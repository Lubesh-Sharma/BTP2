"""Shared inference pipeline: OEM alignment -> encoder features -> HADM matching.

Used by corres.py (test set) and by the validation step inside train.py.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F

from models.modules.orientation_module import normalize_pc_torch, rotate_y_radians


def to_t(a, device):
    return torch.as_tensor(np.asarray(a), dtype=torch.float32, device=device)


@torch.no_grad()
def oem_hypotheses(orient_net, p1, p2, n_votes=4, n_hypotheses=3, min_sep_deg=20.0):
    """Ranked candidate yaw angles (radians) that align target p2 [1,N,3] to source p1.

    OEM test-time voting: softmax bin probabilities and residual vectors are
    accumulated over `n_votes` different point subsets. Hypotheses: best bin,
    best + 180 deg (front/back flip, the dominant OEM failure), second bin.
    """
    p1n, p2n = normalize_pc_torch(p1), normalize_pc_torch(p2)
    was_training = orient_net.training
    orient_net.eval()
    probs, residuals = 0, 0
    base_seed = orient_net.sampling_seed
    try:
        for v in range(max(1, n_votes)):
            orient_net.sampling_seed = base_seed + 101 * v
            out = orient_net(p1n, p2n)
            probs = probs + out['angle_x'].softmax(-1)[0]
            residuals = residuals + out['residual_x'][0]
    finally:
        orient_net.sampling_seed = base_seed
        orient_net.train(was_training)
    residuals = F.normalize(residuals, dim=-1)
    order = torch.argsort(probs, descending=True)
    angles = [float(orient_net.ANGLE[k] + torch.atan2(residuals[k, 1], residuals[k, 0])) for k in order[:2]]
    angles.insert(1, angles[0] + math.pi)
    hyps = []
    for a in angles:
        a = math.atan2(math.sin(a), math.cos(a))
        if all(abs(math.degrees(math.atan2(math.sin(a - h), math.cos(a - h)))) > min_sep_deg for h in hyps):
            hyps.append(a)
        if len(hyps) >= n_hypotheses:
            break
    return hyps, probs / probs.sum()


@torch.no_grad()
def encode(model, shape, pos, device):
    """Encoder embedding [N, C] and predicted DINO [N, D] (or None) in frame `pos`."""
    f = to_t(shape['feat'], device).unsqueeze(0)
    z = model.extract_features(f, pos.unsqueeze(0))
    pred = model.predict_dino(z)
    return z[0], (pred[0] if pred is not None else None)


def shape_dict(shape, pos, z, dino_pred, device, use_dino):
    d = {'pos': pos, 'z': z, 'psi': to_t(shape['psi'], device), 'edges': shape['el']}
    if use_dino:
        d['dino'] = to_t(shape['dino'], device)
        d['dino_conf'] = to_t(shape['dino_conf'], device)
        d['dino_pred'] = dino_pred
    return d


def align_and_match(model, orient_net, matcher, S, T, device, use_dino, oem_cfg=None,
                    method='hadm', src_cache=None):
    """Full pipeline for one pair. Returns p2p (np.int64 [N_S], source -> target) and info."""
    oem_cfg = oem_cfg or {}
    was_training = model.training
    model.eval()
    try:
        p1 = to_t(S['pos'], device)
        p2 = to_t(T['pos'], device)
        key = S['name']
        if src_cache is not None and key in src_cache:
            z1, dp1 = src_cache[key]
        else:
            z1, dp1 = encode(model, S, p1, device)
            if src_cache is not None:
                src_cache[key] = (z1, dp1)
        src = shape_dict(S, p1, z1, dp1, device, use_dino)

        hyps, _ = oem_hypotheses(orient_net, p1.unsqueeze(0), p2.unsqueeze(0),
                                 n_votes=oem_cfg.get('votes', 4), n_hypotheses=oem_cfg.get('hypotheses', 3))
        best = None
        for h in hyps:
            p2_al = rotate_y_radians(p2.unsqueeze(0), -torch.tensor([h], device=device))[0]
            z2, dp2 = encode(model, T, p2_al, device)
            tgt = shape_dict(T, p2_al, z2, dp2, device, use_dino)
            score = matcher.score_alignment(src, tgt)[0] if len(hyps) > 1 else 0.0
            if best is None or score > best[0]:
                best = (score, h, tgt)
        score, angle, tgt = best

        if method == 'greedy':
            p2p, info = matcher.match_greedy(src, tgt)
        else:
            p2p, info = matcher.match(src, tgt)
    finally:
        model.train(was_training)
    info = dict(info)
    info.update({'oem_angle_deg': math.degrees(angle), 'alignment_score': float(score),
                 'oem_hypotheses_deg': [math.degrees(h) for h in hyps]})
    return p2p.cpu().numpy().astype(np.int64), info
