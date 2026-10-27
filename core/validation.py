"""Correspondence validation during training (selects the *best* checkpoint).

Metric (lower is better):
  * 'geodesic'      ground-truth pairs among the validation shapes exist (gt block:
                    .vts templates or SHREC .map pairs). Error = mean geodesic distance on
                    the target between predicted and true vertex / target diameter, with the
                    full inference pipeline (OEM hypotheses + matcher).
                    Geodesics come from <mat_dir>/<name>.mat if present, otherwise from
                    Dijkstra on the point cloud's kNN graph - no mesh is used.
                    (geodesic_error.py normalises by sqrt(area) instead; numbers differ,
                    but the ranking of checkpoints is what matters here.)
  * 'self_identity' no ground truth: each validation shape is matched to an augmented copy
                    of itself (known identity); error = mean biharmonic distance.
"""
import numpy as np
import torch
import torch.nn.functional as F

from core.matching import HeatAnchoredMatcher
from core.pipeline import align_and_match, to_t
from core.intrinsic_losses import augment_view
from utils.gt import gt_pairs, gt_vertex_pairs, geodesic_rows


class CorrespondenceValidator:
    def __init__(self, config, val_shapes, device):
        vc = config.get('validation', {})
        self.gt_cfg = config.get('gt', {})
        self.method = vc.get('method', 'heat_only')
        self.device = device
        self.shapes = {s['name']: s for s in val_shapes}
        self.order = [s['name'] for s in val_shapes]
        self.oem_cfg = config.get('oem', {})
        match_cfg = {k: (dict(v) if isinstance(v, dict) else v)
                     for k, v in config.get('matching', {}).items() if k != 'method'}
        if self.method == 'heat_only':
            match_cfg.setdefault('deform', {})['enabled'] = False
        self.matcher = HeatAnchoredMatcher(match_cfg, device=device)
        self.use_dino = all('dino' in s for s in val_shapes) and self.matcher.cfg['w_dino'] > 0

        pairs = gt_pairs(self.gt_cfg, self.order) if self.gt_cfg else []
        max_pairs = vc.get('max_pairs', 6)
        self.pairs = pairs[:max_pairs] if max_pairs else pairs
        self.mode = 'geodesic' if self.pairs else 'self_identity'
        self._gt = {}
        print(f"  Validation metric: {self.mode} | method={self.method} | "
              f"{len(self.pairs) if self.pairs else len(val_shapes)} cases | DINO={self.use_dino}")

    def _ground_truth(self, s, t):
        """(idx_src, idx_tgt, distance rows on target for idx_tgt, diameter) - cached."""
        if (s, t) not in self._gt:
            S, T = self.shapes[s], self.shapes[t]
            i_s, i_t = gt_vertex_pairs(self.gt_cfg, s, t, len(S['pos']), len(T['pos']))
            rows, inv = np.unique(i_t, return_inverse=True)
            D = geodesic_rows(t, self.gt_cfg, rows, pos=T['pos'], edges=T['el']).astype(np.float32)
            self._gt[(s, t)] = (i_s, inv, D, float(D.max()))
        return self._gt[(s, t)]

    @torch.no_grad()
    def geodesic(self, model, orient_net):
        errors, cache = [], {}
        for s, t in self.pairs:
            S, T = self.shapes[s], self.shapes[t]
            p2p, _ = align_and_match(model, orient_net, self.matcher, S, T, self.device, self.use_dino,
                                     self.oem_cfg, 'greedy' if self.method == 'greedy' else 'hadm', cache)
            i_s, inv, D, diameter = self._ground_truth(s, t)
            errors.append(D[inv, p2p[i_s]].mean() / diameter)
        return float(np.mean(errors))

    @torch.no_grad()
    def self_identity(self, model):
        was_training = model.training
        model.eval()
        state = torch.random.get_rng_state()
        torch.manual_seed(1234)
        errors = []
        try:
            for s in self.shapes.values():
                f = to_t(s['feat'], self.device).unsqueeze(0)
                p = to_t(s['pos'], self.device).unsqueeze(0)
                psi = to_t(s['psi'], self.device)
                z = F.normalize(model.extract_features(f, p)[0], dim=-1)
                pb, ib = augment_view(p, 0.8, 15.0, (0.9, 1.1), 0.003)
                zb = F.normalize(model.extract_features(f[:, ib], pb)[0], dim=-1)
                pred = (zb @ z.T).argmax(1)
                errors.append((psi[pred] - psi[ib]).norm(dim=-1).mean().item())
        finally:
            torch.random.set_rng_state(state)
            model.train(was_training)
        return float(np.mean(errors))

    def run(self, model, orient_net):
        if self.mode == 'geodesic':
            return {'metric': self.geodesic(model, orient_net), 'mode': 'geodesic'}
        return {'metric': self.self_identity(model), 'mode': 'self_identity'}
