"""Point-to-point correspondence: encoder + DINO features -> HADM matcher.

For every test pair (source S, target T):
  1. OEM predicts the relative yaw (test-time voting over point subsets); the top
     hypotheses (best bin, best + 180 deg, 2nd bin) are verified by the matcher's
     anchor-consistency score and the best one is kept.
  2. Encoder features: S in its own frame, T in the chosen aligned frame
     (exactly as during training).
  3. DINO features (cached, alignment-independent) + visibility confidence;
     the encoder's DINO head fills vertices no camera saw.
  4. HeatAnchoredMatcher -> dense map S -> T.

Output: <result_dir>/p2p_<method>/p2p_{T}_to_{S}.txt, one row per SOURCE vertex: "i j"
(source vertex i matches target vertex j), the format geodesic_error.py reads.
Pairs: correspondence.pairs = "all" (every ordered test pair), "consecutive", or "gt"
(only pairs with ground truth, e.g. the SHREC'19 i_j.map files: source i -> target j).
A log of the run is written to <result_dir>/logs/.

Usage:
    python corres.py --config config/FAUST/corres_config.yaml
    python corres.py --config config/FAUST/corres_config.yaml --method greedy     # ablation
"""
import os
import json
import time
import argparse

import numpy as np
import torch
import yaml

from models.asmae import build_asmae
from models.modules.orientation_module import OrientNet
from core.matching import HeatAnchoredMatcher
from core.pipeline import align_and_match
from utils.data import dataset_split, load_shapes
from utils.gt import gt_pairs
from utils.logger import start_logging, save_json


def build_matcher(match_cfg, method, device, verbose=False):
    cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in (match_cfg or {}).items() if k != 'method'}
    if method == 'heat_only':
        cfg.setdefault('deform', {})['enabled'] = False
    return HeatAnchoredMatcher(cfg, device=device, verbose=verbose)


def load_models(ckpt_path, config, device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_asmae(config.get('model', ckpt['config'].get('model')), ckpt['feature_dim'],
                        ckpt.get('dino_dim', 0)).to(device).eval()
    model.load_state_dict(ckpt['model_state_dict'])
    orient_net = OrientNet(**ckpt['config'].get('orientation_model', {})).to(device).eval()
    orient_net.load_state_dict(ckpt['orient_net_state_dict'])
    return model, orient_net, ckpt


def main():
    parser = argparse.ArgumentParser(description="HADM point-to-point correspondence")
    parser.add_argument('--config', default='config/FAUST/corres_config.yaml')
    parser.add_argument('--checkpoint', default=None, help='defaults to correspondence.checkpoint_path')
    parser.add_argument('--method', default=None, choices=['hadm', 'heat_only', 'greedy'],
                        help='override matching.method (ablations)')
    parser.add_argument('--max_pairs', type=int, default=None)
    parser.add_argument('--result_dir', default=None,
                        help='override result_dir, e.g. results/SHREC_to_FAUST for cross-dataset tests')
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    if args.result_dir:
        config['result_dir'] = args.result_dir
    result_dir = config.get('result_dir', 'results')
    start_logging(result_dir, f"corres_{args.method or config.get('matching', {}).get('method', 'hadm')}")
    cc = config['correspondence']
    device = cc.get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    torch.manual_seed(cc.get('seed', 0))

    ckpt_path = args.checkpoint or cc['checkpoint_path']
    model, orient_net, ckpt = load_models(ckpt_path, config, device)
    trained = ckpt.get('config', {})
    print(f"Checkpoint: {ckpt_path} (trained on {trained.get('data_dir', '?')})")
    # cross-dataset testing: features must be computed exactly as during training
    for block, keys in [('geometry', ['k', 't', 'neigvecs', 'n_biharmonic']),
                        ('dino', ['upsampler', 'variant', 'views', 'projection', 'n_azimuth',
                                  'elevations_deg', 'img_size', 'splat', 'colormap', 'smooth_k'])]:
        for k in keys:
            a, b = trained.get(block, {}).get(k), config.get(block, {}).get(k)
            if a is not None and a != b:
                print(f"[!] {block}.{k} differs: checkpoint={a} vs this config={b} - "
                      f"set it to {a} in {args.config} for a fair cross-dataset test")
    method = args.method or config.get('matching', {}).get('method', 'hadm')
    matcher = build_matcher(config.get('matching', {}), method, device, cc.get('verbose', False))
    use_dino = config.get('dino', {}).get('enabled', False) and matcher.cfg['w_dino'] > 0

    _, _, test_files = dataset_split(config)
    if cc.get('max_test_shapes'):
        test_files = test_files[:cc['max_test_shapes']]
    shapes = load_shapes(config, test_files, with_dino=use_dino, label='test shapes')
    if use_dino and model.dino_dim and shapes and shapes[0]['dino'].shape[1] != model.dino_dim:
        raise ValueError(f"DINO features are {shapes[0]['dino'].shape[1]}-d but the checkpoint's DINO head is "
                         f"{model.dino_dim}-d: use the same dino settings as in training.")
    n = len(shapes)
    mode = cc.get('pairs', 'all')
    if mode == 'gt':
        index = {s['name']: k for k, s in enumerate(shapes)}
        pairs = [(index[a], index[b]) for a, b in gt_pairs(config.get('gt', {}), list(index))]
    elif mode == 'consecutive':
        pairs = [(i, i + 1) for i in range(n - 1)]
    else:
        pairs = [(i, j) for i in range(n) for j in range(n) if i != j]
    if args.max_pairs:
        pairs = pairs[:args.max_pairs]

    out_dir = os.path.join(result_dir, f'p2p_{method}')
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, 'matching_log.jsonl')
    save_json(config, os.path.join(out_dir, 'corres_config_used.json'))
    print(f"\nMatching {len(pairs)} pairs | method={method} | DINO={'on' if use_dino else 'off'} | device={device}")

    src_cache = {}
    with open(log_path, 'w') as log:
        for k, (i, j) in enumerate(pairs):
            t0 = time.time()
            S, T = shapes[i], shapes[j]
            p2p, info = align_and_match(model, orient_net, matcher, S, T, device, use_dino,
                                        config.get('oem', {}), method, src_cache)
            np.savetxt(os.path.join(out_dir, f"p2p_{T['name']}_to_{S['name']}.txt"),
                       np.stack([np.arange(len(p2p)), p2p], 1), fmt='%d')
            rec = {'source': S['name'], 'target': T['name'], 'seconds': time.time() - t0,
                   **{kk: v for kk, v in info.items() if np.isscalar(v) or isinstance(v, list)}}
            log.write(json.dumps(rec) + '\n')
            log.flush()
            print(f"[{k + 1}/{len(pairs)}] {S['name']} -> {T['name']} | yaw {info['oem_angle_deg']:.1f} deg "
                  f"| anchors {info.get('anchors', '-')} | {rec['seconds']:.1f}s")
    print(f"\nSaved maps to {out_dir} (per-pair log: {log_path})")
    print(f"Evaluate: python geodesic_error.py --config {args.config} --method {method}")


if __name__ == '__main__':
    main()