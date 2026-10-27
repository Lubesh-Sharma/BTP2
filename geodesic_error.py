"""Mean geodesic error of predicted point-to-point maps (the standard benchmark metric).

    python geodesic_error.py --config config/SHREC/corres_config.yaml --method hadm
    python geodesic_error.py --dataset FAUST                 # = config/FAUST/corres_config.yaml, method hadm

For every ground-truth pair (source S, target T) with a prediction
<result_dir>/p2p_<method>/p2p_{T}_to_{S}.txt  (row i: "i j", source vertex i -> target vertex j):
    error = mean_k  d_T( gt_T[k], pred[gt_S[k]] ) / sqrt(area(T))
Ground truth (gt block of the config):
    vts  - FAUST / SCAPE / SURREAL template correspondences  <corres_dir>/<name>.vts
    map  - SHREC'19 pair maps  <map_dir>/i_j.map  (line v of shape i -> vertex of shape j)
Geodesic distances d_T: <mat_dir>/<T>.mat ('dist') if present, otherwise computed on the
mesh <off_dir>/<T>.off (heat method if potpourri3d is installed, else Dijkstra on mesh edges).
The mesh is used here only - evaluation is the one step that needs it.

Writes <result_dir>/eval_<method>/: pair_errors.csv, summary.json, summary.txt,
and a log in <result_dir>/logs/.
"""
import os
import csv
import argparse
from collections import defaultdict

import numpy as np
import yaml

from utils.gt import gt_pairs, gt_vertex_pairs, geodesic_rows, mesh_area
from utils.logger import start_logging, save_json


def quality_label(error):
    """Common scale for sqrt(area)-normalised geodesic error."""
    if error < 0.02: return "Excellent (SOTA)"
    if error < 0.05: return "Very good"
    if error < 0.1: return "Good"
    if error < 0.2: return "Moderate"
    return "Poor"


def count_vertices_off(path):
    with open(path) as f:
        first = f.readline().split()
        header = first[1:] if len(first) > 1 else f.readline().split()
        return int(header[0])


def main():
    parser = argparse.ArgumentParser(description="Geodesic error evaluation")
    parser.add_argument('--config', default=None, help='dataset corres_config.yaml')
    parser.add_argument('--dataset', default=None, help='FAUST | SCAPE | SURREAL | SHREC (uses its corres config)')
    parser.add_argument('--method', default='hadm', help='which p2p_<method> folder to evaluate')
    parser.add_argument('--p2p_dir', default=None, help='override the prediction folder')
    parser.add_argument('--result_dir', default=None,
                        help='override result_dir (same value as given to corres.py)')
    args = parser.parse_args()
    config_path = args.config or f"config/{(args.dataset or 'FAUST').upper()}/corres_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    result_dir = args.result_dir or config.get('result_dir', 'results')
    start_logging(result_dir, f'eval_{args.method}')

    gt_cfg = config['gt']
    off_dir = gt_cfg.get('off_dir', os.path.join(os.path.dirname(os.path.dirname(
        os.path.normpath(config['data_dir']))), 'off'))
    p2p_dir = args.p2p_dir or os.path.join(result_dir, f'p2p_{args.method}')
    out_dir = os.path.join(result_dir, f'eval_{args.method}')
    os.makedirs(out_dir, exist_ok=True)

    predicted = []
    names = sorted({os.path.splitext(f)[0] for f in os.listdir(off_dir) if f.endswith('.off')})
    for s, t in gt_pairs(gt_cfg, names):
        path = os.path.join(p2p_dir, f"p2p_{t}_to_{s}.txt")
        if os.path.exists(path):
            predicted.append((s, t, path))
    if not predicted:
        print(f"[!] No predictions found in {p2p_dir} for the ground-truth pairs "
              f"(expected names p2p_{{target}}_to_{{source}}.txt)")
        return
    print(f"Evaluating {len(predicted)} pairs from {p2p_dir} | ground truth: {gt_cfg.get('format', 'vts')}")

    by_target = defaultdict(list)
    for s, t, path in predicted:
        by_target[t].append((s, path))

    results = []
    print(f"{'Source':<14} | {'Target':<14} | {'Geo error':<10}")
    print('-' * 46)
    for t, items in by_target.items():
        n_t = count_vertices_off(os.path.join(off_dir, t + '.off'))
        gts = {}
        for s, path in items:
            n_s = count_vertices_off(os.path.join(off_dir, s + '.off'))
            gts[s] = gt_vertex_pairs(gt_cfg, s, t, n_s, n_t)
        rows = np.unique(np.concatenate([g[1] for g in gts.values()]))
        D = geodesic_rows(t, gt_cfg, rows, off_path=os.path.join(off_dir, t + '.off'))
        row_of = np.full(n_t, -1, dtype=np.int64)
        row_of[rows] = np.arange(len(rows))
        norm = np.sqrt(mesh_area(os.path.join(off_dir, t + '.off')))
        for s, path in items:
            data = np.loadtxt(path, dtype=np.int64, ndmin=2)
            pred = np.empty(len(data), dtype=np.int64)
            pred[data[:, 0]] = data[:, 1]
            i_s, i_t = gts[s]
            if pred.max() >= n_t or len(pred) <= i_s.max():
                print(f"[!] {os.path.basename(path)}: indices do not fit the meshes "
                      f"(rows {len(pred)}, max target {pred.max()}, target has {n_t} vertices) - skipped")
                continue
            err = D[row_of[i_t], pred[i_s]].mean() / norm
            results.append({'source': s, 'target': t, 'error': float(err)})
            print(f"{s:<14} | {t:<14} | {err:.6f}")

    errors = np.array([r['error'] for r in results])
    summary = {'dataset_config': config_path, 'method': args.method, 'p2p_dir': p2p_dir,
               'pairs': len(results), 'mean_geodesic_error': float(errors.mean()),
               'median': float(np.median(errors)), 'std': float(errors.std()),
               'quality': quality_label(errors.mean()),
               'best5': sorted(results, key=lambda r: r['error'])[:5],
               'worst5': sorted(results, key=lambda r: -r['error'])[:5]}
    with open(os.path.join(out_dir, 'pair_errors.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['source', 'target', 'error'])
        w.writeheader()
        w.writerows(results)
    save_json(summary, os.path.join(out_dir, 'summary.json'))
    lines = ['=' * 55,
             f"MEAN GEODESIC ERROR ({args.method}, {len(results)} pairs): {summary['mean_geodesic_error']:.6f}",
             f"median {summary['median']:.6f} | std {summary['std']:.6f} | {summary['quality']}",
             '=' * 55, 'Best 5:'] + \
            [f"  {r['source']} -> {r['target']}: {r['error']:.6f}" for r in summary['best5']] + \
            ['Worst 5:'] + [f"  {r['source']} -> {r['target']}: {r['error']:.6f}" for r in summary['worst5']]
    with open(os.path.join(out_dir, 'summary.txt'), 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print('\n' + '\n'.join(lines))
    print(f"\nSaved: {out_dir}/pair_errors.csv, summary.json, summary.txt")


if __name__ == '__main__':
    main()