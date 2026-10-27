"""Ground-truth correspondences + geodesic distances for validation and evaluation.

Config block (top level):

    gt:
      format: "vts"            # "vts": per-shape template indices (FAUST, SCAPE, SURREAL)
                               # "map": per-pair files i_j.map (SHREC'19)
      corres_dir: "input/FAUST/corres"   # .vts files              (format vts)
      map_dir: "input/SHREC/map"         # i_j.map files           (format map)
      mat_dir: "input/FAUST/mat"         # optional geodesic matrices <name>.mat, key 'dist'
      exclude: []                         # shapes excluded from evaluation pairs (SHREC: ['40'], partial)

SHREC'19 .map convention (as in ULRSSM / DV-Matcher evaluation): i_j.map has one line per
vertex of shape i; line v holds the (1-based) vertex of shape j corresponding to vertex v of i.
So the pair is evaluated as source i -> target j, with the error measured on j.
"""
import os
import glob

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra


def _natural_key(name):
    parts = name.replace('-', '_').split('_')
    return tuple(int(p) if p.isdigit() else p for p in parts)


def gt_pairs(gt_cfg, names):
    """Ordered (source, target) name pairs that have ground truth, restricted to `names`."""
    names = list(names)
    have = set(names)
    exclude = set(str(e) for e in gt_cfg.get('exclude', []))
    fmt = gt_cfg.get('format', 'vts')
    if fmt == 'map':
        pairs = []
        files = sorted(glob.glob(os.path.join(gt_cfg['map_dir'], '*.map')),
                       key=lambda f: _natural_key(os.path.splitext(os.path.basename(f))[0]))
        for f in files:
            stem = os.path.splitext(os.path.basename(f))[0]
            if '_' not in stem:
                continue
            i, j = stem.split('_', 1)
            if i in have and j in have and i not in exclude and j not in exclude:
                pairs.append((i, j))
        return pairs
    with_gt = [n for n in names if n not in exclude
               and os.path.exists(os.path.join(gt_cfg.get('corres_dir', ''), n + '.vts'))]
    return [(s, t) for s in with_gt for t in with_gt if s != t]


def gt_vertex_pairs(gt_cfg, src, tgt, n_src, n_tgt):
    """Arrays (idx_src, idx_tgt): vertex idx_src[k] of src corresponds to idx_tgt[k] of tgt."""
    if gt_cfg.get('format', 'vts') == 'map':
        m = np.loadtxt(os.path.join(gt_cfg['map_dir'], f"{src}_{tgt}.map"), dtype=np.int64).reshape(-1) - 1
        if len(m) == n_src and m.max() < n_tgt:
            return np.arange(n_src), m
        if len(m) == n_tgt and m.max() < n_src:          # file written in the opposite convention
            return m, np.arange(n_tgt)
        raise ValueError(f"{src}_{tgt}.map has {len(m)} lines; shapes have {n_src} / {n_tgt} vertices")
    d = gt_cfg['corres_dir']
    vs = np.loadtxt(os.path.join(d, src + '.vts'), dtype=np.int64).reshape(-1) - 1
    vt = np.loadtxt(os.path.join(d, tgt + '.vts'), dtype=np.int64).reshape(-1) - 1
    return vs, vt


def _graph(pos, edges):
    edges = np.asarray(edges)
    if edges.ndim == 2 and edges.shape[1] == 3:
        edges = np.concatenate([edges[:, [0, 1]], edges[:, [1, 2]], edges[:, [2, 0]]])
    i, j = edges[:, 0].astype(np.int64), edges[:, 1].astype(np.int64)
    w = np.linalg.norm(pos[i] - pos[j], axis=1) + 1e-12
    n = len(pos)
    G = sparse.coo_matrix((np.concatenate([w, w]), (np.concatenate([i, j]), np.concatenate([j, i]))),
                          shape=(n, n)).tocsr()
    G.sum_duplicates()
    return G


def geodesic_rows(name, gt_cfg, rows, pos=None, edges=None, off_path=None):
    """Geodesic distances D[rows, :] on shape `name`.

    Priority: <mat_dir>/<name>.mat ('dist')  >  heat method on the .off mesh (potpourri3d,
    if installed)  >  Dijkstra on the .off mesh edges  >  Dijkstra on the point cloud's
    kNN graph (pos, edges). Disconnected parts get the largest finite distance.
    """
    mat = os.path.join(gt_cfg.get('mat_dir', '') or '', name + '.mat')
    if gt_cfg.get('mat_dir') and os.path.exists(mat):
        import scipy.io as sio
        return np.asarray(sio.loadmat(mat)['dist'])[rows]
    if off_path is not None and os.path.exists(off_path):
        from utils.mesh import load_off
        V, _, F = load_off(off_path)
        F = np.asarray(F, dtype=np.int64)
        try:
            import potpourri3d as pp3d
            solver = pp3d.MeshHeatMethodDistanceSolver(np.asarray(V, dtype=np.float64), F)
            D = np.stack([solver.compute_distance(int(r)) for r in rows])
        except ImportError:
            D = dijkstra(_graph(V, F), directed=False, indices=rows)
    else:
        D = dijkstra(_graph(pos, edges), directed=False, indices=rows)
    finite = np.isfinite(D)
    if not finite.all():
        D = np.where(finite, D, D[finite].max())
    return D


def mesh_area(off_path):
    from utils.mesh import load_off
    V, _, F = load_off(off_path)
    F = np.asarray(F, dtype=np.int64)
    return 0.5 * np.linalg.norm(np.cross(V[F[:, 1]] - V[F[:, 0]], V[F[:, 2]] - V[F[:, 0]]), axis=1).sum()
