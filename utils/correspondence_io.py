"""Read the zero-based source-to-target maps exported by corres.py.

File p2p_{TARGET}_to_{SOURCE}.txt has one row per SOURCE vertex: "i j", meaning
source vertex i corresponds to target vertex j (the format geodesic_error.py reads).
"""
import numpy as np


def load_source_to_target_map(path, n_source, n_target):
    """Validate both columns and return target indices in source vertex order."""
    pairs = np.loadtxt(path, ndmin=2)
    if pairs.shape != (n_source, 2):
        raise ValueError(f"Expected {n_source} rows and two columns, got {pairs.shape}: {path}")
    if not np.isfinite(pairs).all() or not np.equal(pairs, np.floor(pairs)).all():
        raise ValueError(f"P2P indices must be finite integers: {path}")
    pairs = pairs.astype(np.int64)
    source, target = pairs[:, 0], pairs[:, 1]
    if not np.array_equal(np.sort(source), np.arange(n_source)):
        raise ValueError(f"Source indices must cover 0..{n_source - 1} exactly once: {path}")
    if (target < 0).any() or (target >= n_target).any():
        raise ValueError(f"Target indices must be in 0..{n_target - 1}: {path}")
    result = np.empty(n_source, dtype=np.int64)
    result[source] = target
    return result
