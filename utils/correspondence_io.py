"""Read the zero-based target-to-source maps exported by correspondence inference."""
import numpy as np


def load_target_to_source_map(path, n_source, n_target):
    """Validate both columns and return source indices in target vertex order."""
    pairs = np.loadtxt(path, ndmin=2)
    if pairs.shape != (n_target, 2):
        raise ValueError(f"Expected {n_target} rows and two columns, got {pairs.shape}: {path}")
    if not np.isfinite(pairs).all() or not np.equal(pairs, np.floor(pairs)).all():
        raise ValueError(f"P2P indices must be finite integers: {path}")
    pairs = pairs.astype(np.int64)
    target, source = pairs[:, 0], pairs[:, 1]
    if not np.array_equal(np.sort(target), np.arange(n_target)):
        raise ValueError(f"Target indices must cover 0..{n_target - 1} exactly once: {path}")
    if (source < 0).any() or (source >= n_source).any():
        raise ValueError(f"Source indices must be in 0..{n_source - 1}: {path}")
    result = np.empty(n_target, dtype=np.int64)
    result[target] = source
    return result
