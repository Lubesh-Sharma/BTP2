"""Shared file selection and loading for generated target-to-source p2p maps."""

from pathlib import Path
import numpy as np


def correspondence_output_dir(config, method='sinkhorn'):
    directory = config['output_dir']
    if method == 'greedy':
        directory = directory.replace('st_te', 'greedy')
    elif method != 'sinkhorn':
        raise ValueError("method must be 'sinkhorn' or 'greedy'")
    return Path(directory)


def select_correspondence_files(files, checkpoint, data_dir, limit=20):
    """Exclude the training partition when matching its original dataset."""
    files = sorted(files)
    if limit == -1:
        return files
    trained_dir = checkpoint.get('train_data_dir')
    same_dataset = trained_dir is None or Path(trained_dir).resolve() == Path(data_dir).resolve()
    if same_dataset:
        if 'train_filenames' in checkpoint:
            reserved = {Path(f).stem for f in checkpoint['train_filenames']}
            files = [f for f in files if Path(f).stem not in reserved]
        else:
            split = checkpoint.get('train_split_index', checkpoint.get('train_size', 0))
            files = files[split:]
    return files[-limit:] if limit > 0 else files


def load_p2p_map(path, source_count, target_count):
    """Read [target_vertex, source_vertex] rows using their explicit indices.

    Both generators write zero-based indices. Do not flatten both columns,
    truncate rows, guess index bases, or clip invalid predictions for display.
    """
    pairs = np.loadtxt(path, dtype=np.int64, ndmin=2)
    if pairs.shape != (target_count, 2):
        raise ValueError(f"{path}: expected {target_count} rows and two columns; got {pairs.shape}")
    target_ids, source_ids = pairs[:, 0], pairs[:, 1]
    if not np.array_equal(np.sort(target_ids), np.arange(target_count)):
        raise ValueError(f"{path}: target indices must cover every target vertex exactly once")
    if np.any(source_ids < 0) or np.any(source_ids >= source_count):
        raise ValueError(f"{path}: predicted source indices exceed the source vertex count")
    mapping = np.empty(target_count, dtype=np.int64)
    mapping[target_ids] = source_ids
    return mapping
