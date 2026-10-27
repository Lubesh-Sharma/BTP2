"""Dataset split + shape loading shared by train.py, test_recon.py and corres.py.

Everything is computed from the point cloud + kNN graph in data_dir (k_10 .obj files).
"""
import os
import sys
import numpy as np

from core.preprocessing import process_geometry


def dataset_split(config):
    """Sorted .obj files -> (train_files, validation_files, test_files).

    split.mode 'sequential' (default; FAUST / SCAPE / SURREAL):
        first train_size-validation_size -> train, next validation_size -> validation,
        next test_size -> test.
    split.mode 'all' (SHREC'19 protocol, unsupervised):
        every shape is used for training AND testing (test pairs come from the GT files).
        Validation shapes = the shapes of the first `split.validation_pairs` GT pairs
        (they are also training shapes; no labels are used for training).
    """
    data_dir = config['data_dir']
    obj_files = sorted(f for f in os.listdir(data_dir) if f.endswith('.obj'))
    split = config.get('split', {})
    if split.get('mode', 'sequential') == 'all':
        from utils.gt import gt_pairs
        names = [os.path.splitext(f)[0] for f in obj_files]
        pairs = gt_pairs(config.get('gt', {}), names)[:split.get('validation_pairs', 4)]
        val_names = sorted({n for p in pairs for n in p}, key=names.index)
        if len(val_names) < 2:
            raise ValueError("split.mode 'all' needs ground-truth pairs (gt block) to pick validation shapes.")
        val = [n + '.obj' for n in val_names]
        return list(obj_files), val, list(obj_files)

    total = len(obj_files)
    train_size = config.get('train_size', 60)
    test_size = config.get('test_size', 40)
    if total < train_size + test_size:
        train_size = int(total * train_size / (train_size + test_size))
        test_size = total - train_size
        print(f"Dataset too small ({total} files). Adjusted to {train_size} train, {test_size} test")
    validation_size = config.get('training', {}).get('validation_size', 2)
    if validation_size < 1 or train_size - validation_size < 2:
        raise ValueError("Reserve at least one validation shape and two training shapes.")
    train = obj_files[:train_size - validation_size]
    val = obj_files[train_size - validation_size:train_size]
    test = obj_files[train_size:train_size + test_size]
    return train, val, test


def load_shapes(config, filenames, with_dino=False, label='shapes'):
    """Load geometry (+ optional DINO) for a list of .obj filenames.

    Each shape dict: name, pos [N,3], el (graph edges / faces), feat [N, k+3],
    psi [N, K] biharmonic embedding, evals, evecs, and if with_dino:
    dino [N, D] (unit norm) and dino_conf [N] in [0, 1].
    """
    geo = config['geometry']
    data_dir = config['data_dir']
    output_dir = config['output_dir']
    cache_dir = config.get('cache_dir', os.path.join(output_dir, 'cache'))
    dino_cfg = config.get('dino', {})
    print(f"\n{'=' * 60}\nLoading {len(filenames)} {label} from: {data_dir}\n{'=' * 60}")
    shapes = []
    for i, fname in enumerate(filenames):
        name = os.path.splitext(fname)[0]
        print(f"[{i + 1}/{len(filenames)}] {fname} ...", end=' ')
        sys.stdout.flush()
        VPos, El, Feat, spec = process_geometry(
            os.path.join(data_dir, fname), geo['k'], geo['t'], geo.get('neigvecs', 300),
            output_dir=output_dir, n_biharmonic=geo.get('n_biharmonic', 100), cache_dir=cache_dir)
        if VPos is None or len(VPos) == 0 or Feat is None or len(Feat) == 0:
            print("SKIP (empty)")
            continue
        shape = {'name': name, 'pos': VPos.astype(np.float32), 'el': El, 'feat': Feat.astype(np.float32),
                 'psi': spec['psi'], 'evals': spec['evals'], 'evecs': spec['evecs']}
        if with_dino:
            # 2D features from the point cloud itself (DV-Matcher-style projection): no mesh needed
            from utils.pc_dino import point_dino_for_shape
            feat, conf = point_dino_for_shape(name, VPos, dino_cfg,
                                              cache_dir=dino_cfg.get('cache_dir', os.path.join(cache_dir, 'dino')))
            shape['dino'], shape['dino_conf'] = feat, conf
        shapes.append(shape)
        print(f"OK ({len(VPos)} vertices, {Feat.shape[1]} features"
              + (f", DINO {shape['dino'].shape[1]}-d, mean conf {shape['dino_conf'].mean():.2f}" if with_dino else '')
              + ")")
    print(f"Loaded {len(shapes)} {label}")
    return shapes
