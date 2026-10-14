"""Fixed validation metrics for checkpoint selection; no optimizer updates."""

from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from scipy.io import loadmat
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from models.oriented_pair import prepare_aligned_pair
from utils.mesh import load_off


@torch.no_grad()
def validate_correspondence(model, oem, shapes, settings, data_dir):
    """Measure ground-truth cross-shape geodesic error where maps exist.

    Otherwise report a distinctly named synthetic self-rotation p2p error.
    Its known vertex identities are not a claim of cross-shape accuracy.
    Matching is greedy, over sampled query vertices and ALL candidate vertices.
    """
    device = next(model.parameters()).device
    model_mode, oem_mode = model.training, oem.training
    model.eval()
    oem.eval()
    root = Path(data_dir).parent.parent
    map_dir = Path(settings.get('map_dir', root / 'map'))
    mat_dir = Path(settings.get('mat_dir', root / 'mat'))
    off_dir = Path(settings.get('off_dir', root / 'off'))
    samples = max(int(settings.get('vertices', 256)), 1)
    max_pairs = max(int(settings.get('max_pairs', 6)), 1)
    tensors = {}
    for shape in shapes:
        tensors[shape['name']] = (
            torch.as_tensor(shape['feat'], device=device, dtype=torch.float32).unsqueeze(0),
            torch.as_tensor(shape['pos'], device=device, dtype=torch.float32).unsqueeze(0))

    def predict(source, target, positions_source=None):
        fx, px = tensors[source['name']]
        fy, py = tensors[target['name']]
        fx, px, fy, py, _ = prepare_aligned_pair(oem, fx, px if positions_source is None else positions_source, fy, py)
        zx = F.normalize(model.extract_features(fx, px), dim=-1).squeeze(0)
        zy = F.normalize(model.extract_features(fy, py), dim=-1).squeeze(0)
        ids = torch.linspace(0, len(zy) - 1, min(samples, len(zy)), device=device).long()
        prediction = (zy[ids] @ zx.T).argmax(-1)
        return ids.cpu().numpy(), prediction.cpu().numpy()

    errors = []
    pair_count = 0
    matrix_cache = {}
    use_matrices = all((mat_dir / f"{s['name']}.mat").exists() for s in shapes)
    try:
        for source in shapes:
            for target in shapes:
                if source['name'] == target['name'] or pair_count >= max_pairs:
                    continue
                # Saved convention: target Y -> source X.
                map_path = map_dir / f"{target['name']}_{source['name']}.map"
                matrix_path = mat_dir / f"{source['name']}.mat"
                mesh_path = off_dir / f"{source['name']}.off"
                if not (map_path.exists() and mesh_path.exists()):
                    continue
                ground_truth = np.loadtxt(map_path, dtype=np.int64).reshape(-1)
                ground_truth -= int(settings.get('map_index_base', 1))
                if len(ground_truth) != len(target['pos']):
                    raise ValueError(f'{map_path}: GT rows differ from target vertex count')
                name = source['name']
                if name not in matrix_cache:
                    vertices, _, faces = load_off(str(mesh_path))
                    if vertices.shape != source['pos'].shape or not np.allclose(vertices, source['pos'], atol=1e-6):
                        raise ValueError(f'{mesh_path}: vertex order differs from matching input')
                    area = 0.5 * np.linalg.norm(np.cross(vertices[faces[:, 1]] - vertices[faces[:, 0]],
                                                       vertices[faces[:, 2]] - vertices[faces[:, 0]]), axis=-1).sum()
                    if area <= 0:
                        raise ValueError(f'{mesh_path}: invalid mesh area')
                    if use_matrices:
                        distances = loadmat(matrix_path)['dist']
                        if distances.shape != (len(vertices), len(vertices)):
                            raise ValueError(f'{matrix_path}: distance matrix has wrong vertex count')
                    else:
                        # No dense geodesic matrix required. Shortest paths on
                        # original triangle edges approximate surface distance.
                        edges = np.concatenate((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
                        edges = np.unique(np.sort(edges, axis=1), axis=0)
                        lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=-1)
                        row = np.concatenate((edges[:, 0], edges[:, 1]))
                        col = np.concatenate((edges[:, 1], edges[:, 0]))
                        distances = coo_matrix((np.tile(lengths, 2), (row, col)),
                                               shape=(len(vertices), len(vertices))).tocsr()
                    matrix_cache[name] = (distances, np.sqrt(area))
                ids, predicted = predict(source, target)
                valid = (ground_truth[ids] >= 0) & (ground_truth[ids] < len(source['pos']))
                distances, scale = matrix_cache[name]
                if valid.any():
                    true_ids = ground_truth[ids[valid]]
                    if use_matrices:
                        values = np.asarray(distances[true_ids, predicted[valid]]).reshape(-1) / scale
                    else:
                        unique, inverse = np.unique(true_ids, return_inverse=True)
                        rows = np.atleast_2d(dijkstra(distances, directed=False, indices=unique))
                        values = rows[inverse, predicted[valid]] / scale
                    values = np.where(np.isfinite(values), values,
                                      float(settings.get('disconnected_penalty', 1.0)))
                    if len(values):
                        errors.extend(values.tolist())
                        pair_count += 1
        if errors:
            values = np.asarray(errors)
            metric = 'validation_greedy_geodesic' if use_matrices else 'validation_greedy_edge_geodesic'
            return {'metric': metric, 'error': float(values.mean()),
                    'pck_005': float((values < 0.05).mean()), 'pairs': pair_count}

        # No GT is fabricated for cross-shape pairs without map/matrix files.
        for shape in shapes[:2]:
            _, positions = tensors[shape['name']]
            for index in settings.get('synthetic_bins', [1, 3, 5, 7]):
                rotated = oem.rotate_point_cloud(positions, torch.tensor([index], device=device), inverse=False)
                ids, predicted = predict(shape, shape, rotated)
                positions_np = shape['pos']
                scale = np.linalg.norm(positions_np - positions_np.mean(axis=0), axis=-1).max()
                values = np.linalg.norm(positions_np[predicted] - positions_np[ids], axis=-1) / max(scale, 1e-12)
                errors.extend(values.tolist())
                pair_count += 1
        if not errors:
            raise ValueError('Validation needs at least one successfully loaded shape')
        return {'metric': 'validation_synthetic_p2p_euclidean', 'error': float(np.mean(errors)),
                'pck_005': float((np.asarray(errors) < 0.05).mean()), 'pairs': pair_count}
    finally:
        model.train(model_mode)
        oem.train(oem_mode)
