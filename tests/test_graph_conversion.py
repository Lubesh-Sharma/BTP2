"""Regression coverage for asymmetric kNN edges and training-time data rebuilds."""
import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import numpy as np

from off_to_obj import convert_off_file
from utils.data import rebuild_training_data
from utils.mesh import build_knn_edges, load_obj, save_off


class GraphConversionTests(unittest.TestCase):
    def test_one_way_neighbor_selection_is_preserved(self):
        # Vertex 2 selects 1, but 1 selects 0. Index filtering used to isolate 2.
        points = np.array([[0., 0., 0.], [0.1, 0., 0.], [10., 0., 0.]])
        vertices, edges = build_knn_edges(points, k=1)
        np.testing.assert_array_equal(vertices, points)
        self.assertEqual(set(map(tuple, edges)), {(0, 1), (1, 2)})

    def test_duplicate_positions_and_large_k_do_not_drop_vertices(self):
        points = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.], [1., 0., 0.]])
        for k in (1, 2, 10):
            _, edges = build_knn_edges(points, k=k)
            self.assertTrue(np.all(edges[:, 0] < edges[:, 1]))
            self.assertEqual(len(edges), len(set(map(tuple, edges))))
            self.assertTrue(np.all(np.bincount(edges.ravel(), minlength=4) >= min(k, 3)))

    def test_invalid_neighbor_count_fails(self):
        points = np.zeros((3, 3))
        for k in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                build_knn_edges(points, k=k)

    def test_disconnected_conversion_does_not_overwrite_existing_obj(self):
        with tempfile.TemporaryDirectory() as directory:
            source, output = Path(directory) / 'shape.off', Path(directory) / 'shape.obj'
            points = np.array([[0., 0., 0.], [0.1, 0., 0.], [10., 0., 0.], [10.1, 0., 0.]])
            save_off(str(source), points, np.array([]), np.empty((0, 3), dtype=int))
            output.write_text('original')
            with self.assertRaisesRegex(ValueError, 'disconnected'):
                convert_off_file(str(source), str(output), k=1)
            self.assertEqual(output.read_text(), 'original')

    def test_rebuild_uses_yaml_paths_and_replaces_stale_geometry_only(self):
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            points = np.stack((np.arange(8) * 0.1, np.zeros(8), np.zeros(8)), axis=1)
            save_off(str(source / 'shape.off'), points, np.array([]), np.empty((0, 3), dtype=int))
            cache = root / 'spectral'
            cache.mkdir()
            stale = cache / 'shape_geo_k99_t100_e300_b100.npz'
            stale.write_bytes(b'stale')
            dino = cache / 'dino'
            dino.mkdir()
            marker = dino / 'shape.npz'
            marker.write_bytes(b'preserved DINO')
            config = dict(preprocessing=dict(off_dir=str(source), knn_k=2),
                          data_dir=str(root / 'graphs'), cache_dir=str(cache),
                          output_dir=str(root / 'fps'), split=dict(mode='all'),
                          geometry=dict(k=3, t=1, neigvecs=6, n_biharmonic=4))
            rebuild_training_data(config)
            vertices, _, first_edges = load_obj(str(root / 'graphs/shape.obj'))
            np.testing.assert_allclose(vertices, points)
            self.assertFalse(stale.exists())
            self.assertEqual(marker.read_bytes(), b'preserved DINO')
            self.assertTrue((root / 'fps/shape.pp').exists())
            cached = cache / 'shape_geo_k3_t1_e6_b4.npz'
            with np.load(cached) as data:
                np.testing.assert_array_equal(data['Elements'], first_edges)
                self.assertTrue(np.isfinite(data['psi']).all())
            # A second rebuild must replace the same cache even though its filename omits graph k.
            config['preprocessing']['knn_k'] = 3
            rebuild_training_data(config)
            _, _, second_edges = load_obj(str(root / 'graphs/shape.obj'))
            self.assertGreater(len(second_edges), len(first_edges))
            with np.load(cached) as data:
                np.testing.assert_array_equal(data['Elements'], second_edges)


if __name__ == '__main__':
    unittest.main()
