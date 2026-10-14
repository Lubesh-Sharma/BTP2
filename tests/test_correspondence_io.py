import tempfile
import unittest
from pathlib import Path

import numpy as np
from utils.correspondence_io import load_target_to_source_map


class CorrespondenceIOTests(unittest.TestCase):
    def load(self, pairs, n_source, n_target):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'p2p_source_to_target.txt'
            np.savetxt(path, pairs)
            return load_target_to_source_map(path, n_source, n_target)

    def test_target_order_and_unequal_vertex_counts(self):
        result = self.load([[2, 4], [0, 1], [1, 3]], 5, 3)
        np.testing.assert_array_equal(result, [1, 3, 4])

    def test_single_vertex(self):
        np.testing.assert_array_equal(self.load([[0, 2]], 3, 1), [2])

    def test_reject_invalid_indices_instead_of_clipping(self):
        for pairs in ([[0, 3], [1, 0]], [[0, -1], [1, 0]],
                      [[0, 1], [0, 2]], [[0, 0.5], [1, 2]]):
            with self.subTest(pairs=pairs), self.assertRaises(ValueError):
                self.load(pairs, 3, 2)

    def test_reject_flattened_or_incomplete_map(self):
        with self.assertRaises(ValueError):
            self.load([[0, 1, 1, 2]], 3, 2)


if __name__ == '__main__':
    unittest.main()
