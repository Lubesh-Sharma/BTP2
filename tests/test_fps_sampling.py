import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is required')
class FPSSamplingTests(unittest.TestCase):
    def test_farthest_coverage_and_no_duplicate_indices(self):
        from models.modules.orientation_module import farthest_point_indices
        points = torch.arange(10, dtype=torch.float32).view(1, 10, 1).expand(-1, -1, 3)
        indices = farthest_point_indices(points, 5, torch.Generator().manual_seed(5))[0]
        self.assertEqual(len(indices.unique()), 5)
        for step in range(1, 5):
            distances = torch.cdist(points, points[:, indices[:step]]).min(-1).values[0]
            distances[indices[:step]] = -1
            self.assertEqual(indices[step].item(), distances.argmax().item())
        duplicates = torch.zeros(1, 10, 3)
        self.assertEqual(farthest_point_indices(duplicates, 5)[0].unique().numel(), 5)

    def test_fraction_reproducibility_full_input_preserved(self):
        from models.modules.orientation_module import OrientNet
        net = OrientNet(input_neighs=3, num_neighs=3, sample_min_points=4,
                        sampling_method='fps', sample_fraction=0.5).eval()
        source, target = torch.randn(1, 15, 3), torch.randn(1, 9, 3)
        original = source.clone()
        a, b = net.sample_pair(source, target)
        again, _ = net.sample_pair(source, target)
        self.assertEqual(a.shape[1], 8)
        self.assertEqual(b.shape[1], 5)
        self.assertTrue(torch.equal(a, again))
        self.assertTrue(torch.equal(source, original))
        small = torch.randn(1, 3, 3)
        result, _ = net.sample_pair(small, target)
        self.assertTrue(torch.equal(result, small))
