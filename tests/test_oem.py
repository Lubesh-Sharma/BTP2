"""CPU checks for OEM rotation supervision and pair-domain adaptation."""
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from core.domain_loss import compute_angle_loss, compute_domain_loss, FocalLoss
    from models.modules.orientation_module import OrientNet


@unittest.skipIf(torch is None, "PyTorch is not installed")
class OEMTests(unittest.TestCase):
    def test_rotation_round_trip_and_inverse_labels(self):
        model = OrientNet()
        points = torch.randn(8, 12, 3) + 5
        bins = torch.arange(8)
        rotated = model.rotate_point_cloud(points, bins, inverse=False)
        restored = model.rotate_point_cloud(rotated, bins, inverse=True)
        torch.testing.assert_close(restored, points, atol=2e-6, rtol=2e-6)
        inverse_bins = (2 - bins) % 8
        opposite = model.rotate_point_cloud(rotated, inverse_bins, inverse=False)
        torch.testing.assert_close(opposite, points, atol=2e-6, rtol=2e-6)
        logits_x = torch.full((8, 8), -10.0)
        logits_y = torch.full((8, 8), -10.0)
        logits_x[torch.arange(8), bins] = 10
        logits_y[torch.arange(8), inverse_bins] = 10
        self.assertLess(compute_angle_loss(logits_x, logits_y, bins).item(), 0.001)

    def test_pair_discriminator_batch_one_unequal_clouds_and_backward(self):
        torch.manual_seed(7)
        model = OrientNet(input_dims=[3, 8], latent_dim=8, output_dim=8,
                          mlps=[16, 8], num_neighs=3, input_neighs=3)
        source = torch.randn(1, 9, 3)
        target = torch.randn(1, 11, 3)
        reference = model(target, target + 0.01 * torch.randn_like(target))
        cross = model(source, target)
        self.assertEqual(cross['global_d_pred'].shape, (1, 2))
        loss = compute_domain_loss(reference['global_d_pred'], cross['global_d_pred'],
                                   FocalLoss(class_num=2, gamma=3))
        loss = loss + compute_angle_loss(reference['angle_x'], reference['angle_y'],
                                        torch.tensor([1]))
        self.assertTrue(torch.isfinite(loss).item())
        loss.backward()
        for parameter in (model.global_netD2.weight, model.input_modules[0].linear.weight,
                          model.classifier.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all().item())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
