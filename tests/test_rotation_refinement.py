import math
import unittest

try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch is required')
class RotationRefinementTests(unittest.TestCase):
    def setUp(self):
        from models.modules.orientation_module import OrientNet
        self.net = OrientNet(input_dims=[3, 8], latent_dim=8, output_dim=8,
                             mlps=[16, 8], num_neighs=3, input_neighs=3)

    def ideal_output(self, angles):
        from models.modules.orientation_module import nearest_angle_bins
        output = {}
        for direction, actual in [('x', angles), ('y', -angles)]:
            delta = actual[:, None] - self.net.ANGLE
            output[f'residual_{direction}'] = torch.stack((delta.cos(), delta.sin()), -1)
            logits = torch.full((len(angles), 8), -10.)
            logits.scatter_(1, nearest_angle_bins(actual, self.net.ANGLE)[:, None], 10)
            output[f'angle_{direction}'] = logits
        return output

    def test_continuous_alignment_wraparound_and_full_target(self):
        from models.modules.orientation_module import rotate_y_radians
        angles = torch.deg2rad(torch.tensor([20., 32., 179., 359., -179.]))
        points = torch.randn(5, 19, 3) + 7
        target = rotate_y_radians(points, angles)
        output = self.ideal_output(angles)
        aligned = self.net.align_target(target, output)
        torch.testing.assert_close(aligned, points, atol=3e-6, rtol=2e-6)
        self.assertEqual(aligned.shape, points.shape)
        coarse = self.net.ANGLE[output['angle_x'].argmax(-1)]
        self.assertGreater((coarse - angles).abs().max().item(), 0.1)

    def test_overlapping_boundary_supervises_both_residuals(self):
        from core.rotation_loss import multibin_rotation_loss
        angles = torch.deg2rad(torch.tensor([22.5]))
        output = self.ideal_output(angles)
        # The exact midpoint can use either neighboring center + correction.
        for direction, indices in [('x', [1, 2]), ('y', [0, 1])]:
            output[f'angle_{direction}'].fill_(-30)
            output[f'angle_{direction}'][:, indices] = 0
        classification, residual = multibin_rotation_loss(output, angles, self.net.ANGLE)
        self.assertAlmostEqual(classification.item(), math.log(2), places=5)
        self.assertAlmostEqual(residual.item(), 0, places=5)
        output['residual_x'] = output['residual_x'].clone()
        output['residual_x'][:, 2] *= -1
        _, worse = multibin_rotation_loss(output, angles, self.net.ANGLE)
        self.assertGreater(worse.item(), 0.4)

    def test_sampling_consistency_separates_modes_and_has_gradients(self):
        from core.rotation_loss import subset_rotation_consistency
        angles = torch.tensor([0.1])
        first = self.ideal_output(angles)
        second = {key: value.clone() for key, value in first.items()}
        self.assertAlmostEqual(subset_rotation_consistency(first, second).item(), 0, places=6)
        for direction in ('x', 'y'):
            second[f'angle_{direction}'] = second[f'angle_{direction}'].roll(4, -1).requires_grad_()
        value = subset_rotation_consistency(first, second)
        self.assertGreater(value.item(), 0.6)
        value.backward()
        self.assertTrue(torch.isfinite(second['angle_x'].grad).all())

    def test_opposite_modes_are_not_averaged(self):
        output = self.ideal_output(torch.tensor([0.]))
        output['angle_x'].fill_(-10)
        output['angle_x'][0, [1, 5]] = 10
        predicted = self.net.predict_angle(output)
        self.assertAlmostEqual(predicted.item(), 0, places=5)

    def test_joint_rotation_loss_trains_residual_head_and_encoder(self):
        from core.rotation_loss import multibin_rotation_loss
        out = self.net(torch.randn(1, 9, 3), torch.randn(1, 11, 3))
        classification, residual = multibin_rotation_loss(
            out, torch.deg2rad(torch.tensor([32.])), self.net.ANGLE)
        (classification + residual).backward()
        for parameter in [self.net.residual_head.weight, self.net.classifier.weight,
                          self.net.input_modules[0].linear.weight]:
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
