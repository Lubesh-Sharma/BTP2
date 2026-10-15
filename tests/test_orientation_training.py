"""Regression tests for the mode mismatch, coordinate frame, and curriculum."""
import copy
import unittest

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    import numpy as np
    from models.asmae import ASMAE
    from models.modules.orientation_module import OrientNet, add_noise_to_pointcloud
    from core.orientation_training import evaluate_rotation, correspondence_loss_scale, relative_rotation_pair, orientation_is_ready, oem_learning_rate


@unittest.skipIf(torch is None, 'PyTorch is not installed')
class OrientationTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.net = OrientNet(input_dims=[3, 8], latent_dim=8, output_dim=8,
                             mlps=[16, 8], num_neighs=3, input_neighs=3)

    def test_identical_train_eval_outputs_after_optimizer_step(self):
        self.assertFalse(any(isinstance(layer, torch.nn.modules.batchnorm._BatchNorm)
                             for layer in self.net.modules()))
        source, target = torch.randn(1, 9, 3), torch.randn(1, 11, 3)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=0.001)
        self.net.train()
        output = self.net(source, target)
        (output['angle_x'].square().mean() + output['global_d_pred'].square().mean()).backward()
        optimizer.step()
        train_output = self.net(source, target)
        self.net.eval()
        eval_output = self.net(source, target)
        for key in train_output:
            torch.testing.assert_close(train_output[key], eval_output[key], atol=1e-6, rtol=1e-6)
        restored = copy.deepcopy(self.net)
        restored.load_state_dict(self.net.state_dict())
        for key, value in restored(source, target).items():
            torch.testing.assert_close(value, eval_output[key])

    def test_xyz_matches_current_positions_without_mutating_targets(self):
        features = torch.randn(1, 7, 8)
        original = features.clone()
        points = torch.randn(1, 7, 3)
        points[..., 2] = 2  # Include a flat axis.
        prepared = ASMAE.features_in_coordinate_frame(features, points)
        torch.testing.assert_close(features, original)
        torch.testing.assert_close(prepared[..., :-3], original[..., :-3])
        torch.testing.assert_close(prepared[..., -1], torch.zeros(1, 7))
        lo, hi = points.amin(dim=1, keepdim=True), points.amax(dim=1, keepdim=True)
        torch.testing.assert_close(prepared[..., -3:], (points - lo) / (hi - lo).clamp_min(1e-8))

    def test_feature_extraction_ignores_stale_xyz(self):
        net = ASMAE(feature_dim=8, embed_dim=8, depth=1, num_heads=2,
                    decoder_embed_dim=8, decoder_depth=1, decoder_num_heads=2,
                    encoder_k=3, aamg_k=3, aamg_emb_dim=8,
                    pos_embed_dim=8, num_mask_queries=4).eval()
        points = torch.randn(1, 9, 3)
        features = torch.randn(1, 9, 8)
        stale = features.clone()
        stale[..., -3:] += 100
        torch.testing.assert_close(net.extract_features(features, points),
                                   net.extract_features(stale, points))

    def test_oem_lr_decay_uses_absolute_epochs(self):
        schedule = dict(start_epoch=31, duration_epochs=20, start_lr=5e-5, min_lr=1e-5)
        self.assertEqual(oem_learning_rate(29, 2.5e-4, schedule), 2.5e-4)
        self.assertAlmostEqual(oem_learning_rate(30, 2.5e-4, schedule), 5e-5)
        self.assertAlmostEqual(oem_learning_rate(49, 2.5e-4, schedule), 1e-5)
        self.assertAlmostEqual(oem_learning_rate(70, 2.5e-4, schedule), 1e-5)
        rates = [oem_learning_rate(epoch, 2.5e-4, schedule) for epoch in range(30, 50)]
        self.assertTrue(all(a >= b for a, b in zip(rates, rates[1:])))
        self.assertEqual(oem_learning_rate(20, 0.001), 0.001)
        with self.assertRaises(ValueError):
            oem_learning_rate(0, 0.001, dict(start_epoch=0, duration_epochs=2, start_lr=1e-4, min_lr=1e-5))

    def test_curriculum_boundaries(self):
        self.assertEqual(correspondence_loss_scale(9, 10, 10), 0)
        self.assertAlmostEqual(correspondence_loss_scale(10, 10, 10), 0.1)
        self.assertEqual(correspondence_loss_scale(19, 10, 10), 1)
        self.assertEqual(correspondence_loss_scale(50, 10, 10), 1)
        self.assertEqual(correspondence_loss_scale(0, 0, 0), 1)
        with self.assertRaises(ValueError):
            correspondence_loss_scale(0, -1, 10)

    def test_radius_scaled_noise_respects_scale_and_translation(self):
        points = torch.randn(1, 1024, 3, dtype=torch.float64)
        generator = torch.Generator().manual_seed(99)
        noisy = add_noise_to_pointcloud(points, 0.000004, True, generator)
        generator = torch.Generator().manual_seed(99)
        transformed = add_noise_to_pointcloud(points * 10 + 20, 0.000004, True, generator)
        torch.testing.assert_close(transformed, noisy * 10 + 20, atol=1e-10, rtol=1e-10)
        radius = (points - points.mean(dim=1, keepdim=True)).norm(dim=-1).max()
        std = ((noisy - points) / radius).std().item()
        self.assertTrue(0.0018 < std < 0.0022)

    def test_zero_noise_is_identity_and_consumes_no_rng(self):
        points = torch.randn(1, 9, 3)
        before = torch.get_rng_state().clone()
        self.assertIs(add_noise_to_pointcloud(points, 0.0, True), points)
        torch.testing.assert_close(torch.get_rng_state(), before)

    def test_sampling_independent_and_eval_reproducible_without_mutation(self):
        self.net.sample_points = 5
        source = torch.arange(45, dtype=torch.float32).reshape(1, 15, 3)
        before = source.clone()
        self.net.train()
        sampled_source, sampled_target = self.net.sample_pair(source, source)
        self.assertEqual(sampled_source.shape, (1, 5, 3))
        self.assertFalse(torch.equal(sampled_source, sampled_target))
        torch.testing.assert_close(source, before)
        self.net.eval()
        rng = torch.get_rng_state().clone()
        first = self.net.sample_pair(source, source)
        second = self.net.sample_pair(source, source)
        torch.testing.assert_close(torch.get_rng_state(), rng)
        for a, b in zip(first, second):
            torch.testing.assert_close(a, b)
        self.assertFalse(torch.equal(first[0], first[1]))

    def test_sampling_validation_compares_identical_subsets_between_modes(self):
        self.net.sample_points = 5
        self.net.eval()
        source, target = self.net.sample_pair(torch.randn(1, 15, 3), torch.randn(1, 17, 3))
        expected = self.net(source, target, sample_inputs=False)
        self.net.train()
        actual = self.net(source, target, sample_inputs=False)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])
        metrics = evaluate_rotation(self.net, [{'pos': np.random.randn(15, 3)}], 'cpu')
        self.assertEqual(metrics['mode_gap'], 0)

    def test_reference_labels_hold_for_every_start_direction(self):
        points = torch.randn(1, 9, 3)
        for base_index in range(8):
            for relative_index in range(8):
                before_rng = torch.get_rng_state().clone()
                base, relative = torch.tensor([base_index]), torch.tensor([relative_index])
                source, target = relative_rotation_pair(self.net, points, base, relative, 0.0)
                torch.testing.assert_close(torch.get_rng_state(), before_rng)
                restored = self.net.rotate_point_cloud(target, relative, inverse=True)
                torch.testing.assert_close(restored, source, atol=2e-6, rtol=2e-6)
        with self.assertRaises(ValueError):
            relative_rotation_pair(self.net, points, base, relative, -1)

    def test_readiness_requires_both_networks_and_both_directions(self):
        good = dict(accuracy=0.9, reverse_accuracy=0.9, angle_error_degrees=10)
        metrics = dict(student_validation=good.copy(), teacher_validation=good.copy())
        self.assertTrue(orientation_is_ready(metrics, 0.85, 15))
        metrics['teacher_validation']['reverse_accuracy'] = 0.5
        self.assertFalse(orientation_is_ready(metrics, 0.85, 15))
        self.assertFalse(orientation_is_ready({}, 0.85, 15))
        metrics['teacher_validation'] = good.copy()
        metrics['student_validation']['angle_error_degrees'] = 20
        self.assertFalse(orientation_is_ready(metrics, 0.85, 15))

    def test_noisy_validation_reproducible_without_changing_training_rng(self):
        shapes = [{'pos': np.random.randn(9, 3)}]
        before = torch.get_rng_state().clone()
        first = evaluate_rotation(self.net, shapes, 'cpu', base_bins=[1, 5], noise_variance=0.0001)
        torch.testing.assert_close(torch.get_rng_state(), before)
        second = evaluate_rotation(self.net, shapes, 'cpu', base_bins=[1, 5], noise_variance=0.0001)
        self.assertEqual(first, second)
        self.assertEqual(first['count'], 16)

    def test_validation_restores_mode_and_does_not_change_weights(self):
        self.net.train()
        before = copy.deepcopy(self.net.state_dict())
        metrics = evaluate_rotation(self.net, [{'pos': np.random.randn(9, 3)}], 'cpu')
        self.assertTrue(self.net.training)
        self.assertLess(metrics['mode_gap'], 1e-6)
        self.assertTrue(0 <= metrics['accuracy'] <= 1)
        self.assertTrue(0 <= metrics['angle_error_degrees'] <= 180.0001)
        for key, value in self.net.state_dict().items():
            torch.testing.assert_close(value, before[key])


if __name__ == '__main__':
    unittest.main()
