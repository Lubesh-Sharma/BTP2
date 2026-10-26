"""Small end-to-end training, checkpoint, and inference regression test (CPU)."""
import contextlib
import copy
import io
import random
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

try:
    import torch
    import numpy as np
    from train_st_te import train_model
    from corres_st_te import load_models
    from corres_greedy import load_model
    from models.asmae import ASMAE
    from models.modules.orientation_module import OrientNet
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'Training dependencies are unavailable')
class TrainingResumeTests(unittest.TestCase):
    def test_fresh_resume_matches_uninterrupted_and_both_inference_loaders(self):
        torch.set_num_threads(1)
        model_config = dict(embed_dim=8, depth=1, num_heads=2, decoder_embed_dim=8,
                            decoder_depth=1, decoder_num_heads=2, mlk_ratio=2,
                            num_mask_queries=8, encoder_k=3, aamg_k=3,
                            aamg_emb_dim=8, pos_embed_dim=8, temperature=0.1)
        rng = np.random.default_rng(3)
        shapes = [dict(name=str(i), pos=rng.normal(size=(32 + i, 3)).astype('float32'),
                       feat=rng.normal(size=(32 + i, 8)).astype('float32')) for i in range(3)]

        def networks():
            student = ASMAE(feature_dim=8, **model_config)
            teacher = copy.deepcopy(student)
            oem = OrientNet(sample_fraction=0.70, sample_min_points=27, sampling_seed=2026, angle_overlap=1.25)
            teacher_oem = copy.deepcopy(oem)
            for module in (teacher, teacher_oem):
                for parameter in module.parameters():
                    parameter.requires_grad_(False)
            return student, teacher, oem, teacher_oem

        def seed():
            random.seed(42)
            np.random.seed(42)
            torch.manual_seed(42)

        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            config = dict(geometry=dict(k=5, t=1, neigvecs=8),
                          orientation_model=dict(sample_fraction=0.70, sample_min_points=27, sampling_seed=2026, angle_overlap=1.25),
                          student_model=model_config, teacher_model=model_config,
                          training=dict(epochs=1, lr=0.0001, device='cpu',
                                        checkpoint_dir=directory, checkpoint_name='resume.pth',
                                        orientation_warmup_epochs=1, orientation_transition_epochs=1,
                                        orientation_validation_base_bins=[1],
                                        noise_variance=0.0,
                                        cycle_n_iter=3, distortion_samples=16))
            seed()
            first = networks()
            initial_asmae = [copy.deepcopy(module.state_dict()) for module in first[:2]]
            initial_oem = copy.deepcopy(first[2].state_dict())
            # The warm-up must never call either ASMAE, even in no-grad mode.
            def reject_asmae_forward(module, inputs):
                raise AssertionError("ASMAE was called during OEM-only warm-up")
            hooks = [module.register_forward_pre_hook(reject_asmae_forward) for module in first[:2]]
            train_model(*first, shapes[:2], config, validation_shapes=shapes[2:])
            for hook in hooks:
                hook.remove()
            for module, before in zip(first[:2], initial_asmae):
                for key, value in module.state_dict().items():
                    torch.testing.assert_close(value, before[key], rtol=0, atol=0)
                self.assertTrue(all(parameter.grad is None for parameter in module.parameters()))
            self.assertTrue(any(not torch.equal(value, initial_oem[key])
                                for key, value in first[2].state_dict().items()))
            self.assertFalse(torch.equal(first[2].residual_head.weight,
                                         initial_oem['residual_head.weight']))
            self.assertFalse(torch.equal(first[2].global_netD2.weight,
                                         initial_oem['global_netD2.weight']))
            self.assertIsNotNone(first[2].global_netD2.weight.grad)
            self.assertGreater(first[2].global_netD2.weight.grad.abs().sum().item(), 0)
            checkpoint = torch.load(Path(directory) / 'resume.pth', weights_only=True)
            self.assertEqual(checkpoint['completed_epochs'], 1)
            self.assertEqual(checkpoint['oem_version'], 10)
            self.assertEqual(checkpoint['joint_start_epoch'], 1)
            self.assertTrue(checkpoint['optimizer_state_dict']['state'])
            self.assertEqual(checkpoint['optimizer_state_dict']['param_groups'][0]['lr'], 0)
            student_ids = checkpoint['optimizer_state_dict']['param_groups'][0]['params']
            self.assertTrue(all(key not in checkpoint['optimizer_state_dict']['state'] for key in student_ids))

            resumed = networks()
            for module, key in zip(resumed, ('model_state_dict', 'teacher_state_dict',
                                            'orient_net_state_dict', 'teacher_orient_net_state_dict')):
                module.load_state_dict(checkpoint[key])
            config['training']['epochs'] = 2
            config['training']['oem_lr_schedule'] = dict(start_epoch=2, duration_epochs=3, start_lr=5e-5, min_lr=1e-5)
            full_shape_sizes = []
            def record_full_shapes(module, inputs):
                full_shape_sizes.append((inputs[1].shape[1], inputs[3].shape[1]))
            shape_hooks = [module.register_forward_pre_hook(record_full_shapes) for module in resumed[:2]]
            train_model(*resumed, shapes[:2], config, validation_shapes=shapes[2:], resume=checkpoint)
            for hook in shape_hooks:
                hook.remove()
            self.assertTrue(full_shape_sizes)
            self.assertEqual(set(full_shape_sizes), {(32, 33), (33, 32)})
            final = torch.load(Path(directory) / 'resume.pth', weights_only=True)
            self.assertEqual(final['completed_epochs'], 2)
            self.assertEqual(final['optimizer_state_dict']['param_groups'][0]['lr'], 0.0001)
            self.assertAlmostEqual(final['optimizer_state_dict']['param_groups'][1]['lr'], 5e-5)
            self.assertTrue(any(not torch.equal(value, initial_asmae[0][key])
                                for key, value in resumed[0].state_dict().items()))

            seed()
            uninterrupted = networks()
            full_config = copy.deepcopy(config)
            full_config['training']['checkpoint_name'] = 'full.pth'
            train_model(*uninterrupted, shapes[:2], full_config, validation_shapes=shapes[2:])
            for actual, expected in zip(resumed, uninterrupted):
                for key, value in actual.state_dict().items():
                    torch.testing.assert_close(value, expected.state_dict()[key], rtol=0, atol=0)

            inference_config = copy.deepcopy(config)
            inference_config['correspondence'] = dict(checkpoint_path=str(Path(directory) / 'resume.pth'))
            for network, offset in [('student', 0), ('teacher', 1)]:
                for loader in (load_models, load_model):
                    asmae, oem = loader(inference_config, 8, 'cpu', network=network)
                    self.assertFalse(asmae.training)
                    self.assertFalse(oem.training)
                    self.assertEqual(oem.sample_fraction, 0.70)
                    self.assertEqual(oem.angle_overlap, 1.25)
                    for loaded, expected in [(asmae, resumed[offset]), (oem, resumed[2 + offset])]:
                        for key, value in loaded.state_dict().items():
                            torch.testing.assert_close(value, expected.state_dict()[key])



if __name__ == '__main__':
    unittest.main()
