"""Small end-to-end training, checkpoint, and inference regression test (CPU)."""
import contextlib
import copy
import io
import random
import tempfile
import unittest
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
            oem = OrientNet()
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
                          student_model=model_config, teacher_model=model_config,
                          training=dict(epochs=1, lr=0.0001, device='cpu',
                                        checkpoint_dir=directory, checkpoint_name='resume.pth',
                                        orientation_warmup_epochs=1, orientation_transition_epochs=1,
                                        cycle_n_iter=3, distortion_samples=16))
            seed()
            first = networks()
            train_model(*first, shapes[:2], config, validation_shapes=shapes[2:])
            checkpoint = torch.load(Path(directory) / 'resume.pth', weights_only=True)
            self.assertEqual(checkpoint['completed_epochs'], 1)
            self.assertEqual(checkpoint['oem_version'], 3)
            self.assertEqual(checkpoint['rotation_metrics']['student_validation']['mode_gap'], 0)
            self.assertTrue(checkpoint['optimizer_state_dict']['state'])

            resumed = networks()
            for module, key in zip(resumed, ('model_state_dict', 'teacher_state_dict',
                                            'orient_net_state_dict', 'teacher_orient_net_state_dict')):
                module.load_state_dict(checkpoint[key])
            config['training']['epochs'] = 2
            train_model(*resumed, shapes[:2], config, validation_shapes=shapes[2:], resume=checkpoint)
            final = torch.load(Path(directory) / 'resume.pth', weights_only=True)
            self.assertEqual(final['completed_epochs'], 2)

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
                    for loaded, expected in [(asmae, resumed[offset]), (oem, resumed[2 + offset])]:
                        for key, value in loaded.state_dict().items():
                            torch.testing.assert_close(value, expected.state_dict()[key])


if __name__ == '__main__':
    unittest.main()
