"""Single-encoder ASMAE + OEM training (no teacher / EMA / Sinkhorn).

Stage 1 (OEM only, `orientation_warmup_epochs`): angle + domain losses train OEM.
Stage 2 (OEM + ASMAE): the encoder is ramped in on OEM-aligned pairs with
    rec      masked feature reconstruction (both directions)           [unchanged]
    contra   contrastive spread of embeddings                          [unchanged]
    cycle    softmax cycle consistency on coordinates                  [unchanged]
    orient   signed-area orientation consistency                       [unchanged]
    dist     metric distortion, now intrinsic (biharmonic) by default  [changed]
    self     InfoNCE between two augmented views of the same shape     [new]
    dino     DINO distillation, confidence-weighted                    [new]
Removed: teacher consistency loss, EMA teacher (ASMAE and OEM), Sinkhorn L_go.

Validation (core/validation.py) runs every `validation.every_iters` encoder steps and at
every epoch end of stage 2, using the full inference pipeline (OEM + matcher):
    checkpoints/<name>_best.pth   lowest validation error so far
    checkpoints/<name>_final.pth  latest state (also used to resume)

Usage:
    python train.py --config config/FAUST/train_config.yaml [--fresh]
"""
import os
import random
import argparse

import numpy as np
import torch
import yaml

from models.asmae import build_asmae
from models.modules.orientation_module import OrientNet, DataAugment, normalize_pc_torch
from core.contrastive_loss import compute_contrastive_loss
from core.cycle_loss import compute_cycle_loss
from core.distortion_loss import compute_distortion_loss
from core.orientation_loss import compute_orientation_loss
from core.intrinsic_losses import (compute_biharmonic_distortion_loss, compute_self_identity_loss,
                                   compute_dino_distill_loss)
from core.orientation_training import (evaluate_rotation, correspondence_loss_scale, relative_rotation_pair,
                                       oem_learning_rate, orientation_loss_weights)
from core.domain_loss import compute_domain_loss, FocalLoss
from core.rotation_loss import multibin_rotation_loss, subset_rotation_consistency
from core.validation import CorrespondenceValidator
from utils.data import dataset_split, load_shapes, rebuild_training_data
from utils.logger import start_logging, save_json

CHECKPOINT_VERSION = 'single-encoder-v1'


def to_t(array, device):
    return torch.as_tensor(np.asarray(array), dtype=torch.float32, device=device).unsqueeze(0)


def train_model(model, orient_net, train_shapes, validation_shapes, config, resume=None):
    tc = config['training']
    device = tc.get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    resume = resume or {}

    num_epochs = tc.get('epochs', 10)
    lr = tc['lr']
    oem_lr_schedule = tc.get('oem_lr_schedule')
    mask_ratio = tc.get('mask_ratio', tc.get('student_mask_ratio', 0.6))
    feat_ratio = tc.get('feature_ratio', tc.get('student_feature_ratio', 0.35))

    contra_margin = tc.get('contrastive_margin', 0.25)
    contra_weight = tc.get('contrastive_weight', 10.0)
    cycle_eps = tc.get('cycle_eps', 0.02)
    cycle_weight = tc.get('cycle_weight', 100.0)
    orient_weight = tc.get('orientation_weight', 100.0)
    dist_weight = tc.get('distortion_weight', 120.0)
    dist_tau = tc.get('distortion_tau', 0.04)
    dist_samples = tc.get('distortion_samples', 384)
    dist_mode = tc.get('distortion_mode', 'biharmonic')
    self_weight = tc.get('self_identity_weight', 5.0)
    self_cfg = dict(keep_ratio=tc.get('self_identity_keep_ratio', 0.8),
                    max_rot_deg=tc.get('self_identity_max_rot_deg', 15.0),
                    noise_std=tc.get('self_identity_noise_std', 0.003),
                    num_queries=tc.get('self_identity_queries', 1024),
                    temperature=tc.get('self_identity_temperature', 0.07))
    dino_weight = tc.get('dino_weight', 50.0) if model.dino_head is not None else 0.0

    angle_weight_target = tc.get('angle_weight', 2.0)
    angle_weight_init = tc.get('angle_weight_init', 8.0)
    angle_warmup_epochs = tc.get('angle_warmup_epochs', 5)
    domain_weight = tc.get('domain_weight', 1.0)
    residual_weight = tc.get('oem_residual_weight', 1.0)
    subset_weight = tc.get('oem_subset_consistency_weight', 0.1)
    oem_warmup = tc.get('orientation_warmup_epochs', 10)
    oem_transition = tc.get('orientation_transition_epochs', 10)
    noise_variance = tc.get('noise_variance', 0.0)
    noise_rel = tc.get('noise_relative_to_radius', False)
    validation_base_bins = tc.get('orientation_validation_base_bins', list(range(orient_net.num_class)))
    validation_every = tc.get('validation_every', 1)          # epochs (OEM rotation report)
    val_every_iters = config.get('validation', {}).get('every_iters', 500)  # encoder steps
    result_dir = config.get('result_dir', 'results')
    base = os.path.splitext(os.path.join(tc['checkpoint_dir'], tc['checkpoint_name']))[0]
    final_path, best_path = base + '_final.pth', base + '_best.pth'

    print(f"\n{'=' * 60}\nSINGLE-ENCODER TRAINING (OEM-first schedule)")
    print(f"  Train shapes: {len(train_shapes)} | Validation: {len(validation_shapes)} | Device: {device}")
    print(f"  OEM-only epochs 1-{oem_warmup}; encoder ramp over {oem_transition} epochs; total {num_epochs}")
    print(f"  Mask ratio (nodes/feats): {mask_ratio} / {feat_ratio}")
    print(f"  Weights: contra={contra_weight} cycle={cycle_weight} dist[{dist_mode}]={dist_weight} "
          f"orient={orient_weight} self={self_weight} dino={dino_weight}")
    print('=' * 60)

    optimizer = torch.optim.Adam([
        {'params': list(model.parameters()), 'lr': lr, 'name': 'asmae'},
        {'params': list(orient_net.parameters()), 'lr': lr, 'name': 'oem'},
    ], lr=lr)
    criterion = torch.nn.L1Loss()
    fl_global = FocalLoss(class_num=2, gamma=3)
    src_aug = DataAugment(operations=["noise"] if noise_variance > 0 else [], noise_variance=noise_variance,
                          noise_relative_to_radius=noise_rel)
    tgt_aug = DataAugment(operations=["rotate", "noise"] if noise_variance > 0 else ["rotate"],
                          rotate_nbins=orient_net.num_class, noise_variance=noise_variance,
                          noise_relative_to_radius=noise_rel)
    model.to(device)
    orient_net.to(device)
    orient_net.train()

    if resume:
        optimizer.load_state_dict(resume['optimizer_state_dict'])
    start_epoch = resume.get('completed_epochs', 0)
    state = {'best_metric': resume.get('best_metric', float('inf')),
             'best_step': resume.get('best_step', -1),
             'global_step': resume.get('global_step', 0),
             'history': resume.get('history', [])}
    validator = CorrespondenceValidator(config, validation_shapes, device)

    def save(path, completed_epochs, tag):
        payload = {
            'version': CHECKPOINT_VERSION,
            'model_state_dict': model.state_dict(),
            'orient_net_state_dict': orient_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'completed_epochs': completed_epochs,
            'feature_dim': train_shapes[0]['feat'].shape[1],
            'dino_dim': model.dino_dim,
            'train_names': [s['name'] for s in train_shapes],
            'validation_names': [s['name'] for s in validation_shapes],
            'config': config,
            **state,
        }
        torch.save(payload, path + '.tmp')
        os.replace(path + '.tmp', path)
        print(f"  [{tag}] checkpoint -> {path}")

    def validate(epoch, completed_epochs, where):
        """Correspondence validation; refresh final, and best when improved."""
        result = validator.run(model, orient_net)
        state['history'].append({'epoch': epoch + 1, 'step': state['global_step'], **result})
        save_json({'best_metric': state['best_metric'], 'best_step': state['best_step'],
                   'history': state['history']}, os.path.join(result_dir, 'validation_history.json'))
        improved = result['metric'] < state['best_metric']
        print(f"  VALIDATION ({where}, epoch {epoch + 1}, step {state['global_step']}): "
              f"{result['mode']} error = {result['metric']:.5f} | best = "
              f"{min(state['best_metric'], result['metric']):.5f}{'  * new best' if improved else ''}")
        if improved:
            state['best_metric'], state['best_step'] = result['metric'], state['global_step']
            save(best_path, completed_epochs, 'best')
        save(final_path, completed_epochs, 'final')

    keys = ['tot', 'rec', 'contra', 'cycle', 'dist', 'orient', 'self', 'dino', 'angle', 'domain']
    try:
        for epoch in range(start_epoch, num_epochs):
            scale = correspondence_loss_scale(epoch, oem_warmup, oem_transition)
            model.requires_grad_(scale > 0)
            model.train(scale > 0)
            # The existing absolute-epoch cosine helper also schedules the encoder.
            optimizer.param_groups[0]['lr'] = oem_learning_rate(epoch, lr, tc.get('encoder_lr_schedule')) * scale
            optimizer.param_groups[1]['lr'] = oem_learning_rate(epoch, lr, oem_lr_schedule)
            w_angle, w_domain = orientation_loss_weights(epoch, angle_weight_init, angle_weight_target,
                                                         angle_warmup_epochs, domain_weight)
            print(f"Epoch {epoch + 1}: {'OEM only' if scale == 0 else 'OEM + encoder'} | encoder scale={scale:.2f} "
                  f"| enc lr={optimizer.param_groups[0]['lr']:.2e} | OEM lr={optimizer.param_groups[1]['lr']:.2e}")
            sums = {k: 0.0 for k in keys}
            n_pairs = 0
            pairs = [(i, j) for i in range(len(train_shapes)) for j in range(len(train_shapes)) if i != j]
            np.random.shuffle(pairs)
            if tc.get('pairs_per_epoch'):
                pairs = pairs[:tc['pairs_per_epoch']]      # random subset of ordered pairs per epoch

            for i1, i2 in pairs:
                s1, s2 = train_shapes[i1], train_shapes[i2]
                p1_raw, f1 = to_t(s1['pos'], device), to_t(s1['feat'], device)
                p2_raw, f2 = to_t(s2['pos'], device), to_t(s2['feat'], device)

                # ---- OEM (unchanged objective) ----
                p1_in = src_aug(p1_raw)
                p2_in, _ = tgt_aug(p2_raw)
                p1n, p2n = normalize_pc_torch(p1_in), normalize_pc_torch(p2_in)
                orient_out = orient_net(p1n, p2n)
                base_bin = torch.randint(orient_net.num_class, (1,), device=device)
                rel_rad = torch.rand(1, device=device) * (2 * torch.pi)
                ref_s, ref_t = relative_rotation_pair(orient_net, p2_raw, base_bin, None, noise_variance,
                                                      noise_relative_to_radius=noise_rel, relative_radians=rel_rad)
                ref_s, ref_t = normalize_pc_torch(ref_s), normalize_pc_torch(ref_t)
                out_ref = orient_net(ref_s, ref_t)
                l_cls, l_res = multibin_rotation_loss(out_ref, rel_rad, orient_net.ANGLE, orient_net.angle_overlap)
                l_sub = l_cls.new_zeros(())
                if subset_weight > 0:
                    out_ref2 = orient_net(ref_s, ref_t)
                    out_cross2 = orient_net(p1n, p2n)
                    c2, r2 = multibin_rotation_loss(out_ref2, rel_rad, orient_net.ANGLE, orient_net.angle_overlap)
                    l_cls, l_res = (l_cls + c2) / 2, (l_res + r2) / 2
                    l_sub = (subset_rotation_consistency(out_ref, out_ref2)
                             + subset_rotation_consistency(orient_out, out_cross2)) / 2
                loss_angle = l_cls + residual_weight * l_res + subset_weight * l_sub
                loss_domain = compute_domain_loss(out_ref["global_d_pred"], orient_out["global_d_pred"], fl_global)
                loss_oem = w_angle * loss_angle + w_domain * loss_domain

                if scale == 0:
                    optimizer.zero_grad(set_to_none=True)
                    loss_oem.backward()
                    optimizer.step()
                    sums['tot'] += loss_oem.item(); sums['angle'] += loss_angle.item()
                    sums['domain'] += loss_domain.item(); n_pairs += 1
                    continue

                with torch.no_grad():
                    p2_al = orient_net.align_target(p2_in, orient_out)
                p1_al = p1_in

                # ---- masked reconstruction (both directions) ----
                pred1, _, _, _, _, m1 = model(f1, p1_al, f2, p2_al, mask_ratio=mask_ratio, feature_ratio=feat_ratio)
                pred2, _, _, _, _, m2 = model(f2, p2_al, f1, p1_al, mask_ratio=mask_ratio, feature_ratio=feat_ratio)
                loss_rec = ((criterion(pred1[m1], f1[m1]) if m1.sum() > 0 else criterion(pred1, f1)) +
                            (criterion(pred2[m2], f2[m2]) if m2.sum() > 0 else criterion(pred2, f2)))

                z1 = model.extract_features(f1, p1_al)
                z2 = model.extract_features(f2, p2_al)
                loss_contra = (compute_contrastive_loss(z1, margin=contra_margin)
                               + compute_contrastive_loss(z2, margin=contra_margin))
                loss_cycle = (compute_cycle_loss(z1, z2, p1_al, eps=cycle_eps)
                              + compute_cycle_loss(z2, z1, p2_al, eps=cycle_eps))
                if dist_mode == 'biharmonic':
                    psi1, psi2 = to_t(s1['psi'], device), to_t(s2['psi'], device)
                    loss_dist = (compute_biharmonic_distortion_loss(z1, z2, psi1, psi2, dist_samples, dist_tau)
                                 + compute_biharmonic_distortion_loss(z2, z1, psi2, psi1, dist_samples, dist_tau)) / 2
                else:
                    loss_dist = (compute_distortion_loss(z1, z2, p1_al, p2_al, dist_samples, dist_tau)
                                 + compute_distortion_loss(z2, z1, p2_al, p1_al, dist_samples, dist_tau)) / 2
                loss_orient = (compute_orientation_loss(z1, z2, p1_al, p2_al, num_samples=500, tau=dist_tau)
                               + compute_orientation_loss(z2, z1, p2_al, p1_al, num_samples=500, tau=dist_tau)) / 2
                loss_self = (compute_self_identity_loss(model, f1, p1_al, z_full=z1, **self_cfg)
                             if self_weight > 0 else z1.new_zeros(()))
                loss_dino = z1.new_zeros(())
                if dino_weight > 0:
                    loss_dino = (compute_dino_distill_loss(model.predict_dino(z1)[0], to_t(s1['dino'], device)[0],
                                                           to_t(s1['dino_conf'], device)[0])
                                 + compute_dino_distill_loss(model.predict_dino(z2)[0], to_t(s2['dino'], device)[0],
                                                             to_t(s2['dino_conf'], device)[0])) / 2

                loss_corr = (loss_rec + contra_weight * loss_contra + cycle_weight * loss_cycle
                             + dist_weight * loss_dist + orient_weight * loss_orient
                             + self_weight * loss_self + dino_weight * loss_dino)
                loss = scale * loss_corr + loss_oem
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                state['global_step'] += 1

                for k, v in [('tot', loss), ('rec', loss_rec), ('contra', loss_contra), ('cycle', loss_cycle),
                             ('dist', loss_dist), ('orient', loss_orient), ('self', loss_self), ('dino', loss_dino),
                             ('angle', loss_angle), ('domain', loss_domain)]:
                    sums[k] += v.item()
                n_pairs += 1
                if val_every_iters and state['global_step'] % val_every_iters == 0:
                    # mid-epoch: a resume restarts this (incomplete) epoch
                    validate(epoch, epoch, f"every {val_every_iters} iters")

            print("  " + " | ".join(f"{k}: {sums[k] / max(1, n_pairs):.4f}" for k in keys))
            if (epoch + 1) % validation_every == 0 or epoch + 1 == num_epochs:
                for split, shapes in [('train_sample', train_shapes[:2]), ('validation', validation_shapes)]:
                    r = evaluate_rotation(orient_net, shapes, device, base_bins=validation_base_bins)
                    print(f"  OEM {split}: acc={r['accuracy']:.1%} | angle err={r['angle_error_degrees']:.2f} deg | "
                          f"within 15 deg={r['accuracy_15deg']:.1%} | >90 deg={r['large_error_fraction']:.1%}")
            if scale > 0:
                validate(epoch, epoch + 1, 'end of epoch')       # also writes final
            else:
                save(final_path, epoch + 1, 'final')              # OEM-only phase: no encoder to rank yet
    except KeyboardInterrupt:
        print("Training interrupted; resume from the final checkpoint.")
    print(f"\nDone. Best {validator.mode} error {state['best_metric']:.5f} at step {state['best_step']}")
    print(f"  best : {best_path}\n  final: {final_path}")
    save_json({'best_metric': state['best_metric'], 'best_step': state['best_step'], 'metric': validator.mode,
               'best_checkpoint': best_path, 'final_checkpoint': final_path, 'history': state['history']},
              os.path.join(result_dir, 'train_summary.json'))
    return model, orient_net


def main():
    parser = argparse.ArgumentParser(description="Single-encoder ASMAE + OEM training")
    parser.add_argument('--config', default='config/FAUST/train_config.yaml')
    parser.add_argument('--fresh', action='store_true', help='ignore an existing final checkpoint')
    parser.add_argument('--rebuild-data', action='store_true',
                        help='rebuild OBJ kNN graphs and geometry caches using this training YAML')
    args = parser.parse_args()
    with open(args.config) as f:
        config = yaml.safe_load(f)
    result_dir = config.get('result_dir', 'results')
    start_logging(result_dir, 'train')
    save_json(config, os.path.join(result_dir, 'train_config_used.json'))

    seed = config['training'].get('seed', 42)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)

    if args.rebuild_data:
        rebuild_training_data(config)
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    train_files, val_files, test_files = dataset_split(config)
    print(f"Split: {len(train_files)} train, {len(val_files)} validation, {len(test_files)} test")
    use_dino = config.get('dino', {}).get('enabled', False) and config['training'].get('dino_weight', 50.0) > 0
    train_shapes = load_shapes(config, train_files, with_dino=use_dino, label='train shapes')
    val_dino = use_dino and config.get('matching', {}).get('w_dino', 1.0) > 0
    val_shapes = load_shapes(config, val_files, with_dino=val_dino, label='validation shapes')
    if len(train_shapes) < 2 or not val_shapes:
        raise ValueError("Need at least two training shapes and one validation shape.")

    feature_dim = train_shapes[0]['feat'].shape[1]
    dino_dim = train_shapes[0]['dino'].shape[1] if use_dino else 0
    model = build_asmae(config.get('model', config.get('student_model')), feature_dim, dino_dim)
    orient_net = OrientNet(**config.get('orientation_model', {}))

    ckpt = os.path.splitext(os.path.join(config['training']['checkpoint_dir'],
                                         config['training']['checkpoint_name']))[0] + '_final.pth'
    resume = None
    if not args.fresh and os.path.exists(ckpt):
        resume = torch.load(ckpt, map_location='cpu', weights_only=False)
        if resume.get('version') != CHECKPOINT_VERSION:
            raise ValueError(f"{ckpt} was written by another pipeline; use --fresh.")
        model.load_state_dict(resume['model_state_dict'])
        orient_net.load_state_dict(resume['orient_net_state_dict'])
        print(f"Resuming after epoch {resume['completed_epochs']}")
    print(f"  Feature dim {feature_dim} | DINO dim {dino_dim} | encoder params "
          f"{sum(p.numel() for p in model.parameters()):,} | OEM params {sum(p.numel() for p in orient_net.parameters()):,}")
    train_model(model, orient_net, train_shapes, val_shapes, config, resume)


if __name__ == "__main__":
    main()
