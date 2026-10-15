import os
import torch
import numpy as np
import sys
import yaml
import argparse
import copy
import random

from core.preprocessing import process_geometry, normalize_pc
from models.asmae import ASMAE
from models.modules.orientation_module import OrientNet, DataAugment, normalize_pc_torch
from core.consistency_loss import compute_consistency_loss
from core.contrastive_loss import compute_contrastive_loss
from core.cycle_loss import compute_cycle_loss
from core.lgo_loss import compute_lgo_loss
from core.distortion_loss import compute_distortion_loss
from core.orientation_loss import compute_orientation_loss
from core.orientation_training import (CHECKPOINT_VERSION, evaluate_rotation,
                                       predicted_alignment_probability)
from core.domain_loss import compute_domain_loss, compute_angle_loss, FocalLoss


def update_teacher_ema(student, teacher, alpha=0.999):
    """
    Updates the teacher parameters using Exponential Moving Average (EMA) of student parameters.
    """
    with torch.no_grad():
        student_state = student.state_dict()
        for name, value in teacher.state_dict().items():
            if value.is_floating_point():
                value.mul_(alpha).add_(student_state[name], alpha=1.0 - alpha)
            else:
                value.copy_(student_state[name])


def load_train_shapes(data_dir, k, t, neigvecs, max_shapes, output_dir, filenames=None):
    """Load all .obj files from directory"""
    print(f"\n{'='*60}")
    print(f"Loading {max_shapes} train shapes from: {data_dir}")
    print(f"{'='*60}")
    
    obj_files = filenames if filenames is not None else sorted([f for f in os.listdir(data_dir) if f.endswith('.obj')])
    if max_shapes:
        obj_files = obj_files[:max_shapes]
        
    shapes = []
    for i, obj_file in enumerate(obj_files):
        print(f"[{i+1}/{len(obj_files)}] Loading: {obj_file}...", end=' ')
        sys.stdout.flush()
        try:
            path = os.path.join(data_dir, obj_file)
            name = os.path.splitext(obj_file)[0]
            VPos, El, Feat, eigvecs = process_geometry(path, k, t, neigvecs, output_dir=output_dir)
            if VPos is None or len(VPos) == 0:
                print("SKIP (no vertices)")
                continue
            if Feat is None or len(Feat) == 0:
                print("SKIP (no features)")
                continue
            if El is None or len(El) == 0:
                El = np.array([])
            shapes.append({
                'name': name,
                'pos': VPos,
                'el': El,
                'feat': Feat,
                'eigvecs': eigvecs
            })
            print(f"OK ({len(VPos)} vertices, {Feat.shape[1]} features)")
        except Exception as e:
            print(f"ERROR: {str(e)[:50]}")
            continue
    print(f"\nSuccessfully loaded {len(shapes)} shapes")
    return shapes


def train_model(student, teacher, orient_net, teacher_orient_net, train_shapes, config, validation_shapes=None, resume=None):
    validation_shapes = validation_shapes or []
    resume = resume or {}
    device = config['training'].get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    num_epochs = config['training'].get('epochs', 10)
    lr = config['training']['lr']
    
    # Extract student params
    student_mask_ratio = config['training'].get('student_mask_ratio', 0.6)
    student_feat_ratio = config['training'].get('student_feature_ratio', 0.35)
    
    # Extract teacher params
    teacher_mask_ratio = config['training'].get('teacher_mask_ratio', 0.1)
    teacher_feat_ratio = config['training'].get('teacher_feature_ratio', 0.1)
    
    # Loss configs
    ema_alpha = config['training'].get('ema_alpha', 0.999)
    cons_weight = config['training'].get('consistency_weight', 50.0)
    contra_margin = config['training'].get('contrastive_margin', 0.25)
    contra_weight = config['training'].get('contrastive_weight', 10.0)
    cycle_eps = config['training'].get('cycle_eps', 0.02)
    cycle_n_iter = config['training'].get('cycle_n_iter', 15)
    cycle_weight = config['training'].get('cycle_weight', 100.0)
    
    # Asymmetric distillation temperatures to drive Lgo down
    student_lgo_eps = config['training'].get('lgo_eps', 0.04)
    target_lgo_eps = config['training'].get('lgo_target_eps', 0.03)
    lgo_weight_target = config['training'].get('lgo_weight', 40.0)
    lgo_weight_init = config['training'].get('lgo_weight_init', 10.0)
    lgo_warmup_epochs = config['training'].get('lgo_warmup_epochs', 5)
    
    orient_weight = config['training'].get('orientation_weight', 100.0)
    dist_weight = config['training'].get('distortion_weight', 120.0)
    dist_tau = config['training'].get('distortion_tau', 0.04)
    dist_samples = config['training'].get('distortion_samples', 384)
    
    # SE-ORNet Orientation & Domain Loss weights with smooth warmup annealing
    angle_weight_target = config['training'].get('angle_weight', 2.0)
    angle_weight_init = config['training'].get('angle_weight_init', 8.0)
    angle_warmup_epochs = config['training'].get('angle_warmup_epochs', 5)
    domain_weight = config['training'].get('domain_weight', 1.0)
    oem_warmup = config['training'].get('orientation_warmup_epochs', 10)
    oem_transition = config['training'].get('orientation_transition_epochs', 10)
    validation_every = config['training'].get('validation_every', 1)
    if validation_every < 1:
        raise ValueError("validation_every must be positive.")
    # Validate durations before starting the expensive training loop.
    predicted_alignment_probability(0, oem_warmup, oem_transition)
    checkpoint_path = os.path.join(config['training']['checkpoint_dir'],
                                   config['training']['checkpoint_name'])
    
    print(f"\n{'='*60}")
    print("STUDENT-TEACHER TRAINING PHASE (WITH DYNAMIC ORIENTATION WARMUP)")
    print(f"  Training shapes: {len(train_shapes)} | Validation shapes: {len(validation_shapes)}")
    print(f"  Known alignment for {oem_warmup} epochs, then {oem_transition} epoch transition to predicted alignment")
    print("  OEM uses GroupNorm: no running-statistics or inference overrides")
    print(f"  Epochs: {num_epochs}")
    print(f"  Device: {device}")
    print(f"  Student Masking Ratio (Nodes/Feats): {student_mask_ratio} / {student_feat_ratio}")
    print(f"  Teacher Masking Ratio (Nodes/Feats): {teacher_mask_ratio} / {teacher_feat_ratio}")
    print(f"  EMA Alpha: {ema_alpha} | Consist. Weight: {cons_weight}")
    print(f"  Cycle Weight: {cycle_weight} | Dist. Weight: {dist_weight} | Orient Weight: {orient_weight}")
    print(f"  Angle Weight: {angle_weight_init} -> {angle_weight_target} (over {angle_warmup_epochs} epochs smooth decay)")
    print(f"  LGO Weight:   {lgo_weight_init} -> {lgo_weight_target} (over {lgo_warmup_epochs} epochs smooth ramp-up)")
    print(f"  LGO Temperatures: student_eps={student_lgo_eps}, target_eps={target_lgo_eps}")
    print(f"{'='*60}\n")
    
    # Joint optimizer for ASMAE student and OrientNet
    optimizer = torch.optim.Adam(
        list(student.parameters()) + list(orient_net.parameters()),
        lr=lr
    )
    criterion = torch.nn.L1Loss()
    fl_global = FocalLoss(class_num=2, gamma=3)
    
    # Point cloud data augmentations
    src_aug = DataAugment(operations=["noise"], noise_variance=0.0001)
    tgt_aug = DataAugment(operations=["rotate", "noise"], rotate_nbins=8, noise_variance=0.0001)
    
    student.to(device)
    teacher.to(device)
    orient_net.to(device)
    teacher_orient_net.to(device)
    
    student.train()
    orient_net.train()
    teacher_orient_net.eval()
    teacher.eval() # Teacher does not get trained by backprop natively
    
    if resume:
        optimizer.load_state_dict(resume['optimizer_state_dict'])
        torch.set_rng_state(resume['torch_rng_state'].cpu())
        if torch.cuda.is_available() and resume.get('cuda_rng_state'):
            torch.cuda.set_rng_state_all([value.cpu() for value in resume['cuda_rng_state']])
        numpy_state = resume['numpy_rng_state']
        np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                             numpy_state[2], numpy_state[3], numpy_state[4]))
        random.setstate(resume['python_rng_state'])
    start_epoch = resume.get('completed_epochs', 0)
    best_rotation_error = resume.get('best_rotation_error', float('inf'))

    def save_progress(completed_epochs, metrics, is_best=False):
        numpy_state = np.random.get_state()
        payload = {
            'model_state_dict': student.state_dict(),
            'teacher_state_dict': teacher.state_dict(),
            'orient_net_state_dict': orient_net.state_dict(),
            'teacher_orient_net_state_dict': teacher_orient_net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'oem_version': CHECKPOINT_VERSION,
            'completed_epochs': completed_epochs,
            'feature_dim': train_shapes[0]['feat'].shape[1],
            'train_size': len(train_shapes),
            'train_names': [shape['name'] for shape in train_shapes],
            'validation_names': [shape['name'] for shape in validation_shapes],
            'config': config,
            'rotation_metrics': metrics,
            'best_rotation_error': best_rotation_error,
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            'numpy_rng_state': (numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]),
            'python_rng_state': random.getstate(),
        }
        temporary = checkpoint_path + '.tmp'
        torch.save(payload, temporary)
        os.replace(temporary, checkpoint_path)
        if is_best:
            best_path = os.path.splitext(checkpoint_path)[0] + '_best_rotation.pth'
            torch.save(payload, best_path + '.tmp')
            os.replace(best_path + '.tmp', best_path)
        print(f"Checkpoint saved after epoch {completed_epochs}: {checkpoint_path}")

    try:
        for epoch in range(start_epoch, num_epochs):
            predicted_probability = predicted_alignment_probability(epoch, oem_warmup, oem_transition)
            # Adversarial adaptation starts with the predicted-alignment transition.
            curr_domain_weight = domain_weight * predicted_probability
            print(f"Alignment curriculum: predicted probability={predicted_probability:.2f}")
            # Smooth cosine annealing for dynamic loss weights (avoids sudden step shocks)
            if epoch < angle_warmup_epochs:
                alpha_angle = 0.5 * (1.0 + np.cos(np.pi * epoch / angle_warmup_epochs))
                curr_angle_weight = angle_weight_target + (angle_weight_init - angle_weight_target) * alpha_angle
            else:
                curr_angle_weight = angle_weight_target
                
            if epoch < lgo_warmup_epochs:
                alpha_lgo = 0.5 * (1.0 - np.cos(np.pi * epoch / lgo_warmup_epochs))
                curr_lgo_weight = lgo_weight_init + (lgo_weight_target - lgo_weight_init) * alpha_lgo
            else:
                curr_lgo_weight = lgo_weight_target
                
            epoch_loss = 0
            epoch_rec_loss = 0
            epoch_cons_loss = 0
            epoch_contra_loss = 0
            epoch_cycle_loss = 0
            epoch_lgo_loss = 0
            epoch_dist_loss = 0
            epoch_orient_loss = 0
            epoch_angle_loss = 0
            epoch_domain_loss = 0
            num_pairs = 0
            
            # All N x N pairs (all pairs i != j)
            pair_indices = [(i, j) for i in range(len(train_shapes)) for j in range(len(train_shapes)) if i != j]
            np.random.shuffle(pair_indices)
            
            for idx1, idx2 in pair_indices:
                s1 = train_shapes[idx1]
                s2 = train_shapes[idx2]
                
                p1_raw = torch.tensor((s1['pos']).copy()).float().unsqueeze(0).to(device)
                f1 = torch.tensor(s1['feat']).float().unsqueeze(0).to(device)
                p2_raw = torch.tensor((s2['pos']).copy()).float().unsqueeze(0).to(device)
                f2 = torch.tensor(s2['feat']).float().unsqueeze(0).to(device)
                
                # -----------------------------------------------------------------
                # Step 1: Data Augmentation & Orientation Estimation (SE-ORNet)
                # -----------------------------------------------------------------
                p1_student = src_aug(p1_raw)
                p2_student, rotated_gt = tgt_aug(p2_raw)
                
                # Pass augmented coordinates through OrientNet for relative orientation & domain logits
                p1_norm_student = normalize_pc_torch(p1_student)
                p2_norm_student = normalize_pc_torch(p2_student)
                
                orient_out = orient_net(p1_norm_student, p2_norm_student)
                
                # Same-shape reference pair supervises rotation and is pair-domain 0.
                p2_norm_clean = normalize_pc_torch(p2_raw)
                orient_out_target = orient_net(p2_norm_clean, p2_norm_student)
                
                loss_angle = compute_angle_loss(
                    orient_out_target["angle_x"],
                    orient_out_target["angle_y"],
                    rotated_gt,
                )
                loss_domain = compute_domain_loss(
                    orient_out_target["global_d_pred"],
                    orient_out["global_d_pred"],
                    fl_global,
                )

                # Start from known augmentation alignment, then transition to the
                # actual cross-shape prediction used at inference. No angle labels
                # are invented for different shapes.
                use_prediction = np.random.random() < predicted_probability
                pred_angle_idx = (orient_out["angle_x"].argmax(dim=-1)
                                  if use_prediction else rotated_gt)
                p2_aligned = orient_net.rotate_point_cloud(p2_student, pred_angle_idx, inverse=True)

                # The EMA teacher has its own OEM and aligns the clean pair.
                with torch.no_grad():
                    teacher_orient_out = teacher_orient_net(
                        normalize_pc_torch(p1_raw), p2_norm_clean,
                    )
                    teacher_angle_idx = teacher_orient_out["angle_x"].argmax(dim=-1)
                    p2_teacher_aligned = (teacher_orient_net.rotate_point_cloud(
                        p2_raw, teacher_angle_idx, inverse=True,
                    ) if use_prediction else p2_raw)
                p1_aligned = p1_student


                
                # -----------------------------------------------------------------
                # Forward Pass 1 (S1 -> S2) through ASMAE
                # -----------------------------------------------------------------
                pred1_s, _, _, _, enc_t1_s, mask1_s = student(
                    f1, p1_aligned, f2, p2_aligned,
                    mask_ratio=student_mask_ratio,
                    feature_ratio=student_feat_ratio
                )
                
                # Teacher predicts lightly masked/unmasked graph using clean coordinates
                with torch.no_grad():
                    pred1_t, _, _, _, enc_t1_t, _ = teacher(
                        f1, p1_raw, f2, p2_teacher_aligned,
                        mask_ratio=teacher_mask_ratio,
                        feature_ratio=teacher_feat_ratio
                    )
                    
                loss1_rec = criterion(pred1_s[mask1_s], f1[mask1_s]) if mask1_s.sum() > 0 else criterion(pred1_s, f1)
                loss1_cons = compute_consistency_loss(pred1_s, pred1_t)
                
                # -----------------------------------------------------------------
                # Forward Pass 2 (S2 -> S1) through ASMAE
                # -----------------------------------------------------------------
                pred2_s, _, _, _, enc_t2_s, mask2_s = student(
                    f2, p2_aligned, f1, p1_aligned,
                    mask_ratio=student_mask_ratio,
                    feature_ratio=student_feat_ratio
                )
                with torch.no_grad():
                    pred2_t, _, _, _, enc_t2_t, _ = teacher(
                        f2, p2_teacher_aligned, f1, p1_raw,
                        mask_ratio=teacher_mask_ratio,
                        feature_ratio=teacher_feat_ratio
                    )
                    
                loss2_rec = criterion(pred2_s[mask2_s], f2[mask2_s]) if mask2_s.sum() > 0 else criterion(pred2_s, f2)
                loss2_cons = compute_consistency_loss(pred2_s, pred2_t)
                
                # -----------------------------------------------------------------
                # Combine and Backprop
                # -----------------------------------------------------------------
                loss_rec = loss1_rec + loss2_rec
                loss_cons = loss1_cons + loss2_cons
                
                # Extract clean features for Shape 1 (N1) and Shape 2 (N2)
                enc1_s = student.extract_features(f1, p1_aligned)
                enc2_s = student.extract_features(f2, p2_aligned)
                
                # Contrastive Loss
                loss_contra1 = compute_contrastive_loss(enc1_s, margin=contra_margin)
                loss_contra2 = compute_contrastive_loss(enc2_s, margin=contra_margin)
                loss_contra = loss_contra1 + loss_contra2
                
                # Cycle Loss (requires matching coordinate tensors p1 [N1] and p2 [N2])
                loss_cycle1 = compute_cycle_loss(enc1_s, enc2_s, p1_aligned, eps=cycle_eps, n_iter=cycle_n_iter)
                loss_cycle2 = compute_cycle_loss(enc2_s, enc1_s, p2_aligned, eps=cycle_eps, n_iter=cycle_n_iter)
                loss_cycle = loss_cycle1 + loss_cycle2
                
                # Global Optimization Loss (L_go) - asymmetric Sinkhorn pseudo-labels
                loss_lgo1 = compute_lgo_loss(enc1_s, enc2_s, eps=student_lgo_eps, target_eps=target_lgo_eps, n_iter=cycle_n_iter)
                loss_lgo2 = compute_lgo_loss(enc2_s, enc1_s, eps=student_lgo_eps, target_eps=target_lgo_eps, n_iter=cycle_n_iter)
                loss_lgo = (loss_lgo1 + loss_lgo2) / 2.0
                
                # Metric Distortion Loss (L_dist)
                loss_dist1 = compute_distortion_loss(enc1_s, enc2_s, p1_aligned, p2_aligned, num_samples=dist_samples, tau=dist_tau)
                loss_dist2 = compute_distortion_loss(enc2_s, enc1_s, p2_aligned, p1_aligned, num_samples=dist_samples, tau=dist_tau)
                loss_dist = (loss_dist1 + loss_dist2) / 2.0
                
                # Signed Area / Orientation Consistency Loss (Method B from Complex Functional Maps)
                # Directly penalizes local normal flips / reflection symmetry errors
                loss_orient1 = compute_orientation_loss(enc1_s, enc2_s, p1_aligned, p2_aligned, num_samples=500, tau=dist_tau)
                loss_orient2 = compute_orientation_loss(enc2_s, enc1_s, p2_aligned, p1_aligned, num_samples=500, tau=dist_tau)
                loss_orient = (loss_orient1 + loss_orient2) / 2.0
                
                # Total loss with dynamic curriculum weights
                loss = (loss_rec + 
                        (cons_weight * loss_cons) + 
                        (contra_weight * loss_contra) + 
                        (cycle_weight * loss_cycle) + 
                        (curr_lgo_weight * loss_lgo) + 
                        (dist_weight * loss_dist) +
                        (orient_weight * loss_orient) +
                        (curr_angle_weight * loss_angle) +
                        (curr_domain_weight * loss_domain))
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update EMA of the teacher model natively after student steps forwards
                update_teacher_ema(student, teacher, alpha=ema_alpha)
                update_teacher_ema(orient_net, teacher_orient_net, alpha=ema_alpha)
                
                epoch_loss += loss.item()
                epoch_rec_loss += loss_rec.item()
                epoch_cons_loss += loss_cons.item()
                epoch_contra_loss += loss_contra.item()
                epoch_cycle_loss += loss_cycle.item()
                epoch_lgo_loss += loss_lgo.item()
                epoch_dist_loss += loss_dist.item()
                epoch_orient_loss += loss_orient.item()
                epoch_angle_loss += loss_angle.item()
                epoch_domain_loss += loss_domain.item()
                num_pairs += 1
                
            avg_loss = epoch_loss / num_pairs if num_pairs > 0 else 0
            avg_rec = epoch_rec_loss / num_pairs if num_pairs > 0 else 0
            avg_cons = epoch_cons_loss / num_pairs if num_pairs > 0 else 0
            avg_contra = epoch_contra_loss / num_pairs if num_pairs > 0 else 0
            avg_cycle = epoch_cycle_loss / num_pairs if num_pairs > 0 else 0
            avg_lgo = epoch_lgo_loss / num_pairs if num_pairs > 0 else 0
            avg_dist = epoch_dist_loss / num_pairs if num_pairs > 0 else 0
            avg_orient = epoch_orient_loss / num_pairs if num_pairs > 0 else 0
            avg_angle = epoch_angle_loss / num_pairs if num_pairs > 0 else 0
            avg_domain = epoch_domain_loss / num_pairs if num_pairs > 0 else 0
            
            print(f"Epoch {epoch+1:3d}/{num_epochs} | Tot: {avg_loss:.4f} | Rec: {avg_rec:.4f} | Cons: {avg_cons:.4f} | Contra: {avg_contra:.4f} | Cycle: {avg_cycle:.4f} | Lgo: {avg_lgo:.4f} | Dist: {avg_dist:.4f} | Orient: {avg_orient:.4f} | Angle: {avg_angle:.4f} (w={curr_angle_weight:.1f}) | Dom: {avg_domain:.4f}")

            metrics = {}
            is_best = False
            if (epoch + 1) % validation_every == 0 or epoch + 1 == num_epochs:
                for label, module in [('student', orient_net), ('teacher', teacher_orient_net)]:
                    for split, shapes in [('train', train_shapes[:2]), ('validation', validation_shapes)]:
                        if not shapes:
                            continue
                        result = evaluate_rotation(module, shapes, device)
                        metrics[f'{label}_{split}'] = result
                        print(f"OEM {label} {split}: accuracy={result['accuracy']:.1%} | "
                              f"reverse={result['reverse_accuracy']:.1%} | "
                              f"angle error={result['angle_error_degrees']:.2f} deg | "
                              f"CE={result['loss']:.4f} | train/eval gap={result['mode_gap']:.2e}")
                validation = metrics.get('student_validation')
                if validation and validation['angle_error_degrees'] < best_rotation_error:
                    best_rotation_error = validation['angle_error_degrees']
                    is_best = True
            save_progress(epoch + 1, metrics, is_best)

    except KeyboardInterrupt:
        print("Training interrupted. Resume from the last completed epoch checkpoint; "
              "the incomplete epoch has not overwritten it.")

    print("Training stopped." if start_epoch >= num_epochs else "Training run finished.")
    return student, orient_net


def main():
    parser = argparse.ArgumentParser(description="ASMAE Student-Teacher Training with Orientation Module")
    parser.add_argument('--config', type=str, default='config/FAUST/train_st_te_config.yaml', help='Path to config file')
    parser.add_argument('--fresh', action='store_true', help='Force training from scratch ignoring existing checkpoint')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    seed = config['training'].get('seed', 42)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    
    config_train_size = config.get('train_size', 60)
    config_test_size = config.get('test_size', 40)
    data_dir = config['data_dir']
    
    obj_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.obj')])
    total_files = len(obj_files)
    
    if total_files < config_train_size + config_test_size:
        train_size = int(total_files * config_train_size / (config_train_size + config_test_size))
        test_size = total_files - train_size
        print(f"\nDataset too small ({total_files} files). Adjusted to ratio: {train_size} train, {test_size} test")
    else:
        train_size = config_train_size
        test_size = config_test_size
        print(f"\nUsing absolute split: {train_size} train, {test_size} test")

    output_dir = config['output_dir']
    k = config['geometry']['k']
    t = config['geometry']['t']
    neigvecs = config['geometry'].get('neigvecs', 300)
    
    validation_size = config['training'].get('validation_size', 2)
    if validation_size < 1 or train_size - validation_size < 2:
        raise ValueError("Reserve at least one validation shape and two training shapes.")
    training_files = obj_files[:train_size - validation_size]
    validation_files = obj_files[train_size - validation_size:train_size]
    print(f"Training-pool split: {len(training_files)} train, {len(validation_files)} validation; "
          f"{test_size} test shapes remain outside training and validation")
    train_shapes = load_train_shapes(data_dir, k, t, neigvecs, max_shapes=len(training_files),
                                     output_dir=output_dir, filenames=training_files)
    validation_shapes = load_train_shapes(data_dir, k, t, neigvecs, max_shapes=len(validation_files),
                                          output_dir=output_dir, filenames=validation_files)
    if not validation_shapes:
        raise ValueError("No validation shapes were loaded.")
    if len(train_shapes) < 2:
        print("Not enough shapes for training.")
        return
        
    feature_dim = train_shapes[0]['feat'].shape[1]
    
    # Initialize Student Model
    student_cfg = config.get('student_model', config.get('model', {}))
    student = ASMAE(
        feature_dim=feature_dim,
        embed_dim=student_cfg['embed_dim'],
        depth=student_cfg['depth'],
        num_heads=student_cfg['num_heads'],
        decoder_embed_dim=student_cfg['decoder_embed_dim'],
        decoder_depth=student_cfg['decoder_depth'],
        decoder_num_heads=student_cfg['decoder_num_heads'],
        mlk_ratio=student_cfg['mlk_ratio'],
        num_mask_queries=student_cfg.get('num_mask_queries', 5000),
        encoder_k=student_cfg.get('encoder_k', 20),
        aamg_k=student_cfg.get('aamg_k', 10),
        aamg_emb_dim=student_cfg.get('aamg_emb_dim', 64),
        pos_embed_dim=student_cfg.get('pos_embed_dim', 64),
        temperature=student_cfg.get('temperature', 1.0)
    )
    
    # Initialize Teacher Model
    teacher_cfg = config.get('teacher_model', config.get('model', {}))
    teacher = ASMAE(
        feature_dim=feature_dim,
        embed_dim=teacher_cfg['embed_dim'],
        depth=teacher_cfg['depth'],
        num_heads=teacher_cfg['num_heads'],
        decoder_embed_dim=teacher_cfg['decoder_embed_dim'],
        decoder_depth=teacher_cfg['decoder_depth'],
        decoder_num_heads=teacher_cfg['decoder_num_heads'],
        mlk_ratio=teacher_cfg['mlk_ratio'],
        num_mask_queries=teacher_cfg.get('num_mask_queries', 5000),
        encoder_k=teacher_cfg.get('encoder_k', 20),
        aamg_k=teacher_cfg.get('aamg_k', 10),
        aamg_emb_dim=teacher_cfg.get('aamg_emb_dim', 64),
        pos_embed_dim=teacher_cfg.get('pos_embed_dim', 64),
        temperature=teacher_cfg.get('temperature', 1.0)
    )
    
    # Initialize OrientNet
    orient_net = OrientNet()
    teacher_orient_net = copy.deepcopy(orient_net)
    
    # Initialize teacher exactly with student's weights initially
    teacher.load_state_dict(student.state_dict())
    
    # Fresh runs never require pretrained weights. Only version-matched complete
    # checkpoints may resume; silently migrating BatchNorm weights would hide the fix.
    checkpoint_path = os.path.join(config['training']['checkpoint_dir'], config['training']['checkpoint_name'])
    resume = None
    if not args.fresh and os.path.exists(checkpoint_path):
        resume = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
        if resume.get('oem_version') != CHECKPOINT_VERSION:
            raise ValueError("Legacy checkpoint uses a different OEM pipeline. "
                             "Use --fresh to train the GroupNorm pipeline from scratch.")
        if resume['train_names'] != [shape['name'] for shape in train_shapes] or \
                resume['validation_names'] != [shape['name'] for shape in validation_shapes]:
            raise ValueError("Checkpoint train/validation split differs from this run.")
        old_training = resume['config']['training']
        mutable = {'epochs', 'device', 'checkpoint_dir', 'checkpoint_name'}
        for key in (old_training.keys() | config['training'].keys()) - mutable:
            if old_training.get(key) != config['training'].get(key):
                raise ValueError(f"Resume must preserve training.{key}; use --fresh for a new experiment.")
        for key in ('geometry', 'student_model', 'teacher_model', 'model'):
            if resume['config'].get(key) != config.get(key):
                raise ValueError(f"Resume must preserve {key}.")
        student.load_state_dict(resume['model_state_dict'])
        teacher.load_state_dict(resume['teacher_state_dict'])
        orient_net.load_state_dict(resume['orient_net_state_dict'])
        teacher_orient_net.load_state_dict(resume['teacher_orient_net_state_dict'])
        print(f"Resuming after completed epoch {resume['completed_epochs']}.")
    else:
        print("Fresh initialization: student, teacher, and OEMs; no pretrained weights loaded.")

    # Lock teacher parameters against standard backprop
    for param in list(teacher.parameters()) + list(teacher_orient_net.parameters()):
        param.requires_grad = False
    
    print(f"  Feature dimension: {feature_dim}")
    num_params_student = sum(p.numel() for p in student.parameters())
    num_params_orient = sum(p.numel() for p in orient_net.parameters())
    print(f"  Student parameters: {num_params_student:,}")
    print(f"  OrientNet parameters: {num_params_orient:,}")

    train_model(student, teacher, orient_net, teacher_orient_net, train_shapes,
                config, validation_shapes=validation_shapes, resume=resume)



if __name__ == "__main__":
    main()
