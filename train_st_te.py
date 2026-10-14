import os
import torch
import numpy as np
import sys
import yaml
import argparse
import copy

from core.preprocessing import process_geometry
from models.asmae import ASMAE
from models.oriented_pair import prepare_aligned_pair, load_orientation_checkpoint, ALIGNMENT_PIPELINE_VERSION
from models.modules.orientation_module import OrientNet, DataAugment, normalize_pc_torch
from core.consistency_loss import compute_consistency_loss
from core.contrastive_loss import compute_contrastive_loss
from core.cycle_loss import compute_cycle_loss
from core.lgo_loss import compute_lgo_loss
from core.distortion_loss import compute_distortion_loss
from core.orientation_loss import compute_orientation_loss
from core.domain_loss import compute_pair_domain_loss, compute_angle_loss, FocalLoss
from core.oem_training import pretrain_oem, evaluate_oem, canonical_angle_labels
from utils.training_validation import validate_correspondence


def update_teacher_ema(student, teacher, alpha=0.999):
    """
    Updates the teacher parameters using Exponential Moving Average (EMA) of student parameters.
    """
    with torch.no_grad():
        for s_param, t_param in zip(student.parameters(), teacher.parameters()):
            t_param.mul_(alpha).add_(s_param, alpha=1.0 - alpha)
        # BatchNorm statistics are part of the teacher, too.
        for s_buffer, t_buffer in zip(student.buffers(), teacher.buffers()):
            if t_buffer.is_floating_point():
                t_buffer.mul_(alpha).add_(s_buffer, alpha=1.0 - alpha)
            else:
                t_buffer.copy_(s_buffer)


def load_train_shapes(data_dir, k, t, neigvecs, max_shapes, output_dir):
    """Load all .obj files from directory"""
    print(f"\n{'='*60}")
    print(f"Loading {max_shapes} train shapes from: {data_dir}")
    print(f"{'='*60}")
    
    obj_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.obj')])
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


def train_model(student, teacher, orient_net, train_shapes, config, teacher_orient_net=None,
                validation_shapes=None, checkpoint_metadata=None, resume_state=None):
    teacher.requires_grad_(False)
    if teacher_orient_net is None:
        teacher_orient_net = copy.deepcopy(orient_net)
    teacher_orient_net.requires_grad_(False)
    device = config['training'].get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    # Move parameters before constructing/loading Adam state, including resume.
    student.to(device)
    teacher.to(device)
    orient_net.to(device)
    teacher_orient_net.to(device)
    
    num_epochs = config['training'].get('epochs', 10)
    lr = config['training']['lr']
    oem_config = config.get('oem', {})
    freeze_epochs = int(oem_config.get('freeze_epochs', 10))
    canonical_labels = canonical_angle_labels(oem_config)
    validation_shapes = validation_shapes or train_shapes[:2]
    validation_settings = config.get('validation', {})
    validation_every = max(int(validation_settings.get('every_epochs', 5)), 1)
    checkpoint_metadata = checkpoint_metadata or {}
    checkpoint_path = os.path.join(config['training']['checkpoint_dir'], config['training']['checkpoint_name'])
    last_path = os.path.splitext(checkpoint_path)[0] + '_last.pth'
    resume_state = resume_state or {}
    best_score = float(resume_state.get('best_validation_error', float('inf')))
    best_metric = resume_state.get('best_validation_metric')
    start_epoch = int(resume_state.get('epoch', 0))
    
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
    lgo_start_epoch = int(config['training'].get('lgo_start_epoch', 5))
    
    orient_weight = config['training'].get('orientation_weight', 100.0)
    dist_weight = config['training'].get('distortion_weight', 120.0)
    dist_tau = config['training'].get('distortion_tau', 0.04)
    dist_samples = config['training'].get('distortion_samples', 384)
    
    # SE-ORNet Orientation & Domain Loss weights with smooth warmup annealing
    angle_weight_target = config['training'].get('angle_weight', 2.0)
    angle_weight_init = config['training'].get('angle_weight_init', 8.0)
    angle_warmup_epochs = config['training'].get('angle_warmup_epochs', 5)
    domain_weight = config['training'].get('domain_weight', 1.0)
    angle_label_smoothing = config['training'].get('angle_label_smoothing', 0.0)
    
    print(f"\n{'='*60}")
    print("STUDENT-TEACHER TRAINING PHASE (WITH DYNAMIC ORIENTATION WARMUP)")
    print(f"  Training shapes: {len(train_shapes)}")
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
    
    # OEM uses a separate, smaller learning rate after its frozen phase.
    optimizer = torch.optim.Adam(
        [{'params': student.parameters(), 'lr': lr},
         {'params': orient_net.parameters(), 'lr': float(oem_config.get('finetune_lr', lr * 0.1))}]
    )
    if 'optimizer_state_dict' in resume_state:
        optimizer.load_state_dict(resume_state['optimizer_state_dict'])
    criterion = torch.nn.L1Loss()
    fl_global = FocalLoss(class_num=2, gamma=3)
    
    # Rotation remains enabled; Gaussian noise is controlled by YAML.
    noise_variance = float(config['training'].get('noise_variance', 0.0))
    if noise_variance < 0:
        raise ValueError("training.noise_variance must be nonnegative")
    noise_operations = ["noise"] if noise_variance > 0 else []
    src_aug = DataAugment(operations=["rotate"] + noise_operations,
                          rotate_nbins=8, noise_variance=noise_variance)
    tgt_aug = DataAugment(operations=noise_operations, noise_variance=noise_variance)
    print(f"  Gaussian noise variance: {noise_variance}")
    
    student.train()
    # OEM uses GroupNorm throughout, with identical train/eval behavior.
    # eval() still permits gradients when OEM is unfrozen.
    orient_net.eval()
    teacher.eval() # Teacher does not get trained by backprop natively
    teacher_orient_net.eval()
    
    completed_epoch = start_epoch
    def save_state(path, completed, validation=None):
        state = dict(checkpoint_metadata)
        state.update({'model_state_dict': student.state_dict(),
                      'teacher_model_state_dict': teacher.state_dict(),
                      'orient_net_state_dict': orient_net.state_dict(),
                      'teacher_orient_net_state_dict': teacher_orient_net.state_dict(),
                      'alignment_pipeline_version': ALIGNMENT_PIPELINE_VERSION,
                      'oem_config': oem_config, 'optimizer_state_dict': optimizer.state_dict(),
                      'epoch': completed, 'best_validation_error': best_score,
                      'best_validation_metric': best_metric, 'validation': validation})
        torch.save(state, path)

    epoch = start_epoch
    try:
        for epoch in range(start_epoch, num_epochs):
            oem_frozen = epoch < freeze_epochs
            orient_net.requires_grad_(not oem_frozen)
            print(f"Epoch {epoch + 1}: OEM {'frozen' if oem_frozen else 'fine-tuning'}; GroupNorm throughout")
            # Smooth cosine annealing for dynamic loss weights (avoids sudden step shocks)
            if epoch < angle_warmup_epochs:
                alpha_angle = 0.5 * (1.0 + np.cos(np.pi * epoch / angle_warmup_epochs))
                curr_angle_weight = angle_weight_target + (angle_weight_init - angle_weight_target) * alpha_angle
            else:
                curr_angle_weight = angle_weight_target
                
            if epoch < lgo_start_epoch:
                curr_lgo_weight = 0.0
            elif epoch - lgo_start_epoch < lgo_warmup_epochs:
                alpha_lgo = 0.5 * (1.0 - np.cos(np.pi * (epoch - lgo_start_epoch) / lgo_warmup_epochs))
                curr_lgo_weight = lgo_weight_init + (lgo_weight_target - lgo_weight_init) * alpha_lgo
            else:
                curr_lgo_weight = lgo_weight_target
            if oem_frozen:
                curr_angle_weight = 0.0
            curr_domain_weight = 0.0 if oem_frozen else domain_weight
                
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
            angle_correct = 0
            angle_count = 0
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
                # Match SE-ORNet's unit-sphere inputs; augmentation noise now
                # has the same magnitude regardless of dataset units.
                p1_raw = normalize_pc_torch(p1_raw)
                p2_raw = normalize_pc_torch(p2_raw)
                
                # -----------------------------------------------------------------
                # Step 1: Data Augmentation & Orientation Estimation (SE-ORNet)
                # -----------------------------------------------------------------
                p1_student, rotated_gt = src_aug(p1_raw)
                p2_student = tgt_aug(p2_raw)

                # OEM(Y_s, X_s) predicts how to align rotated source to target.
                f1_s, p1_aligned, f2_s, p2_aligned, orient_out = prepare_aligned_pair(
                    orient_net, f1, p1_student, f2, p2_student
                )
                # EMA teacher OEM(Y, X) aligns clean source to clean target.
                with torch.no_grad():
                    f1_t, p1_teacher, f2_t, p2_teacher, _ = prepare_aligned_pair(
                        teacher_orient_net, f1, p1_raw, f2, p2_raw
                    )
                
                # Normalize the same-shape rotation pair for OrientNet.
                p1_norm_student = normalize_pc_torch(p1_student)
                # Same-shape reference pair supplies known rotation labels
                # and the pair-domain label 0.
                p1_norm_clean = normalize_pc_torch(p1_raw)
                orient_out_same_source = orient_net(p1_norm_clean, p1_norm_student)
                
                loss_angle_source = compute_angle_loss(
                    orient_out_same_source["angle_x"],
                    orient_out_same_source["angle_y"],
                    rotated_gt,
                    label_smoothing=angle_label_smoothing,
                )
                loss_angle = loss_angle_source
                # Cross-shape angle labels are used ONLY for explicitly
                # annotated native relative orientations.
                native_bin = canonical_labels.get((s1['name'], s2['name']))
                if native_bin is not None:
                    relative_gt = (native_bin + rotated_gt - 1) % 8
                    loss_angle = (loss_angle + compute_angle_loss(
                        orient_out['angle_x'], orient_out['angle_y'], relative_gt,
                        label_smoothing=angle_label_smoothing)) / 2.0
                with torch.no_grad():
                    bins = orient_out_same_source["angle_x"].shape[-1]
                    inverse_gt = (bins // 4 - rotated_gt) % bins
                    angle_correct += (orient_out_same_source["angle_x"].argmax(-1) == rotated_gt).sum().item()
                    angle_correct += (orient_out_same_source["angle_y"].argmax(-1) == inverse_gt).sum().item()
                    angle_count += 2 * rotated_gt.numel()
                
                # SE-ORNet domain adaptation transfers from known-angle
                # same-shape pairs (domain 0) to cross-shape pairs (domain 1).
                loss_domain = compute_pair_domain_loss(
                    orient_out_same_source["global_d_pred"],
                    orient_out["global_d_pred"],
                    fl_global,
                )


                
                # -----------------------------------------------------------------
                # Forward Pass 1 (S1 -> S2) through ASMAE
                # -----------------------------------------------------------------
                pred1_s, _, _, _, enc_t1_s, mask1_s = student(
                    f1_s, p1_aligned, f2_s, p2_aligned,
                    mask_ratio=student_mask_ratio,
                    feature_ratio=student_feat_ratio
                )
                
                # Teacher predicts using its OEM-aligned clean coordinates.
                with torch.no_grad():
                    pred1_t, _, _, _, enc_t1_t, _ = teacher(
                        f1_t, p1_teacher, f2_t, p2_teacher,
                        mask_ratio=teacher_mask_ratio,
                        feature_ratio=teacher_feat_ratio
                    )
                    
                loss1_rec = criterion(pred1_s[mask1_s], f1_s[mask1_s]) if mask1_s.sum() > 0 else criterion(pred1_s, f1_s)
                # XYZ is frame-dependent; distill the intrinsic descriptors.
                loss1_cons = compute_consistency_loss(pred1_s[..., :-3], pred1_t[..., :-3])
                
                # -----------------------------------------------------------------
                # Forward Pass 2 (S2 -> S1) through ASMAE
                # -----------------------------------------------------------------
                pred2_s, _, _, _, enc_t2_s, mask2_s = student(
                    f2_s, p2_aligned, f1_s, p1_aligned,
                    mask_ratio=student_mask_ratio,
                    feature_ratio=student_feat_ratio
                )
                with torch.no_grad():
                    pred2_t, _, _, _, enc_t2_t, _ = teacher(
                        f2_t, p2_teacher, f1_t, p1_teacher,
                        mask_ratio=teacher_mask_ratio,
                        feature_ratio=teacher_feat_ratio
                    )
                    
                loss2_rec = criterion(pred2_s[mask2_s], f2_s[mask2_s]) if mask2_s.sum() > 0 else criterion(pred2_s, f2_s)
                loss2_cons = compute_consistency_loss(pred2_s[..., :-3], pred2_t[..., :-3])
                
                # -----------------------------------------------------------------
                # Combine and Backprop
                # -----------------------------------------------------------------
                loss_rec = loss1_rec + loss2_rec
                loss_cons = loss1_cons + loss2_cons
                
                # Extract clean features for Shape 1 (N1) and Shape 2 (N2)
                enc1_s = student.extract_features(f1_s, p1_aligned)
                enc2_s = student.extract_features(f2_s, p2_aligned)
                
                # Contrastive Loss
                loss_contra1 = compute_contrastive_loss(enc1_s, margin=contra_margin)
                loss_contra2 = compute_contrastive_loss(enc2_s, margin=contra_margin)
                loss_contra = loss_contra1 + loss_contra2
                
                # Cycle Loss (requires matching coordinate tensors p1 [N1] and p2 [N2])
                loss_cycle1 = compute_cycle_loss(enc1_s, enc2_s, p1_aligned, eps=cycle_eps, n_iter=cycle_n_iter)
                loss_cycle2 = compute_cycle_loss(enc2_s, enc1_s, p2_aligned, eps=cycle_eps, n_iter=cycle_n_iter)
                loss_cycle = loss_cycle1 + loss_cycle2
                
                # Global Optimization Loss (L_go) - asymmetric Sinkhorn pseudo-labels
                with torch.set_grad_enabled(curr_lgo_weight > 0):
                    loss_lgo1 = compute_lgo_loss(enc1_s, enc2_s, eps=student_lgo_eps, target_eps=target_lgo_eps, n_iter=cycle_n_iter)
                    loss_lgo2 = compute_lgo_loss(enc2_s, enc1_s, eps=student_lgo_eps, target_eps=target_lgo_eps, n_iter=cycle_n_iter)
                loss_lgo = (loss_lgo1 + loss_lgo2) / 2.0
                
                # Metric Distortion Loss (L_dist)
                loss_dist1 = compute_distortion_loss(enc1_s, enc2_s, p1_aligned, p2_aligned, num_samples=dist_samples, tau=dist_tau)
                loss_dist2 = compute_distortion_loss(enc2_s, enc1_s, p2_aligned, p1_aligned, num_samples=dist_samples, tau=dist_tau)
                loss_dist = (loss_dist1 + loss_dist2) / 2.0
                
                # Signed Area / Orientation Consistency Loss (Method B from Complex Functional Maps)
                # Directly penalizes local normal flips / reflection symmetry errors
                # The legacy world-normal term is diagnostic when weight is 0;
                # do not conceal it by reporting an artificial zero loss.
                with torch.set_grad_enabled(orient_weight > 0):
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
                torch.nn.utils.clip_grad_norm_(list(student.parameters()) + list(orient_net.parameters()),
                                              config['training'].get('grad_clip', 5.0))
                optimizer.step()
                
                # Update EMA of the teacher model natively after student steps forwards
                update_teacher_ema(student, teacher, alpha=ema_alpha)
                if not oem_frozen:
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
            print(f"  Same-shape angle accuracy: {angle_correct / max(angle_count, 1):.2%} (smoothing={angle_label_smoothing})")
            print(f"  Active weights: LGO={curr_lgo_weight:.2f}, Domain={curr_domain_weight:.2f}, "
                  f"Dist={dist_weight:.2f}, Orient={orient_weight:.2f}")
            completed_epoch = epoch + 1
            if completed_epoch % validation_every == 0 or completed_epoch == num_epochs or completed_epoch == 1:
                oem_metrics = evaluate_oem(orient_net, validation_shapes, canonical_labels)
                validation = validate_correspondence(student, orient_net, validation_shapes,
                                                       validation_settings, config['data_dir'])
                print(f"  Held-out OEM eval: forward={oem_metrics['forward_accuracy']:.2%}, "
                      f"inverse={oem_metrics['inverse_accuracy']:.2%}, forward yaw={oem_metrics['yaw_error_degrees']:.2f} deg")
                if oem_metrics['cross_accuracy'] is not None:
                    print(f"  Annotated cross-shape rotation accuracy: {oem_metrics['cross_accuracy']:.2%}")
                print(f"  {validation['metric']}: {validation['error']:.6f} | "
                      f"PCK@0.05={validation['pck_005']:.2%} | pairs={validation['pairs']}")
                if best_metric != validation['metric']:
                    best_score = float('inf')
                    best_metric = validation['metric']
                if validation['error'] < best_score:
                    best_score = validation['error']
                    save_state(checkpoint_path, completed_epoch, validation)
                    print(f"  Saved best validation checkpoint: {checkpoint_path}")
            save_state(last_path, completed_epoch)

        
    except KeyboardInterrupt:
        # A partially completed epoch is replayed on resume; never claim that
        # interrupted weights are a validated best checkpoint.
        save_state(last_path, completed_epoch)
        print(f"\nInterrupted. Latest weights saved to {last_path}; best checkpoint preserved.")
            
    print("\nTraining complete!")
    return student, orient_net, teacher_orient_net


def main():
    parser = argparse.ArgumentParser(description="ASMAE Student-Teacher Training with Orientation Module")
    parser.add_argument('--config', type=str, default='config/SHREC/train_st_te_config.yaml', help='Path to config file')
    parser.add_argument('--fresh', action='store_true', help='Force training from scratch ignoring existing checkpoint')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    seed = int(config['training'].get('seed', 42))
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
    
    reserved_shapes = load_train_shapes(data_dir, k, t, neigvecs, max_shapes=train_size, output_dir=output_dir)
    validation_count = int(config.get('validation', {}).get('shape_count', 4))
    if validation_count < 1 or len(reserved_shapes) < validation_count + 2:
        raise ValueError('Reserve at least one validation shape and two training shapes inside train_size')
    train_shapes = reserved_shapes[:-validation_count]
    validation_shapes = reserved_shapes[-validation_count:]
    print(f"Training shapes: {len(train_shapes)} | Validation shapes: {len(validation_shapes)} | "
          f"Untouched test partition begins at index {train_size}")
        
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
    oem_config = config.get('oem', {})
    orient_net = OrientNet(max_points=int(oem_config.get('max_points', 512)))
    teacher_orient_net = copy.deepcopy(orient_net)
    
    # Initialize teacher exactly with student's weights initially
    teacher.load_state_dict(student.state_dict())
    
    # Check if a checkpoint exists and load weights unless --fresh flag is passed
    checkpoint_path = os.path.join(config['training']['checkpoint_dir'], config['training']['checkpoint_name'])
    ckpt = None
    latest_path = os.path.splitext(checkpoint_path)[0] + '_last.pth'
    resume_path = latest_path if os.path.exists(latest_path) else checkpoint_path
    if not args.fresh and os.path.exists(resume_path):
        print(f"\n[RESUME TRAINING] Found existing checkpoint at: {resume_path}")
        ckpt = torch.load(resume_path, map_location='cpu', weights_only=True)
        if ckpt.get('alignment_pipeline_version') != ALIGNMENT_PIPELINE_VERSION:
            raise ValueError("Existing checkpoint uses the previous OEM pipeline. "
                             "Choose a new checkpoint_name and use --fresh for this design.")
        student.load_state_dict(ckpt['model_state_dict'])
        teacher.load_state_dict(ckpt.get('teacher_model_state_dict', ckpt['model_state_dict']))
        load_orientation_checkpoint(orient_net, ckpt)
        teacher_orient_net.load_state_dict(ckpt.get('teacher_orient_net_state_dict', ckpt['orient_net_state_dict']))
        teacher_orient_net.max_points = orient_net.max_points
        if ckpt.get('validation_filenames') != [s['name'] + '.obj' for s in validation_shapes]:
            raise ValueError('Validation split changed; use --fresh for a new experiment')
        if ckpt.get('geometry_config') != config['geometry']:
            raise ValueError('Descriptor geometry changed; use --fresh')
        if ckpt.get('student_model_config') != student_cfg:
            raise ValueError('Model settings changed; use --fresh')
        if ckpt.get('oem_config', {}).get('max_points', 512) != int(oem_config.get('max_points', 512)):
            raise ValueError('OEM point sampling changed; use --fresh')
        print("Successfully restored student and EMA teacher ASMAE/OEM weights.")
    else:
        if args.fresh:
            print(f"\n[FRESH START] --fresh flag set. Ignoring existing checkpoint and training from scratch.")
        else:
            print(f"\n[NEW TRAINING] No existing checkpoint found. Starting from scratch.")
    
    # Lock teacher parameters against standard backprop
    for param in teacher.parameters():
        param.requires_grad = False

    device = config['training'].get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    orient_net.to(device)
    if ckpt is None:
        warmup_path = os.path.splitext(checkpoint_path)[0] + '_oem_pretrain.pth'
        metrics = pretrain_oem(orient_net, train_shapes, validation_shapes,
                              oem_config, warmup_path, resume=not args.fresh)
        minimum = float(oem_config.get('minimum_eval_accuracy', 0.95))
        if min(metrics['forward_accuracy'], metrics['inverse_accuracy']) < minimum:
            raise RuntimeError(f"OEM warm-up has not reached {minimum:.1%} in both held-out directions. "
                               f"Best OEM saved at {warmup_path}; correspondence training was not started. "
                               "Compare the training-shape eval probe against held-out accuracy before changing epochs.")
        teacher_orient_net.load_state_dict(orient_net.state_dict())
    
    print(f"  Feature dimension: {feature_dim}")
    num_params_student = sum(p.numel() for p in student.parameters())
    num_params_orient = sum(p.numel() for p in orient_net.parameters())
    print(f"  Student parameters: {num_params_student:,}")
    print(f"  OrientNet parameters: {num_params_orient:,}")

    checkpoint_metadata = {
        'feature_dim': feature_dim,
        'train_size': len(train_shapes),
        'train_split_index': train_size,
        'train_data_dir': data_dir,
        'train_filenames': obj_files[:train_size],
        'validation_filenames': [s['name'] + '.obj' for s in validation_shapes],
        'student_model_config': student_cfg,
        'geometry_config': config['geometry'],
    }
    train_model(student, teacher, orient_net, train_shapes, config, teacher_orient_net,
                validation_shapes, checkpoint_metadata, ckpt)
    print(f"\nBest validated weights: {checkpoint_path}")
    print(f"Latest/resumable weights: {os.path.splitext(checkpoint_path)[0]}_last.pth")


if __name__ == "__main__":
    main()
