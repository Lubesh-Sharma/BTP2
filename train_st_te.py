import os
import torch
import numpy as np
import sys
import yaml
import argparse

from core.preprocessing import process_geometry
from models.asmae import ASMAE
from core.consistency_loss import compute_consistency_loss
from core.contrastive_loss import compute_contrastive_loss
from core.cycle_loss import compute_cycle_loss
from core.lgo_loss import compute_lgo_loss

def update_teacher_ema(student, teacher, alpha=0.999):
    """
    Updates the teacher parameters using Exponential Moving Average (EMA) of student parameters.
    """
    with torch.no_grad():
        for s_param, t_param in zip(student.parameters(), teacher.parameters()):
            t_param.data = alpha * t_param.data + (1.0 - alpha) * s_param.data

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

def train_model(student, teacher, train_shapes, config):
    device = config['training'].get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    num_epochs = config['training']['epochs']
    lr = config['training']['lr']
    
    # Extract student params
    student_mask_ratio = config['training'].get('student_mask_ratio', 0.4)
    student_feat_ratio = config['training'].get('student_feature_ratio', 0.2)
    
    # Extract teacher params
    teacher_mask_ratio = config['training'].get('teacher_mask_ratio', 0.1)
    teacher_feat_ratio = config['training'].get('teacher_feature_ratio', 0.05)
    
    # Loss configs
    ema_alpha = config['training'].get('ema_alpha', 0.999)
    cons_weight = config['training'].get('consistency_weight', 1.0)
    contra_margin = config['training'].get('contrastive_margin', 0.5)
    contra_weight = config['training'].get('contrastive_weight', 1.0)
    cycle_eps = config['training'].get('cycle_eps', 0.05)
    cycle_n_iter = config['training'].get('cycle_n_iter', 15)
    cycle_weight = config['training'].get('cycle_weight', 1.0)
    lgo_eps = config['training'].get('lgo_eps', 0.02)
    lgo_weight = config['training'].get('lgo_weight', 500.0)
    
    print(f"\n{'='*60}")
    print("STUDENT-TEACHER TRAINING PHASE")
    print(f"  Training shapes: {len(train_shapes)}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Device: {device}")
    print(f"  Student Masking Ratio (Nodes/Feats): {student_mask_ratio} / {student_feat_ratio}")
    print(f"  Teacher Masking Ratio (Nodes/Feats): {teacher_mask_ratio} / {teacher_feat_ratio}")
    print(f"  EMA Alpha: {ema_alpha} | Consist. Weight: {cons_weight}")
    print(f"{'='*60}\n")
    
    optimizer = torch.optim.Adam(student.parameters(), lr=lr)
    criterion = torch.nn.L1Loss()
    
    student.to(device)
    teacher.to(device)
    
    student.train()
    teacher.eval() # Teacher does not get trained by backprop natively
    
    epoch = 0
    try:
        for epoch in range(num_epochs):
            epoch_loss = 0
            epoch_rec_loss = 0
            epoch_cons_loss = 0
            epoch_contra_loss = 0
            epoch_cycle_loss = 0
            epoch_lgo_loss = 0
            num_pairs = 0
            indices = np.random.permutation(len(train_shapes))
            
            for i in range(0, len(indices)-1, 2):
                idx1 = indices[i]
                idx2 = indices[i+1]
                s1 = train_shapes[idx1]
                s2 = train_shapes[idx2]
                
                p1 = torch.tensor((s1['pos']).copy()).float().unsqueeze(0).to(device)
                f1 = torch.tensor(s1['feat']).float().unsqueeze(0).to(device)
                p2 = torch.tensor((s2['pos']).copy()).float().unsqueeze(0).to(device)
                f2 = torch.tensor(s2['feat']).float().unsqueeze(0).to(device)
                
                # -----------------------------------------------------------------
                # Forward Pass 1 (S1 -> S2)
                # -----------------------------------------------------------------
                # Student predicts deeply masked graph
                pred1_s, _, _, _, enc_t1_s, mask1_s = student(f1, p1, f2, p2, mask_ratio=student_mask_ratio, feature_ratio=student_feat_ratio)
                
                # Teacher predicts lightly masked/unmasked graph
                with torch.no_grad():
                    pred1_t, _, _, _, enc_t1_t, _ = teacher(f1, p1, f2, p2, mask_ratio=teacher_mask_ratio, feature_ratio=teacher_feat_ratio)
                    
                loss1_rec = criterion(pred1_s[mask1_s], f1[mask1_s]) if mask1_s.sum() > 0 else criterion(pred1_s, f1)
                loss1_cons = compute_consistency_loss(pred1_s, pred1_t)
                
                # -----------------------------------------------------------------
                # Forward Pass 2 (S2 -> S1)
                # -----------------------------------------------------------------
                pred2_s, _, _, _, enc_t2_s, mask2_s = student(f2, p2, f1, p1, mask_ratio=student_mask_ratio, feature_ratio=student_feat_ratio)
                with torch.no_grad():
                    pred2_t, _, _, _, enc_t2_t, _ = teacher(f2, p2, f1, p1, mask_ratio=teacher_mask_ratio, feature_ratio=teacher_feat_ratio)
                    
                loss2_rec = criterion(pred2_s[mask2_s], f2[mask2_s]) if mask2_s.sum() > 0 else criterion(pred2_s, f2)
                loss2_cons = compute_consistency_loss(pred2_s, pred2_t)
                
                # -----------------------------------------------------------------
                # Combine and Backdrop
                # -----------------------------------------------------------------
                loss_rec = loss1_rec + loss2_rec
                loss_cons = loss1_cons + loss2_cons
                
                # Extract unmasked full features for the alignment and cycle loss
                z1_s = student.extract_features(f1, p1)
                z2_s = student.extract_features(f2, p2)
                
                # Contrastive Loss
                loss_contra1 = compute_contrastive_loss(z1_s, margin=contra_margin)
                loss_contra2 = compute_contrastive_loss(z2_s, margin=contra_margin)
                loss_contra = loss_contra1 + loss_contra2
                
                # Cycle Consistency Loss
                loss_cycle1 = compute_cycle_loss(z1_s, z2_s, p1, eps=cycle_eps, n_iter=cycle_n_iter)
                loss_cycle2 = compute_cycle_loss(z2_s, z1_s, p2, eps=cycle_eps, n_iter=cycle_n_iter)
                loss_cycle = loss_cycle1 + loss_cycle2
                
                # Global Optimization Loss (L_go)
                loss_lgo1 = compute_lgo_loss(z1_s, z2_s, eps=lgo_eps, n_iter=cycle_n_iter)
                loss_lgo2 = compute_lgo_loss(z2_s, z1_s, eps=lgo_eps, n_iter=cycle_n_iter)
                loss_lgo = loss_lgo1 + loss_lgo2
                
                # We add all losses together using weights from config
                loss = loss_rec + (cons_weight * loss_cons) + (contra_weight * loss_contra) + (cycle_weight * loss_cycle) + (lgo_weight * loss_lgo)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Update EMA of the teacher model natively after student steps forwards
                update_teacher_ema(student, teacher, alpha=ema_alpha)
                
                epoch_loss += loss.item()
                epoch_rec_loss += loss_rec.item()
                epoch_cons_loss += loss_cons.item()
                epoch_contra_loss += loss_contra.item()
                epoch_cycle_loss += loss_cycle.item()
                epoch_lgo_loss += loss_lgo.item()
                num_pairs += 1
                
            avg_loss = epoch_loss / num_pairs if num_pairs > 0 else 0
            avg_rec = epoch_rec_loss / num_pairs if num_pairs > 0 else 0
            avg_cons = epoch_cons_loss / num_pairs if num_pairs > 0 else 0
            avg_contra = epoch_contra_loss / num_pairs if num_pairs > 0 else 0
            avg_cycle = epoch_cycle_loss / num_pairs if num_pairs > 0 else 0
            avg_lgo = epoch_lgo_loss / num_pairs if num_pairs > 0 else 0
            
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch+1:3d}/{num_epochs} | Tot: {avg_loss:.4f} | Rec: {avg_rec:.4f} | Cons: {avg_cons:.4f} | Contra: {avg_contra:.4f} | Cycle: {avg_cycle:.4f} | Lgo: {avg_lgo:.4f}")
        
    except KeyboardInterrupt:
        if epoch >= 100:
            print(f"\nTraining interrupted at epoch {epoch+1}. Saving progress as requested...")
        else:
            print(f"\nTraining interrupted at epoch {epoch+1}. Not saving because < 100 epochs were completed.")
            raise
            
    print("\nTraining complete!")
    return student

def main():
    parser = argparse.ArgumentParser(description="ASMAE Student-Teacher Training")
    parser.add_argument('--config', type=str, default='config/train_st_te_config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
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
    
    train_shapes = load_train_shapes(data_dir, k, t, neigvecs, max_shapes=train_size, output_dir=output_dir)
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
    
    # Initialize Teacher Model (ideally structurally identical, but fully configurable)
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
    
    # Initialize teacher exactly with student's weights initially
    teacher.load_state_dict(student.state_dict())
    
    # Lock teacher parameters against standard backprop
    for param in teacher.parameters():
        param.requires_grad = False
    
    print(f"  Feature dimension: {feature_dim}")
    num_params = sum(p.numel() for p in student.parameters())
    print(f"  Student parameters: {num_params:,}")

    student = train_model(student, teacher, train_shapes, config)
    
    checkpoint_path = os.path.join(config['training']['checkpoint_dir'], config['training']['checkpoint_name'])
    torch.save({
        'model_state_dict': student.state_dict(),
        'feature_dim': feature_dim,
        'train_size': len(train_shapes),
    }, checkpoint_path)
    print(f"\nCheckpoint saved: {checkpoint_path}")

if __name__ == "__main__":
    main()
