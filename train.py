import os
import torch
import numpy as np
import sys
import yaml
import argparse

from core.preprocessing import process_geometry
from models.asmae import ASMAE

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

def train_model(model, train_shapes, config):
    device = config['training'].get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    
    num_epochs = config['training']['epochs']
    lr = config['training']['lr']
    mask_ratio = config['training']['mask_ratio']
    feature_ratio = config['training'].get('feature_ratio', 0.2)
    
    print(f"\n{'='*60}")
    print("TRAINING PHASE")
    print(f"  Training shapes: {len(train_shapes)}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.L1Loss()
    model.to(device)
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
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
            
            pred1, _, _, _, _, mask1 = model(f1, p1, f2, p2, mask_ratio=mask_ratio, feature_ratio=feature_ratio)
            loss1 = criterion(pred1[mask1], f1[mask1]) if mask1.sum() > 0 else criterion(pred1, f1)
            
            pred2, _, _, _, _, mask2 = model(f2, p2, f1, p1, mask_ratio=mask_ratio, feature_ratio=feature_ratio)
            loss2 = criterion(pred2[mask2], f2[mask2]) if mask2.sum() > 0 else criterion(pred2, f2)
            
            loss = loss1 + loss2
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_pairs += 1
            
        avg_loss = epoch_loss / num_pairs if num_pairs > 0 else 0
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | Loss: {avg_loss:.6f}")
            
    print("\nTraining complete!")
    return model

def main():
    parser = argparse.ArgumentParser(description="ASMAE Training")
    parser.add_argument('--config', type=str, default='config/train_config.yaml', help='Path to config file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    os.makedirs('output', exist_ok=True)
    os.makedirs(config['training']['checkpoint_dir'], exist_ok=True)
    
    train_size = config['train_size']
    data_dir = config['data_dir']
    output_dir = config['output_dir']
    k = config['geometry']['k']
    t = config['geometry']['t']
    neigvecs = config['geometry'].get('neigvecs', 300)
    
    train_shapes = load_train_shapes(data_dir, k, t, neigvecs, max_shapes=train_size, output_dir=output_dir)
    if len(train_shapes) < 2:
        print("Not enough shapes for training.")
        return
        
    feature_dim = train_shapes[0]['feat'].shape[1]
    model_cfg = config['model']
    model = ASMAE(
        feature_dim=feature_dim,
        embed_dim=model_cfg['embed_dim'],
        depth=model_cfg['depth'],
        num_heads=model_cfg['num_heads'],
        decoder_embed_dim=model_cfg['decoder_embed_dim'],
        decoder_depth=model_cfg['decoder_depth'],
        decoder_num_heads=model_cfg['decoder_num_heads'],
        mlk_ratio=model_cfg['mlk_ratio'],
        num_mask_queries=model_cfg.get('num_mask_queries', 5000),
        encoder_k=model_cfg.get('encoder_k', 20),
        aamg_k=model_cfg.get('aamg_k', 10),
        aamg_emb_dim=model_cfg.get('aamg_emb_dim', 64),
        pos_embed_dim=model_cfg.get('pos_embed_dim', 64),
        temperature=model_cfg.get('temperature', 1.0)
    )
    
    print(f"  Feature dimension: {feature_dim}")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")

    model = train_model(model, train_shapes, config)
    
    checkpoint_path = os.path.join(config['training']['checkpoint_dir'], config['training']['checkpoint_name'])
    torch.save({
        'model_state_dict': model.state_dict(),
        'feature_dim': feature_dim,
        'train_size': len(train_shapes),
    }, checkpoint_path)
    print(f"\nCheckpoint saved: {checkpoint_path}")

if __name__ == "__main__":
    main()
