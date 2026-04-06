import os
import torch
import numpy as np
import sys
import yaml
import argparse

from core.preprocessing import process_geometry
from models.asmae import ASMAE
from utils.files import save_masked_matrix

def load_test_shapes(data_dir, k, t, neigvecs, start_idx, num_shapes, output_dir):
    """Load test .obj files from directory"""
    print(f"\n{'='*60}")
    print(f"Loading {num_shapes} test shapes from: {data_dir}")
    print(f"{'='*60}")
    
    obj_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.obj')])
    obj_files = obj_files[start_idx:start_idx+num_shapes]
        
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

def test_model(model, test_shapes, config):
    device = config['testing'].get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
        
    output_dir = config['output_dir']
    mask_ratio = config['testing']['mask_ratio']
    feature_ratio = config['testing'].get('feature_ratio', 0.2)
    
    print(f"\n{'='*60}")
    print("TESTING PHASE")
    print(f"  Test shapes: {len(test_shapes)}")
    print(f"  Output dir: {output_dir}")
    print(f"{'='*60}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    model.to(device)
    model.eval()
    
    total_mae = 0.0
    total_masked_elements = 0
    
    with torch.no_grad():
        for i in range(len(test_shapes) - 1):
            s1 = test_shapes[i]
            s2 = test_shapes[i+1]
            
            print(f"[{i+1}/{len(test_shapes)-1}] Testing: {s1['name']} -> {s2['name']}")
            
            p1 = torch.tensor((s1['pos']).copy()).float().unsqueeze(0).to(device)
            f1 = torch.tensor(s1['feat']).float().unsqueeze(0).to(device)
            p2 = torch.tensor((s2['pos']).copy()).float().unsqueeze(0).to(device)
            f2 = torch.tensor(s2['feat']).float().unsqueeze(0).to(device)
            
            pred1, _, _, f1_masked, _, mask1 = model(f1, p1, f2, p2, mask_ratio=mask_ratio, feature_ratio=feature_ratio)
            pred2, _, _, f2_masked, _, mask2 = model(f2, p2, f1, p1, mask_ratio=mask_ratio, feature_ratio=feature_ratio)
            
            if mask1 is not None and mask1.sum() > 0:
                err1 = torch.abs(pred1[mask1] - f1[mask1])
                total_mae += err1.sum().item()
                total_masked_elements += err1.numel()
            
            if mask2 is not None and mask2.sum() > 0:
                err2 = torch.abs(pred2[mask2] - f2[mask2])
                total_mae += err2.sum().item()
                total_masked_elements += err2.numel()
                
            rec1 = f1.clone()
            if mask1 is not None and mask1.sum() > 0:
                rec1[mask1] = pred1[mask1]
            
            rec2 = f2.clone()
            if mask2 is not None and mask2.sum() > 0:
                rec2[mask2] = pred2[mask2]
                
            pair_dir = os.path.join(output_dir, f"{s1['name']}_to_{s2['name']}")
            os.makedirs(pair_dir, exist_ok=True)
            
            save_masked_matrix(f1_masked.squeeze(0).cpu().numpy(), os.path.join(pair_dir, f"masked_{s1['name']}.txt"))
            save_masked_matrix(f2_masked.squeeze(0).cpu().numpy(), os.path.join(pair_dir, f"masked_{s2['name']}.txt"))
            np.savetxt(os.path.join(pair_dir, f"reconstructed_{s1['name']}.txt"), rec1.squeeze(0).cpu().numpy())
            np.savetxt(os.path.join(pair_dir, f"reconstructed_{s2['name']}.txt"), rec2.squeeze(0).cpu().numpy())
            print(f"  -> Saved to {pair_dir}")
            
    if total_masked_elements > 0:
        avg_mae = total_mae / total_masked_elements
        print(f"\nAverage MAE on masked points (test set): {avg_mae:.6f}")
    else:
        print("\nWARNING: No masked elements found during testing")
    print(f"\nTesting complete! All results saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="ASMAE Testing")
    parser.add_argument('--config', type=str, default='config/test_config.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file. Defaults to config file default.')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        checkpoint_path = os.path.join(config['testing']['checkpoint_dir'], config['testing']['checkpoint_name'])
        
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        return
        
    print(f"Loading checkpoint: {checkpoint_path}")
    device = config['testing'].get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
        
    checkpoint = torch.load(checkpoint_path, map_location=device)
    feature_dim = checkpoint['feature_dim']
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
    
    model.load_state_dict(checkpoint['model_state_dict'])
    print("Checkpoint loaded successfully")
    
    data_dir = config['data_dir']
    output_dir = config['output_dir']
    k = config['geometry']['k']
    t = config['geometry']['t']
    neigvecs = config['geometry'].get('neigvecs', 300)
    train_size = config['train_size']
    test_size = config['test_size']
    
    test_shapes = load_test_shapes(data_dir, k, t, neigvecs, start_idx=train_size, num_shapes=test_size, output_dir=output_dir)
    if len(test_shapes) < 2:
        print("Not enough shapes for testing.")
        return
        
    test_model(model, test_shapes, config)

if __name__ == "__main__":
    main()
