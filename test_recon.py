import os
import torch
import numpy as np
import sys
import yaml
import argparse

from models.asmae import build_asmae
from utils.data import dataset_split, load_shapes
from utils.logger import start_logging
from utils.files import save_masked_matrix

def test_model(model, test_shapes, config):
    device = config['testing'].get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
        
    output_dir = os.path.join(config.get('result_dir', 'results'), 'recon')
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
    parser = argparse.ArgumentParser(description="Masked-reconstruction test of the single encoder")
    parser.add_argument('--config', type=str, default='config/FAUST/test_config.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    args = parser.parse_args()
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    start_logging(config.get('result_dir', 'results'), 'test_recon')
    checkpoint_path = args.checkpoint or os.path.join(config['testing']['checkpoint_dir'],
                                                      config['testing']['checkpoint_name'])
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint {checkpoint_path} not found.")
        return
    device = config['testing'].get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        device = 'cpu'
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model = build_asmae(config.get('model', checkpoint['config'].get('model')), checkpoint['feature_dim'],
                        checkpoint.get('dino_dim', 0))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded {checkpoint_path}")
    _, _, test_files = dataset_split(config)
    test_shapes = load_shapes(config, test_files, with_dino=False, label='test shapes')
    if len(test_shapes) < 2:
        print("Not enough shapes for testing.")
        return
    test_model(model, test_shapes, config)


if __name__ == "__main__":
    main()
