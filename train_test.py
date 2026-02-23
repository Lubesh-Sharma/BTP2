"""
Simple Training and Testing Script for ASMAE
- Train on 50-60 shapes (no matrix saving)
- Test on 20 shapes (save matrices and reconstructions)
No external dependencies except torch, numpy
"""

import argparse
import os
import torch
import numpy as np
import sys

# Import your existing modules
from core.preprocessing import process_geometry, normalize_pc
from models.asmae import ASMAE
from utils.files import save_masked_matrix

def load_all_shapes(data_dir, k=30, t=8, max_shapes=None):
    """Load all .obj files from directory"""
    obj_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.obj')])
    
    if max_shapes:
        obj_files = obj_files[:max_shapes]
    
    print(f"\n{'='*60}")
    print(f"Loading {len(obj_files)} shapes from:")
    print(f"  {data_dir}")
    print(f"{'='*60}")
    
    shapes = []
    
    for i, obj_file in enumerate(obj_files):
        print(f"[{i+1}/{len(obj_files)}] Loading: {obj_file}...", end=' ')
        sys.stdout.flush()
        
        try:
            path = os.path.join(data_dir, obj_file)
            name = os.path.splitext(obj_file)[0]
            
            # Load geometry
            VPos, El, Feat, eigvecs = process_geometry(path, k, t)
            
            # Validate
            if VPos is None or len(VPos) == 0:
                print("SKIP (no vertices)")
                continue
            
            if Feat is None or len(Feat) == 0:
                print("SKIP (no features)")
                continue
            
            # Handle point clouds (no faces)
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


def train_model(model, train_shapes, device, num_epochs=100, lr=1e-4):
    """Train model on shape pairs"""
    print(f"\n{'='*60}")
    print("TRAINING PHASE")
    print(f"  Training shapes: {len(train_shapes)}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.L1Loss()
    
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0
        num_pairs = 0
        
        # Train on random pairs
        indices = np.random.permutation(len(train_shapes))
        
        for i in range(0, len(indices)-1, 2):
            idx1 = indices[i]
            idx2 = indices[i+1]
            
            s1 = train_shapes[idx1]
            s2 = train_shapes[idx2]
            
            # Normalize
            pos1_norm = (s1['pos'].copy())
            pos2_norm = (s2['pos'].copy())
            
            # To tensors
            p1 = torch.tensor(pos1_norm).float().unsqueeze(0).to(device)
            f1 = torch.tensor(s1['feat']).float().unsqueeze(0).to(device)
            p2 = torch.tensor(pos2_norm).float().unsqueeze(0).to(device)
            f2 = torch.tensor(s2['feat']).float().unsqueeze(0).to(device)
            
            # Forward pass: s1 -> s2
            pred1, _, _, _, _, mask1 = model(f1, p1, f2, p2, mask_ratio=0.4)
            loss1 = criterion(pred1[mask1], f1[mask1]) if mask1.sum() > 0 else criterion(pred1, f1)
            
            # Forward pass: s2 -> s1
            pred2, _, _, _, _, mask2 = model(f2, p2, f1, p1, mask_ratio=0.4)
            loss2 = criterion(pred2[mask2], f2[mask2]) if mask2.sum() > 0 else criterion(pred2, f2)
            
            # Total loss
            loss = loss1 + loss2
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_pairs += 1
        
        avg_loss = epoch_loss / num_pairs if num_pairs > 0 else 0
        
        # Print progress
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | Loss: {avg_loss:.6f}")
    
    print("\nTraining complete!")
    return model


def test_model(model, test_shapes, device, output_dir):
    """Test model, save matrices, and report masked MAE"""
    print(f"\n{'='*60}")
    print("TESTING PHASE")
    print(f"  Test shapes: {len(test_shapes)}")
    print(f"  Output dir: {output_dir}")
    print(f"{'='*60}\n")
    
    os.makedirs(output_dir, exist_ok=True)
    model.eval()
    
    total_mae = 0.0
    total_masked_elements = 0
    
    with torch.no_grad():
        # Test each shape pair
        for i in range(len(test_shapes) - 1):
            s1 = test_shapes[i]
            s2 = test_shapes[i + 1]
            
            print(f"[{i+1}/{len(test_shapes)-1}] Testing: {s1['name']} -> {s2['name']}")
            
            # Normalize positions
            pos1_norm = (s1['pos'].copy())
            pos2_norm = (s2['pos'].copy())
            
            # To tensors
            p1 = torch.tensor(pos1_norm).float().unsqueeze(0).to(device)
            f1 = torch.tensor(s1['feat']).float().unsqueeze(0).to(device)
            p2 = torch.tensor(pos2_norm).float().unsqueeze(0).to(device)
            f2 = torch.tensor(s2['feat']).float().unsqueeze(0).to(device)
            
            # Forward pass
            pred1, _, _, f1_masked, _, mask1 = model(f1, p1, f2, p2, mask_ratio=0.4)
            pred2, _, _, f2_masked, _, mask2 = model(f2, p2, f1, p1, mask_ratio=0.4)
            
            # ---- Masked MAE computation ----
            if mask1 is not None and mask1.sum() > 0:
                err1 = torch.abs(pred1[mask1] - f1[mask1])
                total_mae += err1.sum().item()
                total_masked_elements += err1.numel()
            
            if mask2 is not None and mask2.sum() > 0:
                err2 = torch.abs(pred2[mask2] - f2[mask2])
                total_mae += err2.sum().item()
                total_masked_elements += err2.numel()
            
            # Reconstruct (blend masked predictions)
            rec1 = f1.clone()
            if mask1 is not None and mask1.sum() > 0:
                rec1[mask1] = pred1[mask1]
            
            rec2 = f2.clone()
            if mask2 is not None and mask2.sum() > 0:
                rec2[mask2] = pred2[mask2]
            
            # Save results
            pair_dir = os.path.join(output_dir, f"{s1['name']}_to_{s2['name']}")
            os.makedirs(pair_dir, exist_ok=True)
            
            save_masked_matrix(
                f1_masked.squeeze(0).cpu().numpy(),
                os.path.join(pair_dir, f"masked_{s1['name']}.txt")
            )
            save_masked_matrix(
                f2_masked.squeeze(0).cpu().numpy(),
                os.path.join(pair_dir, f"masked_{s2['name']}.txt")
            )
            
            np.savetxt(
                os.path.join(pair_dir, f"reconstructed_{s1['name']}.txt"),
                rec1.squeeze(0).cpu().numpy()
            )
            np.savetxt(
                os.path.join(pair_dir, f"reconstructed_{s2['name']}.txt"),
                rec2.squeeze(0).cpu().numpy()
            )
            
            print(f"  -> Saved to {pair_dir}")
    
    # ---- Final MAE report ----
    if total_masked_elements > 0:
        avg_mae = total_mae / total_masked_elements
        print(f"\nAverage MAE on masked points (test set): {avg_mae:.6f}")
    else:
        print("\nWARNING: No masked elements found during testing")
    
    print(f"\nTesting complete! All results saved to: {output_dir}")


# def save_masked_matrix(matrix, filepath):
#     """Save matrix in sparse format (only non-masked rows)"""
#     with open(filepath, 'w') as f:
#         for i, row in enumerate(matrix):
#             # Only save rows that are not all zeros (i.e., not masked)
#             if not np.allclose(row, 0):
#                 f.write(f"{i} " + " ".join(map(str, row)) + "\n")


def main():
    parser = argparse.ArgumentParser(description="ASMAE Train and Test")
    parser.add_argument('--data_dir', type=str, 
                        default='input/diffusion_knn_k=5/FAUST_Dataset',
                        help='Directory with .obj files')
    parser.add_argument('--output_dir', type=str, default='output_test',
                        help='Output directory for test results')
    parser.add_argument('--train_size', type=int, default=60,
                       help='Number of shapes for training')
    parser.add_argument('--test_size', type=int, default=20,
                       help='Number of shapes for testing')
    parser.add_argument('--epochs', type=int, default=1500,
                       help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device: cuda or cpu')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint (for testing only)')
    parser.add_argument('--test_only', action='store_true',
                       help='Skip training, only test')
    
    args = parser.parse_args()
    
    # Device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
    else:
        device = args.device
    
    # Ensure necessary directories exist
    os.makedirs('output', exist_ok=True)
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print("ASMAE TRAINING AND TESTING")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Data directory: {args.data_dir}")
    
    # Load all shapes
    total_needed = args.train_size + args.test_size
    all_shapes = load_all_shapes(args.data_dir, k=30, t=8, max_shapes=total_needed)
    
    if len(all_shapes) < total_needed:
        print(f"\nWARNING: Only loaded {len(all_shapes)} shapes, needed {total_needed}")
        print(f"Adjusting train/test split...")
        args.train_size = int(len(all_shapes) * 0.75)
        args.test_size = len(all_shapes) - args.train_size
    
    # Split
    train_shapes = all_shapes[:args.train_size]
    test_shapes = all_shapes[args.train_size:args.train_size + args.test_size]
    
    print(f"\n{'='*60}")
    print("DATASET SPLIT")
    print(f"{'='*60}")
    print(f"  Training shapes: {len(train_shapes)}")
    print(f"  Testing shapes: {len(test_shapes)}")
    print(f"  Training pairs: {len(train_shapes)//2}")
    print(f"  Test pairs: {len(test_shapes)-1}")
    
    # Initialize model
    feature_dim = train_shapes[0]['feat'].shape[1]
    print(f"  Feature dimension: {feature_dim}")
    
    model = ASMAE(
        feature_dim=feature_dim,
        embed_dim=128,
        depth=4,
        num_heads=4,
        decoder_embed_dim=64,
        decoder_depth=2,
        decoder_num_heads=4,
        mlk_ratio=2.0  # Note: it's mlk_ratio, not mlp_ratio
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {num_params:,}")
    
    # Training or load checkpoint
    if args.test_only or args.checkpoint:
        if args.checkpoint and os.path.exists(args.checkpoint):
            print(f"\nLoading checkpoint: {args.checkpoint}")
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print("Checkpoint loaded successfully")
        else:
            print("\nWARNING: No checkpoint provided or not found, using untrained model")
    else:
        # Train
        model = train_model(model, train_shapes, device, 
                          num_epochs=args.epochs, lr=args.lr)
        
        # Save checkpoint
        os.makedirs('checkpoints', exist_ok=True)
        checkpoint_path = 'checkpoints/trained_model.pth'
        torch.save({
            'model_state_dict': model.state_dict(),
            'feature_dim': feature_dim,
            'train_size': len(train_shapes),
        }, checkpoint_path)
        print(f"\nCheckpoint saved: {checkpoint_path}")
    
    # Test
    test_model(model, test_shapes, device, args.output_dir)
    
    print(f"\n{'='*60}")
    print("ALL DONE!")
    print(f"{'='*60}")
    print(f"Training checkpoint: checkpoints/trained_model.pth")
    print(f"Test results: {args.output_dir}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()