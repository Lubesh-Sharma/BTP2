import torch
import torch.optim as optim
import numpy as np
import os
import argparse
from point_mae_model import MaskedAutoencoder
from trimesh import load_obj, load_off

# Libraries used:
# - torch: Deep learning framework for training the MAE.
# - numpy: For data loading and preprocessing.
# - argparse: For parsing command line arguments.
# - os: For file path handling.

def load_data(obj_path, matrix_path):
    """
    Loads geometry and features for a shape.
    
    Args:
        obj_path (str): Path to .obj/.off file.
        matrix_path (str): Path to .txt feature matrix.
        
    Returns:
        tuple: (VPos [N,3], Features [N, K])
    """
    print(f"Loading geometry from {obj_path}...")
    
    input_obj = obj_path
    if not os.path.exists(input_obj) and os.path.exists(os.path.join("input", obj_path)):
         input_obj = os.path.join("input", obj_path)
         
    if input_obj.endswith('.obj'):
        VPos, _, _ = load_obj(input_obj)
    else:
        # Fallback to OFF if needed
        VPos, _, _ = load_off(input_obj)
    
    print(f"Loading features from {matrix_path}...")
    
    input_mat = matrix_path
    if not os.path.exists(input_mat) and os.path.exists(os.path.join("output", matrix_path)):
         input_mat = os.path.join("output", matrix_path)
         
    features = np.loadtxt(input_mat)
    
    # Check consistency
    if VPos.shape[0] != features.shape[0]:
        print(f"Warning: vertex count {VPos.shape[0]} != feature count {features.shape[0]}")
        # Truncate to min
        n = min(VPos.shape[0], features.shape[0])
        VPos = VPos[:n]
        features = features[:n]
        
    return VPos, features

def main():
    """
    Main training script.
    
    1. Parse arguments.
    2. Load Source and Target data.
    3. Normalize positions.
    4. Initialize PointMAE model.
    5. Run training loop (forward, loss, backward).
    6. Save model checkpoint.
    """
    parser = argparse.ArgumentParser(description="Train PointMAE on source and target shapes")
    parser.add_argument("--source_obj", type=str, default="shape_faust_1.obj")
    # ... (rest of the function)
    parser.add_argument("--target_obj", type=str, default="shape_faust_2.obj")
    parser.add_argument("--source_mat", type=str, default="matrix_shape_faust_1.txt")
    parser.add_argument("--target_mat", type=str, default="matrix_shape_faust_2.txt")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1) # Full batch usually for point clouds if fitting in mem
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--mask_ratio", type=float, default=0.6)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # Output dir
    os.makedirs("output", exist_ok=True)
    
    print("Libraries used: torch, numpy, argparse, os")
    
    # Load Data
    pos_s, feat_s = load_data(args.source_obj, args.source_mat)
    pos_t, feat_t = load_data(args.target_obj, args.target_mat)
    
    # Normalize positions (center and scale to unit sphere)
    def normalize_pc(points):
        centroid = np.mean(points, axis=0)
        points -= centroid
        scale = np.max(np.linalg.norm(points, axis=1))
        points /= scale
        return points
        
    pos_s = normalize_pc(pos_s)
    pos_t = normalize_pc(pos_t)
    
    # Convert to Tensor
    # Model expects [B, N, C]
    # We will treat source and target as a batch of 2
    
    # Check if N is same. If not, we have to pad or batch size=1
    if pos_s.shape[0] != pos_t.shape[0]:
        print("Shapes have different number of vertices. Training with batch_size=1 sequential.")
        data_list = [(pos_s, feat_s), (pos_t, feat_t)]
    else:
        # Stack
        positions = np.stack([pos_s, pos_t])
        features = np.stack([feat_s, feat_t])
        data_list = [(positions, features)]
        
    feature_dim = feat_s.shape[1]
    
    # Initialize Model
    model = MaskedAutoencoder(
        feature_dim=feature_dim,
        embed_dim=128,
        depth=4,
        num_heads=4,
        decoder_embed_dim=64,
        decoder_depth=2,
        decoder_num_heads=4,
        mlp_ratio=2.
    ).to(args.device)
    
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
    
    print(f"Start training on device: {args.device}")
    model.train()
    
    for epoch in range(args.epochs):
        total_loss = 0
        count = 0
        
        # Iterate over data
        # If sizes differ, we loop. If same, data_list has 1 item of batch 2.
        for batch_pos, batch_feat in data_list:
            # Prepare tensors
            if len(batch_pos.shape) == 2: # Single sample [N, 3] -> [1, N, 3]
                 batch_pos = batch_pos[np.newaxis, ...]
                 batch_feat = batch_feat[np.newaxis, ...]
            
            x = torch.from_numpy(batch_feat).float().to(args.device)
            pos = torch.from_numpy(batch_pos).float().to(args.device)
            
            optimizer.zero_grad()
            
            loss, pred, mask = model(x, pos, mask_ratio=args.mask_ratio)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            count += 1
            
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{args.epochs}, Loss: {total_loss/count:.6f}")
            
    print("Training finished.")
    
    # Save model
    torch.save(model.state_dict(), os.path.join("output", "point_mae.pth"))
    print("Model saved to output/point_mae.pth")

if __name__ == "__main__":
    main()
