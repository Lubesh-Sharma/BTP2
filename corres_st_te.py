import os
import torch
import numpy as np
import argparse
import yaml
from scipy.spatial.distance import cdist

from models.asmae import ASMAE
from core.preprocessing import process_geometry, normalize_pc

# -------------------------------------------------
# Sinkhorn (numpy, stable)
# -------------------------------------------------
def sinkhorn(cost, eps=0.05, n_iter=100):
    K = np.exp(-cost / eps)
    u = np.ones(K.shape[0])
    v = np.ones(K.shape[1])

    for _ in range(n_iter):
        u = 1.0 / (K @ v + 1e-8)
        v = 1.0 / (K.T @ u + 1e-8)

    P = (u[:, None] * K) * v[None, :]
    return P

# -------------------------------------------------
# Load ASMAE
# -------------------------------------------------
def load_model(config, feature_dim, device):
    model_cfg = config.get('student_model', config.get('model', {}))
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
    ).to(device)

    checkpoint_path = config['correspondence']['checkpoint_path']
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model

# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ASMAE All-Pairs Correspondence")
    parser.add_argument('--config', type=str, default='config/corres.yaml', help='Path to corres config file')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = config['correspondence'].get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
        
    out_dir = config['output_dir']
    os.makedirs(out_dir, exist_ok=True)
    
    data_dir = config['data_dir']
    k = config['geometry']['k']
    t = config['geometry']['t']
    eps = config['correspondence']['eps']
    n_iter = config['correspondence']['n_iter']
    
    # 1. Scan for all shapes
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.obj') or f.endswith('.off')])
    
    # # Process only the last 20 files, or all if fewer than 20
    # commment out this line in case of the shrec_19
    # all_files = all_files[-20:]
    
    if not all_files:
        print(f"No .obj or .off files found in {data_dir}")
        return
        
    print(f"Found {len(all_files)} shapes. Pre-extracting features for N^2 matching...")
    
    # 2. Pre-load Model
    # We need feature_dim, let's load one file to get it
    temp_V, _, temp_feat, _ = process_geometry(os.path.join(data_dir, all_files[0]), k=k, t=t, output_dir=out_dir)
    feature_dim = temp_feat.shape[1]
    model = load_model(config, feature_dim, device)
    
    # 3. Cache Features (Pre-computing helps for N^2 pairs)
    cached_shapes = {}
    for filename in all_files:
        path = os.path.join(data_dir, filename)
        name = os.path.splitext(filename)[0]
        
        print(f"  -> Caching: {filename}")
        V, El, feat, _ = process_geometry(path, k=k, t=t, output_dir=out_dir)
        V_norm = normalize_pc(V)
        
        with torch.no_grad():
            f_torch = torch.tensor(feat, dtype=torch.float32, device=device).unsqueeze(0)
            p_torch = torch.tensor(V_norm, dtype=torch.float32, device=device).unsqueeze(0)
            z = model.extract_features(f_torch, p_torch).squeeze(0).cpu().numpy()
            
        z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-8
        
        cached_shapes[filename] = {
            'V': V_norm,
            'El': El,
            'Z': z,
            'name': name
        }

    # 4. Compute N*N Correspondences
    print(f"\nProcessing {len(all_files)**2} pairs...")
    
    for i, file1 in enumerate(all_files):
        s1 = cached_shapes[file1]
        for j, file2 in enumerate(all_files):
            s2 = cached_shapes[file2]
            
            # Skip diagonal? (Optional, user asked for all n*n including self-mapping)
            print(f"[{i*len(all_files) + j + 1}/{len(all_files)**2}] {file1} -> {file2}")
            
            # Compute Cost
            cost = cdist(s1['Z'], s2['Z'], metric="sqeuclidean")
            
            # Sinkhorn
            P = sinkhorn(cost, eps=eps, n_iter=n_iter)
            
            # P2P: shape2 -> shape1
            p2p = np.argmax(P.T, axis=1) 
            
            # Save P2P txt
            out_name = f"p2p_{s1['name']}_to_{s2['name']}.txt"
            pairs = np.stack([np.arange(len(s2['V'])), p2p], axis=1)
            np.savetxt(os.path.join(out_dir, out_name), pairs, fmt="%d")

    print(f"\nDone! All results saved to: {out_dir}")
    
if __name__ == "__main__":
    main()