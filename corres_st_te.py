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
    parser = argparse.ArgumentParser(description="ASMAE Correspondence")
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
    shape1_path = os.path.join(data_dir, config['shape1'])
    shape2_path = os.path.join(data_dir, config['shape2'])
    
    k = config['geometry']['k']
    t = config['geometry']['t']
    
    eps = config['correspondence']['eps']
    n_iter = config['correspondence']['n_iter']
    
    print("Loading shapes and geometry...")
    # NOTE: Output dir for preprocessing intermediate files is set to out_dir
    V1, El1, feat1, _ = process_geometry(shape1_path, k=k, t=t, output_dir=out_dir)
    V2, El2, feat2, _ = process_geometry(shape2_path, k=k, t=t, output_dir=out_dir)

    V1 = normalize_pc(V1)
    V2 = normalize_pc(V2)

    N1, N2 = V1.shape[0], V2.shape[0]
    
    feature_dim = feat1.shape[1]

    print("Loading ASMAE model...")
    model = load_model(config, feature_dim, device)

    print("Extracting ASMAE features...")
    with torch.no_grad():
        f1 = torch.tensor(feat1, dtype=torch.float32, device=device).unsqueeze(0)
        f2 = torch.tensor(feat2, dtype=torch.float32, device=device).unsqueeze(0)

        p1 = torch.tensor(V1, dtype=torch.float32, device=device).unsqueeze(0)
        p2 = torch.tensor(V2, dtype=torch.float32, device=device).unsqueeze(0)

        z1 = model.extract_features(f1, p1).squeeze(0).cpu().numpy()
        z2 = model.extract_features(f2, p2).squeeze(0).cpu().numpy()

    # ℓ2-normalize (IMPORTANT)
    z1 /= np.linalg.norm(z1, axis=1, keepdims=True) + 1e-8
    z2 /= np.linalg.norm(z2, axis=1, keepdims=True) + 1e-8

    print("Computing feature cost matrix...")
    cost = cdist(z1, z2, metric="sqeuclidean")

    print("Running Sinkhorn...")
    P = sinkhorn(cost, eps=eps, n_iter=n_iter)

    print("Extracting one-direction P2P (shape2 -> shape1)...")
    p2p = np.argmax(P.T, axis=1)  # shape2 -> shape1

    # -------------------------------------------------
    # SAVE (two-column format)
    # -------------------------------------------------
    shape1_name = os.path.splitext(config['shape1'])[0]
    shape2_name = os.path.splitext(config['shape2'])[0]
    out_path = os.path.join(
        out_dir,
        f"p2p_{shape1_name}_to_{shape2_name}_sinkhorn.txt"
    )

    pairs = np.stack([np.arange(N2), p2p], axis=1)
    np.savetxt(out_path, pairs, fmt="%d")

    print("Saved P2P map to:", out_path)
    print(f"Mapped points: {pairs.shape[0]} / {N2}")
    
    # -------------------------------------------------
    # Color Transfer Visualization
    # -------------------------------------------------
    print("Colorizing Shape 1 (Target) and mapping to Shape 2 (Source)...")
    v_min, v_max = V1.min(axis=0), V1.max(axis=0)
    color1 = (V1 - v_min) / (v_max - v_min + 1e-8)  # XYZ -> RGB (0 to 1)

    color2_mapped = np.zeros_like(V2)
    for i in range(N2):
        color2_mapped[i] = color1[p2p[i]]
        
    c1_path = os.path.join(out_dir, f"{shape1_name}_colored_target.obj")
    c2_path = os.path.join(out_dir, f"{shape2_name}_mapped_source.obj")
    
    from utils.mesh import save_obj
    save_obj(c1_path, V1, color1, El1)
    save_obj(c2_path, V2, color2_mapped, El2)
    print(f"Saved Colored Shape 1: {c1_path}")
    print(f"Saved Mapped Shape 2 (from Shape 1 mapping): {c2_path}")
    
if __name__ == "__main__":
    main()