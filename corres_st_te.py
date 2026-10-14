import os
import torch
import numpy as np
import argparse
import yaml
import csv
from scipy.spatial.distance import cdist

from models.asmae import ASMAE
from models.modules.orientation_module import OrientNet
from models.oriented_pair import prepare_aligned_pair, load_orientation_checkpoint
from core.preprocessing import process_geometry
from utils.correspondence_io import select_correspondence_files

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
# Load ASMAE & OrientNet
# -------------------------------------------------
def load_models(config, feature_dim, device, checkpoint=None):
    checkpoint_path = config['correspondence']['checkpoint_path']
    if checkpoint is None:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model_cfg = checkpoint.get('student_model_config', config.get('student_model', config.get('model', {})))
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

    orient_net = OrientNet().to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    load_orientation_checkpoint(orient_net, checkpoint)
    print("Loaded OrientNet weights from checkpoint.")
    orient_net.eval()

    return model, orient_net

# -------------------------------------------------
# MAIN
# -------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ASMAE All-Pairs Correspondence with Orientation Alignment")
    parser.add_argument('--config', type=str, default='config/SHREC/corres.yaml', help='Path to corres config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Override checkpoint (best validation weights by default)')
    parser.add_argument('--out_dir', type=str, default=None, help='Override output directory')
    parser.add_argument('--alignment', choices=['predicted', 'identity'], default='predicted',
                        help='Use OEM rotation, or compare against the original orientation')
    parser.add_argument('--normalize_pc', action='store_true', default=False, help='Compatibility flag; OEM now always normalizes both clouds as in training')
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
        
    device = config['correspondence'].get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'
        
    out_dir = args.out_dir or config['output_dir']
    if args.out_dir is None and args.alignment == 'identity':
        out_dir += '_identity'
    os.makedirs(out_dir, exist_ok=True)
    
    data_dir = config['data_dir']
    k = config['geometry']['k']
    t = config['geometry']['t']
    eps = config['correspondence']['eps']
    n_iter = config['correspondence']['n_iter']
    neigvecs = config['geometry'].get('neigvecs', 300)
    if args.checkpoint:
        config['correspondence']['checkpoint_path'] = args.checkpoint
    checkpoint = torch.load(config['correspondence']['checkpoint_path'], map_location=device, weights_only=True)
    geometry = checkpoint.get('geometry_config', config['geometry'])
    k, t = geometry['k'], geometry['t']
    neigvecs = geometry.get('neigvecs', 300)
    model, orient_net = load_models(config, checkpoint['feature_dim'], device, checkpoint)
    
    # 1. Scan for all shapes
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.obj') or f.endswith('.off')])
    
    # Match the held-out partition used by training, then limit to 20 shapes.
    all_files = select_correspondence_files(all_files, checkpoint, data_dir)
    
    if not all_files:
        print(f"No held-out .obj or .off files found in {data_dir}")
        return
        
    print(f"Found {len(all_files)} shapes. Computing correspondence with orientation alignment...")
    
    # 3. Load Shape Geometry and Features
    loaded_shapes = {}
    for filename in all_files:
        path = os.path.join(data_dir, filename)
        name = os.path.splitext(filename)[0]
        
        print(f"  -> Loading: {filename}")
        V, El, feat, _ = process_geometry(path, k=k, t=t, neigvecs=neigvecs, output_dir=out_dir)
        p_coords = V.copy()
        
        loaded_shapes[filename] = {
            'V': V,
            'El': El,
            'pos': p_coords,
            'feat': feat,
            'name': name
        }

    # 4. Compute N*N Correspondences with Pairwise Orientation Alignment
    print(f"\nProcessing {len(all_files)**2} pairs...")
    alignment_log = []
    
    for i, file1 in enumerate(all_files):
        s1 = loaded_shapes[file1]
        p1_tensor = torch.tensor(s1['pos'], dtype=torch.float32, device=device).unsqueeze(0)
        f1_tensor = torch.tensor(s1['feat'], dtype=torch.float32, device=device).unsqueeze(0)
        
        for j, file2 in enumerate(all_files):
            s2 = loaded_shapes[file2]
            print(f"[{i*len(all_files) + j + 1}/{len(all_files)**2}] {file1} -> {file2}")
            
            p2_tensor = torch.tensor(s2['pos'], dtype=torch.float32, device=device).unsqueeze(0)
            f2_tensor = torch.tensor(s2['feat'], dtype=torch.float32, device=device).unsqueeze(0)
            
            with torch.no_grad():
                # Use the same cross-shape OEM -> ASMAE inputs as training.
                f1_aligned, p1_aligned, f2_aligned, p2_aligned, outputs = prepare_aligned_pair(
                    orient_net, f1_tensor, p1_tensor, f2_tensor, p2_tensor, args.alignment
                )
                probabilities = outputs['angle_x'].softmax(-1)
                angle_bin = int(probabilities.argmax(-1).item())
                undo_degrees = -float(orient_net.ANGLE[angle_bin].item()) * 180.0 / np.pi
                alignment_log.append([s1['name'], s2['name'], args.alignment, angle_bin,
                                      undo_degrees, float(probabilities.max().item())])
                
                # Extract ASMAE features on orientation-aligned coordinates
                # Source alignment depends on this target; encode it per pair.
                z1 = model.extract_features(f1_aligned, p1_aligned).squeeze(0).cpu().numpy()
                z1 /= np.linalg.norm(z1, axis=1, keepdims=True) + 1e-8
                z2 = model.extract_features(f2_aligned, p2_aligned).squeeze(0).cpu().numpy()
                z2 /= np.linalg.norm(z2, axis=1, keepdims=True) + 1e-8
            
            # Compute Cost
            cost = cdist(z1, z2, metric="sqeuclidean")
            
            # Sinkhorn
            P = sinkhorn(cost, eps=eps, n_iter=n_iter)
            
            # P2P: shape2 -> shape1
            p2p = np.argmax(P.T, axis=1) 
            
            # Save P2P txt
            out_name = f"p2p_{s1['name']}_to_{s2['name']}.txt"
            pairs = np.stack([np.arange(len(s2['V'])), p2p], axis=1)
            np.savetxt(os.path.join(out_dir, out_name), pairs, fmt="%d")

    with open(os.path.join(out_dir, 'alignment.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['source', 'target_reference', 'mode', 'predicted_bin',
                         'predicted_undo_yaw_degrees', 'confidence'])
        writer.writerows(alignment_log)
    print(f"\nDone! All results and alignment.csv saved to: {out_dir}")
    

if __name__ == "__main__":
    main()
