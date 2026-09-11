import os
# Set default OpenGL platform for headless server rendering with DINOv2
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import torch
import numpy as np
import argparse
import yaml

from models.asmae import ASMAE
from core.preprocessing import process_geometry, normalize_pc


def load_model(config, feature_dim, device, checkpoint_path=None):
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

    if checkpoint_path is None:
        checkpoint_path = config.get('correspondence', {}).get('checkpoint_path', 'checkpoints/st_te_model_FAUST.pth')

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {checkpoint_path}")

    return model


def main():
    parser = argparse.ArgumentParser(description="Greedy Cosine Similarity All-Pairs Shape Correspondence")
    parser.add_argument('--config', type=str, default='config/FAUST/corres.yaml', help='Path to corres config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file (overrides config)')
    parser.add_argument('--out_dir', type=str, default=None, help='Output directory for p2p txt results')
    parser.add_argument('--n_shapes', type=int, default=20, help='Number of test shapes to match (last N, default: 20, -1 for all)')
    parser.add_argument('--use_dino', action='store_true', help='Concatenate DINOv2 visual features with ASMAE (as in feature_visualization.ipynb)')
    parser.add_argument('--dino_variant', type=str, default='dinov2_vitb14_reg', help='DINOv2 model variant if --use_dino is set')
    parser.add_argument('--dino_views', type=int, default=4, help='Number of multi-view camera poses around mesh (default: 4)')
    parser.add_argument('--off_dir', type=str, default='input/FAUST/off', help='Directory with .off files (needed for DINOv2)')
    parser.add_argument('--normalize_pc', action='store_true', default=False, help='Apply normalize_pc to coordinates (default: False, to match train_st_te.py)')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = config.get('correspondence', {}).get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = 'cpu'

    out_dir = args.out_dir
    if not out_dir:
        out_dir = config.get('output_dir', 'p2p_results_greedy_FAUST')
        # If output_dir from config is st_te default, distinguish greedy
        if 'st_te' in out_dir:
            out_dir = out_dir.replace('st_te', 'greedy')
        elif out_dir == 'p2p_results_FAUST':
            out_dir = 'p2p_results_greedy_FAUST'
    os.makedirs(out_dir, exist_ok=True)

    data_dir = config['data_dir']
    k = config['geometry']['k']
    t = config['geometry']['t']
    neigvecs = config['geometry'].get('neigvecs', 300)

    # 1. Scan for all shapes
    all_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.obj') or f.endswith('.off')])
    if args.n_shapes > 0 and len(all_files) > args.n_shapes:
        all_files = all_files[-args.n_shapes:]

    if not all_files:
        print(f"No .obj or .off files found in {data_dir}")
        return

    print(f"\n{'='*60}")
    print(f"Greedy All-Pairs Correspondence Matching")
    print(f"  Dataset: {data_dir}")
    print(f"  Shapes to process: {len(all_files)}")
    print(f"  Output directory: {out_dir}")
    print(f"  Using DINOv2: {args.use_dino}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # 2. Pre-load Model
    temp_V, _, temp_feat, _ = process_geometry(os.path.join(data_dir, all_files[0]), k=k, t=t, neigvecs=neigvecs, output_dir=out_dir)
    feature_dim = temp_feat.shape[1]
    ckpt_path = args.checkpoint or config.get('correspondence', {}).get('checkpoint_path', 'checkpoints/st_te_model_FAUST.pth')
    model = load_model(config, feature_dim, device, checkpoint_path=ckpt_path)

    # Optional DINOv2 setup
    if args.use_dino:
        from utils.dino_utils import get_vertex_dino_features
        from utils.mesh import load_off

    # 3. Cache Features
    cached_shapes = {}
    for filename in all_files:
        path = os.path.join(data_dir, filename)
        name = os.path.splitext(filename)[0]

        print(f"  -> Caching features for: {filename}...")
        V, El, feat, _ = process_geometry(path, k=k, t=t, neigvecs=neigvecs, output_dir=out_dir)
        p_coords = normalize_pc(V.copy()) if args.normalize_pc else V.copy()

        with torch.no_grad():
            f_torch = torch.tensor(feat, dtype=torch.float32, device=device).unsqueeze(0)
            p_torch = torch.tensor(p_coords, dtype=torch.float32, device=device).unsqueeze(0)
            z = model.extract_features(f_torch, p_torch).squeeze(0).cpu().numpy()

        # L2-normalize ASMAE features
        z = z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-8)

        if args.use_dino:
            off_file = os.path.join(args.off_dir, f"{name}.off")
            if os.path.exists(off_file):
                V_raw, _, ITris = load_off(off_file)
                dino_feat, _, _ = get_vertex_dino_features(
                    V_raw, ITris, variant=args.dino_variant, n_views=args.dino_views, device=device)
                dino_norm = dino_feat / (np.linalg.norm(dino_feat, axis=1, keepdims=True) + 1e-8)
                z = np.concatenate([z, dino_norm], axis=1)
            else:
                print(f"    [Warning] OFF file not found at {off_file}, skipping DINOv2 for {name}")

        cached_shapes[filename] = {
            'V': V,
            'El': El,
            'Z': z,
            'name': name
        }

    # 4. Compute N*N Greedy Correspondences
    total_pairs = len(all_files) ** 2
    print(f"\nComputing greedy matches for {total_pairs} pairs...")

    pair_count = 0
    for i, file1 in enumerate(all_files):
        s1 = cached_shapes[file1]
        for j, file2 in enumerate(all_files):
            s2 = cached_shapes[file2]
            pair_count += 1

            if pair_count % 50 == 0 or pair_count == total_pairs or pair_count == 1:
                print(f"[{pair_count:4d}/{total_pairs}] Matching {file1} <-> {file2}")

            # -------------------------------------------------------------
            # Greedy Cosine Similarity Matching (feature_visualization.ipynb)
            # -------------------------------------------------------------
            # s2 is the source (column 0 in saved file), s1 is the target (column 1)
            # For each vertex in s2, find the vertex in s1 with maximum cosine similarity:
            # sim matrix shape: [N_s2, N_s1]
            # p2p shape: [N_s2], containing indices in s1
            # -------------------------------------------------------------
            sim = s2['Z'] @ s1['Z'].T
            p2p = np.argmax(sim, axis=1)

            # Save in standard format expected by geodesic_error.py:
            # Filename: p2p_{s1['name']}_to_{s2['name']}.txt
            # Column 0: indices on s2
            # Column 1: mapped indices on s1
            out_name = f"p2p_{s1['name']}_to_{s2['name']}.txt"
            pairs = np.stack([np.arange(len(s2['V'])), p2p], axis=1)
            np.savetxt(os.path.join(out_dir, out_name), pairs, fmt="%d")

    print(f"\nAll {total_pairs} greedy correspondence files saved to: {out_dir}")


if __name__ == "__main__":
    main()

