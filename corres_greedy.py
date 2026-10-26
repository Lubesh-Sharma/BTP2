import os
# Set default OpenGL platform for headless server rendering with DINOv2
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import torch
import numpy as np
import argparse
import yaml

from models.asmae import ASMAE
from models.modules.orientation_module import OrientNet, normalize_pc_torch
from core.preprocessing import process_geometry, normalize_pc


def load_model(config, feature_dim, device, checkpoint_path=None, network="student"):
    """Use the same checkpoint pairing and validation as Sinkhorn inference."""
    from corres_st_te import load_models
    config = dict(config)
    config['correspondence'] = dict(config.get('correspondence', {}))
    if checkpoint_path is not None:
        config['correspondence']['checkpoint_path'] = checkpoint_path
    return load_models(config, feature_dim, device, network=network)


def main():
    parser = argparse.ArgumentParser(description="Greedy Cosine Similarity All-Pairs Shape Correspondence with Orientation Alignment")
    parser.add_argument('--config', type=str, default='config/FAUST/corres.yaml', help='Path to corres config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint file (overrides config)')
    parser.add_argument('--out_dir', type=str, default=None, help='Output directory for p2p txt results')
    parser.add_argument('--n_shapes', type=int, default=None, help='Last N shapes; defaults to config test_size or 20; -1 for all')
    parser.add_argument('--use_dino', action='store_true', help='Concatenate DINOv2 visual features with ASMAE (as in feature_visualization.ipynb)')
    parser.add_argument('--dino_variant', type=str, default='dinov2_vitb14_reg', help='DINOv2 model variant if --use_dino is set')
    parser.add_argument('--dino_views', type=int, default=4, help='Number of multi-view camera poses around mesh (default: 4)')
    parser.add_argument('--off_dir', type=str, default='input/FAUST/off', help='Directory with .off files (needed for DINOv2)')
    parser.add_argument('--normalize_pc', action='store_true', default=False, help='Apply normalize_pc to coordinates (default: False, to match train_st_te.py)')
    parser.add_argument("--network", choices=["student", "teacher"], default="student",
                        help="Select ASMAE and its matching OEM")
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
    n_shapes = args.n_shapes if args.n_shapes is not None else config.get('test_size', 20)
    if n_shapes > 0:
        all_files = all_files[-n_shapes:]

    if not all_files:
        print(f"No .obj or .off files found in {data_dir}")
        return

    print(f"\n{'='*60}")
    print(f"Greedy All-Pairs Correspondence Matching with Orientation Alignment")
    print(f"  Dataset: {data_dir}")
    print(f"  Shapes to process: {len(all_files)}")
    print(f"  Output directory: {out_dir}")
    print(f"  Using DINOv2: {args.use_dino}")
    print(f"  Device: {device}")
    print(f"{'='*60}\n")

    # 2. Pre-load Models
    temp_V, _, temp_feat, _ = process_geometry(os.path.join(data_dir, all_files[0]), k=k, t=t, neigvecs=neigvecs, output_dir=out_dir)
    feature_dim = temp_feat.shape[1]
    ckpt_path = args.checkpoint or config.get('correspondence', {}).get('checkpoint_path', 'checkpoints/st_te_model_FAUST.pth')
    model, orient_net = load_model(config, feature_dim, device, checkpoint_path=ckpt_path, network=args.network)

    # Optional DINOv2 setup
    if args.use_dino:
        from utils.dino_utils import get_vertex_dino_features
        from utils.mesh import load_off

    # 3. Load Shape Geometry and Features
    loaded_shapes = {}
    for filename in all_files:
        path = os.path.join(data_dir, filename)
        name = os.path.splitext(filename)[0]

        print(f"  -> Loading: {filename}...")
        V, El, feat, _ = process_geometry(path, k=k, t=t, neigvecs=neigvecs, output_dir=out_dir)
        p_coords = normalize_pc(V.copy()) if args.normalize_pc else V.copy()

        dino_norm = None
        if args.use_dino:
            off_file = os.path.join(args.off_dir, f"{name}.off")
            if os.path.exists(off_file):
                V_raw, _, ITris = load_off(off_file)
                dino_feat, _, _ = get_vertex_dino_features(
                    V_raw, ITris, variant=args.dino_variant, n_views=args.dino_views, device=device)
                dino_norm = dino_feat / (np.linalg.norm(dino_feat, axis=1, keepdims=True) + 1e-8)
            else:
                print(f"    [Warning] OFF file not found at {off_file}, skipping DINOv2 for {name}")

        loaded_shapes[filename] = {
            'V': V,
            'El': El,
            'pos': p_coords,
            'feat': feat,
            'dino': dino_norm,
            'name': name
        }

    # 4. Compute N*N Greedy Correspondences with Pairwise Orientation Alignment
    total_pairs = len(all_files) ** 2
    print(f"\nComputing greedy matches for {total_pairs} pairs...")

    pair_count = 0
    for i, file1 in enumerate(all_files):
        s1 = loaded_shapes[file1]
        p1_tensor = torch.tensor(s1['pos'], dtype=torch.float32, device=device).unsqueeze(0)
        f1_tensor = torch.tensor(s1['feat'], dtype=torch.float32, device=device).unsqueeze(0)
        
        with torch.no_grad():
            z1 = model.extract_features(f1_tensor, p1_tensor).squeeze(0).cpu().numpy()
            z1 = z1 / (np.linalg.norm(z1, axis=1, keepdims=True) + 1e-8)
            if s1['dino'] is not None:
                z1 = np.concatenate([z1, s1['dino']], axis=1)

        for j, file2 in enumerate(all_files):
            s2 = loaded_shapes[file2]
            pair_count += 1

            if pair_count % 50 == 0 or pair_count == total_pairs or pair_count == 1:
                print(f"[{pair_count:4d}/{total_pairs}] Matching {file1} <-> {file2}")

            p2_tensor = torch.tensor(s2['pos'], dtype=torch.float32, device=device).unsqueeze(0)
            f2_tensor = torch.tensor(s2['feat'], dtype=torch.float32, device=device).unsqueeze(0)
            
            with torch.no_grad():
                # Estimate relative orientation and align p2 to p1's coordinate space
                p1_norm = normalize_pc_torch(p1_tensor)
                p2_norm = normalize_pc_torch(p2_tensor)
                orient_out = orient_net(p1_norm, p2_norm)
                p2_aligned = orient_net.align_target(p2_tensor, orient_out)
                
                # Extract ASMAE features on orientation-aligned coordinates
                z2 = model.extract_features(f2_tensor, p2_aligned).squeeze(0).cpu().numpy()
                z2 = z2 / (np.linalg.norm(z2, axis=1, keepdims=True) + 1e-8)
                if s2['dino'] is not None:
                    z2 = np.concatenate([z2, s2['dino']], axis=1)

            # Cosine similarity matching
            sim = z2 @ z1.T
            p2p = np.argmax(sim, axis=1)

            out_name = f"p2p_{s1['name']}_to_{s2['name']}.txt"
            pairs = np.stack([np.arange(len(s2['V'])), p2p], axis=1)
            np.savetxt(os.path.join(out_dir, out_name), pairs, fmt="%d")

    print(f"\nAll {total_pairs} greedy correspondence files saved to: {out_dir}")


if __name__ == "__main__":
    main()
