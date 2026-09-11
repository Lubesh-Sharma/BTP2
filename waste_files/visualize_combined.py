"""
visualize_combined.py

Visualizes the combined ASMAE + DINOv2 per-vertex features using PCA-RGB coloring.

Pipeline for each pair:
  1. Load mesh, compute HKS+XYZ features (via process_geometry).
  2. Run ASMAE encoder → per-vertex embeddings [N, embed_dim].
  3. Run DINOv2 on a rendered image → per-vertex embeddings [N, D].
  4. L2-normalize and concatenate both: [N, embed_dim + D].
  5. Fit PCA (3 components) on BOTH shapes jointly.
  6. Map PC1→R, PC2→G, PC3→B (normalized to [0,1]).
  7. Save colored .obj files → open in MeshLab to verify correspondence.

Usage:
    python visualize_combined.py --config config/FAUST/train_st_te_config.yaml \\
                                  --checkpoint checkpoints/st_te_model_FAUST.pth \\
                                  --n_pairs 3 \\
                                  --out_dir outputs/dino_viz
    
    # For headless (SSH) servers, prefix with:
    PYOPENGL_PLATFORM=osmesa python visualize_combined.py ...
"""

import os
import argparse
import numpy as np
import torch
import yaml
from sklearn.decomposition import PCA

from core.preprocessing import process_geometry, normalize_pc
from models.asmae import ASMAE
from utils.mesh import load_obj, load_off, save_obj


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def load_asmae_model(config, feature_dim, device):
    """Load trained ASMAE student model from checkpoint."""
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
        temperature=model_cfg.get('temperature', 1.0),
    ).to(device)
    return model


def get_asmae_features(model, features_np, vpos_np, device):
    """
    Extracts ASMAE encoder features for a single shape.

    Args:
        model:       ASMAE in eval mode.
        features_np: np.ndarray [N, C] (HKS + XYZ).
        vpos_np:     np.ndarray [N, 3] vertex positions.
        device:      torch.device.

    Returns:
        np.ndarray [N, embed_dim]
    """
    f = torch.tensor(features_np, dtype=torch.float32).unsqueeze(0).to(device)
    p = torch.tensor(vpos_np.copy(), dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        enc = model.extract_features(f, p)   # [1, N, embed_dim]
    return enc.squeeze(0).cpu().numpy()      # [N, embed_dim]


def pca_rgb_colors(feats_list):
    """
    Fits PCA on all shapes together, returns per-shape RGB arrays.

    Args:
        feats_list: list of np.ndarray, each [N_i, D].

    Returns:
        colors_list: list of np.ndarray [N_i, 3] in [0, 1].
    """
    all_feats = np.concatenate(feats_list, axis=0)        # [sum(N), D]
    pca = PCA(n_components=3)
    all_proj = pca.fit_transform(all_feats)               # [sum(N), 3]

    # Normalize each PC to [0, 1] globally
    for i in range(3):
        col = all_proj[:, i]
        col_min, col_max = col.min(), col.max()
        all_proj[:, i] = (col - col_min) / (col_max - col_min + 1e-8)

    # Split back per shape
    colors_list = []
    start = 0
    for feats in feats_list:
        end = start + len(feats)
        colors_list.append(all_proj[start:end])
        start = end
    return colors_list


def load_faces(path):
    """Load face/triangle indices from .obj or .off file."""
    if path.endswith('.obj'):
        faces = []
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('f '):
                    parts = line.strip().split()[1:]
                    # Handle f v, f v/vt, f v/vt/vn formats
                    indices = [int(p.split('/')[0]) - 1 for p in parts]
                    faces.append(indices)
        return np.array(faces, dtype=np.int32) if faces else np.zeros((0, 3), dtype=np.int32)
    else:
        from utils.mesh import load_off
        _, _, itris = load_off(path)
        return itris


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ASMAE + DINOv2 Visualization")
    parser.add_argument('--config',     type=str, default='config/FAUST/train_st_te_config.yaml')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to .pth checkpoint. Defaults to config checkpoint.')
    parser.add_argument('--n_pairs',    type=int, default=3,
                        help='Number of shape pairs to visualize.')
    parser.add_argument('--out_dir',    type=str, default='outputs/dino_viz')
    parser.add_argument('--dino_variant', type=str, default='dinov2_vitb14_reg',
                        help='DINOv2 variant: dinov2_vitb14_reg, dinov2_vitl14_reg, etc.')
    parser.add_argument('--mesh_dir',  type=str, default=None,
                        help='Directory with .off files for proper triangle mesh rendering. '
                             'If not set, falls back to data_dir (may render blank if edge-based .obj).')
    parser.add_argument('--asmae_only', action='store_true',
                        help='Visualize ASMAE features only (skip DINOv2). Useful for comparison.')
    parser.add_argument('--dino_only',  action='store_true',
                        help='Visualize DINOv2 features only (skip ASMAE). Useful for comparison.')
    args = parser.parse_args()

    # -----------------------------------------------------------------------
    # Setup
    # -----------------------------------------------------------------------
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    data_dir  = config['data_dir']
    output_dir = config['output_dir']
    k  = config['geometry']['k']
    t  = config['geometry']['t']
    nv = config['geometry'].get('neigvecs', 300)

    all_files = sorted([f for f in os.listdir(data_dir)
                        if f.endswith('.obj') or f.endswith('.off')])
    # Use last N files (test set convention)
    test_files = all_files[-min(args.n_pairs * 2, len(all_files)):]

    # -----------------------------------------------------------------------
    # Load ASMAE model
    # -----------------------------------------------------------------------
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        checkpoint_path = os.path.join(
            config['training']['checkpoint_dir'],
            config['training']['checkpoint_name'])

    print(f"\nLoading ASMAE checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    feature_dim = checkpoint['feature_dim']
    print(f"  Checkpoint feature_dim = {feature_dim}  "
          f"(expected {k + 3} = {k} HKS + 3 XYZ)")

    asmae_model = load_asmae_model(config, feature_dim, device)
    asmae_model.load_state_dict(checkpoint['model_state_dict'])
    asmae_model.eval()
    print("  ASMAE model loaded.\n")

    # -----------------------------------------------------------------------
    # DINOv2 model (import only if needed)
    # -----------------------------------------------------------------------
    if not args.asmae_only:
        from utils.dino_utils import get_vertex_dino_features
        print(f"DINOv2 variant: {args.dino_variant}")
        # Trigger model download / load now
        from utils.dino_utils import get_dino_model
        get_dino_model(args.dino_variant, device)

    # -----------------------------------------------------------------------
    # Process pairs
    # -----------------------------------------------------------------------
    pair_idx = 0
    for i in range(0, len(test_files) - 1, 2):
        if pair_idx >= args.n_pairs:
            break

        f1_name = test_files[i]
        f2_name = test_files[i + 1]
        f1_path = os.path.join(data_dir, f1_name)
        f2_path = os.path.join(data_dir, f2_name)

        print(f"\n{'='*60}")
        print(f"Pair {pair_idx+1}: {f1_name}  ↔  {f2_name}")
        print(f"{'='*60}")

        # ---- Preprocess (HKS + XYZ) ----
        print("  [1/4] Preprocessing geometry...")
        V1, El1, feat1, _ = process_geometry(f1_path, k, t, nv, output_dir)
        V2, El2, feat2, _ = process_geometry(f2_path, k, t, nv, output_dir)

        V1_norm = normalize_pc(V1.copy())
        V2_norm = normalize_pc(V2.copy())

        # Load triangle faces for rendering.
        # Use --mesh_dir (.off files) if provided — these have proper triangle connectivity.
        # The k_10 .obj files are edge-based and render as blank white images.
        if args.mesh_dir:
            off1 = os.path.join(args.mesh_dir, os.path.splitext(f1_name)[0] + '.off')
            off2 = os.path.join(args.mesh_dir, os.path.splitext(f2_name)[0] + '.off')
            ITris1 = load_faces(off1) if os.path.exists(off1) else load_faces(f1_path)
            ITris2 = load_faces(off2) if os.path.exists(off2) else load_faces(f2_path)
            # Use raw (non-normalized) positions from the .off file for rendering
            V1_render, _, _ = load_off(off1) if os.path.exists(off1) else (V1, None, None)
            V2_render, _, _ = load_off(off2) if os.path.exists(off2) else (V2, None, None)
        else:
            ITris1 = load_faces(f1_path)
            ITris2 = load_faces(f2_path)
            V1_render, V2_render = V1, V2
        
        if len(ITris1) == 0:
            print("  ⚠ WARNING: No triangle faces found for shape 1. DINOv2 will render a blank image.")
            print("    → Pass --mesh_dir pointing to your .off files to fix this.")
        if len(ITris2) == 0:
            print("  ⚠ WARNING: No triangle faces found for shape 2.")

        # ---- ASMAE features ----
        print("  [2/4] Extracting ASMAE encoder features...")
        asmae_feat1 = get_asmae_features(asmae_model, feat1, V1_norm, device)
        asmae_feat2 = get_asmae_features(asmae_model, feat2, V2_norm, device)
        # L2 normalize
        asmae_feat1 /= np.linalg.norm(asmae_feat1, axis=1, keepdims=True) + 1e-8
        asmae_feat2 /= np.linalg.norm(asmae_feat2, axis=1, keepdims=True) + 1e-8

        pair_dir = os.path.join(args.out_dir,
                                f"{os.path.splitext(f1_name)[0]}_and_"
                                f"{os.path.splitext(f2_name)[0]}")
        os.makedirs(pair_dir, exist_ok=True)

        # ---- DINOv2 features ----
        if not args.asmae_only:
            print("  [3/4] Extracting DINOv2 features (render → project → assign)...")
            dino_feat1, vis1, img1 = get_vertex_dino_features(
                V1_render, ITris1, variant=args.dino_variant, device=device)
            dino_feat2, vis2, img2 = get_vertex_dino_features(
                V2_render, ITris2, variant=args.dino_variant, device=device)

            # Save rendered images for inspection
            from PIL import Image as PILImage
            PILImage.fromarray(img1).save(os.path.join(pair_dir, f"render_{os.path.splitext(f1_name)[0]}.png"))
            PILImage.fromarray(img2).save(os.path.join(pair_dir, f"render_{os.path.splitext(f2_name)[0]}.png"))
            print(f"    Shape 1: {vis1.sum()}/{len(V1)} vertices visible")
            print(f"    Shape 2: {vis2.sum()}/{len(V2)} vertices visible")

        # ---- Build combined features ----
        print("  [4/4] Building combined features and computing PCA colors...")

        if args.asmae_only:
            combined1, combined2 = asmae_feat1, asmae_feat2
            mode = "asmae_only"
        elif args.dino_only:
            combined1, combined2 = dino_feat1, dino_feat2
            mode = "dino_only"
        else:
            combined1 = np.concatenate([asmae_feat1, dino_feat1], axis=1)
            combined2 = np.concatenate([asmae_feat2, dino_feat2], axis=1)
            mode = "asmae_plus_dino"

        # ---- PCA → RGB colors (fit on both shapes jointly) ----
        colors1, colors2 = pca_rgb_colors([combined1, combined2])

        # ---- Save colored .obj files ----
        name1 = os.path.splitext(f1_name)[0]
        name2 = os.path.splitext(f2_name)[0]

        path_out1 = os.path.join(pair_dir, f"{name1}_{mode}_pca.obj")
        path_out2 = os.path.join(pair_dir, f"{name2}_{mode}_pca.obj")

        save_obj(path_out1, V1, colors1, ITris1)
        save_obj(path_out2, V2, colors2, ITris2)

        print(f"\n  ✓ Saved colored meshes to: {pair_dir}/")
        print(f"    {os.path.basename(path_out1)}")
        print(f"    {os.path.basename(path_out2)}")
        print(f"\n  ► Open both .obj files in MeshLab side-by-side.")
        print(f"    Matching body parts should have the same color if")
        print(f"    the {mode} features encode good correspondence.")

        pair_idx += 1

    print(f"\n{'='*60}")
    print(f"Done! Visualizations saved to: {args.out_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
