import argparse
import os
import torch
import yaml
import numpy as np
import torch.nn.functional as F

from core.preprocessing import process_geometry
from core.trainer import train_asmae
from models.asmae import ASMAE


# ---------------------------------------------------------
# Utils
# ---------------------------------------------------------
def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def compute_reconstruction_mse(model, VPos, Feat, device, mask_ratio):
    """
    Compute reconstruction MSE EXACTLY like training.
    Feature-space MSE, not correspondence error.
    """
    model.eval()
    with torch.no_grad():
        p = torch.tensor(VPos).float().unsqueeze(0).to(device)
        f = torch.tensor(Feat).float().unsqueeze(0).to(device)

        pred_f, _, _, _, _, _ = model(
            f, p,        # source
            f, p,        # target (self-reconstruction)
            mask_ratio=mask_ratio
        )

        mse = ((pred_f - f) ** 2).mean().item()

    model.train()
    return mse


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="ASMAE: Train + Reconstruction MSE Test")
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--save_model", type=str, default="output/model.pth")
    args = parser.parse_args()

    config = load_config(args.config)

    dataset_dir = "/data/home/user/Lubesh_22CS30065/btp2/input/diffusion_knn_k=10/FAUST_Dataset"
    output_dir = config.get("output_dir", "output")
    os.makedirs(output_dir, exist_ok=True)

    device = config.get("device", "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    k = config.get("k", 30)
    t = config.get("t", 10000.0)
    mask_ratio = config.get("mask_ratio", 0.8)

    # -----------------------------------------------------
    # Load dataset
    # -----------------------------------------------------
    shape_files = sorted(
        [f for f in os.listdir(dataset_dir) if f.endswith((".off", ".obj"))]
    )

    train_shapes = shape_files[:5]
    test_shapes = shape_files[5:7]

    print(f"Training shapes: {len(train_shapes)}")
    print(f"Testing shapes : {len(test_shapes)}")

    geometries = {}
    print("\nLoading geometries...")
    for fname in train_shapes + test_shapes:
        path = os.path.join(dataset_dir, fname)
        VPos, El, Feat, _ = process_geometry(path, k, t)
        geometries[fname] = (VPos, El, Feat)
        print(f"  {fname}: N={VPos.shape[0]}, C={Feat.shape[1]}")

    feature_dim = next(iter(geometries.values()))[2].shape[1]

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------
    model = ASMAE(
        feature_dim=feature_dim,
        embed_dim=config.get("embed_dim", 128),
        depth=config.get("depth", 4),
        num_heads=config.get("num_heads", 4),
        decoder_embed_dim=config.get("decoder_embed_dim", 64),
        decoder_depth=config.get("decoder_depth", 2),
        decoder_num_heads=config.get("decoder_num_heads", 4),
        mlk_ratio=config.get("mlp_ratio", 2.0),
    ).to(device)

    # -----------------------------------------------------
    # Training
    # -----------------------------------------------------
    print("\n=== TRAINING PHASE ===")

    train_pairs = [
        (train_shapes[i], train_shapes[j])
        for i in range(len(train_shapes))
        for j in range(i, len(train_shapes))
    ]

    for idx, (s1, s2) in enumerate(train_pairs):
        print(f"\nTraining pair {idx + 1}/{len(train_pairs)}: {s1} <-> {s2}")

        VPos1, _, Feat1 = geometries[s1]
        VPos2, _, Feat2 = geometries[s2]

        model, _, _ = train_asmae(
            model,
            VPos1, Feat1,
            VPos2, Feat2,
            config
        )

    torch.save(model.state_dict(), args.save_model)
    print(f"\nModel saved to {args.save_model}")

    # -----------------------------------------------------
    # Reconstruction MSE Evaluation (LIKE TRAINING)
    # -----------------------------------------------------
    print("\n=== RECONSTRUCTION MSE (TRAINING-STYLE) ===")

    print("\nTraining shapes:")
    for s in train_shapes:
        VPos, _, Feat = geometries[s]
        print(Feat)
        mse = compute_reconstruction_mse(
            model, VPos, Feat, device, mask_ratio
        )
        print(f"  {s}: Recon MSE = {mse:.6f}")

    print("\nTest shapes:")
    test_mses = []
    for s in test_shapes:
        VPos, _, Feat = geometries[s]
        print(Feat)
        mse = compute_reconstruction_mse(
            model, VPos, Feat, device, mask_ratio
        )
        test_mses.append(mse)
        print(f"  {s}: Recon MSE = {mse:.6f}")

    print("\n=== FINAL METRIC ===")
    print(f"Average Test Reconstruction MSE: {np.mean(test_mses):.6f}")


if __name__ == "__main__":
    main()
