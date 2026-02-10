import os
import torch
import numpy as np
from scipy.spatial.distance import cdist

from models.asmae import ASMAE
from core.preprocessing import process_geometry, normalize_pc


# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DEVICE = "cuda"
FEATURE_DIM = 30
EPS = 0.05
N_ITER = 100

CHECKPOINT = "/data/home/user/Lubesh_22CS30065/btp2/checkpoints/trained_model.pth"

SHAPE1 = "/data/home/user/Lubesh_22CS30065/btp2/input/diffusion_knn_k=5/FAUST_Dataset/tr_reg_071.obj"
SHAPE2 = "/data/home/user/Lubesh_22CS30065/btp2/input/diffusion_knn_k=5/FAUST_Dataset/tr_reg_075.obj"

OUT_DIR = "p2p_results"
os.makedirs(OUT_DIR, exist_ok=True)


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
def load_model():
    model = ASMAE(
        feature_dim=FEATURE_DIM,
        embed_dim=128,
        depth=4,
        num_heads=4,
        decoder_embed_dim=64,
        decoder_depth=2,
        decoder_num_heads=4,
        mlk_ratio=2.0
    ).to(DEVICE)  # Fixed: use DEVICE constant

    checkpoint = torch.load(CHECKPOINT, map_location=DEVICE)  # Fixed: use constants
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


# -------------------------------------------------
# MAIN
# -------------------------------------------------
print("Loading shapes and geometry...")

V1, _, feat1, _ = process_geometry(SHAPE1, k=30, t=8)
V2, _, feat2, _ = process_geometry(SHAPE2, k=30, t=8)

V1 = normalize_pc(V1)
V2 = normalize_pc(V2)

N1, N2 = V1.shape[0], V2.shape[0]

print("Loading ASMAE model...")
model = load_model()

print("Extracting ASMAE features...")
with torch.no_grad():
    f1 = torch.tensor(feat1, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    f2 = torch.tensor(feat2, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    p1 = torch.tensor(V1, dtype=torch.float32, device=DEVICE).unsqueeze(0)
    p2 = torch.tensor(V2, dtype=torch.float32, device=DEVICE).unsqueeze(0)

    z1 = model.extract_features(f1, p1).squeeze(0).cpu().numpy()
    z2 = model.extract_features(f2, p2).squeeze(0).cpu().numpy()

# ℓ2-normalize (IMPORTANT)
z1 /= np.linalg.norm(z1, axis=1, keepdims=True) + 1e-8
z2 /= np.linalg.norm(z2, axis=1, keepdims=True) + 1e-8


print("Computing feature cost matrix...")
cost = cdist(z1, z2, metric="sqeuclidean")

print("Running Sinkhorn...")
P = sinkhorn(cost, eps=EPS, n_iter=N_ITER)

print("Extracting one-direction P2P (shape2 → shape1)...")
p2p = np.argmax(P.T, axis=1)  # shape2 → shape1

# -------------------------------------------------
# SAVE (two-column format)
# -------------------------------------------------
out_path = os.path.join(
    OUT_DIR,
    "p2p_tr_reg_071_to_tr_reg_075_sinkhorn.txt"
)

pairs = np.stack([np.arange(N2), p2p], axis=1)
np.savetxt(out_path, pairs, fmt="%d")

print("Saved P2P map to:", out_path)
print(f"Mapped points: {pairs.shape[0]} / {N2}")