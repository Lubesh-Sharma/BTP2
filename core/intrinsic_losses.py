"""Losses for single-encoder training (no teacher, no Sinkhorn).

* Biharmonic distortion  - intrinsic replacement of the Euclidean distortion loss.
* Self-identity InfoNCE  - exact self-supervision from two augmented views of one shape.
* DINO distillation      - geometry encoder predicts lifted DINOv2 features,
                           weighted by how well each vertex was observed.
"""
import math
import torch
import torch.nn.functional as F


def compute_biharmonic_distortion_loss(z1, z2, psi1, psi2, num_samples=256, tau=0.05):
    """Pose-invariant metric distortion through a softmax soft map.

    psi*: [B, N, K] biharmonic embeddings (pairwise distance = biharmonic distance,
    normalised so the mean distance is 1 on every shape). For sampled source points
    S, the soft map P (softmax over target vertices) transports psi2; distances
    between transported points must equal the source distances. Blurring a point
    between symmetric candidates (left/right hand) moves its transported embedding
    to the middle and is penalised.
    """
    B, N1, _ = z1.shape
    M = min(num_samples, N1)
    idx = torch.randperm(N1, device=z1.device)[:M]
    z1s = F.normalize(z1[:, idx], dim=-1)
    z2n = F.normalize(z2, dim=-1)
    P = torch.softmax(torch.bmm(z1s, z2n.transpose(1, 2)) / tau, dim=-1)  # [B, M, N2]
    psi_mapped = torch.bmm(P, psi2)                                        # [B, M, K]
    D1 = torch.cdist(psi1[:, idx], psi1[:, idx])
    D2 = torch.cdist(psi_mapped, psi_mapped)
    return F.l1_loss(D2, D1)


def random_rotation_matrices(batch, max_degrees, device):
    axis = F.normalize(torch.randn(batch, 3, device=device), dim=-1)
    angle = (torch.rand(batch, 1, device=device) * 2 - 1) * math.radians(max_degrees)
    K = torch.zeros(batch, 3, 3, device=device)
    K[:, 0, 1], K[:, 0, 2], K[:, 1, 2] = -axis[:, 2], axis[:, 1], -axis[:, 0]
    K = K - K.transpose(1, 2)
    I = torch.eye(3, device=device).expand(batch, 3, 3)
    s, c = angle.sin().unsqueeze(-1), angle.cos().unsqueeze(-1)
    return I + s * K + (1 - c) * torch.bmm(K, K)


def augment_view(pos, keep_ratio, max_rot_deg, scale_range, noise_std):
    """Random vertex subset + small rotation + anisotropic scale + jitter."""
    B, N, _ = pos.shape
    n_keep = max(32, int(round(keep_ratio * N)))
    idx = torch.randperm(N, device=pos.device)[:n_keep].sort().values
    p = pos[:, idx]
    center = p.mean(1, keepdim=True)
    radius = (p - center).norm(dim=-1).amax(1, keepdim=True).unsqueeze(-1)
    R = random_rotation_matrices(B, max_rot_deg, pos.device)
    scale = torch.empty(B, 1, 3, device=pos.device).uniform_(*scale_range)
    p = torch.bmm(p - center, R) * scale + center
    p = p + torch.randn_like(p) * noise_std * radius
    return p, idx


def compute_self_identity_loss(model, feat, pos, keep_ratio=0.8, max_rot_deg=15.0,
                               scale_range=(0.9, 1.1), noise_std=0.003,
                               num_queries=1024, temperature=0.07, z_full=None):
    """InfoNCE between two views of the same shape.

    Every vertex present in both views is its own ground-truth match, so this is
    exact supervision without labels. Candidates are all vertices of view B,
    which also teaches neighbouring vertices to stay distinguishable.
    If `z_full` (embedding of the clean full shape) is given it is used as view A,
    which saves one encoder pass; otherwise both views are augmented.
    """
    pb, ib = augment_view(pos, keep_ratio, max_rot_deg, scale_range, noise_std)
    zb = F.normalize(model.extract_features(feat[:, ib], pb), dim=-1)
    if z_full is not None:
        ia = torch.arange(pos.shape[1], device=pos.device)
        za = F.normalize(z_full, dim=-1)
    else:
        pa, ia = augment_view(pos, keep_ratio, max_rot_deg, scale_range, noise_std)
        za = F.normalize(model.extract_features(feat[:, ia], pa), dim=-1)

    N = pos.shape[1]
    pos_in_b = torch.full((N,), -1, dtype=torch.long, device=pos.device)
    pos_in_b[ib] = torch.arange(len(ib), device=pos.device)
    target = pos_in_b[ia]                               # index in view B or -1
    common = torch.nonzero(target >= 0).squeeze(-1)
    if common.numel() == 0:
        return pos.new_zeros(())
    q = common[torch.randperm(common.numel(), device=pos.device)[:num_queries]]
    logits = za[0, q] @ zb[0].T / temperature           # [Q, N_b]
    return F.cross_entropy(logits, target[q])


def compute_dino_distill_loss(pred, target, conf):
    """Confidence-weighted cosine distance between predicted and lifted DINO features."""
    w = conf.clamp(0, 1)
    cos = (F.normalize(pred, dim=-1) * F.normalize(target, dim=-1)).sum(-1)
    return ((1 - cos) * w).sum() / w.sum().clamp_min(1e-6)
