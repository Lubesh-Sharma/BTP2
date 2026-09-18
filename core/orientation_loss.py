import torch
import torch.nn.functional as F

def compute_orientation_loss(z1, z2, p1, p2, num_samples=1000, tau=0.04, k_spatial=3):
    """
    Signed Area / Orientation Consistency Loss (Method B from Complex Functional Maps).
    
    Enforces local orientation preservation on pure 3D point clouds without mesh faces.
    Under a bilateral symmetry flip (reflection), the local normal cross product flips sign:
      n_1 . n_hat < 0 (reflection / symmetry reversal)
      n_1 . n_hat > 0 (orientation-preserving isometry)
    
    Args:
        z1: [B, N1, C] student features for Shape 1
        z2: [B, N2, C] student features for Shape 2
        p1: [B, N1, 3] 3D coordinates for Shape 1
        p2: [B, N2, 3] 3D coordinates for Shape 2
        num_samples: number of triplets sampled for efficiency
        tau: temperature for soft correspondence mapping
        k_spatial: number of nearest spatial neighbors (default 3: self + 2 neighbors)
    Returns:
        loss_orient: scalar orientation loss penalizing reflection symmetry
    """
    B, N1, _ = p1.shape
    device = p1.device
    
    # 1. Soft correspondence matrix P_12 from Shape 1 to Shape 2
    z1_norm = F.normalize(z1, dim=-1)
    z2_norm = F.normalize(z2, dim=-1)
    sim = torch.bmm(z1_norm, z2_norm.transpose(1, 2)) / tau
    P_12 = F.softmax(sim, dim=-1)  # [B, N1, N2]
    
    # 2. Predicted coordinates on Shape 2
    p1_mapped = torch.bmm(P_12, p2)  # [B, N1, 3]
    
    # 3. Sample a subset of anchor vertices for efficiency
    M = min(num_samples, N1)
    sample_idx = torch.randperm(N1, device=device)[:M]
    p1_anchors = p1[:, sample_idx, :]  # [B, M, 3]
    
    # 4. Find the 2 nearest spatial neighbors in p1 for each sampled anchor
    dists_sq = (torch.sum(p1_anchors ** 2, dim=-1, keepdim=True) +
                torch.sum(p1 ** 2, dim=-1, keepdim=True).transpose(1, 2) -
                2.0 * torch.bmm(p1_anchors, p1.transpose(1, 2)))
    
    _, knn_idx = torch.topk(dists_sq, k=k_spatial, dim=-1, largest=False)  # [B, M, 3]
    
    idx_i = knn_idx[:, :, 0]  # self: [B, M]
    idx_j = knn_idx[:, :, 1]  # neighbor 1: [B, M]
    idx_k = knn_idx[:, :, 2]  # neighbor 2: [B, M]
    
    # Helper to gather coordinates for batch
    def gather_pts(pts, idx):
        B_size, M_size = idx.shape
        idx_expanded = idx.unsqueeze(-1).expand(B_size, M_size, 3)
        return torch.gather(pts, 1, idx_expanded)
    
    # 5. Extract coordinates of triplets on Shape 1
    p1_i = gather_pts(p1, idx_i)
    p1_j = gather_pts(p1, idx_j)
    p1_k = gather_pts(p1, idx_k)
    
    # Signed normal / area vector on Shape 1: (p_j - p_i) x (p_k - p_i)
    v1_j = p1_j - p1_i
    v1_k = p1_k - p1_i
    n1 = torch.cross(v1_j, v1_k, dim=-1)
    n1_norm = F.normalize(n1, dim=-1, eps=1e-8)
    
    # 6. Extract mapped coordinates of the same triplets on Shape 2
    p1_m_i = gather_pts(p1_mapped, idx_i)
    p1_m_j = gather_pts(p1_mapped, idx_j)
    p1_m_k = gather_pts(p1_mapped, idx_k)
    
    v_m_j = p1_m_j - p1_m_i
    v_m_k = p1_m_k - p1_m_i
    n_mapped = torch.cross(v_m_j, v_m_k, dim=-1)
    n_mapped_norm = F.normalize(n_mapped, dim=-1, eps=1e-8)
    
    # 7. Orientation alignment: dot product of unit normals
    # Orientation-preserving: cos_theta -> 1.0 (loss -> 0.0)
    # Symmetry reversal (reflection): cos_theta -> -1.0 (loss -> 2.0)
    cos_theta = torch.sum(n1_norm * n_mapped_norm, dim=-1)  # [B, M]
    loss_orient = (1.0 - cos_theta).mean()
    
    return loss_orient

