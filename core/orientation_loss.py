import torch
import torch.nn.functional as F

def compute_orientation_loss(z1, z2, p1, p2, tau=0.04):
    """
    Global Orientation Preservation Loss (Kabsch / Procrustes Determinant).
    
    Penalizes reflection / mirror / front-back symmetry flips without
    imposing any world-coordinate lock. 
    
    For any valid 3D deformation / rotation between shapes:
      - Proper 3D Rotation (SO(3)): det(R) = +1.0  (Loss = 0.0)
      - Improper Reflection / Flip (O(3) \\ SO(3)): det(R) = -1.0  (Loss = 2.0)
    
    Args:
        z1: [B, N1, C] latent features for Shape 1
        z2: [B, N2, C] latent features for Shape 2
        p1: [B, N1, 3] 3D coordinates for Shape 1
        p2: [B, N2, 3] 3D coordinates for Shape 2
        tau: temperature for soft correspondence mapping
    Returns:
        loss_orient: scalar orientation loss strictly penalizing reflection symmetry
    """
    # 1. Soft correspondence matrix P_12 from Shape 1 to Shape 2
    z1_norm = F.normalize(z1, dim=-1)
    z2_norm = F.normalize(z2, dim=-1)
    sim = torch.bmm(z1_norm, z2_norm.transpose(1, 2)) / tau
    P_12 = F.softmax(sim, dim=-1)  # [B, N1, N2]
    
    # 2. Predicted coordinates on Shape 2
    p1_mapped = torch.bmm(P_12, p2)  # [B, N1, 3]
    
    # 3. Center both point sets
    p1_c = p1 - p1.mean(dim=1, keepdim=True)
    p2_c = p1_mapped - p1_mapped.mean(dim=1, keepdim=True)
    
    # 4. Cross-covariance matrix H: [B, 3, 3]
    H = torch.bmm(p1_c.transpose(1, 2), p2_c)
    
    # 5. SVD to compute optimal rotation R
    U, S, Vh = torch.linalg.svd(H)
    R = torch.bmm(Vh.transpose(1, 2), U.transpose(1, 2))
    det_R = torch.linalg.det(R)
    
    loss_orient = torch.clamp(1.0 - det_R, min=0.0).mean()
    return loss_orient
