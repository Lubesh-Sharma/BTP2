import torch
import torch.nn.functional as F

def compute_orientation_loss(z1, z2, n1, n2, tau=0.04):
    """
    True Outward Normal Compatibility Loss for Point Clouds.
    
    Enforces surface orientation preservation between 3D point clouds without mesh faces.
    Directly aligns source outward normal vectors with mapped target outward normal vectors:
      - When orientation is preserved (e.g. chest -> chest, back -> back):
          n_1 . n_mapped -> +1.0  (loss -> 0.0)
      - Under a front-back or reflection symmetry flip (e.g. chest -> back):
          n_1 . n_mapped -> -1.0  (loss -> 2.0)
    
    Args:
        z1: [B, N1, C] latent features for Shape 1
        z2: [B, N2, C] latent features for Shape 2
        n1: [B, N1, 3] outward unit normals for Shape 1
        n2: [B, N2, 3] outward unit normals for Shape 2
        tau: temperature for soft correspondence mapping
    Returns:
        loss_orient: scalar orientation loss strongly penalizing reflection symmetry
    """
    # 1. Soft correspondence matrix P_12 from Shape 1 to Shape 2
    z1_norm = F.normalize(z1, dim=-1)
    z2_norm = F.normalize(z2, dim=-1)
    sim = torch.bmm(z1_norm, z2_norm.transpose(1, 2)) / tau
    P_12 = F.softmax(sim, dim=-1)  # [B, N1, N2]
    
    # 2. Predicted outward normals on Shape 2
    n1_mapped = torch.bmm(P_12, n2)  # [B, N1, 3]
    n1_mapped_norm = F.normalize(n1_mapped, dim=-1, eps=1e-8)
    n1_norm = F.normalize(n1, dim=-1, eps=1e-8)
    
    # 3. Unit normal alignment dot product
    # cos_theta = +1.0 for valid orientation, -1.0 for inverted reflection
    cos_theta = torch.sum(n1_norm * n1_mapped_norm, dim=-1)  # [B, N1]
    loss_orient = (1.0 - cos_theta).mean()
    
    return loss_orient
