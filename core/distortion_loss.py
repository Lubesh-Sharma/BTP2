import torch
import torch.nn.functional as F

def compute_distortion_loss(z1, z2, p1, p2, num_samples=256, tau=0.05):
    """
    Pairwise Metric Distortion Loss (L_dist).
    
    Penalizes correspondence predictions that distort pairwise metric distances
    between points on the manifold. Under non-rigid shape deformation, the true
    correspondence approximately preserves local and geodesic distances, while
    bilateral symmetry flips (e.g. left arm mapping to right arm) severely tear
    and distort metric relationships to the rest of the body.
    
    Args:
        z1: [B, N1, D] Clean embeddings for Shape 1.
        z2: [B, N2, D] Clean embeddings for Shape 2.
        p1: [B, N1, 3] Point cloud coordinates for Shape 1.
        p2: [B, N2, 3] Point cloud coordinates for Shape 2.
        num_samples: Number of landmark points to subsample for distance computation (default: 256).
        tau: Softmax temperature for correspondence soft mapping (default: 0.05).
        
    Returns:
        loss_dist: scalar tensor with autograd gradients for z1 and z2.
    """
    B, N1, _ = p1.shape
    _, N2, _ = p2.shape
    
    # 1. Compute soft correspondence matrix P_12: [B, N1, N2]
    z1_norm = F.normalize(z1, dim=-1)
    z2_norm = F.normalize(z2, dim=-1)
    sim = torch.bmm(z1_norm, z2_norm.transpose(1, 2)) / tau
    P_12 = torch.softmax(sim, dim=2)
    
    # 2. Map coordinates of Shape 1 into Shape 2's space
    p1_in_2 = torch.bmm(P_12, p2)  # [B, N1, 3]
    
    # 3. Subsample landmark points for pairwise distance computation
    if num_samples is not None and N1 > num_samples:
        indices = torch.linspace(0, N1 - 1, num_samples, dtype=torch.long, device=p1.device)
        p1_sub = p1[:, indices, :]
        p1_in_2_sub = p1_in_2[:, indices, :]
    else:
        p1_sub = p1
        p1_in_2_sub = p1_in_2
        
    # 4. Compute pairwise Euclidean distance matrices
    D1 = torch.cdist(p1_sub, p1_sub, p=2)
    D1_in_2 = torch.cdist(p1_in_2_sub, p1_in_2_sub, p=2)
    
    # 5. Metric distortion: L1 loss between source distance and mapped distance
    loss_dist = F.l1_loss(D1_in_2, D1)
    return loss_dist

