import torch

def compute_cycle_loss(z1, z2, p1, eps=0.05, n_iter=15):
    """Cycle consistency mapping coordinates 1 -> 2 -> 1 using Softmax (Differentiable & Non-Orthogonal)."""
    z1_norm = torch.nn.functional.normalize(z1, dim=-1)
    z2_norm = torch.nn.functional.normalize(z2, dim=-1)
    
    # Calculate similarity with temperature scaling 
    sim_matrix = torch.bmm(z1_norm, z2_norm.transpose(1, 2)) / eps
    
    # Use standard Softmax instead of Sinkhorn for training gradients
    P_12 = torch.softmax(sim_matrix, dim=2)
    P_21 = torch.softmax(sim_matrix.transpose(1, 2), dim=2)
    
    # Cycle mapping: P_12 maps 1 to 2, P_21 maps 2 to 1
    cycle_p1 = torch.bmm(P_12, torch.bmm(P_21, p1))
    return torch.nn.functional.mse_loss(cycle_p1, p1)
