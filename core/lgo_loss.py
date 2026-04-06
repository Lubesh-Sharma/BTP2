import torch
from core.sinkhorn import sinkhorn_pytorch

def compute_lgo_loss(z1, z2, eps=0.05, n_iter=15):
    """Global Optimization Loss (L_go) using Sinkhorn pseudo-labels."""
    z1_norm = torch.nn.functional.normalize(z1, dim=-1)
    z2_norm = torch.nn.functional.normalize(z2, dim=-1)
    
    # Cosine similarity matrix S: [B, N, N]
    S = torch.bmm(z1_norm, z2_norm.transpose(1, 2))
    
    # Cost matrix for Sinkhorn: 1 - S. Detach to use as pseudo-labels naturally
    cost = (1.0 - S).detach()
    
    # Generate optimal mapping planner T* using Sinkhorn
    T_star = sinkhorn_pytorch(cost, eps=eps, n_iter=n_iter).detach()
    
    # L_go = CE(T*, S)
    logits = S / eps
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)
    
    # T_star has row probabilities = 1
    loss_go = -torch.sum(T_star * log_probs, dim=2).mean()
    return loss_go
