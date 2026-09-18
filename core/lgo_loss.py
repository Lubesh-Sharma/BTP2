import torch
from core.sinkhorn import sinkhorn_pytorch

def compute_lgo_loss(z1, z2, eps=0.04, target_eps=0.025, n_iter=15):
    """
    Global Optimization Loss (L_go) using asymmetric Sinkhorn pseudo-labels.
    
    - target_eps (0.025): sharp temperature for Sinkhorn T*, driving entropy to ~4.0
    - eps (0.04): student logit temperature, ensuring non-vanishing gradients (q - T* != 0)
    """
    z1_norm = torch.nn.functional.normalize(z1, dim=-1)
    z2_norm = torch.nn.functional.normalize(z2, dim=-1)
    
    # Cosine similarity matrix S: [B, N, N]
    S = torch.bmm(z1_norm, z2_norm.transpose(1, 2))
    
    # Cost matrix for Sinkhorn: 1 - S. Detach to use as pseudo-labels naturally
    cost = (1.0 - S).detach()
    
    # Generate sharp optimal mapping planner T* using stabilized Sinkhorn at target_eps
    T_star = sinkhorn_pytorch(cost, eps=target_eps, n_iter=n_iter).detach()
    
    # Student logits with smooth temperature
    logits = S / eps
    log_probs = torch.nn.functional.log_softmax(logits, dim=2)
    
    # L_go = CE(T*, q)
    loss_go = -torch.sum(T_star * log_probs, dim=2).mean()
    return loss_go
