import torch

def sinkhorn_pytorch(cost, eps=0.025, n_iter=15):
    """
    Differentiable Sinkhorn using PyTorch with numerical stabilization.
    Subtracts row-wise minimum cost to center maximum exponent at 0 (exp(0) = 1.0),
    preventing float32 underflow/NaNs when eps is small (<= 0.03).
    """
    cost_min = torch.min(cost, dim=-1, keepdim=True)[0]
    cost_stabilized = cost - cost_min
    K = torch.exp(-cost_stabilized / eps)
    
    u = torch.ones_like(K[:, :, 0])
    v = torch.ones_like(K[:, 0, :])
    for _ in range(n_iter):
        u = 1.0 / (torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + 1e-8)
        v = 1.0 / (torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + 1e-8)
    P = u.unsqueeze(-1) * K * v.unsqueeze(1)
    # Ensure exact row stochasticity (sum to 1.0)
    P = P / (torch.sum(P, dim=-1, keepdim=True) + 1e-8)
    return P
