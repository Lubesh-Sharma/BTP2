import torch

def sinkhorn_pytorch(cost, eps=0.05, n_iter=15):
    """Differentiable Sinkhorn using PyTorch."""
    K = torch.exp(-cost / eps)
    u = torch.ones_like(K[:, :, 0])
    v = torch.ones_like(K[:, 0, :])
    for _ in range(n_iter):
        u = 1.0 / (torch.bmm(K, v.unsqueeze(-1)).squeeze(-1) + 1e-8)
        v = 1.0 / (torch.bmm(K.transpose(1, 2), u.unsqueeze(-1)).squeeze(-1) + 1e-8)
    P = u.unsqueeze(-1) * K * v.unsqueeze(1)
    return P
