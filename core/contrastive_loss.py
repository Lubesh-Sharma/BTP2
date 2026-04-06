import torch

def compute_contrastive_loss(z, margin=0.5):
    """Multi-target contrastive loss via InfoNCE-style penalty on off-diagonals."""
    z_norm = torch.nn.functional.normalize(z, dim=-1)
    sim = torch.bmm(z_norm, z_norm.transpose(1, 2)).squeeze(0)
    N = sim.shape[0]
    mask = ~torch.eye(N, dtype=torch.bool, device=sim.device)
    return torch.clamp(sim[mask] - margin, min=0).mean()
