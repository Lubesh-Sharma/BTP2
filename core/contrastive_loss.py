import torch

def compute_contrastive_loss(z, margin=0.5):
    """Multi-target contrastive loss via penalty on off-diagonals."""
    z_norm = torch.nn.functional.normalize(z, dim=-1)
    sim = torch.bmm(z_norm, z_norm.transpose(1, 2))  # [B, N, N]
    B, N, _ = sim.shape
    eye = torch.eye(N, dtype=torch.bool, device=sim.device).unsqueeze(0).expand(B, -1, -1)
    return torch.clamp(sim[~eye] - margin, min=0).mean()
