import torch
import numpy as np
import torch.nn.functional as F

def sinkhorn_algorithm(M_mat, epsilon=0.05, max_iter=50):
    """
    Solves the Optimal Transport problem using the Sinkhorn Algorithm.
    Minimizes Tr(T.M) - epsilon * H(T)
    """
    B, N, M = M_mat.shape
    K = torch.exp(-M_mat / epsilon)
    
    u = torch.ones(B, N, device=M_mat.device) / N
    v = torch.ones(B, M, device=M_mat.device) / M
    
    for _ in range(max_iter):
        Kv = torch.bmm(K, v.unsqueeze(2)).squeeze(2)
        u = (1.0/N) / (Kv + 1e-9)
        KTu = torch.bmm(K.transpose(1, 2), u.unsqueeze(2)).squeeze(2)
        v = (1.0/M) / (KTu + 1e-9)
        
    T = torch.bmm(torch.diag_embed(u), K)
    T = torch.bmm(T, torch.diag_embed(v))
    return T

def compute_nearest_neighbor_match(emb_s, emb_t):
    """
    Computes Nearest Neighbor matches based on Cosine Similarity.
    
    Args:
        emb_s: [B, N, C] or [N, C] Source embeddings.
        emb_t: [B, M, C] or [M, C] Target embeddings.
        
    Returns:
        matches: [N] Array of indices where matches[i] is the index of the point in Target matching point i in Source.
    """
    if emb_s.dim() == 2:
        emb_s = emb_s.unsqueeze(0)
        emb_t = emb_t.unsqueeze(0)
        
    # Normalize
    emb_s_norm = F.normalize(emb_s, p=2, dim=-1)
    emb_t_norm = F.normalize(emb_t, p=2, dim=-1)
    
    # Cosine Similarity: [B, N, M]
    sim_matrix = torch.bmm(emb_s_norm, emb_t_norm.transpose(1, 2))
    
    # Argmax over Target dimension (dim 2)
    # For each source point, find best target point
    matches = torch.argmax(sim_matrix, dim=2) # [B, N]
    
    return matches.squeeze(0).cpu().numpy().astype(np.int32)

def compute_sinkhorn_match(emb_s, emb_t, epsilon=0.05, max_iter=50):
    """
    Computes matches using the Sinkhorn Algorithm (Optimal Transport).
    
    Args:
        emb_s: [B, N, C] Source embeddings.
        emb_t: [B, M, C] Target embeddings.
    
    Returns:
        matches: [N] Array of indices mapping Source -> Target.
    """
    if emb_s.dim() == 2:
        emb_s = emb_s.unsqueeze(0)
        emb_t = emb_t.unsqueeze(0)

    # Normalize
    emb_s_norm = F.normalize(emb_s, p=2, dim=-1)
    emb_t_norm = F.normalize(emb_t, p=2, dim=-1)
    
    # Cost Matrix C = 1 - CosineSimilarity
    sim_matrix = torch.bmm(emb_s_norm, emb_t_norm.transpose(1, 2))
    cost_matrix = 1.0 - sim_matrix
    
    # Compute Transport Plan T
    # Note: sinkhorn_algorithm returns T [B, N, M]
    with torch.no_grad():
        T = sinkhorn_algorithm(cost_matrix, epsilon=epsilon, max_iter=max_iter)
        
    # Argmax over Target dimension to finds hard correspondence
    matches = torch.argmax(T, dim=2) # [B, N]
    
    return matches.squeeze(0).cpu().numpy().astype(np.int32)
