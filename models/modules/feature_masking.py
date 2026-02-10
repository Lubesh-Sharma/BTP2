import torch
import torch.nn.functional as F

def get_feature_mask_indices(x, mask_ratio):
    """
    Computes indices of features to mask based on similarity-based clustering.
    
    Strategy: "Antipodal Similarity Masking"
    1. Select Random Feature A.
    2. Select Feature B that is maximally dissimilar to A.
    3. Select cluster of features highly correlated to A.
    4. Select cluster of features highly correlated to B.
    5. Mask both clusters.
    
    Args:
        x: [B, N, C] Input features.
        mask_ratio: Float, fraction of C features to mask.
        
    Returns:
        mask_indices: LongTensor [m] indices of features to zero out.
    """
    B, N, C = x.shape
    device = x.device
    
    m = int(C * mask_ratio)

    if m == 0: return torch.tensor([], dtype=torch.long, device=device)
    
    # 1. Compute Global Feature Correlation Matrix [C, C]
    # Reshape to [B*N, C] to treat all points as samples
    features_flat = x.reshape(-1, C) 
    
    # Normalize columns to compute cosine similarity
    # We want similarity between *Columns* (Features)
    # Transpose first? No, features_flat is [Samples, Features]
    # We norm along dim 0.
    features_norm = F.normalize(features_flat, dim=0)
    
    # Similarity: [C, C] = F.T @ F
    sim_matrix = features_norm.t() @ features_norm
    
    # 2. Select Seed A (Random)
    idx_a = torch.randint(0, C, (1,), device=device).item()
    
    # 3. Select Seed B (Most dissimilar to A)
    # We look for min value in sim_matrix[idx_a]
    # argmin gives index of feature with lowest correlation
    idx_b = torch.argmin(sim_matrix[idx_a]).item()
    
    # 4. Define Cluster Sizes
    m_b = m // 2
    m_a = m - m_b
    
    # 5. Get Cluster A (Most similar to A)
    # topk returns values, indices. We want indices.
    # We include A itself (it has sim 1.0 with itself)
    _, cluster_a_indices = torch.topk(sim_matrix[idx_a], k=m_a)
    
    # 6. Get Cluster B (Most similar to B)
    _, cluster_b_indices = torch.topk(sim_matrix[idx_b], k=m_b)
    
    # 7. Combine
    combined_indices = torch.cat([cluster_a_indices, cluster_b_indices])
    
    # Unique/Distinct check? 
    # If A and B are somehow close (unlikely if B is argmin), clusters might overlap.
    # unique() ensures we handle overlap, though m might slightly decrease.
    # Given B is argmin, overlap is minimal/impossible for HKS.
    combined_indices = torch.unique(combined_indices)
    
    return combined_indices

def apply_feature_mask(x, feature_indices, point_mask):
    """
    Applies the feature mask to the selected points.
    
    Args:
        x: [B, N, C] Input features.
        feature_indices: [m] Indices of feature dimensions to zero.
        point_mask: [B, N] Binary mask (1=Mask this point).
    """
    if feature_indices.numel() == 0:
        return x.clone(), torch.zeros_like(x, dtype=torch.bool)
        
    x_masked = x.clone()
    
    # Logic: x_masked[b, n, f] = 0 IF point_mask[b, n] == 1 AND f in feature_indices
    # Construct a mask tensor efficiently
    
    B, N, C = x.shape
    device = x.device
    
    # Create base boolean mask for features [C]
    feat_mask_vec = torch.zeros(C, dtype=torch.bool, device=device)
    feat_mask_vec[feature_indices] = True
    
    # Expand to [B, N, C]
    # mask_bool: [B, N] -> [B, N, 1]
    point_mask_bool = (point_mask > 0).unsqueeze(-1) 
    
    # expanded_feat_mask: [1, 1, C] -> [B, N, C]
    feat_mask_exp = feat_mask_vec.view(1, 1, C).expand(B, N, C)
    
    # Final mask: AND condition
    final_mask = point_mask_bool & feat_mask_exp
    
    x_masked[final_mask] = 0.0
    
    return x_masked, final_mask
