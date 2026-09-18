import torch
import torch.nn.functional as F

def get_feature_mask_indices(x, mask_ratio):
    """
    Computes indices of features to mask based on similarity-based clustering.

    Strategy: "Antipodal Similarity Masking" on Geometric Descriptors.
    
    The last 3 feature dimensions are XYZ vertex coordinates (appended
    by preprocessing.py). Positional coordinates act as spatial conditioning
    tokens and are NEVER masked, ensuring the network always knows lateral
    placement (preventing bilateral symmetry flip).

    The entire masking budget is applied to the descriptor channels
    (0 .. C-4, which include HKS and Chirality features) via Antipodal strategy:
      1. Select Random Descriptor Feature A.
      2. Select Descriptor Feature B that is maximally dissimilar to A.
      3. Select clusters of features correlated to A and B respectively.
      4. Union of clusters A and B = final mask indices.

    Args:
        x: [B, N, C] Input features. Last 3 dims are XYZ positions.
        mask_ratio: Float, fraction of descriptor features to mask.

    Returns:
        mask_indices: LongTensor [m] indices of features to zero out (strictly < C-3).
    """
    B, N, C = x.shape
    device = x.device

    desc_C = C - 3  # descriptor dims (e.g. HKS + Chirality)
    if desc_C <= 0:
        return torch.tensor([], dtype=torch.long, device=device)

    m = int(desc_C * mask_ratio)
    if m == 0:
        return torch.tensor([], dtype=torch.long, device=device)

    # ---------------------------------------------------------------
    # Antipodal masking restricted strictly to descriptor dims (0 .. desc_C - 1).
    # Positional coordinates (C-3, C-2, C-1) remain 100% visible at all times.
    # ---------------------------------------------------------------
    features_flat = x[:, :, :desc_C].reshape(-1, desc_C)
    features_norm = F.normalize(features_flat, dim=0)
    sim_matrix = features_norm.t() @ features_norm  # [desc_C, desc_C]

    # Seed A: random descriptor dim
    idx_a = torch.randint(0, desc_C, (1,), device=device).item()

    # Seed B: most dissimilar descriptor dim to A
    idx_b = torch.argmin(sim_matrix[idx_a]).item()

    # Split budget between clusters A and B
    m_b = m // 2
    m_a = m - m_b

    # Clamp k to available dims
    k_a = min(m_a, desc_C)
    k_b = min(m_b, desc_C)

    _, cluster_a_indices = torch.topk(sim_matrix[idx_a], k=k_a)
    _, cluster_b_indices = torch.topk(sim_matrix[idx_b], k=k_b)

    # Combine clusters
    combined_indices = torch.cat([cluster_a_indices, cluster_b_indices])
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
