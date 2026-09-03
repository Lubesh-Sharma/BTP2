import torch
import torch.nn.functional as F

def get_feature_mask_indices(x, mask_ratio):
    """
    Computes indices of features to mask based on similarity-based clustering.

    Strategy: "Antipodal Similarity Masking" + Forced Positional Masking
    
    Assumes the last 3 feature dimensions are XYZ vertex positions (appended
    by preprocessing.py). The strategy is:
    1. ALWAYS mask one random positional dim (x, y, or z) so the model learns
       to predict spatial placement from HKS context.
    2. Use the remaining masking budget on HKS dims via Antipodal strategy:
       - Select Random HKS Feature A.
       - Select HKS Feature B that is maximally dissimilar to A.
       - Select clusters of HKS features correlated to A and B respectively.
    3. Union of HKS cluster + forced positional dim = final mask.

    Args:
        x: [B, N, C] Input features. Last 3 dims are XYZ positions.
        mask_ratio: Float, fraction of C features to mask.

    Returns:
        mask_indices: LongTensor [m] indices of features to zero out.
    """
    B, N, C = x.shape
    device = x.device

    m = int(C * mask_ratio)

    if m == 0:
        return torch.tensor([], dtype=torch.long, device=device)

    # ---------------------------------------------------------------
    # STEP 1: Always force-mask one positional dimension (x, y, or z).
    # The last 3 feature dims (indices C-3, C-2, C-1) are XYZ.
    # ---------------------------------------------------------------
    forced_pos_dim = torch.randint(C - 3, C, (1,), device=device)  # one of {C-3, C-2, C-1}

    # Reserve 1 slot for the forced positional dim; rest goes to HKS masking
    m_hks = m - 1
    if m_hks <= 0:
        # Edge case: mask_ratio so small only 1 dim fits → just mask positional
        return forced_pos_dim

    # ---------------------------------------------------------------
    # STEP 2: Antipodal masking restricted to HKS dims (0 .. C-4).
    # ---------------------------------------------------------------
    hks_C = C - 3  # number of pure HKS dimensions

    # Correlation matrix over HKS dims only [hks_C, hks_C]
    features_flat = x[:, :, :hks_C].reshape(-1, hks_C)
    features_norm = F.normalize(features_flat, dim=0)
    sim_matrix = features_norm.t() @ features_norm  # [hks_C, hks_C]

    # Seed A: random HKS dim
    idx_a = torch.randint(0, hks_C, (1,), device=device).item()

    # Seed B: most dissimilar HKS dim to A
    idx_b = torch.argmin(sim_matrix[idx_a]).item()

    # Split remaining budget between clusters A and B
    m_b = m_hks // 2
    m_a = m_hks - m_b

    # Clamp k to available dims (safety for small hks_C)
    k_a = min(m_a, hks_C)
    k_b = min(m_b, hks_C)

    _, cluster_a_indices = torch.topk(sim_matrix[idx_a], k=k_a)
    _, cluster_b_indices = torch.topk(sim_matrix[idx_b], k=k_b)

    # ---------------------------------------------------------------
    # STEP 3: Combine HKS cluster indices + forced positional dim
    # ---------------------------------------------------------------
    combined_indices = torch.cat([cluster_a_indices, cluster_b_indices, forced_pos_dim])
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
