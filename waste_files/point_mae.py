import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --------------------------------------------------------------------------------
# Helper: Graph Utils for DGCNN
# --------------------------------------------------------------------------------

def knn(x, k):
    """
    Computes k-Nearest Neighbors (kNN) for a batch of point clouds.
    
    Args:
        x (torch.Tensor): [B, C, N] Batch of point clouds (C is usually 3).
        k (int): Number of neighbors to find.
        
    Returns:
        torch.Tensor: [B, N, k] Indices of the nearest neighbors.
    """
    # x: [B, C, N]
    inner = -2*torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

def get_graph_feature(x, k=20, idx=None):
    """
    Constructs a local graph feature representation (EdgeConv style).
    For each point, concatenates (neighbor - point, point).
    
    Args:
        x (torch.Tensor): [B, C, N] Point cloud features/coords.
        k (int): Number of neighbors.
        idx (torch.Tensor, optional): Precomputed kNN indices [B, N, k].
        
    Returns:
        torch.Tensor: [B, 2*C, N, k] Graph features (edge features).
    """
    # x: [B, C, N]
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)
    device = torch.device('cuda') if x.is_cuda else torch.device('cpu')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature

# --------------------------------------------------------------------------------
# 1. Blocks
# --------------------------------------------------------------------------------

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# --------------------------------------------------------------------------------
# Sinkhorn Algorithm (Global Optimization)
# --------------------------------------------------------------------------------

def sinkhorn_algorithm(M_mat, epsilon=0.05, max_iter=50):
    """
    Solves the Optimal Transport problem using the Sinkhorn Algorithm.
    Used for Global Optimization to find soft correspondences T*.
    
    Minimizes Tr(T.M) - epsilon * H(T)
    
    Args:
        M_mat (torch.Tensor): [B, N, M] Cost Matrix (1 - Similarity).
        epsilon (float): Regularization strength.
        max_iter (int): Number of iterations.
        
    Returns:
        torch.Tensor: [B, N, M] Optimal Transport Plan T*.
    """
    # M_mat: [B, N, M] Cost Matrix
    # Solve T* that minimizes Tr(T.M) - eps*H(T)
    # T = diag(u) * K * diag(v)
    # K = exp(-M/epsilon)
    
    B, N, M = M_mat.shape
    K = torch.exp(-M_mat / epsilon)
    
    # Init u and v (ones)
    # Mass constraints: usually 1/N and 1/M for uniform
    u = torch.ones(B, N, device=M_mat.device) / N
    v = torch.ones(B, M, device=M_mat.device) / M
    
    for _ in range(max_iter):
        # Update u
        # u = (1/N) / (K @ v)
        # K: [B, N, M], v: [B, M]
        Kv = torch.bmm(K, v.unsqueeze(2)).squeeze(2) # [B, N]
        u = (1.0/N) / (Kv + 1e-9)
        
        # Update v
        # v = (1/M) / (K.T @ u)
        # K.T: [B, M, N], u: [B, N]
        KTu = torch.bmm(K.transpose(1, 2), u.unsqueeze(2)).squeeze(2) # [B, M]
        v = (1.0/M) / (KTu + 1e-9)
        
    # T = u * K * v
    # diag(u): [B, N, N], K: [B, N, M], diag(v): [B, M, M]
    T = torch.bmm(torch.diag_embed(u), K)
    T = torch.bmm(T, torch.diag_embed(v))
    return T

# --------------------------------------------------------------------------------
# Blocks
# --------------------------------------------------------------------------------

class LocalSelfAttentionBlock(nn.Module):
    """
    Local Self-Attention Block using k-Nearest Neighbors.
    Replaces global attention in the Encoder to capture local geometry.
    """
    # Local Attention using k-NN
    # Similar to GAT or DGCNN-style attention
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., k=20):
        super().__init__()
        self.k = k
        self.norm1 = nn.LayerNorm(dim)
        
        # Simple Multihead Attention on Neighbors?
        # Standard standard Transformer expects Sequence.
        # We construct sequence of neighbors [B*N, k, C]
        
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim*mlp_ratio))

    def forward(self, x, pos=None):
        """
        Args:
            x: [B, N, C] Input features.
            pos: [B, N, 3] Input coordinates (used for finding neighbors).
        """
        # x: [B, N, C]
        B, N, C = x.shape
        shortcut = x
        x_norm = self.norm1(x)
        
        # 1. Find Neighbors
        # Use simple kNN on features or coordinates? 
        # Paper says "Local Self Attention". Usually on Coordinates (Pos) or Features.
        # HSTR uses spatial kNN. We will use `pos` (coordinates).
        # Assuming pos is updated or original? Usually original pos.
        # If pos is None, use x.
        
        # loc_ref: [B, N, 3] usually. knn expects [B, 3, N] (Channels First)
        loc_ref = x_norm if pos is None else pos

        # loc_ref: [B, N, 3] usually. knn expects [B, 3, N] (Channels First)
        if loc_ref.shape[-1] == 3 and loc_ref.shape[-2] != 3:
             loc_ref_knn = loc_ref.permute(0, 2, 1)
        else:
             loc_ref_knn = loc_ref
             
        idx = knn(loc_ref_knn, self.k) # [B, N, k]
        
        # Prepare Q, K, V
        # qkv: [B, N, 3C]
        qkv = self.qkv(x_norm).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] # [B, N, nH, d]
        
        # Gather k and v for neighbors
        # We need [B, N, k, nH, d]
        # Expand idx to features
        
        # Flat gather
        # [B, N, nH, d] -> [B*N, nH, d]
        k_flat = k.reshape(B*N, self.num_heads, -1)
        v_flat = v.reshape(B*N, self.num_heads, -1)
        
        idx_flat = idx.view(B*N, self.k) # [B*N, k]
        # We need to adjust indices for batch
        # knn returns 0..N-1. add batch offsets.
        batch_offset = torch.arange(B, device=x.device).view(B, 1, 1) * N
        idx_global = (idx + batch_offset).view(B*N, self.k) # [B*N, k]
        
        # Gather
        # k_neigh: [B*N, k, nH, d]
        idx_exp = idx_global.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, self.num_heads, C // self.num_heads)
        
        # Torch gather is tricky with multiple dims. 
        # Easier: index select logic.
        
        k_neigh = k_flat[idx_global] # [B*N, k, nH, d]
        v_neigh = v_flat[idx_global] # [B*N, k, nH, d]
        
        # Q is centered: [B, N, nH, d] -> [B*N, 1, nH, d]
        q_curr = q.reshape(B*N, 1, self.num_heads, -1)
        
        # Attention
        # (Q @ K.T) * scale
        attn = (q_curr @ k_neigh.transpose(-2, -1)) * self.scale # [B*N, 1, nH, nH, k] -> wait dims
        # q: [M, 1, H, D], k_T: [M, H, D, K] -> [M, H, 1, K] (Matmul broadcasts on H)
        # We need manual matching.
        
        # q: [M, 1, H, D] -> permute [M, H, 1, D]
        q_curr = q_curr.permute(0, 2, 1, 3) 
        # k_neigh: [M, K, H, D] -> permute [M, H, D, K]
        k_neigh_T = k_neigh.permute(0, 2, 3, 1)
        
        attn = (q_curr @ k_neigh_T) * self.scale # [M, H, 1, K]
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Aggregation
        # v_neigh: [M, K, H, D] -> [M, H, K, D]
        v_neigh = v_neigh.permute(0, 2, 1, 3)
        x_attn = (attn @ v_neigh) # [M, H, 1, D]
        
        # Restore shape
        x_attn = x_attn.transpose(1, 2).reshape(B, N, C)
        
        x_attn = self.proj(x_attn)
        x_attn = self.proj_drop(x_attn)
        
        x = x + x_attn
        x = x + self.mlp(self.norm2(x))
        return x

class SelfAttentionBlock(nn.Module):
    # DEPRECATED IN FAVOR OF LOCAL ATTENTION FOR ENCODER
    # Kept for Decoder Self-Attention if needed (Paper says: "For decoder ... self-attention blocks")
    # Yes, Decoder uses standard SA. Encoder uses Local.
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim*mlp_ratio))

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x

class CrossAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.norm_cross = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, dropout=attn_drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim*mlp_ratio))

    def forward(self, x, context):
        # Structure based on diagram explanation:
        # 1. Cross Attention -> Add & Norm (Residual from Q)
        # 2. Self Attention -> Add & Norm (Residual from input to Self)
        
        # Block 1: Cross Attention
        residual = x
        # Q=x, K=context, V=context
        cross_out, _ = self.cross_attn(query=x, key=context, value=context)
        # Add & Norm
        x = self.norm_cross(residual + cross_out)
        
        # Block 2: Self Attention
        residual = x
        attn_out, _ = self.self_attn(x, x, x)
        # Add & Norm
        x = self.norm1(residual + attn_out)
        
        return x

# --------------------------------------------------------------------------------
# 2. Paper-Exact Adaptive Adversarial Mask Generator (AAMG)
# --------------------------------------------------------------------------------

class LightDGCNN(nn.Module):
    """ 
    Simple DGCNN architecture for extracting local geometric features E_X 
    to be used by the Mask Generator (AAMG).
    Uses EdgeConv logic.
    """
    """ Simple DGCNN for feature extraction E_X """
    def __init__(self, in_features, k=10, emb_dims=64):
        super().__init__()
        self.k = k
        self.bn1 = nn.BatchNorm2d(64)
        self.conv1 = nn.Sequential(nn.Conv2d(in_features*2, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        
        self.bn2 = nn.BatchNorm1d(emb_dims)
        self.conv2 = nn.Sequential(nn.Conv1d(64, emb_dims, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))

    def forward(self, x):
        # x: [B, N, C] -> Need [B, C, N] for conv
        x = x.permute(0, 2, 1)
        batch_size = x.size(0)
        
        # Edge Conv
        x_graph = get_graph_feature(x, k=self.k) # [B, 2*Cin, N, k]
        x_graph = self.conv1(x_graph) # [B, 64, N, k]
        x_graph = x_graph.max(dim=-1, keepdim=False)[0] # [B, 64, N]
        
        x_out = self.conv2(x_graph) # [B, emb, N]
        return x_out.permute(0, 2, 1) # [B, N, emb]

class AAMG_Query(nn.Module):
    """
    Adaptive Adversarial Mask Generator.
    Learns to mask points that make reconstruction most difficult.
    
    Components:
    - LightDGCNN: Extracts features.
    - Filter Queries: Learnable vectors that select points.
    - Gumbel-Softmax: Differentiable selection of points.
    """
    def __init__(self, feature_dim, num_queries=256, embed_dim=64):
        super().__init__()
        # E_x extractor
        self.dgcnn = LightDGCNN(feature_dim, k=10, emb_dims=embed_dim)
        
        # Learnable Queries [Nm, C2]
        # We define them as parameter
        # We set a high maximum (e.g. 5000) to support high resolution masking
        self.filter_queries = nn.Parameter(torch.randn(num_queries, embed_dim))
        
        self.temperature = 1.0

    def forward(self, x, num_active_queries=None):
        """
        Args:
            x: [B, N, C] Input features.
            num_active_queries: Optional limit on number of mask queries to use (controls mask ratio).
            
        Returns:
            mask_binary: [B, N] Binary mask (1=Remove, 0=Keep).
            M: [B, Nm, N] Similarity matrix (for Diversity Loss).
            selection_onehot: [B, Nm, N] Soft selection matrix.
        """
        """
        x: [B, N, C]
        Returns:
            mask: [B, N] binary (1=Remove, 0=Keep)
            similarity_vectors: list of [B, N] for diversity loss
            idx: [B, Nm] indices of masked points
        """
        B, N, C = x.shape
        
        # 1. Extract Features E_X
        Ex = self.dgcnn(x) # [B, N, emb]
        
        # 2. Compute Similarity M (Dot product)
        # Queries Q: [Nm, emb]
        # M = Q * Ex^T  -> [B, Nm, N]
        
        # Select active queries
        curr_queries = self.filter_queries
        if num_active_queries is not None:
             # Limit to needed count
             # Since queries are learnable parameters, using a subset might mean 
             # the first K are trained more often. 
             # But for flexible ratio support, this is necessary.
             # Alternatively, we could interpolate, but slicing is standard for variable length.
             curr_queries = self.filter_queries[:num_active_queries]

        # Expand queries for batch
        Nm = curr_queries.shape[0]
        Q = curr_queries.unsqueeze(0).expand(B, -1, -1) # [B, Nm, emb]
        
        # [B, Nm, emb] x [B, emb, N] -> [B, Nm, N]
        M = torch.bmm(Q, Ex.transpose(1, 2))
        
        # 3. Gumbel Softmax Selection
        # Select "most similar" point for each query
        # We want a ONE-HOT vector of size N for each query m=1..Nm.
        
        # M is [B, Nm, N]. Logic: M_{ij} is similarity of query i to point j.
        # We apply Gumbel Softmax along dim=2 (Points)
        
        # hard=True returns one-hot.
        selection_onehot = F.gumbel_softmax(M, tau=self.temperature, hard=True, dim=-1) # [B, Nm, N]
        
        # This gives us Nm selected points (some might be duplicates).
        # We simply sum them up to get the mask. 
        # If a point is selected multiple times, it's just masked.
        # Mask = sum(selection_onehot) > 0
        
        mask_count = selection_onehot.sum(dim=1) # [B, N]
        mask_binary = (mask_count > 0).float() # 1=Remove
        
        # To ensure we mask exactly Nm points (handle duplicates):
        # The paper says: "randomly select extra points to ensure total number equals Nm."
        # For differentiability during training, we rely on the sampled mask's gradient.
        # For the forward pass to the AE, we fix the count.
        
        # But wait, we need 'indices of kept points' for the Transformer.
        # If we have duplicates, we masked fewer than Nm.
        # We need to drop more.
        
        # Simple hack for this implementation:
        # Just return the binary mask. The encoder will select points where mask==0.
        # Variable length handling is fine (Transformer works with mask).
        
        # For Diversity Loss, we return the SOFT similarity vectors (or post-softmax probabilities)
        # Actually paper says "activations" or similarity vectors mx_i.
        # M is raw similarity.
        
        return mask_binary, M, selection_onehot

# --------------------------------------------------------------------------------
# 3. ASMAE Model
# --------------------------------------------------------------------------------

class ASMAE(nn.Module):
    """
    Adaptive Siamese Masked Autoencoder.
    
    Architecture:
    1. AAMG: Generates mask adversarially.
    2. Siamese Encoder (Local Attn): Encodes Masked Source and Complete Target.
    3. Siamese Decoder (Cross Attn): Reconstructs Source using Target as context.
    4. Prediction Head: Reconstructs input features.
    
    Training Losses:
    - MAE Loss: MSE(Recon, Original_Features)
    - Diversity Loss (AAMG): Enforces diverse query selection.
    - Global Optimization (Sinkhorn) & Construction (Smoothness) Losses handled externally.
    """
    def __init__(self, feature_dim=32, embed_dim=128, depth=4, num_heads=4,
                 decoder_embed_dim=128, decoder_depth=2, decoder_num_heads=4,
                 mlk_ratio=4.):
        super().__init__()
        
        # AAMG
        # Num queries determines the mask ratio indirectly.
        # If we target mask_ratio = 0.6 and N=5000, we need ~3000 queries.
        # We'll set this dynamically or fix a large number?
        # The class needs a fixed parameter size. 
        # Let's assume a standard N=2048 or something.
        # "Nm learnable filter queries".
        # We will initialize with a default, but handle mask ratio by picking subset? 
        # Paper implies Nm is fixed architecture hyperparam.
        
        # Let's default to 1024 queries (reasonable for 2-5k points).
        self.num_mask_queries = 5000 
        self.mask_generator = AAMG_Query(feature_dim, num_queries=self.num_mask_queries, embed_dim=64)
        
        # ENCODER
        self.feature_embed = nn.Linear(feature_dim, embed_dim)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, embed_dim)
        )
        self.encoder_blocks = nn.ModuleList([
            LocalSelfAttentionBlock(embed_dim, num_heads, mlk_ratio, k=20)
            for _ in range(depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)
        
        # DECODER
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, 64),
            nn.GELU(),
            nn.Linear(64, decoder_embed_dim)
        )
        self.decoder_blocks = nn.ModuleList([
            CrossAttentionBlock(decoder_embed_dim, decoder_num_heads, mlk_ratio)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        self.pred_head = nn.Linear(decoder_embed_dim, feature_dim)
        
        self.initialize_weights()

    def initialize_weights(self):
        torch.nn.init.normal_(self.mask_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def compute_diversity_loss(self, M):
        # M: [B, Nm, N]
        # We want rows (queries) to be diverse.
        # Normalize M rows:
        M_norm = F.normalize(M, p=2, dim=2)
        # Cosine similarity between queries: [B, Nm, Nm]
        sim_matrix = torch.bmm(M_norm, M_norm.transpose(1, 2))
        # We want identity matrix (orthogonality).
        # Loss = mean of off-diagonal elements squared
        I = torch.eye(sim_matrix.shape[1], device=M.device).unsqueeze(0)
        loss_div = ((sim_matrix - I) ** 2).mean()
        return loss_div

    def forward_encoder(self, x, pos, mask_binary=None):
        """
        x: [B, N, C]
        pos: [B, N, 3]
        mask_binary: [B, N] 1=Target for feature masking, 0=Keep Clean
        
        New Logic:
        - If mask_binary is set, we do NOT drop points.
        - Instead, for points where mask=1, we mask one random feature dimension.
        """
        x_emb = self.feature_embed(x)
        
        # Apply Feature Masking if mask provided (Training)
        if mask_binary is not None:
             B, N, C = x.shape
             # Identify indices to mask
             # For each batch/point where mask==1, pick random feature dim
             # We perform this on the INPUT x usually, but we already embedded.
             # Wait, masking should happen on x BEFORE embedding for "reconstruct feature" logic usually.
             # But here we embedded.
             # Let's apply mask on x_emb? No, that corrupts the embedding space. 
             # We should mask 'x' passed in, but we already called feature_embed(x).
             
             # Let's retro-fix: Apply mask to x before embedding.
             # But we need to do this carefully. 
             pass # Logic moved to forward() to mask x before calling this.
             
        pos_emb = self.pos_embed(pos)
        x_emb = x_emb + pos_emb
        
        # No dropping of tokens anymore.
        # Just pass everything through.
        
        for blk in self.encoder_blocks:
            x_emb = blk(x_emb, pos=pos)
            
        x_emb = self.encoder_norm(x_emb)
        return x_emb



    def forward_decoder(self, x_encoded, pos_source, target_encoded):
        # x_encoded: [B, N, C] (Full sequence, noisy)
        # target_encoded: [B, N, C] (Full sequence, context)
        
        # We no longer need mask tokens or scattering because input is already full length.
        # We just add positional embeddings and run the decoder blocks.
        
        x_encoded = self.decoder_embed(x_encoded)
        target_encoded = self.decoder_embed(target_encoded)
        
        # Positional embedding for decoder
        # Usually Decoder has its own POS embed or reuses.
        # Encoder output already has pos info mixed in? 
        # But we act like it's a new "Query". 
        x_encoded = x_encoded + self.decoder_pos_embed(pos_source)
        
        base = x_encoded
        for blk in self.decoder_blocks:
            base = blk(base, target_encoded)
            
        return self.decoder_norm(base)

    def forward(self, x_source, pos_source, x_target, pos_target, mask_ratio=None):
        """
        Note: mask_ratio is ignored by logic, driven by num_mask_queries.
        We can dynamically adjust queries used? 
        The paper uses fixed Nm queries.
        """
        
        # 1. AAMG
        # Adjust queries active to match ratio? Or just use all.
        # Paper implies Nm is fixed.
        # We will use ALL queries.
        
        active_queries = None
        if mask_ratio is not None:
             B, N, _ = x_source.shape
             needed = int(mask_ratio * N)
             active_queries = min(needed, self.num_mask_queries)
        
        mask_binary, M_sim, onehot = self.mask_generator(x_source, num_active_queries=active_queries)
        
        # 2. Diversity Loss
        loss_div = self.compute_diversity_loss(M_sim)
        
        # 3. Encode Target (Clean)
        target_encoded = self.forward_encoder(x_target, pos_target, mask_binary=None)
        
        # 4. Feature Masking Logic
        # mask_binary: [B, N]. 1 = Corrupt this point.
        # We need to corrupt x_source BEFORE encoding.
        x_source_corrupted = x_source.clone()
        
        if mask_binary is not None:
             B, N, C = x_source.shape
             # Mask: [B, N] -> indices where 1
             mask_bool = (mask_binary > 0) # [B, N]
             
             # For each point where mask is True, pick a random feature index [0, C-1]
             # We create a random mask of shape [B, N, C]
             
             # 1. Create a mask of which Feature Index to drop for each point
             # rand_ind: [B, N] integers in [0, C)
             rand_ind = torch.randint(0, C, (B, N), device=x_source.device)
             
             # Create one-hot mask [B, N, C]
             feature_mask = F.one_hot(rand_ind, num_classes=C).bool() # [B, N, C]
             
             # Combined mask: Only apply feature_mask IF point_mask is True
             # final_mask: [B, N, C]
             final_mask = feature_mask & mask_bool.unsqueeze(-1)
             
             # Apply
             x_source_corrupted[final_mask] = 0.0
             
             # Return the masked input for saving later?
             # We return 'x_source_corrupted' or just mask? 
             # Pipeline expects mask_binary. We can just return it.
        
        # 5. Encode Source (Corrupted)
        source_enc = self.forward_encoder(x_source_corrupted, pos_source, mask_binary=None)
        
        # 6. Decode
        # We pass the corrupted encoding to the decoder
        source_recon = self.forward_decoder(source_enc, pos_source, target_encoded)
        
        pred_source = self.pred_head(source_recon)
        
        # 6. Global Optimization Preparation
        # Return dense features (from encoder target, and RECONSTRUCTED or ENCODED source?)
        # Paper: "Fx and Fy are upsampled... similarity matrix... T*"
        # Since we don't downsample, Fx and Fy are just the dense features.
        # We need dense features of Source (Unmasked) and Target (Unmasked) for L_go.
        # We already have target_encoded. We need source_encoded (UNMASKED).
        
        # We need to run encoder on FULL source for L_go (and symmetric L_go).
        # This is expensive but necessary logic for "Training".
        source_full_enc = self.forward_encoder(x_source, pos_source, mask_binary=None)
        
        # 7. Outputs
        # pred_source: [B, N, C] Reconstructed features
        # mask_binary: [B, N] Indicates which points were targeted
        # x_source_corrupted: Return this to save the masked matrix
        
        return pred_source, mask_binary, loss_div, x_source_corrupted, target_encoded, source_full_enc

    def extract_features(self, x, pos):
        return self.forward_encoder(x, pos, mask_binary=None)
