import torch
import torch.nn as nn
from .layers import Mlp, knn

class LocalSelfAttentionBlock(nn.Module):
    """
    Local Self-Attention Block using k-Nearest Neighbors.
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0., k=20):
        super().__init__()
        self.k = k
        self.norm1 = nn.LayerNorm(dim)
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim*mlp_ratio))
        # Complex Phase Directional Projection (Method A from Complex Functional Maps)
        # Encodes tangent angle (cos theta, sin theta); sin(theta) flips sign under reflection
        self.phase_proj = nn.Linear(2, num_heads, bias=False)
        nn.init.zeros_(self.phase_proj.weight)

    def forward(self, x, pos=None):
        B, N, C = x.shape
        shortcut = x
        x_norm = self.norm1(x)
        loc_ref = x_norm if pos is None else pos
        if loc_ref.shape[-1] == 3 and loc_ref.shape[-2] != 3:
             loc_ref_knn = loc_ref.permute(0, 2, 1)
        else:
             loc_ref_knn = loc_ref
        idx = knn(loc_ref_knn, self.k) # [B, N, k]
        
        qkv = self.qkv(x_norm).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 1, 3, 4)
        q, k, v = qkv[0], qkv[1], qkv[2] 
        
        k_flat = k.reshape(B*N, self.num_heads, -1)
        v_flat = v.reshape(B*N, self.num_heads, -1)
        idx_flat = idx.view(B*N, self.k)
        batch_offset = torch.arange(B, device=x.device).view(B, 1, 1) * N
        idx_global = (idx + batch_offset).view(B*N, self.k)
        
        k_neigh = k_flat[idx_global]
        v_neigh = v_flat[idx_global]
        q_curr = q.reshape(B*N, 1, self.num_heads, -1)
        
        q_curr = q_curr.permute(0, 2, 1, 3) 
        k_neigh_T = k_neigh.permute(0, 2, 3, 1)
        
        # Method A: Complex Phase Tangent Directional Attention
        if pos is not None and self.k >= 3:
            pos_flat = pos.reshape(B * N, 3)
            pos_neigh = pos_flat[idx_global].view(B, N, self.k, 3)
            delta_p = pos_neigh - pos.unsqueeze(2)  # [B, N, k, 3]
            
            # Local tangent basis using nearest spatial neighbors
            e1 = torch.nn.functional.normalize(delta_p[:, :, 1, :], dim=-1, eps=1e-8)
            cross_12 = torch.cross(e1, delta_p[:, :, 2, :], dim=-1)
            n_loc = torch.nn.functional.normalize(cross_12, dim=-1, eps=1e-8)
            e2 = torch.cross(n_loc, e1, dim=-1)
            
            # Project displacements onto local tangent frame (u, v)
            u = torch.sum(delta_p * e1.unsqueeze(2), dim=-1)  # [B, N, k]
            v = torch.sum(delta_p * e2.unsqueeze(2), dim=-1)  # [B, N, k]
            r = torch.sqrt(u**2 + v**2 + 1e-12)
            cos_theta = u / r
            sin_theta = v / r
            
            phase_feat = torch.stack([cos_theta, sin_theta], dim=-1)  # [B, N, k, 2]
            phase_bias = self.phase_proj(phase_feat).permute(0, 1, 3, 2).unsqueeze(3).reshape(B * N, self.num_heads, 1, self.k)
            attn = (q_curr @ k_neigh_T) * self.scale + phase_bias
        else:
            attn = (q_curr @ k_neigh_T) * self.scale
            
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        v_neigh = v_neigh.permute(0, 2, 1, 3)
        x_attn = (attn @ v_neigh)
        x_attn = x_attn.transpose(1, 2).reshape(B, N, C)
        x_attn = self.proj(x_attn)
        x_attn = self.proj_drop(x_attn)
        
        x = x + x_attn
        x = x + self.mlp(self.norm2(x))
        return x
