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
