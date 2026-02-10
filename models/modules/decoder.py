import torch.nn as nn
from .layers import Mlp

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
        # 1. Cross Attention
        residual = x
        cross_out, _ = self.cross_attn(query=x, key=context, value=context)
        x = self.norm_cross(residual + cross_out)
        
        # 2. Self Attention
        residual = x
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(residual + attn_out)
        
        return x
