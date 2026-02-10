import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules.aamg import AAMG_Query
from .modules.encoder import LocalSelfAttentionBlock
from .modules.decoder import CrossAttentionBlock
from .modules.feature_masking import get_feature_mask_indices, apply_feature_mask

class ASMAE(nn.Module):
    """
    Adaptive Siamese Masked Autoencoder.
    """
    def __init__(self, feature_dim=32, embed_dim=128, depth=4, num_heads=4,
                 decoder_embed_dim=128, decoder_depth=2, decoder_num_heads=4,
                 mlk_ratio=4.):
        super().__init__()
        
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
        M_norm = F.normalize(M, p=2, dim=2)
        sim_matrix = torch.bmm(M_norm, M_norm.transpose(1, 2))
        I = torch.eye(sim_matrix.shape[1], device=M.device).unsqueeze(0)
        loss_div = ((sim_matrix - I) ** 2).mean()
        return loss_div

    def forward_encoder(self, x, pos, mask_binary=None):
        x_emb = self.feature_embed(x)
        if mask_binary is not None:
             pass 
        pos_emb = self.pos_embed(pos)
        x_emb = x_emb + pos_emb
        
        for blk in self.encoder_blocks:
            x_emb = blk(x_emb, pos=pos)
        x_emb = self.encoder_norm(x_emb)
        return x_emb

    def forward_decoder(self, x_encoded, pos_source, target_encoded):
        x_encoded = self.decoder_embed(x_encoded)
        target_encoded = self.decoder_embed(target_encoded)
        x_encoded = x_encoded + self.decoder_pos_embed(pos_source)
        base = x_encoded
        for blk in self.decoder_blocks:
            base = blk(base, target_encoded)
        return self.decoder_norm(base)

    def forward(self, x_source, pos_source, x_target, pos_target, mask_ratio=None, feature_ratio=0.2):
        active_queries = None
        if mask_ratio is not None:
             B, N, _ = x_source.shape
             needed = int(mask_ratio * N)
             active_queries = min(needed, self.num_mask_queries)
        
        mask_binary, M_sim, onehot = self.mask_generator(x_source, num_active_queries=active_queries)
        loss_div = self.compute_diversity_loss(M_sim)
        
        target_encoded = self.forward_encoder(x_target, pos_target, mask_binary=None)
        
        x_source_corrupted = x_source.clone()
        final_mask = None
        if mask_binary is not None:
             feat_indices = get_feature_mask_indices(x_source, feature_ratio)
             x_source_corrupted, final_mask = apply_feature_mask(x_source, feat_indices, mask_binary)
        
        source_enc = self.forward_encoder(x_source_corrupted, pos_source, mask_binary=None)
        source_recon = self.forward_decoder(source_enc, pos_source, target_encoded)
        pred_source = self.pred_head(source_recon)
        
        return pred_source, mask_binary, loss_div, x_source_corrupted, target_encoded, final_mask

    def extract_features(self, x, pos):
        return self.forward_encoder(x, pos, mask_binary=None)
