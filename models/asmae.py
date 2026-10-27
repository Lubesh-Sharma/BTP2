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
                 mlk_ratio=4., num_mask_queries=5000, encoder_k=20, aamg_k=10,
                 aamg_emb_dim=64, pos_embed_dim=64, temperature=1.0, dino_dim=0):
        super().__init__()
        self.dino_dim = dino_dim
        
        self.num_mask_queries = num_mask_queries 
        self.mask_generator = AAMG_Query(feature_dim, num_queries=self.num_mask_queries, embed_dim=aamg_emb_dim, k=aamg_k, temperature=temperature)
        
        # ENCODER
        self.feature_embed = nn.Linear(feature_dim, embed_dim)
        self.pos_embed = nn.Sequential(
            nn.Linear(3, pos_embed_dim),
            nn.GELU(),
            nn.Linear(pos_embed_dim, embed_dim)
        )
        
        # Multi-scale hierarchical receptive fields across encoder layers:
        # Breaks local receptive field symmetry blindness by expanding from local (k=16) to global (k=128)
        if isinstance(encoder_k, (list, tuple)):
            k_list = list(encoder_k)
            if len(k_list) < depth:
                k_list = k_list + [k_list[-1]] * (depth - len(k_list))
        elif encoder_k == 20 or encoder_k is None:
            # Default hierarchical progression for depth 4: [16, 32, 64, 128]
            k_list = [min(128, 16 * (2 ** i)) for i in range(depth)]
            # Localized hierarchical progression for depth 4: [16, 24, 32, 48]
            # Keeps receptive fields tightly focused on limb geometry and prevents hand-to-leg feature leakage
            default_k = [16, 24, 32, 48]
            k_list = default_k if depth == 4 else [min(48, 16 + i * 8) for i in range(depth)]
        else:
            k_list = [min(128, max(8, int(encoder_k * (2 ** (i - 1))))) if i > 0 else encoder_k for i in range(depth)]
            k_list = [min(48, max(8, int(encoder_k * (1.2 ** i)))) for i in range(depth)]
            
        self.encoder_k_list = k_list
        self.encoder_blocks = nn.ModuleList([
            LocalSelfAttentionBlock(embed_dim, num_heads, mlk_ratio, k=k_list[i])
            for i in range(depth)
        ])
        self.encoder_norm = nn.LayerNorm(embed_dim)
        
        # DECODER
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        self.decoder_pos_embed = nn.Sequential(
            nn.Linear(3, pos_embed_dim),
            nn.GELU(),
            nn.Linear(pos_embed_dim, decoder_embed_dim)
        )
        self.decoder_blocks = nn.ModuleList([
            CrossAttentionBlock(decoder_embed_dim, decoder_num_heads, mlk_ratio)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(decoder_embed_dim)
        
        self.pred_head = nn.Linear(decoder_embed_dim, feature_dim)

        # Optional DINO distillation head: geometry-only encoder -> DINOv2 space.
        # Lets the encoder predict semantic features for vertices that no camera saw.
        self.dino_head = None
        if dino_dim > 0:
            self.dino_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2), nn.GELU(), nn.Linear(embed_dim * 2, dino_dim))
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

    @staticmethod
    def features_in_coordinate_frame(x, pos):
        """Keep appended XYZ channels in the actual encoder coordinate frame.

        Preprocessing appends per-axis [0, 1] coordinates; descriptor channels
        and the caller's reconstruction targets remain unchanged.
        """
        if x.shape[-1] < 3 or x.shape[:2] != pos.shape[:2]:
            raise ValueError("Expected descriptors with appended XYZ and matching points.")
        minimum = pos.amin(dim=1, keepdim=True)
        extent = pos.amax(dim=1, keepdim=True) - minimum
        xyz = (pos - minimum) / extent.clamp_min(1e-8)
        return torch.cat((x[..., :-3], xyz), dim=-1)

    def forward_encoder(self, x, pos, mask_binary=None):
        x = self.features_in_coordinate_frame(x, pos)
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
        x_source = self.features_in_coordinate_frame(x_source, pos_source)
        x_target = self.features_in_coordinate_frame(x_target, pos_target)
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

    def predict_dino(self, z):
        """Encoder embedding [B, N, embed] -> unit-norm predicted DINO feature."""
        if self.dino_head is None:
            return None
        return F.normalize(self.dino_head(z), dim=-1)


def build_asmae(model_cfg, feature_dim, dino_dim=0):
    """Construct ASMAE from a YAML model block (shared by train / test / corres)."""
    return ASMAE(
        feature_dim=feature_dim,
        embed_dim=model_cfg['embed_dim'],
        depth=model_cfg['depth'],
        num_heads=model_cfg['num_heads'],
        decoder_embed_dim=model_cfg['decoder_embed_dim'],
        decoder_depth=model_cfg['decoder_depth'],
        decoder_num_heads=model_cfg['decoder_num_heads'],
        mlk_ratio=model_cfg['mlk_ratio'],
        num_mask_queries=model_cfg.get('num_mask_queries', 5000),
        encoder_k=model_cfg.get('encoder_k', 20),
        aamg_k=model_cfg.get('aamg_k', 10),
        aamg_emb_dim=model_cfg.get('aamg_emb_dim', 64),
        pos_embed_dim=model_cfg.get('pos_embed_dim', 64),
        temperature=model_cfg.get('temperature', 1.0),
        dino_dim=dino_dim,
    )
