import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class OrientationModule(nn.Module):
    """
    SE-ORNet / DV-Matcher Orientation Estimation Module (Deng et al., CVPR 2023).
    
    Predicts relative 3D orientation between a Source and Target point cloud pair
    using a Feature Interaction Module (FIM) and discrete angle classification.
    
    Key Properties:
    1. Feature Interaction Module (FIM): Computes cross k-NN edge features
       between Source and Target in coordinate/feature space with residual connection.
    2. Discrete Angle Classification: Predicts probability distribution over M discrete
       angle bins (default M=8, covering 360 degrees in 45-degree bins).
    3. Pure Point Cloud: Requires no triangular faces, no mesh topology, and no surface
       normals. Operates on general point clouds (humans, quadrupeds/animals, arbitrary 3D shapes).
    """
    def __init__(self, in_channels=3, num_bins=8, k=16):
        super().__init__()
        self.num_bins = num_bins
        self.k = k
        
        # Point feature encoder (EdgeConv / PointNet layers)
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.norm1 = nn.InstanceNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.norm2 = nn.InstanceNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.norm3 = nn.InstanceNorm1d(256)
        
        # Feature Interaction Module (FIM): MLP on spatial position differences and feature differences
        # Edge spatial: (p_i, q_ij - p_i) -> 6 channels
        # Edge feature: (F_s, F_t_gathered - F_s) -> 512 channels (Total: 518 channels)
        self.fim_mlp = nn.Sequential(
            nn.Conv2d(6 + 512, 256, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fim_skip = nn.Conv1d(256, 256, 1)
        
        # Refinement Conv
        self.refine = nn.Sequential(
            nn.Conv1d(256, 256, 1),
            nn.InstanceNorm1d(256),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Angle classification head: Max + Avg pooling -> 512 channels
        self.head = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(128, num_bins)
        )
        
        # Initialize head bias to zero for neutral uniform class probability initially
        self.head[-1].bias.data.zero_()

    def extract_point_features(self, p_c):
        """
        Extracts 256-dim point features from centered point coordinates.
        p_c: [B, N, 3] centered coordinates
        """
        p_t = p_c.transpose(1, 2)
        x1 = F.leaky_relu(self.norm1(self.conv1(p_t)), 0.2)
        x2 = F.leaky_relu(self.norm2(self.conv2(x1)), 0.2)
        x3 = F.leaky_relu(self.norm3(self.conv3(x2)), 0.2)
        return x3  # [B, 256, N]

    def forward(self, p_src, p_tgt=None):
        """
        Predicts relative rotation classification logits between p_src and p_tgt.
        Args:
            p_src: [B, N_s, 3] Source point cloud
            p_tgt: [B, N_t, 3] Target point cloud (if None, defaults to p_src)
        Returns:
            logits: [B, num_bins] Angle classification logits
        """
        if p_tgt is None:
            p_tgt = p_src
            
        B, N_s, _ = p_src.shape
        _, N_t, _ = p_tgt.shape
        
        # Subsample for lightweight computation if N > 1024
        if N_s > 1024:
            step_s = max(1, N_s // 1024)
            p_s_sub = p_src[:, ::step_s, :][:, :1024, :]
        else:
            p_s_sub = p_src
            
        if N_t > 1024:
            step_t = max(1, N_t // 1024)
            p_t_sub = p_tgt[:, ::step_t, :][:, :1024, :]
        else:
            p_t_sub = p_tgt
            
        N_sub_s = p_s_sub.shape[1]
        N_sub_t = p_t_sub.shape[1]
        
        c_s = torch.mean(p_s_sub, dim=1, keepdim=True)
        c_t = torch.mean(p_t_sub, dim=1, keepdim=True)
        p_s_c = p_s_sub - c_s
        p_t_c = p_t_sub - c_t
        
        # Extract features
        f_s = self.extract_point_features(p_s_c)  # [B, 256, N_sub_s]
        f_t = self.extract_point_features(p_t_c)  # [B, 256, N_sub_t]
        
        # Feature Interaction Module: Cross k-NN in feature space
        dist = torch.cdist(f_s.transpose(1, 2), f_t.transpose(1, 2))  # [B, N_sub_s, N_sub_t]
        k = min(self.k, N_sub_t)
        knn_idx = torch.topk(dist, k=k, dim=-1, largest=False)[1]  # [B, N_sub_s, k]
        
        # Gather target coordinates
        idx_expanded = knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        p_t_gathered = torch.gather(p_t_c.unsqueeze(1).expand(-1, N_sub_s, -1, -1), 2, idx_expanded)  # [B, N_sub_s, k, 3]
        p_s_exp = p_s_c.unsqueeze(2).expand(-1, -1, k, -1)  # [B, N_sub_s, k, 3]
        
        # Edge spatial: (p_i, q_ij - p_i) -> [B, 6, N_sub_s, k]
        edge_spatial = torch.cat([p_s_exp, p_t_gathered - p_s_exp], dim=-1).permute(0, 3, 1, 2)
        f_s_exp = f_s.unsqueeze(-1).expand(-1, -1, -1, k)  # [B, 256, N_sub_s, k]
        
        # Edge feature: (f_s, f_t_gathered - f_s) -> [B, 512, N_sub_s, k]
        idx_f = knn_idx.unsqueeze(1).expand(-1, 256, -1, -1)
        f_t_gathered = torch.gather(f_t.unsqueeze(2).expand(-1, -1, N_sub_s, -1), 3, idx_f)
        edge_feat = torch.cat([edge_spatial, f_s_exp, f_t_gathered - f_s_exp], dim=1)  # [B, 518, N_sub_s, k]
        
        # FIM MLP and MaxPool
        e = self.fim_mlp(edge_feat)  # [B, 256, N_sub_s, k]
        p_out = torch.max(e, dim=-1)[0]  # [B, 256, N_sub_s]
        p_out = p_out + self.fim_skip(f_s)  # residual skip connection
        
        # Refinement Conv
        p_hat = self.refine(p_out)  # [B, 256, N_sub_s]
        
        # Global Max + Avg Pooling
        g_max = torch.max(p_hat, dim=2)[0]
        g_avg = torch.mean(p_hat, dim=2)
        global_feat = torch.cat([g_max, g_avg], dim=-1)  # [B, 512]
        
        logits = self.head(global_feat)  # [B, num_bins]
        return logits

    def predict_rotation(self, p_src, p_tgt=None):
        """
        Predicts the relative 3x3 rotation matrix R to align p_src into p_tgt.
        Returns:
            R: [B, 3, 3] rotation matrix
            logits: [B, num_bins] classification logits
        """
        if p_tgt is None:
            B = p_src.shape[0]
            R = torch.eye(3, device=p_src.device, dtype=p_src.dtype).unsqueeze(0).expand(B, -1, -1)
            logits = torch.zeros(B, self.num_bins, device=p_src.device, dtype=p_src.dtype)
            logits[:, 0] = 1.0
            return R, logits
            
        logits = self.forward(p_src, p_tgt)
        pred_bin = torch.argmax(logits, dim=-1)  # [B]
        
        # Discretized angle bin center
        angle = pred_bin.float() * (2.0 * math.pi / self.num_bins)
        
        cos_t = torch.cos(angle)
        sin_t = torch.sin(angle)
        zero = torch.zeros_like(cos_t)
        one = torch.ones_like(cos_t)
        
        # Rotation around vertical axis (Y-axis)
        row0 = torch.stack([cos_t, zero, sin_t], dim=-1)
        row1 = torch.stack([zero, one, zero], dim=-1)
        row2 = torch.stack([-sin_t, zero, cos_t], dim=-1)
        R = torch.stack([row0, row1, row2], dim=1)  # [B, 3, 3]
        return R, logits

    def align(self, p_src, p_tgt=None):
        """
        Aligns p_src to p_tgt by predicting R and applying it:
        p_aligned = (p_src - c_src) @ R + c_src
        """
        if p_tgt is None:
            B = p_src.shape[0]
            R = torch.eye(3, device=p_src.device, dtype=p_src.dtype).unsqueeze(0).expand(B, -1, -1)
            return p_src, R
            
        R, _ = self.predict_rotation(p_src, p_tgt)
        c_src = torch.mean(p_src, dim=1, keepdim=True)
        p_aligned = torch.bmm(p_src - c_src, R) + c_src
        return p_aligned, R
