import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class OrientationModule(nn.Module):
    """
    SE-ORNet / DV-Matcher Orientation Estimation Module (Deng et al., CVPR 2023).
    
    Predicts relative 3D orientation between a Source and Target point cloud pair
    using discrete angle classification into M=12 bins (30-degree bins covering 360 degrees).
    
    Key Features:
    1. Cross-Covariance Tensor: Integrates centered second-moment cross-covariance
       H = (1/N) * P_src^T @ P_tgt in SO(3), providing direct geometric rotation signals.
    2. Deep Point Encoders: Permutation-invariant PointNet feature extraction for both shapes.
    3. Discrete Angle Classification: Predicts probability distribution over M=12 bins (30 deg each).
    4. Pure Point Cloud: Requires no triangular faces, no mesh topology, and no surface normals.
       Operates on arbitrary 3D shapes (humans, quadrupeds/animals, general manifolds).
    """
    def __init__(self, in_channels=3, num_bins=12, k=16):
        super().__init__()
        self.num_bins = num_bins
        self.k = k
        
        # Point feature encoder for source and target
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.norm1 = nn.InstanceNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.norm2 = nn.InstanceNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.norm3 = nn.InstanceNorm1d(256)
        
        # Angle classification head:
        # Inputs: source global feat (256) + target global feat (256) + cross-covariance H (9) = 521 channels
        self.head = nn.Sequential(
            nn.Linear(256 + 256 + 9, 256),
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
        Returns: [B, 256] global pooled features
        """
        p_t = p_c.transpose(1, 2)
        x1 = F.leaky_relu(self.norm1(self.conv1(p_t)), 0.2)
        x2 = F.leaky_relu(self.norm2(self.conv2(x1)), 0.2)
        x3 = F.leaky_relu(self.norm3(self.conv3(x2)), 0.2)
        g_feat = torch.max(x3, dim=-1)[0]  # [B, 256]
        return g_feat

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
            
        c_s = torch.mean(p_s_sub, dim=1, keepdim=True)
        c_t = torch.mean(p_t_sub, dim=1, keepdim=True)
        p_s_c = p_s_sub - c_s
        p_t_c = p_t_sub - c_t
        
        # 1. Direct cross-covariance tensor in SO(3): H = (1/N) * P_s_c^T @ P_t_c
        min_n = min(p_s_c.shape[1], p_t_c.shape[1])
        H = torch.bmm(p_s_c[:, :min_n].transpose(1, 2), p_t_c[:, :min_n]) / float(min_n)
        H_flat = H.reshape(B, 9)
        
        # 2. Extract shape representations
        f_s = self.extract_point_features(p_s_c)  # [B, 256]
        f_t = self.extract_point_features(p_t_c)  # [B, 256]
        
        # 3. Concatenate shape representations + cross-covariance
        joint_feat = torch.cat([f_s, f_t, H_flat], dim=-1)  # [B, 521]
        logits = self.head(joint_feat)  # [B, num_bins]
        return logits

    def predict_rotation(self, p_src, p_tgt=None, soft=None, temperature=0.5):
        """
        Predicts the relative 3x3 rotation matrix R to align p_src into p_tgt.
        If soft is True (or during training by default), returns a differentiable
        expectation over rotation matrices so correspondence and area losses
        can backpropagate gradients into the orientation module.
        """
        if p_tgt is None:
            B = p_src.shape[0]
            R = torch.eye(3, device=p_src.device, dtype=p_src.dtype).unsqueeze(0).expand(B, -1, -1)
            logits = torch.zeros(B, self.num_bins, device=p_src.device, dtype=p_src.dtype)
            logits[:, 0] = 1.0
            return R, logits
            
        logits = self.forward(p_src, p_tgt)
        device = p_src.device
        dtype = p_src.dtype
        B = p_src.shape[0]

        if soft is None:
            soft = self.training

        # All candidate rotation matrices for discrete bins
        bin_angles = torch.arange(self.num_bins, device=device, dtype=dtype) * (2.0 * math.pi / self.num_bins)
        cos_all = torch.cos(bin_angles)
        sin_all = torch.sin(bin_angles)
        zeros_all = torch.zeros_like(cos_all)
        ones_all = torch.ones_like(cos_all)

        # [num_bins, 3, 3] rotation matrices around Y
        r0 = torch.stack([cos_all, zeros_all, sin_all], dim=-1)
        r1 = torch.stack([zeros_all, ones_all, zeros_all], dim=-1)
        r2 = torch.stack([-sin_all, zeros_all, cos_all], dim=-1)
        R_all = torch.stack([r0, r1, r2], dim=1)  # [num_bins, 3, 3]

        if soft:
            # Differentiable soft rotation matrix (expectation)
            probs = F.softmax(logits / max(temperature, 1e-4), dim=-1)  # [B, num_bins]
            # R: [B, 3, 3] = sum_b probs[b] * R_all[b]
            R = torch.einsum('bm,mij->bij', probs, R_all)
        else:
            pred_bin = torch.argmax(logits, dim=-1)  # [B]
            R = R_all[pred_bin]  # [B, 3, 3]

        return R, logits

    def align(self, p_src, p_tgt=None, soft=None):
        """
        Aligns p_src to p_tgt by predicting R and applying it:
        p_aligned = (p_src - c_src) @ R.T + c_src
        """
        if p_tgt is None:
            B = p_src.shape[0]
            R = torch.eye(3, device=p_src.device, dtype=p_src.dtype).unsqueeze(0).expand(B, -1, -1)
            return p_src, R
            
        R, _ = self.predict_rotation(p_src, p_tgt, soft=soft)
        c_src = torch.mean(p_src, dim=1, keepdim=True)
        # Apply rotation (transpose for row vector multiplication)
        p_aligned = torch.bmm(p_src - c_src, R.transpose(1, 2)) + c_src
        return p_aligned, R

