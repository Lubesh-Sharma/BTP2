import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def knn(ref: torch.Tensor, query: torch.Tensor, k: int = 24):
    """
    K-Nearest Neighbor retrieval via native PyTorch cdist.
    Fast, reliable, and requires no external CUDA C++ extensions (no knn_cuda).
    Args:
        ref: [B, N_ref, C]
        query: [B, N_query, C]
        k: number of nearest neighbors
    Returns:
        idx: [B, N_query, k]
    """
    dist = torch.cdist(query, ref)
    return dist.topk(k=k, largest=False, dim=-1)[1]

def get_graph_feature(ref: torch.Tensor, query: torch.Tensor, k: int = 20, idx=None, ref_xyz=None):
    """
    Extracts graph neighborhood features as in DGCNN and SE-ORNet (Deng et al., CVPR 2023).
    Args:
        ref: [B, N, C]
        query: [B, N, C]
        k: int, number of neighbors
        idx: optional precomputed neighbor indices [B, N, k]
        ref_xyz: optional reference 3D coordinates [B, N, 3]
    Returns:
        feature: [B, N, k, C]
        xyz: [B, N, k, 3] (if ref_xyz is provided)
    """
    batch_size, num_points, num_dims = ref.size()
    if idx is None:
        idx = knn(ref, query, k=k)
    idx_base = torch.arange(0, batch_size, device=idx.device).view(-1, 1, 1) * num_points
    idx = (idx + idx_base).view(-1)

    feature = ref.reshape(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    if ref_xyz is not None:
        xyz = ref_xyz.reshape(batch_size * num_points, -1)[idx, :]
        xyz = xyz.view(batch_size, num_points, k, 3)
        return feature, xyz
    return feature

class OrientModule(nn.Module):
    """
    Feature Interaction Module (FIM) from SE-ORNet (CVPR 2023).
    Correlates cross-shape edge graph features and coordinates.
    """
    def __init__(self, k: int, input_size: int, output_size: int):
        super().__init__()
        self.k = k
        self.linear = nn.Conv2d((input_size + 3) * 2, output_size, kernel_size=1, bias=False)
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.norm = nn.GroupNorm(1, output_size)

    def forward(self, xyz_s, xyz_t, feature_s, feature_t):
        feature, xyz = get_graph_feature(
            feature_s.transpose(2, 1),
            feature_t.transpose(2, 1),
            k=self.k,
            ref_xyz=xyz_s,
        )
        xyz_t_rep = xyz_t.unsqueeze(2).repeat(1, 1, self.k, 1)
        feature_t_rep = feature_t.transpose(2, 1).unsqueeze(2).repeat(1, 1, self.k, 1)
        feature_cat = torch.cat(
            (feature - feature_t_rep, feature, xyz - xyz_t_rep, xyz_t_rep), dim=-1
        )
        feature_cat = feature_cat.permute(0, 3, 1, 2).contiguous()
        feature_cat = self.relu(self.norm(self.linear(feature_cat)))
        output, _ = feature_cat.max(dim=-1, keepdim=False)
        return output

class EdgeConvModule(nn.Module):
    """
    EdgeConv module from DGCNN / SE-ORNet with GroupNorm for batch-size invariance.
    """
    def __init__(self, k: int, input_size: int, output_size: int):
        super().__init__()
        self.k = k
        self.linear = nn.Conv2d(input_size * 2, output_size, kernel_size=1, bias=False)
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.norm = nn.GroupNorm(1, output_size)

    def forward(self, input_tensor, idx=None):
        feature = get_graph_feature(
            input_tensor.transpose(2, 1), input_tensor.transpose(2, 1), k=self.k, idx=idx
        )
        input_rep = input_tensor.transpose(2, 1).unsqueeze(2).repeat(1, 1, self.k, 1)
        feature_cat = torch.cat((feature - input_rep, input_rep), dim=-1)
        feature_cat = feature_cat.permute(0, 3, 1, 2).contiguous()
        feature_cat = self.relu(self.norm(self.linear(feature_cat)))
        output, _ = feature_cat.max(dim=-1, keepdim=False)
        return output

class OrientNet(nn.Module):
    """
    SE-ORNet Core Orientation Network (Deng et al., CVPR 2023).
    Multi-scale EdgeConv + Bidirectional Feature Interaction Module (FIM).
    """
    def __init__(
        self,
        input_dims=[3, 64, 128, 256],
        output_dim=256,
        latent_dim=256,
        mlps=[256, 128, 128],
        num_neighs=24,
        input_neighs=27,
        num_class=8,
    ):
        super().__init__()
        self.num_neighs = num_neighs
        self.input_neighs = input_neighs

        # 1. Multi-scale input EdgeConv modules
        self.input_modules = nn.ModuleList()
        in_dim = input_dims[0]
        for out_dim in input_dims[1:]:
            self.input_modules.append(EdgeConvModule(self.input_neighs, in_dim, out_dim))
            in_dim = out_dim

        # 2. Bidirectional Feature Interaction Module (FIM) & Refinement EdgeConv
        self.orient_module = OrientModule(self.num_neighs, in_dim, latent_dim)
        self.edgeconv = EdgeConvModule(self.num_neighs, latent_dim, output_dim)

        # 3. Angle Classification MLPs
        self.mlps = nn.ModuleList()
        mlp_in = 2 * (latent_dim + output_dim)
        for dim in mlps:
            self.mlps.append(nn.Sequential(
                nn.Conv1d(mlp_in, dim, kernel_size=1, bias=False),
                nn.GroupNorm(1, dim),
                nn.LeakyReLU(negative_slope=0.2),
            ))
            mlp_in = dim

        self.num_class = num_class
        self.classifier = nn.Conv1d(mlp_in, self.num_class, 1, bias=False)

    def forward(self, xyz_s, xyz_t):
        batch_size = xyz_s.shape[0]
        idx_s = knn(xyz_s, xyz_s, k=self.input_neighs)
        idx_t = knn(xyz_t, xyz_t, k=self.input_neighs)

        feat_s = xyz_s.transpose(1, 2)
        feat_t = xyz_t.transpose(1, 2)
        for input_module in self.input_modules:
            feat_s = input_module(feat_s, idx=idx_s)
            feat_t = input_module(feat_t, idx=idx_t)

        latent_s_0 = self.orient_module(xyz_s, xyz_t, feat_s, feat_t)
        latent_t_0 = self.orient_module(xyz_t, xyz_s, feat_t, feat_s)

        latent_s_1 = self.edgeconv(latent_s_0)
        latent_t_1 = self.edgeconv(latent_t_0)

        x = torch.cat((latent_s_0, latent_s_1), dim=1)
        y = torch.cat((latent_t_0, latent_t_1), dim=1)

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        y1 = F.adaptive_max_pool1d(y, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)
        y2 = F.adaptive_avg_pool1d(y, 1).view(batch_size, -1)

        x = torch.cat((x1, x2), 1).unsqueeze(-1)
        y = torch.cat((y1, y2), 1).unsqueeze(-1)

        for m in self.mlps:
            x = m(x)
            y = m(y)

        angle_x = self.classifier(x).squeeze(-1)
        angle_y = self.classifier(y).squeeze(-1)

        return {'angle_x': angle_x, 'angle_y': angle_y}

class OrientationModule(nn.Module):
    """
    SE-ORNet / DV-Matcher Relative 3D Orientation Module (Deng et al., CVPR 2023).
    
    Predicts canonical relative yaw rotation between 3D shapes using discrete angle
    classification into M=8 bins (45-degree bins covering 360 degrees):
      - bin 0: -45°
      - bin 1:   0° (identity / upright pose)
      - bin 2: +45°
      - bin 3: +90°
      - bin 4: +135°
      - bin 5: +180° (mirror / bilateral flip)
      - bin 6: -135° (+225°)
      - bin 7: -90° (+270°)
    """
    def __init__(self, in_channels=3, num_bins=8, num_points=512):
        super().__init__()
        self.num_bins = num_bins
        self.num_points = num_points

        self.orientnet = OrientNet(
            input_dims=[in_channels, 64, 128, 256],
            output_dim=256,
            latent_dim=256,
            mlps=[256, 128, 128],
            num_neighs=24,
            input_neighs=27,
            num_class=num_bins
        )

        offset = math.pi / 4.0 if num_bins == 8 else 0.0
        angles = torch.arange(num_bins, dtype=torch.float32) * (2.0 * math.pi / num_bins) - offset
        self.register_buffer('angles', angles)

    def normalize_pc(self, pc):
        """Centered and unit-sphere normalized point cloud matching SE-ORNet."""
        c = torch.mean(pc, dim=1, keepdim=True)
        pc_c = pc - c
        scale = torch.max(torch.sqrt(torch.sum(pc_c ** 2, dim=-1, keepdim=True)), dim=1, keepdim=True)[0]
        return pc_c / (scale + 1e-8), c, scale

    def forward(self, p_src, p_tgt=None, return_dict=False):
        if p_tgt is None:
            B = p_src.shape[0]
            identity_bin = 1 if self.num_bins == 8 else 0
            logits = torch.full((B, self.num_bins), -10.0, device=p_src.device, dtype=p_src.dtype)
            logits[:, identity_bin] = 10.0
            if return_dict:
                return {'angle_x': logits, 'angle_y': logits}
            return logits

        B, N_s, _ = p_src.shape
        _, N_t, _ = p_tgt.shape

        if N_s > self.num_points:
            step_s = max(1, N_s // self.num_points)
            p_s_sub = p_src[:, ::step_s, :][:, :self.num_points, :]
        else:
            p_s_sub = p_src

        if N_t > self.num_points:
            step_t = max(1, N_t // self.num_points)
            p_t_sub = p_tgt[:, ::step_t, :][:, :self.num_points, :]
        else:
            p_t_sub = p_tgt

        p_s_norm, _, _ = self.normalize_pc(p_s_sub)
        p_t_norm, _, _ = self.normalize_pc(p_t_sub)

        out = self.orientnet(p_s_norm, p_t_norm)
        if return_dict:
            return out
        return out['angle_x']

    def predict_rotation(self, p_src, p_tgt=None, soft=None, temperature=0.5):
        """
        Predicts 3x3 rotation matrix R to align p_src into p_tgt's coordinate frame.
        """
        B = p_src.shape[0]
        device = p_src.device
        dtype = p_src.dtype
        identity_bin = 1 if self.num_bins == 8 else 0

        if p_tgt is None:
            R = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
            logits = torch.full((B, self.num_bins), -10.0, device=device, dtype=dtype)
            logits[:, identity_bin] = 10.0
            return R, logits

        logits = self.forward(p_src, p_tgt, return_dict=False)

        cos_all = torch.cos(self.angles)
        sin_all = torch.sin(self.angles)
        zeros_all = torch.zeros_like(cos_all)
        ones_all = torch.ones_like(cos_all)

        r0 = torch.stack([cos_all, zeros_all, sin_all], dim=-1)
        r1 = torch.stack([zeros_all, ones_all, zeros_all], dim=-1)
        r2 = torch.stack([-sin_all, zeros_all, cos_all], dim=-1)
        R_all = torch.stack([r0, r1, r2], dim=1)  # [num_bins, 3, 3]

        if soft:
            probs = F.softmax(logits / max(temperature, 1e-4), dim=-1)
            R = torch.einsum('bm,mij->bij', probs, R_all)
        else:
            pred_bin = torch.argmax(logits, dim=-1)
            R = R_all[pred_bin]

        return R, logits

    def align(self, p_src, p_tgt=None, soft=None):
        """
        Aligns p_src into p_tgt's frame:
          p_aligned = (p_src - c_src) @ R.T + c_src
        """
        if p_tgt is None:
            B = p_src.shape[0]
            R = torch.eye(3, device=p_src.device, dtype=p_src.dtype).unsqueeze(0).expand(B, -1, -1)
            return p_src, R

        R, _ = self.predict_rotation(p_src, p_tgt, soft=soft)
        c_src = torch.mean(p_src, dim=1, keepdim=True)
        p_aligned = torch.bmm(p_src - c_src, R.transpose(1, 2)) + c_src
        return p_aligned, R
