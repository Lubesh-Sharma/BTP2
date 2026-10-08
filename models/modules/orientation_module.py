import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha=1.0):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * -ctx.alpha, None


def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)


def normalize_pc_torch(points: torch.Tensor):
    """
    Center and scale point cloud tensor to unit sphere on GPU.
    Args:
        points: [B, N, 3] or [N, 3]
    Returns:
        normalized_points: same shape as input
    """
    if points.dim() == 2:
        centroid = torch.mean(points, dim=0, keepdim=True)
        centered = points - centroid
        scale = torch.max(torch.sqrt(torch.sum(centered ** 2, dim=-1)))
        return centered / (scale + 1e-8)
    elif points.dim() == 3:
        centroid = torch.mean(points, dim=1, keepdim=True)
        centered = points - centroid
        scale = torch.max(torch.sqrt(torch.sum(centered ** 2, dim=-1)), dim=1, keepdim=True)[0].unsqueeze(-1)
        return centered / (scale + 1e-8)
    return points


def knn_points(ref: torch.Tensor, query: torch.Tensor, k: int):
    """
    Find k nearest neighbors in ref for each point in query.
    Supports varying number of points between ref (N_ref) and query (N_query).
    Args:
        ref: [B, N_ref, C]
        query: [B, N_query, C]
        k: int
    Returns:
        idx: [B, N_query, k] indices in ref
    """
    dists = torch.cdist(query, ref)  # [B, N_query, N_ref]
    idx = torch.topk(dists, k=k, dim=-1, largest=False)[1]  # [B, N_query, k]
    return idx


def get_graph_feature(ref: torch.Tensor, query: torch.Tensor, k: int = 20, idx=None, ref_xyz=None):
    """
    Extract graph edge features between query and ref.
    Supports asymmetric point cloud sizes where ref (N_ref) != query (N_query).
    Args:
        ref: [B, N_ref, C]
        query: [B, N_query, C]
        k: int
        idx: [B, N_query, k] optional precomputed neighbor indices in ref
        ref_xyz: [B, N_ref, 3] optional reference 3D coordinates
    Returns:
        feature: [B, N_query, k, C] (or tuple with xyz [B, N_query, k, 3] if ref_xyz is provided)
    """
    batch_size, num_ref, num_dims = ref.size()
    num_query = query.size(1)

    if idx is None:
        idx = knn_points(ref, query, k=k)  # [B, N_query, k]

    idx_base = torch.arange(0, batch_size, device=ref.device).view(-1, 1, 1) * num_ref
    idx_flat = (idx + idx_base).view(-1)

    ref_flat = ref.reshape(batch_size * num_ref, -1)
    feature = ref_flat[idx_flat, :].view(batch_size, num_query, k, num_dims)

    if ref_xyz is not None:
        xyz_flat = ref_xyz.reshape(batch_size * num_ref, -1)
        xyz = xyz_flat[idx_flat, :].view(batch_size, num_query, k, 3)
        return feature, xyz

    return feature


class EdgeConvModule(nn.Module):
    def __init__(self, k: int, input_size: int, output_size: int):
        super(EdgeConvModule, self).__init__()
        self.k = k
        self.linear = nn.Conv2d(input_size * 2, output_size, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(output_size)
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x: torch.Tensor, idx=None):
        """
        Args:
            x: [B, C_in, N]
        Returns:
            output: [B, C_out, N]
        """
        x_trans = x.transpose(2, 1)  # [B, N, C_in]
        feature = get_graph_feature(x_trans, x_trans, k=self.k, idx=idx)  # [B, N, k, C_in]
        x_expanded = x_trans.unsqueeze(2).repeat(1, 1, self.k, 1)  # [B, N, k, C_in]
        edge_feat = torch.cat((feature - x_expanded, x_expanded), dim=-1)  # [B, N, k, 2*C_in]
        edge_feat = edge_feat.permute(0, 3, 1, 2).contiguous()  # [B, 2*C_in, N, k]
        edge_feat = self.relu(self.bn(self.linear(edge_feat)))  # [B, C_out, N, k]
        output, _ = edge_feat.max(dim=-1, keepdim=False)  # [B, C_out, N]
        return output


class OrientModule(nn.Module):
    def __init__(self, k: int, input_size: int, output_size: int):
        super(OrientModule, self).__init__()
        self.k = k
        self.linear = nn.Conv2d((input_size + 3) * 2, output_size, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(output_size)
        self.relu = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, xyz_s: torch.Tensor, xyz_t: torch.Tensor, feature_s: torch.Tensor, feature_t: torch.Tensor):
        """
        Extract cross-shape orientation features between source (N_s) and target (N_t).
        Args:
            xyz_s: [B, N_s, 3]
            xyz_t: [B, N_t, 3]
            feature_s: [B, C_in, N_s]
            feature_t: [B, C_in, N_t]
        Returns:
            output: [B, C_out, N_t]
        """
        feature, xyz = get_graph_feature(
            feature_s.transpose(2, 1),
            feature_t.transpose(2, 1),
            k=self.k,
            ref_xyz=xyz_s
        )  # [B, N_t, k, C_in], [B, N_t, k, 3]

        xyz_t_exp = xyz_t.unsqueeze(2).repeat(1, 1, self.k, 1)  # [B, N_t, k, 3]
        feature_t_exp = feature_t.transpose(2, 1).unsqueeze(2).repeat(1, 1, self.k, 1)  # [B, N_t, k, C_in]

        feat_cat = torch.cat(
            (feature - feature_t_exp, feature, xyz - xyz_t_exp, xyz_t_exp),
            dim=-1
        )  # [B, N_t, k, 2*(C_in + 3)]
        feat_cat = feat_cat.permute(0, 3, 1, 2).contiguous()  # [B, 2*(C_in + 3), N_t, k]
        feat_cat = self.relu(self.bn(self.linear(feat_cat)))  # [B, C_out, N_t, k]
        output, _ = feat_cat.max(dim=-1, keepdim=False)  # [B, C_out, N_t]
        return output


class OrientNet(nn.Module):
    """
    Orientation Estimation Module (OEM) from SE-ORNet.
    Predicts relative orientation angle bins and domain logits between two point clouds.
    Fully supports varying vertex counts (N_s != N_t).
    """
    def __init__(
        self,
        input_dims=[3, 64, 128, 256],
        latent_dim=256,
        output_dim=256,
        mlps=[256, 128, 128],
        num_neighs: int = 24,
        input_neighs: int = 27,
        num_class: int = 8,
    ):
        super(OrientNet, self).__init__()
        self.num_neighs = num_neighs
        self.input_neighs = input_neighs
        self.num_class = num_class
        
        # Precompute discrete angle bins (nbins=8: [-pi/4, 0, pi/4, pi/2, 3pi/4, pi, 5pi/4, 3pi/2])
        angles = torch.arange(num_class, dtype=torch.float32) * 2.0 * np.pi / num_class - np.pi / 4.0
        self.register_buffer("ANGLE", angles)

        # 1. Coordinate Feature Encoder E (EdgeConv modules: 3 -> 64 -> 128 -> 256)
        self.input_modules = nn.ModuleList()
        in_dim = input_dims[0]
        for out_d in input_dims[1:]:
            self.input_modules.append(EdgeConvModule(self.input_neighs, in_dim, out_d))
            in_dim = out_d

        # 2. Cross-shape Orientation Module and second EdgeConv
        self.orient_module = OrientModule(self.num_neighs, in_dim, latent_dim)
        self.edgeconv = EdgeConvModule(self.num_neighs, latent_dim, output_dim)

        # 3. Domain Discriminator with Gradient Reversal Layer (GRL)
        self.global_netD = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(128, 64),
            nn.LeakyReLU(negative_slope=0.2),
            nn.Linear(64, 2)
        )

        # 4. Angle Classification MLPs
        self.mlps = nn.ModuleList()
        mlp_in = 2 * (latent_dim + output_dim)
        for dim in mlps:
            self.mlps.append(
                nn.Sequential(
                    nn.Conv1d(mlp_in, dim, kernel_size=1, bias=False),
                    nn.GroupNorm(1, dim),
                    nn.LeakyReLU(negative_slope=0.2),
                )
            )
            mlp_in = dim

        self.classifier = nn.Conv1d(mlp_in, self.num_class, 1, bias=False)

    def forward(self, xyz_s: torch.Tensor, xyz_t: torch.Tensor):
        """
        Forward pass for orientation estimation.
        Args:
            xyz_s: [B, N_s, 3] source point cloud
            xyz_t: [B, N_t, 3] target point cloud
        Returns:
            dict containing:
                angle_x: [B, num_class] orientation logits for source w.r.t target
                angle_y: [B, num_class] orientation logits for target w.r.t source
                d_pred_src: [B, 2] domain prediction logits for source
                d_pred_tgt: [B, 2] domain prediction logits for target
                feat_s: [B, C] global feature descriptor for source
                feat_t: [B, C] global feature descriptor for target
        """
        batch_size = xyz_s.shape[0]
        idx_s = knn_points(xyz_s, xyz_s, k=self.input_neighs)
        idx_t = knn_points(xyz_t, xyz_t, k=self.input_neighs)

        feature_s = xyz_s.transpose(1, 2)  # [B, 3, N_s]
        feature_t = xyz_t.transpose(1, 2)  # [B, 3, N_t]

        for input_module in self.input_modules:
            feature_s = input_module(feature_s, idx=idx_s)  # [B, C, N_s]
            feature_t = input_module(feature_t, idx=idx_t)  # [B, C, N_t]

        # Extract pure individual shape global descriptors for domain discrimination & invariance
        feat_s_global = F.adaptive_max_pool1d(feature_s, 1).view(batch_size, -1)  # [B, 256] clean domain
        feat_t_global = F.adaptive_max_pool1d(feature_t, 1).view(batch_size, -1)  # [B, 256] rotated domain

        d_pred_src = self.global_netD(grad_reverse(feat_s_global))  # [B, 2]
        d_pred_tgt = self.global_netD(grad_reverse(feat_t_global))  # [B, 2]

        latent_s_0 = self.orient_module(xyz_s, xyz_t, feature_s, feature_t)  # [B, 256, N_t]
        latent_t_0 = self.orient_module(xyz_t, xyz_s, feature_t, feature_s)  # [B, 256, N_s]
        latent_s_1 = self.edgeconv(latent_s_0)  # [B, 256, N_t]
        latent_t_1 = self.edgeconv(latent_t_0)  # [B, 256, N_s]

        x = torch.cat((latent_s_0, latent_s_1), dim=1)  # [B, 512, N_t]
        y = torch.cat((latent_t_0, latent_t_1), dim=1)  # [B, 512, N_s]

        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)  # [B, 512]
        y1 = F.adaptive_max_pool1d(y, 1).view(batch_size, -1)  # [B, 512]
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)  # [B, 512]
        y2 = F.adaptive_avg_pool1d(y, 1).view(batch_size, -1)  # [B, 512]

        x = torch.cat((x1, x2), 1).unsqueeze(-1)  # [B, 1024, 1]
        y = torch.cat((y1, y2), 1).unsqueeze(-1)  # [B, 1024, 1]

        # Angle prediction
        for m in self.mlps:
            x = m(x)
            y = m(y)

        angle_x = self.classifier(x).squeeze(-1)  # [B, num_class]
        angle_y = self.classifier(y).squeeze(-1)  # [B, num_class]

        return {
            "angle_x": angle_x,
            "angle_y": angle_y,
            "d_pred_src": d_pred_src,
            "d_pred_tgt": d_pred_tgt,
            "feat_s": feat_s_global,
            "feat_t": feat_t_global,
        }



    def rotate_point_cloud(self, xyz: torch.Tensor, angle_indices: torch.Tensor, inverse: bool = True):
        """
        Rotate point cloud batch along the vertical Y-axis using the predicted angle index.
        Args:
            xyz: [B, N, 3] point cloud coordinates
            angle_indices: [B] discrete angle bin indices
            inverse: If True, rotates by -ANGLE[idx] to align back to canonical frame
        Returns:
            rotated_xyz: [B, N, 3]
        """
        B = xyz.shape[0]
        device = xyz.device
        rotated = torch.zeros_like(xyz)
        
        for b in range(B):
            idx = angle_indices[b]
            rot_angle = -self.ANGLE[idx] if inverse else self.ANGLE[idx]
            cosval = torch.cos(rot_angle)
            sinval = torch.sin(rot_angle)
            
            # Rotation matrix around Y-axis
            R = torch.tensor([
                [cosval, 0.0, sinval],
                [0.0,    1.0, 0.0],
                [-sinval, 0.0, cosval]
            ], dtype=torch.float32, device=device)
            
            pts = xyz[b, :, 0:3]
            centroid = torch.mean(pts, dim=0, keepdim=True)
            pts_centered = pts - centroid
            rotated[b, :, 0:3] = torch.mm(pts_centered, R) + centroid
            
        return rotated


# -------------------------------------------------------------------------
# Data Augmentation Utilities
# -------------------------------------------------------------------------

def rotate_by_y_axis(batch_data: torch.Tensor, nbins: int = 8):
    """
    Rotate point cloud batch along vertical Y-axis by a random discrete angle bin.
    Args:
        batch_data: [B, N, 3]
        nbins: int (default: 8)
    Returns:
        rotated_data: [B, N, 3]
        rotated_gt: [B] ground truth bin index
    """
    B = batch_data.shape[0]
    device = batch_data.device
    angles = torch.arange(nbins, dtype=torch.float32, device=device) * 2.0 * np.pi / nbins - np.pi / 4.0
    rotated_gt = torch.randint(0, nbins, (B,), device=device)
    rot_angles = angles[rotated_gt]
    
    rotated_data = torch.zeros_like(batch_data)
    for b in range(B):
        cosval = torch.cos(rot_angles[b])
        sinval = torch.sin(rot_angles[b])
        R = torch.tensor([
            [cosval, 0.0, sinval],
            [0.0,    1.0, 0.0],
            [-sinval, 0.0, cosval]
        ], dtype=torch.float32, device=device)
        
        pts = batch_data[b, :, 0:3]
        centroid = torch.mean(pts, dim=0, keepdim=True)
        pts_centered = pts - centroid
        rotated_data[b, :, 0:3] = torch.mm(pts_centered, R) + centroid
        
    return rotated_data, rotated_gt


def add_noise_to_pointcloud(batch_data: torch.Tensor, noise_variance: float = 0.0001):
    """Add Gaussian noise to point cloud."""
    noise = torch.randn_like(batch_data) * np.sqrt(noise_variance)
    return batch_data + noise


def scale_pointcloud(batch_data: torch.Tensor, scale_range=[0.95, 1.05]):
    """Scale point cloud by random factor."""
    B = batch_data.shape[0]
    device = batch_data.device
    scales = torch.empty(B, 1, 1, device=device).uniform_(scale_range[0], scale_range[1])
    return batch_data * scales


class DataAugment(nn.Module):
    """
    Point cloud data augmentation for orientation and robustness training.
    """
    def __init__(
        self,
        operations=["rotate", "noise"],
        scale_range=[0.95, 1.05],
        rotate_nbins=8,
        noise_variance=0.0001,
    ):
        super().__init__()
        self.operations = operations
        self.scale_range = scale_range
        self.rotate_nbins = rotate_nbins
        self.noise_variance = noise_variance

    def forward(self, batch_data: torch.Tensor):
        B = batch_data.shape[0]
        rotated_gt = None
        
        if "scale" in self.operations:
            batch_data = scale_pointcloud(batch_data, self.scale_range)
        if "rotate" in self.operations:
            batch_data, rotated_gt = rotate_by_y_axis(batch_data, self.rotate_nbins)
        if "noise" in self.operations:
            batch_data = add_noise_to_pointcloud(batch_data, self.noise_variance)
            
        if "rotate" in self.operations:
            return batch_data, rotated_gt
        else:
            return batch_data
