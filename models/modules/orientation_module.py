import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


@torch.no_grad()
def farthest_point_indices(points, count, generator=None):
    """FPS without replacement; random initial vertex, then farthest coverage."""
    batch, size, _ = points.shape
    indices = torch.empty((batch, count), dtype=torch.long, device=points.device)
    distances = torch.full((batch, size), float('inf'), device=points.device)
    selected = torch.zeros((batch, size), dtype=torch.bool, device=points.device)
    current = torch.randint(size, (batch,), device=points.device, generator=generator)
    rows = torch.arange(batch, device=points.device)
    for step in range(count):
        indices[:, step] = current
        selected[rows, current] = True
        delta = points - points[rows, current].unsqueeze(1)
        distances = torch.minimum(distances, (delta * delta).sum(-1))
        current = distances.masked_fill(selected, -1).argmax(-1)
    return indices


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
        self.bn = nn.GroupNorm(1, output_size)
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
        self.bn = nn.GroupNorm(1, output_size)
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
    Predicts relative orientation angle bins and pair-domain logits between two point clouds.
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
        sample_points=None,
        sampling_seed: int = 2026,
        sampling_method: str = 'random',
        sample_fraction=None,
        sample_min_points: int = 1024,
    ):
        super(OrientNet, self).__init__()
        self.num_neighs = num_neighs
        self.input_neighs = input_neighs
        self.num_class = num_class
        if sample_points is not None and sample_points < max(num_neighs, input_neighs):
            raise ValueError("OEM sample_points must cover its largest KNN neighborhood.")
        self.sample_points = sample_points
        self.sampling_seed = sampling_seed
        if sampling_method not in ('random', 'fps'):
            raise ValueError('sampling_method must be random or fps')
        if sample_fraction is not None and not 0 < sample_fraction <= 1:
            raise ValueError('sample_fraction must be in (0, 1]')
        if sample_min_points < max(num_neighs, input_neighs):
            raise ValueError('sample_min_points must cover KNN neighborhoods')
        self.sampling_method = sampling_method
        self.sample_fraction = sample_fraction
        self.sample_min_points = sample_min_points
        
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

        # Pair discriminator: same-shape reference pairs versus cross-shape pairs.
        pair_dim = 2 * (latent_dim + output_dim)
        self.global_netD1 = nn.Sequential(
            nn.Conv1d(pair_dim, 256, 1),
            nn.GroupNorm(1, 256),
            nn.ReLU(),
            nn.Conv1d(256, 128, 1),
            nn.GroupNorm(1, 128),
            nn.ReLU(),
        )
        self.global_netD2 = nn.Linear(128, 2)

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

    def sample_pair(self, xyz_s, xyz_t):
        """Independent OEM subsets; never mutate, reorder, or truncate caller inputs.

        Training draws fresh subsets. Evaluation uses private fixed seeds for
        reproducible subsets without consuming the training random stream.
        """
        def sample(points, stream):
            count = self.sample_points
            if self.sample_fraction is not None:
                count = max(self.sample_min_points, math.ceil(self.sample_fraction * points.shape[1]))
            if count is None or points.shape[1] <= count:
                return points
            generator = None
            if not self.training:
                generator = torch.Generator(device=points.device).manual_seed(self.sampling_seed + stream)
            if self.sampling_method == 'fps':
                indices = farthest_point_indices(points, count, generator)
            else:
                indices = torch.stack([
                    torch.randperm(points.shape[1], device=points.device, generator=generator)[:count]
                    for _ in range(points.shape[0])
                ])
            return points.gather(1, indices.unsqueeze(-1).expand(-1, -1, points.shape[-1]))
        return sample(xyz_s, 0), sample(xyz_t, 1)

    def forward(self, xyz_s: torch.Tensor, xyz_t: torch.Tensor, sample_inputs=True):
        """
        Forward pass for orientation estimation.
        Args:
            xyz_s: [B, N_s, 3] source point cloud
            xyz_t: [B, N_t, 3] target point cloud
        Returns:
            dict containing:
                angle_x: [B, num_class] orientation logits for source w.r.t target
                angle_y: [B, num_class] orientation logits for target w.r.t source
                global_d_pred: [B, 2] pair-domain prediction logits
        """
        if sample_inputs:
            xyz_s, xyz_t = self.sample_pair(xyz_s, xyz_t)
        batch_size = xyz_s.shape[0]
        idx_s = knn_points(xyz_s, xyz_s, k=self.input_neighs)
        idx_t = knn_points(xyz_t, xyz_t, k=self.input_neighs)

        feature_s = xyz_s.transpose(1, 2)  # [B, 3, N_s]
        feature_t = xyz_t.transpose(1, 2)  # [B, 3, N_t]

        for input_module in self.input_modules:
            feature_s = input_module(feature_s, idx=idx_s)  # [B, C, N_s]
            feature_t = input_module(feature_t, idx=idx_t)  # [B, C, N_t]

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

        pair_features = torch.cat((x, y), dim=-1)  # [B, pair_dim, 2]
        domain_features = self.global_netD1(grad_reverse(pair_features)).mean(dim=-1)
        global_d_pred = self.global_netD2(domain_features)

        # Angle prediction
        for m in self.mlps:
            x = m(x)
            y = m(y)

        angle_x = self.classifier(x).squeeze(-1)  # [B, num_class]
        angle_y = self.classifier(y).squeeze(-1)  # [B, num_class]

        return {
            "angle_x": angle_x,
            "angle_y": angle_y,
            "global_d_pred": global_d_pred,
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

def nearest_angle_bins(radians, angle_centers):
    """Nearest bin by circular distance; ties choose the first bin index."""
    delta = radians.unsqueeze(-1) - angle_centers.to(radians)
    return torch.atan2(delta.sin(), delta.cos()).abs().argmin(dim=-1)


def rotate_by_y_axis(batch_data: torch.Tensor, nbins: int = 8):
    """Apply a random exact bin-center Y rotation and return its bin label."""
    centers = torch.arange(nbins, device=batch_data.device, dtype=batch_data.dtype)
    centers = centers * (2 * math.pi / nbins) - math.pi / 4
    labels = torch.randint(nbins, (batch_data.shape[0],), device=batch_data.device)
    radians = centers[labels]
    c, s = radians.cos(), radians.sin()
    zero, one = torch.zeros_like(c), torch.ones_like(c)
    matrix = torch.stack((c, zero, s, zero, one, zero, -s, zero, c), -1).reshape(-1, 3, 3)
    center = batch_data.mean(dim=1, keepdim=True)
    return torch.bmm(batch_data - center, matrix) + center, labels


def add_noise_to_pointcloud(batch_data: torch.Tensor, noise_variance: float = 0.0001,
                            relative_to_radius: bool = False, generator=None):
    """Gaussian noise, optionally scaled by each cloud's centered maximum radius."""
    if noise_variance < 0:
        raise ValueError("noise_variance cannot be negative.")
    if noise_variance == 0:
        return batch_data
    scale = 1.0
    if relative_to_radius:
        centered = batch_data - batch_data.mean(dim=1, keepdim=True)
        scale = centered.norm(dim=-1).amax(dim=1, keepdim=True).unsqueeze(-1)
    noise = torch.randn(batch_data.shape, dtype=batch_data.dtype, device=batch_data.device,
                        generator=generator) * noise_variance ** 0.5
    return batch_data + noise * scale


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
        noise_relative_to_radius=False,
    ):
        super().__init__()
        self.operations = operations
        self.scale_range = scale_range
        self.rotate_nbins = rotate_nbins
        self.noise_variance = noise_variance
        self.noise_relative_to_radius = noise_relative_to_radius

    def forward(self, batch_data: torch.Tensor):
        B = batch_data.shape[0]
        rotated_gt = None
        
        if "scale" in self.operations:
            batch_data = scale_pointcloud(batch_data, self.scale_range)
        if "rotate" in self.operations:
            batch_data, rotated_gt = rotate_by_y_axis(batch_data, self.rotate_nbins)
        if "noise" in self.operations:
            batch_data = add_noise_to_pointcloud(batch_data, self.noise_variance, self.noise_relative_to_radius)
            
        if "rotate" in self.operations:
            return batch_data, rotated_gt
        else:
            return batch_data
