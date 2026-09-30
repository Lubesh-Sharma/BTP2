import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_rotation_matrix_from_ortho6d(ortho6d):
    """
    Computes a 3x3 rotation matrix from a continuous 6D representation
    (Zhou et al., CVPR 2019: 'On the Continuity of Rotation Representations in Neural Networks').
    
    Args:
        ortho6d: [B, 6] continuous 6D representation
    Returns:
        matrix: [B, 3, 3] valid rotation matrix in SO(3) with det(R) = +1
    """
    x_raw = ortho6d[:, 0:3]
    y_raw = ortho6d[:, 3:6]

    x = F.normalize(x_raw, dim=-1, eps=1e-8)
    z = torch.cross(x, y_raw, dim=-1)
    z = F.normalize(z, dim=-1, eps=1e-8)
    y = torch.cross(z, x, dim=-1)

    matrix = torch.stack([x, y, z], dim=-1)
    return matrix

class OrientationModule(nn.Module):
    """
    SE-ORNet-style Orientation Estimation Module (Deng et al., CVPR 2023).
    
    Predicts a global 3D rotation matrix R in SO(3) from input 3D point cloud
    coordinates to transform the point cloud into a canonical, shared orientation frame.
    
    Uses LayerNorm and InstanceNorm so it is robust to batch size B=1.
    Initializes to the Identity transformation (R = I_3x3).
    """
    def __init__(self, in_channels=3, feat_dim=256):
        super().__init__()
        # Permutation-invariant feature extractor (PointNet-style)
        self.conv1 = nn.Conv1d(in_channels, 64, 1)
        self.norm1 = nn.InstanceNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.norm2 = nn.InstanceNorm1d(128)
        self.conv3 = nn.Conv1d(128, feat_dim, 1)
        self.norm3 = nn.InstanceNorm1d(feat_dim)
        
        # Global MLP head predicting 6D continuous rotation
        self.fc1 = nn.Linear(feat_dim, 128)
        self.norm4 = nn.LayerNorm(128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 6)
        
        # Initialize to Identity transformation (R = I_3x3)
        # Bias [1, 0, 0, 0, 1, 0] produces x=[1, 0, 0], y=[0, 1, 0], z=[0, 0, 1] -> I
        nn.init.constant_(self.fc3.weight, 0)
        self.fc3.bias.data.copy_(torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float32))

    def forward(self, p):
        """
        Args:
            p: [B, N, 3] point cloud coordinates
        Returns:
            R: [B, 3, 3] rotation matrix in SO(3)
        """
        B, N, _ = p.shape
        # Center the point cloud for translation invariance
        centroid = torch.mean(p, dim=1, keepdim=True)
        p_centered = p - centroid
        
        # [B, 3, N]
        x = p_centered.transpose(1, 2)
        x = F.relu(self.norm1(self.conv1(x)))
        x = F.relu(self.norm2(self.conv2(x)))
        x = F.relu(self.norm3(self.conv3(x)))
        
        # Global max pooling across points: [B, feat_dim]
        global_feat = torch.max(x, dim=2)[0]
        
        # MLP head
        net = F.relu(self.norm4(self.fc1(global_feat)))
        net = F.relu(self.fc2(net))
        ortho6d = self.fc3(net)
        
        R = compute_rotation_matrix_from_ortho6d(ortho6d)
        return R

    def align(self, p):
        """
        Predicts R and applies canonical alignment: p_align = (p - center) @ R + center
        Args:
            p: [B, N, 3]
        Returns:
            p_align: [B, N, 3] canonically aligned coordinates
            R: [B, 3, 3] predicted rotation matrix
        """
        centroid = torch.mean(p, dim=1, keepdim=True)
        p_centered = p - centroid
        R = self.forward(p)
        p_align = torch.bmm(p_centered, R) + centroid
        return p_align, R
