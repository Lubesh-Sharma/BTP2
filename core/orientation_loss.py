import math
import torch
import torch.nn.functional as F

def rotate_point_cloud_by_angle(batch_data: torch.Tensor, angles: torch.Tensor):
    """
    Rotates batch of point clouds around vertical Y-axis by given angles.
    Args:
        batch_data: [B, N, 3] Point clouds
        angles: [B] Rotation angles in radians
    Returns:
        rotated_data: [B, N, 3] Rotated point clouds centered at their centroids
        R: [B, 3, 3] Rotation matrices
    """
    cos_t = torch.cos(angles)
    sin_t = torch.sin(angles)
    zero = torch.zeros_like(cos_t)
    one = torch.ones_like(cos_t)

    # [B, 3, 3] rotation matrix around Y
    row0 = torch.stack([cos_t, zero, sin_t], dim=-1)
    row1 = torch.stack([zero, one, zero], dim=-1)
    row2 = torch.stack([-sin_t, zero, cos_t], dim=-1)
    R = torch.stack([row0, row1, row2], dim=1)

    centroid = torch.mean(batch_data, dim=1, keepdim=True)
    rotated_data = torch.bmm(batch_data - centroid, R) + centroid
    return rotated_data, R

def compute_orientation_loss(student_ori, p1, p2, teacher_ori=None, num_bins=None, sigma_noise=0.005):
    """
    SE-ORNet Relative Angle Loss (Deng et al., CVPR 2023).
    
    Supervises the OrientationModule using discrete angle classification into M=8 bins
    with bidirectional cyclic consistency (angle_x vs angle_y):
      - Angle codebook: ANGLE = m * (2*pi / 8) - pi / 4
      - Identity bin: bin 1 = 0.0 degrees (canonical forward-facing)
      - Inverse angle mapping: rev_gt = (10 - gt) % 8
    
    Multi-objective supervision:
      1. Self-Rotation (P2 -> P2_rot): Exact rigid transformation ground-truth with 0 pose noise (SE-ORNet core).
      2. Identity Pair (P1 -> P2): Canonical forward-facing alignment (gt = bin 1 = 0°).
         Directly prevents front-to-back 180° flip between front-facing dataset shapes.
      3. Cross-Shape Rotation (P1 -> P2_rot): Invariance to cross-shape relative yaw.
    """
    if num_bins is None:
        num_bins = getattr(student_ori, 'num_bins', 8)
    device = p1.device
    B = p1.shape[0]
    identity_bin = 1 if num_bins == 8 else 0

    # Sample random ground-truth discrete rotation bins
    gt_bins = torch.randint(0, num_bins, (B,), device=device)
    angles = student_ori.angles[gt_bins]

    # SE-ORNet inverse ground-truth formula
    if num_bins == 8:
        rev_bins = (10 - gt_bins) % 8
    else:
        rev_bins = (num_bins - gt_bins) % num_bins

    # Rotate target point cloud by ground-truth angle (SE-ORNet data augmentation)
    p2_rot, _ = rotate_point_cloud_by_angle(p2, angles)

    if sigma_noise > 0:
        p2_rot = p2_rot + torch.randn_like(p2_rot) * sigma_noise

    # --- 1. Pure Self-Rotation Loss (SE-ORNet core: P2 vs P2_rot) ---
    # Unambiguous rigid 3D rotation supervision with zero non-rigid deformation noise.
    out_self = student_ori(p2, p2_rot, return_dict=True)
    l_self_x = F.cross_entropy(out_self['angle_x'], gt_bins)
    l_self_y = F.cross_entropy(out_self['angle_y'], rev_bins)
    loss_self = 0.5 * (l_self_x + l_self_y)

    # --- 2. Identity Canonical Alignment Loss (P1 vs P2) ---
    # Both P1 and P2 in the dataset are canonical front-facing shapes (relative angle = 0°).
    # Explicitly teaches the network to predict bin 1 (0°) and strictly prevents 180° flips!
    id_targets = torch.full((B,), identity_bin, dtype=torch.long, device=device)
    out_ident = student_ori(p1, p2, return_dict=True)
    l_ident_x = F.cross_entropy(out_ident['angle_x'], id_targets)
    l_ident_y = F.cross_entropy(out_ident['angle_y'], id_targets)
    loss_ident = 0.5 * (l_ident_x + l_ident_y)

    # --- 3. Cross-Shape Rotated Loss (P1 vs P2_rot) ---
    out_cross = student_ori(p1, p2_rot, return_dict=True)
    l_cross_x = F.cross_entropy(out_cross['angle_x'], gt_bins)
    l_cross_y = F.cross_entropy(out_cross['angle_y'], rev_bins)
    loss_cross = 0.5 * (l_cross_x + l_cross_y)

    total_loss = 0.4 * loss_self + 0.3 * loss_ident + 0.3 * loss_cross

    # Optional Teacher Consistency (Self-Ensembling)
    if teacher_ori is not None:
        with torch.no_grad():
            out_teacher = teacher_ori(p2, p2_rot, return_dict=True)
        l_cons_x = F.kl_div(
            F.log_softmax(out_self['angle_x'], dim=-1),
            F.softmax(out_teacher['angle_x'], dim=-1),
            reduction='batchmean'
        )
        total_loss = total_loss + 0.1 * l_cons_x

    return total_loss
