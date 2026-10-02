import math
import torch
import torch.nn.functional as F

def sample_rotation_around_axis(B, device, dtype, angle):
    """
    Constructs a 3x3 rotation matrix around the vertical axis for given batch angles.
    """
    cos_t = torch.cos(angle)
    sin_t = torch.sin(angle)
    zero = torch.zeros_like(cos_t)
    one = torch.ones_like(cos_t)
    
    row0 = torch.stack([cos_t, zero, sin_t], dim=-1)
    row1 = torch.stack([zero, one, zero], dim=-1)
    row2 = torch.stack([-sin_t, zero, cos_t], dim=-1)
    R = torch.stack([row0, row1, row2], dim=1)  # [B, 3, 3]
    return R

def compute_orientation_loss(student_ori, p1, p2, teacher_ori=None, num_bins=8, sigma_noise=0.02):
    """
    SE-ORNet / DV-Matcher Relative Orientation Angle Loss (Deng et al., CVPR 2023).
    
    Supervises the OrientationModule using discrete angle classification:
    
    1. Sample random target angle bin b in {0, ..., num_bins-1} covering [0, 2pi).
    2. Synthetically rotate Shape 1 by theta(b) + Gaussian noise: p1_rot = (p1 - c) @ R.T + c + n.
    3. Pass (p1_rot, p1) into the OrientationModule to predict relative rotation logits:
       L_self = CrossEntropy(logits_self, target_bin).
    4. Reverse self-relative rotation: (p1, p1_rot) -> (num_bins - target_bin) % num_bins.
    5. Pass (p1_rot, p2) into the OrientationModule to predict cross-shape relative rotation:
       L_cross = CrossEntropy(logits_cross, target_bin).
    6. Pass (p1, p2) unrotated to enforce identity alignment (bin 0):
       L_id = CrossEntropy(logits_id, 0).
       
    The loss directly trains down from ln(num_bins) (~2.08) down to < 0.1 as the
    orientation classifier learns to detect relative 3D orientations accurately.
    
    Args:
        student_ori: student.orientation_module (OrientationModule)
        p1: [B, N1, 3] coordinates of Shape 1
        p2: [B, N2, 3] coordinates of Shape 2
        teacher_ori: optional teacher orientation module (for consistency)
        num_bins: number of discrete angle bins (default: 8)
        sigma_noise: standard deviation of Gaussian noise perturbation
        
    Returns:
        total_loss: scalar Cross-Entropy orientation loss
    """
    B = p1.shape[0]
    device = p1.device
    dtype = p1.dtype
    
    # 1. Sample synthetic relative rotation bin and angle
    bin_target = torch.randint(0, num_bins, (B,), device=device)
    bin_width = 2.0 * math.pi / num_bins
    # Add small jitter within the bin for continuous coverage
    angle = bin_target.to(dtype) * bin_width + (torch.rand(B, device=device, dtype=dtype) - 0.5) * 0.4 * bin_width
    
    R = sample_rotation_around_axis(B, device, dtype, angle)
    
    # Synthetically rotate Shape 1 around its centroid with small noise
    c1 = torch.mean(p1, dim=1, keepdim=True)
    noise = torch.randn_like(p1) * sigma_noise if sigma_noise > 0 else 0.0
    p1_rot = torch.bmm(p1 - c1, R.transpose(1, 2)) + c1 + noise
    
    # 2. Self-Relative Rotation Loss: p1_rot relative to p1 should be bin_target
    logits_self = student_ori(p1_rot, p1)
    loss_self = F.cross_entropy(logits_self, bin_target)
    
    # Reverse self-relative rotation: p1 relative to p1_rot should be (num_bins - bin_target) % num_bins
    bin_rev = (num_bins - bin_target) % num_bins
    logits_rev = student_ori(p1, p1_rot)
    loss_rev = F.cross_entropy(logits_rev, bin_rev)
    
    # 3. Cross-Relative Rotation Loss: p1_rot relative to p2 should also be bin_target
    logits_cross = student_ori(p1_rot, p2)
    loss_cross = F.cross_entropy(logits_cross, bin_target)
    
    # 4. Identity Alignment Loss: unrotated pair (p1, p2) should predict bin 0 (0 degrees)
    logits_id = student_ori(p1, p2)
    bin_zero = torch.zeros(B, dtype=torch.long, device=device)
    loss_id = F.cross_entropy(logits_id, bin_zero)
    
    total_loss = (loss_self + loss_rev + loss_cross + loss_id) / 4.0
    
    # Optional Teacher consistency (SE-ORNet self-ensembling)
    if teacher_ori is not None:
        with torch.no_grad():
            teacher_logits = teacher_ori(p1_rot, p1)
        loss_teacher = F.kl_div(
            F.log_softmax(logits_self, dim=-1),
            F.softmax(teacher_logits, dim=-1),
            reduction='batchmean'
        )
        total_loss = total_loss + 0.5 * loss_teacher
        
    return total_loss
