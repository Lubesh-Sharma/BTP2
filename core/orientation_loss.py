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
    bin_width = 2.0 * math.pi / num_bins
    
    # 1. Synthetically rotate Shape 1 by random angle bin1
    bin1 = torch.randint(0, num_bins, (B,), device=device)
    angle1 = bin1.to(dtype) * bin_width + (torch.rand(B, device=device, dtype=dtype) - 0.5) * 0.4 * bin_width
    R1 = sample_rotation_around_axis(B, device, dtype, angle1)
    c1 = torch.mean(p1, dim=1, keepdim=True)
    noise1 = torch.randn_like(p1) * sigma_noise if sigma_noise > 0 else 0.0
    p1_rot = torch.bmm(p1 - c1, R1.transpose(1, 2)) + c1 + noise1

    logits1 = student_ori(p1_rot, p1)
    loss1 = F.cross_entropy(logits1, bin1)
    logits1_rev = student_ori(p1, p1_rot)
    loss1_rev = F.cross_entropy(logits1_rev, (num_bins - bin1) % num_bins)

    # 2. Synthetically rotate Shape 2 by random angle bin2
    bin2 = torch.randint(0, num_bins, (B,), device=device)
    angle2 = bin2.to(dtype) * bin_width + (torch.rand(B, device=device, dtype=dtype) - 0.5) * 0.4 * bin_width
    R2 = sample_rotation_around_axis(B, device, dtype, angle2)
    c2 = torch.mean(p2, dim=1, keepdim=True)
    noise2 = torch.randn_like(p2) * sigma_noise if sigma_noise > 0 else 0.0
    p2_rot = torch.bmm(p2 - c2, R2.transpose(1, 2)) + c2 + noise2

    logits2 = student_ori(p2_rot, p2)
    loss2 = F.cross_entropy(logits2, bin2)
    logits2_rev = student_ori(p2, p2_rot)
    loss2_rev = F.cross_entropy(logits2_rev, (num_bins - bin2) % num_bins)

    # 3. Identity alignment (unrotated self-pair with small noise)
    p1_noisy = p1 + torch.randn_like(p1) * sigma_noise
    bin_zero = torch.zeros(B, dtype=torch.long, device=device)
    loss_id1 = F.cross_entropy(student_ori(p1_noisy, p1), bin_zero)
    p2_noisy = p2 + torch.randn_like(p2) * sigma_noise
    loss_id2 = F.cross_entropy(student_ori(p2_noisy, p2), bin_zero)

    # 4. Cross-Shape Relative Rotation Equivariance (Shape-Agnostic)
    # If p1 is rotated by bin1, the predicted relative orientation (p1_rot, p2)
    # must be exactly the unrotated prediction (p1, p2) shifted by bin1.
    logits_cross_unrot = student_ori(p1, p2)
    logits_cross_rot = student_ori(p1_rot, p2)
    # Target distribution is the unrotated prediction rolled by bin1
    probs_cross_unrot = F.softmax(logits_cross_unrot.detach(), dim=-1)
    target_shifted = torch.zeros_like(probs_cross_unrot)
    for b_idx in range(B):
        target_shifted[b_idx] = torch.roll(probs_cross_unrot[b_idx], shifts=int(bin1[b_idx].item()), dims=0)
    loss_equiv = F.kl_div(
        F.log_softmax(logits_cross_rot, dim=-1),
        target_shifted,
        reduction='batchmean'
    )

    total_loss = (loss1 + loss1_rev + loss2 + loss2_rev + loss_id1 + loss_id2 + 2.0 * loss_equiv) / 8.0
    
    # Optional Teacher consistency (SE-ORNet self-ensembling)
    if teacher_ori is not None:
        with torch.no_grad():
            teacher_logits1 = teacher_ori(p1_rot, p1)
            teacher_logits2 = teacher_ori(p2_rot, p2)
        loss_teacher1 = F.kl_div(
            F.log_softmax(logits1, dim=-1),
            F.softmax(teacher_logits1, dim=-1),
            reduction='batchmean'
        )
        loss_teacher2 = F.kl_div(
            F.log_softmax(logits2, dim=-1),
            F.softmax(teacher_logits2, dim=-1),
            reduction='batchmean'
        )
        total_loss = total_loss + 0.25 * (loss_teacher1 + loss_teacher2)
        
    return total_loss
