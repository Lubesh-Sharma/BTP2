import math
import torch
import torch.nn.functional as F

def sample_orientation_perturbation(B, device, dtype=torch.float32, max_yaw_deg=20.0, jitter_deg=5.0, flip_prob=0.5):
    """
    Samples physically realistic 3D orientation perturbations for upright meshes:
    - Bounded yaw perturbation around vertical (Y) axis within [-max_yaw_deg, +max_yaw_deg].
    - Discrete 180-degree front/back flip with probability flip_prob to actively train the
      OrientationModule to shatter the 180-degree bilateral/front-back ambiguity.
    - Small pitch and roll jitter around horizontal axes within [-jitter_deg, +jitter_deg].
    """
    max_yaw_rad = math.radians(max_yaw_deg)
    theta = (torch.rand(B, device=device, dtype=dtype) * 2.0 - 1.0) * max_yaw_rad
    if flip_prob > 0:
        flips = (torch.rand(B, device=device) < flip_prob).to(dtype) * math.pi
        theta = theta + flips
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    zero = torch.zeros_like(cos_t)
    one = torch.ones_like(cos_t)
    
    # R_y(theta): [cos, 0, sin; 0, 1, 0; -sin, 0, cos]
    row0 = torch.stack([cos_t, zero, sin_t], dim=-1)
    row1 = torch.stack([zero, one, zero], dim=-1)
    row2 = torch.stack([-sin_t, zero, cos_t], dim=-1)
    R_y = torch.stack([row0, row1, row2], dim=1)  # [B, 3, 3]
    
    if jitter_deg > 0:
        max_rad = math.radians(jitter_deg)
        rx = (torch.rand(B, device=device, dtype=dtype) * 2.0 - 1.0) * max_rad
        rz = (torch.rand(B, device=device, dtype=dtype) * 2.0 - 1.0) * max_rad
        cx, sx = torch.cos(rx), torch.sin(rx)
        cz, sz = torch.cos(rz), torch.sin(rz)
        
        # Pitch around X
        Rx_row0 = torch.stack([one, zero, zero], dim=-1)
        Rx_row1 = torch.stack([zero, cx, -sx], dim=-1)
        Rx_row2 = torch.stack([zero, sx, cx], dim=-1)
        Rx = torch.stack([Rx_row0, Rx_row1, Rx_row2], dim=1)
        
        # Roll around Z
        Rz_row0 = torch.stack([cz, -sz, zero], dim=-1)
        Rz_row1 = torch.stack([sz, cz, zero], dim=-1)
        Rz_row2 = torch.stack([zero, zero, one], dim=-1)
        Rz = torch.stack([Rz_row0, Rz_row1, Rz_row2], dim=1)
        
        Q = torch.bmm(Rz, torch.bmm(Rx, R_y))
    else:
        Q = R_y
    return Q

def compute_orientation_loss(student_ori, teacher_ori, p1, p2):
    """
    SE-ORNet Orientation Loss (Deng et al., CVPR 2023).
    
    Supervises the OrientationModule to canonicalize 3D orientations
    and maintain consistency across paired shapes and under perturbation:
    
    1. Cross-Shape Alignment:
       Both shape 1 and shape 2 are guided to share the exact same canonical frame:
         || R_1 - R_2 ||_F^2 -> 0
         
    2. Perturbation Equivariance:
       Under physical orientation perturbation Q, the module predicts R_rot such that:
         Q @ R_rot = R_s  <=>  || R_rot - Q.T @ R_s.detach() ||_F^2 -> 0
         
    3. Self-Ensembling Consistency (Student-Teacher):
       The student canonical rotation is stabilized by the smooth EMA teacher:
         0.5 * (|| R_1 - R_1_teacher.detach() ||_F^2 + || R_2 - R_2_teacher.detach() ||_F^2) -> 0
    
    Args:
        student_ori: student.orientation_module (OrientationModule)
        teacher_ori: teacher.orientation_module (OrientationModule or None)
        p1: [B, N1, 3] coordinates of Shape 1
        p2: [B, N2, 3] coordinates of Shape 2
        
    Returns:
        loss_orient: scalar orientation loss (~0.005 - 0.02)
    """
    B = p1.shape[0]
    device = p1.device
    dtype = p1.dtype
    
    # 1. Predict rotations for both shapes
    R1 = student_ori(p1)
    R2 = student_ori(p2)
    
    # Cross-shape canonical frame alignment
    loss_cross = F.mse_loss(R1, R2)
    
    # 2. Perturbation equivariance for both shapes
    loss_equiv = 0.0
    for p, R_s in [(p1, R1), (p2, R2)]:
        c = torch.mean(p, dim=1, keepdim=True)
        p_c = p - c
        Q = sample_orientation_perturbation(B, device, dtype=dtype, max_yaw_deg=20.0, jitter_deg=5.0)
        p_rot = torch.bmm(p_c, Q) + c
        R_rot = student_ori(p_rot)
        R_target = torch.bmm(Q.transpose(1, 2), R_s.detach())
        loss_equiv = loss_equiv + F.mse_loss(R_rot, R_target)
    loss_equiv = loss_equiv / 2.0
    
    # 3. Student-Teacher consistency
    if teacher_ori is not None:
        with torch.no_grad():
            R1_t = teacher_ori(p1)
            R2_t = teacher_ori(p2)
        loss_teacher = 0.5 * (F.mse_loss(R1, R1_t.detach()) + F.mse_loss(R2, R2_t.detach()))
    else:
        loss_teacher = 0.0
        
    total_loss = loss_cross + loss_equiv + loss_teacher
    return total_loss
