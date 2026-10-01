import math
import torch
import torch.nn.functional as F

def sample_dataset_rotation(B, device, dtype=torch.float32, jitter_deg=10.0):
    """
    Samples physically realistic 3D rotations for upright humanoid/animal meshes:
    - Continuous yaw rotation around the vertical (Y) axis in [-pi, pi], which
      directly captures the bilateral front-back 180-degree flip ambiguity.
    - Small jitter (pitch and roll) around horizontal axes (default +-10 degrees)
      to ensure robustness to natural pose tilting.
    """
    theta = (torch.rand(B, device=device, dtype=dtype) * 2.0 - 1.0) * math.pi
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
    SE-ORNet Self-Supervised Orientation Loss (Deng et al., CVPR 2023).
    
    Supervises the OrientationModule to resolve 3D spatial/reflection/180-degree
    rotational ambiguity on upright meshes:
    
    1. Rotation Equivariance:
       For realistic rotation Q (yaw in [-pi, pi] + pitch/roll jitter), if P_rot = (P - c) @ Q + c:
       The module must predict R_rot such that:
         Q @ R_rot = R_s  <=>  R_rot_target = Q.T @ R_s
         || R_rot - Q.T @ R_s.detach() ||_F^2 -> 0
         
    2. Self-Ensembling Consistency (Student-Teacher):
       The student rotation is stabilized by penalizing deviation from the EMA teacher:
         || R_student - R_teacher.detach() ||_F^2 -> 0
    
    Args:
        student_ori: student.orientation_module (OrientationModule)
        teacher_ori: teacher.orientation_module (OrientationModule or None)
        p1: [B, N1, 3] coordinates of Shape 1
        p2: [B, N2, 3] coordinates of Shape 2
        
    Returns:
        loss_orient: scalar orientation loss in [0, 1]
    """
    B = p1.shape[0]
    device = p1.device
    
    total_loss = 0.0
    shapes = [p1, p2]
    
    for p in shapes:
        c = torch.mean(p, dim=1, keepdim=True)
        p_c = p - c
        
        # 1. Student rotation on original shape
        R_s = student_ori(p)
        
        # 2. Sample realistic rotation containing yaw and 180-deg flip
        Q = sample_dataset_rotation(B, device, dtype=p.dtype, jitter_deg=10.0)
        p_rot = torch.bmm(p_c, Q) + c
        
        # Student rotation on rotated shape
        R_rot = student_ori(p_rot)
        
        # Equivariance target: Q @ R_rot = R_s => R_rot_target = Q.T @ R_s
        R_target = torch.bmm(Q.transpose(1, 2), R_s.detach())
        loss_rot = F.mse_loss(R_rot, R_target)
        
        # 3. Student-Teacher consistency loss
        if teacher_ori is not None:
            with torch.no_grad():
                R_t = teacher_ori(p)
            loss_teacher = F.mse_loss(R_s, R_t.detach())
        else:
            loss_teacher = 0.0
            
        shape_loss = loss_rot + loss_teacher
        total_loss = total_loss + shape_loss
        
    return total_loss / len(shapes)
