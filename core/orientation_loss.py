import torch
import torch.nn.functional as F

def random_rotation_matrix(B, device, dtype=torch.float32):
    """
    Generates uniform random 3D rotation matrices in SO(3) with det(R) = +1.
    Uses QR decomposition with sign correction to guarantee valid orthogonal rotation.
    """
    A = torch.randn(B, 3, 3, device=device, dtype=dtype)
    Q, R = torch.linalg.qr(A)
    diag = torch.diagonal(R, dim1=-2, dim2=-1)
    sign = torch.sign(diag)
    sign[sign == 0] = 1.0
    Q = Q * sign.unsqueeze(-2)
    det = torch.linalg.det(Q)
    col0 = Q[:, :, 0] * det.unsqueeze(-1)
    Q = torch.stack([col0, Q[:, :, 1], Q[:, :, 2]], dim=-1)
    return Q

def compute_orientation_loss(student_ori, teacher_ori, p1, p2):
    """
    SE-ORNet Orientation Loss (Deng et al., CVPR 2023).
    
    Supervises the OrientationModule to resolve 3D spatial/reflection/180-degree
    rotational ambiguity without manual annotations by enforcing:
    
    1. Rotation Equivariance:
       For random 3D rotation Q in SO(3), if P_rot = (P - c) @ Q + c:
       The module must predict R_rot such that:
         R_rot @ Q.T = R  <=>  || R_rot - Q.T @ R ||_F^2 -> 0
         and canonically aligned points match:
         || (P_rot - c) @ R_rot - (P - c) @ R ||_2^2 -> 0
         
    2. Self-Ensembling Consistency (Student-Teacher):
       The student rotation is stabilized by penalizing deviation from the EMA teacher:
         || R_student - R_teacher.detach() ||_F^2 -> 0
    
    Args:
        student_ori: student.orientation_module (OrientationModule)
        teacher_ori: teacher.orientation_module (OrientationModule or None)
        p1: [B, N1, 3] coordinates of Shape 1
        p2: [B, N2, 3] coordinates of Shape 2
        
    Returns:
        loss_orient: scalar orientation loss
    """
    B = p1.shape[0]
    device = p1.device
    
    total_loss = 0.0
    shapes = [p1, p2]
    
    for p in shapes:
        # Centroid
        c = torch.mean(p, dim=1, keepdim=True)
        p_c = p - c
        
        # 1. Student rotation on original shape
        R_s = student_ori(p)
        p_canon_s = torch.bmm(p_c, R_s.detach())
        
        # 2. Sample random rotation Q in SO(3)
        Q = random_rotation_matrix(B, device, dtype=p.dtype)
        p_rot = torch.bmm(p_c, Q) + c
        p_rot_c = p_rot - c
        
        # Student rotation on randomly rotated shape
        R_rot = student_ori(p_rot)
        p_canon_rot = torch.bmm(p_rot_c, R_rot)
        
        # Equivariance target: (P_c @ Q) @ R_rot = P_c @ R_s
        # => Q @ R_rot = R_s => R_rot_target = Q.T @ R_s
        R_target = torch.bmm(Q.transpose(1, 2), R_s.detach())
        
        # Rotation matrix equivariance loss
        loss_rot = F.mse_loss(R_rot, R_target)
        
        # Canonical coordinate alignment loss
        loss_coord = F.mse_loss(p_canon_rot, p_canon_s)
        
        # 3. Student-Teacher consistency loss (if teacher is available)
        if teacher_ori is not None:
            with torch.no_grad():
                R_t = teacher_ori(p)
            loss_teacher = F.mse_loss(R_s, R_t.detach())
        else:
            loss_teacher = 0.0
            
        shape_ori_loss = loss_rot + loss_coord + loss_teacher
        total_loss = total_loss + shape_ori_loss
        
    return total_loss / len(shapes)
