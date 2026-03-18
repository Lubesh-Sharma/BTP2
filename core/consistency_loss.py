import torch.nn.functional as F

def compute_consistency_loss(pred_student, pred_teacher):
    """
    Computes the consistency loss (Mean Squared Error) between 
    student's predictions and teacher's predictions.
    """
    f_s = F.normalize(pred_student, dim=-1)
    f_t = F.normalize(pred_teacher, dim=-1)
    return F.mse_loss(f_s, f_t)
