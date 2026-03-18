import torch
import torch.nn.functional as F

def compute_contrastive_loss(feat_student, feat_teacher, temperature=0.5):
    """
    Computes InfoNCE contrastive loss by matching the point embeddings
    of the student and the teacher networks.
    
    Assumes features are of shape [B, N, C].
    """
    B, N, C = feat_student.shape
    loss = 0.0
    for b in range(B):
        # Normalize features
        f_s = F.normalize(feat_student[b], dim=-1) # [N, C]
        f_t = F.normalize(feat_teacher[b], dim=-1) # [N, C]
        
        # Cross-correlation matrix between student and teacher points
        sim_matrix = torch.matmul(f_s, f_t.T) / temperature # [N, N]
        
        # Self-contrastive labels (Point i of student matches Point i of teacher)
        labels = torch.arange(N, device=feat_student.device)
        
        loss += F.cross_entropy(sim_matrix, labels)
        
    return loss / max(B, 1)
