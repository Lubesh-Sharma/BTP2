import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for classification / domain discrimination.
    """
    def __init__(self, class_num=2, alpha=None, gamma=3, size_average=True):
        super(FocalLoss, self).__init__()
        self.class_num = class_num
        self.gamma = gamma
        self.size_average = size_average
        if alpha is None:
            self.alpha = torch.ones(class_num, 1)
        else:
            if isinstance(alpha, torch.Tensor):
                self.alpha = alpha
            else:
                self.alpha = torch.tensor(alpha, dtype=torch.float32)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor):
        """
        Args:
            inputs: [B, C] logits
            targets: [B] class labels
        Returns:
            loss: scalar tensor
        """
        N = inputs.size(0)
        C = inputs.size(1)
        device = inputs.device
        
        P = F.softmax(inputs, dim=-1)
        class_mask = torch.zeros((N, C), dtype=torch.float32, device=device)
        class_mask.scatter_(1, targets.view(-1, 1), 1.0)

        alpha = self.alpha.to(device)[targets.view(-1)]
        probs = (P * class_mask).sum(dim=1, keepdim=True)
        log_p = torch.log(probs.clamp(min=1e-8))
        batch_loss = -alpha * (torch.pow(1.0 - probs, self.gamma)) * log_p

        if self.size_average:
            return batch_loss.mean()
        else:
            return batch_loss.sum()


def compute_angle_loss(angle_pred_x: torch.Tensor, angle_pred_y: torch.Tensor, rotated_gt: torch.Tensor):
    """
    Compute cross-entropy loss for angle predictions.
    Args:
        angle_pred_x: [B, 8] logits for angle of source w.r.t target
        angle_pred_y: [B, 8] logits for angle of target w.r.t source
        rotated_gt: [B] ground truth rotation angle bin index
    Returns:
        loss_angle: scalar tensor
    """
    angle_pred_combined = torch.cat([angle_pred_x, angle_pred_y], dim=0)  # [2*B, 8]
    rotated_gt_combined = torch.cat([rotated_gt, (10 - rotated_gt) % 8], dim=0)  # [2*B]
    loss_angle = F.cross_entropy(angle_pred_combined, rotated_gt_combined)
    return loss_angle


def compute_domain_loss(domain_pred_S: torch.Tensor, domain_pred_T: torch.Tensor, focal_loss_fn: FocalLoss = None):
    """
    Compute domain discriminator loss using Focal Loss.
    Args:
        domain_pred_S: [B, 2] source domain prediction logits (label 0)
        domain_pred_T: [B, 2] target domain prediction logits (label 1)
        focal_loss_fn: FocalLoss instance
    Returns:
        loss_domain: scalar tensor
    """
    if focal_loss_fn is None:
        focal_loss_fn = FocalLoss(class_num=2, gamma=3)
        
    device = domain_pred_S.device
    domain_S = torch.zeros(domain_pred_S.size(0), dtype=torch.long, device=device)
    domain_T = torch.ones(domain_pred_T.size(0), dtype=torch.long, device=device)
    
    loss_S = focal_loss_fn(domain_pred_S, domain_S)
    loss_T = focal_loss_fn(domain_pred_T, domain_T)
    
    return loss_S + loss_T
