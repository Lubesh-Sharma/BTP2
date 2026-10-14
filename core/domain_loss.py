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


def compute_angle_loss(angle_pred_x: torch.Tensor, angle_pred_y: torch.Tensor, rotated_gt: torch.Tensor, label_smoothing: float = 0.0):
    """
    Compute cross-entropy loss for angle predictions with label smoothing.
    Args:
        angle_pred_x: [B, num_bins] logits for angle of source w.r.t target
        angle_pred_y: [B, num_bins] logits for angle of target w.r.t source
        rotated_gt: [B] ground truth rotation angle bin index
        label_smoothing: label smoothing factor to eliminate flickering on near-symmetric pairs
    Returns:
        loss_angle: scalar tensor
    """
    num_bins = angle_pred_x.size(-1)
    # Offset by +2 bins to account for the -pi/4 grid shift in OrientNet (k=1 is 0 deg)
    inv_rotated_gt = (10 - rotated_gt) % num_bins
    angle_pred_combined = torch.cat([angle_pred_x, angle_pred_y], dim=0)  # [2*B, num_bins]
    rotated_gt_combined = torch.cat([rotated_gt, inv_rotated_gt], dim=0)  # [2*B]
    loss_angle = F.cross_entropy(angle_pred_combined, rotated_gt_combined, label_smoothing=label_smoothing)
    return loss_angle


def compute_domain_loss(reference_logits: torch.Tensor, cross_shape_logits: torch.Tensor,
                        focal_loss_fn: FocalLoss = None):
    """Adapt OEM pair features from same-shape references (0) to cross-shape pairs (1)."""
    if focal_loss_fn is None:
        focal_loss_fn = FocalLoss(class_num=2, gamma=3)
    reference_labels = torch.zeros(reference_logits.size(0), dtype=torch.long,
                                   device=reference_logits.device)
    cross_labels = torch.ones(cross_shape_logits.size(0), dtype=torch.long,
                              device=cross_shape_logits.device)
    return (focal_loss_fn(reference_logits, reference_labels)
            + focal_loss_fn(cross_shape_logits, cross_labels))
