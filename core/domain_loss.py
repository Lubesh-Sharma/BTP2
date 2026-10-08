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


def compute_angle_loss(angle_pred_x: torch.Tensor, angle_pred_y: torch.Tensor, rotated_gt: torch.Tensor, label_smoothing: float = 0.08):
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
    inv_rotated_gt = (num_bins - rotated_gt) % num_bins
    angle_pred_combined = torch.cat([angle_pred_x, angle_pred_y], dim=0)  # [2*B, num_bins]
    rotated_gt_combined = torch.cat([rotated_gt, inv_rotated_gt], dim=0)  # [2*B]
    loss_angle = F.cross_entropy(angle_pred_combined, rotated_gt_combined, label_smoothing=label_smoothing)
    return loss_angle


def compute_domain_loss(domain_pred_S: torch.Tensor, domain_pred_T: torch.Tensor, focal_loss_fn: FocalLoss = None, feat_clean: torch.Tensor = None, feat_rot: torch.Tensor = None):
    """
    Compute domain discriminator loss and Latent Feature Invariance.
    Args:
        domain_pred_S: [B, 2] source/clean domain prediction logits (label 0)
        domain_pred_T: [B, 2] target/rotated domain prediction logits (label 1)
        focal_loss_fn: Optional FocalLoss instance (defaults to standard Cross-Entropy for 0.4-0.8 -> 0.05-0.18 scale)
        feat_clean: [B, D] clean global feature representation
        feat_rot: [B, D] rotated global feature representation
    Returns:
        loss_domain: scalar tensor
    """
    device = domain_pred_S.device
    domain_S = torch.zeros(domain_pred_S.size(0), dtype=torch.long, device=device)
    domain_T = torch.ones(domain_pred_T.size(0), dtype=torch.long, device=device)
    
    # Standard cross-entropy domain classification (Initial: ~0.693, Target: 0.050 - 0.180)
    loss_S = F.cross_entropy(domain_pred_S, domain_S)
    loss_T = F.cross_entropy(domain_pred_T, domain_T)
    loss_disc = (loss_S + loss_T) / 2.0
    
    if feat_clean is not None and feat_rot is not None:
        # Cosine distance between clean and rotated global embeddings
        feat_clean_norm = F.normalize(feat_clean, dim=-1)
        feat_rot_norm = F.normalize(feat_rot, dim=-1)
        loss_inv = (1.0 - torch.sum(feat_clean_norm * feat_rot_norm, dim=-1)).mean()
        return loss_disc + loss_inv
        
    return loss_disc


