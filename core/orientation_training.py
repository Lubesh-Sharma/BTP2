"""Rotation curriculum and evaluation shared by fresh OEM training and tests."""
import torch
from core.domain_loss import compute_angle_loss
from models.modules.orientation_module import normalize_pc_torch

CHECKPOINT_VERSION = 3


def predicted_alignment_probability(epoch, warmup_epochs, transition_epochs):
    """Epoch is zero-based; transition gradually replaces known augmentation labels."""
    if warmup_epochs < 0 or transition_epochs < 0:
        raise ValueError("Orientation curriculum durations cannot be negative.")
    if epoch < warmup_epochs:
        return 0.0
    if transition_epochs == 0:
        return 1.0
    return min(1.0, (epoch - warmup_epochs + 1) / transition_epochs)


@torch.no_grad()
def evaluate_rotation(net, shapes, device):
    """Evaluate every discrete rotation with normal eval behavior, without augmentation noise.

    This measures same-shape rotation recovery, not cross-shape physical orientation.
    """
    if not shapes:
        raise ValueError("Rotation evaluation requires at least one shape.")
    was_training = net.training
    correct = reverse_correct = count = 0
    total_loss = total_error = max_mode_gap = 0.0
    try:
        net.eval()
        for shape in shapes:
            source = torch.as_tensor(shape['pos'], dtype=torch.float32, device=device).unsqueeze(0)
            source_norm = normalize_pc_torch(source)
            for index in range(net.num_class):
                label = torch.tensor([index], device=device)
                target = normalize_pc_torch(net.rotate_point_cloud(source, label, inverse=False))
                out = net(source_norm, target)
                pred = out['angle_x'].argmax(dim=-1)
                reverse = out['angle_y'].argmax(dim=-1)
                difference = net.ANGLE[pred] - net.ANGLE[label]
                total_error += torch.atan2(difference.sin(), difference.cos()).abs().sum().item()
                # Resolve inverse labels from the actual registered angle grid.
                inverse_difference = net.ANGLE + net.ANGLE[index]
                inverse_label = torch.atan2(inverse_difference.sin(), inverse_difference.cos()).abs().argmin()
                correct += (pred == label).sum().item()
                reverse_correct += (reverse == inverse_label).sum().item()
                count += 1
                total_loss += compute_angle_loss(out['angle_x'], out['angle_y'], label).item()
                if index == 1:
                    # Same weights/inputs must produce the same logits in both modes.
                    net.train()
                    train_out = net(source_norm, target)
                    net.eval()
                    gap = max((train_out[key] - out[key]).abs().max().item()
                              for key in ('angle_x', 'angle_y'))
                    max_mode_gap = max(max_mode_gap, gap)
    finally:
        net.train(was_training)
    return dict(accuracy=correct / count, reverse_accuracy=reverse_correct / count,
                angle_error_degrees=total_error / count * 180.0 / torch.pi,
                loss=total_loss / count, mode_gap=max_mode_gap)
