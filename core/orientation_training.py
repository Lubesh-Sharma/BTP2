"""Rotation curriculum and evaluation shared by fresh OEM training and tests."""
import torch
from core.domain_loss import compute_angle_loss
from models.modules.orientation_module import normalize_pc_torch

CHECKPOINT_VERSION = 5


def correspondence_loss_scale(epoch, warmup_epochs, transition_epochs):
    """Zero-based epoch: OEM-only warm-up, then ramp ASMAE objectives to full weight."""
    if warmup_epochs < 0 or transition_epochs < 0:
        raise ValueError("Orientation curriculum durations cannot be negative.")
    if epoch < warmup_epochs:
        return 0.0
    if transition_epochs == 0:
        return 1.0
    return min(1.0, (epoch - warmup_epochs + 1) / transition_epochs)


def orientation_is_ready(metrics, minimum_accuracy, maximum_error):
    """Both OEMs must pass clean held-out forward/reverse rotation checks."""
    if not 0 <= minimum_accuracy <= 1 or not 0 <= maximum_error <= 180:
        raise ValueError("Readiness thresholds must be accuracy [0,1], error [0,180].")
    for name in ('student_validation', 'teacher_validation'):
        result = metrics.get(name)
        if result is None or not (
            result['accuracy'] >= minimum_accuracy
            and result['reverse_accuracy'] >= minimum_accuracy
            and result['angle_error_degrees'] <= maximum_error
        ):
            return False
    return True


def relative_rotation_pair(net, points, base_indices, relative_indices, noise_variance=0.0,
                           generator=None):
    """Random starting direction plus a known relative Y rotation; no cross-shape labels."""
    if noise_variance < 0:
        raise ValueError("noise_variance cannot be negative.")
    source = net.rotate_point_cloud(points, base_indices, inverse=False)
    target = net.rotate_point_cloud(source, relative_indices, inverse=False)
    # At zero variance do not allocate noise or consume random numbers.
    if noise_variance > 0:
        target = target + torch.randn(target.shape, device=target.device, dtype=target.dtype,
                                      generator=generator) * noise_variance ** 0.5
    return source, target


@torch.no_grad()
def evaluate_rotation(net, shapes, device, base_bins=None, noise_variance=0.0):
    """Same-shape relative rotations at deterministic starting directions in eval mode.

    Uses a private RNG for optional noise, leaving training randomness untouched.
    This is not a measurement of cross-shape physical orientation or P2P quality.
    """
    if not shapes:
        raise ValueError("Rotation evaluation requires at least one shape.")
    base_bins = [1] if base_bins is None else list(base_bins)
    if not base_bins or any(index < 0 or index >= net.num_class for index in base_bins):
        raise ValueError("Invalid starting rotation bins.")
    if noise_variance < 0:
        raise ValueError("noise_variance cannot be negative.")
    generator = torch.Generator(device=device).manual_seed(12345)
    was_training = net.training
    correct = reverse_correct = count = 0
    total_loss = total_error = max_mode_gap = 0.0
    try:
        net.eval()
        for shape in shapes:
            points = torch.as_tensor(shape['pos'], dtype=torch.float32, device=device).unsqueeze(0)
            for base_bin in base_bins:
                base = torch.tensor([base_bin], device=device)
                for index in range(net.num_class):
                    label = torch.tensor([index], device=device)
                    source, target = relative_rotation_pair(net, points, base, label,
                                                            noise_variance, generator)
                    source, target = normalize_pc_torch(source), normalize_pc_torch(target)
                    out = net(source, target)
                    pred = out['angle_x'].argmax(dim=-1)
                    reverse = out['angle_y'].argmax(dim=-1)
                    difference = net.ANGLE[pred] - net.ANGLE[label]
                    total_error += torch.atan2(difference.sin(), difference.cos()).abs().sum().item()
                    inverse_difference = net.ANGLE + net.ANGLE[index]
                    inverse_label = torch.atan2(inverse_difference.sin(), inverse_difference.cos()).abs().argmin()
                    correct += (pred == label).sum().item()
                    reverse_correct += (reverse == inverse_label).sum().item()
                    count += 1
                    total_loss += compute_angle_loss(out['angle_x'], out['angle_y'], label).item()
                    if index == 1:
                        net.train()
                        train_out = net(source, target)
                        net.eval()
                        gap = max((train_out[key] - out[key]).abs().max().item()
                                  for key in ('angle_x', 'angle_y'))
                        max_mode_gap = max(max_mode_gap, gap)
    finally:
        net.train(was_training)
    return dict(accuracy=correct / count, reverse_accuracy=reverse_correct / count,
                angle_error_degrees=total_error / count * 180.0 / torch.pi,
                loss=total_loss / count, mode_gap=max_mode_gap, count=count)
