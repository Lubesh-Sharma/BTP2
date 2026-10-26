"""Rotation curriculum and evaluation shared by fresh OEM training and tests."""
import torch
from core.rotation_loss import multibin_rotation_loss
from models.modules.orientation_module import (normalize_pc_torch, add_noise_to_pointcloud,
                                               nearest_angle_bins, rotate_y_radians)

CHECKPOINT_VERSION = 10


def correspondence_loss_scale(epoch, warmup_epochs, transition_epochs):
    """Zero-based epoch: OEM-only warm-up, then ramp ASMAE objectives to full weight."""
    if warmup_epochs < 0 or transition_epochs < 0:
        raise ValueError("Orientation curriculum durations cannot be negative.")
    if epoch < warmup_epochs:
        return 0.0
    if transition_epochs == 0:
        return 1.0
    return min(1.0, (epoch - warmup_epochs + 1) / transition_epochs)


def orientation_loss_weights(epoch, angle_initial, angle_target, duration, domain_initial):
    """Start both losses at full weight; preserve their ratio during cosine decay."""
    if angle_initial <= 0 or angle_target < 0 or domain_initial < 0 or duration < 0:
        raise ValueError("Invalid angle/domain loss-weight schedule")
    import math
    fraction = 0.5 * (1 + math.cos(math.pi * epoch / duration)) if duration and epoch < duration else 0.0
    angle = angle_target + (angle_initial - angle_target) * fraction
    return angle, domain_initial * angle / angle_initial


def relative_rotation_pair(net, points, base_indices, relative_indices, noise_variance=0.0,
                           generator=None, noise_relative_to_radius=False, relative_radians=None):
    """Known relative Y rotation (exact radians or bin center); no cross-shape labels."""
    if noise_variance < 0:
        raise ValueError("noise_variance cannot be negative.")
    source = net.rotate_point_cloud(points, base_indices, inverse=False)
    target = (net.rotate_point_cloud(source, relative_indices, inverse=False) if relative_radians is None
              else rotate_y_radians(source, relative_radians))
    # At zero variance do not allocate noise or consume random numbers.
    if noise_variance > 0:
        source = add_noise_to_pointcloud(source, noise_variance, noise_relative_to_radius, generator)
        target = add_noise_to_pointcloud(target, noise_variance, noise_relative_to_radius, generator)
    return source, target


@torch.no_grad()
def evaluate_rotation(net, shapes, device, base_bins=None, noise_variance=0.0,
                      noise_relative_to_radius=False, arbitrary_angles=True):
    """Deterministic synthetic rotation evaluation, not cross-pose ground truth.

    Tests the applied continuous prediction. Exact-grid tests remain available
    with arbitrary_angles=False. All randomness is private to evaluation.
    """
    if not shapes:
        raise ValueError("Rotation evaluation requires at least one shape.")
    base_bins = [1] if base_bins is None else list(base_bins)
    if not base_bins or any(i < 0 or i >= net.num_class for i in base_bins):
        raise ValueError("Invalid starting rotation bins.")
    if noise_variance < 0:
        raise ValueError("noise_variance cannot be negative.")
    generator = torch.Generator(device=device).manual_seed(12345)
    was_training = net.training
    correct = reverse_correct = coarse_correct = count = 0
    total_loss = total_error = total_residual = total_coarse_error = 0.0
    hits = {5: 0, 10: 0, 15: 0}
    large_errors = flips = 0
    try:
        net.eval()
        for shape in shapes:
            points = torch.as_tensor(shape['pos'], dtype=torch.float32, device=device).unsqueeze(0)
            for base_bin in base_bins:
                base = torch.tensor([base_bin], device=device)
                for index in range(net.num_class):
                    actual = net.ANGLE[index].reshape(1)
                    if arbitrary_angles:
                        offset = torch.rand((1,), device=device, generator=generator) - 0.5
                        actual = actual + offset * (2 * torch.pi / net.num_class)
                    label = nearest_angle_bins(actual, net.ANGLE)
                    inverse_label = nearest_angle_bins(-actual, net.ANGLE)
                    source, target = relative_rotation_pair(
                        net, points, base, label, noise_variance, generator,
                        noise_relative_to_radius, relative_radians=actual)
                    out = net(normalize_pc_torch(source), normalize_pc_torch(target))
                    predicted = net.predict_angle(out)
                    reverse = net.predict_angle(out, 'y')
                    delta = predicted - actual
                    error = torch.rad2deg(torch.atan2(delta.sin(), delta.cos()).abs()).item()
                    coarse_bin = out['angle_x'].argmax(-1)
                    coarse_delta = net.ANGLE[coarse_bin] - actual
                    total_coarse_error += torch.rad2deg(torch.atan2(coarse_delta.sin(), coarse_delta.cos()).abs()).item()
                    correct += (nearest_angle_bins(predicted, net.ANGLE) == label).sum().item()
                    reverse_correct += (nearest_angle_bins(reverse, net.ANGLE) == inverse_label).sum().item()
                    coarse_correct += (coarse_bin == label).sum().item()
                    for threshold in hits:
                        hits[threshold] += int(error <= threshold)
                    large_errors += int(error > 90)
                    flips += int(error >= 150)
                    count += 1
                    total_error += error
                    classification, residual = multibin_rotation_loss(out, actual, net.ANGLE, net.angle_overlap)
                    total_loss += classification.item()
                    total_residual += residual.item()
    finally:
        net.train(was_training)
    return dict(accuracy=correct / count, reverse_accuracy=reverse_correct / count,
                coarse_accuracy=coarse_correct / count,
                angle_error_degrees=total_error / count,
                coarse_angle_error_degrees=total_coarse_error / count,
                loss=total_loss / count, residual_loss=total_residual / count, count=count,
                large_error_fraction=large_errors / count, flip_fraction=flips / count,
                **{f'accuracy_{threshold}deg': value / count for threshold, value in hits.items()})


def oem_learning_rate(epoch, base_lr, schedule=None):
    """Absolute-epoch cosine decay; YAML start_epoch is one-based.

    Stateless scheduling allows a version-5 run to adopt this schedule on resume
    without resetting Adam moments or changing completed-epoch bookkeeping.
    """
    if not schedule:
        return base_lr
    start = schedule['start_epoch']
    duration = schedule['duration_epochs']
    initial, minimum = schedule['start_lr'], schedule['min_lr']
    if start < 1 or duration < 1 or not 0 < minimum <= initial:
        raise ValueError("Invalid OEM learning-rate schedule.")
    if epoch + 1 < start:
        return base_lr
    import math
    progress = min(1.0, (epoch + 1 - start) / max(1, duration - 1))
    return minimum + (initial - minimum) * 0.5 * (1.0 + math.cos(math.pi * progress))
