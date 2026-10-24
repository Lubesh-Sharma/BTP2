"""Rotation curriculum and evaluation shared by fresh OEM training and tests."""
import torch
from core.domain_loss import compute_angle_loss
from models.modules.orientation_module import (normalize_pc_torch, add_noise_to_pointcloud,
                                               nearest_angle_bins, rotate_y_radians)

CHECKPOINT_VERSION = 7


def correspondence_loss_scale(epoch, warmup_epochs, transition_epochs):
    """Zero-based epoch: OEM-only warm-up, then ramp ASMAE objectives to full weight."""
    if warmup_epochs < 0 or transition_epochs < 0:
        raise ValueError("Orientation curriculum durations cannot be negative.")
    if epoch < warmup_epochs:
        return 0.0
    if transition_epochs == 0:
        return 1.0
    return min(1.0, (epoch - warmup_epochs + 1) / transition_epochs)


def relative_rotation_pair(net, points, base_indices, relative_indices, noise_variance=0.0,
                           generator=None, noise_relative_to_radius=False, relative_radians=None):
    """Known relative Y rotation, optionally in exact radians; no cross-shape labels."""
    if noise_variance < 0:
        raise ValueError("noise_variance cannot be negative.")
    source = net.rotate_point_cloud(points, base_indices, inverse=False)
    target = (net.rotate_point_cloud(source, relative_indices, inverse=False)
              if relative_radians is None else rotate_y_radians(source, relative_radians))
    # At zero variance do not allocate noise or consume random numbers.
    if noise_variance > 0:
        source = add_noise_to_pointcloud(source, noise_variance, noise_relative_to_radius, generator)
        target = add_noise_to_pointcloud(target, noise_variance, noise_relative_to_radius, generator)
    return source, target


@torch.no_grad()
def evaluate_rotation(net, shapes, device, base_bins=None, noise_variance=0.0,
                      noise_relative_to_radius=False, arbitrary_angles=False):
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
    total_loss = total_error = total_quantization_error = 0.0
    try:
        net.eval()
        for shape in shapes:
            points = torch.as_tensor(shape['pos'], dtype=torch.float32, device=device).unsqueeze(0)
            for base_bin in base_bins:
                base = torch.tensor([base_bin], device=device)
                for index in range(net.num_class):
                    label = torch.tensor([index], device=device)
                    actual = net.ANGLE[label]
                    if arbitrary_angles:
                        # Stratified uniform angles: cover every bin's interior.
                        offset = (torch.rand((1,), device=device, generator=generator) - 0.5)
                        actual = actual + offset * (2 * torch.pi / net.num_class)
                        label = nearest_angle_bins(actual, net.ANGLE)
                    inverse_label = nearest_angle_bins(-actual, net.ANGLE)
                    source, target = relative_rotation_pair(net, points, base, label,
                                                            noise_variance, generator, noise_relative_to_radius,
                                                            relative_radians=actual if arbitrary_angles else None)
                    source, target = normalize_pc_torch(source), normalize_pc_torch(target)
                    # Evaluation uses deterministic OEM subsets.
                    source, target = net.sample_pair(source, target)
                    out = net(source, target, sample_inputs=False)
                    pred = out['angle_x'].argmax(dim=-1)
                    reverse = out['angle_y'].argmax(dim=-1)
                    difference = net.ANGLE[pred] - actual
                    total_error += torch.atan2(difference.sin(), difference.cos()).abs().sum().item()
                    quantization = net.ANGLE[label] - actual
                    total_quantization_error += torch.atan2(quantization.sin(), quantization.cos()).abs().sum().item()
                    correct += (pred == label).sum().item()
                    reverse_correct += (reverse == inverse_label).sum().item()
                    count += 1
                    total_loss += compute_angle_loss(out['angle_x'], out['angle_y'], label,
                                                    reverse_gt=inverse_label).item()
    finally:
        net.train(was_training)
    return dict(accuracy=correct / count, reverse_accuracy=reverse_correct / count,
                angle_error_degrees=total_error / count * 180.0 / torch.pi,
                quantization_error_degrees=total_quantization_error / count * 180.0 / torch.pi,
                loss=total_loss / count, count=count)


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
