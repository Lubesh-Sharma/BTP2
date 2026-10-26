"""MultiBin-inspired continuous yaw supervision and resampling consistency.

No mesh faces, body parts, ground-truth correspondence, or pretrained weights.
This adapts published ideas; it is not a reproduction of an entire paper.
"""
import math
import torch
import torch.nn.functional as F


def multibin_rotation_loss(output, radians, centers, overlap=1.5):
    """Overlapping classification targets and circular residual loss in both directions."""
    classification = radians.new_zeros(())
    residual = radians.new_zeros(())
    half_width = overlap * math.pi / centers.numel()
    for direction, actual in [('x', radians), ('y', -radians)]:
        delta = actual[:, None] - centers.to(actual)[None, :]
        delta = torch.atan2(delta.sin(), delta.cos())
        valid = delta.abs() <= half_width + 1e-6
        targets = valid.to(actual.dtype)
        targets = targets / targets.sum(-1, keepdim=True).clamp_min(1)
        classification = classification - (targets * output[f'angle_{direction}'].log_softmax(-1)).sum(-1).mean()
        desired = torch.stack((delta.cos(), delta.sin()), -1)
        vectors = output[f'residual_{direction}']
        error = 1 - (vectors * desired).sum(-1)
        residual = residual + (targets * error).sum(-1).mean()
    return classification / 2, residual / 2


def subset_rotation_consistency(first, second):
    """Symmetric JS of mode probabilities plus confidence-weighted residual agreement.

    Compares modes separately, avoiding an average between opposite directions.
    Both independently sampled predictions receive gradients.
    """
    loss = first['angle_x'].new_zeros(())
    for direction in ('x', 'y'):
        logp = first[f'angle_{direction}'].log_softmax(-1)
        logq = second[f'angle_{direction}'].log_softmax(-1)
        p, q = logp.exp(), logq.exp()
        average = (p + q) / 2
        log_average = average.clamp_min(1e-8).log()
        js = 0.5 * ((p * (logp - log_average)).sum(-1) + (q * (logq - log_average)).sum(-1))
        agreement = 1 - (first[f'residual_{direction}'] * second[f'residual_{direction}']).sum(-1)
        loss = loss + (js + (average.detach() * agreement).sum(-1)).mean()
    return loss / 2
