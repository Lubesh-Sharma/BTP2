"""Shared OEM -> aligned descriptor/coordinate inputs for ASMAE.

SE-ORNet predicts the rotation of the second cloud relative to the first,
then rotates the second cloud by the inverse predicted bin. No vertex order
changes, so correspondence indices remain in the original mesh indexing.
To match the paper figure, pass target Y first and source X second to OEM.
"""

import torch

from .modules.orientation_module import normalize_pc_torch

ALIGNMENT_PIPELINE_VERSION = 4


def features_in_aligned_frame(features, positions):
    """Keep descriptors and replace appended XYZ with the aligned unit-sphere XYZ.

    XYZ uses the fixed transform (xyz + 1) / 2 in the common pair frame. Unlike
    independent per-axis min/max scaling, this preserves the alignment.
    """
    if features.shape[:-1] != positions.shape[:-1] or features.shape[-1] <= 3:
        raise ValueError("Expected matching point counts and descriptor + XYZ channels")
    return torch.cat((features[..., :-3], (positions + 1.0) / 2.0), dim=-1)


def prepare_aligned_pair(orient_net, features_source, positions_source,
                         features_target, positions_target, alignment='predicted'):
    """Normalize both clouds and align source X to target Y, as in the figure.

    Called by the student OEM for augmented pairs, EMA teacher OEM for clean
    pairs, and the saved student OEM during evaluation. Discrete bin selection
    follows the SE-ORNet implementation; angle supervision trains the OEM.
    """
    source = normalize_pc_torch(positions_source)
    target = normalize_pc_torch(positions_target)
    # OEM's angle_x describes its second input relative to its first.
    # Keep the target as the reference and rotate the source, for both the
    # clean teacher pair (Y, X) and augmented student pair (Y_s, X_s).
    outputs = orient_net(target, source)
    angle_indices = outputs["angle_x"].argmax(dim=-1)
    if alignment == 'predicted':
        source = orient_net.rotate_point_cloud(source, angle_indices, inverse=True)
    elif alignment != 'identity':
        raise ValueError("alignment must be 'predicted' or 'identity'")
    return (features_in_aligned_frame(features_source, source), source,
            features_in_aligned_frame(features_target, target), target, outputs)


def load_orientation_checkpoint(orient_net, checkpoint):
    """Require an OEM trained with the same alignment/domain pipeline."""
    if checkpoint.get("alignment_pipeline_version") != ALIGNMENT_PIPELINE_VERSION:
        raise ValueError("Checkpoint uses the previous OEM pipeline. Train with the "
                         "updated train_st_te.py before evaluating aligned pairs.")
    if "orient_net_state_dict" not in checkpoint:
        raise ValueError("Checkpoint has no trained orientation module")
    orient_net.max_points = checkpoint.get('oem_config', {}).get('max_points', 512)
    orient_net.load_state_dict(checkpoint["orient_net_state_dict"])
