"""Rotation warm-up and fixed evaluation-mode diagnostics, run on SSH."""

import torch
import csv
import math
from pathlib import Path

from core.domain_loss import compute_angle_loss, compute_pair_domain_loss, FocalLoss
from models.modules.orientation_module import normalize_pc_torch
from models.oriented_pair import ALIGNMENT_PIPELINE_VERSION


def canonical_angle_labels(config):
    """Explicit native relative yaw labels only; never assume poses share a frame.

    Entries name source X, target Y and X's native rotation bin relative to Y.
    Bin 1 means already aligned. The reverse entry is derived automatically.
    """
    labels = {}
    for pair in config.get('canonical_pairs', []):
        source, target = str(pair['source']), str(pair['target'])
        index = int(pair['angle_bin'])
        if not 0 <= index < 8:
            raise ValueError('canonical_pairs.angle_bin must be between 0 and 7')
        for key, value in (((source, target), index), ((target, source), (2 - index) % 8)):
            if key in labels and labels[key] != value:
                raise ValueError(f'Conflicting canonical pair annotation: {key}')
            labels[key] = value
    return labels


@torch.no_grad()
def evaluate_oem(net, shapes, canonical_labels=None):
    """All eight rotations on fixed shapes, never training-mode minibatch accuracy."""
    device = next(net.parameters()).device
    was_training = net.training
    net.eval()
    counts = {'forward': 0, 'inverse': 0, 'total': 0, 'ce': 0.0, 'degrees': 0.0}
    cross_correct, cross_count = 0, 0
    by_name = {s['name']: s for s in shapes}
    details = []
    confusion = [[0 for _ in range(8)] for _ in range(8)]
    for shape in shapes:
        clean = normalize_pc_torch(torch.as_tensor(shape['pos'], device=device,
                                                   dtype=torch.float32).unsqueeze(0))
        for index in range(8):
            label = torch.tensor([index], device=device)
            rotated = net.rotate_point_cloud(clean, label, inverse=False)
            outputs = net(clean, rotated)
            forward = outputs['angle_x'].argmax(-1)
            inverse = outputs['angle_y'].argmax(-1)
            counts['forward'] += (forward == label).sum().item()
            counts['inverse'] += (inverse == (2 - label) % 8).sum().item()
            counts['ce'] += compute_angle_loss(outputs['angle_x'], outputs['angle_y'], label).item()
            delta = (forward - label).abs()
            degrees = torch.minimum(delta, 8 - delta).sum().item() * 45
            counts['degrees'] += degrees
            predicted = int(forward.item())
            confusion[index][predicted] += 1
            details.append({'shape': shape['name'], 'gt_bin': index,
                            'forward_bin': predicted, 'inverse_gt_bin': (2 - index) % 8,
                            'inverse_bin': int(inverse.item()), 'forward_error_degrees': degrees})
            counts['total'] += 1
    for (source, target), base_bin in (canonical_labels or {}).items():
        if source not in by_name or target not in by_name or source == target:
            continue
        source_pc = normalize_pc_torch(torch.as_tensor(by_name[source]['pos'], device=device,
                                                       dtype=torch.float32).unsqueeze(0))
        target_pc = normalize_pc_torch(torch.as_tensor(by_name[target]['pos'], device=device,
                                                       dtype=torch.float32).unsqueeze(0))
        for index in range(8):
            rotated = net.rotate_point_cloud(source_pc, torch.tensor([index], device=device), inverse=False)
            expected = (base_bin + index - 1) % 8
            outputs = net(target_pc, rotated)
            cross_correct += (outputs['angle_x'].argmax(-1) == expected).sum().item()
            cross_count += 1
    net.train(was_training)
    total = max(counts['total'], 1)
    return {'forward_accuracy': counts['forward'] / total,
            'inverse_accuracy': counts['inverse'] / total,
            'accuracy': (counts['forward'] + counts['inverse']) / (2 * total),
            'angle_ce': counts['ce'] / total, 'yaw_error_degrees': counts['degrees'] / total,
            'cross_accuracy': cross_correct / cross_count if cross_count else None,
            'details': details, 'forward_confusion': confusion}


def pretrain_oem(net, train_shapes, validation_shapes, config, checkpoint_path, resume=True):
    """Learn rotation before any predicted angle controls ASMAE training."""
    epochs = int(config.get('pretrain_epochs', 30))
    if epochs <= 0:
        return evaluate_oem(net, validation_shapes)
    device = next(net.parameters()).device
    initial_lr = float(config.get('lr', 0.00025))
    optimizer = torch.optim.Adam(net.parameters(), lr=initial_lr,
                                 weight_decay=float(config.get('weight_decay', 0.0001)))
    focal = FocalLoss(class_num=2, gamma=3)
    labels = canonical_angle_labels(config)
    cache = {s['name']: normalize_pc_torch(torch.as_tensor(s['pos'], dtype=torch.float32,
                                                           device=device).unsqueeze(0))
             for s in train_shapes}
    names = sorted(cache)
    best_accuracy, best_state, best_metrics = -1.0, None, None
    probe_count = min(int(config.get('train_eval_shapes', 4)), len(train_shapes))
    if probe_count < 1:
        raise ValueError('oem.train_eval_shapes must be positive')
    train_probe = [train_shapes[i * len(train_shapes) // probe_count] for i in range(probe_count)]
    start_epoch = 0
    if resume and Path(checkpoint_path).exists():
        saved = torch.load(checkpoint_path, map_location=device, weights_only=True)
        if saved.get('alignment_pipeline_version') != ALIGNMENT_PIPELINE_VERSION:
            raise ValueError('Warm-up checkpoint uses the previous OEM normalization. Restart with --fresh.')
        expected = [s['name'] for s in validation_shapes]
        if saved.get('validation_shapes') != expected:
            raise ValueError('OEM warm-up validation split changed; use --fresh')
        if saved.get('oem_config', {}).get('max_points', 512) != net.max_points:
            raise ValueError('OEM point sampling changed; use --fresh')
        net.load_state_dict(saved['orient_net_state_dict'])
        if 'optimizer_state_dict' in saved:
            optimizer.load_state_dict(saved['optimizer_state_dict'])
        start_epoch = int(saved.get('epoch', 0))
        best_metrics = evaluate_oem(net, validation_shapes, labels)
        best_accuracy = min(best_metrics['forward_accuracy'], best_metrics['inverse_accuracy'])
        best_state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
        print(f"Resuming OEM warm-up from epoch {start_epoch}")
    for epoch in range(start_epoch, epochs):
        net.train()
        # Reduce late-epoch oscillations rather than keep fitting at full LR.
        factor = 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * epoch / max(epochs - 1, 1)))
        for group in optimizer.param_groups:
            group['lr'] = initial_lr * factor
            group['weight_decay'] = float(config.get('weight_decay', 0.0001))
        domain_start = int(config.get('domain_start_epoch', 10))
        ramp = max(int(config.get('domain_warmup_epochs', 10)), 1)
        domain_weight = float(config.get('domain_weight', 0.1)) * max(0.0, min(1.0, (epoch - domain_start + 1) / ramp))
        total_loss, steps = 0.0, 0
        # Every shape sees every bin, avoiding random-bin imbalance.
        for name_index in torch.randperm(len(names)).tolist():
            name = names[name_index]
            clean = cache[name]
            other_name = names[(name_index + 1 + epoch % max(len(names) - 1, 1)) % len(names)]
            target = cache[other_name]
            for index in torch.randperm(8).tolist():
                label = torch.tensor([index], device=device)
                rotated = net.rotate_point_cloud(clean, label, inverse=False)
                same = net(clean, rotated)
                loss = compute_angle_loss(same['angle_x'], same['angle_y'], label,
                                          float(config.get('label_smoothing', 0.0)))
                if domain_weight > 0 or (name, other_name) in labels:
                    cross = net(target, rotated)
                    if domain_weight > 0:
                        loss = loss + domain_weight * compute_pair_domain_loss(
                            same['global_d_pred'], cross['global_d_pred'], focal)
                    if (name, other_name) in labels:
                        relative = torch.tensor([(labels[(name, other_name)] + index - 1) % 8], device=device)
                        loss = loss + compute_angle_loss(cross['angle_x'], cross['angle_y'], relative)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), float(config.get('grad_clip', 5.0)))
                optimizer.step()
                total_loss += loss.item()
                steps += 1
        metrics = evaluate_oem(net, validation_shapes, labels)
        train_metrics = evaluate_oem(net, train_probe)
        print(f"OEM warm-up {epoch + 1}/{epochs} | Loss: {total_loss / max(steps, 1):.4f} | "
              f"Eval forward: {metrics['forward_accuracy']:.2%} | Eval inverse: {metrics['inverse_accuracy']:.2%} | "
              f"Forward yaw error: {metrics['yaw_error_degrees']:.2f} deg | Domain weight: {domain_weight:.3f} | "
              f"LR: {optimizer.param_groups[0]['lr']:.7f}")
        print(f"  Training-shape eval probe: forward={train_metrics['forward_accuracy']:.2%}, "
              f"inverse={train_metrics['inverse_accuracy']:.2%}, CE={train_metrics['angle_ce']:.4f} "
              f"({len(train_probe)} fixed training shapes)")
        for shape in validation_shapes:
            rows = [row for row in metrics['details'] if row['shape'] == shape['name']]
            correct = sum(row['forward_bin'] == row['gt_bin'] for row in rows)
            yaw = sum(row['forward_error_degrees'] for row in rows) / max(len(rows), 1)
            print(f"  Validation shape {shape['name']}: forward={correct}/{len(rows)}, yaw={yaw:.2f} deg")
        diagnostic_path = Path(checkpoint_path).with_suffix('')
        diagnostic_path = Path(str(diagnostic_path) + '_predictions.csv')
        with diagnostic_path.open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['shape', 'gt_bin', 'forward_bin',
                                                  'inverse_gt_bin', 'inverse_bin', 'forward_error_degrees'])
            writer.writeheader()
            writer.writerows(metrics['details'])
        if metrics['cross_accuracy'] is not None:
            print(f"  Annotated held-out cross-shape rotation accuracy: {metrics['cross_accuracy']:.2%}")
        score = min(metrics['forward_accuracy'], metrics['inverse_accuracy'])
        if score > best_accuracy or (score == best_accuracy
                                                  and metrics['angle_ce'] < best_metrics['angle_ce']):
            best_accuracy, best_metrics = score, metrics
            best_state = {key: value.detach().cpu().clone() for key, value in net.state_dict().items()}
            torch.save({'orient_net_state_dict': best_state, 'oem_config': config,
                        'alignment_pipeline_version': ALIGNMENT_PIPELINE_VERSION,
                        'oem_validation': metrics, 'epoch': epoch + 1,
                        'train_eval_probe': train_metrics,
                        'optimizer_state_dict': optimizer.state_dict(),
                        'validation_shapes': [s['name'] for s in validation_shapes]}, checkpoint_path)
    net.load_state_dict(best_state)
    net.eval()
    return best_metrics
