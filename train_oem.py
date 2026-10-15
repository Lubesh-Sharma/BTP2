"""Standalone OEM experiment: coordinates only; no ASMAE or geometry features."""
import argparse
import copy
import math
from pathlib import Path
import random

import torch
import yaml

from core.domain_loss import compute_angle_loss
from core.orientation_training import evaluate_rotation, relative_rotation_pair, oem_learning_rate
from models.modules.orientation_module import OrientNet, normalize_pc_torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='config/SHREC/train_oem_fps_config.yaml')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text())
    training = config['training']
    device = torch.device(training['device'])
    seed = training.get('seed', 42)
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)
    path = Path(training['checkpoint_dir']) / training['checkpoint_name']
    if path.exists() and not args.resume:
        raise FileExistsError(f'{path} exists. Use --resume to continue it.')

    # Read vertices only: no mesh features, eigensolvers, or ASMAE imports.
    files = sorted(Path(config['data_dir']).glob('*.obj'))
    pool = config['train_size']
    val_size = training['validation_size']
    if len(files) < pool + config['test_size'] or not 0 < val_size < pool:
        raise ValueError('Dataset does not support the configured train/validation/test split')

    def load(paths):
        shapes = []
        for filename in paths:
            vertices = []
            with filename.open() as handle:
                for line in handle:
                    fields = line.split()
                    if fields and fields[0] == 'v':
                        vertices.append([float(v) for v in fields[1:4]])
            if not vertices:
                raise ValueError(f'No vertices in {filename}')
            shapes.append({'name': filename.stem,
                           'pos': torch.tensor(vertices, dtype=torch.float32)})
        return shapes

    train = load(files[:pool - val_size])
    validation = load(files[pool - val_size:pool])
    net = OrientNet(**config['orientation_model']).to(device)
    teacher = copy.deepcopy(net).eval().requires_grad_(False)
    optimizer = torch.optim.Adam(net.parameters(), lr=training['lr'])
    start = 0
    best = float('inf')
    if args.resume:
        saved = torch.load(path, map_location='cpu', weights_only=True)
        if saved['config'] != config:
            raise ValueError('Resume requires the same experiment config')
        net.load_state_dict(saved['orient_net_state_dict'])
        teacher.load_state_dict(saved['teacher_orient_net_state_dict'])
        optimizer.load_state_dict(saved['optimizer_state_dict'])
        start = saved['completed_epochs']
        best = saved['best_angle_error']
        random.setstate(saved['python_rng'])
        torch.set_rng_state(saved['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(saved['cuda_rng'])

    print('OEM ONLY: no ASMAE, no P2P, no geometry features, no noise.', flush=True)
    print('Training:', [s['name'] for s in train], flush=True)
    print('Validation:', [s['name'] for s in validation], flush=True)
    print('Sampling:', config['orientation_model'], flush=True)
    for shape in train + validation:
        n = len(shape['pos'])
        print(f"Shape {shape['name']}: {n} input vertices", flush=True)

    for epoch in range(start, training['epochs']):
        net.train()
        lr = oem_learning_rate(epoch, training['lr'], training.get('oem_lr_schedule'))
        for group in optimizer.param_groups:
            group['lr'] = lr
        # Match the full trainer's 23 reference updates per training shape.
        jobs = [i for i in range(len(train)) for _ in range(len(train) - 1)]
        random.shuffle(jobs)
        total = 0.0
        for step, index in enumerate(jobs, 1):
            points = train[index]['pos'].to(device).unsqueeze(0)
            base = torch.randint(8, (1,), device=device)
            label = torch.randint(8, (1,), device=device)
            source, target = relative_rotation_pair(net, points, base, label)
            out = net(normalize_pc_torch(source), normalize_pc_torch(target))
            loss = compute_angle_loss(out['angle_x'], out['angle_y'], label)
            optimizer.zero_grad(set_to_none=True)
            # Preserve the baseline warm-up angle-loss scaling.
            warm = training['angle_warmup_epochs']
            fraction = 0.5 * (1 + math.cos(math.pi * epoch / warm)) if warm and epoch < warm else 0.0
            weight = training['angle_weight'] + (
                training['angle_weight_init'] - training['angle_weight']) * fraction
            (weight * loss).backward()
            optimizer.step()
            with torch.no_grad():
                current_state = net.state_dict()
                for name, value in teacher.state_dict().items():
                    current = current_state[name]
                    if value.is_floating_point():
                        value.lerp_(current, 1 - training['ema_alpha'])
                    else:
                        value.copy_(current)
            total += loss.item()
            if step % 48 == 0:
                print(f'Epoch {epoch + 1}: {step}/{len(jobs)} updates', flush=True)

        metrics = {}
        for name, model in [('student', net), ('teacher', teacher)]:
            result = evaluate_rotation(model, validation, device, base_bins=list(range(8)))
            metrics[name] = result
            print(f"Epoch {epoch + 1} {name}: CE={total / len(jobs):.4f} | "
                  f"val forward={result['accuracy']:.2%}, reverse={result['reverse_accuracy']:.2%}, "
                  f"angle={result['angle_error_degrees']:.2f} degrees", flush=True)
        improved = metrics['student']['angle_error_degrees'] < best
        best = min(best, metrics['student']['angle_error_degrees'])
        state = dict(kind='oem_only', config=config, completed_epochs=epoch + 1,
                     train_names=[s['name'] for s in train],
                     validation_names=[s['name'] for s in validation],
                     orient_net_state_dict=net.state_dict(),
                     teacher_orient_net_state_dict=teacher.state_dict(),
                     optimizer_state_dict=optimizer.state_dict(), metrics=metrics,
                     best_angle_error=best, python_rng=random.getstate(),
                     torch_rng=torch.get_rng_state(),
                     cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else [])
        path.parent.mkdir(parents=True, exist_ok=True)
        targets = [path]
        if improved:
            targets.append(path.with_stem(path.stem + '_best_rotation'))
        for target_path in targets:
            temporary = target_path.with_suffix('.tmp')
            torch.save(state, temporary)
            temporary.replace(target_path)


if __name__ == '__main__':
    main()
