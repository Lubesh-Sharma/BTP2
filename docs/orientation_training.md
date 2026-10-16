# Fresh OEM-first student–teacher training

Use `config/SHREC/train_st_te_config.yaml` for the normal SHREC pipeline.
No pretrained checkpoints are required. Version 6 prevents accidental resume of
old readiness-gated experiments.

## Schedule

- Epochs 1–30: train student OEM by synthetic rotation cross-entropy; update its EMA OEM. ASMAE receives no forward passes, gradients, or EMA changes.
- Epochs 31–40: continue OEM angle training; ramp ASMAE loss scale and learning rate from 0.1 to 1.0. Ramp the cross-shape domain-loss weight too.
- Epochs 41–70: train the full pipeline together at full correspondence weight.

Validation reports forward/reverse rotation accuracy, circular angle error, and
CE. It never gates or delays the schedule. There are no training/evaluation
normalization comparisons. GroupNorm remains in OEM. Noise variance is zero.
The 24/2/18 train/validation/test shape split remains unchanged.

## Sampling and alignment

OEM independently samples each input uniformly without replacement. The count is
`min(N, max(1024, ceil(0.70 * N)))`: 3,500 points for a 5,000-point shape.
Training draws new subsets; evaluation uses fixed private seeds. Sampling does
not mutate the full inputs. Predict rotation on subsets, apply it to the full
target, and give all vertices to ASMAE. Feature XYZ channels follow the aligned
coordinates. Both correspondence loaders and the visualization notebook restore
the sampling configuration from the saved full-pipeline checkpoint.

## Run

```bash
python train_st_te.py --config config/SHREC/train_st_te_config.yaml --fresh
```

`--fresh` overwrites the configured checkpoint as epochs complete; preserve an
existing file first if needed. Omit `--fresh` to resume the last completed epoch.
Checkpoints include both ASMAEs, both OEMs, optimizer, configuration, split, and RNG
states. An interrupted incomplete epoch does not overwrite completed progress.
The `_best_rotation.pth` file tracks OEM validation, not best P2P performance;
use the final main checkpoint to evaluate the fully trained pipeline.

```bash
python corres_st_te.py --config config/SHREC/corres.yaml --output-dir p2p_results_st_te_SHREC
python corres_greedy.py --config config/SHREC/corres.yaml --out_dir p2p_results_greedy_SHREC
```

Both default to the student; pass `--network teacher` for the EMA pair. DINO is
disabled unless explicitly enabled in the greedy script. Run
`feature_visualization.ipynb` from the top. Run `checker.ipynb` afterward, selecting
which output directory to inspect. This repository has no `check.ipynb`.

The standalone OEM trainer and its FPS/random50 configs, older noise/sampling
experiment configs, and obsolete sampling correspondence config have been removed.
