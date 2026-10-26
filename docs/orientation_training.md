# Fresh OEM-first student–teacher training

Use `config/SHREC/train_st_te_config.yaml` for the normal SHREC pipeline.
No pretrained checkpoints are required. Version 10 prevents accidental resume of
old bin-only OEM checkpoints. The new residual head requires fresh training.

## Schedule

- Epochs 1–30: train student OEM by overlapping-bin classification, continuous residual supervision, subset consistency, and cross-shape adversarial domain loss from the first update; update its EMA OEM. ASMAE receives no forward passes, gradients, or EMA changes.
- Epochs 31–40: continue OEM angle training; ramp ASMAE loss scale and learning rate from 0.1 to 1.0. Domain adaptation continues independently of the ASMAE ramp.
- Epochs 41–70: train the full pipeline together at full correspondence weight.

Validation tests continuous relative rotations and reports refined forward/reverse
nearest-bin accuracy, actual circular angle error, coarse-only error, accuracy
within 5/10/15 degrees, large-error fraction, and classification loss. It never gates or delays the schedule. There are no training/evaluation
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

## Rotation estimator and losses

The encoder/FIM retains the released SE-ORNet structure with our GroupNorm.
No mesh faces, anatomical labels, pretrained weights, or ground-truth P2P maps
are used. This is a combination of ideas, not an exact reproduction of a paper.

The angle head has eight confidences plus a normalized cosine/sine residual per
bin in each direction. The angular coverage is 67.5 degrees for centers spaced
by 45 degrees (`angle_overlap: 1.5`). At overlap angles, classification targets
share probability uniformly among covering bins, and each covering residual
receives circular supervision. The final prediction is the most confident bin
center plus its own atan2 residual. Opposite hypotheses are not averaged.

Training and validation cover the full relative-angle circle, including boundaries.
Exact-center augmentation and exclusion intervals are superseded: nonzero residuals
need non-center training examples. Synthetic pairs provide the actual angle and
its negative for forward/reverse supervision. Validation uses private seeded,
stratified angles; it does not consume training RNG. `arbitrary_angles=False`
retains exact-grid evaluation for comparison.

Two independent student OEM subset pairs are drawn for each reference pair and
each real cross-shape pair. References receive angle supervision on both views.
The subset loss combines symmetric JS divergence of bin probabilities and
confidence-weighted agreement of per-bin residual vectors. It compares identical
underlying full inputs/rotations, with only sampling changing. It does not average
angles between modes. The extra views approximately double OEM forward work and
increase training memory. No extra ASMAE forwards are introduced.

`Angle` in the log is classification + `oem_residual_weight` * residual +
`oem_subset_consistency_weight` * subset consistency. Defaults are 1.0 and 0.1;
the raw residual and subset losses are also printed. The subset coefficient is
an experimental starting value, not a paper-validated optimum for this dataset.

Domain adaptation remains active from the first update, independent of P2P:
`domain(epoch) = domain_weight * angle_weight(epoch) / angle_weight_init`.
With the current YAML, angle weight decays 350 to 250 over ten epochs and domain
150 to approximately 107.143. The refined rotation is applied to all target
vertices with no P2P gradient through the alignment; OEM learns from its dedicated
losses. Teacher alignment, both correspondence scripts, and visualization all
use the same `align_target` helper. ASMAE still receives full shapes.

The YAML only adds `angle_overlap`, `oem_residual_weight`, and
`oem_subset_consistency_weight`. Sampling remains 70%, noise stays zero,
GroupNorm and fixed EMA remain, and the schedule stays 30 + 10 + 30.

## Research basis and limits

- [SE-ORNet](https://arxiv.org/pdf/2304.05395): pairwise orientation and synthetic/real-pair domain adaptation.
- [MultiBin, Section 4.1](https://arxiv.org/html/1612.00496v2): overlapping coarse hypotheses with sine/cosine residual refinement. Our confidence supervision is an explicit soft-label adaptation for overlap.
- [Shape–Pose Disentanglement, Section 3.3.2](https://arxiv.org/html/2204.01159v1): pose-preserving resampling consistency; adapted here to pairwise rotation distributions/residuals.

No Vector Neuron backbone, semantic torso detector, or implicit orientation-density
model is introduced. Subset agreement and a fine residual cannot guarantee that
front/back ambiguity is resolved. Synthetic validation is not a measurement of
cross-pose semantic orientation. Actual improvement requires an experiment.
