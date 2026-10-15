# Fresh orientation and correspondence training

Start SHREC training with random weights:

```bash
python train_st_te.py --config config/SHREC/train_st_te_config.yaml --fresh
```

No historical checkpoint is required. OEM uses GroupNorm throughout, including
its pair discriminator. It has no running statistics or inference overrides.
ASMAE regenerates appended XYZ input channels from its actual coordinates in
both training and feature extraction. Original descriptor reconstruction targets
are retained; XYZ channels were already excluded from feature masking.

The SHREC configuration reserves two shapes from the 26-shape training pool:
24 optimize the model, two validate rotation, and the existing 18 test shapes
remain outside optimization and validation. File ordering follows the existing
lexicographic split. Checkpoint metadata records the actual shape names.

OEM reference training first applies a random discrete starting rotation to a
shape, then a second, known relative rotation to its copy. The label describes
the second rotation, not either shape's absolute facing direction.

ASMAE stays frozen for at least `orientation_warmup_epochs` (10). Afterwards it
starts only when BOTH student and teacher pass the clean held-out rotation gate:
forward and reverse accuracy at least 85%, mean forward circular angle error at
most 15 degrees, for two consecutive validation checks. These are configurable
experimental thresholds, not guarantees of cross-shape alignment. If readiness
is not reached by `orientation_max_warmup_epochs` (30), training saves progress
and stops instead of automatically training P2P on unreliable alignments.

Once the gate passes, ASMAE's existing loss and learning-rate multiplier ramp
from 0.1 to 1.0 over ten epochs. All correspondence inputs use OEM predictions,
including during the ramp. OEM angle learning continues, and pair-domain
adaptation ramps with ASMAE. LGO's own ramp starts when ASMAE activates.
Known augmentation labels never supervise a different-shape angle. P2P losses
cannot differentiate through the hard angle argmax; OEM learns through its angle
and domain objectives.

`training.noise_variance: 0.0` disables all coordinate noise: no random noise is
allocated or sampled. When explicitly enabled in a later experiment, the same
variance controls training noise and additional noisy validation checks. Clean
validation always runs and controls readiness. Validation uses deterministic
starting bins `[1, 5]` (0 and 180 degrees) and all eight relative angle bins;
optional validation noise uses a private RNG and cannot perturb training RNG.

Each validation interval reports normal-evaluation rotation accuracy, reverse
accuracy, circular angle error, cross-entropy, and maximum train/eval logit gap
for student and teacher. The printed training metric is explicitly a two-shape
`train_sample`, not the accuracy over all 24 optimization shapes. The gap should be zero up to numerical precision.
These are same-shape known-rotation tests, not measurements of cross-shape
physical orientation or P2P quality. Use the notebook's correspondence and
geodesic checks separately; a low rotation loss does not guarantee low P2P error.

The main checkpoint is written atomically after every completed epoch. It contains
both ASMAEs, both OEMs, Adam state, completed epochs, configuration, shape splits,
random-generator states, readiness streak, and the actual ASMAE start epoch. Interrupting an epoch leaves the previous completed
epoch checkpoint available; that incomplete epoch is repeated when resumed.
The `_best_rotation.pth` file selects the lowest student validation circular
angle error and is not a best-geodesic-error checkpoint.

Resume this pipeline by omitting `--fresh`. `epochs` is the total desired number
of epochs, not the number to add. Training settings and splits must match, apart
from total epochs, device, checkpoint location, and maximum warm-up duration.
The latter can be increased to resume an unready run without forcing ASMAE on. Checkpoints from previous training schedules are rejected
rather than silently converted from BatchNorm to GroupNorm.

Inference retains the existing flags:

```bash
python corres_st_te.py --config config/SHREC/corres.yaml --network student --output-dir p2p_student
python corres_greedy.py --config config/SHREC/corres.yaml --network teacher --out_dir p2p_teacher
```

Both loaders use the architecture recorded in the selected version-5 checkpoint
and check feature preprocessing parameters. The visualization notebook uses this
same loader. Its default ASMAE-only checks do not require DINO.

Run the CPU regression tests in an environment with PyTorch and the training
script dependencies installed:

```bash
python -m unittest discover -s tests -v
```
