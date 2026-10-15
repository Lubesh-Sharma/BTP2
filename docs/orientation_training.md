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

During epochs 1–10, the known inverse augmentation aligns student coordinates;
the teacher receives clean coordinates. OEM learns reference-pair rotation
classification. During epochs 11–20, the probability of using student/teacher
OEM predictions increases from 0.1 to 1.0; pair-domain adaptation ramps with it.
Subsequent epochs always use predicted pairwise alignment, as inference does.
The known augmentation label is never used as an angle label for different shapes.
Correspondence losses cannot differentiate through the hard angle argmax; OEM
is trained by its angle and domain objectives.

Each validation interval reports normal-evaluation rotation accuracy, reverse
accuracy, circular angle error, cross-entropy, and maximum train/eval logit gap
for student and teacher. The gap should be zero up to numerical precision.
These are same-shape known-rotation tests, not measurements of cross-shape
physical orientation or P2P quality. Use the notebook's correspondence and
geodesic checks separately; a low rotation loss does not guarantee low P2P error.

The main checkpoint is written atomically after every completed epoch. It contains
both ASMAEs, both OEMs, Adam state, completed epochs, configuration, shape splits,
and random-generator states. Interrupting an epoch leaves the previous completed
epoch checkpoint available; that incomplete epoch is repeated when resumed.
The `_best_rotation.pth` file selects the lowest student validation circular
angle error and is not a best-geodesic-error checkpoint.

Resume this pipeline by omitting `--fresh`. `epochs` is the total desired number
of epochs, not the number to add. Training settings and splits must match, apart
from total epochs, device, and checkpoint location. Legacy checkpoints are rejected
rather than silently converted from BatchNorm to GroupNorm.

Inference retains the existing flags:

```bash
python corres_st_te.py --config config/SHREC/corres.yaml --network student --output-dir p2p_student
python corres_greedy.py --config config/SHREC/corres.yaml --network teacher --out_dir p2p_teacher
```

Both loaders use the architecture recorded in the selected version-3 checkpoint
and check feature preprocessing parameters. The visualization notebook uses this
same loader. Its default ASMAE-only checks do not require DINO.

Run the CPU regression tests in an environment with PyTorch and the training
script dependencies installed:

```bash
python -m unittest discover -s tests -v
```
