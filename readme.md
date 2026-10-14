# ASMAE with staged SE-ORNet-style orientation learning

HKS/FPS preprocessing and ASMAE remain the descriptor/correspondence algorithm.
OEM retains the three uses shown in the SE-ORNet figure: clean teacher alignment,
augmented student alignment, and known-angle same-shape supervision. Training now
stabilizes rotation estimation before it controls correspondence inputs.

## Pipeline on source X and target Y

1. Compute HKS/FPS descriptors. With k=50, ASMAE inputs have 50 descriptor channels
   plus three XYZ channels. OEM receives coordinates, not HKS.
2. Reserve validation shapes inside the training partition. SHREC uses 22 shapes
   for gradients, 4 for validation, and leaves the final 18 for testing.
3. Pretrain OEM on original X and its rotated copy X_s. Every training shape sees
   all eight rotation bins each epoch. Optional annotated cross-shape angles are
   also supervised; no native orientation labels are assumed for other pairs.
4. Evaluate OEM on fixed validation shapes, all eight rotations, in eval mode.
   Log forward and inverse accuracy separately and mean yaw error in degrees.
   Restore the best OEM and initialize the teacher OEM from those exact weights.
5. During correspondence training, call student OEM(Y_s, X_s), rotate X_s by the
   inverse predicted angle toward Y_s, then pass the aligned pair to student ASMAE.
   Call teacher OEM(Y, X), align clean X toward Y, then pass it to teacher ASMAE.
   Original X and rotated X_s still go through student OEM for angle supervision.
6. Freeze OEM for the first 10 correspondence epochs. Its encoder normalization
   statistics remain fixed throughout correspondence training, matching inference.
   Then fine-tune its angle/domain objectives at a smaller learning rate. ASMAE
   learns through reconstruction, consistency and correspondence objectives.
   The discrete angle decision still blocks correspondence gradients through
   rotation, following the reference implementation.
7. EMA updates teacher ASMAE, and updates teacher OEM during OEM fine-tuning.
   HKS is preserved; XYZ channels are rebuilt from the actual aligned coordinates.
   Descriptor consistency excludes frame-dependent XYZ.

The pair discriminator uses GroupNorm instead of BatchNorm on only two pooled
directional samples. Pair domains are same-shape = 0 and cross-shape = 1, with
gradient reversal. Domain influence starts later during pretraining and stays
small; its focal loss is not expected to approach zero.

OEM uses a deterministic sample of at most 512 vertices for its quadratic neighbor
searches; ASMAE still receives every vertex. `oem.max_points: 0` uses the full cloud.
Noise remains disabled with `training.noise_variance: 0.0`.

## Loss settings

LGO is logged from the start but initially excluded from optimization. It starts
after epoch 5 and ramps over 15 epochs; SHREC's maximum weight is 50 rather than
250. This is an initial configuration to validate, not a proven optimal weight.

The old geometric orientation formula compares world-space normal directions
across poses. It is still calculated and logged, but `orientation_weight: 0.0`
makes it diagnostic only. OEM angle loss remains active in its training stages.
Setting a positive orientation weight explicitly enables the legacy term.
Distance, contrastive, cycle and reconstruction formulas have not been replaced
in this stabilization pass. Printed active weights distinguish improving a
metric from simply disabling its influence.

## Validation and checkpoints

Correspondence validation uses fixed held-out shapes and greedy matching over
sampled query vertices against all candidate vertices. Ground-truth maps are used
only for validation, not as training labels. Where dense geodesic matrices exist,
error is normalized by sqrt(mesh area). Otherwise shortest paths on original OFF
triangle edges provide **approximate edge geodesics**, explicitly named in logs.
Unreachable predictions receive a configurable penalty rather than being omitted.

Without compatible maps/meshes, validation reports a separately named synthetic
self-rotation p2p coordinate error. That fallback does not establish cross-shape
correspondence accuracy. `validation.map_index_base` declares GT indexing (1 for
SHREC); generated predictions always use zero-based indices.

The main checkpoint is selected by the validation mapping error, not total training
loss. Separate `_last.pth` files preserve latest weights, optimizer state and epoch.
Training automatically resumes the latest staged checkpoint if present. A partly
completed interrupted epoch is replayed. OEM warm-up also retains its best weights
and optimizer state separately.

By default, both held-out OEM directions must reach 95% before correspondence
training begins. This is a configurable readiness criterion, not a guarantee of
cross-shape alignment. If warm-up does not reach it, the OEM checkpoint is kept
and correspondence training stops. Inspect the reported yaw errors, or increase
`oem.pretrain_epochs` and rerun without `--fresh` to continue warm-up. Existing
cross-shape accuracy is reported only when explicit validation angle annotations
are available.

## Run on SSH

No model code or tests were executed locally. Existing models/results remain
separate: new names end in `_oem_staged`. From the repository root:

```bash
python train_st_te.py --config config/SHREC/train_st_te_config.yaml --fresh
python corres_st_te.py --config config/SHREC/corres.yaml
# Or:
python corres_greedy.py --config config/SHREC/corres.yaml
```

Both p2p scripts load the best validated checkpoint by default. To diagnose OEM
on the SAME checkpoint, run `--alignment identity`; results go to a separate
`_identity` directory. `alignment.csv` records each source/reference pair's
predicted bin, inverse yaw and confidence. Confidence alone does not establish
rotation correctness. `--checkpoint` and `--out_dir` allow explicit overrides.

Open `checker.ipynb` or `feature_visualization.ipynb`, restart its kernel and run
cells from the top. Set the notebook method to `sinkhorn` or `greedy` to match the
script. Set `results_dir_override` to inspect identity-mode or custom output paths.
Both notebooks display the saved p2p map by default; DINO is optional and off.

`p2p_X_to_Y.txt` preserves the existing Y-to-X convention: rows are
`[Y_vertex_index, predicted_X_vertex_index]`. Rotation never reorders vertices.
The maps can be many-to-one. Same-dataset inference excludes both training and
validation filenames; matching another dataset does not reuse that split.

The full schema changed, so previous `_oem.pth` checkpoints are rejected rather
than partially loaded into the new discriminator. No test files were added.
