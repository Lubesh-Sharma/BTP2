# ASMAE with SE-ORNet-style orientation alignment

The orientation pipeline follows the three OEM roles in the figure and the
SHREC path of the local SE-ORNet implementation in
`SE-ORNet/models/modules/dgcnn_with_angle.py` and
`SE-ORNet/models/runners/PointCorrWithAngle.py`. ASMAE and its correspondence
objectives remain the algorithm after alignment.

## Training on two shapes X and Y

1. Preprocessing computes FPS-anchor heat descriptors and appends XYZ channels.
   Coordinates are centered and normalized to a unit sphere before augmentation.
2. The student receives a randomly rotated source X_s and the unchanged target Y_s.
   Rotation uses eight discrete Y-axis bins.
   Gaussian noise is disabled by `training.noise_variance: 0.0` in every dataset
   training YAML. Setting a positive variance enables noise on both clouds.
   To match the figure's source/target naming, OEM receives (Y_s, X_s): the
   reference implementation always rotates its second input toward its first.
3. The student OEM encodes both augmented clouds with shared EdgeConv weights.
   Its cross-shape orientation layers predict their relative rotation. The
   highest-scoring angle bin rotates X_s toward Y_s's frame before ASMAE.
4. The aligned pair enters student ASMAE: adaptive point/feature masking,
   local attention encoder, cross-attention decoder, and descriptor reconstruction.
   Clean embedding extraction on the same aligned pair supplies the existing
   contrastive, cycle, LGO, distortion, and geometric orientation objectives.
5. The clean X/Y pair passes through the teacher's own EMA OEM as (Y, X), which
   aligns source X to target Y before teacher ASMAE. Teacher predictions supervise the student's
   descriptor predictions. XYZ is excluded from consistency because student
   and teacher alignments can differ (and noise can differ if enabled later).
6. An additional same-shape pair (clean X, rotated X_s) goes through student OEM.
   Its known augmentation angle supervises both directional angle classifiers.
   Cross-shape pairs are not assigned an invented ground-truth angle.
7. As in SE-ORNet, an OEM pair discriminator with gradient reversal distinguishes
   same-shape rotation pairs (domain 0) from cross-shape pairs (domain 1).
   This encourages the angle representation to transfer between pair types.
8. Backpropagation updates student ASMAE and student OEM. OEM learns through
   angle/domain losses; the discrete angle choice blocks correspondence gradients
   through the rotation decision, matching the reference implementation.
   EMA updates both teacher ASMAE and teacher OEM, including normalization buffers.

The common alignment implementation is `models/oriented_pair.py`. It replaces
the final XYZ feature channels with `(aligned_unit_sphere_xyz + 1) / 2` and
preserves descriptor channels and vertex ordering. Both ASMAE's position tensor
and appended XYZ therefore describe the same frame.

## Evaluation and p2p mapping

New shapes X/Y pass through the saved student OEM as (Y, X). The predicted inverse
rotation aligns source X to target Y before the ASMAE encoder. Source features are
recomputed for each target because source alignment is pair-dependent. Then
cosine-normalized embeddings produce p2p assignments through
either Sinkhorn (`corres_st_te.py`) or nearest similarity (`corres_greedy.py`).
Teacher, augmentation, angle labels, and domain losses are unnecessary at inference.
The discriminator is still computed by OEM's shared forward function but is unused.
`test_st_te.py` uses the same alignment before reconstruction testing; its MAE
is a reconstruction metric, not a correspondence accuracy metric.

Both p2p scripts default to the saved held-out partition and at most 20 shapes.
The greedy script's `--n_shapes -1` explicitly includes all shapes. The split
assumes the same dataset directory and sorted filenames as training.

P2p output retains the existing convention: `p2p_X_to_Y.txt` stores two columns,
`[Y_vertex_index, predicted_X_vertex_index]`, i.e. a Y-to-X map. Argmax assignments
can be many-to-one. Rotation does not reorder vertices or change mesh indexing.

The OEM predicts eight discrete yaw bins (45-degree steps), not arbitrary 3D
rotation. Correct alignment and correspondence on unseen shapes must be measured;
the design alone does not guarantee them. Existing distortion/orientation loss
formulas remain unchanged, now evaluated on normalized aligned coordinates.
This follows the SHREC/figure pair-supervision pattern; other dataset branches in
the original SE-ORNet runner use different angle-supervision schemes.

## Running on the SSH machine

New checkpoints use `_oem.pth` filenames in every dataset config. Previous
checkpoints are preserved, but incompatible pipelines are rejected. Train the new
pipeline from scratch. For SHREC, run from the repository root:

```bash
python train_st_te.py --config config/SHREC/train_st_te_config.yaml --fresh
python test_st_te.py --config config/SHREC/test_st_te_config.yaml
python corres_st_te.py --config config/SHREC/corres.yaml
python corres_greedy.py --config config/SHREC/corres.yaml
```

Checkpoints contain student and teacher ASMAE/OEM weights, the pipeline version,
and training split index. Loading restores weights; optimizer state and epoch
position are not saved by this training script.

The default run order is training, either p2p script, then `checker.ipynb` or
`feature_visualization.ipynb`. Scripts and notebooks default to SHREC. For another
dataset, pass its YAML to the scripts and select it in the notebook settings.
The checkpoint records the actual model configuration, descriptor geometry,
training directory, and reserved training filenames for matching.

In the notebooks, choose `sinkhorn` after `corres_st_te.py` or `greedy` after
`corres_greedy.py`. Both notebooks display the saved map by default and select
shapes from generated results. No flattening, truncation, index-base guessing,
or clipping is used. The feature notebook's DINO features are disabled by default;
explicitly recomputed greedy matches are available with `USE_SAVED_P2P=False`.
Its feature plots align each cloud to the first displayed shape through OEM.
New p2p output directories end in `_oem` to keep prior predictions separate.
Start each notebook with a fresh kernel and run cells from the top.

The requested test files have been removed. No project code was executed locally.
