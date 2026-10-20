# Synthetic before/after transport experiment

This experiment replaces inconsistent AI concept sheets with directly generated,
geometrically coherent 3D material samples. It is **not** a reconstruction of the
image sheets, a calibrated rock simulation, or a physical velocity measurement.
No additional raster images are needed because the point geometry is generated directly.

## Data

`synthetic_flow.py` samples concentric quarry benches and walls. A compact smooth
region moves inward/downward according to prescribed collapse parameters. Each
point has a stable integer ID, before/after coordinates, RGB, and trajectory.
Basalt, limestone, and sandstone select appearance, not measured mechanical laws.

There are 24 training scenes per material (72 total, 2,048 points each). Three
held-out scene seeds produce 12,000-point before/after pairs. Training and test
scene seeds differ, but all come from the same simple procedural family.

Coordinates and time are dimensionless. With `d = x1 - x0` and prescribed `a`:

```
x(t) = x0 + (3t² - 2t³)d + t(1-t)a
u(t) = 6t(1-t)d + (1-2t)a
```

The path is chosen, not inferred from two endpoint clouds. Two endpoints alone
cannot identify a unique physical velocity history. Particles are persistent
material samples: no fracture, collision/contact solver, surface resampling,
occlusion, fragmentation, or mass creation is modeled.

## Model and loss

The first model was an MLP receiving `xt, x0, t, condition`. It fit sampled velocities
but drifted under ODE rollout and lost to a no-motion baseline on endpoint error.
That failed experiment remains on the Pod under `outputs/synthetic-flow-v1`.

The revised model is a **source-anchored conditional Lagrangian velocity model**.
It receives source coordinates, time, relative source position to the affected
region, and supplied collapse conditions (region center/radius, drop, spread,
and material one-hot). A compact support gate derived from the supplied region
forces velocity to zero outside it. The network accepts `xt` for the ODE interface
but deliberately ignores it. It does not use target `x1` at inference.

An MLP (14 → 128 → 128 → 128 → 3, SiLU) minimizes squared analytic velocity error
at random times. Batches equally sample all points and moving points to avoid
being dominated by stationary background. Moving-point labels are used for
training sampling and evaluation only, not as inference inputs.

This is a supervised conditional flow-matching-style baseline, not an unpaired
optimal-transport solver, full generative model, or learned material physics.
It regresses vector fields along prescribed paths, related to the objective in
[Lipman et al., Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747).
Inference uses a midpoint ODE integrator, 64 steps, and stores nine states.
A 128-step comparison checks integration sensitivity.

## Evaluation and interpretation

After inspecting v1's failure on seeds 9001/9002/9003, the architecture was changed.
The revised evaluation uses fresh seeds 9101/9102/9103. These were evaluated once
for v2; they are not evidence of broad out-of-distribution generalization.

`outputs/synthetic-flow-v2/metrics.json` contains mean Euclidean endpoint error
(EPE), moving/stationary subsets, no-motion baseline, velocity error on true paths,
and integration-refinement error. Moving means endpoint displacement > 0.005 in
synthetic units. Report moving-subset error because full-cloud error is dominated
by stationary points. Values are per point, not per coordinate.

There is only one held-out scene per material. There is no confidence interval,
physical calibration, conservation guarantee, or evidence of transfer to real scans.
The known affected-region gate and supplied motion parameters make this a deliberately
easy sanity check. The next research step is varied scene-level evaluation with
unknown motion conditions, and then uncertain/partial correspondence for real clouds.
Gaussian splatting is an optional appearance representation, not required for this
point-transport experiment.

## Reproduce

Use a CUDA-compatible PyTorch installation, NumPy, and optionally Pillow for previews.
The GPU run used system PyTorch 2.8.0+cu128 on an RTX 6000 Ada.

```bash
python3 synthetic_flow.py --output data/synthetic-v2
python3 train_flow.py --data data/synthetic-v2 --output outputs/new-flow-run --steps 3000
python3 -m unittest discover -s tests -v
```

The output directory must be new to avoid overwriting experiments. The generator
is deterministic by scene seed; CUDA training results can vary slightly across
hardware/runs. All raw positions remain in common coordinates within a pair.

Each material's output contains `before.ply`, `after_true.ply`,
`after_predicted.ply`, and `prediction.npz` with matching IDs and predicted paths.
The run also includes model weights and metrics. The generator writes analytic
ground-truth positions and velocities separately for each held-out scene.

Export learned velocity vectors along the predicted paths (and corresponding
analytic target vectors) with:

```bash
python3 scripts/export_velocities.py --run outputs/new-flow-run --data data/synthetic-v2
```

This creates `{material}/velocities.npz` with `predicted`, `target`, `times`,
`target_positions`, and stable `ids`. The interactive preview can be rebuilt with:

```bash
python3 scripts/build_flow_viewer.py --run outputs/new-flow-run --output outputs/new-flow-run/viewer.html
```

The viewer source is an inline visualization fragment; it is designed for the
Codex visualization renderer. Full 12,000-point PLY and NPZ exports remain the
analysis data; the preview samples tracked points and static context for size.
