# Box alignment, ball correspondence, and endpoint motion

The interactive viewer is `outputs/box-clouds-3d.html`. Serve the `outputs` directory with `python3 -m http.server 8775 --bind 127.0.0.1 --directory outputs`, then visit `http://127.0.0.1:8775/box-clouds-3d.html`.

Both observed clouds share the **before reconstruction's arbitrary coordinate units**. Drag to orbit, shift-drag to pan, and scroll to zoom. The right-panel selector switches between observed after points and a motion hypothesis. Select a match to inspect its two IDs and a competing assignment. The time slider is normalized from 0 to 1; it is not elapsed seconds.

## Observations and extraction

There are 14 visible balls in each state, including 12 white dimpled golf balls and two orange balls. `configs/box-ball-seeds.json` records visually annotated circular extraction regions in IMG_3261 and IMG_3265. These are manual assistant annotations, not automatic detections or established cross-state identities. Dense reconstructed points are projected into those regions, filtered by color, and fitted with robust spheres. The model uses observed surfaces, not generated 3D points.

`outputs/ball-motion/cross-view-validation.jpg` shows the fitted centers projected into all ten photos; the other eight views did not supply extraction seeds. Projection circles are approximate perspective radius indicators. White-ball radii have a 1.66% relative standard deviation across both states. All four orange-ball fits have weak surface support and are explicitly flagged; their centers/radii should be treated as provisional. Low shell residual measures fit consistency, not absolute accuracy.

## Alignment

Brown pixels on the static cardboard supply SIFT descriptors associated with existing sparse 3D observations. Mutual ratio-tested correspondences produce 49 candidate 3D pairs; similarity-transform RANSAC retains 27. The after-to-before transform has scale 0.9036112. Median support residual is 0.01770 arbitrary units, approximately 0.0322 median golf-ball diameters. Conditional leave-one-out median residual is 0.01888, p95 0.03123. This cross-validation reuses RANSAC-selected support and is not independent absolute calibration.

No moved-ball correspondences are used to determine the alignment. The viewer applies a single additional orientation/scale to both states for presentation; exported PLYs retain the before reconstruction coordinate system.

## Correspondence and motion limitations

The first correspondence model minimizes total 3D center displacement under one-to-one and color constraints. **All 14 identities remain unverified.** It does not use logos, track rotation, infer impulses, or recover actual paths. A small alternative assignment gap indicates a competing global assignment; it is not a match probability. A large gap also does not establish identity because minimum travel is an assumption.

The endpoint model uses a minimum-jerk straight interpolation, `q(t)=10t³−15t⁴+6t⁵`. Positions are `a+q(t)(b−a)` and normalized velocities are `q'(t)(b−a)`. The chosen model starts/ends at rest by assumption. Five pairs overlap in the sampled interpolation by more than 10% of the smaller fitted radius. It is therefore an explicit **nonphysical baseline**, not a collision-resolved simulation or recovered blast dynamics. No elapsed time or metric calibration is available. Do not interpret normalized velocity as m/s.

The motion viewer translates sampled before-ball surfaces without rotation; static before-box points remain fixed. At t=1 this is a translated before-surface hypothesis, not the observed after surface. Choose “Observed after” to see the actual reconstructed target cloud.

## Outputs

All analysis outputs are under `outputs/ball-motion/`:

- `before_aligned.ply`, `after_aligned.ply`: full colored aligned clouds (223,420 / 307,055 points).
- `alignment.json`, `alignment_matches.npz`: transformation and feature support.
- `before_centers.csv`, `after_centers.csv`: 3D centers, radii, support, and fit quality.
- `correspondences.csv`: proposed assignments, displacement, competing target, cost gap, verification flag.
- `balls_and_motion.json`: complete ball metadata, assignment hypotheses, and collision diagnostics.
- `motion.npz`: 101 normalized time samples, 14 center positions and velocities per sample.
- `validation.json`: checks and uncertainty limitations.
- `before_IMG_*.jpg`, `after_IMG_*.jpg`: per-view projected sphere overlays.

## Reproduction

Run from the repository root in the GPU Python environment used for reconstruction (NumPy, SciPy, OpenCV, PyCOLMAP, Pillow):

```bash
python scripts/align_box_states.py
python scripts/extract_box_balls.py
python scripts/validate_ball_motion.py
python -m unittest discover -s tests
python scripts/build_box_viewer.py --output /absolute/path/box-clouds.html
```

The viewer build needs only NumPy. Its fragment is based on `viewers/box-clouds.template.html`; the standalone output is a wrapped copy. The source observations and raw reconstructions are preserved. Analysis is deterministic and does not retrain MV-RoMa or the repository's synthetic flow network.

Next refinement should use multiview logos/dimple appearance to constrain identities, improve the low-texture orange-ball reconstruction, and then test a collision-aware trajectory hypothesis. More before/after pairs or intermediate video are needed to validate a learned dynamics model.
