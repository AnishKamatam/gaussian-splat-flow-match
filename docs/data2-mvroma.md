# Reconstructing the controlled box experiment

`IMG_3257–3261` are assigned **before**, `IMG_3262–3266` **after**, by capture order at the user's direction. These are two static five-view sets, one before/after pair. All ten originals remain in `data2/`.

## Pipeline

1. Decode HEIC with libheif, apply orientation, prepare images at a 2000-pixel maximum dimension, and record checksums and camera metadata.
2. Run the official indoor MV-RoMa model with UFM priors, each image serving once as source with the other four as targets. Coarse inference is 560×560, refined correspondence fields 840×840.
3. Sample every fourth source pixel. Require confidence ≥0.8 in both directions, valid image bounds, and round-trip error ≤2 model-output pixels.
4. Consolidate nearby observations in 1-pixel cells in prepared-image coordinates, retaining averaged subpixel locations. Enforce one-to-one assignments within each image pair and run COLMAP geometric verification with a 2-pixel RANSAC threshold.
5. Initialize focal lengths from 26 mm-equivalent EXIF metadata, recover camera poses and optimize geometry with COLMAP. Retain the box and background for future alignment.
6. Remove tracks with duplicate image observations or fewer than three distinct views, then run bundle adjustment and export the colored cloud. Raw reconstructions remain in `raw_model/` for auditing.
7. Where camera recovery is successful, run CUDA PatchMatch stereo and fusion at up to 1600 pixels for denser surface coverage.

MV-RoMa's released repository supplies inference, not the paper's complete SfM reconstruction pipeline. This repository's adapter is a custom integration, not an exact reproduction of that post-processing method.

## Local preparation

Requires `heif-convert` (libheif), Pillow and NumPy. macOS `sips` produced black previews for these HEIC files, so it is not used.

```bash
python scripts/prepare_data2.py --input data2 --output outputs/data2-input
```

The output includes `manifest.json`, `contact.jpg`, and `box/{before,after}/images/*.jpg`. Existing output directories are refused to prevent accidental overwrite. Both source and prepared-image checksums are recorded; source EXIF focal and orientation information is in the manifest.

## GPU setup and reconstruction

The completed setup uses Python 3.12, the host's PyTorch 2.8.0+cu128 / torchvision 0.23.0+cu128, and PyCOLMAP CUDA 4.2.1. This differs from upstream's documented PyTorch 2.5.1+cu118; compatibility is validated by an actual inference run. The RTX 6000 Ada has 48 GB VRAM.

Copy this repository's scripts/modules and prepared inputs to a dedicated workspace, then:

```bash
bash scripts/setup_mvroma_gpu.sh /workspace/box-reconstruction
cd /workspace/box-reconstruction
OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 .venv/bin/python -u scripts/reconstruct_mvroma.py \
  --input outputs/data2-input \
  --mvroma vendor/MV-RoMa \
  --weights weights/indoor_final.pth \
  --cache outputs/mvroma-cache-v2 \
  --output outputs/mvroma-final
```

Each new reconstruction needs a fresh `--output` directory. Matching can resume from its cache; `--geometry-only` reuses complete cached predictions without loading the matcher. Cache signatures include prepared inputs, checkpoint SHA256, inference source, upstream code fingerprints, UFM weights revision and torch versions. Unprovenanced or mismatched caches are rejected.

The official checkpoint contains obsolete `conv_backbones` and rotary-period keys. These known unused keys are logged in `checkpoint_ignored_keys.json`; all live model parameters still load strictly.

```bash
.venv/bin/python scripts/densify.py \
  --sparse-run outputs/mvroma-final \
  --input-root outputs/data2-input \
  --resume-run outputs/mvroma-dense
.venv/bin/python -m unittest discover -s tests -v
```

## Outputs and interpretation

The sparse run has `summary.json`, per-state metrics, `sparse.ply`, COLMAP camera/point models, match counts and verified-pair statistics. The dense run has `dense.ply`, dense depth maps, fusion outputs and its own summary. Logs and an environment package listing are retained separately.

These clouds use independent arbitrary coordinate frames. They have not yet been aligned, scaled to physical units, or segmented into individual balls. A low reprojection error measures image consistency, not physical accuracy or completeness. Hidden ball surfaces are not recovered. Before/after states do not establish persistent ball identities or the actual intervening trajectory.

## Verified sparse results (2026-10-21)

| State | Registered views | Retained points | Mean reprojection error | Mean track length |
| --- | ---: | ---: | ---: | ---: |
| Before | 5 / 5 | 42,865 | 0.487 px | 3.51 |
| After | 5 / 5 | 87,602 | 0.405 px | 3.95 |

Every retained sparse point was audited to have finite coordinates, positive depth in its observing cameras, and observations in three to five distinct images with no duplicate image entries. Track filtering removed 1,208 ambiguous and 1,144 two-view tracks before, and 3,799 ambiguous and 443 two-view tracks after. These counts describe the sparse reconstruction; dense stereo uses separate depth-map consistency checks.

The full repository suite passes 19 tests on the GPU host. The clean point clouds and camera models are in `outputs/mvroma-final/box/{before,after}/`; sparse previews are in `outputs/mvroma-final/renders/`.

## Dense results

| State | Registered views | Dense points | Local cloud |
| --- | ---: | ---: | --- |
| Before | 5 / 5 | 223,420 | `outputs/mvroma-dense/box/before/dense.ply` |
| After | 5 / 5 | 307,055 | `outputs/mvroma-dense/box/after/dense.ply` |

Both dense clouds were downloaded and checked for finite coordinates and exact agreement with their reported vertex counts. SHA256 checksums are in `outputs/mvroma-final/deliverables-audit.json`; the sparse track/depth audit is in `outputs/mvroma-final/audit.json`. Final environment versions and test/reconstruction logs are in `outputs/mvroma-final/logs/`.

Visual inspection shows the box and substantially improved ball-surface coverage in the dense clouds. There are still holes and peripheral background fragments; these are surface reconstructions, not complete watertight ball models. The preview independently normalizes and orients each state and must not be used to measure motion. Next work is alignment on unchanged box structure, scale calibration if desired, and ball segmentation/correspondence.

The remote workspace retains matching caches and stereo depth maps at `/workspace/box-reconstruction`. The final clouds, sparse camera models, provenance and logs are saved locally; the GPU has no reconstruction jobs running.
