# Before / after quarry point clouds

Reconstruct basalt, limestone, and sandstone separately for both scene states.
The input is `data/split/{material}/{before|after}/images/01.png` through `10.png`.
The Python driver uses only the standard library (Python 3.9+); COLMAP is an
external executable. This is photogrammetric reconstruction, not NeRF training.

## GPU machine setup

Copy this repository **including `data/split`** to the GPU machine. The images
are currently local files; cloning an uncommitted repository will not copy them.

Install [COLMAP](https://colmap.github.io/install.html) on that machine.
For dense reconstruction use an NVIDIA CUDA GPU, working NVIDIA drivers, and a
COLMAP build with CUDA enabled. A GPU alone is insufficient; some packaged
COLMAP builds omit CUDA. Apple/AMD GPUs cannot run this dense stage.
On Linux check the driver with `nvidia-smi`, then check COLMAP:

```bash
colmap -h
python3 reconstruct.py --check
```

`--check` validates the six input folders and CLI GPU-option compatibility; it
does not prove CUDA works or that images can be reconstructed. The driver detects
both older `SiftExtraction/SiftMatching` and newer `FeatureExtraction/FeatureMatching`
GPU flag names. Headless runs set `QT_QPA_PLATFORM=offscreen`.

## Run

Start with a sparse reconstruction of one scene to assess consistency:

```bash
python3 reconstruct.py --materials basalt --states before
```

Generate sparse **and dense** point clouds for all three before/after pairs:

```bash
python3 reconstruct.py --dense
```

Or generate all six sparse clouds without CUDA:

```bash
python3 reconstruct.py --cpu
```

Other options: `--colmap /path/to/colmap`, `--gpu-index 0` (dense stereo device),
`--max-image-size 1600`, `--input-root data/split`, `--output-root outputs`.
The size limit never adds real image detail; our crops are only about 330 × 390 px.
Feature extraction/matching uses COLMAP's default GPU selection unless `--cpu`.

Every invocation writes a fresh `outputs/<UTC timestamp>/` directory:

```text
summary.json
basalt/
  before/
    database.db
    sparse/           # COLMAP cameras, poses, points; may contain multiple models
    sparse.ply        # Largest model by registered camera count, then point count
    sparse_result.json
    dense/            # Undistorted images and stereo depth maps, with --dense
    dense.ply         # Colored fused cloud, with --dense
    result.json
    logs/
  after/              # Same structure
limestone/            # Same before/after structure
sandstone/            # Same before/after structure
```

Each set runs feature extraction, exhaustive matching, incremental mapping, and
PLY export. Dense mode adds image undistortion, geometric PatchMatch, and stereo
fusion, following the [COLMAP CLI workflow](https://colmap.github.io/cli.html).
Intrinsics are estimated per image because sheet crops differ in dimensions and
unknown principal points. No camera poses are invented. All models are retained;
only the largest connected model is exported, so check registered image coverage.

Failures are recorded per set, and remaining sets are still attempted. The process
exits nonzero if any set fails. Read `summary.json` and individual logs. If dense
processing fails, `sparse.ply` and `sparse_result.json` remain available. A nonempty
cloud and three registered cameras are minimal output checks, not proof of quality.
Open PLY files in CloudCompare, MeshLab, or COLMAP to inspect them.

## What these outputs mean

The source sheets explicitly label these as AI-generated concept views with
geometric consistency not guaranteed. Ten low-resolution views may not provide
consistent feature tracks. A failed reconstruction is a valid result; this code
does not replace it with fabricated points. More GPU memory will not fix changing
geometry between views.

Before and after are reconstructed independently. Their scale, rotation, and
translation are unrelated, and the view numbers are not verified camera pairs.
These clouds are not yet aligned or suitable for physical change/volume
measurement. Alignment needs verified correspondences on unchanged surroundings;
metric measurements also need a known scale. There is no alignment stage here.
COLMAP sparse models can later initialize Gaussian splatting if reconstruction
quality is adequate.

## Tests

```bash
python3 -m unittest discover -s tests -v
```

Tests cover model selection, empty results, dataset discovery, and continuing
after one scene fails. They do not replace a real COLMAP/GPU integration run.

## Dense reconstruction with the official CUDA Python wheel

On a Linux NVIDIA host you can densify an existing sparse run without compiling
the COLMAP CLI. The [official PyCOLMAP docs](https://colmap.github.io/pycolmap/index.html)
provide the `pycolmap-cuda12` package:

```bash
uv venv .venv
uv pip install --python .venv/bin/python pycolmap-cuda12==4.2.1
.venv/bin/python scripts/densify.py --sparse-run outputs/<sparse-run> --input-root data/split
```

Copy the entire sparse-run directory and split images to that host first. This
runner resolves models relative to `--sparse-run`, so reports containing paths
from another machine are supported. Sets with fewer than three recovered cameras
are skipped. Results and dense PLY files go in `outputs/dense-<timestamp>/`.
The process exits nonzero if any scene is skipped or fails; the summary distinguishes
these cases. This command densifies recovered cameras; it does not retry sparse mapping.

For an orthographic preview (requires NumPy and Pillow):

```bash
python3 scripts/render_clouds.py outputs/<sparse-run>
python3 scripts/render_clouds.py outputs/<dense-run> --dense
```

Previews retain all points, normalize each cloud separately, and show three PCA
projections. They do not align before and after or provide physical scale.

## Learning before/after point motion

See [SYNTHETIC_EXPERIMENT.md](SYNTHETIC_EXPERIMENT.md) for the controlled synthetic
benchmark, dataset generation, GPU training, evaluation, and limitations. This
uses directly generated coherent geometry and persistent point IDs; it does not
claim to recover physical motion from the image sheets.
