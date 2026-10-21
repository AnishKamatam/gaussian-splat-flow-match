# Data2 reconstruction reconnaissance — 2026-10-21

## Repository map

- `reconstruct.py`: standard-library COLMAP CLI driver; SIFT/exhaustive matching, mapping, largest-model selection, PLY export, optional CUDA dense stereo. Currently restricts material names to basalt/limestone/sandstone and discovers PNG/JPEG under material/state/images.
- `scripts/densify.py`: CUDA PyCOLMAP densification of existing sparse models.
- `scripts/render_clouds.py`: orthographic PLY previews, independently normalized.
- `synthetic_flow.py`: procedural geometry with persistent point IDs and prescribed trajectories.
- `train_flow.py`: supervised vector-field fitting using known synthetic displacement and arc targets. Cannot directly train on independently reconstructed real clouds.
- `scripts/export_velocities.py`, `scripts/build_flow_viewer.py`, `viewers/`: synthetic motion exports and viewers.
- `tests/`: reconstruction helpers and synthetic/model tests.
- `data/`: earlier quarry inputs and generated synthetic data.
- `data2/`: ten original HEIC images, currently untracked; no reconstructed clouds yet.

## Image inspection

All ten source images report 5712 x 4284 pixels before orientation handling.
Visual inspection of libheif-decoded previews finds two arrangements:

| Group | Files | Interpretation |
| --- | --- | --- |
| A | IMG_3257.HEIC through IMG_3261.HEIC | Five views of the first arrangement |
| B | IMG_3262.HEIC through IMG_3266.HEIC | Five views of the second arrangement |

Assigned at user direction: A is before and B is after, one experiment, using capture order. The box contains white and orange balls with repeated appearance, so match ambiguity and partial surface visibility need explicit inspection.

macOS `sips` produced black JPEG previews despite successful exit status. `heif-convert` produced visible images. Original files were not modified. Production preparation must check decoded pixel content and preserve orientation and camera metadata explicitly.

## Upstream findings

Paper: https://arxiv.org/html/2603.27542v1
Repository: https://github.com/IceTea-CV/MV-RoMa
Demo: https://github.com/IceTea-CV/MV-RoMa/blob/main/demo.py

The released demo produces dense source-to-multiple-target correspondence maps and confidence logits. It does not itself export reconstructed cameras or a point cloud. Installation documents Python >=3.10, PyTorch 2.5.1 CUDA 11.8, UFM/UniCeption, and downloaded model weights. An integration must add geometric verification, consistent track construction, camera estimation, triangulation, and bundle adjustment. Do not describe a custom integration as an exact reproduction of the paper's SfM post-processing without verifying that implementation.

## Proposed first reconstruction milestone

1. Decode original HEIC inputs to oriented images; retain original names, hashes, dimensions and metadata in a manifest. Confirm group labels.
2. Run MV-RoMa within each static five-image group, rotating the source view for coverage; cache correspondences and confidence.
3. Filter matches geometrically, consolidate observations into tracks, and use COLMAP/PyCOLMAP for cameras, triangulation and bundle adjustment.
4. Export colored PLY and camera models per state, with registered-view coverage, reprojection errors and track statistics. Aim for all five views registered; explicitly report partial reconstruction.
5. Inspect ball surfaces and box structure. Add dense stereo if sparse geometry is sound and coverage needs improvement.

Keep enough unchanged box/background structure for later alignment; do not fit before and after moving balls as one static SfM scene. Independent reconstructions have unrelated similarity frames. Later comparison requires stable anchors and known scale. Before/after endpoints alone do not establish physical intermediate trajectories or persistent ball identities.

## Execution context and remaining inputs

User GPU: one RTX 6000 Ada, 16 vCPUs, 62 GB RAM, 30 GB container disk. Use a Linux CUDA environment and measure memory on a single group before increasing resolution. SSH access supplied and verified; remote workspace is `/workspace/box-reconstruction`. Budget disk for the CUDA environment, model weights and stereo caches.

Local COLMAP is available at `/opt/homebrew/bin/colmap`. Default local Python lacks torch and Pillow; inspection used bundled Pillow and installed libheif. This describes the initial reconnaissance. See `docs/data2-mvroma.md` and `outputs/mvroma-final` for the implemented GPU pipeline and results.
