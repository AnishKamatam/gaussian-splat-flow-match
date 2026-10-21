# Data2 MV-RoMa Reconstruction Implementation Plan

> **For agentic workers:** Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Reconstruct and inspect the two real five-view box arrangements on the provided GPU.
**Architecture:** Preserve HEIC originals; prepare oriented images and a manifest. Run the released MV-RoMa matcher, build shared image observations from sampled dense matches, verify geometry and reconstruct cameras/points with PyCOLMAP. Cache matches independently from geometry so adjustments do not repeat expensive inference.
**Tech Stack:** Python, libheif, Pillow, NumPy, MV-RoMa/UFM, PyTorch CUDA, PyCOLMAP.
**Spec:** docs/data2-reconstruction-index.md

## Global Constraints
- Before = IMG_3257–3261; after = IMG_3262–3266, assigned by capture order at user direction.
- Keep states separate; no invented poses, correspondences or points.
- Preserve original inputs and existing quarry/synthetic pipelines.
- Report partial registration and geometry quality; point count alone is insufficient.
- User supplied the GPU and requested reconstruction; implement and run in this session.

## Review Focus
- Pixel-center coordinates and resized image dimensions must agree with COLMAP.
- Repeated ball appearance must be filtered geometrically.
- Shared observations must support multi-view tracks without duplicating one image in a track.
- Cached results must not silently mix configurations or inputs.
- Failure must retain intermediate outputs and report incomplete coverage.

### Task 1: Prepare inputs
- [x] Add `scripts/prepare_data2.py`, converting with libheif, checking decoded content, retaining source SHA256/EXIF/dimensions and generating a manifest plus contact sheet under outputs.
- [x] Inspect all ten prepared views and manifest; originals stay unchanged.

### Task 2: Match and reconstruct
- [x] Test coordinate conversion, filtering invalid/low-confidence observations, and per-image keypoint consolidation with hand-derived fixtures in `tests/test_mvroma_geometry.py`.
- [x] Implement `mvroma_geometry.py` and `scripts/reconstruct_mvroma.py`, with cached inference, database export, geometric verification, mapping and PLY/metrics export.
- [x] Set up dependencies in an isolated environment on the supplied GPU. Pin upstream commits and record installed versions.
- [x] Run tests locally and on GPU, then reconstruct both states. Inspect registration, reprojection error and point coverage.

### Task 3: Deliver results
- [x] If geometry is sound, densify with CUDA stereo for useful surface coverage.
- [x] Retrieve point clouds, camera models, metrics, logs and previews to local outputs.
- [x] Document reproduction commands, actual results and unresolved limitations.

## Progress
- SSH connected successfully after retry; RTX 6000 Ada 49140 MiB, Python 3.12, torch 2.8 CUDA 12.8 preinstalled.
- Ruling: reuse the installed CUDA torch in a system-site-packages venv first; upstream specifies 2.5.1, so record and test compatibility before changing the large base environment.
- Task 1 complete: ten images decoded, oriented, hashed and visually inspected; metadata gives 26 mm equivalent focal length for all views.
- Task 2: coordinate/filter tests RED→GREEN; GPU inference completed all ten source views (~3 seconds each, 10.55 GiB peak allocated). Live model keys load strictly; known obsolete checkpoint keys logged separately.
- Ruling: use custom reciprocal filtering, 1-pixel observation cells and COLMAP verification; not an exact implementation of the paper's SfM sampler.
- Integration failure reproduced: PyCOLMAP import_images requires an existing database. Added Database.open(...).close() before import; next run reached mapping and bundle adjustment.
- Independent reviewer found two cache-provenance gaps. Fixed missing-signature rejection (RED→GREEN) and added source/dependency fingerprints plus pinned UFM weights revision. The final run uses a fresh cache to exercise these changes.
- Final sparse run complete: before 42,865 points / 5 cameras / 0.487 px mean error; after 87,602 / 5 / 0.405 px.
- Ruling: COLMAP produced some tracks with repeated image observations. Preserve raw models, remove ambiguous and two-view tracks, and re-optimize; this trades point count for verifiable three-view support.
- All retained points passed unique-image track, finite-coordinate and positive-depth audits. Full regression suite passed 19/19 on GPU after copying the older quarry fixtures required by two existing tests.
- Dense stereo running using the successful camera models; final sparse outputs copied locally and previewed.

- Task 3 complete: dense before 223,420 points; dense after 307,055 points. Both clouds downloaded, finite/vertex-count checks passed, previews visually inspected. Saved sparse models, audit, package versions and final logs locally.
- Final verification: full GPU suite 19/19; Python compilation, shell syntax and git diff whitespace checks pass. GPU idle after completion.
- Final review fixes: missing-provenance rejection and dirty-source fingerprints covered by regression tests; all active checkpoint parameters remain strictly required. No deferred review findings.
