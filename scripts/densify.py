#!/usr/bin/env python3
"""Densify existing COLMAP models using the official CUDA PyCOLMAP wheel.

Install: uv pip install pycolmap-cuda12==4.2.1 (Linux/NVIDIA CUDA only).
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from reconstruct import best_model, ply_vertices


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sparse-run', required=True, type=Path)
    parser.add_argument('--input-root', type=Path, default=Path('data/split'))
    parser.add_argument('--output-root', type=Path, default=Path('outputs'))
    parser.add_argument('--gpu-index', default='0')
    parser.add_argument('--resume-run', type=Path, help='Reuse a dense workspace and cached depth maps')
    args = parser.parse_args()
    import pycolmap
    if not pycolmap.has_cuda:
        parser.error('This PyCOLMAP build has no CUDA support')
    run = args.resume_run.resolve() if args.resume_run else args.output_root.resolve() / ('dense-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ'))
    run.mkdir(parents=True, exist_ok=bool(args.resume_run))
    results = []
    for source in json.loads((args.sparse_run / 'summary.json').read_text()):
        material, state = source['material'], source['state']
        work = run / material / state
        work.mkdir(parents=True, exist_ok=bool(args.resume_run))
        result = dict(material=material, state=state, workspace=str(work))
        try:
            sparse = args.sparse_run / material / state / 'sparse'
            model, registered, points = best_model(sparse)
            result.update(registered_images=registered, sparse_points=points,
                          input_images=source.get('input_images', 10))
            if registered < 3:
                result.update(status='skipped', error='Fewer than three registered images')
            else:
                images = (args.input_root / material / state / 'images').resolve()
                print(f'Densifying {material}/{state}: {registered} cameras, {points} points', flush=True)
                if not (work / 'stereo' / 'patch-match.cfg').exists():
                    pycolmap.undistort_images(work, model.resolve(), images)
                pycolmap.patch_match_stereo(work, options=pycolmap.PatchMatchOptions(
                    gpu_index=args.gpu_index, geom_consistency=True, max_image_size=1600))
                cloud = work / 'dense.ply'
                fused_model = work / 'fused_model'
                fused_model.mkdir(exist_ok=True)
                fused = pycolmap.stereo_fusion(fused_model, work, input_type='geometric')
                fused.export_PLY(cloud)
                result.update(status='success', dense_points=ply_vertices(cloud), dense_cloud=str(cloud))
                print(json.dumps(result), flush=True)
        except Exception as exc:
            result.update(status='failed', error=str(exc))
            traceback.print_exc()
        results.append(result)
        (work / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        (run / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
    print(f'Results: {run / "summary.json"}', flush=True)
    return int(any(r['status'] != 'success' for r in results))


if __name__ == '__main__':
    sys.exit(main())
