#!/usr/bin/env python3
"""Reconstruct independent before/after point clouds using the COLMAP CLI."""
import argparse
import json
import os
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import sys
from datetime import datetime, timezone

MATERIALS = ['basalt', 'limestone', 'sandstone']
STATES = ['before', 'after']


def discover(root, materials, states):
    sets = []
    for material in materials:
        for state in states:
            folder = root / material / state / 'images'
            images = [p for p in folder.glob('*') if p.suffix.lower() in {'.png', '.jpg', '.jpeg'}]
            if len(images) < 3:
                raise RuntimeError(f'{folder}: need at least 3 images, found {len(images)}')
            sets.append((material, state, folder.resolve()))
    return sets


def binary_count(path):
    with path.open('rb') as stream:
        return struct.unpack('<Q', stream.read(8))[0]


def best_model(sparse):
    candidates = []
    for folder in sparse.iterdir():
        if all((folder / name).is_file() for name in ('images.bin', 'points3D.bin', 'cameras.bin')):
            candidates.append((folder, binary_count(folder / 'images.bin'), binary_count(folder / 'points3D.bin')))
    if not candidates:
        raise RuntimeError('No sparse model recovered. Inspect matching/mapper logs; views may be inconsistent.')
    return max(candidates, key=lambda item: (item[1], item[2]))


def ply_vertices(path):
    count = 0
    with path.open('rb') as stream:
        if stream.readline().strip() != b'ply':
            raise RuntimeError(f'Invalid PLY: {path}')
        for _ in range(1000):
            line = stream.readline()
            if line.startswith(b'element vertex '):
                count = int(line.split()[-1])
            if line.strip() == b'end_header':
                break
        else:
            raise RuntimeError(f'Invalid PLY header: {path}')
    if count < 1:
        raise RuntimeError(f'Point cloud is empty: {path}')
    return count


class Colmap:
    def __init__(self, executable):
        self.executable = executable
        self.env = dict(os.environ, QT_QPA_PLATFORM='offscreen')

    def gpu_option(self, command, modern, legacy):
        result = subprocess.run([self.executable, command, '-h'], env=self.env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for option in (modern, legacy):
            if option in result.stdout:
                return option
        raise RuntimeError(f'Cannot detect GPU option for {command}: {result.stdout[-1000:]}')

    def run(self, command, log, **options):
        args = [self.executable, command]
        for key, value in options.items():
            args.extend(['--' + key, str(value)])
        print(shlex.join(args), flush=True)
        with log.open('w') as stream:
            stream.write(shlex.join(args) + '\n')
            stream.flush()
            result = subprocess.run(args, env=self.env, stdout=stream, stderr=subprocess.STDOUT)
        if result.returncode:
            raise RuntimeError(f'{command} exited {result.returncode}; see {log}')


def reconstruct(colmap, images, work, args):
    logs = work / 'logs'
    logs.mkdir()
    sparse = work / 'sparse'
    sparse.mkdir()
    database = work / 'database.db'
    extract_gpu = colmap.gpu_option('feature_extractor', 'FeatureExtraction.use_gpu', 'SiftExtraction.use_gpu')
    match_gpu = colmap.gpu_option('exhaustive_matcher', 'FeatureMatching.use_gpu', 'SiftMatching.use_gpu')
    colmap.run('feature_extractor', logs / '01_features.log', **{
        'database_path': database, 'image_path': images,
        'ImageReader.camera_model': 'SIMPLE_RADIAL',
        # Crops have different dimensions/principal points: do not force shared intrinsics.
        'ImageReader.single_camera': 0, extract_gpu: int(not args.cpu),
    })
    colmap.run('exhaustive_matcher', logs / '02_matching.log', **{
        'database_path': database, match_gpu: int(not args.cpu),
    })
    colmap.run('mapper', logs / '03_mapper.log', **{
        'database_path': database, 'image_path': images, 'output_path': sparse,
    })
    model, registered, points = best_model(sparse)
    if registered < 3 or points < 1:
        raise RuntimeError(f'Insufficient reconstruction: {registered} registered images, {points} points')
    cloud = work / 'sparse.ply'
    colmap.run('model_converter', logs / '04_export.log', **{
        'input_path': model, 'output_path': cloud, 'output_type': 'PLY',
    })
    summary = {'registered_images': registered, 'input_images': len(list(images.glob('*'))),
               'sparse_points': ply_vertices(cloud), 'selected_model': str(model),
               'sparse_cloud': str(cloud)}
    # Persist sparse success even if dense reconstruction subsequently fails.
    (work / 'sparse_result.json').write_text(json.dumps(summary, indent=2) + '\n')
    if args.dense:
        dense = work / 'dense'
        dense.mkdir()
        colmap.run('image_undistorter', logs / '05_undistort.log', **{
            'image_path': images, 'input_path': model, 'output_path': dense,
            'output_type': 'COLMAP', 'max_image_size': args.max_image_size,
        })
        colmap.run('patch_match_stereo', logs / '06_stereo.log', **{
            'workspace_path': dense, 'workspace_format': 'COLMAP',
            'PatchMatchStereo.geom_consistency': 'true',
            'PatchMatchStereo.max_image_size': args.max_image_size,
            'PatchMatchStereo.gpu_index': args.gpu_index,
        })
        dense_cloud = work / 'dense.ply'
        colmap.run('stereo_fusion', logs / '07_fusion.log', **{
            'workspace_path': dense, 'workspace_format': 'COLMAP',
            'input_type': 'geometric', 'output_path': dense_cloud,
        })
        summary.update(dense_cloud=str(dense_cloud), dense_points=ply_vertices(dense_cloud))
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input-root', type=Path, default=Path('data/split'))
    parser.add_argument('--output-root', type=Path, default=Path('outputs'))
    parser.add_argument('--materials', nargs='+', choices=MATERIALS, default=MATERIALS)
    parser.add_argument('--states', nargs='+', choices=STATES, default=STATES)
    parser.add_argument('--colmap', default='colmap', help='COLMAP executable path')
    parser.add_argument('--dense', action='store_true', help='Also run CUDA PatchMatch stereo and fusion')
    parser.add_argument('--cpu', action='store_true', help='Use CPU feature extraction/matching (sparse only)')
    parser.add_argument('--gpu-index', default='0', help='CUDA device index for dense stereo')
    parser.add_argument('--max-image-size', type=int, default=1600)
    parser.add_argument('--check', action='store_true', help='Validate inputs and executable without reconstruction')
    args = parser.parse_args(argv)
    if args.cpu and args.dense:
        parser.error('--dense requires CUDA; do not combine with --cpu')
    if args.max_image_size < 1:
        parser.error('--max-image-size must be positive')
    try:
        sets = discover(args.input_root, args.materials, args.states)
        for material, state, folder in sets:
            print(f'{material}/{state}: {len(list(folder.glob("*")))} images in {folder}')
        executable = shutil.which(args.colmap)
        if not executable:
            raise RuntimeError('COLMAP not found. Install COLMAP and ensure it is on PATH, or pass --colmap.')
        colmap = Colmap(executable)
        if args.check:
            colmap.gpu_option('feature_extractor', 'FeatureExtraction.use_gpu', 'SiftExtraction.use_gpu')
            colmap.gpu_option('exhaustive_matcher', 'FeatureMatching.use_gpu', 'SiftMatching.use_gpu')
            print('Inputs and COLMAP CLI checked. CUDA capability is only verified by an actual GPU run.')
            return 0
    except (RuntimeError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        return 1
    # Unique run directories prevent old databases/models from contaminating new results.
    run = args.output_root.resolve() / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    run.mkdir(parents=True)
    results = []
    for material, state, images in sets:
        work = run / material / state
        work.mkdir(parents=True)
        result = {'material': material, 'state': state, 'workspace': str(work)}
        try:
            result.update(reconstruct(colmap, images, work, args), status='success')
        except (RuntimeError, OSError, ValueError, struct.error) as exc:
            result.update(status='failed', error=str(exc))
            print(f'{material}/{state}: {exc}', file=sys.stderr)
        results.append(result)
        (work / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        (run / 'summary.json').write_text(json.dumps(results, indent=2) + '\n')
    print(f'Results: {run / "summary.json"}')
    return int(any(r['status'] == 'failed' for r in results))


if __name__ == '__main__':
    sys.exit(main())
