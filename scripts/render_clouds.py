"""Render COLMAP binary PLY clouds as three orthographic projections.

Requires numpy and Pillow. No points are removed or synthesized.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def read_ply(path):
    with path.open('rb') as f:
        header = []
        while True:
            line = f.readline()
            if not line:
                raise ValueError('Missing PLY header')
            header.append(line.decode('ascii').strip())
            if line.strip() == b'end_header':
                break
        if 'format binary_little_endian 1.0' not in header:
            raise ValueError('Expected binary little-endian PLY')
        count = int(next(x.split()[-1] for x in header if x.startswith('element vertex ')))
        types = {'float': '<f4', 'double': '<f8', 'uchar': 'u1'}
        properties = []
        in_vertices = False
        for line in header:
            if line.startswith('element '):
                in_vertices = line.startswith('element vertex ')
            elif in_vertices and line.startswith('property '):
                _, kind, name = line.split()
                properties.append((name, types[kind]))
        points = np.frombuffer(f.read(), dtype=np.dtype(properties), count=count)
        xyz = np.column_stack([points[k] for k in ('x', 'y', 'z')]).astype(float)
        if not np.isfinite(xyz).all():
            raise ValueError('Nonfinite coordinates')
        return xyz, np.column_stack([points[k] for k in ('red', 'green', 'blue')])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--dense', action='store_true')
    args = parser.parse_args()
    results = json.loads((args.run / 'summary.json').read_text())
    rows = [r for r in results if r['status'] == 'success']
    if not rows:
        raise SystemExit('No successful point clouds to render')
    width, height = 1560, 155 + len(rows) * 475
    canvas = Image.new('RGB', (width, height), '#101722')
    d = ImageDraw.Draw(canvas)
    fontpath = '/System/Library/Fonts/Supplemental/Arial.ttf'
    def font(size):
        return ImageFont.truetype(fontpath, size)
    d.text((30, 20), 'COLMAP ' + ('dense' if args.dense else 'sparse') + ' point clouds — AFTER sets', font=font(30), fill='#f0f4fa')
    d.text((30, 63), 'Actual reconstructed points • orthographic projections • original RGB colors • no surfaces or interpolation', font=font(18), fill='#bac7d8')
    d.text((30, 94), 'Each cloud centered and rotated to its own PCA axes. Arbitrary units; rows are not spatially aligned.', font=font(18), fill='#bac7d8')
    axes = [(0, 1, 2, 'Principal axes 1 / 2'), (0, 2, 1, 'Principal axes 1 / 3'), (1, 2, 0, 'Principal axes 2 / 3')]
    for row, result in enumerate(rows):
        xyz, rgb = read_ply(args.run / result['material'] / result['state'] / ('dense.ply' if args.dense else 'sparse.ply'))
        centered = xyz - xyz.mean(axis=0)
        _, _, rotation = np.linalg.svd(centered, full_matrices=False)
        coords = centered @ rotation.T
        coords -= (coords.min(axis=0) + coords.max(axis=0)) / 2
        scale = 320 / max(np.ptp(coords, axis=0).max(), 1e-12)
        y = 145 + row * 475
        title = f"{result['material'].capitalize()} / {result['state']}     {result['registered_images']}/10 cameras     {len(xyz):,} points"
        d.text((30, y), title, font=font(23), fill='#f0f4fa')
        for col, (a, b, depth, label) in enumerate(axes):
            left, top = 30 + col * 510, y + 42
            d.rectangle((left, top, left + 490, top + 395), fill='#1b2635', outline='#35465b')
            for i in np.argsort(coords[:, depth]):
                px = left + 245 + coords[i, a] * scale
                py = top + 195 - coords[i, b] * scale
                color = tuple(int(v) for v in rgb[i])
                d.ellipse((px-2.5, py-2.5, px+2.5, py+2.5), fill=color, outline='#8b99a8', width=1)
            d.text((left + 12, top + 365), label + ' (arbitrary units)', font=font(16), fill='#bac7d8')
    out = args.run / 'renders'
    out.mkdir(exist_ok=True)
    canvas.save(out / 'point_clouds.png')
    print(out / 'point_clouds.png')


if __name__ == '__main__':
    main()
