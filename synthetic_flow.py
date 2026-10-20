#!/usr/bin/env python3
"""Controlled synthetic point transport, not calibrated rock/blast physics.

Coordinates are dimensionless; t spans a normalized transition [0, 1].
Point IDs are persistent material samples, not independently resampled surfaces.
"""
import argparse
import json
from pathlib import Path
import numpy as np

MATERIALS = ['basalt', 'limestone', 'sandstone']
TRAIN_SEEDS = tuple(range(100, 124))
TEST_SEEDS = (9101, 9102, 9103)


def scene(material, seed, count):
    rng = np.random.default_rng(seed + 100000 * material)
    angle = rng.uniform(-np.pi, np.pi)
    center = .58 * np.array([np.cos(angle), np.sin(angle)])
    reach, drop, spread = rng.uniform(.34, .48), rng.uniform(.22, .34), rng.uniform(.14, .24)
    # Concentric terraces: horizontal benches plus vertical exposed walls.
    levels = rng.integers(0, 4, count)
    theta = rng.uniform(-np.pi, np.pi, count)
    inner = .26 + .18 * levels
    wall = rng.random(count) < .38
    radius = np.sqrt(rng.uniform(inner**2, (inner+.18)**2))
    radius[wall] = inner[wall]
    z = -.54 + .135*levels
    z[wall] -= rng.uniform(0,.135,wall.sum())
    z += .006*np.sin(theta*13+radius*9) + rng.normal(0,.002,count)
    x0 = np.column_stack([radius*np.cos(theta),radius*np.sin(theta),z])
    local = x0[:,:2]-center
    dist2 = np.sum(local**2,axis=1)/reach**2
    # Compact smooth local deformation. Parameters are supplied as conditioning.
    weight = np.maximum(1-dist2,0)**2
    d = np.zeros_like(x0)
    inward = -center/np.linalg.norm(center)
    d[:,:2] = weight[:,None]*(inward*spread + local*.28)
    d[:,2] = -drop*weight
    arc = np.zeros_like(x0)
    arc[:,2] = .10*weight
    palette = np.array([[86,91,94],[179,174,151],[174,113,80]])[material]
    shade = rng.uniform(.72,1.20,(count,1)) + .07*np.sin(theta[:,None]*9)
    rgb = np.clip(palette*shade,0,255).astype(np.uint8)
    condition = np.concatenate([center,[reach,drop,spread],np.eye(3)[material]])
    return dict(x0=x0,x1=x0+d,displacement=d,arc=arc,rgb=rgb,
                condition=condition,ids=np.arange(count,dtype=np.int64),seed=np.array(seed),material=np.array(material))


def path(s, t):
    q = t*t*(3-2*t)
    return s['x0'] + q*s['displacement'] + t*(1-t)*s['arc']


def velocity(s, t):
    return 6*t*(1-t)*s['displacement'] + (1-2*t)*s['arc']


def write_ply(filename, xyz, rgb):
    filename.parent.mkdir(parents=True,exist_ok=True)
    points = np.empty(len(xyz),dtype=[('xyz','<f4',(3,)),('rgb','u1',(3,))])
    points['xyz'],points['rgb'] = xyz,rgb
    header = ('ply\nformat binary_little_endian 1.0\nelement vertex '+str(len(xyz))+ '\n'
              'property float x\nproperty float y\nproperty float z\n'
              'property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n')
    with filename.open('wb') as f:
        f.write(header.encode('ascii'))
        f.write(points.tobytes())


def generate(root, train_points=2048, test_points=12000):
    root.mkdir(parents=True,exist_ok=True)
    manifest = {'synthetic':True,'units':'dimensionless','time':'normalized [0,1]',
                'trajectory':'x(t)=x0+(3t^2-2t^3)*(x1-x0)+t*(1-t)*arc',
                'limitations':['Analytic deformation, not physics simulation',
                               'No fracture, occlusion, mass creation, or surface resampling',
                               'Material labels select appearance; not calibrated mechanical properties'],
                'train_scenes':[],'test_scenes':[]}
    for material,name in enumerate(MATERIALS):
        for split,seeds,n in [('train',TRAIN_SEEDS,train_points),('test',[TEST_SEEDS[material]],test_points)]:
            for seed in seeds:
                s=scene(material,seed,n)
                folder=root/split/name/str(seed)
                folder.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(folder/'scene.npz',**s)
                manifest[split+'_scenes'].append(str((folder/'scene.npz').relative_to(root)))
                if split=='test':
                    write_ply(folder/'before.ply',s['x0'],s['rgb'])
                    write_ply(folder/'after.ply',s['x1'],s['rgb'])
                    times=np.linspace(0,1,21)
                    np.savez_compressed(folder/'ground_truth.npz',times=times,
                                        positions=np.stack([path(s,t) for t in times]),
                                        velocities=np.stack([velocity(s,t) for t in times]),ids=s['ids'])
    (root/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Generated {len(manifest["train_scenes"])} training scenes and 3 held-out scene pairs in {root}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('data/synthetic'))
    parser.add_argument('--train-points',type=int,default=2048)
    parser.add_argument('--test-points',type=int,default=12000)
    args=parser.parse_args()
    generate(args.output,args.train_points,args.test_points)
