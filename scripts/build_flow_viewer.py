#!/usr/bin/env python3
"""Embed a sampled synthetic experiment in the interactive viewer fragment."""
import argparse
import json
from pathlib import Path
import numpy as np

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',type=Path,default=Path('outputs/synthetic-flow-v2'))
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
metrics=json.loads((args.run/'metrics.json').read_text())
data={}
for row in metrics['results']:
    name=row['material']
    p=dict(np.load(args.run/name/'prediction.npz'))
    v=dict(np.load(args.run/name/'velocities.npz'))
    ids=np.random.default_rng(12).choice(len(p['ids']),600,replace=False)
    # Same origin, rotation, and scale for source, true endpoint, learned endpoint.
    center=(p['x0'].min(axis=0)+p['x0'].max(axis=0))/2
    scale=np.linalg.norm(p['x0']-center,axis=1).max()
    def transform(x,position=True):
        a=(x-center if position else x)/scale
        return np.round(a[...,[0,2,1]]*np.array([1,1,-1]),4).tolist()
    stationary=np.where(np.linalg.norm(p['x1']-p['x0'],axis=1)==0)[0]
    stationary=np.setdiff1d(stationary,ids)[:1500]
    data[name]=dict(name=name.capitalize(),seed=row['test_seed'],
        background=transform(p['x0'][stationary]),background_rgb=p['rgb'][stationary].tolist(),
        x0=transform(p['x0'][ids]),delta=transform((p['x1']-p['x0'])[ids],False),
        arc=transform(p['arc'][ids],False),rgb=p['rgb'][ids].tolist(),
        positions=transform(p['positions'][:,ids]),velocities=transform(v['predicted'][:,ids],False),
        error=row['moving_endpoint_epe'],baseline=row['no_motion_moving_epe'],
        improvement=100*(1-row['moving_endpoint_epe']/row['no_motion_moving_epe']))
template=Path(__file__).resolve().parents[1]/'viewers/synthetic-flow.template.html'
text=template.read_text().replace('__FLOW_DATA__',json.dumps(data,separators=(',',':')))
args.output.parent.mkdir(parents=True,exist_ok=True)
args.output.write_text(text)
assert args.output.stat().st_size<1000000,'Reduce preview points to keep viewer below 1 MB'
print(args.output,args.output.stat().st_size,'bytes')
