#!/usr/bin/env python3
"""Export model velocity vectors along the predicted ODE paths and analytic targets."""
import argparse
from pathlib import Path
import sys
import json
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from train_flow import VelocityField, load_scene, target_path, target_velocity

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--run',type=Path,required=True)
parser.add_argument('--data',type=Path,default=Path('data/synthetic-v2'))
args=parser.parse_args()
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=VelocityField().to(device)
model.load_state_dict(torch.load(args.run/'model.pt',map_location=device,weights_only=True)['model'])
model.eval()
manifest=json.loads((args.data/'manifest.json').read_text())
for relative in manifest['test_scenes']:
    s,x0,d,arc,condition=load_scene(args.data/relative,device)
    material=['basalt','limestone','sandstone'][int(s['material'])]
    folder=args.run/material
    prediction=dict(np.load(folder/'prediction.npz'))
    velocities=[];true_velocities=[];true_positions=[]
    with torch.no_grad():
        for time,positions in zip(prediction['times'],prediction['positions']):
            t=torch.full((len(x0),1),float(time),device=device)
            xt=torch.as_tensor(positions,device=device)
            velocities.append(model(xt,x0,t,condition).cpu().numpy())
            true_velocities.append(target_velocity(d,arc,t).cpu().numpy())
            true_positions.append(target_path(x0,d,arc,t).cpu().numpy())
    np.savez_compressed(folder/'velocities.npz',predicted=np.stack(velocities),
                        target=np.stack(true_velocities),target_positions=np.stack(true_positions),
                        times=prediction['times'],ids=prediction['ids'])
    print(material, 'exported',len(velocities),'velocity fields')
