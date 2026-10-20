#!/usr/bin/env python3
"""Supervised conditional vector-field fitting on procedural motion paths.

Not an unpaired distribution matcher or physical velocity estimator. Requires torch/numpy.
"""
import argparse
import json
from pathlib import Path
import random
import numpy as np
import torch
from torch import nn
from synthetic_flow import write_ply


class VelocityField(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(14,128),nn.SiLU(),nn.Linear(128,128),nn.SiLU(),
                               nn.Linear(128,128),nn.SiLU(),nn.Linear(128,3))

    def forward(self,xt,x0,t,condition):
        # Source-anchored conditional field. Current position is accepted by the
        # ODE interface, but deliberately excluded to avoid off-path feedback.
        # The affected region is known conditioning, not inferred from the target.
        relative=(x0[:,:2]-condition[:,:2])/condition[:,2:3]
        weight=torch.clamp(1-relative.square().sum(dim=1,keepdim=True),min=0).square()
        features=torch.cat([x0,relative,t,condition],dim=1)
        return weight*self.net(features)



def target_path(x0,d,arc,t):
    return x0+t*t*(3-2*t)*d+t*(1-t)*arc


def target_velocity(d,arc,t):
    return 6*t*(1-t)*d+(1-2*t)*arc


@torch.no_grad()
def integrate(model,x0,condition,steps=64):
    x=x0.clone()
    frames=[x.cpu().numpy()]
    dt=1/steps
    for step in range(steps):
        t=torch.full((len(x),1),step*dt,device=x.device)
        half=x+.5*dt*model(x,x0,t,condition)
        x=x+dt*model(half,x0,t+.5*dt,condition)
        if (step+1)%(steps//8)==0:
            frames.append(x.cpu().numpy())
    return np.stack(frames)


def load_scene(path,device):
    s=dict(np.load(path))
    x0=torch.as_tensor(s['x0'],dtype=torch.float32,device=device)
    d=torch.as_tensor(s['displacement'],dtype=torch.float32,device=device)
    arc=torch.as_tensor(s['arc'],dtype=torch.float32,device=device)
    c=torch.as_tensor(s['condition'],dtype=torch.float32,device=device).expand(len(x0),-1)
    return s,x0,d,arc,c


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('data/synthetic'))
    parser.add_argument('--output',type=Path,default=Path('outputs/synthetic-flow'))
    parser.add_argument('--steps',type=int,default=3000)
    parser.add_argument('--seed',type=int,default=17)
    parser.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=parser.parse_args()
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    if args.output.exists():
        parser.error('Output already exists; use a new --output directory to preserve experiments')
    args.output.mkdir(parents=True)
    device=torch.device(args.device)
    manifest=json.loads((args.data/'manifest.json').read_text())
    tensors=[load_scene(args.data/p,device)[1:] for p in manifest['train_scenes']]
    x0,d,arc,c=[torch.cat([s[i] for s in tensors]) for i in range(4)]
    moving=torch.where(torch.linalg.vector_norm(d,dim=1)>.005)[0]
    if not len(moving):raise RuntimeError('No moving training points')
    model=VelocityField().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.00001)
    history=[]
    for step in range(args.steps):
        idx=torch.cat([torch.randint(len(x0),(1024,),device=device),
                       moving[torch.randint(len(moving),(1024,),device=device)]])
        t=torch.rand((len(idx),1),device=device)
        xt=target_path(x0[idx],d[idx],arc[idx],t)
        pred=model(xt,x0[idx],t,c[idx])
        loss=(pred-target_velocity(d[idx],arc[idx],t)).square().mean()
        optimizer.zero_grad(set_to_none=True);loss.backward();optimizer.step()
        if step%250==0 or step==args.steps-1:
            item=dict(step=step,loss=float(loss.detach()))
            history.append(item);print(json.dumps(item),flush=True)
    model.eval()
    torch.save(dict(model=model.state_dict(),seed=args.seed,steps=args.steps,
                    architecture='source-anchored gated VelocityField 14->128->128->128->3',
                    units='dimensionless per normalized time'),args.output/'model.pt')
    results=[]
    for relative in manifest['test_scenes']:
        s,source,delta,bend,condition=load_scene(args.data/relative,device)
        with torch.no_grad():
            velocity_errors=[]
            for time in [.1,.3,.5,.7,.9]:
                t=torch.full((len(source),1),time,device=device)
                xt=target_path(source,delta,bend,t)
                velocity_errors.append(torch.linalg.vector_norm(model(xt,source,t,condition)-target_velocity(delta,bend,t),dim=1).cpu().numpy())
            predicted=integrate(model,source,condition)
            # Check integration resolution without using test error to select a model.
            fine=integrate(model,source,condition,steps=128)[-1]
        endpoint=predicted[-1]
        errors=np.linalg.norm(endpoint-s['x1'],axis=1)
        baseline=np.linalg.norm(s['displacement'],axis=1)
        changed=baseline>.005
        material=['basalt','limestone','sandstone'][int(s['material'])]
        folder=args.output/material;folder.mkdir()
        write_ply(folder/'before.ply',s['x0'],s['rgb'])
        write_ply(folder/'after_true.ply',s['x1'],s['rgb'])
        write_ply(folder/'after_predicted.ply',endpoint,s['rgb'])
        np.savez_compressed(folder/'prediction.npz',positions=predicted,times=np.linspace(0,1,len(predicted)),
                            ids=s['ids'],x0=s['x0'],x1=s['x1'],arc=s['arc'],rgb=s['rgb'])
        result=dict(material=material,test_seed=int(s['seed']),points=len(source),moving_points=int(changed.sum()),
                    endpoint_epe=float(errors.mean()),moving_endpoint_epe=float(errors[changed].mean()),
                    stationary_endpoint_epe=float(errors[~changed].mean()),
                    no_motion_epe=float(baseline.mean()),no_motion_moving_epe=float(baseline[changed].mean()),
                    velocity_epe=float(np.mean(velocity_errors)),
                    integration_refinement_epe=float(np.linalg.norm(fine-endpoint,axis=1).mean()))
        results.append(result);print(json.dumps(result),flush=True)
    report=dict(experiment='supervised synthetic conditional flow',train_scenes=len(tensors),test_scenes=3,
                split='disjoint scene seeds, same procedural family',seed=args.seed,training_steps=args.steps,
                device=str(device),torch_version=torch.__version__,units='dimensionless; t in [0,1]',
                targets='prescribed analytic motion, not recovered physical motion',
                limitation='Known affected region and collapse parameters; anchored Lagrangian field ignores xt; real correspondence unsolved',
                results=results,history=history)
    (args.output/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
