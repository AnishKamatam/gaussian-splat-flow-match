#!/usr/bin/env python3
"""Embed observed aligned PLY samples and explicitly hypothetical endpoint motion."""
import argparse,base64,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);p.add_argument('--points',type=int,default=25000);args=p.parse_args()
out=Path('outputs/ball-motion');analysis=json.loads((out/'balls_and_motion.json').read_text());balls=analysis['balls'];raw=[np.load(out/f'{s}_aligned.npz') for s in ['before','after']]
centers=np.array([b['center'] for b in balls['before']]);origin=centers.mean(0);_,_,rot=np.linalg.svd(centers-origin,full_matrices=False)
# Orient common axes using the annotated reference photograph: rightward and upward.
x=centers[1]-centers[0];normal=rot[2];x-=normal*(x@normal);x/=np.linalg.norm(x);y=np.cross(normal,x)
if (centers[0]-centers[10])@y<0:normal=-normal;y=-y
rot=np.array([x,y,normal]);q=(raw[0]['xyz']-origin)@rot.T;scale=np.max(np.quantile(np.abs(q),.995,axis=0))/.88
clouds=[]
for index,state in enumerate(['before','after']):
 xyz,rgb=raw[index]['xyz'],raw[index]['rgb'];q=(xyz-origin)@rot.T/scale
 ids=np.random.default_rng(17).choice(len(q),min(args.points,len(q)),replace=False)
 c=np.array([b['center'] for b in balls[state]]);rad=np.array([b['radius'] for b in balls[state]])
 distances=np.linalg.norm(xyz[ids,None]-c[None],axis=2)/rad;labels=distances.argmin(1)+1;labels[distances.min(1)>1.12]=0
 color=rgb[ids].astype(float);brown=(color[:,0]>color[:,1]*1.05)&(color[:,1]>color[:,2]*1.08)&(color[:,0]<color[:,1]*1.65);labels[brown]=0
 packed=np.empty(len(ids),dtype=[('xyz','<i2',(3,)),('rgb','u1',(3,)),('ball','u1')]);packed['xyz']=np.clip(np.round(q[ids]*10000),-32768,32767).astype(np.int16);packed['rgb']=rgb[ids];packed['ball']=labels
 clouds.append(dict(packed=base64.b64encode(packed.tobytes()).decode(),total=len(q)))
 for ball in balls[state]:ball['center']=((np.array(ball['center'])-origin)@rot.T/scale).round(6).tolist();ball['radius']/=scale
np.savez(out/'viewer_frame.npz',origin=origin,rotation=rot,scale=scale)
data=dict(clouds=clouds,balls=balls,matches=analysis['matches'],caption='Aligned observed surfaces · 25,000 sampled points per state · 14 fitted centers per state · all cross-state identities unverified.',collisionCount=len(analysis['collision_warnings']))
text=Path('viewers/box-clouds.template.html').read_text().replace('__BOX_DATA__',json.dumps(data,separators=(',',':')))
args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(text)
assert args.output.stat().st_size<1000000
print(args.output)
