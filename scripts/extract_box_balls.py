"""Fit spheres from photo-seeded observed dense surfaces, then match endpoints."""
import sys,json
from pathlib import Path
import numpy as np
import cv2,pycolmap
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from ball_motion import fit_sphere,assign_balls,interpolate
from scripts.render_clouds import read_ply

def main():
 out=Path('outputs/ball-motion');seeds=json.loads(Path('configs/box-ball-seeds.json').read_text());align=json.loads((out/'alignment.json').read_text());s=align['scale'];r=np.array(align['rotation']);t=np.array(align['translation']);balls={}
 for state in ['before','after']:
  rec=pycolmap.Reconstruction(f'outputs/mvroma-final/box/{state}/sparse/0');im=next(i for i in rec.images.values() if i.name==seeds[state]['image']);cam=rec.cameras[im.camera_id]
  xyz,rgb=read_ply(Path(f'outputs/mvroma-dense/box/{state}/dense.ply'));matrix=im.cam_from_world().matrix();camera=xyz@matrix[:,:3].T+matrix[:,3];uv=cam.img_from_cam(camera)
  pic=cv2.imread(f'outputs/data2-input/box/{state}/images/{im.name}');factor=np.array([pic.shape[1],pic.shape[0]])/seeds['display_size'];entries=[]
  for idx,(x,y,rad,color) in enumerate(seeds[state]['circles']):
   center2d=np.array([x,y])*factor;rad*=factor.mean();dist=np.linalg.norm(uv-center2d,axis=1)
   col=rgb.astype(float);white=(col.min(1)>85)&((col.max(1)-col.min(1))<65);orange=(col[:,0]>col[:,1]*1.35)&(col[:,1]>col[:,2]*1.3)
   mask=(dist<rad*.92)&(camera[:,2]>0)&(white if color=='white' else orange)
   p=xyz[mask];c,radius,err=fit_sphere(p)
   keep=err<radius*.2
   c,radius,err=fit_sphere(p[keep]);support=p[keep]
   if state=='after':c=s*c@r.T+t;radius*=s;err*=s;support=s*support@r.T+t
   quality='weak surface support' if len(support)<200 or np.median(err)/radius>.02 else 'supported sphere fit'
   entries.append(dict(quality=quality,id=f'{state[0].upper()}{idx+1:02}',center=c.tolist(),radius=radius,color=color,points=len(support),median_shell_error=float(np.median(err)),p95_shell_error=float(np.quantile(err,.95)),seed_image=im.name,seed_xy=center2d.tolist()))
   cv2.circle(pic,tuple(center2d.astype(int)),int(rad),(0,255,0),3);cv2.putText(pic,entries[-1]['id'],tuple((center2d+[-rad,-rad]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),3)
   print(state,idx+1,len(p),round(radius,4),round(float(np.median(err)),4),flush=True)
  cv2.imwrite(str(out/f'{state}_seeds.jpg'),pic);balls[state]=entries
  # Check fitted centers in every original view, including four unseeded views.
  for view in rec.images.values():
   photo=cv2.imread(f'outputs/data2-input/box/{state}/images/{view.name}')
   vc=rec.cameras[view.camera_id];vm=view.cam_from_world().matrix()
   for ball in entries:
    center=np.array(ball['center']);radius=ball['radius']
    if state=='after':center=(center-t)@r/s;radius/=s
    cp=vm[:,:3]@center+vm[:,3];xy=vc.img_from_cam(cp);edge=vc.img_from_cam(cp+np.array([radius,0,0]));rr=np.linalg.norm(edge-xy)
    cv2.circle(photo,tuple(xy.astype(int)),int(rr),(0,255,0),2);cv2.putText(photo,ball['id'],tuple((xy+[-rr,-rr]).astype(int)),cv2.FONT_HERSHEY_SIMPLEX,.8,(0,0,255),2)
   cv2.imwrite(str(out/f'{state}_{view.name}'),photo)

 a=np.array([v['center'] for v in balls['before']]);b=np.array([v['center'] for v in balls['after']]);matches=assign_balls(a,b,[v['color'] for v in balls['before']],[v['color'] for v in balls['after']]);target=b[[m['target'] for m in matches]]
 time=np.linspace(0,1,101);pos,vel=interpolate(a,target,time);radii=np.array([v['radius'] for v in balls['before']]);overlaps=[]
 for i in range(len(a)):
  for j in range(i):
   d=np.linalg.norm(pos[:,i]-pos[:,j],axis=1);penetration=radii[i]+radii[j]-d
   if penetration.max()>.1*min(radii[i],radii[j]):overlaps.append(dict(source_ids=[i,j],max_overlap=float(penetration.max()),normalized_time=float(time[penetration.argmax()])))
 np.savez_compressed(out/'motion.npz',time=time,positions=pos,normalized_velocity=vel)
 result=dict(balls=balls,matches=matches,model='Minimum-jerk straight endpoint interpolation; arbitrary spatial units and normalized time. Not recovered trajectories or physical velocities. Ball rotation not estimated.',correspondence='Color-constrained minimum-total-displacement assignment. All identities unverified; alternative gap is an optimization diagnostic, not a probability.',collision_warnings=overlaps)
 (out/'balls_and_motion.json').write_text(json.dumps(result,indent=2));print('collision warnings',len(overlaps))
if __name__=='__main__':main()
