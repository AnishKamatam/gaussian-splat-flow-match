"""Align independent reconstructions using static brown image features."""
from pathlib import Path
import sys,json
import numpy as np
import cv2,pycolmap
from scipy.spatial import cKDTree
sys.path.insert(0,str(Path(__file__).resolve().parent))
from render_clouds import read_ply

def similarity(a,b):
 ac=a-a.mean(0);bc=b-b.mean(0)
 u,s,v=np.linalg.svd(bc.T@ac);d=np.ones(3);d[-1]=np.linalg.det(u@v)
 r=(u*d)@v; scale=(s*d).sum()/(ac*ac).sum()
 return scale,r,b.mean(0)-scale*a.mean(0)@r.T

def main():
 out=Path('outputs/ball-motion');out.mkdir(exist_ok=True,parents=True)
 recs={s:pycolmap.Reconstruction(f'outputs/mvroma-final/box/{s}/sparse/0') for s in ['before','after']}
 features={};sift=cv2.SIFT_create(nfeatures=18000)
 for state,rec in recs.items():
  features[state]=[]
  for im in rec.images.values():
   pic=cv2.imread(f'outputs/data2-input/box/{state}/images/{im.name}')
   b,g,r=cv2.split(pic.astype(float));mask=((r>g*1.05)&(g>b*1.08)&(r<g*1.65)).astype('uint8')*255
   kp,desc=sift.detectAndCompute(pic,mask)
   obs=[p for p in im.points2D if p.has_point3D()]
   xy=np.array([p.xy for p in obs]);ids=np.array([p.point3D_id for p in obs])
   dist,ix=cKDTree(xy).query([k.pt for k in kp]);keep=dist<2.5
   features[state].append((desc[keep],ids[ix[keep]]))
   print(state,im.name,int(keep.sum()),flush=True)
 pairs=set();bf=cv2.BFMatcher()
 for da,ia in features['before']:
  for db,ib in features['after']:
   reverse={m.queryIdx:m.trainIdx for m,n in bf.knnMatch(db,da,k=2) if m.distance<.75*n.distance}
   for m,n in bf.knnMatch(da,db,k=2):
    if m.distance<.75*n.distance and reverse.get(m.trainIdx)==m.queryIdx:pairs.add((int(ia[m.queryIdx]),int(ib[m.trainIdx])))
 pairs=sorted(pairs);a=np.array([recs['after'].points3D[j].xyz for i,j in pairs]);b=np.array([recs['before'].points3D[i].xyz for i,j in pairs])
 extent=np.linalg.norm(np.quantile(b,.95,axis=0)-np.quantile(b,.05,axis=0));tol=.008*extent
 rng=np.random.default_rng(42);best=np.zeros(len(a),bool)
 for _ in range(10000):
  sample=rng.choice(len(a),3,replace=False)
  if np.linalg.matrix_rank(a[sample]-a[sample].mean(0))<2:continue
  sc,r,t=similarity(a[sample],b[sample]);err=np.linalg.norm(sc*a@r.T+t-b,axis=1);ok=err<tol
  if ok.sum()>best.sum():best=ok
 if best.sum()<20:raise RuntimeError(f'Insufficient alignment support: {best.sum()}')
 for _ in range(5):
  sc,r,t=similarity(a[best],b[best]);err=np.linalg.norm(sc*a@r.T+t-b,axis=1);best=err<tol
 result=dict(scale=float(sc),rotation=r.tolist(),translation=t.tolist(),candidate_pairs=len(a),inliers=int(best.sum()),threshold=float(tol),median_residual=float(np.median(err[best])),p95_residual=float(np.quantile(err[best],.95)),units='before reconstruction arbitrary units',method='mutual SIFT on brown pixels, associated to sparse tracks, Sim3 RANSAC')
 (out/'alignment.json').write_text(json.dumps(result,indent=2))
 np.savez_compressed(out/'alignment_matches.npz',after=a,before=b,inliers=best)
 for state in recs:
  xyz,rgb=read_ply(Path(f'outputs/mvroma-dense/box/{state}/dense.ply'))
  if state=='after':xyz=sc*xyz@r.T+t
  np.savez_compressed(out/f'{state}_aligned.npz',xyz=xyz,rgb=rgb)
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
