"""Export reusable geometry and audit the reconstruction-derived endpoint model."""
from pathlib import Path
import csv,json,sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from scripts.align_box_states import similarity

def write_ply(path,xyz,rgb):
 data=np.empty(len(xyz),dtype=[('x','<f4'),('y','<f4'),('z','<f4'),('red','u1'),('green','u1'),('blue','u1')])
 for i,key in enumerate(['x','y','z']):data[key]=xyz[:,i]
 for i,key in enumerate(['red','green','blue']):data[key]=rgb[:,i]
 with path.open('wb') as f:
  f.write(('ply\nformat binary_little_endian 1.0\nelement vertex '+str(len(xyz))+'\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n').encode());f.write(data.tobytes())

def main():
 out=Path('outputs/ball-motion');d=json.loads((out/'balls_and_motion.json').read_text());align=json.loads((out/'alignment.json').read_text());pairs=np.load(out/'alignment_matches.npz');a=pairs['after'][pairs['inliers']];b=pairs['before'][pairs['inliers']]
 # Leave-one-correspondence-out diagnostic on RANSAC-selected support. This is
 # conditional cross-validation, not an independent absolute calibration test.
 errors=[]
 for i in range(len(a)):
  keep=np.arange(len(a))!=i;s,r,t=similarity(a[keep],b[keep]);errors.append(float(np.linalg.norm(s*a[i]@r.T+t-b[i])))
 white=np.array([x['radius'] for state in d['balls'].values() for x in state if x['color']=='white']);diameter=float(2*np.median(white));weak=[]
 for state in ['before','after']:
  cloud=np.load(out/f'{state}_aligned.npz');assert np.isfinite(cloud['xyz']).all();write_ply(out/f'{state}_aligned.ply',cloud['xyz'],cloud['rgb'])
  with (out/f'{state}_centers.csv').open('w') as f:
   w=csv.writer(f);w.writerow(['id','x','y','z','radius','color','surface_points','median_shell_error','quality'])
   for ball in d['balls'][state]:
    w.writerow([ball['id'],*ball['center'],ball['radius'],ball['color'],ball['points'],ball['median_shell_error'],ball['quality']])
    if ball['quality']!='supported sphere fit':weak.append(ball['id'])
 with (out/'correspondences.csv').open('w') as f:
  w=csv.writer(f);w.writerow(['before_id','after_id','displacement_arbitrary_units','displacement_in_median_golf_diameters','alternative_after_id','global_assignment_cost_gap','identity_verified'])
  for m in d['matches']:w.writerow([d['balls']['before'][m['source']]['id'],d['balls']['after'][m['target']]['id'],m['distance'],m['distance']/diameter,d['balls']['after'][m['alternative_target']]['id'],m['alternative_gap'],False])
 motion=np.load(out/'motion.npz');before=np.array([x['center'] for x in d['balls']['before']]);after=np.array([d['balls']['after'][m['target']]['center'] for m in d['matches']]);np.testing.assert_allclose(motion['positions'][0],before);np.testing.assert_allclose(motion['positions'][-1],after);assert np.isfinite(motion['positions']).all();assert len(set(m['target'] for m in d['matches']))==14
 report=dict(alignment=align,conditional_leave_one_out_median=float(np.median(errors)),conditional_leave_one_out_p95=float(np.quantile(errors,.95)),alignment_median_in_golf_diameters=align['median_residual']/diameter,median_golf_diameter_arbitrary_units=diameter,white_radius_relative_std=float(white.std()/white.mean()),weak_sphere_fits=weak,matched_balls=14,verified_identities=0,collision_pairs=len(d['collision_warnings']),endpoint_checks='passed',caveat='Conditional alignment cross-validation reuses RANSAC-selected data. No metric calibration, real time, rotations, or intermediate trajectory observations.')
 (out/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
