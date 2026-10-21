"""Endpoint geometry hypotheses; time is normalized, not elapsed seconds."""
import numpy as np
from scipy.optimize import least_squares,linear_sum_assignment

def fit_sphere(points):
 p=np.asarray(points,float);origin=p.mean(0);q=p-origin
 if len(p)<20 or np.linalg.matrix_rank(q)<3:raise ValueError('Insufficient 3D sphere support')
 sol=np.linalg.lstsq(np.c_[2*q,np.ones(len(q))],(q*q).sum(1),rcond=None)[0]
 radius=np.sqrt(max(sol[3]+sol[:3]@sol[:3],1e-12))
 fit=least_squares(lambda s:np.linalg.norm(q-s[:3],axis=1)-s[3],np.r_[sol[:3],radius],loss='soft_l1',f_scale=radius*.03,bounds=([-np.inf]*3+[radius*.5],[np.inf]*3+[radius*2]))
 return fit.x[:3]+origin,float(fit.x[3]),np.abs(fit.fun)

def assign_balls(a,b,color_a,color_b):
 costs=np.linalg.norm(a[:,None]-b[None,:],axis=2)
 costs[np.asarray(color_a)[:,None]!=np.asarray(color_b)[None,:]]=np.inf
 rows,cols=linear_sum_assignment(costs);base=costs[rows,cols].sum();result=[]
 for i,j in zip(rows,cols):
  alternate=costs.copy();alternate[i,j]=np.inf
  try:
   ar,ac=linear_sum_assignment(alternate);gap=float(alternate[ar,ac].sum()-base);other=int(ac[np.where(ar==i)[0][0]])
  except ValueError:gap=None;other=None
  result.append(dict(source=int(i),target=int(j),distance=float(costs[i,j]),alternative_gap=gap,alternative_target=other,identity_status='unverified minimum-displacement hypothesis'))
 return result

def interpolate(a,b,time):
 t=np.asarray(time)[:,None,None];q=10*t**3-15*t**4+6*t**5;dq=30*t**2-60*t**3+30*t**4
 return a[None]+q*(b-a)[None],dq*(b-a)[None]
