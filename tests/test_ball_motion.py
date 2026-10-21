import unittest
import numpy as np
from ball_motion import fit_sphere, assign_balls, interpolate
class GeometryTests(unittest.TestCase):
 def test_partial_sphere_with_measurement_noise(self):
  rng=np.random.default_rng(5);p=rng.normal(size=(1000,3));p/=np.linalg.norm(p,axis=1)[:,None];p[:,2]=abs(p[:,2]);c=np.array([1.,2.,3.]);p=c+.4*p+rng.normal(scale=.001,size=p.shape)
  center,radius,err=fit_sphere(p)
  np.testing.assert_allclose(center,c,atol=.002);self.assertAlmostEqual(radius,.4,places=2)
 def test_assignment_respects_color_and_reports_ambiguity(self):
  a=np.array([[0.,0,0],[2,0,0],[1,4,0]]);b=np.array([[1.,0,0],[1,0,0],[1,4,0]])
  rows=assign_balls(a,b,[1,1,2],[1,1,2])
  self.assertEqual(rows[2]['target'],2);self.assertLess(rows[0]['alternative_gap'],1e-9)
 def test_motion_has_exact_endpoints_and_zero_endpoint_velocity(self):
  a=np.array([[0.,0,0]]);b=np.array([[2.,3,1]])
  p,v=interpolate(a,b,np.array([0.,.5,1.]))
  np.testing.assert_allclose(p[0],a);np.testing.assert_allclose(p[-1],b);np.testing.assert_allclose(v[[0,-1]],0)

 def test_similarity_recovers_scale_rotation_translation(self):
  from scripts.align_box_states import similarity
  rng=np.random.default_rng(9);a=rng.normal(size=(30,3));rotation=np.array([[0.,-1,0],[1,0,0],[0,0,1]])
  b=1.7*a@rotation.T+np.array([3.,-2,1]);scale,r,t=similarity(a,b)
  np.testing.assert_allclose(scale*a@r.T+t,b,atol=1e-12)
