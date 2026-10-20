import unittest
try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, 'PyTorch required for model tests; run these on the GPU environment')
class FlowModelTests(unittest.TestCase):
    def test_midpoint_integrates_constant_velocity(self):
        import train_flow as m
        class ConstantField:
            def __call__(self,xt,x0,t,c):
                return torch.ones_like(xt)*.25
        x0=torch.zeros((7,3))
        states=m.integrate(ConstantField(),x0,torch.zeros((7,8)),steps=64)
        self.assertEqual(states.shape,(9,7,3))
        self.assertTrue(abs(states[-1]-.25).max()<1e-6)

    def test_target_velocity_matches_autograd(self):
        import train_flow as m
        t=torch.tensor([[.3]],requires_grad=True)
        x0=torch.tensor([[.1,.2,.3]])
        delta=torch.tensor([[.2,-.3,.4]])
        arc=torch.tensor([[0.,0.,.1]])
        x=m.target_path(x0,delta,arc,t)
        expected=m.target_velocity(delta,arc,t)
        actual=torch.stack([torch.autograd.grad(x[0,i],t,retain_graph=True)[0].squeeze() for i in range(3)])
        torch.testing.assert_close(actual,expected[0])

    def test_field_outputs_finite_velocity(self):
        import train_flow as m
        model=m.VelocityField()
        output=model(torch.zeros(8,3),torch.zeros(8,3),torch.ones(8,1)*.5,torch.tensor([[0.,0.,.4,.3,.2,1.,0.,0.]]).expand(8,-1))
        self.assertEqual(output.shape,(8,3))
        self.assertTrue(torch.isfinite(output).all())

    def test_unaffected_points_have_exactly_zero_velocity(self):
        import train_flow as m
        model=m.VelocityField()
        source=torch.ones((5,3))*2
        condition=torch.zeros((5,8))
        condition[:,2]=.4
        output=model(source,source,torch.ones((5,1))*.5,condition)
        torch.testing.assert_close(output,torch.zeros_like(output))
