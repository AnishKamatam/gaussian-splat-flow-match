import importlib.util
from pathlib import Path
import unittest
import numpy as np

MODULE = Path(__file__).resolve().parents[1] / 'synthetic_flow.py'

class SyntheticTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(MODULE.exists(), 'Synthetic benchmark is not implemented')
        spec = importlib.util.spec_from_file_location('synthetic_flow', MODULE)
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)

    def test_repeatable_corresponding_endpoints(self):
        a = self.m.scene(0, 42, 1000)
        b = self.m.scene(0, 42, 1000)
        np.testing.assert_array_equal(a['x0'], b['x0'])
        np.testing.assert_allclose(self.m.path(a, 0), a['x0'])
        np.testing.assert_allclose(self.m.path(a, 1), a['x1'])
        self.assertEqual(len(np.unique(a['ids'])), 1000)
        self.assertTrue(np.isfinite(a['x1']).all())
        self.assertGreater(np.linalg.norm(a['x1']-a['x0'],axis=1).max(), .05)

    def test_velocity_is_path_derivative(self):
        a = self.m.scene(1, 31, 500)
        t, h = .37, 1e-4
        numerical = (self.m.path(a,t+h)-self.m.path(a,t-h))/(2*h)
        np.testing.assert_allclose(numerical, self.m.velocity(a,t), atol=1e-8)

    def test_train_and_test_use_distinct_scenes(self):
        self.assertTrue(set(self.m.TRAIN_SEEDS).isdisjoint(self.m.TEST_SEEDS))

if __name__ == '__main__':
    unittest.main()
