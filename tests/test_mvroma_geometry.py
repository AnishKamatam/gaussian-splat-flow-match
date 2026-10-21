"""Catch coordinate offsets, invalid matches, and broken shared observations."""
import unittest
import tempfile
import subprocess
from pathlib import Path
import numpy as np
from mvroma_geometry import normalized_to_colmap, sample_pair, KeypointIndex, checkpoint_for_model, validate_cache


class GeometryTests(unittest.TestCase):
    def test_pixel_centers_resize_without_half_pixel_bias(self):
        got = normalized_to_colmap(np.array([[-.75, -.5], [.75, .5]]), (8, 4))
        np.testing.assert_allclose(got, [[1, 1], [7, 3]])

    def test_invalid_low_confidence_and_nonreciprocal_matches_rejected(self):
        # Identity on a 2x2 image, one low score and one invalid target.
        flow = np.array([[[-.5, .5], [-.5, 2.]], [[-.5, -.5], [.5, .5]]])
        score = np.array([[.95, .1], [.99, .99]])
        reverse = np.array([[[-.5, .5], [.5, .5]], [[-.5, -.5], [.5, .5]]])
        a,b,c = sample_pair(flow, score, reverse, np.ones((2,2)),
                            (4,4), (8,8), stride=1, confidence=.8, cycle_px=.1)
        np.testing.assert_allclose(a, [[1,1]])
        np.testing.assert_allclose(b, [[2,2]])
        np.testing.assert_allclose(c, [.95])

    def test_shared_keypoints_keep_indices_across_pairs(self):
        index=KeypointIndex(cell_size=1.)
        ids=index.add(np.array([[10.1,20.1],[10.2,20.2],[12.1,20.1]]))
        np.testing.assert_array_equal(ids,[0,0,1])
        np.testing.assert_array_equal(index.add(np.array([[12.2,20.2]])),[1])
        np.testing.assert_allclose(index.points(),[[10.15,20.15],[12.15,20.15]])

    def test_checkpoint_requires_all_live_parameters_and_only_known_extras(self):
        weights={'live':1,'conv_backbones.8.0.weight':2,
                 'tracktention_modules.0.sampler.attn.rope.periods':3}
        filtered,ignored=checkpoint_for_model(weights, {'live'})
        self.assertEqual(filtered,{'live':1})
        self.assertEqual(len(ignored),2)
        with self.assertRaises(ValueError): checkpoint_for_model(weights, {'missing'})
        with self.assertRaises(ValueError): checkpoint_for_model({'live':1,'unknown':2}, {'live'})

    def test_cache_rejects_unprovenanced_predictions_and_changed_signature(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            validate_cache(root,{'version':1})
            (root/'0.npz').write_bytes(b'prediction')
            validate_cache(root,{'version':1})
            with self.assertRaises(ValueError): validate_cache(root,{'version':2})
            (root/'config.json').unlink()
            with self.assertRaises(ValueError): validate_cache(root,{'version':1})

    def test_source_fingerprint_detects_dirty_inference_code(self):
        from scripts.reconstruct_mvroma import source_identity
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            subprocess.run(['git','init','-q',str(root)],check=True)
            (root/'model.py').write_text('value = 1\n')
            subprocess.run(['git','-C',str(root),'add','model.py'],check=True)
            subprocess.run(['git','-C',str(root),'-c','user.name=Test','-c','user.email=test@example.invalid',
                            'commit','-qm','fixture'],check=True)
            original=source_identity(root)
            (root/'model.py').write_text('value = 2\n')
            changed=source_identity(root)
            self.assertEqual(original['commit'],changed['commit'])
            self.assertNotEqual(original['source_sha256'],changed['source_sha256'])

    def test_nan_predictions_are_not_exported(self):
        flow=np.full((2,2,2),np.nan)
        reverse=np.zeros_like(flow)
        a,b,c=sample_pair(flow,np.ones((2,2)),reverse,np.ones((2,2)),(2,2),(2,2),stride=1)
        self.assertEqual(a.shape,(0,2))
        self.assertEqual(b.shape,(0,2))
        self.assertEqual(len(c),0)

if __name__ == '__main__':
    unittest.main()
