import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / 'reconstruct.py'

class ReconstructionTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.exists(), 'Reconstruction pipeline has not been implemented')
        spec = importlib.util.spec_from_file_location('reconstruct', SCRIPT)
        self.m = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.m)

    def test_selects_model_with_most_registered_images(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name, images, points in [('0', 3, 100), ('1', 8, 20)]:
                folder = root / name
                folder.mkdir()
                (folder / 'images.bin').write_bytes(struct.pack('<Q', images))
                (folder / 'points3D.bin').write_bytes(struct.pack('<Q', points))
                (folder / 'cameras.bin').write_bytes(struct.pack('<Q', 1))
            model, count, points = self.m.best_model(root)
            self.assertEqual((model.name, count, points), ('1', 8, 20))

    def test_empty_reconstruction_is_explicit_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(RuntimeError, 'No sparse model'):
                self.m.best_model(Path(tmp))

    def test_rejects_empty_cloud(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'cloud.ply'
            path.write_text('ply\nformat ascii 1.0\nelement vertex 0\nend_header\n')
            with self.assertRaisesRegex(RuntimeError, 'empty'):
                self.m.ply_vertices(path)
            path.write_text('ply\nformat ascii 1.0\nelement vertex 12\nend_header\n')
            self.assertEqual(self.m.ply_vertices(path), 12)

    def test_discovers_six_separate_sets(self):
        root = SCRIPT.parent / 'data' / 'split'
        sets = self.m.discover(root, ['basalt', 'limestone', 'sandstone'], ['before', 'after'])
        self.assertEqual(len(sets), 6)
        self.assertTrue(all(len(list(p.glob('*.png'))) == 10 for _, _, p in sets))

    def test_failed_scene_does_not_stop_remaining_sets(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(self.m.shutil, 'which', return_value='/fake/colmap'):
                with patch.object(self.m, 'reconstruct', side_effect=[RuntimeError('No matches'), {'sparse_points': 25}]) as run:
                    code = self.m.main(['--input-root', str(SCRIPT.parent / 'data/split'),
                                        '--output-root', tmp, '--materials', 'basalt'])
            import json
            summary = json.loads(next(Path(tmp).glob('*/summary.json')).read_text())
            self.assertEqual(code, 1)
            self.assertEqual(run.call_count, 2)
            self.assertEqual([r['status'] for r in summary], ['failed', 'success'])
            self.assertEqual(summary[0]['error'], 'No matches')

if __name__ == '__main__':
    unittest.main()
