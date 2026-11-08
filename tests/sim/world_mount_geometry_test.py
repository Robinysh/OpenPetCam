"""Room surfaces require a mechanical datum, not an unclassified echo peak."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from sim.world.room import fuse


class TestMountGeometry(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix='opc-mount-geometry-')
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        (self.root / 'calib.json').write_text('{}')
        (self.root / 'picks.json').write_text(json.dumps({'surfaces': [], 'note': 'independent synthetic fixture'}))
        self.ctx = SimpleNamespace(path=lambda *parts: str(self.root.joinpath(*parts)))

    def echoes(self, near=.179):
        return SimpleNamespace(data={'items': [
            {'echoes': [[near, .9], [.807, .8]]},
            {'echoes': [[near, .85], [.807, .82]]},
        ]})

    def test_unclassified_peak_cannot_create_surface(self):
        room = fuse(self.ctx, self.echoes())
        self.assertIsNone(room.overhead_m, 'An echo-only proxy must not become mounting-surface geometry')
        self.assertNotIn('countertop overhang', [p[0] for p in room.planes()])
        self.assertAlmostEqual(room.floor_height, .807)
        self.assertEqual(room.provenance['overhead_status'], 'unknown_mount_geometry')
        self.assertAlmostEqual(room.provenance['unclassified_near_echo_proxy_m'], .179)

    def test_declared_mechanical_datum_is_independent_of_peak(self):
        for near in (.179, .220):
            room = fuse(self.ctx, self.echoes(near), mount_clearance_m=.073,
                        mount_clearance_source='synthetic independent mount measurement')
            self.assertAlmostEqual(room.overhead_m, .073)
            plane = next(p for p in room.planes() if p[0] == 'countertop overhang')
            self.assertAlmostEqual(plane[1][1], .073)
            self.assertEqual(room.provenance['overhead_status'], 'declared_mount_geometry')
            self.assertAlmostEqual(room.provenance['unclassified_near_echo_proxy_m'], near)

    def test_explicit_datum_requires_valid_value_and_source(self):
        for value in (0., -.02, float('nan'), float('inf')):
            with self.subTest(value=value), self.assertRaises(ValueError):
                fuse(self.ctx, self.echoes(), mount_clearance_m=value, mount_clearance_source='fixture')
        for source in (None, '', '  '):
            with self.subTest(source=source), self.assertRaises(ValueError):
                fuse(self.ctx, self.echoes(), mount_clearance_m=.073, mount_clearance_source=source)

    def test_tape_floor_path_is_unchanged(self):
        room = fuse(self.ctx, self.echoes(), tape_m=.792)
        self.assertAlmostEqual(room.floor_height, .792)
        self.assertIsNone(room.overhead_m)
        self.assertIn('tape measure', room.provenance['floor_height'])


if __name__ == '__main__':
    unittest.main()
