"""Synthetic coordinate, boundary and air-cell contracts for the wave-room adapter."""
import hashlib
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.acoustics.wave3d import Grid, _stencils
from sim.acoustics.wave_room import air_point, build_room, candidates


class AcousticWaveRoomTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'room.json'
        self.room = {'floor_height_m': .8, 'ceiling_m': 2.6,
                     'walls': [{'a': [1.25, 0], 'b': [1.25, 2], 'label': 'finite furniture face'},
                               {'a': [-.75, 3.25], 'b': [1.5, 3.25], 'label': 'disconnected wall'}]}

    def make(self, **kwargs):
        self.path.write_text(json.dumps({'room': self.room}))
        params = dict(spacing_m=.25, padding_m=.5)
        params.update(kwargs)
        return build_room(self.path, **params)

    def node(self, grid, point):
        return tuple(np.rint((np.asarray(point) - grid.origin_m) / grid.spacing_m).astype(int))

    def test_saved_wall_endpoints_are_xz_and_extrusion_is_y(self):
        room = self.make()
        np.testing.assert_array_equal(room.metadata['declared_segment_bounds_xz_m'], [[-.75, 1.5], [0, 3.25]])
        first = room.metadata['wall_segments'][0]
        self.assertEqual(first['a_xz_m'], [1.25, 0])
        self.assertEqual(first['b_xz_m'], [1.25, 2])
        self.assertTrue(first['full_height_extrusion_assumed'])
        g = room.grid
        near = self.node(g, [1.125, -.175, .625])
        beyond = self.node(g, [1.125, -.175, 2.625])
        self.assertTrue(g.solid[near])
        self.assertFalse(g.solid[beyond])
        np.testing.assert_array_equal(g.solid[:, 0, :], g.solid[:, -1, :])

    def test_rigid_box_completion_is_explicit_and_never_claims_walkability(self):
        room = self.make()
        self.assertFalse(room.metadata['physical_room_validated'])
        assumptions = ' '.join(room.metadata['assumptions'])
        self.assertIn('Artificial rectangular rigid completion', assumptions)
        self.assertIn('no enclosed/walkable footprint inferred', assumptions)
        self.assertIn('including possible furniture faces', assumptions)
        self.assertIn('not calibrated wall impedance', assumptions)
        self.assertEqual(room.metadata['source_sha256'], hashlib.sha256(self.path.read_bytes()).hexdigest())
        self.assertEqual(room.metadata['solid_sha256'], hashlib.sha256(room.grid.solid.tobytes()).hexdigest())

    def test_outer_faces_are_half_a_cell_outside_sample_centres(self):
        room = self.make()
        g, h = room.grid, room.grid.spacing_m
        bounds = np.asarray(room.metadata['realized_rigid_bounds_xyz_m'])
        for axis in range(3):
            self.assertAlmostEqual(bounds[axis, 0], g.origin_m[axis] - h / 2)
            self.assertAlmostEqual(bounds[axis, 1], g.origin_m[axis] + (g.shape_xyz[axis] - .5) * h)
        # A free node is sampled pressure, not a rigid face; the face lies another half cell away.
        self.assertTrue(air_point(g, g.origin_m))
        for axis in range(3):
            point = np.asarray(g.origin_m).copy()
            point[axis] = bounds[axis, 0]
            self.assertFalse(air_point(g, point))

    def test_floor_and_ceiling_quantization_preserves_requested_and_realized_values(self):
        room = self.make()
        meta = room.metadata
        self.assertAlmostEqual(meta['requested_floor_y_m'], -.8)
        self.assertAlmostEqual(meta['requested_ceiling_y_m'], 1.8)
        self.assertEqual(room.grid.shape_xyz[1], math.ceil(2.6 / .25))
        actual = meta['realized_rigid_bounds_xyz_m'][1]
        self.assertAlmostEqual(actual[0], -.8)
        self.assertGreaterEqual(actual[1], 1.8)
        self.assertLess(actual[1] - 1.8, .25)
        shifted = self.make(floor_shift_m=.1, ceiling_shift_m=.07)
        self.assertAlmostEqual(shifted.metadata['requested_floor_y_m'], -.7)
        self.assertAlmostEqual(shifted.metadata['requested_ceiling_y_m'], 1.87)
        self.assertAlmostEqual(shifted.metadata['realized_rigid_bounds_xyz_m'][1][0], -.7)
        self.assertGreaterEqual(shifted.metadata['realized_rigid_bounds_xyz_m'][1][1], 1.87)

    def test_air_point_requires_every_positive_trilinear_weight_to_be_free(self):
        solid = np.zeros((5, 5, 5), dtype=bool)
        solid[2, 2, 2] = True
        grid = Grid(origin_m=(0, 0, 0), shape_xyz=(5, 5, 5), spacing_m=1, solid=solid)
        self.assertTrue(air_point(grid, [1, 2, 2]))  # adjacent solid has zero interpolation weight here
        self.assertFalse(air_point(grid, [1.1, 2, 2]))
        # The adapter must not approve a stencil that the propagation engine rejects.
        self.assertFalse(air_point(grid, [1 + 1e-7, 1 + 1e-7, 2]))
        self.assertFalse(air_point(grid, [1.5, 1.5, 1.5]))
        self.assertFalse(air_point(grid, [2, 2, 2]))
        self.assertFalse(air_point(grid, [-.01, 0, 0]))
        self.assertFalse(air_point(grid, [4.01, 0, 0]))
        for point in ([np.nan, 1, 1], [1, np.inf, 1], [1, 2], [1, 2, 3, 4]):
            self.assertFalse(air_point(grid, point))
        _stencils(grid, np.array([[1., 2, 2]]))
        with self.assertRaisesRegex(ValueError, 'solid voxel'):
            _stencils(grid, np.array([[1.1, 2, 2]]))
        with self.assertRaisesRegex(ValueError, 'solid voxel'):
            _stencils(grid, np.array([[1 + 1e-7, 1 + 1e-7, 2]]))

    def test_candidate_grid_is_fixed_by_declared_bounds_and_air_mask_not_hidden_truth(self):
        room = self.make()
        before = candidates(room, horizontal_spacing=.75, heights_above_floor=(.35, .7))
        self.assertGreater(len(before), 0)
        self.assertEqual(len(before), len(np.unique(before, axis=0)))
        self.assertTrue(all(air_point(room.grid, point) for point in before))
        self.assertTrue(all(point[0] in (-.75, 0, .75) for point in before))
        self.assertTrue(all(point[2] in (0, .75, 1.5, 2.25, 3) for point in before))
        self.assertTrue(all(any(np.isclose(point[1], -.8 + h) for h in (.35, .7)) for point in before))
        room.metadata['hidden_emitter_xyz_m'] = [.123, -.31, 1.234]
        room.metadata['hidden_held_out_truth'] = [[-.5, -.4, 2.9]]
        after = candidates(room, horizontal_spacing=.75, heights_above_floor=(.35, .7))
        np.testing.assert_array_equal(before, after)
        # This is a mask of interpolatable fluid cells, not evidence of accessible household floor.
        _stencils(room.grid, after)

    def test_wall_shift_is_seeded_and_changes_voxel_mask_when_resolvable(self):
        nominal = self.make()
        first = self.make(wall_shift_m=.5, seed=11)
        repeat = self.make(wall_shift_m=.5, seed=11)
        other = self.make(wall_shift_m=.5, seed=12)
        self.assertEqual(first.metadata['wall_segments'], repeat.metadata['wall_segments'])
        np.testing.assert_array_equal(first.grid.solid, repeat.grid.solid)
        self.assertNotEqual(first.metadata['solid_sha256'], nominal.metadata['solid_sha256'])
        self.assertNotEqual(first.metadata['solid_sha256'], other.metadata['solid_sha256'])
        self.assertTrue(all(abs(w['normal_offset_m']) <= .5 for w in first.metadata['wall_segments']))
        for original, shifted in zip(nominal.metadata['wall_segments'], first.metadata['wall_segments']):
            original_delta = np.asarray(original['b_xz_m']) - original['a_xz_m']
            np.testing.assert_allclose(np.asarray(shifted['b_xz_m']) - shifted['a_xz_m'], original_delta)

    def test_invalid_floor_geometry_and_parameters_are_rejected(self):
        for floor in (False, 0, -.1, 2.6, 3, np.nan, np.inf):
            with self.subTest(floor=floor), self.assertRaises(ValueError):
                self.room['floor_height_m'] = floor
                self.make()
        self.room['floor_height_m'] = .8
        for kw in ({'spacing_m': 0}, {'spacing_m': -.1}, {'spacing_m': np.nan}, {'padding_m': 0},
                   {'padding_m': True}, {'sound_speed': 0}, {'damping': -1}, {'wall_shift_m': -.1},
                   {'wall_thickness_m': 0}, {'wall_thickness_m': np.inf}, {'ceiling_shift_m': -3}):
            with self.subTest(parameters=kw), self.assertRaises(ValueError):
                self.make(**kw)
        for walls in ([], [{'a': [0, 0], 'b': [0, 0]}], [{'a': [0, 0, 0], 'b': [1, 1, 1]}],
                      [{'a': [0, np.nan], 'b': [1, 1]}]):
            with self.subTest(walls=walls), self.assertRaises(ValueError):
                self.room['walls'] = walls
                self.make()

    def test_invalid_candidate_layers_and_spacing_are_not_silently_accepted(self):
        room = self.make()
        for spacing in (0, -.1, np.nan, np.inf, True):
            with self.subTest(spacing=spacing), self.assertRaises(ValueError):
                candidates(room, horizontal_spacing=spacing)
        for heights in ((), (np.nan,), (np.inf,), (-.1,), (False,), (.35, np.nan)):
            with self.subTest(heights=heights), self.assertRaises(ValueError):
                candidates(room, heights_above_floor=heights)


if __name__ == '__main__':
    unittest.main()
