"""Isolated uncertainty factors keep unrelated forward quantities unchanged."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.acoustics.localization_model import RoomModel, Surface, build_paths
from sim.acoustics.localization_sensitivity import isolated_models


class SensitivityTests(unittest.TestCase):
    def setUp(self):
        self.model = RoomModel((Surface('floor', (0, -1, 0), (0, 1, 0)),
                                Surface('ceiling', (0, 2, 0), (0, -1, 0)),
                                Surface('wall', (3, 0, 0), (-1, 0, 0))))
        self.factors = isolated_models(self.model, 172)

    def test_material_factor_changes_no_surface_position_or_speed(self):
        material, speed = self.factors['material_only']
        self.assertEqual(speed, 343)
        for base, changed in zip(self.model.surfaces, material.surfaces):
            self.assertEqual(base.point_m, changed.point_m)
            self.assertEqual(base.normal, changed.normal)
            self.assertEqual(base.vertices_m, changed.vertices_m)
        self.assertTrue(any(s.reflection != .5 for s in material.surfaces))

    def test_geometry_levels_share_deviates_and_keep_material_speed_nominal(self):
        small, speed_small = self.factors['geometry_2cm']
        large, speed_large = self.factors['geometry_10cm']
        self.assertEqual((speed_small, speed_large), (343, 343))
        for base, first, second in zip(self.model.surfaces, small.surfaces, large.surfaces):
            np.testing.assert_allclose(np.asarray(second.point_m) - base.point_m,
                                       5 * (np.asarray(first.point_m) - base.point_m), atol=1e-14)
            self.assertEqual((first.reflection, second.reflection), (.5, .5))
        floor, _ = self.factors['floor_2cm']
        walls, _ = self.factors['walls_2cm']
        self.assertEqual(floor.surfaces[1:], self.factors['nominal'][0].surfaces[1:])
        self.assertEqual(walls.surfaces[:2], self.factors['nominal'][0].surfaces[:2])

    def test_speed_factor_changes_delay_but_not_geometry_path_gain_or_length(self):
        nominal, _ = self.factors['nominal']
        speed_model, speed = self.factors['speed_only']
        self.assertEqual(nominal, speed_model)
        first = build_paths(nominal, [1, 0, 1], [0, 0, 0], sound_speed=343)
        second = build_paths(speed_model, [1, 0, 1], [0, 0, 0], sound_speed=speed)
        self.assertEqual([p.length_m for p in first], [p.length_m for p in second])
        self.assertEqual([p.gain for p in first], [p.gain for p in second])
        self.assertNotEqual(first[0].delay_s, second[0].delay_s)


if __name__ == '__main__':
    unittest.main()
