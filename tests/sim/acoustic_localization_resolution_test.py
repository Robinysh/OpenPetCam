"""Controls for global lattice placement, power prefixes and missing coverage."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.acoustics.localization_model import load_room
from sim.acoustics.localization_resolution import describe, low_height_grid, prefix_mean_powers


class ResolutionTests(unittest.TestCase):
    def setUp(self):
        self.room = load_room({"floor_height_m": .8, "ceiling_m": 2.6,
                              "walls": [{"a": [-2, -1], "b": [2, 3]}]})

    def test_grids_are_global_nested_and_do_not_take_source_labels(self):
        coarse = low_height_grid(self.room, .5)
        fine = low_height_grid(self.room, .125)
        self.assertTrue(all(tuple(point) in set(map(tuple, fine)) for point in coarse))
        np.testing.assert_allclose(coarse[:, 1], -.8 + .45)
        np.testing.assert_allclose(fine[:, [0, 2]] / .125, np.round(fine[:, [0, 2]] / .125))
        self.assertTrue(np.all(np.hypot(fine[:, 0], fine[:, 2]) >= .6))
        np.testing.assert_array_equal(coarse, low_height_grid(self.room, .5))

    def test_repeated_power_uses_same_nested_linear_prefix_and_preserves_input(self):
        powers = np.tile(np.arange(1, 17, dtype=np.float32)[:, None], (1, 3))
        original = powers.copy()
        result = prefix_mean_powers(powers)
        np.testing.assert_array_equal(result[1], [1, 1, 1])
        np.testing.assert_array_equal(result[4], [2.5, 2.5, 2.5])
        np.testing.assert_array_equal(result[16], [8.5, 8.5, 8.5])
        np.testing.assert_array_equal(powers, original)

    def test_short_or_negative_power_rejects_instead_padding(self):
        with self.assertRaises(ValueError):
            prefix_mean_powers(np.ones((15, 3)))
        with self.assertRaises(ValueError):
            prefix_mean_powers(-np.ones((16, 3)))

    def test_unlocalized_failures_stay_in_success_denominators(self):
        rows = [{"distance_error_m": .2, "azimuth_error_deg": 5},
                {"distance_error_m": None, "azimuth_error_deg": None}]
        result = describe(rows)
        self.assertEqual(result["within_1m_fraction"], .5)
        self.assertEqual(result["within_0p5m_fraction"], .5)
        self.assertEqual(result["unlocalized_cases"], 1)


if __name__ == "__main__":
    unittest.main()
