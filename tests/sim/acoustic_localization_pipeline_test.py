"""Research composition preserves power inputs and fails closed without evidence."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.acoustics.localization_model import RoomModel, Surface
from sim.acoustics.localization_pipeline import ResearchMatcher


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.frequency = np.fft.rfftfreq(8192, 1 / 16000)
        model = RoomModel((Surface('floor', (0, -1, 0), (0, 1, 0)),),
            declared_wall_extents=({'a_xz_m': [-3, -3], 'b_xz_m': [3, 3]},))
        self.matcher = ResearchMatcher(model, np.array([[1, -.55, 1], [2, -.55, 2]]),
            np.zeros((2, 3, 871)), [0, 0, 0], self.frequency)

    def test_no_signal_does_not_fabricate_a_refinement_seed(self):
        powers = np.zeros((4, len(self.frequency)))
        result = self.matcher.match(powers, harmonic=True, refine=True)
        self.assertIsNone(result['global']['position_m'])
        self.assertIsNone(result['final']['position_m'])
        self.assertEqual(result['global_center_indexes'], [])
        self.assertEqual(result['new_candidates_m'], [])
        self.assertFalse(result['final']['accepted'])

    def test_power_transforms_never_mutate_the_archived_inputs(self):
        powers = np.random.default_rng(771).exponential(size=(4, len(self.frequency)))
        original = powers.copy()
        result = self.matcher.match(powers, harmonic=True, refine=True)
        np.testing.assert_array_equal(powers, original)
        self.assertIsNone(result['final']['position_m'], 'an entirely uninformative model supplies no candidate')

    def test_invalid_or_nonfinite_power_grids_are_rejected(self):
        with self.assertRaises(ValueError):
            self.matcher.match(np.ones((4, 20)))
        with self.assertRaises(ValueError):
            self.matcher.match(np.full((4, len(self.frequency)), np.nan))


if __name__ == '__main__':
    unittest.main()
