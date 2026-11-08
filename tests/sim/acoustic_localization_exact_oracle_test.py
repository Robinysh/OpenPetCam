"""Physical single-path power has no contrast; fractional kernels are not truth."""
from dataclasses import replace
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.acoustics.localization import rank_candidates, ripple_features
from sim.acoustics.localization_exact_oracle import LOW, exact_feature
from sim.acoustics.localization_model import AcousticPath, render_ir


class ExactOracleTests(unittest.TestCase):
    def setUp(self):
        self.frequency = np.fft.rfftfreq(8192, 1 / 16000)
        self.path = AcousticPath(0, 2.3, 2.3 / 343., 1 / (4 * np.pi * 2.3),
                                 ((0., 0., 2.3), (0., 0., 0.)))

    def test_fractional_single_path_is_uninformative_despite_rendered_ir_ripple(self):
        feature, diagnostic = exact_feature([self.path], self.frequency)
        h = render_ir([self.path], 16000, .25, .01)
        rendered, rms = ripple_features(abs(np.fft.rfft(h, n=8192)) ** 2, self.frequency, **LOW)
        self.assertFalse(diagnostic['informative'])
        self.assertLess(diagnostic['exact_log_power_ripple_rms'], 1e-10)
        self.assertEqual(np.linalg.norm(feature), 0)
        self.assertGreater(float(rms), 1e-8)
        self.assertAlmostEqual(np.linalg.norm(rendered), 1)
        dictionary = np.array([rendered, -rendered])
        rank = rank_candidates(feature, dictionary, [[0, 0, 2], [2, 0, 0]])
        self.assertIsNone(rank['position_m'])
        self.assertFalse(rank['accepted'])
        self.assertEqual(rank_candidates(rendered, dictionary, [[0, 0, 2], [2, 0, 0]])['best_index'], 0)

    def test_empty_and_zero_gain_paths_do_not_fabricate_information(self):
        for paths in ([], [replace(self.path, gain=0.)]):
            feature, diagnostic = exact_feature(paths, self.frequency)
            self.assertFalse(diagnostic['informative'])
            self.assertEqual(diagnostic['active_paths'], 0)
            self.assertEqual(np.linalg.norm(feature), 0)

    def test_single_signed_path_is_still_uninformative(self):
        for gain in (-1., .5, 20.):
            feature, diagnostic = exact_feature([replace(self.path, gain=gain)], self.frequency)
            self.assertFalse(diagnostic['informative'])
            self.assertEqual(np.linalg.norm(feature), 0)

    def test_distinct_relative_delays_have_power_interference(self):
        echo = replace(self.path, order=1, gain=self.path.gain * .6, delay_s=self.path.delay_s + .006)
        feature, diagnostic = exact_feature([self.path, echo], self.frequency)
        self.assertTrue(diagnostic['informative'])
        self.assertGreater(diagnostic['exact_log_power_ripple_rms'], .1)
        self.assertAlmostEqual(np.linalg.norm(feature), 1)
        shifted = [replace(path, delay_s=path.delay_s + .023) for path in (self.path, echo)]
        invariant, _ = exact_feature(shifted, self.frequency)
        np.testing.assert_allclose(invariant, feature, atol=1e-12)

    def test_same_delay_paths_can_have_flat_power_despite_count_two(self):
        feature, diagnostic = exact_feature([self.path, replace(self.path, gain=self.path.gain * .5)], self.frequency)
        self.assertEqual(diagnostic['active_paths'], 2)
        self.assertFalse(diagnostic['informative'])
        self.assertEqual(np.linalg.norm(feature), 0)


if __name__ == '__main__':
    unittest.main()
