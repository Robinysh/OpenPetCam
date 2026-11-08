"""Analytical controls for a source-comb prior and narrow observed-peak masks."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.acoustics.localization_harmonic import attenuate_lines, detect_comb


class HarmonicTests(unittest.TestCase):
    def setUp(self):
        self.frequency = np.fft.rfftfreq(8192, 1 / 16000)

    def comb(self, f0=220., width=2., amplitude=100.):
        background = np.exp(-self.frequency / 5000) * (1.2 + .2 * np.cos(2 * np.pi * self.frequency / 91.))
        power = background.copy()
        for order in range(1, 12):
            power *= 1 + amplitude / order**.3 / (1 + ((self.frequency - f0 * order) / width)**2)
        return power

    def test_unknown_fundamental_is_observed_and_only_masked_bins_change(self):
        power = self.comb()
        original = power.copy()
        corrected, diagnostic, mask = attenuate_lines(power, self.frequency)
        self.assertTrue(diagnostic['gated'])
        self.assertAlmostEqual(diagnostic['estimated_f0_hz'], 220., delta=.5)
        self.assertGreaterEqual(diagnostic['matched_count'], 6)
        self.assertLess(diagnostic['low_band_masked_fraction'], .2)
        np.testing.assert_array_equal(power, original)
        np.testing.assert_array_equal(corrected[~mask], power[~mask])
        self.assertLess(corrected[np.argmax(power)] / power.max(), .2)

    def test_broad_room_comb_does_not_meet_narrow_peak_width_gate(self):
        power = (1.1 + np.cos(2 * np.pi * self.frequency / 220.))**2
        corrected, diagnostic, mask = attenuate_lines(power, self.frequency)
        self.assertFalse(diagnostic['gated'])
        self.assertFalse(mask.any())
        np.testing.assert_array_equal(corrected, power)

    def test_zero_and_smooth_background_are_unchanged(self):
        for power in (np.zeros(len(self.frequency)), np.exp(-self.frequency / 2000)):
            corrected, diagnostic, mask = attenuate_lines(power, self.frequency)
            self.assertFalse(diagnostic['gated'])
            self.assertFalse(mask.any())
            np.testing.assert_array_equal(corrected, power)

    def test_same_narrow_comb_cannot_identify_source_versus_room_origin(self):
        observed_source_comb = self.comb()
        observed_room_comb = observed_source_comb.copy()
        first = detect_comb(observed_source_comb, self.frequency)
        second = detect_comb(observed_room_comb, self.frequency)
        self.assertTrue(first['gated'])
        self.assertEqual(first, second, 'A single identical power spectrum contains no source-origin label')

    def test_invalid_signed_or_nonfinite_power_rejected(self):
        with self.assertRaises(ValueError):
            detect_comb(-np.ones(len(self.frequency)), self.frequency)
        with self.assertRaises(ValueError):
            detect_comb(np.full(len(self.frequency), np.nan), self.frequency)


if __name__ == '__main__':
    unittest.main()
