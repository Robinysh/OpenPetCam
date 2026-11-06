"""Behavioral controls for conditional single-microphone localization."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.acoustics.localization import errors, rank_candidates, ripple_features, spatial_partition, spectrum


class LocalizationTests(unittest.TestCase):
    def setUp(self):
        self.f = np.fft.rfftfreq(8192, 1 / 16000)

    def test_anechoic_unknown_gain_is_not_location_information(self):
        transfers = np.stack([np.ones_like(self.f) / r**2 for r in (1, 2, 4)])
        feat, rms = ripple_features(transfers, self.f)
        result = rank_candidates(feat[0], feat, [[1, 0, 0], [2, 0, 0], [4, 0, 0]])
        self.assertTrue(np.all(rms < 1e-8))
        self.assertEqual(result["status"], "uninformative")
        self.assertFalse(result["accepted"])
        self.assertIsNone(result["position_m"])

    def test_distinct_echo_spacing_recovers_candidate_under_gain_and_smooth_color(self):
        delays = [.004, .007, .011]
        powers = np.array([abs(1 + .55 * np.exp(-2j * np.pi * self.f * d))**2 for d in delays])
        model, _ = ripple_features(powers, self.f)
        observed, _ = ripple_features(powers[1] * 100 * np.exp(-self.f / 5000), self.f)
        result = rank_candidates(observed, model, [[1, 0, 1], [2, 0, 2], [3, 0, 3]])
        self.assertEqual(result["best_index"], 1)
        self.assertFalse(result["accepted"], "raw similarity is not calibrated confidence")

    def test_symmetric_transfer_reports_ambiguity(self):
        power = abs(1 + .7 * np.exp(-2j * np.pi * self.f * .006))**2
        feature, _ = ripple_features(power, self.f)
        result = rank_candidates(feature, np.array([feature, feature]), [[-3, 0, 2], [3, 0, 2]],
                                 min_margin=0, min_similarity=.1)
        self.assertAlmostEqual(result["score_margin"], 0)
        self.assertEqual(len(result["near_best_candidates"]), 2)
        self.assertEqual(result["near_best_box_diagonal_m"], 6)
        self.assertFalse(result["accepted"])

    def test_unconstrained_source_can_explain_multiple_locations(self):
        rng = np.random.default_rng(71)
        source = np.fft.rfft(rng.normal(size=8192))
        h1 = 1 + .4 * np.exp(-2j * np.pi * self.f * .005)
        h2 = 1 + .6 * np.exp(-2j * np.pi * self.f * .014)
        recording = h1 * source
        alternative_source = recording / h2
        self.assertLess(np.linalg.norm(recording - h2 * alternative_source) / np.linalg.norm(recording), 1e-14)
        self.assertGreater(np.linalg.norm(source - alternative_source), 1)

    def test_tiles_and_height_variants_never_split(self):
        result = spatial_partition([[1.1, -.5, 2.1], [1.8, .6, 2.7], [-1.1, 0, 2.1]])
        self.assertEqual(result[0], result[1])
        self.assertEqual(result, spatial_partition([[1.1, -.5, 2.1], [1.8, .6, 2.7], [-1.1, 0, 2.1]]))

    def test_azimuth_wrap(self):
        result = errors([.01, 0, -1], [-.01, 0, -1])
        self.assertLess(result["azimuth_error_deg"], 2)

    def test_spectrum_does_not_mutate_signed_audio(self):
        x = np.random.default_rng(4).normal(size=16000)
        original = x.copy()
        f, p = spectrum(x, 16000)
        np.testing.assert_array_equal(x, original)
        self.assertEqual(len(f), len(p))
        self.assertTrue(np.all(p >= 0))

    def test_invalid_features_fail_closed(self):
        with self.assertRaises(ValueError):
            ripple_features(np.full_like(self.f, np.nan), self.f)
        with self.assertRaises(ValueError):
            spectrum(np.zeros((2, 400)), 16000)


if __name__ == "__main__":
    unittest.main()
