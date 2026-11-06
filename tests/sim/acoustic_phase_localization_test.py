"""Independent analytic Green sums test the source/phase information ladder."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.acoustics.phase_localization import (arbitrary_source_control, arbitrary_source_control_spectra,
    blind_fir, blind_fir_spectra, known_source, known_source_spectra, smooth_source_basis)


def analytic(frequency, lengths, gains):
    """A separate explicit image-path sum, with no fitter/render helper calls."""
    return sum(gain * np.exp(-2j * np.pi * frequency * length / 343.) / (4 * np.pi * length)
               for length, gain in zip(lengths, gains))


class PhaseLocalizationTest(unittest.TestCase):
    def setUp(self):
        self.f = np.linspace(80., 220., 71)
        self.g = np.array([analytic(self.f, [2, 2.9], [1, .6]),
                           analytic(self.f, [2.3, 4.7], [1, -.5]),
                           analytic(self.f, [3.1, 4.1], [1, .8])])
        rng = np.random.default_rng(742)
        self.s = rng.normal(size=len(self.f)) + 1j * rng.normal(size=len(self.f))

    def test_known_multipath_recovers_index_real_gain_and_one_common_onset(self):
        y = 1.7 * self.s * self.g[1] * np.exp(-2j * np.pi * self.f * .003)
        result = known_source_spectra(y, self.s, self.g, self.f, onsets_s=[-.003, 0, .003])
        self.assertEqual(result['best_index'], 1)
        self.assertEqual(result['best_onset_s'], .003)
        self.assertLess(result['normalized_residual'], 1e-12)
        self.assertAlmostEqual(result['profiles'][1]['gain'], 1.7)
        self.assertFalse(result['accepted'])
        self.assertFalse(result['real_world_validated'])
        self.assertEqual(len(result['normalized_residual_profiles']), 3)
        self.assertTrue(all(len(row) == 3 for row in result['normalized_residual_profiles']))

    def test_positive_gain_is_linear_and_source_polarity_is_explicit(self):
        for multiplier in (.2, 3.5):
            result = known_source_spectra(multiplier * self.g[1] * self.s, self.s, self.g, self.f)
            self.assertEqual(result['best_index'], 1)
            self.assertAlmostEqual(result['profiles'][1]['gain'], multiplier)
        negative = -2 * self.g[1] * self.s
        positive_only = known_source_spectra(negative, self.s, self.g[1:2], self.f)
        self.assertIsNone(positive_only['best_index'])
        self.assertEqual(positive_only['profiles'][0]['gain'], 0)
        self.assertEqual(positive_only['profiles'][0]['normalized_residual'], 1)
        declared_signed = known_source_spectra(negative, self.s, self.g[1:2], self.f, positive_gain=False)
        self.assertEqual(declared_signed['best_index'], 0)
        self.assertAlmostEqual(declared_signed['profiles'][0]['gain'], -2)

    def test_single_path_unknown_level_and_onset_are_spatially_ambiguous(self):
        greens = np.array([analytic(self.f, [2], [1]), analytic(self.f, [3], [1])])
        y = self.s * greens[0]
        result = known_source_spectra(y, self.s, greens, self.f, onsets_s=[-1/343., 0, 1/343.])
        self.assertEqual(result['status'], 'ambiguous')
        self.assertIsNone(result['best_index'])
        self.assertEqual(result['near_best_indices'], [0, 1])
        self.assertLess(max(row['normalized_residual'] for row in result['profiles']), 1e-12)

    def test_onset_is_bounded_and_arbitrary_frequency_phase_cannot_be_fitted(self):
        quadrature = 1j * self.s * self.g[0]
        result = known_source_spectra(quadrature, self.s, self.g[:1], self.f, onsets_s=[0])
        self.assertIsNone(result['best_index'])
        self.assertAlmostEqual(result['profiles'][0]['normalized_residual'], 1)
        shifted = self.s * self.g[0] * np.exp(-2j * np.pi * self.f * .007)
        bounded = known_source_spectra(shifted, self.s, self.g[:1], self.f, onsets_s=[0, .001])
        self.assertIn(bounded['profiles'][0]['onset_s'], [0, .001])
        self.assertGreater(bounded['profiles'][0]['normalized_residual'], .1)

    def test_tiny_physical_green_and_observation_scales_do_not_become_silence(self):
        baseline = known_source_spectra(self.s * self.g[1], self.s, self.g, self.f)
        for scale in (1e-9, 1e-12, 1e-100):
            result = known_source_spectra(self.s * self.g[1] * scale, self.s, self.g * scale, self.f)
            self.assertEqual(result['best_index'], baseline['best_index'])
            np.testing.assert_allclose([r['normalized_residual'] for r in result['profiles']],
                                       [r['normalized_residual'] for r in baseline['profiles']], atol=1e-14)
            self.assertAlmostEqual(result['profiles'][1]['gain'], 1)

    def test_zero_observation_excitation_and_green_groups_retain_every_candidate(self):
        for y, s, g in [(np.zeros_like(self.s), self.s, self.g),
                        (self.s, np.zeros_like(self.s), self.g),
                        (self.s, self.s, np.zeros_like(self.g))]:
            result = known_source_spectra(y, s, g, self.f)
            self.assertIsNone(result['best_index'])
            self.assertEqual(len(result['profiles']), 3)
            self.assertFalse(result['accepted'])
        g = np.concatenate((np.zeros_like(self.g[:1]), self.g[1:2]))
        result = known_source_spectra(self.s * self.g[1], self.s, g, self.f)
        self.assertIsNone(result['profiles'][0]['objective'])
        self.assertEqual(result['profiles'][0]['active_frequency_bins'], 0)
        self.assertEqual(result['best_index'], 1)

    def test_arbitrary_source_control_fits_every_nonzero_green_but_never_returns_coordinates(self):
        y = self.s * self.g[1]
        result = arbitrary_source_control_spectra(y, self.g, self.f)
        self.assertEqual(result['compatible_indices'], [0, 1, 2])
        self.assertTrue(all(row['normalized_residual'] == 0 for row in result['profiles']))
        self.assertIsNone(result['best_index'])
        self.assertEqual(result['status'], 'source_unidentifiable')
        self.assertFalse(result['accepted'])
        partial = self.g.copy(); partial[0, :10] = 0
        result = arbitrary_source_control_spectra(y, partial, self.f)
        self.assertGreater(result['profiles'][0]['normalized_residual'], 0)
        self.assertNotIn(0, result['compatible_indices'])

    def test_short_causal_unknown_wavelet_can_recover_multipath_conditionally(self):
        rate = 1000.
        wavelet = np.array([.4, -.2, .3])
        source = np.exp(-2j * np.pi * self.f[:, None] * np.arange(3) / rate) @ wavelet
        result = blind_fir_spectra(self.g[1] * source, self.g, self.f, rate, source_length=3, ridge=0)
        self.assertEqual(result['best_index'], 1)
        self.assertLess(result['normalized_residual'], 1e-20)
        np.testing.assert_allclose(result['profiles'][1]['source_fir'], wavelet, atol=1e-12)
        self.assertEqual(result['profiles'][1]['source_design_rank'], 3)
        self.assertFalse(result['accepted'])

    def test_flexible_source_fir_absorbs_distinct_greens_and_causes_abstention(self):
        n = 64
        f = np.fft.rfftfreq(n, 1/1000.)
        irs = np.zeros((3, n))
        irs[:, 0] = 1
        irs[0, 4], irs[1, 7], irs[2, 11] = .5, -.4, .3
        g = np.fft.rfft(irs, axis=1)
        wavelet = np.random.default_rng(9).normal(size=n)
        y = g[1] * np.fft.rfft(wavelet)
        result = blind_fir_spectra(y, g, f, 1000, source_length=n, ridge=0)
        self.assertIsNone(result['best_index'])
        self.assertEqual(result['near_best_indices'], [0, 1, 2])
        self.assertTrue(all(row['normalized_residual'] < 1e-20 for row in result['profiles']))

    def test_smooth_causal_source_basis_is_explicit_and_orthonormal(self):
        basis = smooth_source_basis(12, 3)
        np.testing.assert_allclose(basis.T @ basis, np.eye(3), atol=1e-15)
        wavelet = basis @ np.array([.2, -.1, .4])
        spectrum = np.exp(-2j * np.pi * self.f[:, None] * np.arange(12) / 1000.) @ wavelet
        result = blind_fir_spectra(self.g[1] * spectrum, self.g, self.f, 1000,
                                  source_length=12, source_basis=basis, ridge=0)
        self.assertEqual(result['source_basis_columns'], 3)
        self.assertEqual(result['source_length'], 12)
        self.assertEqual(result['best_index'], 1)

    def test_ridge_invertibility_does_not_fabricate_measured_source_design_rank(self):
        y = self.s * self.g[0]
        result = blind_fir_spectra(y, self.g[:1], self.f, 5488, source_length=32, ridge=.01)
        self.assertLess(result['profiles'][0]['source_design_rank'], 32)
        self.assertEqual(len(result['profiles'][0]['source_design_singular_values']), 32)
        scaled = blind_fir_spectra(y * 1e-12, self.g[:1] * 1e-12, self.f, 5488,
                                  source_length=32, ridge=.01)
        self.assertAlmostEqual(result['profiles'][0]['objective'], scaled['profiles'][0]['objective'])
        json.dumps(result, allow_nan=False)

    def test_signed_waveform_and_ir_dictionary_use_discrete_convolution_without_extra_dt(self):
        irs = np.zeros((3, 24)); irs[:, 1] = 1
        irs[0, 5], irs[1, 8], irs[2, 13] = .4, -.7, .5
        source = np.array([.2, -.3, .4, .1])
        observed = np.convolve(source, irs[1]) * 2.3
        known = known_source(observed, source, irs, 1000, green_kind='ir')
        self.assertEqual(known['best_index'], 1)
        self.assertAlmostEqual(known['profiles'][1]['gain'], 2.3)
        blind = blind_fir(observed, irs, 1000, green_kind='ir', source_length=4, ridge=0)
        self.assertEqual(blind['best_index'], 1)
        np.testing.assert_allclose(blind['profiles'][1]['source_fir'], source * 2.3, atol=1e-12)
        control = arbitrary_source_control(observed, irs, 1000, green_kind='ir')
        self.assertIsNone(control['best_index'])

    def test_inputs_are_immutable_results_json_safe_and_frequency_masks_declared(self):
        y = self.s * self.g[1]
        originals = y.copy(), self.s.copy(), self.g.copy(), self.f.copy()
        result = known_source_spectra(y, self.s, self.g, self.f, band_hz=(100, 180))
        self.assertEqual(result['comparison_frequency_bins'], 41)
        json.dumps(result, allow_nan=False)
        for actual, original in zip((y, self.s, self.g, self.f), originals):
            np.testing.assert_array_equal(actual, original)

    def test_invalid_inputs_and_unbounded_nuisance_are_rejected(self):
        y = self.s * self.g[1]
        for kw in [dict(onsets_s=[]), dict(onsets_s=np.linspace(0, 1, 130)),
                   dict(onsets_s=[11]), dict(positive_gain=1), dict(band_hz=(500, 900)),
                   dict(min_active_bins=0), dict(tie_tolerance=-1)]:
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                known_source_spectra(y, self.s, self.g, self.f, **kw)
        with self.assertRaises(ValueError):
            known_source_spectra(y, self.s[:-1], self.g, self.f)
        with self.assertRaises(ValueError):
            known_source_spectra(y, self.s, self.g * np.nan, self.f)
        for kw in [dict(source_length=129), dict(ridge=-1), dict(source_length=3, source_basis=np.ones((3, 2))),
                   dict(source_length=3, source_basis=np.ones((3, 1), complex))]:
            with self.subTest(kw=kw), self.assertRaises(ValueError):
                blind_fir_spectra(y, self.g, self.f, 1000, **kw)


if __name__ == '__main__':
    unittest.main()
