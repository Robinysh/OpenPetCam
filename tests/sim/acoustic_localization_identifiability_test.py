"""Analytical controls for oracle uniqueness and polar ambiguity summaries."""
import sys
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.acoustics.localization_identifiability import (circular_span_deg, material_profile_matrix,
    normalized_profile_matrix, oracle_rows, summarize)


class IdentifiabilityTests(unittest.TestCase):
    def test_identical_features_always_tie_even_if_first_index_is_truth(self):
        features = np.array([[1., 0.], [1., 0.]])
        profile, valid = material_profile_matrix(features, query_variant=0)
        rows = oracle_rows(profile, valid, [[0, 0, 1], [0, 0, 4]], [0, 0, 0])
        self.assertEqual(rows[0]["best_index"], 0)
        self.assertFalse(rows[0]["oracle_uniquely_identifiable"])
        self.assertFalse(rows[1]["oracle_uniquely_identifiable"])
        self.assertEqual(rows[0]["score_margin"], 0)
        self.assertEqual(rows[0]["near_range_span_m"], 3)
        self.assertEqual(rows[0]["near_max_azimuth_error_deg"], 0)
        self.assertEqual(summarize(rows)["near_range_ambiguous_azimuth_stable"], 2)

    def test_material_profile_retains_cross_variant_ties(self):
        features = np.array([[[1., 0.], [0., 1.]], [[0., 1.], [1., 0.]]])
        profile, valid = material_profile_matrix(features, query_variant=1)
        rows = oracle_rows(profile, valid, [[1, 0, 0], [0, 0, 1]], [0, 0, 0])
        self.assertEqual(rows[0]["exact_other_candidates"], [1])
        self.assertEqual(rows[1]["exact_other_candidates"], [0])
        self.assertTrue(all(not row["oracle_uniquely_identifiable"] for row in rows))

    def test_zero_signature_has_no_fabricated_candidate_or_ambiguity_span(self):
        features = np.array([[0., 0.], [1., 0.], [0., 1.]])
        profile, valid = material_profile_matrix(features, query_variant=0)
        rows = oracle_rows(profile, valid, [[0, 0, 1], [0, 0, 2], [1, 0, 0]], [0, 0, 0])
        self.assertIsNone(rows[0]["best_index"])
        self.assertIsNone(rows[0]["near_box_diagonal_m"])
        summary = summarize(rows)
        self.assertEqual(summary["uninformative"], 1)
        self.assertEqual(summary["oracle_unique_candidates"], 2)
        self.assertEqual(summary["oracle_unique_all_candidate_fraction"], 2 / 3)

    def test_unknown_gain_removes_raw_power_range_information(self):
        powers = np.array([[1., 1., 1.], [.25, .25, .25], [.0625, .0625, .0625]])
        profile, valid = normalized_profile_matrix(powers)
        np.testing.assert_allclose(profile, 1)
        self.assertTrue(valid.all())

    def test_complex_scalar_shape_removes_global_phase_and_gain_only(self):
        phase = np.array([1, 1j, -1, -1j])
        vectors = np.array([phase, phase * 2j, [1, 1, 1, 1]], complex)
        profile, _ = normalized_profile_matrix(vectors, complex_scalar=True)
        self.assertAlmostEqual(profile[0, 1], 1)
        self.assertLess(profile[0, 2], .1)

    def test_azimuth_wrap_uses_small_circular_span(self):
        angle = np.radians([179., -179.])
        points = np.column_stack([np.sin(angle), np.zeros(2), np.cos(angle)])
        rows = oracle_rows(np.ones((2, 2)), np.ones(2, bool), points, [0, 0, 0])
        self.assertAlmostEqual(circular_span_deg([179, -179]), 2)
        self.assertAlmostEqual(rows[0]["near_azimuth_span_deg"], 2)
        self.assertAlmostEqual(rows[0]["near_max_azimuth_error_deg"], 2)


if __name__ == "__main__":
    unittest.main()
