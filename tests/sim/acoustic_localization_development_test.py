"""Paired repeats, stationary responses and explicit conditional-prior controls."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.acoustics import localization_development as study
from sim.acoustics.geometry import Rig
from sim.acoustics.localization import spatial_partition
from sim.acoustics.localization_model import RoomModel, Surface


class DevelopmentTests(unittest.TestCase):
    def test_mean_power_uses_nested_identical_draws_without_mutation(self):
        powers = np.arange(16 * 100, dtype=float).reshape(16, 100)
        original = powers.copy()
        means = study.nested_mean_powers(powers)
        np.testing.assert_array_equal(means[1], original[0])
        np.testing.assert_allclose(means[4], original[:4].mean(0))
        np.testing.assert_allclose(means[16], original.mean(0))
        np.testing.assert_array_equal(powers, original)
        with self.assertRaises(ValueError):
            study.nested_mean_powers(-np.ones((16, 100)))

    def test_prior_range_uses_receiver_and_wrong_priors_stay_explicit(self):
        positions = np.array([[10, -.55, 1], [10, .2, 1], [10, -.55, 5], [10, .2, 5]])
        receiver, floor = np.array([10, 0, 0]), -1
        priors = study.priors(positions, receiver, floor)
        self.assertEqual({k: int(v.sum()) for k, v in priors.items()},
                         dict(zip(study.PRIOR_NAMES, (4, 2, 2, 1))))
        support = study.prior_support([10, .2, 5], receiver, floor)
        self.assertTrue(support['all_candidates'])
        self.assertFalse(support['low_height'])
        self.assertFalse(support['range_4m'])
        self.assertTrue(all(value is None for value in study.prior_support(None, receiver, floor).values()))

    def test_cases_never_leave_development_and_no_emitter_gets_no_source_truth(self):
        positions = np.array([[x, -.35, z] for x in np.arange(-3, 4, .5) for z in np.arange(-3, 4, .5)])
        visible = np.arange(len(positions)) % 3 != 0
        cases = study.freeze_cases(positions, visible, count=48, spacing=.5, seed=23)
        for case in cases:
            self.assertEqual(spatial_partition([case['simulation_anchor_m']])[0], 'development')
            self.assertEqual(len(set(case['burst_seeds'])), 16)
            if case['scenario'] == 'no_source_background':
                self.assertIsNone(case['source_m'])
            else:
                self.assertEqual(case['source_m'], case['simulation_anchor_m'])
            if case['scenario'] == 'occluded_source':
                self.assertFalse(case['anchor_direct_visible'])
        for start in range(0, len(cases), len(study.SCENARIOS)):
            independent, replay = cases[start:start + 2]
            self.assertEqual(independent['source_m'], replay['source_m'])
            self.assertEqual(independent['burst_seeds'], replay['burst_seeds'])
            np.testing.assert_array_equal(study.source_burst(independent, independent['burst_seeds'][0]),
                                          study.source_burst(replay, replay['burst_seeds'][0]))

    def test_replay_source_is_identical_but_independent_sources_are_not(self):
        case = dict(scenario='replayed_broadband', replay_seed=918)
        np.testing.assert_array_equal(study.source_burst(case, 1), study.source_burst(case, 2))
        case['scenario'] = 'independent_broadband'
        self.assertGreater(np.linalg.norm(study.source_burst(case, 1) - study.source_burst(case, 2)), 1)

    def test_model_and_tail_stay_fixed_within_source_case(self):
        model = RoomModel((Surface('floor', (0, -1, 0), (0, 1, 0)),))
        case = dict(scenario='diffuse_tail', room_seed=417, simulation_anchor_m=[2, -.55, 2])
        first, diagnostic = study.fixed_transfer(model, case, Rig().pose(0, 0))
        second, _ = study.fixed_transfer(model, case, Rig().pose(0, 0))
        np.testing.assert_array_equal(first, second)
        self.assertTrue(diagnostic['fixed_across_bursts'])
        self.assertGreater(diagnostic['ir_energy'], 0)

    def test_background_has_nonzero_power_without_inventing_source_error(self):
        case = dict(scenario='no_source_background', replay_seed=17, burst_seeds=list(range(16)))
        powers = study.burst_spectra(case, np.zeros(4000))
        self.assertEqual(powers.shape[0], 16)
        self.assertGreater(powers.sum(), 0)
        rows = [dict(source_m=None, best_index=0, score_margin=.1, feature_cosine_to_known_ir=None)]
        report = study.describe(rows)
        self.assertEqual(report['source_cases'], 0)
        self.assertEqual(report['background_conditional_candidates'], 1)
        self.assertEqual(report['all_source_cases'], {'cases': 0})

    def test_subset_best_index_maps_back_to_frozen_global_grid(self):
        features = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], float)
        positions = np.array([[i, 0, 0] for i in range(4)], float)
        result = study.rank_subset(np.array([0, 1]), features, positions, np.array([False, True, False, True]))
        self.assertEqual(result['best_index'], 1)
        self.assertEqual(result['position_m'], [1, 0, 0])
        self.assertFalse(result['accepted'])


if __name__ == '__main__':
    unittest.main()
