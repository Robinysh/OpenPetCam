"""Coordinate and mass invariants for the offline angular search proxy."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'sim'))
from search_prior import group_candidates, load_prior


class SearchPriorTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.params = Path(self.tmp.name) / 'params.json'
        self.room_map = Path(self.tmp.name) / 'room_map.json'
        self.p = {'geometry': {'pan_max': 220, 'tilt_max': 150, 'pan_frame': 120, 'tilt_frame': 80},
                  'pet': {'room_map': [[100, 50, 3], [200, 50, 1]]}}
        self.m = {'version': 1, 'pan_step': 100, 'tilt_step': 50, 'min_floor': .02, 'pose_shift': [20, 10],
                  'poses': [{'pan': x, 'tilt': y, 'floor_share': (.01 if (x, y) == (100, 0) else .5)}
                            for x in (0, 100, 200, 300) for y in (0, 50, 100, 150)],
                  'stops': {'3': [[117, 62, .7], [216, 58, .2]]}, 'exclude_day': None}

    def load(self, **kw):
        self.params.write_text(json.dumps(self.p))
        self.room_map.write_text(json.dumps(self.m))
        return load_prior(self.params, self.room_map, **kw)

    def test_grid_shift_matches_reverse_lookup_and_physical_limits(self):
        prior = self.load()
        # One low-floor pose and seven physically unreachable poses are excluded.
        self.assertEqual(len(prior.candidates), 8)
        self.assertEqual(prior.provenance['out_of_bounds_candidates_filtered'], 7)
        self.assertIn([20, 10], prior.candidates.tolist())
        self.assertIn([220, 110], prior.candidates.tolist())
        self.assertNotIn([120, 10], prior.candidates.tolist())
        self.assertTrue(np.all(prior.candidates <= [220, 150]))
        self.assertEqual(prior.floor_share(20, 10), .5)
        self.assertEqual(prior.floor_share(120, 10), .01)
        self.assertEqual(prior.floor_share(118, 12), .01)
        self.assertTrue(np.diag(prior.visibility).all())

    def test_saved_stops_do_not_receive_grid_shift_or_mutate_source(self):
        prior = self.load()
        expected = [[117, 62, .7], [216, 58, .2]]
        self.assertEqual(prior.current_stops(3), expected)
        self.assertEqual(prior.current_stops(27), expected)
        prior.current_stops(3)[0][0] = -1
        self.assertEqual(prior.current_stops(3), expected)
        self.assertEqual(prior.current_stops(4), [])

    def test_histogram_and_broad_mass_are_normalized_on_known_floor(self):
        prior = self.load(uniform_mix=.3)
        self.assertAlmostEqual(float(prior.weights.sum()), 1)
        self.assertTrue((prior.weights >= .3 / len(prior.weights) - 1e-12).all())
        loc = {tuple(point): i for i, point in enumerate(prior.targets)}
        self.assertAlmostEqual(prior.weights[loc[120, 60]], .7 * .75 + .3 / 8)
        self.assertAlmostEqual(prior.weights[loc[220, 60]], .7 * .25 + .3 / 8)
        self.assertNotIn((120, 10), loc)
        self.assertFalse(prior.provenance['held_out_claim'])

    def test_floor_gate_and_unknown_views_are_not_counted_as_visible(self):
        self.m['poses'] = [row for row in self.m['poses'] if (row['pan'], row['tilt']) != (0, 150)]
        prior = self.load()
        # Both footprints reach known targets, but the first is low-floor and the second unknown.
        poses = [[120, 10], [20, 150]]
        self.assertFalse(prior.visibility_for(poses).any())
        self.assertTrue(prior.visibility_for(poses, floor_mask=False).any())
        self.assertFalse(prior.visibility_for([[-1, 60], [221, 60]], floor_mask=False).any())

    def test_visibility_is_discrete_footprint_not_continuous_pan_sweep(self):
        prior = self.load()
        vis = prior.visibility_for([[20, 60], [220, 60]], floor_mask=False)
        middle = np.where((prior.targets == [120, 60]).all(axis=1))[0][0]
        self.assertFalse(vis[:, middle].any())
        self.assertTrue(prior.visibility_for([[120, 60]])[0, middle])

    def test_coarse_targets_preserve_uniform_mass_and_all_candidates(self):
        full = self.load(uniform_mix=1)
        coarse = self.load(uniform_mix=1, target_step=(200, 100))
        np.testing.assert_array_equal(coarse.candidates, full.candidates)
        groups = group_candidates(full.candidates, (200, 100))
        self.assertEqual(len(coarse.targets), len(groups))
        np.testing.assert_allclose(coarse.weights, [len(g) / 8 for g in groups])
        self.assertAlmostEqual(coarse.weights.sum(), 1)
        self.assertEqual(coarse.visibility.shape, (8, len(groups)))

    def test_grouping_is_complete_stable_and_does_not_claim_equivalence(self):
        prior = self.load()
        groups = prior.candidate_groups((100, 50))
        self.assertEqual(sorted(i for g in groups for i in g), list(range(8)))
        self.assertEqual(groups, prior.candidate_groups((100, 50)))

    def test_explicit_histogram_mount_sensitivity_and_source_hashes(self):
        prior = self.load(prior_shift=(0, 0))
        self.assertEqual(prior.provenance['histogram_shift'], [0, 0])
        self.assertEqual(prior.provenance['grid_pose_shift'], [20, 10])
        self.assertEqual(prior.provenance['sources']['params']['sha256'], hashlib.sha256(self.params.read_bytes()).hexdigest())
        self.assertEqual(prior.provenance['sources']['room_map']['sha256'], hashlib.sha256(self.room_map.read_bytes()).hexdigest())
        json.dumps(prior.provenance, allow_nan=False)
        with self.assertRaises(ValueError):
            prior.weights[0] = 1

    def test_zero_histogram_uses_uniform_without_invented_empirical_mass(self):
        self.p['pet']['room_map'] = []
        prior = self.load(uniform_mix=0)
        np.testing.assert_allclose(prior.weights, np.ones(8) / 8)
        self.assertEqual(prior.provenance['uniform_mix_effective'], 1)

    def test_invalid_inputs_fail_explicitly(self):
        for mix in [-.1, 1.1, float('nan'), True]:
            with self.subTest(mix=mix), self.assertRaises(ValueError):
                self.load(uniform_mix=mix)
        self.p['pet']['room_map'] = [[100, 50, -1]]
        with self.assertRaises(ValueError):
            self.load()
        self.p['pet']['room_map'] = []
        self.m['poses'] = []
        with self.assertRaisesRegex(ValueError, 'no floor-bearing candidate'):
            self.load()


if __name__ == '__main__':
    unittest.main()
