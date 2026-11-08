"""Truth-independent lattice construction and separated score-seed controls."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.acoustics.localization_adaptive import distinct_centers, local_positions


class AdaptiveTests(unittest.TestCase):
    def test_score_seeds_are_spatially_distinct_and_skip_uninformative_points(self):
        positions = np.array([[1, 0, 1], [1.1, 0, 1], [2, 0, 1], [3, 0, 1]], float)
        self.assertEqual(distinct_centers([.9, .89, .8, None], positions, count=2), [0, 2])
        self.assertEqual(distinct_centers([None] * 4, positions), [])

    def test_local_lattice_preserves_height_and_has_no_existing_duplicates(self):
        parent = np.array([[2, -.357, 2], [2, .393, 2]])
        before = parent.copy()
        local = local_positions(parent[:1], parent, (1, 3, 1, 3))
        self.assertEqual(local.shape, (24, 3))
        self.assertTrue(np.all(local[:, 1] == parent[0, 1]))
        self.assertLessEqual(np.max(np.abs(local[:, [0, 2]] - parent[0, [0, 2]])), .0625)
        self.assertFalse(any(tuple(p) in {tuple(x) for x in parent} for p in local))
        np.testing.assert_array_equal(parent, before)

    def test_only_declared_bounds_clip_the_candidate_lattice(self):
        parent = np.array([[2, -.357, 2]])
        local = local_positions(parent, parent, (2, 3, 1, 3))
        self.assertEqual(len(local), 14)
        self.assertTrue(np.all(local[:, 0] >= 2))

    def test_no_informative_centers_produce_no_invented_local_candidates(self):
        parent = np.array([[2, -.357, 2]])
        local = local_positions(np.empty((0, 3)), parent, (1, 3, 1, 3))
        self.assertEqual(local.shape, (0, 3))


if __name__ == '__main__':
    unittest.main()
