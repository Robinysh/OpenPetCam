"""Behavioral checks for genuinely held-out optical-profile validation."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.world.calibration_validation import (capture_overlap, nominal_rotations,
    project_bearings, score_profile, undistort_points, validation_split, _stats, _skew, _right_jacobian)


PROFILE = {"focal_px": 1720., "k1": -.24, "k2": -.09,
           "deg_per_step_pan": .085, "deg_per_step_tilt": .08,
           "level_tilt": 720., "roll_deg": .2}


def observations():
    rng = np.random.default_rng(777)
    poses = [{"pan": 1850 + 45 * i, "tilt": 700 + 12 * (i % 2)} for i in range(7)]
    rr = nominal_rotations(poses, PROFILE)
    pairs = []
    for i in range(len(poses)):
        for j in range(i + 1, len(poses)):
            rays = np.column_stack([rng.uniform(-.38, .38, 160), rng.uniform(-.23, .23, 160), np.ones(160)])
            world = rays @ rr[i].T
            other = world @ rr[j]
            a, _ = project_bearings(rays, PROFILE)
            b, good = project_bearings(other, PROFILE)
            good &= (b[:, 0] >= 0) & (b[:, 0] < 1920) & (b[:, 1] >= 0) & (b[:, 1] < 1080)
            pairs.append((i, j, a[good], b[good]))
    return poses, pairs


class CalibrationValidationTest(unittest.TestCase):
    def test_analytic_rotation_derivative_matches_finite_differences(self):
        from scipy.spatial.transform import Rotation
        for th in [np.zeros(3), np.array([.2, -.08, .04]), np.array([-.7, .3, -.2])]:
            v = np.array([.3, -.1, 1.])
            rr = Rotation.from_rotvec(th).as_matrix()
            actual = -rr @ _skew(v[None])[0] @ _right_jacobian(th[None])[0]
            numeric = np.column_stack([(Rotation.from_rotvec(th + np.eye(3)[k] * 1e-6).apply(v) -
                                        Rotation.from_rotvec(th - np.eye(3)[k] * 1e-6).apply(v)) / 2e-6 for k in range(3)])
            np.testing.assert_allclose(actual, numeric, atol=2e-10)

    def test_independent_radial_inverse_recovers_edge_rays(self):
        rays = np.array([[0., 0., 1.], [.65, .3, 1.], [-.6, -.28, 1.]])
        uv, good = project_bearings(rays, PROFILE)
        actual, invertible = undistort_points(uv, PROFILE)
        self.assertTrue(good.all() and invertible.all())
        np.testing.assert_allclose(actual, rays / np.linalg.norm(rays, axis=1)[:, None], atol=1e-12)

    def test_pair_and_feature_holdouts_are_disjoint_and_connected(self):
        poses, pairs = observations()
        fit, score = validation_split(pairs, len(poses))
        self.assertFalse({(i, j) for i, j, _, _ in fit} & {(i, j) for i, j, _, _ in score})
        def features(edges):
            return {(frame, *np.rint(pt * 100).astype(int)) for i, j, a, b in edges
                    for frame, points in [(i, a), (j, b)] for pt in points}
        self.assertFalse(features(fit) & features(score))
        visited = {0}
        while True:
            previous = len(visited)
            for i, j, _, _ in fit:
                if i in visited or j in visited:
                    visited.update([i, j])
            if previous == len(visited):
                break
        self.assertEqual(len(visited), len(poses))
        fit2, score2 = validation_split(pairs, len(poses))
        self.assertEqual([(i, j) for i, j, _, _ in score], [(i, j) for i, j, _, _ in score2])
        self.assertEqual(features(fit), features(fit2))

    def test_truth_beats_wrong_lens_on_unseen_pairs_and_features(self):
        poses, pairs = observations()
        fit, score = validation_split(pairs, len(poses))
        truth = score_profile(PROFILE, poses, fit, score, max_nfev=50)
        wrong = score_profile(dict(PROFILE, k1=0., k2=0.), poses, fit, score, max_nfev=50)
        self.assertLess(truth["all"]["median_px"], 1e-7)
        self.assertGreater(wrong["all"]["median_px"], 1.)
        self.assertEqual(truth["all"]["matches"], wrong["all"]["matches"])

    def test_invalid_projections_cannot_reduce_denominator(self):
        stats = _stats(np.array([0., 0., 0.]), np.array([True, False, False]))
        self.assertEqual(stats["matches"], 3)
        self.assertEqual(stats["invalid"], 2)
        self.assertGreater(stats["median_px"], 1000)
        _, front = project_bearings(np.array([[0., 0., -1.]]), PROFILE)
        self.assertFalse(front[0])

    def test_radial_foldback_is_invalid_even_when_it_returns_inside_image(self):
        p = dict(PROFILE, k1=-.3, k2=-.1)
        uv, valid = project_bearings(np.array([[1.3, 0., 1.]]), p)
        self.assertTrue(0 <= uv[0, 0] < 1920)
        self.assertFalse(valid[0], "a folded radial prediction must not earn a small pixel error")
        stats = _stats(np.zeros(1), valid)
        self.assertEqual(stats["matches"], 1)
        self.assertEqual(stats["invalid"], 1)
        self.assertEqual(stats["median_px"], 3840.)

    def test_sweep_overlap_uses_pixels_not_filenames(self):
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp) / "a", Path(tmp) / "b"
            a.mkdir(); b.mkdir()
            for root in [a, b]:
                (root / "poses.json").write_text(json.dumps({"poses": [{"file": "frame.jpg"}]}))
            (a / "frame.jpg").write_bytes(b"identical capture")
            (b / "frame.jpg").write_bytes(b"identical capture")
            self.assertFalse(capture_overlap(a, b)["independent_captures"])
            (b / "frame.jpg").write_bytes(b"new capture")
            self.assertTrue(capture_overlap(a, b)["independent_captures"])


if __name__ == "__main__":
    unittest.main()
