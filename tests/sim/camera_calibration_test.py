"""Regression tests for calibration geometry, without images or a live camera.

Run with ``python3 tests/sim/camera_calibration_test.py``.
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sim.world import camera  # noqa: E402


class _ObjectiveCaptured(Exception):
    pass


class CalibrationObjectiveTest(unittest.TestCase):
    def initial_objective(self, target_pan, target_xy=(960.0, 540.0)):
        """Evaluate the actual calibrate() solver objective before any optimization.

        Mocking feature extraction supplies a known correspondence; intercepting
        the solver avoids testing the optimizer's own convergence by accident.
        """
        captured = []

        def inspect_objective(fun, x0, **kwargs):
            captured.append(np.asarray(fun(x0), float))
            raise _ObjectiveCaptured

        poses = [{"file": "a.jpg", "pan": 1850, "tilt": 728},
                 {"file": "b.jpg", "pan": target_pan, "tilt": 728}]
        pairs = [(0, 1, np.array([[960.0, 540.0]]), np.array([target_xy]))]
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "poses.json").write_text(json.dumps({"poses": poses}))
            with patch.object(camera, "_features", return_value=([], [])), \
                    patch.object(camera, "_sift_pairs", return_value=pairs), \
                    patch("scipy.optimize.least_squares", side_effect=inspect_objective):
                with self.assertRaises(_ObjectiveCaptured):
                    camera.calibrate(tmp, os.path.join(tmp, "calib.json"))
        self.assertEqual(len(captured), 1)
        return captured[0]

    def test_out_of_frame_correspondence_cannot_zero_objective(self):
        # Centre ray from frame A lands beyond the right border of frame B.
        residual = self.initial_objective(2250)
        self.assertTrue(np.isfinite(residual).all())
        self.assertGreater(np.linalg.norm(residual), 100.0,
                           "Out-of-frame matches must retain their reprojection error")

    def test_behind_camera_correspondence_cannot_zero_objective(self):
        # The two views face nearly opposite directions: no valid match exists.
        residual = self.initial_objective(0)
        self.assertTrue(np.isfinite(residual).all())
        self.assertGreater(np.linalg.norm(residual), 100.0,
                           "Behind-camera matches must incur a finite positive penalty")

    def test_correct_correspondence_has_zero_projection_error(self):
        residual = self.initial_objective(1850)
        self.assertLess(np.linalg.norm(residual), 1e-9)


class CameraProjectionTest(unittest.TestCase):
    def setUp(self):
        self.cam = camera.Camera({"focal_px": 1700.0, "k1": -0.20, "k2": -0.04,
                                  "deg_per_step_pan": 0.086, "deg_per_step_tilt": 0.081,
                                  "level_tilt": 716.0, "roll_deg": 0.0})

    def test_unbounded_projection_scores_forward_points_outside_raster(self):
        # This forward direction is outside both the raster and the old r² mask.
        d = np.array([[0.8, 0.0, 1.0]])
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        u, v, valid = self.cam.project_unbounded(1850, 716, d)
        self.assertTrue(valid.all())
        self.assertGreater(u[0], 1920)
        self.assertTrue(np.isfinite([u, v]).all())
        self.assertFalse(self.cam.project(1850, 716, d)[2].any(),
                         "Rendering must continue to mask pixels outside the raster")

    def test_corner_inside_raster_is_scored_despite_conservative_render_mask(self):
        # A distorted corner can lie inside the image while its ideal radius
        # exceeds the conservative renderer's threshold of sqrt(0.45).
        d = np.array([[0.61, 0.30, 1.0]])
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        u, v, valid = self.cam.project_unbounded(1850, 716, d)
        self.assertTrue(valid.all())
        self.assertTrue(0 < u[0] < 1919 and 0 < v[0] < 1079)
        self.assertFalse(self.cam.project(1850, 716, d)[2].any())

    def test_corner_inverse_is_accurate_for_stronger_barrel_profile(self):
        c = camera.Camera(dict(self.cam.c, k1=-0.23, k2=-0.13))
        x, y = np.array([0, 1919, 0, 1919]), np.array([0, 0, 1079, 1079])
        rays = c.rays(1850, 716, x, y)
        u, v, valid = c.project_unbounded(1850, 716, rays)
        self.assertTrue(valid.all())
        self.assertLess(float(np.max(np.hypot(u - x, v - y))), 0.01,
                        "Inverse radial rays must reproduce full-sensor corners within 0.01 pixels")

    def test_invalid_and_nonfinite_matches_have_finite_positive_residuals(self):
        poses = [{"pan": 1850, "tilt": 716}, {"pan": 0, "tilt": 716}]
        pairs = [(0, 1, np.array([[960.0, 540.0], [np.nan, 540.0]]),
                  np.array([[960.0, 540.0], [960.0, 540.0]]))]
        residual = camera.calibration_residuals(self.cam, poses, pairs, np.zeros((2, 3)))
        self.assertTrue(np.isfinite(residual).all())
        self.assertGreater(np.linalg.norm(residual), 100.0)

    def test_vectorized_solver_preserves_errors_and_penalties(self):
        poses = [{"pan": 1850, "tilt": 716}, {"pan": 2240, "tilt": 620},
                 {"pan": 100, "tilt": 800}]
        rng = np.random.default_rng(749)
        pairs = []
        for i, j, count in [(0, 1, 21), (1, 2, 14), (2, 0, 10)]:
            a = rng.uniform([100, 80], [1820, 1000], (count, 2))
            b = rng.uniform([100, 80], [1820, 1000], (count, 2))
            pairs.append((i, j, a, b))
        pairs[0][2][0, 0] = np.nan
        pairs[1][3][0, 1] = np.inf
        offsets = np.array([[0.2, -0.4, 0.8], [-0.5, 0.7, -0.1], [0.4, 1.2, -0.8]])
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            expected = camera.calibration_residuals(self.cam, poses, pairs, offsets, scale=0.7)
            actual = camera._CalibrationObservations(poses, pairs).residuals(self.cam, offsets, scale=0.7)
        self.assertTrue(np.isfinite(actual).all())
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-8)


def _independent_projection(c, pose, directions):
    """Generate observations from analytic optics, without Camera.rays/project.

    SciPy's intrinsic Y-X-Z rotation supplies an independent Euler conversion.
    The image coordinates are the standard two-term radial model.
    """
    angles = [(1850 - pose["pan"]) * c["deg_per_step_pan"],
              -(pose["tilt"] - c["level_tilt"]) * c["deg_per_step_tilt"],
              c["roll_deg"]]
    local = directions @ Rotation.from_euler("YXZ", angles, degrees=True).as_matrix()
    xy = local[:, :2] / local[:, 2:]
    r2 = np.sum(xy ** 2, axis=1)
    xy *= (1 + c["k1"] * r2 + c["k2"] * r2 ** 2)[:, None]
    pixels = xy * [c["focal_px"], -c["focal_px"]] + [960.0, 540.0]
    ok = (local[:, 2] > 0.3) & (r2 < 0.43)
    ok &= ((pixels > [200, 100]) & (pixels < [1720, 980])).all(axis=1)
    return pixels, ok


class SyntheticCalibrationTest(unittest.TestCase):
    def test_recovers_optics_and_predicts_unseen_directions(self):
        truth = {"focal_px": 1680.0, "k1": -0.22, "k2": -0.04,
                 "deg_per_step_pan": 0.086, "deg_per_step_tilt": 0.081,
                 "level_tilt": 716.0, "roll_deg": 0.2}
        poses = [{"file": f"{i}.jpg", "pan": p, "tilt": t} for i, (p, t) in enumerate(
            [(1600, 600), (1850, 600), (2100, 600), (1600, 850), (1850, 850), (2100, 850)])]
        rng = np.random.default_rng(748)
        directions = np.column_stack([rng.uniform(-1.0, 1.0, 500),
                                      rng.uniform(-0.7, 0.7, 500), np.ones(500)])
        observed = [_independent_projection(truth, p, directions) for p in poses]
        pairs = []
        for i in range(len(poses)):
            for j in range(i + 1, len(poses)):
                ok = observed[i][1] & observed[j][1]
                if ok.sum() >= 12:
                    pairs.append((i, j, observed[i][0][ok], observed[j][0][ok]))
        self.assertGreater(len(pairs), 6)
        fitted, offsets, _result = camera.fit_matches(poses, pairs, prior_deg=0.01, max_nfev=150)
        self.assertAlmostEqual(fitted["focal_px"], truth["focal_px"], delta=0.5)
        self.assertAlmostEqual(fitted["k1"], truth["k1"], delta=0.002)
        self.assertAlmostEqual(fitted["k2"], truth["k2"], delta=0.005)
        # A second independent set of directions never entered the optimization.
        unseen = np.column_stack([rng.uniform(-1.0, 1.0, 150),
                                  rng.uniform(-0.7, 0.7, 150), np.ones(150)])
        cam = camera.Camera(fitted)
        errors = []
        for p, off in zip(poses, offsets):
            expected, ok = _independent_projection(truth, p, unseen)
            u, v, valid = cam.project_unbounded(p["pan"], p["tilt"], unseen, off=off)
            self.assertTrue(valid[ok].all())
            errors.extend(np.hypot(u[ok] - expected[ok, 0], v[ok] - expected[ok, 1]))
        self.assertGreater(len(errors), 100)
        self.assertLess(max(errors), 0.1)


class NaturalSceneOpticsTest(unittest.TestCase):
    def test_projection_jacobian_including_optimizer_extensions(self):
        from sim.world.calibration_selfcal import project_with_jac
        profile = {"focal_px": 1700.0, "k1": -0.24, "k2": -0.09}
        # The last three rows exercise the radial continuation and clipped-z
        # regions used during optimization, including a back-facing ray.
        rays = np.array([[0.0, 0.0, 1.0], [0.4, 0.2, 1.0], [0.9, 0.7, 1.0],
                         [0.03, -0.02, 0.03], [0.01, 0.02, -0.3]])
        _, analytic, valid = project_with_jac(rays, profile)
        self.assertEqual(valid.tolist(), [True, True, False, False, False])
        for k in range(3):
            low, high = rays.copy(), rays.copy()
            low[:, k] -= 1e-6
            high[:, k] += 1e-6
            numerical = (project_with_jac(high, profile)[0] - project_with_jac(low, profile)[0]) / 2e-6
            np.testing.assert_allclose(analytic[:, :, k], numerical, rtol=2e-6, atol=1e-3)

    def test_optical_parameterization_is_invertible_and_monotonic(self):
        from sim.world.calibration_selfcal import optical_profile, optics_parameters, RADIUS, DIAGONAL
        rng = np.random.default_rng(750)
        for _ in range(10):
            th = np.array([rng.uniform(np.log(1.005), np.log(1.5)),
                           rng.uniform(0.2, 1.5), rng.uniform(0.02, 1.5)])
            c = optical_profile(th, {})
            np.testing.assert_allclose(optics_parameters(c), th, rtol=1e-12, atol=1e-12)
            r = np.linspace(0, RADIUS, 1001)
            derivative = 1 + 3 * c["k1"] * r**2 + 5 * c["k2"] * r**4
            self.assertGreater(derivative.min(), 0.0)
            reach = c["focal_px"] * RADIUS * (1 + c["k1"] * RADIUS**2 + c["k2"] * RADIUS**4)
            self.assertGreaterEqual(reach, DIAGONAL * 1.005)

    def test_fitted_rotation_offsets_follow_camera_convention(self):
        from sim.world.calibration_validation import nominal_rotations
        c = {"focal_px": 1700.0, "k1": -0.24, "k2": -0.09,
             "deg_per_step_pan": 0.086, "deg_per_step_tilt": 0.081,
             "level_tilt": 716.0, "roll_deg": 0.2}
        cam = camera.Camera(c)
        rng = np.random.default_rng(751)
        for pan, tilt in [(0, 50), (1850, 716), (3600, 950)]:
            base = nominal_rotations([{"pan": pan, "tilt": tilt}], c)[0]
            fitted = base @ Rotation.from_rotvec(rng.uniform(-0.2, 0.2, 3)).as_matrix()
            angles = Rotation.from_matrix(base.T @ fitted).as_euler("YXZ", degrees=True)
            off = [angles[1], angles[0], angles[2]]
            np.testing.assert_allclose(cam.R(pan, tilt, off), fitted, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
