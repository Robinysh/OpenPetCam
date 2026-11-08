"""Analytic acoustic geometry and regressions; no camera, server or model data.

Run: python3 tests/sim/acoustic_geometry_test.py
"""
import math
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sim.world import nearfield  # noqa: E402
from sim.world.geometry import directions  # noqa: E402
from sim.acoustics.geometry import Rig, angles_from_motor, one_surface_paths, ray_plane_ranges  # noqa: E402


class LegacyGeometryRegressionTests(unittest.TestCase):
    def test_face_override_produces_finite_distance_sensitive_planes(self):
        rays = directions(72, 36)
        original = np.ones((36, 72))
        near = nearfield.face_override(original, rays, 0.25)
        far = nearfield.face_override(original, rays, 0.48)
        changed = ~np.isclose(near, original, equal_nan=True)
        self.assertGreater(int((changed & np.isfinite(near)).sum()), 0,
                           "candidate face must add finite ray-plane intersections")
        self.assertFalse(np.array_equal(near, far, equal_nan=True),
                         "different candidate face distances must produce different geometry")

    def test_pan_body_speaker_is_independent_of_head_tilt(self):
        shape = nearfield.grid_shape()
        _, level_source, _ = nearfield.camera_solid(shape, np.array([0., 0., 1.]), 0)
        p = math.radians(30)
        _, raised_source, _ = nearfield.camera_solid(shape, np.array([0., math.sin(p), math.cos(p)]), 0)
        np.testing.assert_allclose(level_source, raised_source, atol=1e-12, rtol=0,
                                   err_msg="tilting the head must not translate the pan-body speaker")


class RigTests(unittest.TestCase):
    def test_reference_lens_and_motor_signs(self):
        rig = Rig()
        np.testing.assert_allclose(rig.pose(0., 0.).lens_m, [0., 0., 0.], atol=1e-15)
        calib = {"deg_per_step_pan": .1, "deg_per_step_tilt": .2, "level_tilt": 700.}
        yaw, pitch = angles_from_motor(calib, 950., 850.)
        self.assertAlmostEqual(yaw, math.pi / 2)
        self.assertAlmostEqual(pitch, math.pi / 6)
        pose = rig.pose_motor(calib, 950., 850.)
        np.testing.assert_allclose(pose.receiver_axis, [math.cos(pitch), math.sin(pitch), 0.], atol=1e-14)

    def test_fixed_pan_pivot_speaker_tilt_independence_and_rigid_head(self):
        rig = Rig(tilt_offset_m=(.005, -.01, .002))
        for yaw in (-1.3, 0., 2.1):
            reference = rig.pose(yaw, 0.)
            for pitch in (-.5, .4, .8):
                pose = rig.pose(yaw, pitch)
                np.testing.assert_allclose(pose.pan_origin_m, rig.pan_origin_m, atol=1e-15)
                np.testing.assert_allclose(pose.tilt_origin_m, reference.tilt_origin_m, atol=1e-15)
                np.testing.assert_allclose(pose.source_m, reference.source_m, atol=1e-15)
                np.testing.assert_allclose(pose.source_axis, reference.source_axis, atol=1e-15)
                self.assertAlmostEqual(np.linalg.norm(pose.receiver_m - pose.lens_m),
                                       np.linalg.norm(np.array(rig.mic_offset_m) - rig.lens_offset_m), places=13)
                np.testing.assert_allclose(pose.head_rotation.T @ pose.head_rotation, np.eye(3), atol=1e-14)
                self.assertAlmostEqual(np.linalg.det(pose.head_rotation), 1.)

    def test_pan_origin_does_not_orbit_around_lens(self):
        rig = Rig()
        level = rig.pose(0., 0.)
        quarter = rig.pose(math.pi / 2, 0.)
        np.testing.assert_allclose(level.pan_origin_m, quarter.pan_origin_m, atol=1e-15)
        np.testing.assert_allclose(quarter.lens_m, [.029, 0., -.029], atol=1e-14)
        self.assertGreater(np.linalg.norm(level.lens_m - quarter.lens_m), .04)

    def test_bad_calibration_or_offsets_are_rejected(self):
        with self.assertRaises(ValueError):
            Rig(mic_offset_m=(0., float("nan"), 0.))
        with self.assertRaises(ValueError):
            Rig().pose(float("inf"), 0.)
        with self.assertRaises(ValueError):
            angles_from_motor({"deg_per_step_pan": 1., "deg_per_step_tilt": 1., "level_tilt": 0.}, 0., float("nan"))


class PlanePathTests(unittest.TestCase):
    def test_monostatic_delay_is_twice_perpendicular_distance(self):
        p = one_surface_paths([0., 0., 0.], [0., 0., 0.], [0., 0., 1.], 2.)
        self.assertTrue(p.valid)
        self.assertEqual(p.direct_m, 0.)
        self.assertEqual(p.reflected_m, 4.)
        self.assertAlmostEqual(float(p.reflected_m - p.direct_m) / 343., 4. / 343.)
        np.testing.assert_allclose(p.reflection_points_m, [0., 0., 2.], atol=1e-14)

    def test_bistatic_matches_analytic_image_source_and_specular_point(self):
        source = np.array([0., 0., 0.])
        receiver = np.array([.3, 0., 0.])
        p = one_surface_paths(source, receiver, [0., 0., 1.], 2.)
        self.assertAlmostEqual(float(p.direct_m), .3)
        self.assertAlmostEqual(float(p.reflected_m), math.hypot(4., .3))
        np.testing.assert_allclose(p.reflection_points_m, [.15, 0., 2.], atol=1e-14)
        via_point = np.linalg.norm(source - p.reflection_points_m) + np.linalg.norm(receiver - p.reflection_points_m)
        self.assertAlmostEqual(float(p.reflected_m), via_point, places=13)

    def test_reciprocity_normal_sign_and_broadcast(self):
        sources = np.array([[.1, .2, .3], [-.4, .2, -.1], [.3, -.1, .7]])
        receiver = np.array([.03, -.04, .2])
        n = np.array([1., 2., 3.]) / math.sqrt(14)
        forward = one_surface_paths(sources, receiver, n, 1.8)
        reverse = one_surface_paths(receiver, sources, n, 1.8)
        flipped = one_surface_paths(sources, receiver, -n, -1.8)
        self.assertTrue(forward.valid.all())
        np.testing.assert_allclose(forward.direct_m, reverse.direct_m, atol=1e-14)
        np.testing.assert_allclose(forward.reflected_m, reverse.reflected_m, atol=1e-14)
        np.testing.assert_allclose(forward.reflection_points_m, reverse.reflection_points_m, atol=1e-14)
        np.testing.assert_allclose(forward.reflected_m, flipped.reflected_m, atol=1e-14)
        np.testing.assert_allclose(forward.reflection_points_m @ n, 1.8, atol=1e-14)

    def test_opposite_sides_on_plane_or_nonfinite_are_invalid(self):
        sources = np.array([[0., 0., 0.], [0., 0., 1.], [float("nan"), 0., 0.], [0., 0., 2.]])
        receivers = np.array([[0., 0., 2.], [0., 0., 0.], [0., 0., 0.], [0., 0., 3.]])
        p = one_surface_paths(sources, receivers, [0., 0., 1.], 1.)
        np.testing.assert_array_equal(p.valid, [False, False, False, True])
        self.assertTrue(np.isnan(p.reflected_m[:3]).all())
        self.assertTrue(np.isnan(p.reflection_points_m[:3]).all())
        self.assertEqual(p.reflected_m[-1], 3.)

    def test_plane_movement_produces_expected_delay_change(self):
        near = one_surface_paths([0., 0., 0.], [0., 0., 0.], [0., 0., -1.], .4)
        far = one_surface_paths([0., 0., 0.], [0., 0., 0.], [0., 0., -1.], .45)
        self.assertAlmostEqual(float(far.reflected_m - near.reflected_m), .1)

    def test_nonunit_normal_is_rejected(self):
        with self.assertRaises(ValueError):
            one_surface_paths([0., 0., 0.], [0., 0., 0.], [0., 0., 2.], 1.)
        with self.assertRaises(ValueError):
            one_surface_paths(0., [0., 0., 0.], [0., 0., 1.], 1.)


class RayPlaneTests(unittest.TestCase):
    def test_normal_orientation_nonunit_rays_and_origin(self):
        rays = np.array([[0., 0., -2.], [1., 0., -1.], [1., 0., 0.], [0., 0., 1.]])
        result = ray_plane_ranges(rays, [0., 0., -1.], .5)
        np.testing.assert_allclose(result[:2], [.5, math.sqrt(.5)], atol=1e-14)
        self.assertTrue(np.isnan(result[2:]).all())
        np.testing.assert_allclose(result, ray_plane_ranges(rays, [0., 0., 1.], -.5), equal_nan=True)
        self.assertAlmostEqual(float(ray_plane_ranges([0., 0., -1.], [0., 0., -1.], .5, origin_m=(0., 0., -.2))), .3)


if __name__ == "__main__":
    unittest.main()
