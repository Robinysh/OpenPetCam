"""Synthetic physics and qualification checks for active near-IR measurements.

No camera, saved room photographs, or model weights are used. Expected signals
are generated independently from the Lambertian near-light equation, rather
than from the estimator being tested.
"""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sim.world.nir import capture_quality, difference_signal, geometry_response  # noqa: E402


def verified_capture():
    """Synthetic readbacks describe one genuinely fixed, linear acquisition."""
    readbacks = []
    for led in ("off", "on", "off"):
        readbacks.append({"verified": True, "exposure_us": 10000.,
                          "analog_gain": 2., "digital_gain": 1.,
                          "processing_signature": "fixed-linear-test-pipeline",
                          "night_mode": "night", "ircut": "out",
                          "pan": 1850, "tilt": 716, "led": led})
    return {"schema_version": 2, "protocol": "nir-led-off-on-off",
            "controls_verified": True, "exposure_locked": True,
            "gain_locked": True, "processing_locked": True,
            "same_night_mode": True, "ircut_unchanged": True,
            "pose_fixed": True, "controls_readbacks": readbacks,
            "radiometry": {"response_model": "linear", "verified": True,
                           "black_level": 0., "white_level": 1.}}


class CaptureQualificationTest(unittest.TestCase):
    def test_verified_fixed_readbacks_are_required(self):
        self.assertTrue(capture_quality(verified_capture())["quantitative"])
        for missing in ({}, {"schema_version": 1}, {"controls_verified": True}):
            self.assertFalse(capture_quality(missing)["quantitative"])
        for flag in ("controls_verified", "exposure_locked", "gain_locked", "processing_locked",
                     "same_night_mode", "ircut_unchanged", "pose_fixed"):
            metadata = verified_capture()
            metadata[flag] = False
            self.assertFalse(capture_quality(metadata)["quantitative"], flag)

    def test_claimed_locks_cannot_override_changed_readbacks(self):
        for field, changed in (("exposure_us", 20000.), ("analog_gain", 4.),
                               ("digital_gain", 2.), ("processing_signature", "automatic-tone"),
                               ("night_mode", "day"), ("ircut", "in"), ("pan", 1860), ("tilt", 720)):
            metadata = verified_capture()
            metadata["controls_readbacks"][1][field] = changed
            result = capture_quality(metadata)
            self.assertFalse(result["quantitative"], field)
            self.assertTrue(result["reasons"], field)

    def test_missing_or_unverified_radiometry_is_not_linear_light(self):
        for change in ({"verified": False}, {"response_model": "unknown-jpeg"}):
            metadata = verified_capture()
            metadata["radiometry"].update(change)
            self.assertFalse(capture_quality(metadata)["quantitative"])
        metadata = verified_capture()
        metadata["controls_readbacks"][1]["verified"] = False
        self.assertFalse(capture_quality(metadata)["quantitative"])

    def test_equal_but_invalid_numeric_controls_are_not_verified_measurements(self):
        for field in ("exposure_us", "analog_gain", "digital_gain"):
            for value in (0., -1., np.inf, np.nan, True, "10000"):
                metadata = verified_capture()
                for reading in metadata["controls_readbacks"]:
                    reading[field] = value
                self.assertFalse(capture_quality(metadata)["quantitative"], (field, value))
        for field in ("pan", "tilt"):
            metadata = verified_capture()
            for reading in metadata["controls_readbacks"]:
                reading[field] = np.inf
            self.assertFalse(capture_quality(metadata)["quantitative"], field)

    def test_actual_led_state_sequence_must_match_the_claimed_protocol(self):
        for states in (("off", "off", "off"), ("on", "off", "on"), (None, "on", "off")):
            metadata = verified_capture()
            for reading, state in zip(metadata["controls_readbacks"], states):
                reading["led"] = state
            self.assertFalse(capture_quality(metadata)["quantitative"], states)


class DifferenceSignalTest(unittest.TestCase):
    def test_stable_off_subtraction_preserves_known_added_illumination(self):
        off = np.full((4, 5), .1)
        on = off + np.linspace(.1, .3, off.size).reshape(off.shape)
        result = difference_signal(off, on, off.copy(), verified_capture())
        self.assertTrue(result["quality"]["quantitative"])
        self.assertTrue(result["valid"].all())
        np.testing.assert_allclose(result["signal"], on - off, rtol=1e-10, atol=1e-12)

    def test_off_baseline_uses_actual_on_time_not_unweighted_mean(self):
        before = np.full((3, 4), .1)
        after = np.full((3, 4), .102)
        on = np.full((3, 4), .2)
        result = difference_signal(before, on, after, verified_capture(), times=(0., 1., 4.))
        self.assertTrue(result["valid"].all())
        np.testing.assert_allclose(result["signal"], .0995, atol=1e-12)

    def test_inverse_response_is_applied_to_each_frame_and_channel_first(self):
        metadata = verified_capture()
        metadata["radiometry"].update(response_model="lut", lut=(np.linspace(0, 1, 11)**2).tolist())
        before = np.array([[[.1, .1, .4]]])
        on = np.array([[[.5, .6, .6]]])
        result = difference_signal(before, on, before.copy(), metadata)
        self.assertTrue(result["valid"].all())
        expected = np.mean(on**2 - before**2, axis=-1)
        np.testing.assert_allclose(result["signal"], expected, atol=1e-12)
        self.assertGreater(abs(expected.item() - result["encoded_delta"].item()**2), .1)

    def test_legacy_jpeg_difference_is_qualitative_only(self):
        before = np.full((3, 4), 60.)
        on = before + 10
        result = difference_signal(before, on, before, {"pos": [1850, 716]})
        self.assertFalse(result["quality"]["quantitative"])
        self.assertFalse(result["valid"].any())
        self.assertTrue(np.isnan(result["signal"]).all())
        np.testing.assert_allclose(result["encoded_delta"], 10.)

    def test_saturation_ambient_drift_and_negative_difference_are_invalid(self):
        before = np.full((3, 4), .1)
        on = np.full((3, 4), .3)
        after = before.copy()
        on[0, 0] = 1.                 # clipped sensor output
        after[0, 1] = .8              # changing ambient light or moving object
        on[0, 2] = .05               # impossible positive LED contribution
        result = difference_signal(before, on, after, verified_capture())
        self.assertFalse(result["valid"][0, 0])
        self.assertFalse(result["valid"][0, 1])
        self.assertFalse(result["valid"][0, 2])
        self.assertTrue(result["valid"][1:, :].all())

    def test_invalid_code_values_and_motion_masks_never_become_signal(self):
        before = np.full((2, 3), .1)
        on = np.full((2, 3), .3)
        before[0, 0] = -.1
        on[0, 1] = np.nan
        motion = np.zeros((2, 3), bool)
        motion[0, 2] = True
        result = difference_signal(before, on, before.copy(), verified_capture(), motion_mask=motion)
        self.assertFalse(result["valid"][0].any())
        self.assertTrue(result["valid"][1].all())

    def test_off_frames_must_bracket_the_on_capture(self):
        image = np.full((2, 3), .1)
        for times in ((0., 0., 2.), (0., 3., 2.), (np.nan, 1., 2.)):
            with self.assertRaises(ValueError):
                difference_signal(image, image + .2, image, verified_capture(), times=times)


class GeometryResponseTest(unittest.TestCase):
    def compare_equal(self, response):
        mask = response["valid"]
        self.assertTrue(mask.all())
        values = response["response"][mask]
        self.assertGreater(float(values.min()), 0.)
        np.testing.assert_allclose(values / np.median(values), 1., atol=1e-8)

    def test_inverse_square_falloff_does_not_become_a_material_change(self):
        ranges = np.array([1., 2., 4.])
        directions = np.tile([0., 0., 1.], (3, 1))
        normals = -directions
        signal = .4 / ranges**2
        self.compare_equal(geometry_response(signal, ranges, normals, directions))

    def test_incidence_and_known_beam_pattern_are_corrected(self):
        ranges = np.array([1., 2., 3.])
        cosine = np.array([1., .8, .4])
        beam = np.array([1., .7, .5])
        directions = np.tile([0., 0., 1.], (3, 1))
        normals = np.column_stack([np.sqrt(1 - cosine**2), np.zeros(3), -cosine])
        signal = .3 * cosine * beam / ranges**2
        self.compare_equal(geometry_response(signal, ranges, normals, directions, beam=beam))

    def test_material_response_ratio_survives_common_normalization(self):
        directions = np.tile([0., 0., 1.], (2, 1))
        result = geometry_response(np.array([.2, .4]), np.ones(2), -directions, directions)
        self.assertTrue(result["valid"].all())
        self.assertAlmostEqual(result["response"][1] / result["response"][0], 2., places=10)

    def test_led_offset_uses_light_to_surface_distance_and_direction(self):
        ranges = np.array([1., 2., 3.])
        directions = np.tile([0., 0., 1.], (3, 1))
        normals = -directions
        light = np.array([1., 0., 0.])
        points = ranges[:, None] * directions
        toward_light = light - points
        distance = np.linalg.norm(toward_light, axis=1)
        incidence = np.sum(normals * toward_light / distance[:, None], axis=1)
        signal = .4 * incidence / distance**2
        self.compare_equal(geometry_response(signal, ranges, normals, directions, light_position=light))

    def test_grazing_backlit_or_invalid_geometry_is_not_silently_clamped(self):
        directions = np.tile([0., 0., 1.], (7, 1))
        normals = -directions.copy()
        normals[1] = [np.sqrt(1 - .01**2), 0., -.01]
        normals[2] = [0., 0., 1.]
        normals[5] = [np.nan, 0., -1.]
        ranges = np.array([1., 1., 1., 0., np.nan, 1., 1.])
        signal = np.array([.2, .2, .2, .2, .2, .2, -.2])
        result = geometry_response(signal, ranges, normals, directions)
        np.testing.assert_array_equal(result["valid"], [True, False, False, False, False, False, False])
        self.assertTrue(np.isnan(result["response"][1:]).all())

    def test_caller_masks_and_unilluminated_beam_are_respected(self):
        directions = np.tile([0., 0., 1.], (4, 1))
        result = geometry_response(np.full(4, .2), np.ones(4), -directions, directions,
                                   valid=np.array([True, False, True, True]), beam=np.array([1., 1., 0., np.nan]))
        np.testing.assert_array_equal(result["valid"], [True, False, False, False])
        self.assertTrue(np.isnan(result["response"][1:]).all())

    def test_depth_uncertainty_is_a_real_input_to_geometry_qualification(self):
        directions = np.tile([0., 0., 1.], (5, 1))
        result = geometry_response(np.full(5, .2), np.ones(5), -directions, directions,
                                   range_spread=np.array([.1, .3, np.nan, np.inf, -.1]))
        np.testing.assert_array_equal(result["valid"], [True, False, False, False, False])
        self.assertTrue(np.isnan(result["response"][1:]).all())


if __name__ == "__main__":
    unittest.main()
