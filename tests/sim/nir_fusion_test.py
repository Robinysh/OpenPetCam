"""Do not promote an old display overlay to a measured NIR property."""
import copy
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from sim.world import fuse  # noqa: E402
from sim.world.geometry import directions  # noqa: E402
from sim.world.sensors.nir_active import ActivePassiveNIR  # noqa: E402
from nir_sensor_test import _context, _write_capture  # noqa: E402


class LegacyNIRFusionTest(unittest.TestCase):
    def test_legacy_led_jpeg_alone_cannot_create_reflectivity(self):
        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp)
            width, height = 16, 8
            np.full((height, width), 2., np.float32).tofile(model / "range.f32")
            # This is a visualization, whose colour mapping clips and loses
            # signal values. Its presence is not radiometric provenance.
            Image.fromarray(np.full((height, width, 3), [200, 70, 190], np.uint8)).save(model / "pano_led_light.jpg")
            ctx = SimpleNamespace(model=str(model), _cache={}, room=None)
            manifest = {"room": {"floor_height_m": .8}, "layers": [
                {"id": "fused.surface", "data": {"file": "range.f32", "width": width, "height": height, "objects": []}},
                {"id": "acoustic.echoes", "data": {"items": []}},
            ]}
            with np.errstate(invalid="ignore"):
                summary, attributes = fuse.build(ctx, manifest)
            measured = attributes.get("nir_reflectance", np.full((height, width), np.nan))
            self.assertFalse(np.isfinite(measured).any(),
                             "A legacy JPEG without controlled capture evidence must not become reflectivity")
            self.assertNotIn("nir_reflectance", summary["textures"])


class NumericNIRFusionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.ctx = _context(self.root)
        _write_capture(self.root)
        self.layers = {layer.id: layer.to_json() for layer in ActivePassiveNIR(self.ctx).layers()}
        self.ranges = np.full((8, 16), 2.)
        self.directions = directions(16, 8)
        self.normals = -self.directions
        np.savez_compressed(self.root / "model" / "depth_evidence.npz",
                            spread=np.full((8, 16), .1), views=np.full((8, 16), 3))

    def fields(self, layers=None):
        return fuse.infrared_fields(self.ctx, self.layers if layers is None else layers,
                                    self.ranges, self.normals, self.directions)

    def test_controlled_numeric_signal_creates_geometry_response_without_a_jpeg(self):
        (self.root / "model" / "pano_led_light.jpg").unlink()
        fields, status = self.fields()
        np.testing.assert_allclose(fields["nir_led_signal"], 20 / 255, atol=1e-7)
        np.testing.assert_allclose(fields["nir_response"][1:-1], 1., atol=1e-7)
        self.assertFalse(status["reflectivity_valid"])
        self.assertTrue(status["response_valid"])
        self.assertNotIn("nir_reflectance", fields)

    def test_stale_numeric_file_without_qualifying_layer_is_not_admitted(self):
        fields, status = self.fields({})
        self.assertNotIn("nir_led_signal", fields)
        self.assertNotIn("nir_response", fields)
        self.assertFalse(status["response_valid"])

    def test_qualitative_layer_cannot_promote_even_valid_numeric_samples(self):
        layers = copy.deepcopy(self.layers)
        layers["infrared.led"]["data"].update(quantitative_observations=0, response_valid=False)
        fields, status = self.fields(layers)
        self.assertNotIn("nir_led_signal", fields)
        self.assertNotIn("nir_response", fields)
        self.assertFalse(status["response_valid"])

    def test_qualitative_numeric_difference_has_code_units_without_reflectivity(self):
        layers = copy.deepcopy(self.layers)
        layers["infrared.led"]["data"].update(quantitative_observations=0, response_valid=False,
                                                legacy_observations=1, qualitative_observations=1)
        fields, status = self.fields(layers)
        np.testing.assert_allclose(fields["nir_led_difference"], 20., atol=1e-7)
        self.assertEqual(status["status"], "qualitative_only")
        self.assertEqual(status["qualitative_observations"], 1)
        for unsupported in ("nir_led_signal", "nir_response", "nir_reflectance"):
            self.assertNotIn(unsupported, fields)

    def test_numeric_measurement_hash_is_required_and_enforced(self):
        layers = copy.deepcopy(self.layers)
        layers["infrared.led"]["data"].pop("numeric_sha256")
        fields, status = self.fields(layers)
        self.assertNotIn("nir_led_signal", fields,
                         "Unhashed numeric data has no verified link to its acquisition metadata")
        self.assertFalse(status["response_valid"])

    def test_changed_numeric_file_is_rejected(self):
        with open(self.root / "model" / "nir_signal.npz", "ab") as output:
            output.write(b"changed after provenance recorded")
        fields, status = self.fields()
        self.assertNotIn("nir_led_signal", fields)
        self.assertFalse(status["response_valid"])

    def test_old_calibration_panorama_is_not_used_after_camera_model_changes(self):
        self.ctx.camera.c = {"test_model": "changed-projection"}
        fields, status = self.fields()
        self.assertNotIn("nir_led_signal", fields,
                         "A numeric panorama projected with an old lens model is stale geometry")
        self.assertFalse(status["response_valid"])

    def test_missing_camera_provenance_is_not_assumed_current(self):
        layers = copy.deepcopy(self.layers)
        layers["infrared.led"]["data"].pop("calibration_sha256")
        fields, status = self.fields(layers)
        self.assertNotIn("nir_led_signal", fields)
        self.assertFalse(status["response_valid"])

    def test_uncertain_depth_preserves_signal_but_withholds_response(self):
        np.savez_compressed(self.root / "model" / "depth_evidence.npz",
                            spread=np.full((8, 16), .5), views=np.full((8, 16), 3))
        fields, status = self.fields()
        self.assertIn("nir_led_signal", fields)
        self.assertNotIn("nir_response", fields)
        self.assertFalse(status["response_valid"])

    def test_depth_evidence_on_another_grid_does_not_qualify_surface_normals(self):
        np.savez_compressed(self.root / "model" / "depth_evidence.npz",
                            spread=np.full((9, 16), .1), views=np.full((9, 16), 3))
        fields, status = self.fields()
        self.assertIn("nir_led_signal", fields)
        self.assertNotIn("nir_response", fields)
        self.assertFalse(status["response_valid"])

    def test_depth_jump_boundary_does_not_become_a_reflectivity_edge(self):
        self.ranges[:, 8:] = 4.
        fields, status = self.fields()
        response = fields["nir_response"]
        self.assertTrue(np.isnan(response[:, 7:9]).all())
        self.assertTrue(np.isfinite(response[1:-1, 2:5]).all())
        self.assertTrue(np.isfinite(response[1:-1, 10:13]).all())
        self.assertFalse(status["reflectivity_valid"])


if __name__ == "__main__":
    unittest.main()
