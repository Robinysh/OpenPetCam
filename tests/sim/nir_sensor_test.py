"""Active NIR ingestion retains provenance and keeps exposure scales separate."""
import copy
import hashlib
import json
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
from sim.world.sensors.nir_active import ActivePassiveNIR  # noqa: E402
from nir_reflectance_test import verified_capture  # noqa: E402


class _FixedPixelCamera:
    c = {"test_model": "fixed-pixel"}

    def project(self, pan, tilt, directions, w, h, off=(0, 0, 0)):
        shape = directions.shape[:-1]
        return np.full(shape, w / 2), np.full(shape, h / 2), np.ones(shape, bool)


def _context(root):
    model = root / "model"
    model.mkdir()
    return SimpleNamespace(model=str(model), camera=_FixedPixelCamera(),
                           path=lambda *parts: str(root.joinpath(*parts)))


def _write_capture(root, name="a", *, exposure=10000., added=20, burst=1):
    folder = root / "raw" / "ir" / name
    folder.mkdir(parents=True)
    metadata = verified_capture()
    metadata["radiometry"]["white_level"] = 255.
    metadata.update(pos=[1850, 716], status="complete", frames=[])
    for phase, led in enumerate(("off", "on", "off")):
        controls = copy.deepcopy(metadata["controls_readbacks"][phase])
        controls["exposure_us"] = exposure
        for k in range(burst):
            filename = f"{phase}_{k}.png"
            pixels = np.full((32, 48, 3), 20 + (added if led == "on" else 0), np.uint8)
            Image.fromarray(pixels).save(folder / filename)
            metadata["frames"].append({"file": filename, "phase": phase, "led": led,
                                       "t": 1. + phase * 2 + k * .1,
                                       "sha256": hashlib.sha256((folder / filename).read_bytes()).hexdigest(),
                                       "controls": copy.deepcopy(controls), "controls_after": copy.deepcopy(controls)})
    (folder / "log.json").write_text(json.dumps(metadata))
    return folder, metadata


def _led_layer(layers):
    return next(layer for layer in layers if layer.id == "infrared.led")


class NIRSensorTest(unittest.TestCase):
    def test_verified_capture_keeps_numeric_signal_instead_of_overlay_colours(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctx = _context(root)
            _write_capture(root)
            layer = _led_layer(ActivePassiveNIR(ctx).layers())
            self.assertEqual(layer.data["quantitative_observations"], 1)
            self.assertFalse(layer.data["reflectivity_valid"])
            with np.load(root / "model" / layer.data["numeric_file"]) as z:
                self.assertTrue(z["valid"].all())
                np.testing.assert_allclose(z["led_signal"], 20 / 255, atol=1e-7)
                self.assertEqual(z["led_signal"].dtype, np.float32)

    def test_changed_exposure_captures_do_not_share_a_response_scale(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctx = _context(root)
            _write_capture(root, "a", exposure=10000., added=20)
            _write_capture(root, "b", exposure=20000., added=60)
            layer = _led_layer(ActivePassiveNIR(ctx).layers())
            self.assertEqual(layer.data["quantitative_observations"], 1)
            self.assertEqual(layer.data["legacy_observations"], 1)
            with np.load(root / "model" / layer.data["numeric_file"]) as z:
                np.testing.assert_allclose(z["led_signal"], 20 / 255, atol=1e-7)
                self.assertTrue((z["observations"] == 1).all())

    def test_missing_after_readback_is_not_filled_in_from_before(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctx = _context(root)
            folder, metadata = _write_capture(root)
            metadata["frames"][1].pop("controls_after")
            (folder / "log.json").write_text(json.dumps(metadata))
            layer = _led_layer(ActivePassiveNIR(ctx).layers())
            self.assertEqual(layer.data["quantitative_observations"], 0)
            self.assertFalse(layer.data["response_valid"])

    def test_switching_led_during_a_frame_cannot_hide_in_global_protocol(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctx = _context(root)
            folder, metadata = _write_capture(root)
            # Flattened readbacks still compress to off/on/off, but the first
            # off exposure was not bracketed by the same LED state.
            metadata["frames"][0]["controls_after"]["led"] = "on"
            (folder / "log.json").write_text(json.dumps(metadata))
            layer = _led_layer(ActivePassiveNIR(ctx).layers())
            self.assertEqual(layer.data["quantitative_observations"], 0)
            self.assertFalse(layer.data["response_valid"])

    def test_modified_capture_file_is_not_a_verified_numeric_measurement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctx = _context(root)
            folder, metadata = _write_capture(root)
            Image.fromarray(np.full((32, 48, 3), 90, np.uint8)).save(folder / metadata["frames"][1]["file"])
            layer = _led_layer(ActivePassiveNIR(ctx).layers())
            self.assertEqual(layer.data["quantitative_observations"], 0)
            self.assertFalse(layer.data["response_valid"])

    def test_even_encoded_bursts_are_not_promoted_through_nonlinear_response(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ctx = _context(root)
            folder, metadata = _write_capture(root, burst=2)
            metadata["radiometry"].update(response_model="lut", lut=(np.linspace(0, 1, 256)**2).tolist())
            (folder / "log.json").write_text(json.dumps(metadata))
            layer = _led_layer(ActivePassiveNIR(ctx).layers())
            self.assertEqual(layer.data["quantitative_observations"], 0)


if __name__ == "__main__":
    unittest.main()
