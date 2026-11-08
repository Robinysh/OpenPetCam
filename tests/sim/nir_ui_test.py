"""IR bundle views advertise only the acquisition quality their metadata supports."""
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sim.world.app_bundle import build  # noqa: E402


class InfraredBundleTest(unittest.TestCase):
    def make_bundle(self, quality=None):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        world, out = root / "world", root / "bundle"
        model = world / "model"
        model.mkdir(parents=True)
        textures = {k: k + ".png" for k in ("nir_excess", "nir_led_difference", "nir_led_signal", "nir_response", "nir_reflectance")}
        for name in textures.values():
            Image.new("RGB", (4, 2), (80, 120, 150)).save(model / name)
        np.array([1., 2.], np.float32).tofile(model / "range.f32")
        surf = {"textures": textures, "texture_ranges": {
            "uncertainty": [-2, 0], "echo_support": [0, 3], "echo_coverage": [0, 2],
            "dwell": [0, 1], "floor_residual": [-.15, .15], "light_range": [0, 120]}}
        if quality is not None:
            surf["infrared"] = quality
        manifest = {"room": {"floor_height_m": .8, "ceiling_m": 2.6, "outline": [], "walls": []},
                    "layers": [{"id": "fused.surface", "data": {"width": 2, "height": 1, "file": "range.f32"}},
                               {"id": "fused.surfels", "data": surf}]}
        (model / "manifest.json").write_text(json.dumps(manifest))
        (world / "calib.json").write_text(json.dumps({k: v for k, v in zip(
            ("focal_px", "k1", "k2", "deg_per_step_pan", "deg_per_step_tilt", "level_tilt", "roll_deg"),
            (1700., -.2, -.05, .085, .082, 716., 0.))}))
        return build(str(world), str(out)), out

    def test_legacy_textures_cannot_advertise_quantitative_ir(self):
        doc, out = self.make_bundle()
        self.assertIn("nir_excess", doc["textures"])
        for key in ("nir_led_difference", "nir_led_signal", "nir_response", "nir_reflectance"):
            self.assertNotIn(key, doc["textures"])
            self.assertFalse((out / f"room_{key}.png").exists())
        self.assertFalse(doc["infrared"]["reflectivity_valid"])
        self.assertFalse(doc["infrared"]["response_valid"])

    def test_qualitative_capture_exports_encoded_led_difference_only(self):
        quality = {"status": "qualitative_only", "legacy_observations": 1,
                   "response_valid": False, "reflectivity_valid": False,
                   "quantitative_observations": 0}
        doc, out = self.make_bundle(quality)
        self.assertEqual(doc["infrared"]["qualitative_observations"], 1)
        self.assertIn("nir_led_difference", doc["textures"])
        self.assertTrue((out / doc["textures"]["nir_led_difference"]).exists())
        legend = doc["legends"]["nir_led_difference"]
        self.assertEqual((legend["lo"], legend["hi"]), ("-16 DN", "+16 DN"))
        self.assertIn("qualitative", legend["title"])
        for key in ("nir_led_signal", "nir_response", "nir_reflectance"):
            self.assertNotIn(key, doc["textures"])

    def test_controlled_response_stays_distinct_from_calibrated_reflectivity(self):
        quality = {"status": "geometry_corrected_response", "response_valid": True,
                   "reflectivity_valid": False, "quantitative_observations": 2,
                   "limitations": ["LED beam uncalibrated"]}
        doc, out = self.make_bundle(quality)
        for key in ("nir_led_signal", "nir_response"):
            self.assertIn(key, doc["textures"])
            self.assertIn(key, doc["legends"])
            self.assertTrue((out / doc["textures"][key]).exists())
        self.assertNotIn("nir_reflectance", doc["textures"])
        self.assertEqual(doc["infrared"]["limitations"], quality["limitations"])

    def test_calibrated_reflectivity_requires_explicit_quality(self):
        doc, out = self.make_bundle({"status": "relative_reflectivity", "response_valid": True,
                                    "reflectivity_valid": True, "quantitative_observations": 2})
        self.assertIn("nir_reflectance", doc["textures"])
        self.assertIn("nir_reflectance", doc["legends"])
        self.assertTrue((out / doc["textures"]["nir_reflectance"]).exists())
        # Reusing an output folder must not re-advertise a stale calibrated
        # texture when a later manifest has no acquisition-quality evidence.
        world = out.parent / "world"
        source = world / "model" / "manifest.json"
        manifest = json.loads(source.read_text())
        manifest["layers"][1]["data"].pop("infrared")
        source.write_text(json.dumps(manifest))
        rebuilt = build(str(world), str(out))
        self.assertNotIn("nir_reflectance", rebuilt["textures"])
        self.assertFalse(rebuilt["infrared"]["reflectivity_valid"])


@unittest.skipUnless(shutil.which("node"), "JavaScript runtime unavailable")
class LegacyInfraredCopyTest(unittest.TestCase):
    def test_room_texture_gate_keeps_qualitative_and_controlled_ir_separate(self):
        room = (ROOT / "server" / "room.js").read_text()
        match = re.search(r"export function roomTextureAllowed\([^)]*\) \{.*?^\}", room, re.S | re.M)
        self.assertIsNotNone(match)
        program = match.group().replace("export ", "", 1) + """
const keys = ['nir_led_difference', 'nir_led_signal', 'nir_response', 'nir_reflectance'];
const cases = [{}, {legacy_observations: 1}, {qualitative_observations: 1}, {quantitative_observations: 1},
  {response_valid: true}, {reflectivity_valid: true}];
console.log(JSON.stringify(cases.map(infrared => keys.map(key => roomTextureAllowed({infrared}, key)))));
"""
        response = subprocess.run(["node", "-e", program], capture_output=True, text=True, check=True)
        self.assertEqual(json.loads(response.stdout), [[False, False, False, False],
                         [True, False, False, False], [True, False, False, False],
                         [False, True, False, False], [False, False, True, False],
                         [False, False, False, True]])

    def test_old_bundle_wording_is_qualified_without_repainting_its_colours(self):
        room = (ROOT / "server" / "room.js").read_text()
        viewer = (ROOT / "sim" / "world" / "viewer" / "index.html").read_text()
        helpers = []
        for source, name in ((room, "roomLegend"), (viewer, "qualifyInfraredLayer")):
            match = re.search(r"export function " + name + r"\([^)]*\) \{.*?^\}", source, re.S | re.M)
            self.assertIsNotNone(match)
            helpers.append(match.group().replace("export ", "", 1))
        program = "\n".join(helpers) + """
const oldLegend = {kind: 'gradient', stops: ['#123456', '#abcdef'], lo: 'less near-IR', hi: 'more near-IR', title: 'Materials'};
const oldLayer = {id: 'infrared.nir_excess', title: 'Materials', method: 'plants, fabrics and fur glow', band: {note: 'materials'}, data: {image: 'old.jpg'}};
const oldLED = {id: 'infrared.led', method: 'reflectance from distant surfaces', band: {}, data: {image: 'led.jpg'}};
console.log(JSON.stringify({legend: roomLegend({legends: {nir_excess: oldLegend}}, 'nir_excess'), layer: qualifyInfraredLayer(oldLayer), led: qualifyInfraredLayer(oldLED), original: oldLegend}));
"""
        response = subprocess.run(["node", "-e", program], capture_output=True, text=True, check=True)
        result = json.loads(response.stdout)
        self.assertEqual(result["legend"]["stops"], result["original"]["stops"])
        self.assertEqual(result["legend"]["title"], "filter response (qualitative)")
        self.assertEqual(result["legend"]["lo"], "lower response")
        self.assertEqual(result["legend"]["hi"], "higher response")
        self.assertEqual(result["layer"]["data"]["image"], "old.jpg")
        self.assertIn("not a material identity", result["layer"]["method"])
        self.assertIn("without verified", result["led"]["method"])
        self.assertEqual(result["original"]["title"], "Materials")


if __name__ == "__main__":
    unittest.main()
