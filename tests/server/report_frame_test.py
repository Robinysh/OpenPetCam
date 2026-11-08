"""A report's picture must show what the owner was looking at when they opened the form, not what the camera shows
once they finish typing. While a search runs the camera can be a whole view away by then: the report then holds a
picture without the thing the note describes, and the owner's words get checked against the wrong moment."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("PETCAM_SITE_ENV", str(ROOT / "config/site.env.example"))
sys.path.insert(0, str(ROOT / "server"))
import app


class ReportFrameTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)
        self.clock = 1000.0
        for target, value in [
            (patch.object(app, "REPORTS_DIR", temp.name), None),
            (patch.object(app, "live_frame", side_effect=lambda: f"at-{self.clock:.0f}".encode()), None),
            (patch.object(app, "_report_clip"), None),
            (patch.object(app, "read_live", return_value={}), None),
            (patch.object(app, "get_settings", return_value={}), None),
            (patch.object(app, "list_zones", return_value=[]), None),
            (patch.object(app, "_db_rows", return_value=[]), None),
            (patch.object(app.time, "time", side_effect=lambda: self.clock), None),
            (patch.dict(app._report_last, clear=True), None),
            (patch.dict(app._report_held, clear=True), None),
        ]:
            target.start()
            self.addCleanup(target.stop)

    def send(self, device="phone"):
        rid = app.file_report(device, "Someone", {"note": "look", "client": {}})["id"]
        return self.dir / rid

    def test_report_keeps_the_picture_from_when_the_form_opened(self):
        app.hold_report_frame("phone")
        self.clock += 27
        d = self.send()
        self.assertEqual((d / "frame.jpg").read_bytes(), b"at-1000",
                         "the report kept the picture from when it was sent, after the camera had moved on")
        self.assertEqual((d / "frame_sent.jpg").read_bytes(), b"at-1027")
        rep = json.loads((d / "report.json").read_text())
        self.assertEqual(rep["frame_t"], 1000.0)

    def test_without_a_held_picture_the_report_uses_the_current_one(self):
        d = self.send()
        self.assertEqual((d / "frame.jpg").read_bytes(), b"at-1000")
        self.assertFalse((d / "frame_sent.jpg").exists())
        self.assertEqual(json.loads((d / "report.json").read_text())["frame_t"], 1000.0)

    def test_a_held_picture_is_used_once_and_only_by_its_device(self):
        app.hold_report_frame("phone")
        self.clock += 5
        other = self.send("laptop")
        self.assertEqual((other / "frame.jpg").read_bytes(), b"at-1005")
        self.clock += 5
        first = self.send("phone")
        self.assertEqual((first / "frame.jpg").read_bytes(), b"at-1000")
        self.clock += 30
        second = self.send("phone")
        self.assertEqual((second / "frame.jpg").read_bytes(), b"at-1040")

    def test_a_form_abandoned_long_ago_does_not_supply_the_picture(self):
        app.hold_report_frame("phone")
        self.clock += app.REPORT_HOLD_S + 1
        d = self.send()
        self.assertEqual((d / "frame.jpg").read_bytes(), b"at-1601")
        self.assertFalse((d / "frame_sent.jpg").exists())

    def test_a_failed_new_hold_does_not_leave_the_older_picture_behind(self):
        app.hold_report_frame("phone")
        with patch.object(app, "live_frame", side_effect=app.Busy("The camera is paused. Resume it first.")):
            with self.assertRaises(app.Busy):
                app.hold_report_frame("phone")
        self.assertNotIn("phone", app._report_held)

    def test_paused_camera_keeps_no_picture_at_all(self):
        app.hold_report_frame("phone")
        with patch.object(app, "live_frame", side_effect=app.Busy("The camera is paused. Resume it first.")):
            d = self.send()
        self.assertFalse((d / "frame.jpg").exists(), "a held picture was written while the camera was paused")
        self.assertFalse((d / "frame_sent.jpg").exists())
        self.assertIn("paused", json.loads((d / "report.json").read_text())["frame_error"])
        self.assertNotIn("phone", app._report_held)

    def test_unreachable_camera_at_sending_still_keeps_the_held_picture(self):
        app.hold_report_frame("phone")
        with patch.object(app, "live_frame", side_effect=TimeoutError("no answer")):
            d = self.send()
        self.assertEqual((d / "frame.jpg").read_bytes(), b"at-1000")
        self.assertFalse((d / "frame_sent.jpg").exists())
        self.assertIn("no answer", json.loads((d / "report.json").read_text())["frame_error"])

    def test_turning_privacy_on_forgets_held_pictures(self):
        app.hold_report_frame("phone")
        with patch.object(app, "_privacy_operation"), patch.object(app, "_privacy_transition", return_value={}):
            app.set_privacy(False)
            self.assertIn("phone", app._report_held)
            app.set_privacy(True)
        self.assertNotIn("phone", app._report_held)


class ReportFormTest(unittest.TestCase):
    def test_opening_the_form_asks_for_the_picture_and_survives_a_failure(self):
        source = (ROOT / "server/requests.js").read_text()
        handler = re.search(r"\$\('report-new'\)\.onclick = (\(\) => \{.*?\n    \});", source, re.S)
        self.assertIsNotNone(handler)
        js = """
const calls = [];
const node = () => ({ classList: { remove() {}, add() {} }, focus() {}, textContent: '' });
const nodes = {};
const $ = id => nodes[id] || (nodes[id] = node());
const snapshot = () => ({ shown_status: 'x' });
const api = path => { calls.push(path); return Promise.reject(new Error('camera paused')); };
const open = %s;
process.on('unhandledRejection', () => { console.log('UNHANDLED'); process.exit(1); });
open();
setTimeout(() => console.log(JSON.stringify({ calls, shown: nodes['report-shown'].textContent })), 20);
""" % handler.group(1)
        result = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        out = json.loads(result.stdout)
        self.assertEqual(out["calls"], ["/api/report/hold"])
        self.assertIn("x", out["shown"])


class CaseFileTest(unittest.TestCase):
    def test_the_case_file_says_which_moment_the_picture_shows(self):
        site = types.SimpleNamespace(ROOT=str(ROOT), SITE={"TZ": "UTC"}, SSH=[])     # no install config in a checkout
        with patch.dict(sys.modules, {"siteconf": site}), patch.object(sys, "path", sys.path + [str(ROOT / "tools")]):
            sys.modules.pop("triage", None)
            import triage
            with tempfile.TemporaryDirectory() as temp:
                report = {"id": "r", "ts": 1027.0, "frame_t": 1000.0, "local_time": "t"}
                Path(temp, "report.json").write_text(json.dumps(report))
                triage.summarize(temp)
                self.assertNotIn("frame_sent.jpg", Path(temp, "summary.md").read_text())
                Path(temp, "frame_sent.jpg").write_bytes(b"x")
                triage.summarize(temp)
                summary = Path(temp, "summary.md").read_text()
        self.assertIn("27 s before sending", summary)
        self.assertIn("frame_sent.jpg", summary)


if __name__ == "__main__":
    unittest.main()
