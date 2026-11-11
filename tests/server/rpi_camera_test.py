"""A Raspberry Pi camera without motors (CAMERA=rpi, PTZ=0): every move is refused without touching the
network, the pose is a fixed (0, 0), the picture comes from MediaMTX instead of the camera's CGIs, and the
web app is told what to hide. Run: python3 tests/server/rpi_camera_test.py (no camera or server needed)."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
_tmp = tempfile.TemporaryDirectory()
_env = Path(_tmp.name) / "site.env"
_env.write_text("PET_NAME=Test\nCAMERA=rpi\nPTZ=0\n")
os.environ["PETCAM_SITE_ENV"] = str(_env)
sys.path.insert(0, str(ROOT / "server"))
import app  # noqa: E402
import petconf  # noqa: E402
import watch  # noqa: E402


def no_network(*a, **k):
    raise AssertionError("a camera or network call was made")


class RpiNoMotors(unittest.TestCase):
    def setUp(self):
        self.state = tempfile.TemporaryDirectory()
        self.addCleanup(self.state.cleanup)
        for p in (patch.object(app, "STATE_DIR", self.state.name),
                  patch.object(app, "cam_get", no_network), patch.object(app, "cam_login", no_network),
                  patch.object(app._camsh, "run", no_network),
                  patch("urllib.request.urlopen", no_network), patch("http.client.HTTPConnection", no_network)):
            p.start()
            self.addCleanup(p.stop)

    def test_settings(self):
        self.assertFalse(petconf.PTZ)
        self.assertFalse(petconf.AUDIO)
        self.assertFalse(petconf.THINGINO)
        self.assertEqual(petconf.caps(), {"ptz": False, "audio": False, "imaging": False, "talk": False})
        self.assertIn('"caps"', petconf.template_vars()["PET_JS"])

    def test_moves_are_refused_without_io(self):
        for call in (lambda: app.goto(10, 10), lambda: app.aim(0.1, 0.1), lambda: app.nudge(1, 0),
                     lambda: app.motor_cmd("g", 1, 1), lambda: app.recalibrate(), lambda: app.request_find(),
                     lambda: app.add_preset("sofa")):
            with self.assertRaises(app.NoMotors):
                call()
        self.assertTrue(issubclass(app.NoMotors, app.Busy))   # the routes already answer Busy with 409
        self.assertFalse(os.path.exists(f"{self.state.name}/find_request"))

    def test_fixed_pose_and_state(self):
        self.assertEqual(app.motor_status(), ("0", 0, 0))
        self.assertEqual(app.position(), (0, 0))
        self.assertEqual(app.wait_idle(), (0, 0))
        s = app.state()
        self.assertFalse(s["ptz"])
        self.assertEqual(s["presets"], [])

    def test_picture_comes_from_mediamtx(self):
        seen = []

        def run(cmd, **k):
            seen.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout=b"\xff\xd8jpeg\xff\xd9", stderr=b"")
        with patch("subprocess.run", run):
            self.assertEqual(app.live_frame(), b"\xff\xd8jpeg\xff\xd9")
        self.assertEqual(seen[0][0], "ffmpeg")
        self.assertIn("rtsp://127.0.0.1:8554/sub", seen[0])

    def test_no_talk_or_imaging(self):
        with self.assertRaises(app.Busy):
            app.set_imaging("night")
        with self.assertRaises(app.Busy):
            app.talk("d", "Phone", b"x", "audio/webm")


class RpiPrivacy(unittest.TestCase):
    """Pause releases the camera through MediaMTX. Live finding (2026-11-10): MediaMTX rejects a patch that leaves the
    secondary 'sub' stream without its rpiCamera primary, so pause is sub then cam and resume is cam then sub."""
    def setUp(self):
        self.state = tempfile.TemporaryDirectory()
        self.addCleanup(self.state.cleanup)
        self.source = {"cam": "rpiCamera", "sub": "rpiCamera"}
        self.calls = []

        def mtx(method, path, body=None):
            name = path.rsplit("/", 1)[1]
            if method == "PATCH":
                new = dict(self.source, **{name: body["source"]})
                if new["sub"] == "rpiCamera" and new["cam"] != "rpiCamera":
                    raise OSError("cannot find a primary RPI Camera stream to associate with the secondary stream")
                self.source = new
                self.calls.append((name, body["source"]))
                return {"status": "ok"}
            on = self.source[name] == "rpiCamera"
            return {"ready": on, "source": {"type": "rpiCameraSource"} if on else None}
        for p in (patch.object(app, "STATE_DIR", self.state.name), patch.object(app, "_mtx", mtx),
                  patch.object(app, "PRIVACY_RESUME_WAIT", 1)):
            p.start()
            self.addCleanup(p.stop)

    def test_pause_then_resume(self):
        app.set_privacy(True)
        self.assertEqual(self.calls, [("sub", "publisher"), ("cam", "publisher")])
        self.assertTrue(app.privacy_requested())
        self.assertFalse(app._mtx_capture_running())
        app.set_privacy(False)
        self.assertEqual(self.calls[2:], [("cam", "rpiCamera"), ("sub", "rpiCamera")])
        self.assertFalse(app.privacy_requested())
        self.assertTrue(app._mtx_capture_running())

    def test_refresh_repauses_after_a_restart(self):
        app.set_privacy(True)
        self.source = {"cam": "rpiCamera", "sub": "rpiCamera"}     # MediaMTX restarted and reopened the camera
        app.ensure_privacy_paused()
        self.assertFalse(app._mtx_capture_running())


class RpiWatcher(unittest.TestCase):
    def setUp(self):
        self.state = tempfile.TemporaryDirectory()
        self.addCleanup(self.state.cleanup)
        for p in (patch.object(watch, "RESTART_CAMERA", f"{self.state.name}/restart_camera"),
                  patch.object(watch, "SETTINGS", f"{self.state.name}/settings.json"),
                  patch.object(watch, "STATE", self.state.name),
                  patch.object(watch, "privacy_flag", lambda: False),
                  patch("urllib.request.urlopen", no_network)):
            p.start()
            self.addCleanup(p.stop)

    def test_frames_come_from_the_substream_only(self):
        fs = watch.FrameSource.__new__(watch.FrameSource)
        fs.want_until, fs.fast_until, fs.push_down_until = 0.0, 0.0, 0.0
        self.assertIsNone(fs.desired_mode())
        fs.want_until = watch.now() + 60
        self.assertEqual(fs.desired_mode(), "sub")      # never push, idle or main (those are Thingino-only)
        with patch.object(watch, "privacy_flag", lambda: True):
            self.assertIsNone(fs.desired_mode())

    def test_restart_means_restart_mediamtx(self):
        self.assertTrue(watch.cam_action("/x/restart-prudynt.cgi"))     # no HTTP: urlopen is patched to fail
        self.assertTrue(os.path.exists(watch.RESTART_CAMERA))
        self.assertFalse(watch.cam_action("/x/reboot.cgi"))

    def test_follow_and_find_are_off(self):
        w = watch.Watch.__new__(watch.Watch)
        Path(watch.SETTINGS).write_text('{"follow": true}')
        self.assertFalse(w.setting("follow", False))
        Path(f"{self.state.name}/find_request").write_text(str(watch.now()))
        w.last_find_req = 0.0
        self.assertFalse(w.find_requested(watch.now()))

    def test_imaging_is_empty(self):
        self.assertEqual(watch.imaging_state(), {})


if __name__ == "__main__":
    unittest.main()
