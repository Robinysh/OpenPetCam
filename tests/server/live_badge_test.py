"""The LIVE badge and the pet pill must agree with "Camera offline".

Report 20261105-212631: the camera was offline (watcher status "offline", WebRTC connected to the relay
but the video element had readyState 0, no picture), yet the badge still said LIVE, and the pill said
"out of sight", which reads as "she left" when really the camera could not see anything.

Runs badgeState() and petText() from server/index.html in node. Usage: python3 tests/server/live_badge_test.py
"""
import json
from pathlib import Path
import re
import shutil
import subprocess
import time
import unittest

HTML = Path(__file__).resolve().parents[2] / "server/index.html"


def extract(src, name):
    m = re.search(r"^function " + name + r"\(.*?^}\n", src, re.S | re.M)
    assert m, f"{name} not found in index.html"
    return m.group(0)


def run(expr):
    src = HTML.read_text()
    js = "\n".join(['const PET = {"name": "Biscuit"};', 'const fmtTime = () => "12:00";',
                    *(extract(src, n) for n in ["fmtDur", "ago", "petText", "badgeState"]),
                    f"console.log(JSON.stringify({expr}));"])
    out = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


@unittest.skipUnless(shutil.which("node"), "node not installed")
class LiveBadgeTest(unittest.TestCase):
    def badge(self, live, video=None):
        return run(f"badgeState({json.dumps(live)}, {json.dumps(video)})")

    def test_offline_camera_is_not_live(self):
        text, down = self.badge({"status": "offline", "present": True}, {"readyState": 0})
        self.assertNotEqual(text, "LIVE"); self.assertTrue(down)
        # Even if the phone still holds an old picture, the server says the camera is gone.
        self.assertNotEqual(self.badge({"status": "offline"}, {"readyState": 4})[0], "LIVE")
        self.assertNotEqual(self.badge({"status": "frozen"}, {"readyState": 4})[0], "LIVE")
        self.assertNotEqual(self.badge(None)[0], "LIVE")

    def test_connected_without_picture_is_not_live(self):
        self.assertEqual(self.badge({"status": "ok"}, {"readyState": 0})[0], "CONNECTING")

    def test_healthy_playing_stream_is_live(self):
        self.assertEqual(self.badge({"status": "ok"}, {"readyState": 4}), ["LIVE", False])
        # The video can play while the watcher is slow: the picture really is live.
        self.assertEqual(self.badge({"status": "watcher_down"}, {"readyState": 4}), ["LIVE", False])

    def test_pill_does_not_say_out_of_sight_when_camera_offline(self):
        seen = time.time() - 300
        text = run(f"petText({json.dumps({'status': 'offline', 'present': True, 'last_seen': seen})})")[0]
        self.assertNotIn("out of sight", text)
        self.assertIn("last seen 5 min ago", text)
        # A working camera that lost her still says out of sight.
        text = run(f"petText({json.dumps({'status': 'ok', 'present': True, 'last_seen': seen})})")[0]
        self.assertIn("out of sight", text)


if __name__ == "__main__":
    unittest.main()
