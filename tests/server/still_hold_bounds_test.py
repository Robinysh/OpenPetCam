"""The still hold, end to end through the real presence state machine, and its bounds.

A still, resting dog the detector misses for several cycles stays present, and the app says so honestly ("last seen by
the detector N min ago", never a fresh sighting). Motion in her patch, a privacy pause or resume, or the settings flag
off ends it at once. held_since rides on the published status and never causes a write of its own: publish_tick still
writes only for a new frame.
  python3 tests/server/still_hold_bounds_test.py
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
import types

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "server"))
import watch  # noqa: E402

watch.log = lambda *a: None
tmp = tempfile.mkdtemp()
watch.SETTINGS = os.path.join(tmp, "settings.json")
watch.HEARTBEAT = os.path.join(tmp, "heartbeat")
watch.LIVE = os.path.join(tmp, "live.json")
watch.add_event = lambda *a, **k: None

rng = np.random.default_rng(1)
room = rng.integers(40, 200, size=(360, 640), dtype=np.uint8)
dog = room.copy()
dog[120:200, 280:420] = 170 + rng.integers(0, 25, size=(80, 140), dtype=np.uint8)
box = [280 / 640, 120 / 360, 420 / 640, 200 / 360]


def img(a, noise=2):
    return Image.fromarray(np.clip(a.astype(int) + rng.integers(-noise, noise + 1, a.shape), 0, 255).astype(np.uint8))


def make():
    w = watch.Watch.__new__(watch.Watch)
    w.pose = (1200, 500)
    w.camera_moved_near = lambda t, before=2.0, after=0: False
    w.present, w.last_seen, w.gone_since = False, 0.0, None
    w.present_since = w.rest_since = w.moving_since = None
    w.clips = types.SimpleNamespace(want=lambda *a: None)
    w.save_thumb = lambda *a: None
    w.imaging, w.view_blocked, w.find_result, w.last_bark = None, False, None, None
    w.audio = types.SimpleNamespace(last_ok=0.0, pause=lambda: None)
    w.frames = types.SimpleNamespace(pause=lambda: None)
    return w


def step(w, t, picture, detected):
    """One watcher cycle's presence part, exactly as Watch.cycle does it."""
    seen = {"box": box} if detected else None
    score = watch.PRESENT_SCORE if detected else 0.0
    held = w.still_hold(t, picture, seen, False)
    if held is not None and score < watch.PRESENT_SCORE:
        w.update_pet(None, t, watch.PRESENT_SCORE, False, b"", held)
    else:
        w.update_pet(None, t, score, False, b"", box if detected else None)
    w.publish(status="ok")
    with open(watch.LIVE) as f:
        return json.load(f)


# 1) Seen once, then missed by the detector for 20 minutes while she lies still: present the whole time, and the
#    published status says when the detector last saw her (not "now").
w = make()
step(w, 0.0, img(dog), True)
for t in range(12, 1200, 12):
    live = step(w, float(t), img(dog), False)
    assert live["present"], f"lost a still, missed dog at {t} s"
assert live["held_since"] == 0.0, live
print("still dog missed for 20 min: present, held_since = last detection")

# 1b) Frame ticks between cycles keep held_since next to frame_age/det_age, and a repeated frame writes nothing.
saved_now = watch.now
writes = []
saved_write = watch.write_json
watch.write_json = lambda path, obj: (writes.append(obj), saved_write(path, obj))
try:
    w.privacy_on = lambda: False
    w.live_written = 0.0                      # publish() stamped the wall clock; this test runs on its own clock
    watch.now = lambda: 1190.0
    w.publish_tick(1189.9, 0.0)
    assert len(writes) == 1 and writes[-1]["held_since"] == 0.0, writes
    assert "frame_age" in writes[-1] and "det_age" in writes[-1], writes[-1]
    watch.now = lambda: 1191.0
    w.publish_tick(1189.9, 0.0)               # same frame again: no write, held_since or not
    assert len(writes) == 1, "a repeated frame was published"
finally:
    watch.now, watch.write_json = saved_now, saved_write
print("frame ticks carry held_since; a repeated frame is not published")

# 2) Motion in her patch (she got up) ends it at once; after LEAVE_AFTER she has left.
live = step(w, 1212.0, img(room), False)
assert live["held_since"] is None
for t in range(1224, 1224 + int(watch.LEAVE_AFTER) + 30, 12):
    live = step(w, float(t), img(room), False)
assert not live["present"], "the patch changed but she stayed present"
print("change in her patch ends the hold")

# 3) A privacy pause ends it, and so does resume: after privacy the same unchanged picture must not bring her back
#    without a new sighting.
w = make()
step(w, 0.0, img(dog), True)
assert step(w, 12.0, img(dog), False)["held_since"] == 0.0
w.last_privacy_check = float("inf")          # no camera call from the test
w.privacy_cycle()
assert getattr(w, "still_ref", None) is None, "privacy pause kept the still hold"
w = make()
step(w, 0.0, img(dog), True)
step(w, 12.0, img(dog), False)
saved = watch.Tracker, watch.CamAdapter
watch.Tracker = lambda *a, **k: types.SimpleNamespace()
watch.CamAdapter = lambda: None
w.frames = types.SimpleNamespace(pause=lambda: None, motion_track=None, wake_decoder=types.SimpleNamespace(set=lambda: None))
w.resume_after_privacy()
watch.Tracker, watch.CamAdapter = saved
assert getattr(w, "still_ref", None) is None, "privacy resume kept the still hold"
assert w.still_hold(20.0, img(dog), None, False) is None
print("privacy pause and resume end the hold")

# 4) The settings flag turns it off.
with open(watch.SETTINGS, "w") as f:
    json.dump({"still_hold": False}, f)
w = make()
step(w, 0.0, img(dog), True)
assert w.still_hold(12.0, img(dog), None, False) is None, "still_hold: false in settings did not turn it off"
os.remove(watch.SETTINGS)
print("settings flag off: no hold")

# 5) The app's words: held -> "resting here, last seen by the detector ...", never "resting 5m" / "is here".
if shutil.which("node"):
    import re
    src = open(os.path.join(HERE, "..", "..", "server", "index.html")).read()

    def extract(name):
        m = re.search(r"^function " + name + r"\(.*?^}\n", src, re.S | re.M)
        assert m, name
        return m.group(0)
    now = 10_000.0
    live = {"status": "ok", "present": True, "last_seen": now, "rest_since": now - 900, "held_since": now - 600}
    js = "\n".join(['const PET = {"name": "Biscuit"};', 'const fmtTime = () => "12:00";',
                    f"Date.now = () => {now * 1000};", extract("fmtDur"), extract("ago"), extract("petText"),
                    f"console.log(JSON.stringify(petText({json.dumps(live)})));"])
    out = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    text = json.loads(out.stdout)[0]
    assert "last seen by the detector 10 min ago" in text, text
    print("app says:", text)
else:
    print("node not found: app text not checked")
shutil.rmtree(tmp, ignore_errors=True)
print("still hold bounds ok")
