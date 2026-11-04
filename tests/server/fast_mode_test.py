"""Fast mode (main-stream frames while Follow is engaged) and its CPU governor.
  python3 tests/server/fast_mode_test.py"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "server"))
sys.path.insert(0, "/opt/petcam")
import watch  # noqa: E402

watch.FAST_ENABLED = True

clock = [1000.0]
watch.now = lambda: clock[0]
logs = []
watch.log = lambda *a: logs.append(" ".join(map(str, a)))

stat = tempfile.NamedTemporaryFile("w", delete=False)
cpu = {"total": 0, "steal": 0}


def write():
    with open(stat.name, "w") as f:   # user nice system idle iowait irq softirq steal
        f.write(f"cpu  {cpu['total'] - cpu['steal']} 0 0 0 0 0 0 {cpu['steal']} 0 0\n")


def tick(secs, steal_share):
    for _ in range(int(secs // 10)):
        clock[0] += 10
        cpu["total"] += 1000
        cpu["steal"] += int(1000 * steal_share)
        write()
        ok = g.ok()
    return ok


write()
g = watch.StealGovernor(stat.name)
assert tick(120, 0.0), "no steal: fast mode allowed"
assert not tick(70, 0.30), "steal over the limit: fast mode off"
assert any("fast mode off" in m for m in logs), logs
assert not tick(300, 0.0), "stays off for the cooldown even once steal is gone"
assert tick(400, 0.0), "back on after the cooldown"
assert watch.StealGovernor("/nonexistent").ok(), "an unreadable /proc/stat never blocks fast mode"

# The frame source picks the stream: main only while fast mode is wanted and the governor allows it.
f = watch.FrameSource()
f.governor = type("G", (), {"ok": lambda self: True})()
assert f.desired_mode() == "push", "live frames come from the camera's JPEG push by default"
f.push_down_until = clock[0] + 600
assert f.desired_mode() == "sub", "a failed push falls back to the substream"
f.push_down_until = 0
f.want_fast(20)
assert f.desired_mode() == "main" and f.wanted()
clock[0] += 21
assert f.desired_mode() == "push", "fast mode expires"
assert f.wanted(), "the decoder itself keeps running (STREAM_HOLD) on the substream"
f.want_fast(20)
f.governor = type("G", (), {"ok": lambda self: False})()
assert f.desired_mode() == "push", "the governor wins"
assert watch.STREAM_LAG["main"] < watch.STREAM_LAG["sub"]
watch.FAST_ENABLED = False
g2 = watch.FrameSource()
g2.governor = type("G", (), {"ok": lambda self: True})()
g2.want_fast(20)
assert g2.desired_mode() == "push" and g2.wanted(), "disabled: live frames as before, never the main stream"
os.unlink(stat.name)

# The push parser: header line + JPEG bytes; noise lines skipped; a torn JPEG skipped; end of stream -> None.
import io  # noqa: E402
from PIL import Image  # noqa: E402
buf = io.BytesIO()
Image.new("RGB", (640, 360), (200, 10, 10)).save(buf, "JPEG")
j = buf.getvalue()
small = io.BytesIO()
Image.new("RGB", (320, 180), (10, 200, 10)).save(small, "JPEG")
k = small.getvalue()
data = b"motd noise\n" + b"J %d\n" % len(j) + j + b"J 5\nxxxxx" + b"J %d\n" % len(k) + k + b"J 999\nshort"
proc = type("P", (), {"stdout": io.BytesIO(data)})()
fs = watch.FrameSource()
a = fs.read_frame(proc, "push", 0)
assert a.shape == (360, 640, 3) and a[180, 320, 0] > 150, "a pushed JPEG decodes to the frame size"
assert fs.read_frame(proc, "push", 0).size == 0, "a torn JPEG is skipped, not fatal"
b = fs.read_frame(proc, "push", 0)
assert b.shape == (360, 640, 3) and b[90, 160, 1] > 150, "another size is scaled to the frame size"
assert fs.read_frame(proc, "push", 0) is None, "a truncated frame ends the source"
for bad in (b"J abc\n", b"J\n", b"J -5\n", b"J 99999999\n" + b"x" * 10):
    pb = type("P", (), {"stdout": io.BytesIO(bad)})()
    assert fs.read_frame(pb, "push", 0) is None, ("broken framing must end the session, not crash", bad[:12])
cmd = " ".join(fs.push_cmd())
assert "/tmp/.push.pid" in cmd and "|| break" in cmd and str(watch.PUSH_MAX_S) in cmd, "one loop, stops on a failed write, bounded"
print("fast mode: ok")
