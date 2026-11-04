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
assert f.desired_mode() == "sub"
f.want_fast(20)
assert f.desired_mode() == "main" and f.wanted()
clock[0] += 21
assert f.desired_mode() == "sub", "fast mode expires"
assert f.wanted(), "the decoder itself keeps running (STREAM_HOLD) on the substream"
f.want_fast(20)
f.governor = type("G", (), {"ok": lambda self: False})()
assert f.desired_mode() == "sub", "the governor wins"
assert watch.STREAM_LAG["main"] < watch.STREAM_LAG["sub"]
watch.FAST_ENABLED = False
g2 = watch.FrameSource()
g2.governor = type("G", (), {"ok": lambda self: True})()
g2.want_fast(20)
assert g2.desired_mode() == "sub" and g2.wanted(), "disabled: live frames as before, never the main stream"
os.unlink(stat.name)
print("fast mode: ok")
