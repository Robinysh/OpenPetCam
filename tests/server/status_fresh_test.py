"""The app's status (live.json) follows the idle frames: every new frame is published, nothing is published without one,
and each status says how old its camera picture and detector result are.
  python3 tests/server/status_fresh_test.py"""
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "server"))
sys.path.insert(0, "/opt/petcam")
import watch  # noqa: E402

clock = [1000.0]
watch.now = lambda: clock[0]
watch.log = lambda *a: None
watch.privacy_flag = lambda: False
writes = []
watch.write_json = lambda path, obj: writes.append(dict(obj))


def fresh_watch():
    w = watch.Watch.__new__(watch.Watch)
    w.live_last = {"t": clock[0], "status": "ok", "present": True, "det_t": clock[0] - 7.0}
    w.live_written = clock[0]
    return w


# 1. Idle frames 0.5 s apart (IDLE_FPS 2) with camera/decoder jitter: every one of them reaches the app.
rng = random.Random(7)
w = fresh_watch()
writes.clear()
frames = []
ft = clock[0]
for _ in range(400):
    ft += 1 / watch.IDLE_FPS + rng.uniform(-0.03, 0.03)
    clock[0] = ft
    frames.append(ft)
    w.publish_tick(ft, 0.4)                          # IR grain: under the detection threshold
assert len(writes) == len(frames), f"{len(writes)} of {len(frames)} idle frames published; each must be"
assert [x["frame_t"] for x in writes] == frames

# 2. No publish without a new frame: the same frame offered again (any time later) does not refresh the timestamp.
n = len(writes)
clock[0] += 0.45
w.publish_tick(frames[-1], 0.4)
clock[0] += 0.6
w.publish_tick(frames[-1], 9.0)
assert len(writes) == n, "a status was rewritten without a new frame"

# 3. Each status carries the age of its camera picture (arrival plus the stream's lag) and of its detector result.
w = fresh_watch()
w.frames = type("F", (), {"lag": watch.STREAM_LAG["idle"]})()
writes.clear()
clock[0] += 1.0
arrived = clock[0]
clock[0] += 0.05                                     # published a little after arrival
w.publish_tick(arrived, 0.4)
x = writes[-1]
assert abs(x["frame_age"] - (0.05 + watch.STREAM_LAG["idle"])) < 0.011, x
assert abs(x["det_age"] - (clock[0] - w.live_last["det_t"])) < 0.011, x
assert x["t"] == clock[0] and x["det_t"] == w.live_last["det_t"], "the detector's own time is kept, never refreshed"

# 4. A cycle's own publish carries the detector age too, and none when there is no detector result yet.
w = fresh_watch()
for k, v in dict(present=False, last_seen=None, gone_since=None, rest_since=None, moving_since=None, imaging=None,
                 view_blocked=False, find_result=None, last_bark=None).items():
    setattr(w, k, v)
w.audio = type("A", (), {"last_ok": clock[0]})()
writes.clear()
w.publish(status="ok", det_t=clock[0] - 3.0)
assert abs(writes[-1]["det_age"] - 3.0) < 0.011, writes[-1]
w.publish(status="privacy", det_t=None, frame_t=None)
assert writes[-1]["det_age"] is None and writes[-1]["frame_age"] is None, writes[-1]
print("status freshness: ok")
