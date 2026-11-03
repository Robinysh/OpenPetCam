"""Stream-stall watchdog with a simulated outage: live video gone, snapshots fine (2026-11-02 10:28).
  python3 tests/server/stall_test.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "server"))
sys.path.insert(0, "/opt/petcam")
import watch  # noqa: E402

clock = [1000.0]
watch.now = lambda: clock[0]
actions, events = [], []
watch.cam_action = lambda path: actions.append((clock[0], path)) or True
watch.add_event = lambda con, kind, detail=None, thumb=None, ts=None: events.append((kind, detail))
watch.db = lambda: type("C", (), {"close": lambda self: None})()

w = watch.Watch.__new__(watch.Watch)
w.last_fix, w.stream_down_since = 0.0, 1000.0
for _ in range(20):                       # 60 s down: nothing yet
    clock[0] += 3; w.check_stream_stall()
assert not actions, actions
for _ in range(12):                       # past 90 s: one restart
    clock[0] += 3; w.check_stream_stall()
assert len(actions) == 1 and actions[0][1] == "/x/restart-prudynt.cgi", actions
for _ in range(100):                      # still down for 5 more min: rate limit holds
    clock[0] += 3; w.check_stream_stall()
assert len(actions) == 1, actions
for _ in range(120):                      # after 10 min: tries again
    clock[0] += 3; w.check_stream_stall()
assert len(actions) == 2, actions
print("stall watchdog ok:", [round(t - 1000) for t, _ in actions], events[0])

# Sound lost while the mic is on (2026-11-02 18:30: after pause/resume the audio track stayed empty for 20 min).
actions.clear()
# just started: no audio reading yet (last_ok 0) must not count as 1.8 billion seconds of silence
w.audio = type("A", (), {"last_ok": 0.0})()
w.last_audio_fix, w.audio_watch_from = 0.0, clock[0]
w.check_audio_lost(False, {"mic_enabled": 1})
assert not actions, ("restarted on startup", actions)
w.audio.last_ok = clock[0]
for _ in range(20):                       # 60 s without sound: nothing yet
    clock[0] += 3; w.check_audio_lost(False, {"mic_enabled": 1})
assert not actions, actions
for _ in range(12):                       # past 90 s: one restart
    clock[0] += 3; w.check_audio_lost(False, {"mic_enabled": 1})
assert len(actions) == 1, actions
actions.clear(); w.last_audio_fix = 0.0
for _ in range(60):                       # paused, or the mic switched off on purpose: never restarts
    clock[0] += 3; w.check_audio_lost(True, {"mic_enabled": 0}); w.check_audio_lost(False, {"mic_enabled": 0})
assert not actions, actions
print("sound watchdog ok: restarts the streamer after 90 s of silence with the mic on, never while paused")
