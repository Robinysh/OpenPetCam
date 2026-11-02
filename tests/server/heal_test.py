"""Streamer self-heal decisions with a simulated outage (camera actions and clock mocked)."""
import sys
sys.path.insert(0, "/opt/petcam")
import watch
calls = []
watch.cam_action = lambda p: calls.append(p) or True
watch.add_event = lambda con, kind, detail=None, thumb=None, ts=None: None
clock = [1_000_000.0]
watch.now = lambda: clock[0]
w = watch.Watch.__new__(watch.Watch)
w.last_ok, w.last_fix = clock[0], 0.0
for step in range(0, 3000, 30):          # 50 minutes of failed snapshots, camera web reachable
    clock[0] = 1_000_000.0 + step
    before = len(calls)
    w.heal_streamer()
    if len(calls) > before:
        print(f"t+{step:4d}s -> {calls[-1]}")
print("total actions in 50 min:", len(calls))
