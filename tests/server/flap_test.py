"""Day/night flapping locks night briefly, and longer only if it flaps again right after (report 2026-11-04 08:07:
"auto light is not working": a dawn flap at 06:13 held a 2 h night lock, so the picture stayed infrared in daylight).
  python3 tests/server/flap_test.py
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "server"))
import watch  # noqa: E402

modes, events = [], []
watch.camapi.set_imaging = lambda m: modes.append(m)
watch.add_event = lambda con, kind, detail=None, thumb=None, ts=None: events.append((kind, detail))
watch.FLAP_LOCK_FILE = os.path.join(tempfile.mkdtemp(), "lock")
w = watch.Watch.__new__(watch.Watch)
w.flap_lock_until, w.mode_changes, w.last_mode = 0.0, [], None


def flap(t0):
    """Four day/night switches in two minutes, as the camera's auto mode does at dawn or facing a close wall."""
    t = t0
    for m in ("day", "night", "day", "night", "day"):
        w.imaging = {"daynight_mode": m, "daynight_enabled": 1}
        w.check_flapping(None, t)
        t += 30
    return t


def tick(t):
    w.imaging = {"daynight_mode": "night", "daynight_enabled": 1}
    w.check_flapping(None, t)


t = flap(0)
assert modes[-1] == "night" and events[-1][1]["locked_s"] == watch.FLAP_LOCK == 1200, (modes, events)
tick(t + 1300)
assert modes[-1] == "auto", "the first lock ends after 20 min, not 2 h"
t = flap(t + 1400)                                   # flaps again right away: a real problem, lock longer
assert events[-1][1]["locked_s"] == 2400, events[-1]
tick(t + 2500)
t = flap(t + 2600)
assert events[-1][1]["locked_s"] == 4800, events[-1]
tick(t + 4900)
t = flap(t + 5000)
assert events[-1][1]["locked_s"] == watch.FLAP_LOCK_MAX, "capped at 2 h"
tick(t + 7300)
t = flap(t + 7300 + watch.FLAP_RELAPSE + 60)         # quiet for a while: back to the short lock
assert events[-1][1]["locked_s"] == watch.FLAP_LOCK, events[-1]
print("flap lock ok: 20 min, doubling on relapse to 2 h, short again after a quiet spell")
