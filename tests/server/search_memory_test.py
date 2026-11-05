"""Empty-search budget and cooldown survive watcher restarts, without storing visual tracks."""
import inspect
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "server"))
sys.path.insert(0, HERE)
import tracker as T  # noqa: E402
from tracker_test import World  # noqa: E402


def make(w, path):
    # Exercise the old behavior too: the baseline fails for a new sweep, not a missing argument.
    kwargs = {"memory": path} if "memory" in inspect.signature(T.Tracker).parameters else {}
    return T.Tracker(w, w.find_pet, lambda: w.t, log=lambda m: None, sleep=w.sleep,
                     find_now=w.find_now, **kwargs)


with tempfile.TemporaryDirectory() as d:
    path = os.path.join(d, "search_memory.json")
    w = World(pet=-10000)
    w.t = 10000.0
    tr = make(w, path)
    for _ in range(T.GIVE_UP_AFTER):
        w.advance(T.SEARCH_EVERY + 1)
        assert "not found" in tr.step(None, False)
    assert tr.failed_searches == T.GIVE_UP_AFTER
    before = len(w.moves)
    w.advance(1)
    restarted = make(w, path)
    for _ in range(int(T.SEARCH_AFTER + 20)):
        w.advance(1)
        restarted.step(None, False)
    assert len(w.moves) == before, "restart must not spend a new empty-room search budget"
    assert restarted.failed_searches == T.GIVE_UP_AFTER
    # Restart carries only search limits: no old box, velocity, or camera-move pause.
    assert restarted.last_seen_c is None and restarted.last_vel == (0.0, 0.0)
    # A dog sound can still re-arm within the existing sound cooldown, and Find remains explicit.
    w.advance(T.SOUND_SEARCH_EVERY + 1)
    assert "not found" in restarted.step(None, False, sound=True)
    assert "not found" in restarted.search(reason="you asked")
    assert make(w, path).last_search == restarted.last_search, "manual Find also persists its cooldown"
    # A real sighting clears the empty-search count and persists that reset.
    w.advance(T.POST_MOVE_IGNORE + 1)
    restarted.step([0.4, 0.4, 0.6, 0.6], False, box_t=w.t)
    assert make(w, path).failed_searches == 0
    for bad in ({"last_search": w.t + 1, "failed_searches": 2},
                {"last_search": w.t - 3601, "failed_searches": 2},
                {"last_search": float("nan"), "failed_searches": 2},
                {"last_search": w.t, "failed_searches": -1},
                {"last_search": w.t, "failed_searches": "oops"}, []):
        with open(path, "w") as f:
            json.dump(bad, f)
        assert make(w, path).failed_searches == 0, bad
    with open(path, "w") as f:
        f.write("broken json")
    assert make(w, path).failed_searches == 0
print("search memory: ok")
