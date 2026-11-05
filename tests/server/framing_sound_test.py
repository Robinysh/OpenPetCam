"""Bounded framing and household-sound hints, using the real Follow state machine."""
import inspect
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "server"))
sys.path.insert(0, HERE)
import tracker as T  # noqa: E402
from tracker_test import World  # noqa: E402


def make(w, memory=None):
    return T.Tracker(w, w.find_pet, lambda: w.t, log=lambda m: None, sleep=w.sleep, memory=memory)


def test_framing():
    w = World()
    tr = make(w)
    cut = [0.1, 0.0, 0.9, 0.99]  # body centred, upper edge cut: the report's missed-head geometry
    w.advance(1)
    tr.step(cut, True, box_t=w.t)
    assert not w.moves, "one box is not enough evidence for framing"
    stamp = w.t
    w.advance(1)
    tr.step(cut, True, box_t=stamp)
    assert not w.moves, "cached detection cannot confirm framing"
    w.advance(1)
    tr.step(cut, True, box_t=w.t)
    assert w.Y > 500, "two fresh moving clipped boxes should tilt upward to reveal the head"
    for _ in range(8):
        w.advance(T.MOVE_EVERY + T.POST_MOVE_IGNORE)
        tr.step(cut, True, box_t=w.t)
    assert 1 <= len(w.moves) <= 2, "a persistent clipped box cannot drive an unbounded tilt"
    for moving, weak, manual, top_stop in ((False, False, False, False), (True, True, False, False),
                                          (True, False, True, False), (True, False, False, True)):
        q = World(Y=1000 if top_stop else 500)
        f = make(q)
        for _ in range(3):
            q.advance(3)
            if manual:
                q.moved = q.t
            f.step(None if weak else cut, moving, box_t=q.t, weak=[cut] if weak else None)
        assert not q.moves, (moving, weak, manual, top_stop, q.moves)


def activity_step(tr, enabled=True):
    kwargs = {"activity_sound": enabled} if "activity_sound" in inspect.signature(tr.step).parameters else {}
    return tr.step(None, False, **kwargs)


def test_sound():
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "memory.json")
        w = World(pet=-10000)
        w.t = 10000
        tr = make(w, path)
        tr.last_seen_t = w.t - 1000
        tr.last_search = w.t - T.SEARCH_EVERY - 1
        tr.failed_searches = T.GIVE_UP_AFTER
        assert "household sound" in (activity_step(tr) or ""), "one return-home sound can re-arm the exhausted search"
        after = len(w.moves)
        w.advance(T.SEARCH_EVERY + 1)
        activity_step(tr)
        assert len(w.moves) == after, "continuing noise cannot re-arm another search in the same absence"
        f = make(w, path)
        f.last_seen_t = w.t - 1000
        activity_step(f)
        assert len(w.moves) == after, "restart does not renew the one household hint per absence"
        # A real sighting starts another episode; the existing 600-second cooldown still applies.
        f.step([0.4, 0.4, 0.6, 0.6], False, box_t=w.t)
        f.last_seen_t = w.t - 1000
        f.last_search = w.t - 100
        activity_step(f)
        assert len(w.moves) == after
        w.advance(T.SEARCH_EVERY + 1)
        activity_step(f)
        assert len(w.moves) > after


if len(sys.argv) == 1 or "framing" in sys.argv:
    test_framing()
if len(sys.argv) == 1 or "sound" in sys.argv:
    test_sound()
print("framing + household sound: ok")
