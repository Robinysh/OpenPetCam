"""A motion chase must not run the camera into a wall and leave it there (owner report 2026-11-05 14:56: three turns
from her resting spot ended at pan 3666, facing the wall beside the mount, until a search minutes later)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "server"))
sys.path.insert(0, HERE)
import tracker as T  # noqa: E402
from tracker_test import World, XM  # noqa: E402


class NoFloorAbove:
    """A room map whose poses beyond pan x0 show no floor (the wall next to the mount)."""
    def __init__(self, x0): self.x0 = x0
    def has_floor(self, x, y): return x < self.x0
    def floor_share(self, x, y): return 0.3 if x < self.x0 else 0.0
    def stops(self, hour): return []
    def places(self, hour): return []
    def tiles(self, x, y): return []


def tracker(w):
    tr = T.Tracker(w, w.find_pet, clock=lambda: w.t, log=lambda m: None, sleep=w.sleep, find_now=w.find_now)
    tr.last_seen_t = w.t - 30                              # lost a while ago, nothing in view now
    tr.failed_searches, tr.last_search = T.GIVE_UP_AFTER, w.t  # timed searches given up: only the chase acts
    return tr


def chase(w, tr, mx):
    """Motion at the edge (mx), then the cycles that follow: returns the actions taken."""
    acts = []
    for k in range(12):
        w.advance(1.0)
        motion = (w.t - 0.2, mx, 0.0) if k == 0 else None
        a = tr.step(None, moving=False, motion=motion)
        if a:
            acts.append(a)
        w.wait_idle()
    return acts


def test_no_floor_stops_the_chase_and_returns():
    T.ROOM_MAP = NoFloorAbove(3000)
    try:
        w = World(X=1816, pet=900)                         # she is far to the other side; motion at the left edge
        tr = tracker(w)
        acts = chase(w, tr, -0.41)                         # -fx turns toward higher pan (toward the wall)
        assert any(a.startswith("chase: motion at the edge") for a in acts), acts
        assert max(x for k, x in w.moves) < 3000, w.moves  # never drove onto the floorless poses
        assert any("back to where it was looking (1816" in a for a in acts), acts
        assert abs(w.X - 1816) < 2, w.X
    finally:
        T.ROOM_MAP = None


def test_end_stop_without_a_map_returns():
    T.ROOM_MAP = None
    w = World(X=2900, pet=600)
    tr = tracker(w)
    acts = chase(w, tr, -0.41)
    assert any("back to where it was looking (2900" in a for a in acts), acts
    assert abs(w.X - 2900) < 2 and w.X < XM, w.X           # not left at the end stop


def test_seeing_her_cancels_the_return():
    T.ROOM_MAP = None
    w = World(X=1816, pet=1816 + 733 * 0.75)               # she is exactly where the first turn looks
    tr = tracker(w)
    w.advance(1.0)
    tr.step(None, moving=False, motion=(w.t - 0.2, -0.41, 0.0))
    w.wait_idle()
    for _ in range(4):
        w.advance(1.0)
        tr.step(w.box(), moving=True)
    assert tr.chase_origin is None and tr.chase_dir is None


if __name__ == "__main__":
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            fn()
            print("ok", n)
