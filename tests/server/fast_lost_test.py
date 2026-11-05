"""Follow when the detector has lost her and only the motion estimate is left (tracker._fast_box).
The estimate is a stand-in with fasttrack.FastTrack's anchor/shift/drop/estimate, so this runs without numpy:
  python3 tests/server/fast_lost_test.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "server"))
sys.path.insert(0, HERE)
import tracker as T  # noqa: E402
from tracker_test import World  # noqa: E402


class Estimate:
    """fasttrack.FastTrack without its motion image: motion(t, x, y) is a frame whose changed pixels moved the estimate."""
    MAX_AGE = 12.0

    def __init__(self):
        self.c = self.t_anchor = self.t_update = None
        self.size, self.n = (0.1, 0.1), 0

    def anchor(self, t, box):
        self.c = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
        self.size = (box[2] - box[0], box[3] - box[1])
        self.t_anchor, self.t_update, self.n = t, t, 0

    def motion(self, t, x, y):
        self.c, self.t_update = (x, y), t
        self.n += 1

    def shift(self, fx, fy):
        self.c = (min(1.0, max(0.0, self.c[0] - fx)), min(1.0, max(0.0, self.c[1] - fy)))

    def drop(self):
        self.c = self.t_anchor = self.t_update = None

    def estimate(self, now):
        if self.c is None or now - self.t_anchor > self.MAX_AGE:
            return None
        return (self.c[0] - 0.5, self.c[1] - 0.5, self.n, now - self.t_anchor, self.t_update, *self.size)


def lost_after_a_fast_aim(motion_goes_on):
    """A big close-up box drifting left, then the detector loses her and one burst of motion on the left sends a fast
    aim. Afterwards the picture is still (or, with motion_goes_on, she keeps moving in it). Returns what Follow logged
    after the aim, by seconds since it, and the tracker."""
    w = World(X=2600, pet=2600)
    tr = T.Tracker(w, w.find_pet, clock=lambda: w.t, log=lambda m: None, sleep=w.sleep, find_now=w.find_now)
    est = tr.fast = Estimate()
    was = T.FAST_FOLLOW
    T.FAST_FOLLOW = True
    try:
        for cx in (0.10, 0.03, -0.03):                # inside EDGE: no aim yet
            w.advance(1.0)
            box = [cx + 0.5 - 0.33, 0.11, cx + 0.5 + 0.33, 0.99]
            est.anchor(w.t, box)
            assert tr.step(box, moving=True, box_t=w.t, fast=est.estimate(w.t)) is None
        w.advance(1.0)
        est.motion(w.t, 0.5 - 0.28, 0.57)             # the motion that moves the estimate
        aim = tr.step(None, moving=False, fast=est.estimate(w.t))
        assert aim and aim.startswith("track (fast): aim (-"), aim
        aimed_at = w.t
        after = []
        for k in range(1, 13):
            w.advance(1.0)
            if motion_goes_on and est.c is not None:
                est.motion(w.t, est.c[0] - 0.02, est.c[1])
            act = tr.step(None, moving=False, fast=est.estimate(w.t))
            if act:
                after.append((k, act))
        return after, tr, aimed_at
    finally:
        T.FAST_FOLLOW = was


def test_frozen_estimate_does_not_cancel_the_chase():
    # Owner report 2026-11-05 17:47: the estimate stopped moving after the aim but kept standing in for the missing box
    # for FAST_LOST_MAX, as if she were seen. Her speed decayed to zero meanwhile, so the chase never fired and the
    # camera stayed on an empty floor.
    after, tr, aimed_at = lost_after_a_fast_aim(motion_goes_on=False)
    chases = [(k, a) for k, a in after if a.startswith("chase: she left the frame heading (-")]
    assert chases and chases[0][0] <= int(T.CHASE_WINDOW[1]), after


def test_frozen_estimate_is_not_a_sighting():
    # With no new motion the "seen" clock stops at the aim instead of running on.
    w = World(X=2600, pet=2600)
    tr = T.Tracker(w, w.find_pet, clock=lambda: w.t, log=lambda m: None, sleep=w.sleep, find_now=w.find_now)
    was = T.FAST_FOLLOW
    T.FAST_FOLLOW = True
    try:
        w.advance(1.0)
        tr.step([0.4, 0.3, 0.6, 0.7], moving=False, box_t=w.t)
        w.advance(1.0)
        frozen = (0.03, 0.0, 4, 1.0, w.t, 0.2, 0.4)   # inside EDGE, so nothing aims
        tr.step(None, moving=False, fast=frozen)
        seen_at = tr.last_seen_t
        assert seen_at == w.t, "a new estimate is a sighting"
        w.advance(1.0)
        tr.step(None, moving=False, fast=frozen)
        assert tr.last_seen_t == seen_at, "the same estimate again is not"
    finally:
        T.FAST_FOLLOW = was


def test_estimate_that_keeps_moving_still_stands_in():
    # She walks on with no detection: following that motion is what the tracker is for, so no chase while it updates.
    after, tr, aimed_at = lost_after_a_fast_aim(motion_goes_on=True)
    assert not [a for k, a in after if k <= T.FAST_LOST_MAX - 2 and a.startswith("chase")], after
    assert tr.last_seen_t > aimed_at + 3, (tr.last_seen_t, aimed_at)


def test_old_detector_box_cannot_aim_after_a_slow_cycle():
    # Report 20261103-210058: recording an arrival stalled the watcher; a 13-second-old
    # left-edge box then aimed away from the pet's actual new position near the centre.
    w = World(X=2600, pet=2600)
    tr = T.Tracker(w, w.find_pet, clock=lambda: w.t, log=lambda m: None, sleep=w.sleep)
    w.advance(1)
    captured = w.t
    w.advance(13)
    tr.step([0.0, 0.2, 0.3, 0.8], moving=True, box_t=captured)
    assert not w.moves, "stale detector evidence cannot command an aim"
    tr.step([0.0, 0.2, 0.3, 0.8], moving=True, box_t=w.t)
    assert w.moves, "the next fresh detection can still aim"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
