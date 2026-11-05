"""The per-frame motion tracker (server/fasttrack.py) and its hand-off to Follow (tracker.FAST_FOLLOW)."""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "server"))
sys.path.insert(0, HERE)
import fasttrack  # noqa: E402
from fasttrack import FastTrack, W, H  # noqa: E402
import tracker as T  # noqa: E402
from tracker_test import World  # noqa: E402


def blob_frame(cx, cy, r=6, base=60):
    f = np.full((H, W), base, np.int16)
    yy, xx = np.mgrid[:H, :W]
    f[(xx - cx * W) ** 2 + (yy - cy * H) ** 2 <= r * r] = 200
    return f


def test_follows_a_walking_blob():
    ft = FastTrack()
    ft.anchor(0.0, [0.25, 0.35, 0.35, 0.55])         # box centred at (0.30, 0.45)
    prev = blob_frame(0.30, 0.45)
    for k in range(1, 21):                            # walks right 0.02 of the frame per frame
        cur = blob_frame(0.30 + 0.02 * k, 0.45)
        ft.frame(k * 0.2, FastTrack.mask(cur, prev))
        prev = cur
    e = ft.estimate(4.0)
    assert e is not None and e[2] > 10, e
    assert abs(e[0] - (0.70 - 0.5)) < 0.06, e         # near where the blob is now, not where the box was


def test_ignores_small_flicker_near_a_still_dog():
    ft = FastTrack()
    ft.anchor(0.0, [0.25, 0.35, 0.45, 0.65])
    rng = np.random.default_rng(0)
    prev = np.full((H, W), 60, np.int16)
    for k in range(1, 30):
        cur = prev.copy()
        y, x = rng.integers(30, 60), rng.integers(40, 75)
        cur[y:y + 2, x:x + 2] = 200                   # a 4-pixel flicker (well under MIN_FRAC of her box)
        ft.frame(k * 0.2, FastTrack.mask(cur, prev))
        prev = np.full((H, W), 60, np.int16)
    e = ft.estimate(6.0)
    assert e[2] == 0 and abs(e[0] - (0.35 - 0.5)) < 1e-9, e


def test_late_box_catches_up_on_buffered_motion():
    ft = FastTrack()
    prev = blob_frame(0.30, 0.45)
    for k in range(1, 11):
        cur = blob_frame(0.30 + 0.02 * k, 0.45)
        ft.frame(k * 0.2, FastTrack.mask(cur, prev))
        prev = cur
    ft.anchor(0.1, [0.25, 0.35, 0.35, 0.55])          # the box arrives late, for the frame taken at 0.1 s
    e = ft.estimate(2.0)
    assert abs(e[0] - (0.50 - 0.5)) < 0.06, e


def test_own_move_and_camera_motion():
    ft = FastTrack()
    ft.anchor(0.0, [0.55, 0.4, 0.65, 0.6])
    ft.shift(0.2, 0.0)                                # we aimed 0.2 right: she is 0.2 further left in the picture
    assert abs(ft.estimate(0.5)[0] - (0.40 - 0.5)) < 1e-9
    ft.frame(0.6, np.ones((H, W), bool), camera_moving=True)   # the motor's own whole-picture change
    assert ft.estimate(0.7)[2] == 0
    assert ft.estimate(0.0 + fasttrack.MAX_AGE + 1) is None


def test_follow_handoff():
    w = World(X=2600, pet=2600)
    tr = T.Tracker(w, w.find_pet, clock=lambda: w.t, log=lambda m: None, sleep=w.sleep, find_now=w.find_now)
    stale = [0.62, 0.4, 0.72, 0.6]                    # the last verified box, handed again between detector runs
    w.advance(1.0); tr.step(stale, moving=True, box_t=w.t)
    w.advance(3.0)
    fast = (0.40, 0.0, 12, 3.0, w.t, 0.10, 0.20)      # motion says she has walked on to the right edge
    T.FAST_FOLLOW = False
    try:
        assert tr.step(stale, moving=True, box_t=w.t - 3, fast=fast) is None     # off: the stale box, inside EDGE
        T.FAST_FOLLOW = True
        act = tr.step(stale, moving=True, box_t=w.t - 3, fast=fast)
        assert act and act.startswith("track (fast): aim (+") and float(act.split("(")[2][1:5]) > 0.3, act   # toward her, beyond the stale box
        w.advance(3.0)
        near = (0.18 + 0.01, 0.0, 3, 1.0, w.t, 0.10, 0.20)
        assert tr._fast_box(w.t, [0.63, 0.4, 0.73, 0.6], near) is None             # hardly moved: keep the box
        assert tr._fast_box(w.t + 20, None, fast) is None                           # lost too long ago
        assert tr._fast_box(w.t, None, (0.1, 0, 0, 1.0, w.t, 0.1, 0.2)) is None    # never followed any motion
    finally:
        T.FAST_FOLLOW = False


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
