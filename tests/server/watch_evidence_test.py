"""Run the actual Watch cycle with controlled frames: visual evidence and sound routing."""
import collections
import inspect
import os
import queue
import sys
import tempfile
import time
import types

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "server"))
sys.path.insert(0, HERE)
import watch  # noqa: E402
import tracker as T  # noqa: E402
from tracker_test import World  # noqa: E402


def cycle_case(d, visible=False, privacy=False, follow=True, tiled=False):
    world = World()
    world.t = 10000.0
    watch.now = lambda: world.t
    watch.HEARTBEAT = os.path.join(d, "heartbeat")
    watch.MOVED = os.path.join(d, "moved")
    rows, searches = [], []
    con = types.SimpleNamespace(execute=lambda *a: rows.append(a), close=lambda: None)
    watch.db = lambda: con
    watch.add_event = lambda *a: None
    watch.log = lambda *a: None
    w = watch.Watch.__new__(watch.Watch)
    w.tracker = T.Tracker(world, world.find_pet, lambda: world.t, log=lambda m: None, sleep=world.sleep)
    # The pet disappeared 10 s ago; arrival/left debounce still reports presence for 3 minutes.
    w.tracker.last_seen_t = world.t - 10
    w.tracker.search = lambda reason: searches.append(reason) or "search: not found"
    w.present, w.last_seen = True, world.t - 10
    w.present_since, w.gone_since = world.t - 100, None
    w.rest_since, w.moving_since = world.t - 100, None
    w.cam_moves, w.move_box0 = collections.deque(maxlen=20), None
    w.dog_sound_pending = True
    w.source, w.frame_mode = "stream", "push"
    w.offline, w.last_motion_t, w.last_detect = False, 0.0, 0.0
    w.prev_small, w.prev_box, w.verifier = None, None, None
    if tiled:
        w.verifier = types.SimpleNamespace(threshold=0.8, prob=lambda *a: 0.99)
        w.still_anchor = (world.t - 10, [0.4, 0.4, 0.6, 0.6])
        w.tiled_box = lambda *a, **k: (0.99, [0.4, 0.4, 0.6, 0.6])
    w.last_found, w.policy_arm, w.ppo, w.code_version = [], "base", None, "test"
    w.last_imaging, w.last_house = world.t, time.strftime("%Y-%m-%d")
    w.det = types.SimpleNamespace(detect=lambda *a, **k: ([{"label": "dog", "score": 1.0,
                                                         "box": [0.4, 0.4, 0.6, 0.6]}] if visible else []))
    w.frames = types.SimpleNamespace(latest=lambda: Image.new("RGB", (160, 90)), got_jpeg=b"test",
                                    got=(world.t, 0.55, "push"), motion_c=None,
                                    motion_track=types.SimpleNamespace(anchor=lambda *a: None, estimate=lambda *a: None),
                                    want=lambda *a: None, want_fast=lambda *a: None)
    w.check_frozen = lambda *a: False
    w.check_blocked = lambda *a: None
    w.camera_moved_recently = lambda: False
    w.privacy_on = lambda: privacy
    w.setting = lambda *a: follow
    w.find_requested = lambda *a: False
    w.apply_policy = lambda *a: None
    w.current_pose = lambda *a: (1850, 500)
    for method in ("save_candidate", "save_search_evidence", "check_zones", "drain_events", "accumulate", "publish"):
        setattr(w, method, lambda *a, **k: None)
    w.cycle()
    return w, searches


def test_sound(d):
    w, searches = cycle_case(d)
    assert w.present, "arrival/left debounce remains active"
    assert len(searches) == 1, "a dog sound must reach Follow when no pet box is visible, even during leave debounce"
    assert not w.dog_sound_pending, "the sound hint is consumed once"
    assert not cycle_case(d, visible=True)[1], "a sound does not search away from a visible pet"
    assert not cycle_case(d, privacy=True)[1], "privacy forbids sound-driven search"
    assert not cycle_case(d, follow=False)[1], "Follow off forbids sound-driven search"


def test_activity(d):
    watch.SNAPS = d
    w = watch.Watch.__new__(watch.Watch)
    w.present, w.minute, w.acc = True, None, None
    w.audio = types.SimpleNamespace(take_minute=lambda: (None, 0))
    saved = []
    con = types.SimpleNamespace(execute=lambda sql, row: saved.append(row))
    # One actual visible/moving sample, then 19 empty samples during the leave debounce.
    for i in range(21):
        kwargs = {"visible": i == 0} if "visible" in inspect.signature(w.accumulate).parameters else {}
        w.accumulate(con, 1200 + 3 * i, b"jpeg", 0.0, i == 0, False, **kwargs)
    assert saved[0][1:4] == (20, 1, 1), "minute visibility must count evidence, not 3-minute presence debounce: %r" % (saved,)


def test_household_queue():
    watch.now = lambda: 10000.0
    watch.add_event = lambda *a: 1
    for label, speaker, ts, capped, want in (
        ("Bathtub (filling or washing)", False, 9999.0, False, True),
        ("Door", False, 9999.0, False, True),
        ("Walk, footsteps", False, 9999.0, False, True),
        ("Music", False, 9999.0, False, False),
        ("Speech", False, 9999.0, False, False),
        ("Door", True, 9999.0, False, False),
        ("Door", False, 9900.0, False, False),
        ("Door", False, 9999.0, True, True),
        ("Bathtub (filling or washing)", False, 9999.0, True, True),
    ):
        w = watch.Watch.__new__(watch.Watch)
        w.events = queue.Queue()
        w.events.put(("sound", {"kind": "loud_sound", "yamnet": {"top": label}}, ts))
        w.present, w.last_seen = False, 9000.0
        w.activity_sound_pending = 0.0
        emitted = []
        watch.add_event = lambda *a: emitted.append(a) or 1
        w.owner_talking = lambda *a: speaker
        w.save_thumb = lambda *a: None
        con = types.SimpleNamespace(execute=lambda *a: types.SimpleNamespace(
            fetchone=lambda: (watch.LOUD_PER_HOUR if capped else 0,)))
        w.drain_events(con, b"")
        assert bool(w.activity_sound_pending) == want, (label, speaker, ts, capped)
        if capped:
            assert not emitted, "the independent search hint must not bypass the moment/notification cap"


def test_still_recovery(d):
    assert not watch.STILL_RECHECK_ENABLED, "negative gate rejected this candidate: it must stay off by default"
    w, _ = cycle_case(d, tiled=True)
    assert not w.last_found, "research recovery must not silently run in normal Follow"
    watch.STILL_RECHECK_ENABLED = True
    w, searches = cycle_case(d, tiled=True)
    assert w.last_found and w.last_found[-1].get("verified", 0) >= 0.8, "fresh zoom evidence must recover a recent resting pet missed in the whole frame"
    assert not searches, "a recovered pet must prevent an out-of-sight sound search"
    watch.STILL_RECHECK_ENABLED = False


def test_still_bounds(d):
    watch.STILL_RECHECK_ENABLED = True
    watch.MOVED = os.path.join(d, "manual_move")
    w = watch.Watch.__new__(watch.Watch)
    w.verifier = types.SimpleNamespace(threshold=0.8)
    w.tracker = types.SimpleNamespace(last_move=-1e9)
    w.setting = lambda *a: True
    w.last_found = []
    w.still_anchor = (1000.0, [0.4, 0.4, 0.6, 0.6])
    w.frame_capture_t = 1001.0
    calls = []
    w.tiled_box = lambda *a, **k: calls.append(k["tiles"]) or None
    # A failed ROI gets full coverage once; repeated frames cannot spend the budget.
    w.recover_still_pet(None, 1001, 0, False)
    assert [len(c) for c in calls] == [1, 8]
    w.recover_still_pet(None, 1002, 0, False)
    assert len(calls) == 2
    w.recover_still_pet(None, 1013, 0, False)
    assert [len(c) for c in calls] == [1, 8, 1]
    w.recover_still_pet(None, 1061, 0, False)
    assert [len(c) for c in calls] == [1, 8, 1, 1, 8]
    w.recover_still_pet(None, 1121, 0, False)
    assert len(calls) == 5, "expired anchor cannot sustain periodic searching"
    # A new manual move invalidates the old ROI even once the usual move-settling window ends.
    with open(watch.MOVED, "w") as f:
        f.write("moved")
    os.utime(watch.MOVED, (1002, 1002))
    w.recover_still_pet(None, 1016, 0, False)
    assert w.still_anchor is None and len(calls) == 5
    os.unlink(watch.MOVED)
    for moving, enabled, person in ((True, True, False), (False, False, False), (False, True, True)):
        w.still_anchor = (2000.0, [0.4, 0.4, 0.6, 0.6])
        w.last_found = [{"label": "person", "score": 0.8}] if person else []
        w.setting = lambda *a: enabled
        w.recover_still_pet(None, 2001, watch.MOTION_DETECT + 1 if moving else 0, False)
        assert len(calls) == 5, "motion, Follow off, and a person inhibit extra inference"
    w.last_found = []
    w.setting = lambda *a: True
    w.tracker.last_move = 2000.5
    w.recover_still_pet(None, 2002, 0, False)
    assert w.still_anchor is None and len(calls) == 5
    watch.STILL_RECHECK_ENABLED = False


with tempfile.TemporaryDirectory() as d:
    # Individually selectable to demonstrate both independent baseline failures.
    if len(sys.argv) == 1 or "sound" in sys.argv:
        test_sound(d)
    if len(sys.argv) == 1 or "activity" in sys.argv:
        test_activity(d)
    if len(sys.argv) == 1 or "household" in sys.argv:
        test_household_queue()
    if len(sys.argv) == 1 or "still" in sys.argv:
        test_still_recovery(d)
        test_still_bounds(d)
print("watch evidence: ok")
