"""Find by sound (server/soundfind.py and the tracker's sound-first stops), without hardware.

  1. The fingerprint ignores loudness and keeps the room: one source through the same simulated room path matches
     itself at any gain better than it matches the same source through another path.
  2. The guess returns the views where the most similar stored sounds came from; too small a memory gives none.
  3. The ledger writes what the guess and the baseline would have said BEFORE a label is stored, and the gate opens
     only on search labels (never on in-view ones) when the guess beats the baseline by the declared margin.
  4. aim_pose uses app.aim's signs; snippets are written as 16 kHz WAV and pruned by age.
  5. The tracker records the sound's guess in every search plan, and only goes there first when SOUND_FIND_USE is on.
  python3 tests/server/soundfind_test.py
"""
import json
import os
import sys
import tempfile
import time
import wave

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "server"))
import soundfind as S  # noqa: E402
import tracker as T  # noqa: E402
from roommap import RoomMap  # noqa: E402
from tracker_test import World, tracker_for  # noqa: E402

rng = np.random.default_rng(1)
FRAME = (733, 424)


def path_ir(delays_ms, gains):
    h = np.zeros(int(0.06 * S.SR))
    h[0] = 1.0
    for d, g in zip(delays_ms, gains):
        h[int(d * S.SR / 1000)] += g
    return h


def burst(n=0.4):
    x = rng.normal(size=int(n * S.SR)) * np.hanning(int(n * S.SR))
    return np.convolve(x, [1, -0.6])[: len(x)]          # a coloured (not white) source


def through(x, h, gain):
    y = np.convolve(x, h)[: len(x) + len(h)] * gain
    y = y / np.abs(y).max() * 0.5 * gain
    return np.concatenate([np.zeros(1600), y, np.zeros(1600)]).clip(-1, 1) * 32767


A = path_ir([2.1, 5.3, 9.8], [0.7, 0.5, 0.4])
B = path_ir([3.7, 7.9, 12.5], [0.6, 0.6, 0.3])

# 1. same path at different gains vs another path
same, other = [], []
for _ in range(10):
    src = burst()
    fa1 = S.fingerprint(through(src, A, 1.0).astype(np.int16))
    fa2 = S.fingerprint(through(burst(), A, 0.2).astype(np.int16))       # another sound, same spot, much quieter
    fb = S.fingerprint(through(src, B, 1.0).astype(np.int16))            # the same sound, another spot
    same.append(np.dot(fa1["ripple"], fa2["ripple"]))
    other.append(np.dot(fa1["ripple"], fb["ripple"]))
assert np.median(same) > np.median(other) + 0.2, (np.median(same), np.median(other))
assert S.fingerprint(np.zeros(800, dtype=np.int16)) is None                # too short / silent
print(f"fingerprint: same spot {np.median(same):.2f}, other spot {np.median(other):.2f}")

# 2-3. guess, ledger and gate
d = tempfile.mkdtemp()
sf = S.SoundFind(d, log=lambda *a: None)
pa, pb = (1000, 500), (2800, 600)
assert sf.guess(S.fingerprint(through(burst(), A, 1).astype(np.int16)), FRAME) == []      # empty memory: no guess
for i in range(S.MIN_MEMORY):
    sf.learn(S.fingerprint(through(burst(), A if i % 2 else B, 1).astype(np.int16)), pa if i % 2 else pb, "view", FRAME, [pb])
g = sf.guess(S.fingerprint(through(burst(), A, 0.5).astype(np.int16)), FRAME)
assert g and S.same_view(g[0]["pose"], pa, FRAME), g
print("guess:", g[:2])

ok, summary = sf.gate()
assert not ok and summary["n"] == 0, summary                 # in-view labels never open the gate
for i in range(S.GATE_MIN_N):
    row = sf.learn(S.fingerprint(through(burst(), A, 0.7).astype(np.int16)), pa, "search", FRAME, [pb])
    assert row["guess"] and row["hit2"] and row["base_hit2"] is False, row
ok, summary = sf.gate()
assert ok and summary["guess_top2"] - summary["baseline_top2"] >= S.GATE_MARGIN, summary
assert sf.use_guess("auto") and not sf.use_guess("shadow") and sf.use_guess("on")
sf2 = S.SoundFind(d, log=lambda *a: None)                   # survives a restart
assert len(sf2.memory) == len(sf.memory) and sf2.gate()[0]
# a guess that stops working is switched off again (judged on the most recent rows)
for i in range(S.GATE_RECENT):
    sf2.learn(S.fingerprint(through(burst(), A, 0.7).astype(np.int16)), pb, "search", FRAME, [pb])
assert not sf2.gate()[0], sf2.gate()
print("gate:", summary, "-> after a run of misses:", sf2.gate()[1])

# 4. aim pose and snippets
assert S.aim_pose((1000, 500), [0.6, 0.4, 0.8, 0.8], FRAME) == [1000 - round(0.2 * 733), 500 - round(0.1 * 424)]
p = sf.keep_snippet(np.arange(-8000, 8000, dtype=np.int16), time.time(), "barking", {"yamnet": {"dog": 0.9}})
with wave.open(p) as w:
    assert w.getframerate() == 16000 and w.getnchannels() == 1 and w.getnframes() == 16000
assert json.load(open(p[:-4] + ".json"))["kind"] == "barking"
old = os.path.join(d, "sounds", "2020-01-01")
os.makedirs(old)
open(os.path.join(old, "x.wav"), "w").close()
sf.prune()
assert not os.path.exists(old) and os.path.exists(p)
print("snippets ok")

# 5. tracker: shadow vs used


def doc(stops):
    poses = [{"pan": pan, "tilt": tilt, "floor_share": 0.3, "tiles": [3, 4, 5]}
             for pan in range(0, 3701, 100) for tilt in range(0, 1001, 100)]
    return {"version": 1, "pan_step": 100, "tilt_step": 100, "min_floor": 0.02, "poses": poses,
            "stops": {str(h): stops for h in range(24)}, "places": {}}


try:
    T.ROOM_MAP = RoomMap(doc([[1000, 500, 0.7]]))
    w = World(X=1850, pet=3000)
    tr = tracker_for(w)
    tr.sound_hint = {"t": w.t, "guess": [{"pose": [3000, 500], "score": 0.8}]}
    T.SOUND_FIND_USE = False
    st = tr.search_stops()
    assert (st[0]["x"], st[0]["y"]) == (1000, 500), st                 # shadow: the map's order is unchanged
    assert tr.search_plan["guess"] == [[3000, 500]] and tr.search_plan["used"] is False, tr.search_plan
    T.SOUND_FIND_USE = True
    res = tr.search("dog sound while out of sight")
    assert "found her at the view her sound came from (3000, 500)" in res, res
    assert w.moves[0] == ("goto", 3000), w.moves
    assert tr.search_plan["used"] is True
    tr.sound_hint["t"] = w.t - T.SOUND_HINT_MAX_AGE - 1                 # an old sound says nothing
    tr.search_stops()
    assert tr.search_plan is None
    print("tracker:", res)
finally:
    T.ROOM_MAP = None
    T.SOUND_FIND_USE = False
print("soundfind ok")

# 6. The watcher: a sound with her in view (no person) is a label; with nobody in view, a hint for the next search;
#    a search that then finds her labels that sound with where she was.
import collections  # noqa: E402
import watch  # noqa: E402

d2 = tempfile.mkdtemp()
watch.SOUNDFIND = S.SoundFind(d2, log=lambda *a: None)
watch.log = lambda *a: None
clock = [5000.0]
watch.now = lambda: clock[0]
w = watch.Watch.__new__(watch.Watch)
world = World(X=1850, pet=1850)
w.tracker = tracker_for(world)
w.privacy_on = lambda: False
w.last_seen = 0
w.sound_views = collections.deque(maxlen=40)
watch.camapi.PAN_STEPS_PER_FRAME, watch.camapi.TILT_STEPS_PER_FRAME = FRAME
fp = S.fingerprint(through(burst(), A, 1).astype(np.int16))
w.sound_views.append((5000.0, [0.4, 0.4, 0.6, 0.6], (1200, 500), 0.0))
w.sound_print({"fp": fp, "kind": "barking", "dog": True, "dur": 1.0}, 4999.0)
assert len(watch.SOUNDFIND.memory) == 1 and watch.SOUNDFIND.memory[0] == [1200, 500], watch.SOUNDFIND.memory
w.sound_views.append((5001.0, [0.4, 0.4, 0.6, 0.6], (1200, 500), 0.9))      # a person in view: not her label
w.sound_print({"fp": fp, "kind": "sound", "dog": False, "dur": 1.0}, 5001.0)
assert len(watch.SOUNDFIND.memory) == 1
clock[0] = 6000.0
w.sound_print({"fp": fp, "kind": "barking", "dog": True, "dur": 1.0}, 6000.0)          # nobody in view: a hint
assert w.tracker.sound_hint and w.tracker.sound_hint["t"] == 6000.0 and w.tracker.sound_hint["guess"] == []
w.tracker.last_search = 6020.0
watch.camapi.motor_status = lambda: ("0", 2400, 520)
w.sound_search_label("search (dog sound while out of sight): found her at the saved view (2400, 500) in 9 s")
assert watch.SOUNDFIND.memory[-1] == [2400, 520] and watch.SOUNDFIND.ledger[-1]["source"] == "search"
assert w.tracker.sound_hint is None
print("watcher: in-view label, person skipped, hint, search label")
watch.SOUNDFIND = None
print("soundfind watcher ok")
