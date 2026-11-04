"""Follow / chase / search against a simulated room (no hardware). Runs locally or on the server:
  python3 tests/server/tracker_test.py
World: pan position in motor steps; one frame is FW steps wide. A pet at world x=wx is at frame
offset fx = (X - wx) / FW from the centre when the camera is at X (turning right lowers x)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "server"))
import tracker as T  # noqa: E402

FW, FH, XM, YM = 733.0, 424.0, 3700, 1000


class World:
    """A room on a pan/tilt mount. goto() pans over time at the quiet speed (like the real motor);
    aim() is the small tracking move, treated as instant. Frames seen while panning lag 0.7 s."""
    SPEED, LAG = 250.0, 0.7

    def __init__(self, X=1850, pet=1850, pet_y=500, Y=500):
        self.X, self.Y, self.tx, self.pet, self.pet_y, self.t, self.moved = float(X), float(Y), float(X), pet, pet_y, 0.0, -1e9
        self.moves, self.hist = [], [(0.0, float(X), float(Y))]
        self.moving = False

    def advance(self, dt):
        step = self.SPEED * dt
        self.X = self.tx if abs(self.tx - self.X) <= step else self.X + step * (1 if self.tx > self.X else -1)
        self.t += dt
        self.hist.append((self.t, self.X, self.Y))
        self.hist = self.hist[-400:]

    def sleep(self, s):
        for _ in range(max(1, int(round(s / 0.1)))):
            self.advance(0.1)

    # camera API used by the tracker
    def position(self): return round(self.X), round(self.Y)
    def limits(self): return XM, YM
    def frame_steps(self): return FW, FH
    def presets(self): return []
    def home(self): return 1850, 500
    def moved_at(self): return self.moved
    def busy(self): return self.X != self.tx
    def stop(self): self.tx = self.X
    def wait_idle(self):
        while self.busy():
            self.advance(0.1)
        return self.position()
    def goto(self, x, y):
        self.tx, self.Y = float(max(0, min(XM, x))), float(y); self.moved = self.t; self.moves.append(("goto", round(self.tx)))
    def aim(self, fx, fy):
        self.X = self.tx = max(0, min(XM, self.X - fx * FW)); self.Y = self.Y - fy * FH; self.moved = self.t
        self.moves.append(("aim", round(self.X)))

    def box_at(self, X, Y):
        f, g = (X - self.pet) / FW, (Y - self.pet_y) / FH
        return [0.5 + f - 0.08, 0.5 + g - 0.1, 0.5 + f + 0.08, 0.5 + g + 0.1] if abs(f) < 0.45 and abs(g) < 0.45 else None

    def fx(self): return (self.X - self.pet) / FW
    def box(self): return self.box_at(self.X, self.Y)
    def find_pet(self, after):              # a settled look: waits for a fresh frame
        self.sleep(0.4)
        return self.box()
    def find_now(self, since):              # mid-pan: the newest frame shows where the camera was LAG ago
        past = [h for h in self.hist if h[0] <= self.t - self.LAG]
        _, X, Y = past[-1] if past else self.hist[0]
        return self.box_at(X, Y)


def run(world, tr, seconds, pet_speed=0.0, sound_at=None, dt=1.0):
    """Advance the simulation; the pet walks at pet_speed steps/s (negative = to the camera's right).
    The frame reader reports motion where she is whenever she moves inside (or at the edge of) the frame."""
    log = []
    for _ in range(int(seconds / dt)):
        f_before = world.fx()
        world.advance(dt)
        world.pet += pet_speed * dt
        b = world.box()
        motion = None
        if pet_speed and abs(f_before) < 0.5:                 # she moved through the visible part of the picture
            mx = max(-0.49, min(0.49, world.fx()))           # the 5 fps reader last sees her where she leaves the frame
            motion = (world.t - 0.2, mx, 0.0)
        a = tr.step(b, moving=pet_speed != 0 and b is not None, sound=sound_at is not None and abs(world.t - sound_at) < dt / 2,
                    motion=motion)
        if a:
            log.append((round(world.t), a))
    return log


def tracker_for(w):
    tr = T.Tracker(w, w.find_pet, clock=lambda: w.t, log=lambda m: None, sleep=w.sleep, find_now=w.find_now)
    tr.last_seen_t = w.t
    return tr


# 1. Walking across the room: the camera keeps her in view the whole way.
w = World(X=3400, pet=3400); tr = tracker_for(w); seen = 0
for _ in range(25):
    run(w, tr, 1, pet_speed=-110)          # ~0.15 frame/s to the right, across most of the pan range
    seen += w.box() is not None
assert seen == 25, f"lost her while walking: in view {seen}/25 s, moves {w.moves}"
print(f"walk across the room 25 s: in view 25/25, {len(w.moves)} moves")

# 2. She trots out of the frame (0.5 frame/s, faster than one tracking move keeps up with) and lies
#    down by the door: chase/track keep up, and she is in view once she stops.
w = World(X=2600, pet=2600); tr = tracker_for(w); log, seen = [], 0
for _ in range(5):
    log += run(w, tr, 1, pet_speed=-370)
    seen += w.box() is not None
log += run(w, tr, 6)                       # stops; give the tracker a few cycles
assert w.box() is not None, (log, w.moves, round(w.fx(), 2))
print(f"trots out and lies down: in view at the end; during the trot {seen}/5 s;",
      sum("chase" in a for _, a in log), "chase(s),", sum("track" in a for _, a in log), "track move(s)")

# 3. The 2026-11-02 10:04 case: she lies down two frames away, out of view. Search finds her.
w = World(X=1850, pet=1850 + 2 * 733); tr = tracker_for(w)
log = run(w, tr, 60)
found = [a for _, a in log if a.startswith("search")]
assert found and "found her" in found[0] and w.box() is not None, (log, w.moves)
print("asleep out of view:", found[0])

# 4. Not anywhere in reach (another room): search returns to where it was, and waits 10 min to retry.
w = World(X=1200, pet=99999); tr = tracker_for(w)
log = run(w, tr, 300)
s = [a for _, a in log if a.startswith("search")]
assert len(s) == 1 and "not found" in s[0] and w.X == 1200, (s, w.X)
print("not in reach:", s[0])

# 5. A dog sound while she is out of view searches right away (not after 45 s).
w = World(X=1850, pet=1850 - 2 * 733); tr = tracker_for(w)
log = run(w, tr, 10, sound_at=5)
assert any("dog sound" in a and "found her" in a for _, a in log), log
print("barks out of view:", [a for _, a in log][0][:70])

# 6. A person turns the camera: hands off for USER_PAUSE.
w = World(X=1850, pet=1850 + 2 * 733); tr = tracker_for(w)
w.t += 1; w.X = 1000; w.moved = w.t       # the owner moved it (not the tracker)
log = run(w, tr, 120)
assert len(log) == 1 and "paused" in log[0][1] and w.moves == [], (log, w.moves)   # said once, then quiet
log = run(w, tr, 120)
assert any("found her" in a for _, a in log), log
print("owner moved the camera: paused 3 min, then searched and found her")

# 7. Resting, cut off at the side of the picture: centred once after 20 s, not repeatedly (when enabled);
#    while CENTRE_RESTING_ON is off (owner: no automatic moves while she rests until noise is measured): never moves.
w = World(X=1850, pet=1850 - 0.38 * 733); tr = tracker_for(w)
log = run(w, tr, 60)
c = [a for _, a in log if "centre" in a]
if T.CENTRE_RESTING_ON:
    assert len(c) == 1 and abs(w.fx()) < 0.1, (log, w.fx())
    print("resting at the edge:", c[0])
else:
    assert not c and not w.moves, (log, w.moves)
    print("resting at the edge: left alone (automatic moves near a resting dog are off)")
# 8. Lying on the floor below the view (owner, 2026-11-02: searches never tilted). The second row finds her.
w = World(X=1850, Y=600, pet=1850 + 733, pet_y=600 - 0.6 * 424); tr = tracker_for(w)
log = run(w, tr, 120)
s8 = [a for _, a in log if a.startswith("search")]
assert s8 and "found her" in s8[0] and w.box() is not None, (s8, w.position())
print("on the floor below the view:", s8[0])

# 10. Live test 2026-11-02: camera left tilted down (300), the dog higher up on the rug. Rows around HOME find her.
w = World(X=3300, Y=300, pet=2600, pet_y=500 + 0.5 * 424); tr = tracker_for(w)
log = run(w, tr, 150)
s10 = [a for _, a in log if a.startswith("search")]
assert s10 and "found her" in s10[0] and w.box() is not None, (s10, w.position())
print("camera left tilted down, she is higher up:", s10[0])

# 9. Only visible part-way along a pan, with the 0.7 s detection lag: seen while panning, stopped,
#    confirmed on a fresh frame and centred (the lag leaves her behind the centre after the stop).
w = World(X=200, pet=2200); tr = tracker_for(w)
log = run(w, tr, 120)
s9 = [a for _, a in log if a.startswith("search")]
assert s9 and "found her" in s9[0] and w.box() is not None and abs(w.fx()) < 0.2, (s9, round(w.fx(), 2))
print("passed mid-pan despite the lag:", s9[0])

# 11. Out on a walk (2026-11-02 14:50-15:40): two empty searches, then no more timed sweeps for an hour;
#     something moving in the picture re-arms the search.
w = World(X=1200, pet=99999); tr = tracker_for(w)
log = run(w, tr, 3600)
s11 = [a for _, a in log if a.startswith("search")]
assert len(s11) == 2 and "waiting for movement" in s11[1], s11
w.pet = 1200 + 733                                   # back home, walks in at the edge of the picture
log = []
for _ in range(700):
    w.advance(1.0)
    log += [a for a in [tr.step(w.box(), moving=False, motion=(w.t - 0.2, 0.1, 0.0))] if a]
assert any("found her" in a for a in log), log
print(f"out on a walk: {len(s11)} searches in an hour, then searched again on movement")

# 12. Report 2026-11-02 16:51: the dog asleep out of view, people moving about at the edges of the picture.
#     Every bit of edge motion started a chase (+2 continuations): 18 motor moves in 4 min. After one
#     motion-only chase, no new one for MOTION_CHASE_QUIET unless she is seen.
w = World(X=1850, pet=99999); tr = tracker_for(w)
tr.failed_searches = T.GIVE_UP_AFTER; tr.last_search = w.t       # timed searches already given up
edges = [(-0.19 - 0.2, 0.26), (0.39, -0.2), (-0.31, 0.03), (-0.36, 0.32), (-0.37, -0.28), (-0.30, 0.19)]
log = []
for i in range(240):
    w.advance(1.0)
    m = edges[(i // 15) % len(edges)] if i % 15 == 0 else None    # a person crosses an edge every 15 s
    a = tr.step(w.box(), moving=False, motion=(w.t - 0.2, m[0], m[1]) if m else None)
    if a:
        log.append((round(w.t), a))
chases = [a for _, a in log if a.startswith("chase")]
starts = [a for a in chases if "motion at the edge" in a]
assert len(starts) <= 2 and len(chases) <= 6, (len(starts), len(chases), chases)
# ...but once she is seen, motion may start a chase again right away
w.pet = w.X; tr.step(w.box(), moving=False); w.pet = 99999; w.advance(T.REST_MISS_HOLD + 1.0)
assert (tr.step(None, moving=False, motion=(w.t - 0.2, 0.45, 0.0)) or "").startswith("chase: motion"), "chase not re-armed"
print(f"people moving about while she sleeps out of view: {len(chases)} chase moves in 4 min")

# 13. Report 2026-11-02 16:51:06: the watcher restarted in the middle of a search. The OLD process's
#     last search move had touched the 'camera moved' mark 10 s before, the new one took that for a
#     person, logged "paused for 3 min (a person moved the camera)" and stayed on a blank wall.
w = World(X=900, pet=99999); w.t = 1000.0
w.moved = w.t - 10                                   # the old process's own goto, just before the restart
tr = tracker_for(w)                                  # the new process starts
a = tr.step(None, moving=False)
assert not (a or "").startswith("paused"), f"took its own pre-restart move for a person: {a}"
assert not tr.user_paused(), "paused by a mark older than the process"
# ...a person moving the camera after the start still pauses it
w.advance(5.0); w.moved = w.t
assert (tr.step(None, moving=False) or "").startswith("paused"), "a real person's move no longer pauses"
print("restart mid-search: an earlier process's move is not taken for a person")

# 14. Report 2026-11-02 18:31:13: after a restart the watcher ran on camera snapshots (no live stream)
#     for 3.5 min. Its timer search panned 3 rows with no frames to judge, passed the dog on the rug by
#     the TV, and reported "not found". Without the stream, a timer search waits instead of sweeping blind,
#     then runs (and finds her) once the stream is back.
w = World(X=3700, pet=3700 - 2 * 733)
stream = {"up": False}
tr = T.Tracker(w, w.find_pet, clock=lambda: w.t, log=lambda m: None, sleep=w.sleep, find_now=w.find_now,
               can_scan=lambda: stream["up"])
tr.last_seen_t = w.t
log = run(w, tr, 120)
assert not [a for _, a in log if a.startswith("search")] and not w.moves, ("swept blind on snapshots", log, w.moves)
assert tr.failed_searches == 0
stream["up"] = True
log = run(w, tr, 60)
s = [a for _, a in log if a.startswith("search")]
assert s and "found her" in s[0] and w.box() is not None, (log, w.moves)
print("no live stream: waits instead of a blind sweep; with the stream back:", s[0])
# 15. A search holds the watcher's loop for minutes: it must keep calling tick() so the heartbeat and app stay fresh.
w = World(X=1200, pet=99999); ticks = []
tr = T.Tracker(w, w.find_pet, clock=lambda: w.t, log=lambda m: None, sleep=w.sleep, find_now=w.find_now, tick=lambda: ticks.append(w.t))
tr.last_seen_t = w.t - 100
res = tr.search("test")
gaps = [b - a for a, b in zip(ticks, ticks[1:])]
assert "not found" in res and ticks and max(gaps) < 3.5, (res, len(ticks), max(gaps) if gaps else None)
print(f"long sweep keeps the watcher alive: {len(ticks)} ticks, longest gap {max(gaps):.1f} s")
# 16. Report 2026-11-03 08:31: a chase left the camera at the pan end stop (0, 471); the empty search went
#     "back to (0, 471)", a corner of the room. With a saved view named "Corridor", it waits there instead.
#     Without one it still goes back to where it was (case 4).
w = World(X=0, pet=99999)
w.presets = lambda: [{"name": "Sofa", "x": 2900, "y": 600}, {"name": "Corridor", "x": 2252, "y": 704}]
tr = tracker_for(w); tr.last_seen_t = w.t - 100
res = tr.search("test")
w.wait_idle()
assert "not found" in res and "Corridor" in res and (w.X, w.Y) == (2252, 704), (res, w.X, w.Y)
print("empty search parks at the corridor view:", res)

# 17. Night false finds (2026-11-02/03: 31 of 43 "found her" were at night, where an infrared clothes pile or
#     slipper can pass the verifier on one frame). One stray verified frame must not end a search; two frames
#     showing her in the same place do; two hits in different places do not; and a frame that misses a
#     curled-up dog does not stop her being confirmed.
def looker(seq):
    w = World(X=1850, pet=99999); it = iter(seq)
    w.find_pet = lambda after: (w.sleep(0.2), next(it, None))[1]
    return T.Tracker(w, w.find_pet, clock=lambda: w.t, log=lambda m: None, sleep=w.sleep, find_now=w.find_now)
A, B = [0.40, 0.40, 0.50, 0.50], [0.80, 0.10, 0.90, 0.20]
assert looker([None, A, None, None]).look() is None, "one stray frame confirmed a find"
assert looker([A, None, A, None]).look() == A, "two agreeing frames with a miss between did not confirm"
assert looker([A, B, None, None]).look() is None, "two hits in different places confirmed a find"
assert looker([None, None, None, A]).look() is None
calls = []
tr = looker([None, None, None, None]); f = tr.find_pet
tr.find_pet = lambda after: (calls.append(1), f(after))[1]
assert tr.look() is None and len(calls) == T.LOOK_FRAMES - T.CONFIRM_FRAMES + 1, ("did not stop early", len(calls))
print(f"confirmation: {T.CONFIRM_FRAMES} agreeing of {T.LOOK_FRAMES} frames; one stray frame is not a find; gives up after {len(calls)} empty frames")

# 18. Report 2026-11-03 12:22: asleep in the middle of the picture for two hours; she stirred, then the detector
#     missed her on one frame just after the stream restarted, while the frame reader reported motion at the
#     left edge. A motion chase turned the camera left 3 times, away from her, and the search missed her.
#     A pet seen lying still a moment ago is not chased on motion alone; the camera stays put.
w = World(X=1850, pet=1850); tr = tracker_for(w)
log = run(w, tr, 60)                                   # resting, in view
assert not w.moves, w.moves
seen_box = w.box(); w.advance(1.0)
a = tr.step(None, moving=False, motion=(w.t - 0.2, -0.48, -0.06))   # one missed frame + edge motion
w.advance(2.0)
b = tr.step(seen_box, moving=False)                    # the next frame sees her again
assert not (a or "").startswith("chase") and not w.moves and w.box() is not None, (a, b, w.moves)
print("resting pet missed for one frame: no motion chase away from her")

# 19. Report 2026-11-03 16:10: nobody in view for 1.5 h, yet a full sweep ran every 10 min. Each empty search
#     ended with a slow drive back to the saved 'Corridor' view; the frame reader ignores motion only 4 s after a
#     move command, so the picture sliding past 5 s later counted as "something moved since the last sweep" and
#     re-armed the timer. The camera's own drive home must not re-arm it: still only GIVE_UP_AFTER sweeps an hour.
w = World(X=1200, pet=99999)
w.presets = lambda: [{"name": "Corridor", "x": 2252, "y": 704}]   # ~9 s drive from the pan end stop where the sweep ends
tr = tracker_for(w); log = []; last_motion = None  # the watcher passes the LAST motion it saw (frames.motion_c)
for _ in range(3600):
    w.advance(1.0)
    if w.busy() and w.t - w.moved > 4.0:            # still driving after the reader's 4 s blind window
        last_motion = (w.t - 0.2, 0.3, 0.0)
    a = tr.step(None, moving=False, motion=last_motion)
    if a:
        log.append((round(w.t), a))
s19 = [a for _, a in log if a.startswith("search")]
assert len(s19) == T.GIVE_UP_AFTER, (len(s19), s19[:4])
print(f"empty room: {len(s19)} sweeps in an hour; the drive back to the saved view does not re-arm the timer")
print("tracker ok")
