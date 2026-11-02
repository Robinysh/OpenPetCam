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
    def __init__(self, X=1850, pet=1850):
        self.X, self.Y, self.pet, self.t, self.moved = X, 500, pet, 0.0, -1e9
        self.moves = []

    # camera API used by the tracker
    def position(self): return self.X, self.Y
    def limits(self): return XM, YM
    def frame_steps(self): return FW, FH
    def presets(self): return []
    def moved_at(self): return self.moved
    def wait_idle(self): return self.X, self.Y
    def goto(self, x, y):
        self.X, self.Y = max(0, min(XM, x)), y; self.moved = self.t; self.moves.append(("goto", round(self.X)))
    def aim(self, fx, fy):
        self.X = max(0, min(XM, self.X - fx * FW)); self.moved = self.t; self.moves.append(("aim", round(self.X)))

    def fx(self): return (self.X - self.pet) / FW
    def box(self):
        f = self.fx()
        return [0.5 + f - 0.08, 0.40, 0.5 + f + 0.08, 0.60] if abs(f) < 0.45 else None


def run(world, tr, seconds, pet_speed=0.0, sound_at=None, dt=1.0):
    """Advance the simulation; the pet walks at pet_speed steps/s (negative = to the camera's right).
    The frame reader reports motion where she is whenever she moves inside (or at the edge of) the frame."""
    log = []
    for _ in range(int(seconds / dt)):
        world.t += dt
        f_before = world.fx()
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
    tr = T.Tracker(w, lambda after: w.box(), clock=lambda: w.t, log=lambda m: None)
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

# 7. Resting, cut off at the side of the picture: centred once after 20 s, not repeatedly.
w = World(X=1850, pet=1850 - 0.38 * 733); tr = tracker_for(w)
log = run(w, tr, 60)
c = [a for _, a in log if "centre" in a]
assert len(c) == 1 and abs(w.fx()) < 0.1, (log, w.fx())
print("resting at the edge:", c[0])
print("tracker ok")
