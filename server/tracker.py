"""Following the pet: track, chase, search. Used by petcam-watch when "Follow" is on.

Why (2026-11-02 10:04): the pet walked through the frame and lay down by the door, out of view. The
old follow only re-centred on a pet it could see near the edge (two detections in a row, at most
every 10 s), so a quick walk-through left the camera staring at an empty room for 15 minutes until
the owner turned it by hand. This keeps the pet in view in three ways:

  track   every verified detection updates her position and velocity; the camera aims where she is
          going (lead), not where she was, as soon as that leaves the middle of the frame.
  chase   she leaves the frame while moving toward an edge: turn that way at once.
  search  she has been out of sight for a while, or a dog sound is heard while she isn't visible:
          sweep the room (saved views first, then a grid across the pan range), stop where she is
          found, otherwise go back to where the camera was.

The owner always wins: any move they make pauses all of this for USER_PAUSE seconds.
Pure logic over an injected camera, detector and clock, so it is tested without hardware
(tests/server/tracker_test.py).
"""
import collections
import math

TRACK_WINDOW = 4.0      # seconds of detections used for the velocity estimate
LEAD = 1.0              # aim where she will be this many seconds ahead
EDGE = 0.20             # re-aim when the predicted centre is this far from the middle (frame = 1.0 wide)
MOVE_EVERY = 2.5        # seconds between tracking moves
CENTRE_RESTING = 0.36   # a resting pet this far off-centre (partly cut off) for RESTING_WAIT s gets centred once
RESTING_WAIT = 20.0
CHASE_WINDOW = (1.0, 8.0)   # chase if she was last seen this many seconds ago...
CHASE_EDGE = 0.20       # ...off-centre toward an edge (or heading out of the frame), and was moving
SEARCH_AFTER = 45.0     # out of sight this long (after having been followed) -> search
SEARCH_EVERY = 600.0    # at most one unprompted search per 10 min
SOUND_SEARCH_EVERY = 180.0  # a dog sound while she is not visible may search every 3 min
USER_PAUSE = 180.0      # a move by a person pauses following/searching this long
MAX_STOPS = 8
SETTLE = 0.9            # seconds after a move before a frame is trusted (motion blur, stream latency)


class Tracker:
    def __init__(self, cam, find_pet, clock, log=print, sleep=None):
        """cam: position() -> (x, y); goto(x, y); aim(fx, fy); wait_idle(); limits() -> (xmax, ymax);
                 frame_steps() -> (pan steps per frame width, tilt steps per frame height);
                 presets() -> [{"x":..,"y":..}]; moved_at() -> mtime of the 'camera moved' mark.
           find_pet(after_t) -> box [x0, y0, x1, y1] of the verified pet in a frame newer than after_t, or None."""
        self.cam, self.find_pet, self.clock, self.log = cam, find_pet, clock, log
        self.sleep = sleep or (lambda s: None)
        self.hist = collections.deque()
        self.last_seen_t = clock()     # starting up counts as "just lost": an unseen pet still gets searched for
        self.last_seen_c = None
        self.last_vel = (0.0, 0.0)
        self.last_move = -1e9
        self.own_move_t = -1e9
        self.last_moving = False
        self.chase_dir = None         # (fx, fy) of the last chase; continued while she is still not found
        self.chase_steps = 0
        self.chased_for = None      # last_seen_t of the loss we already chased
        self.last_search = -1e9
        self.offcentre_since = None
        self.searching = False
        self.was_paused = False

    # ---- helpers -------------------------------------------------------------------------
    def _move(self, fn, *a):
        fn(*a)
        self.own_move_t = self.last_move = self.clock()
        self.hist.clear()                    # positions in the old view no longer apply

    def user_paused(self):
        try:
            m = self.cam.moved_at()
        except OSError:
            return False
        return m > self.own_move_t + 1.5 and self.clock() - m < USER_PAUSE

    def velocity(self):
        if len(self.hist) < 2:
            return 0.0, 0.0
        (t0, x0, y0), (t1, x1, y1) = self.hist[0], self.hist[-1]
        dt = t1 - t0
        return ((x1 - x0) / dt, (y1 - y0) / dt) if dt >= 0.8 else (0.0, 0.0)

    # ---- every cycle ---------------------------------------------------------------------
    def step(self, box, moving, sound=False, motion=None):
        """Call once per watcher cycle with the verified pet box (or None), and the latest motion
        centroid (t, fx, fy) from the frame reader. Returns what it did, or None."""
        t = self.clock()
        if self.user_paused():
            self.hist.clear()
            if not self.was_paused:
                self.was_paused = True
                return f"paused for {USER_PAUSE / 60:.0f} min (a person moved the camera)"
            return None
        self.was_paused = False
        if box is not None:
            return self._track(t, box, moving)
        return self._lost(t, sound, motion)

    def _track(self, t, box, moving):
        cx, cy = (box[0] + box[2]) / 2 - 0.5, (box[1] + box[3]) / 2 - 0.5
        self.hist.append((t, cx, cy))
        while self.hist and t - self.hist[0][0] > TRACK_WINDOW:
            self.hist.popleft()
        recent = t - self.last_seen_t < 3.0
        self.chase_dir, self.chase_steps = None, 0
        self.last_moving = bool(moving)
        self.last_seen_t, self.last_seen_c = t, (cx, cy)
        if len(self.hist) >= 2 and self.hist[-1][0] - self.hist[0][0] >= 0.8:
            vx, vy = self.velocity()
        elif recent:
            vx, vy = self.last_vel          # just after our own move: her speed in the room is unchanged
        else:
            vx, vy = 0.0, 0.0
        self.last_vel = (vx, vy)
        if t - self.last_move < MOVE_EVERY:
            return None
        px, py = cx + vx * LEAD, cy + vy * LEAD
        speed = math.hypot(vx, vy)
        if (moving or speed > 0.05) and (abs(px) > EDGE or abs(py) > EDGE * 1.25):
            fx, fy = max(-0.6, min(0.6, px * 0.9)), max(-0.6, min(0.6, py * 0.8))
            self._move(self.cam.aim, fx, fy)
            self.offcentre_since = None
            return f"track: aim ({fx:+.2f}, {fy:+.2f}), speed {speed:.2f}/s"
        # Lying down but cut off at the side of the picture: centre her once, but only if she is mostly
        # out of the picture. A motor next to a sleeping dog startles her (owner, 2026-11-02).
        if not moving and max(abs(cx), abs(cy)) > CENTRE_RESTING:
            self.offcentre_since = self.offcentre_since or t
            if t - self.offcentre_since >= RESTING_WAIT:
                self.offcentre_since = None
                self._move(self.cam.aim, cx * 0.9, cy * 0.8)
                return f"centre resting pet ({cx:+.2f}, {cy:+.2f})"
        else:
            self.offcentre_since = None
        return None

    def _chase(self, fx, fy, why):
        self.chase_dir, self.chase_steps = (fx, fy), self.chase_steps + 1
        self._move(self.cam.aim, fx, fy)
        return f"chase: {why}, turned ({fx:+.2f}, {fy:+.2f})"

    def _lost(self, t, sound, motion=None):
        gone = t - self.last_seen_t
        # keep going the way she went: one turn rarely catches up with a trotting dog
        if self.chase_dir and self.chase_steps < 3 and t - self.last_move >= MOVE_EVERY:
            return self._chase(*self.chase_dir, f"still not in view, continuing ({self.chase_steps + 1}/3)")
        # chase on motion: she crossed too fast to be detected twice (or at all), but the frame reader
        # saw where the picture changed. Motion near an edge, just now, with no pet in view: look there.
        if motion and t - motion[0] < 3.0 and motion[0] > self.own_move_t + 2.0 and motion[0] != self.chased_for:
            mx, my = motion[1], motion[2]
            if max(abs(mx), abs(my) * 1.2) > 0.3 and t - self.last_move > MOVE_EVERY:
                self.chased_for, self.chase_steps = motion[0], 0
                fx, fy = (math.copysign(0.75, mx), 0.0) if abs(mx) >= abs(my) * 1.2 else (0.0, math.copysign(0.6, my))
                return self._chase(fx, fy, f"motion at the edge ({mx:+.2f}, {my:+.2f}) and no pet in view")
        # chase: she just walked out at an edge
        # (only if she was moving: a resting pet the detector briefly misses must not be "chased" away)
        if self.last_seen_c and self.last_moving and self.chased_for != self.last_seen_t and \
                CHASE_WINDOW[0] <= gone <= CHASE_WINDOW[1]:
            cx, cy = self.last_seen_c
            vx, vy = self.last_vel
            look = min(gone, 3.0)
            ex, ey = cx + vx * look, cy + vy * look         # where she is likely to be by now
            if max(abs(ex), abs(ey)) > CHASE_EDGE:
                self.chased_for, self.chase_steps = self.last_seen_t, 0
                if abs(ex) >= abs(ey):
                    fx, fy = math.copysign(0.75, ex), 0.0
                else:
                    fx, fy = 0.0, math.copysign(0.6, ey)
                return self._chase(fx, fy, f"she left the frame heading ({ex:+.2f}, {ey:+.2f})")
        due = (gone > SEARCH_AFTER and t - self.last_search > SEARCH_EVERY) or \
              (sound and t - self.last_search > SOUND_SEARCH_EVERY)
        if due:
            return self.search(reason="dog sound while out of sight" if sound and gone <= SEARCH_AFTER else
                               f"out of sight {gone:.0f} s")
        return None

    # ---- search ---------------------------------------------------------------------------
    def stops(self, x0, y0):
        """Saved views first (where people expect her), then a grid across the pan range at the
        current tilt, nearest to the side she was last heading first."""
        xm, ym = self.cam.limits()
        fw, _ = self.cam.frame_steps()
        pts = [(p["x"], p["y"]) for p in self.cam.presets()]
        n = max(2, int(math.ceil(xm / (fw * 0.85))) + 1)
        grid = [(round(i * xm / (n - 1)), y0) for i in range(n)]
        lean = -1 if (self.last_seen_c and self.last_seen_c[0] > 0) else 1   # +fx (right) means x down
        grid.sort(key=lambda p: (abs(p[0] - x0) + (0 if (p[0] - x0) * lean >= 0 else fw * 0.5)))
        out = []
        for p in pts + grid:
            if all(abs(p[0] - q[0]) > fw * 0.4 or abs(p[1] - q[1]) > fw * 0.25 for q in out + [(x0, y0)]):
                out.append(p)
        return out[:MAX_STOPS]

    def search(self, reason):
        t0 = self.clock()
        self.last_search = t0
        self.searching = True
        try:
            x0, y0 = self.cam.position()
            visited = 0
            for (x, y) in self.stops(x0, y0):
                if self.user_paused():
                    return f"search ({reason}): stopped, a person moved the camera"
                self._move(self.cam.goto, x, y)
                # goto returns once the camera ACCEPTS the move; a long pan takes seconds. Judging a
                # frame before the motor stops credited the pet (seen mid-pan) to the wrong stop (live test
                # 2026-11-02: "found" at a blank wall). Wait for the motor, then let the picture settle.
                self.cam.wait_idle()
                self.sleep(SETTLE)
                visited += 1
                box = self.find_pet(self.clock())
                if box is not None:
                    cx, cy = (box[0] + box[2]) / 2 - 0.5, (box[1] + box[3]) / 2 - 0.5
                    if max(abs(cx), abs(cy)) > 0.15:
                        self._move(self.cam.aim, cx * 0.9, cy * 0.8)
                    self.last_seen_t, self.last_seen_c = self.clock(), (0.0, 0.0)
                    return f"search ({reason}): found her at stop {visited} ({x}, {y}) in {self.clock() - t0:.0f} s"
            self._move(self.cam.goto, x0, y0)
            return f"search ({reason}): not found at {visited} stops, back to ({x0}, {y0})"
        finally:
            self.searching = False
