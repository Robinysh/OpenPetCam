"""Following the pet: track, chase, search. Used by petcam-watch when "Follow" is on.

Why (2026-11-02 10:04): the dog walked through the frame and lay down by the door, out of view. The
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
CENTRE_RESTING_ON = False   # owner, 2026-11-02: no automatic moves while she rests until the motor noise is measured
                            # (tools/qa.py --hardware noise); re-enable from the measurement, not by guess
CHASE_WINDOW = (1.0, 8.0)   # chase if she was last seen this many seconds ago...
CHASE_EDGE = 0.20       # ...off-centre toward an edge (or heading out of the frame), and was moving
SEARCH_AFTER = 45.0     # out of sight this long (after having been followed) -> search
SEARCH_EVERY = 600.0    # at most one unprompted search per 10 min
SOUND_SEARCH_EVERY = 180.0  # a dog sound while she is not visible may search every 3 min
GIVE_UP_AFTER = 2       # this many unprompted searches in a row found nothing: stop searching on a timer
                        # until something moves in the picture or a dog sound is heard (2026-11-02: the
                        # camera swept the empty flat every 10 min for an hour while she was out on a walk)
MOTION_CHASE_QUIET = 180.0  # after a chase set off by motion alone, no new motion chase for this long unless
                        # she is seen (2026-11-02 16:44-16:48: people moving about while she slept on the rug
                        # set off 6 motion chases, 18 motor moves in 4 min; the owner wants fewer, quieter moves)
USER_PAUSE = 180.0      # a move by a person pauses following/searching this long
SETTLE = 0.9            # seconds after a move before a frame is trusted (motion blur, stream latency)
ROW_STEP = 0.55         # the second search row is this many frame heights toward the floor
LOOK_FRAMES = 3         # frames judged when stopped (one frame can miss her)
SCAN_POLL = 0.2         # check a new frame this often while panning (the stream delivers 5 fps)


class Tracker:
    def __init__(self, cam, find_pet, clock, log=print, sleep=None, find_now=None, can_scan=None, tick=None):
        """cam: position() -> (x, y); home() -> (x, y); goto(x, y) (returns at once); aim(fx, fy); wait_idle(); busy(); stop();
                 limits() -> (xmax, ymax);
                 frame_steps() -> (pan steps per frame width, tilt steps per frame height);
                 presets() -> [{"x":..,"y":..}]; moved_at() -> mtime of the 'camera moved' mark.
           find_pet(after_t) -> box [x0, y0, x1, y1] of the verified pet in a frame newer than after_t, or None.
           find_now(since) -> the same for the newest frame if it arrived after `since`, without waiting.
           can_scan() -> False while frames come from slow snapshots instead of the live stream: a pan
           then sees almost nothing, so a timer search waits for the stream instead of sweeping blind."""
        self.cam, self.find_pet, self.clock, self.log = cam, find_pet, clock, log
        self.find_now = find_now or (lambda since: None)   # newest frame since `since`, no waiting
        self.sleep = sleep or (lambda s: None)
        self.can_scan = can_scan or (lambda: True)
        # tick(): called every few seconds while a search holds the watcher's loop, so it can keep its heartbeat
        # and the app's status fresh (report 2026-11-02 16:51: a 2-min sweep showed 'Watcher not running').
        self.tick = tick or (lambda: None)
        self.hist = collections.deque()
        self.last_seen_t = clock()     # starting up counts as "just lost": an unseen pet still gets searched for
        self.last_seen_c = None
        self.last_vel = (0.0, 0.0)
        self.last_move = -1e9
        self.own_move_t = -1e9
        # Report 2026-11-02 16:51: a restart mid-search found the 'camera moved' mark from the OLD
        # process's own search moves, took it for a person and froze on a blank wall for 3 min.
        # A mark older than this process can't be told apart from our own moves, so it doesn't pause.
        self.started_t = clock()
        self.last_moving = False
        self.chase_dir = None         # (fx, fy) of the last chase; continued while she is still not found
        self.chase_steps = 0
        self.chased_for = None      # last_seen_t of the loss we already chased
        self.motion_chase_t = -1e9    # when motion alone last started a chase (cleared when she is seen)
        self.last_search = -1e9
        self.failed_searches = 0      # unprompted searches in a row that found nothing
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
        return m > max(self.own_move_t + 1.5, self.started_t) and self.clock() - m < USER_PAUSE

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
        self.motion_chase_t = -1e9
        self.failed_searches = 0
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
        if CENTRE_RESTING_ON and not moving and max(abs(cx), abs(cy)) > CENTRE_RESTING:
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
            if max(abs(mx), abs(my) * 1.2) > 0.3 and t - self.last_move > MOVE_EVERY and \
                    t - self.motion_chase_t >= MOTION_CHASE_QUIET:
                self.chased_for, self.chase_steps, self.motion_chase_t = motion[0], 0, t
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
        # After GIVE_UP_AFTER empty sweeps she is probably not home: a timer alone no longer moves the
        # camera. Something moving in the picture since the last sweep (not our own move) re-arms it.
        stirred = bool(motion) and motion[0] > max(self.last_search, self.own_move_t) + 2.0
        timer_ok = self.failed_searches < GIVE_UP_AFTER or stirred
        due = (gone > SEARCH_AFTER and t - self.last_search > SEARCH_EVERY and timer_ok) or \
              (sound and t - self.last_search > SOUND_SEARCH_EVERY)
        # Report 2026-11-02 18:31: after a restart the watcher ran 3.5 min on camera snapshots; the search
        # panned all 3 rows with no frames to check, "found nothing" past her on the rug by the TV, and
        # counted as an empty search. Sweeping blind is only motor noise: wait for the live stream.
        if due and not self.can_scan():
            return None
        if due:
            res = self.search(reason="dog sound while out of sight" if sound and gone <= SEARCH_AFTER else
                              f"out of sight {gone:.0f} s")
            if "not found" in res:
                self.failed_searches += 1
                if self.failed_searches == GIVE_UP_AFTER:
                    res += f"; {GIVE_UP_AFTER} empty searches, waiting for movement or a sound before the next"
            elif "found her" in res:
                self.failed_searches = 0
            return res
        return None

    # ---- search ---------------------------------------------------------------------------
    def rows(self, y0):
        """Tilts to sweep, around the camera's HOME tilt, not wherever it happens to point: level, a step
        up (sofa, bed) and a step down (floor). Live test 2026-11-02: left tilted down at 300, rows taken
        from there both looked lower still and missed the dog on the rug above. Nearest row to y0 first."""
        _, ym = self.cam.limits()
        _, fh = self.cam.frame_steps()
        lo, hi = 40, ym - 40                                 # the app's tilt margin
        try:
            home = self.cam.home()[1]
        except Exception:  # noqa: BLE001
            home = ym / 2
        rows = []
        for y in (home, home + ROW_STEP * fh, home - ROW_STEP * fh):
            y = round(max(lo, min(hi, y)))
            if all(abs(y - r) > fh * 0.3 for r in rows):
                rows.append(y)
        return sorted(rows, key=lambda r: abs(r - y0))

    def _wait_idle(self, limit=20.0):
        """Wait for the motor, ticking: a slow move can take 15 s, longer than the app tolerates silence."""
        t0 = self.clock()
        while self.cam.busy() and self.clock() - t0 < limit:
            self.tick()
            self.sleep(0.3)
        self.cam.wait_idle()

    def look(self, frames=LOOK_FRAMES):
        """Judge a few fresh frames after the camera settled: one frame can miss a curled-up dog."""
        self.tick()
        for _ in range(frames):
            box = self.find_pet(self.clock())
            if box is not None:
                return box
        return None

    def _found(self, box, t0, where, reason):
        cx, cy = (box[0] + box[2]) / 2 - 0.5, (box[1] + box[3]) / 2 - 0.5
        if max(abs(cx), abs(cy)) > 0.15:
            self._move(self.cam.aim, cx * 0.9, cy * 0.8)
        self.last_seen_t, self.last_seen_c = self.clock(), (0.0, 0.0)
        return f"search ({reason}): found her {where} in {self.clock() - t0:.0f} s"

    def _scan_to(self, b, y, t0, reason):
        """Pan (slowly) to (b, y), checking every frame on the way. Returns the result text if found."""
        x_now, y_now = self.cam.position()
        if abs(x_now - b) < 5 and abs(y_now - y) < 5:
            return None
        self._move(self.cam.goto, b, y)              # returns at once; the motor keeps going
        seen_t = self.clock()
        while self.cam.busy():
            self.tick()
            if self.user_paused():
                self.cam.stop()
                return f"search ({reason}): stopped, a person moved the camera"
            box = self.find_now(seen_t)
            seen_t = self.clock()
            if box is not None:
                self.cam.stop()
                self._wait_idle()
                self.sleep(SETTLE)
                confirm = self.look()
                if confirm is not None:
                    x, _ = self.cam.position()
                    return self._found(confirm, t0, f"while panning, near ({x}, {y})", reason)
                self._move(self.cam.goto, b, y)      # glimpse not confirmed: carry on
            self.sleep(SCAN_POLL)
        self.own_move_t = self.last_move = self.clock()
        self.sleep(SETTLE)
        box = self.look()
        if box is not None:
            return self._found(box, t0, f"at ({b}, {y})", reason)
        return None

    def search(self, reason):
        """Saved views first (stop and look), then a zigzag over two tilt rows. Along each row the
        camera pans slowly (quiet speed) and EVERY frame is checked while it moves, instead of only
        stopping and looking: a stop-and-look sweep could pass her between stops, or judge one bad
        frame. A sighting mid-pan stops the motor, waits for the picture to catch up (the stream lags
        ~0.7 s, so she is now behind the centre) and confirms on fresh frames before aiming."""
        t0 = self.clock()
        self.last_search = t0
        self.searching = True
        try:
            x0, y0 = self.cam.position()
            xm, _ = self.cam.limits()
            fw, _ = self.cam.frame_steps()
            for p in self.cam.presets():
                if self.user_paused():
                    return f"search ({reason}): stopped, a person moved the camera"
                if abs(p["x"] - x0) < fw * 0.4 and abs(p["y"] - y0) < fw * 0.25:
                    continue
                self._move(self.cam.goto, p["x"], p["y"])
                self._wait_idle()
                self.sleep(SETTLE)
                box = self.look()
                if box is not None:
                    return self._found(box, t0, f"at the saved view ({p['x']}, {p['y']})", reason)
            # Look here first, then scan this row toward the side she was last heading, then the rest of it,
            # then the second row back the other way: no long blind move before scanning starts.
            box = self.look()
            if box is not None:
                return self._found(box, t0, "where the camera already was", reason)
            heading_right = bool(self.last_seen_c and self.last_seen_c[0] > 0)   # +fx (right) = lower x
            end_a, end_b = (0, xm) if heading_right else (xm, 0)
            plan, end = [], end_a                             # zigzag: alternate direction row by row
            for n, ry in enumerate(self.rows(y0)):
                if n == 0:
                    plan += [(end_a, ry), (end_b, ry)]
                    end = end_b
                else:
                    other = end_a if end == end_b else end_b
                    plan += [(end, ry), (other, ry)]
                    end = other
            for (tx, ty) in plan:
                if self.user_paused():
                    return f"search ({reason}): stopped, a person moved the camera"
                res = self._scan_to(tx, ty, t0, reason)
                if res:
                    return res
            self._move(self.cam.goto, x0, y0)
            return f"search ({reason}): not found on {len(self.rows(y0))} rows, back to ({x0}, {y0})"
        finally:
            self.searching = False
