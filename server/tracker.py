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
REST_MISS_HOLD = 8.0  # she was seen lying still this recently: edge motion is not her, so no motion chase
                        # (report 2026-11-03 12:22: two hours asleep on the sofa, one missed frame right after the
                        # stream restarted, and a motion chase turned the camera 3 times away from her)
# A learned Follow policy (sim/train_policy.py) may set only these constants, clamped to these bounds; the watcher
# alternates it with the constants above in 2-hour blocks (live A/B) and logs which one ran (track.policy).
POLICY_BOUNDS = {"LEAD": (0.0, 2.0), "EDGE": (0.10, 0.35), "MOVE_EVERY": (1.0, 5.0), "CHASE_EDGE": (0.10, 0.40),
                 "WEAK_GATE": (0.05, 0.25), "WEAK_MAX": (3.0, 15.0), "POST_MOVE_IGNORE": (0.8, 2.5),
                 "TRACK_WINDOW": (2.0, 6.0), "MOTION_CHASE_QUIET": (30.0, 600.0), "SEARCH_AFTER": (15.0, 120.0)}


SEARCH_STOPS = []      # a learned search order [(pan, tilt), ...] (sim/train_search.py); empty = the saved views


def policy_stops(doc, xm=3700, ym=1000, most=10):
    """A policy document's learned search stops, validated: pairs of numbers, clamped inside the motor range."""
    out = []
    for p in (doc or {}).get("search_stops") or []:
        try:
            x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError):
            continue
        if x == x and y == y:                       # not NaN
            out.append((int(min(xm, max(0, x))), int(min(ym - 40, max(40, y)))))
        if len(out) == most:                        # validate first, then cap: bad entries cannot crowd out good ones
            break
    return out


def policy_constants(doc):
    """The whitelisted, clamped constants of a policy document ({"constants": {...}}); anything else is dropped."""
    out = {}
    for k, v in ((doc or {}).get("constants") or {}).items():
        if k in POLICY_BOUNDS and isinstance(v, (int, float)) and v == v:
            lo, hi = POLICY_BOUNDS[k]
            out[k] = float(min(hi, max(lo, v)))
    return out


REST_NAMES = ("default", "corridor")   # a saved view named with one of these is where a failed search waits
USER_PAUSE = 180.0     # a move by a person pauses following/searching this long
SETTLE = 0.9            # seconds after a move before a frame is trusted (motion blur, stream latency)
ROW_STEP = 0.55         # the second search row is this many frame heights toward the floor
LOOK_FRAMES = 4         # frames judged when stopped (one frame can miss her)
CONFIRM_FRAMES = 2      # ...and this many of them must show her in the same place before "found her":
                        # 2026-11-02/03, 31 of 43 finds were at night, where one infrared frame of a clothes pile
                        # or a slipper could pass the verifier and end the search. Two agreeing frames out of four
                        # still tolerate one frame that misses a curled-up dog.
GLIMPSE_P = 0.97        # a single verified box is followed up (aim + look again) only this sure. Fitted by eye on the 12
                        # failed searches 2026-11-03/04 that had one: the 8 real ones scored >= 0.977, the 4 false ones
                        # (one toy under the stool at night) <= 0.956. 12 cases, no held-out set: revisit with more.
AGREE = 0.15            # two boxes agree when their centres are within this share of the frame
SCAN_POLL = 0.2         # check a new frame this often while panning (the stream delivers 5 fps)
# Keeping the track through weak detections (ByteTrack's idea, arXiv 2110.06864): a moving, blurred dog often scores
# just under the verifier's cut-off. A dog-like box where the track predicts her keeps following her, but only
# for WEAK_MAX seconds since the last box the verifier accepted, so a weak chain can never start or keep a track
# on its own. Boxes the verifier scores near zero (her toys) are never passed in (server/watch.py WEAK_MIN).
WEAK_MAX = 8.0
WEAK_GATE = 0.12        # a weak box counts only this close (share of the frame) to where she is predicted, plus speed x time
POST_MOVE_IGNORE = 1.6  # after our own aim, frames this young may predate the move (stream lag + settle): don't re-aim on them


class SearchHalted(Exception):
    """A running search must stop now (cancelled, or a person moved the camera); the message is its result."""


class Tracker:
    def __init__(self, cam, find_pet, clock, log=print, sleep=None, find_now=None, can_scan=None, tick=None, cancelled=None):
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
        # cancelled(since) -> True once the owner pressed "Stop looking" after `since` (report 2026-11-03 07:31:
        # a 2-min sweep they asked for could not be stopped). Checked wherever user_paused() is.
        self.cancelled = cancelled or (lambda since: False)
        self.search_t0 = None
        self.hist = collections.deque()
        self.last_seen_t = clock()     # starting up counts as "just lost": an unseen pet still gets searched for
        self.last_seen_c = None
        self.last_vel = (0.0, 0.0)
        self.last_move = -1e9
        self.own_move_t = -1e9
        self.last_verified_t = -1e9   # last box the verifier accepted (weak boxes only extend a track from here)
        self.last_aim_t = -1e9        # our last small tracking move (frames just after it may predate it)
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
    def _move(self, fn, *a, shift=None):
        # During a search every motor command is a boundary: a cancel or a person's move since the last check stops
        # it here, before the camera moves (review 2026-11-04: cancelled at 8.5 s, aimed at 8.9 s, reported found).
        if self.searching and self.search_t0 is not None:
            halt = self._halt(getattr(self, "search_reason", "search"))
            if halt:
                raise SearchHalted(halt)
        fn(*a)
        self.n_motor = getattr(self, "n_motor", 0) + 1        # every motor command, for the A/B (a search is many)
        self.own_move_t = self.last_move = self.clock()
        if shift is None:
            self.hist.clear()                # a long move: positions in the old view no longer apply
        else:
            # A small aim of (fx, fy) moves the picture by the same amount: where she was at offset c in the old view
            # she is at c - (fx, fy) now. Shifting the history keeps her speed instead of forgetting it.
            fx, fy = shift
            self.hist = collections.deque((t, x - fx, y - fy) for t, x, y in self.hist)
            if self.last_seen_c:
                self.last_seen_c = (self.last_seen_c[0] - fx, self.last_seen_c[1] - fy)

    def user_paused(self):
        try:
            m = self.cam.moved_at()
        except OSError:
            return False
        return m > max(self.own_move_t + 1.5, self.started_t) and self.clock() - m < USER_PAUSE

    def _halt(self, reason):
        """Why the running search must stop now (a person moved the camera, or the owner cancelled), or None."""
        if self.search_t0 is not None and self.cancelled(self.search_t0):
            return f"search ({reason}): stopped, you cancelled it"
        if self.user_paused():
            return f"search ({reason}): stopped, a person moved the camera"
        return None

    def velocity(self):
        if len(self.hist) < 2:
            return 0.0, 0.0
        (t0, x0, y0), (t1, x1, y1) = self.hist[0], self.hist[-1]
        dt = t1 - t0
        return ((x1 - x0) / dt, (y1 - y0) / dt) if dt >= 0.8 else (0.0, 0.0)

    # ---- every cycle ---------------------------------------------------------------------
    def step(self, box, moving, sound=False, motion=None, weak=None, box_t=None):
        """Call once per watcher cycle with the verified pet box (or None), the latest motion centroid (t, fx, fy)
        from the frame reader, and `weak`: dog-like boxes from this cycle that the verifier scored below its cut-off
        but not near zero. box_t: when the detector produced `box` (the watcher reuses its last answer between
        detector runs). Returns what it did, or None."""
        t = self.clock()
        if self.user_paused():
            self.hist.clear()
            if not self.was_paused:
                self.was_paused = True
                return f"paused for {USER_PAUSE / 60:.0f} min (a person moved the camera)"
            return None
        self.was_paused = False
        # Report 2026-11-03 18:17: the detector runs only every few seconds and the watcher passes its last box
        # in between. After our own aim that box shows her where she was BEFORE the turn; read as new, it looked
        # like she had jumped by the aim's size, so Follow aimed again and again (+0.19, +0.20, +0.22 in 7 s)
        # and turned the camera off her while she lay still. A box from before our last move says nothing new.
        if box is not None and box_t is not None and box_t < self.last_move + POST_MOVE_IGNORE:
            return None
        if box is not None:
            self.last_verified_t = t
            return self._track(t, box, moving)
        wbox = self._weak_match(t, weak)
        if wbox is not None:
            return self._track(t, wbox, moving)
        return self._lost(t, sound, motion)

    def _weak_match(self, t, weak):
        """The weak box closest to where the track predicts her, if the track is alive and the box is close enough."""
        if not weak or t - self.last_verified_t > WEAK_MAX or self.last_seen_c is None or t - self.last_seen_t > 3.0:
            return None
        dt = t - self.last_seen_t
        vx, vy = self.last_vel
        px, py = self.last_seen_c[0] + vx * dt, self.last_seen_c[1] + vy * dt
        gate = WEAK_GATE + math.hypot(vx, vy) * dt
        best = None
        for b in weak:
            cx, cy = (b[0] + b[2]) / 2 - 0.5, (b[1] + b[3]) / 2 - 0.5
            d = math.hypot(cx - px, cy - py)
            if d <= gate and (best is None or d < best[0]):
                best = (d, b)
        return best[1] if best else None

    def lay_still(self, span=3.0, spread=0.06):
        """She was lying still when last seen: her track covers at least `span` s and her centre stayed within
        `spread` of the frame. Judged from where she was, not from one frame's noisy moving flag: a walking dog's
        last sighting is often flagged still (follow benchmark, 2026-11-03), and that must not switch off the chase."""
        if len(self.hist) < 2 or self.hist[-1][0] - self.hist[0][0] < span:
            return False
        xs, ys = [h[1] for h in self.hist], [h[2] for h in self.hist]
        return max(xs) - min(xs) <= spread and max(ys) - min(ys) <= spread

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
        if t - self.last_move < MOVE_EVERY or t - self.last_aim_t < POST_MOVE_IGNORE:
            return None
        px, py = cx + vx * LEAD, cy + vy * LEAD
        speed = math.hypot(vx, vy)
        if (moving or speed > 0.05) and (abs(px) > EDGE or abs(py) > EDGE * 1.25):
            fx, fy = max(-0.6, min(0.6, px * 0.9)), max(-0.6, min(0.6, py * 0.8))
            self._move(self.cam.aim, fx, fy, shift=(fx, fy))
            self.last_aim_t = self.clock()
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
        resting_here = gone < REST_MISS_HOLD and self.lay_still()
        if motion and t - motion[0] < 3.0 and motion[0] > self.own_move_t + 2.0 and motion[0] != self.chased_for \
                and not resting_here:
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

    def sweep_plan(self, y0):
        """The pan targets of the zigzag after the saved views. Scan this row toward the side she was last
        heading, then the rest of it, then the next row back the other way: no long blind move before scanning
        starts. tools/coverage.py draws this plan's coverage, so it lives here and not inside search()."""
        xm, _ = self.cam.limits()
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
        return plan

    def _wait_idle(self, limit=20.0):
        """Wait for the motor, ticking: a slow move can take 15 s, longer than the app tolerates silence."""
        t0 = self.clock()
        while self.cam.busy() and self.clock() - t0 < limit:
            self.tick()
            if self.search_t0 is not None and self.cancelled(self.search_t0):
                self.cam.stop()                      # the caller's next check returns 'you cancelled it'
                break
            self.sleep(0.3)
        self.cam.wait_idle()

    def look(self, frames=LOOK_FRAMES, need=CONFIRM_FRAMES):
        """Judge fresh frames after the camera settled and return her box only when `need` of up to `frames`
        frames show her in about the same place. One frame can miss a curled-up dog; one frame can also
        mistake a slipper for her at night. Stops as soon as she is confirmed, or can no longer be."""
        self.tick()
        hits = []
        self.last_glimpse = None
        pose = None
        # One boundary for the whole look: every frame must be taken after the camera settled, but each later frame
        # need only be a NEW one, not wait the transport delay again (that wait came back per frame: ~2.7 s for
        # two confirmations on the substream).
        boundary = self.clock()
        for n in range(frames):
            if self.searching and self.search_t0 is not None:
                halt = self._halt(getattr(self, "search_reason", "search"))
                if halt:
                    raise SearchHalted(halt)
            box = self.find_pet(boundary)
            if box is not None:
                c = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)
                agree = [h for h in hits if abs(h[0][0] - c[0]) <= AGREE and abs(h[0][1] - c[1]) <= AGREE]
                if len(agree) + 1 >= need:
                    return box
                hits.append((c, box))
                if getattr(box, "p", 1.0) >= GLIMPSE_P:
                    self.last_glimpse = box
                    # Across looks: a sure sighting where an earlier look of this search had one (room coordinates,
                    # the camera is stopped) confirms her. 2026-11-03 20:54 and 2026-11-04 09:35 saw her 3 and 8
                    # times, never twice within one look, and ended "not found".
                    pose = pose or self.cam.position()
                    fw, fh = self.cam.frame_steps()
                    rx, ry = pose[0] - (c[0] - 0.5) * fw, pose[1] - (c[1] - 0.5) * fh
                    seen = getattr(self, "search_hits", None)
                    if seen is not None:
                        if any(abs(h[0] - rx) <= AGREE * fw and abs(h[1] - ry) <= AGREE * fh for h in seen):
                            self.log(f"search: sighting at ({rx:.0f}, {ry:.0f}) agrees with an earlier look: confirmed")
                            return box
                        seen.append((rx, ry, self.clock()))
            best = max([sum(1 for k in hits if abs(k[0][0] - h[0][0]) <= AGREE and abs(k[0][1] - h[0][1]) <= AGREE)
                        for h in hits] or [0])
            if best + (frames - n - 1) < need:       # even if every remaining frame agrees, not enough
                return None
        return None

    def _glimpse(self, t0, where, reason):
        """A verified sighting that did not reach CONFIRM_FRAMES: aim at it, so she is fully in the picture, and
        confirm on fresh frames as usual (the confirmation rule itself is unchanged). Report 2026-11-04 17:40: she
        lay at the top edge of a stop, cut off; the verifier gave 0.977 on one of 4 frames, the others missed, and
        the search moved on while she was in view. Returns the result text, or None."""
        g = getattr(self, "last_glimpse", None)
        if g is None:
            return None
        cx, cy = (g[0] + g[2]) / 2 - 0.5, (g[1] + g[3]) / 2 - 0.5
        self.log(f"search: one glimpse at ({cx:+.2f}, {cy:+.2f}), not yet confirmed: aiming at it to look again")
        self._move(self.cam.aim, cx * 0.9, cy * 0.8)
        self._wait_idle()
        self.sleep(SETTLE)
        box = self.look()
        if box is not None:
            return self._found(box, t0, f"{where}, after aiming at a glimpse of her", reason, centre=False)
        return None

    def _found(self, box, t0, where, reason, centre=True):
        cx, cy = (box[0] + box[2]) / 2 - 0.5, (box[1] + box[3]) / 2 - 0.5
        if centre and max(abs(cx), abs(cy)) > 0.15:
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
            halt = self._halt(reason)
            if halt:
                self.cam.stop()
                return halt
            box = self.find_now(seen_t)
            seen_t = self.clock()
            if box is not None:
                self.cam.stop()
                self._wait_idle()
                self.sleep(SETTLE)
                confirm = self.look()
                x, _ = self.cam.position()
                if confirm is not None:
                    return self._found(confirm, t0, f"while panning, near ({x}, {y})", reason)
                res = self._glimpse(t0, f"while panning, near ({x}, {y})", reason)
                if res:
                    return res
                self._move(self.cam.goto, b, y)      # glimpse not confirmed: carry on
            self.sleep(SCAN_POLL)
        self.own_move_t = self.last_move = self.clock()
        self.sleep(SETTLE)
        box = self.look()
        if box is not None:
            return self._found(box, t0, f"at ({b}, {y})", reason)
        return self._glimpse(t0, f"at ({b}, {y})", reason)

    def rest_view(self):
        """The saved view a failed search parks at: the first one whose name contains a REST_NAMES word
        (the owner saves it from the app, e.g. "Corridor" or "Default"), else None (stay where it was)."""
        try:
            presets = self.cam.presets()
        except Exception:  # noqa: BLE001
            return None
        for word in REST_NAMES:
            for p in presets:
                if word in str(p.get("name", "")).lower():
                    return p
        return None

    def search(self, reason):
        """The current view first (no move), then saved views (stop and look), then a zigzag over the tilt rows. Along each row the
        camera pans slowly (quiet speed) and EVERY frame is checked while it moves, instead of only
        stopping and looking: a stop-and-look sweep could pass her between stops, or judge one bad
        frame. A sighting mid-pan stops the motor, waits for the picture to catch up (the stream lags
        ~0.7 s, so she is now behind the centre) and confirms on fresh frames before aiming."""
        t0 = self.clock()
        self.last_search = t0
        self.search_t0 = t0
        self.searching = True
        self.search_reason = reason
        self.search_hits = []            # sure sightings of this search, room coordinates (pan, tilt steps)
        try:
            x0, y0 = self.cam.position()
            fw, _ = self.cam.frame_steps()
            # Look here before moving anywhere (2026-11-02 13:45 and 18:31: asleep in view, a detector dropout
            # or a restart started the search, it drove to the saved views first and came back "not found";
            # she was detected in the same view 38 s / 9 s later). Found here: no motor move at all, not
            # even to centre her, since a motor next to a resting dog is what the owners want avoided.
            box = self.look()
            if box is None and getattr(self, "last_glimpse", None) is not None:
                box = self.look(frames=2 * LOOK_FRAMES)      # she may be here: look longer before any move
            if box is not None:
                return self._found(box, t0, "where the camera already was", reason, centre=False)
            # Still only glimpses: aim at her rather than drive off to the saved views (a smaller move than leaving)
            res = self._glimpse(t0, "where the camera already was", reason)
            if res:
                return res
            stops = [{"x": x, "y": y} for x, y in SEARCH_STOPS] or self.cam.presets()   # learned order, else saved views
            for p in stops:
                halt = self._halt(reason)
                if halt:
                    return halt
                if abs(p["x"] - x0) < fw * 0.4 and abs(p["y"] - y0) < fw * 0.25:
                    continue
                self._move(self.cam.goto, p["x"], p["y"])
                self._wait_idle()
                halt = self._halt(reason)
                if halt:
                    return halt
                self.sleep(SETTLE)
                box = self.look()
                where = f"at {'the learned stop' if SEARCH_STOPS else 'the saved view'} ({p['x']}, {p['y']})"
                if box is not None:
                    return self._found(box, t0, where, reason)
                res = self._glimpse(t0, where, reason)
                if res:
                    return res
            for (tx, ty) in self.sweep_plan(y0):
                halt = self._halt(reason)
                if halt:
                    return halt
                res = self._scan_to(tx, ty, t0, reason)
                if res:
                    return res
            # Not found: wait at the view that sees the most (owner, report 2026-11-03 08:31: "make default
            # position the one with the most view (corridor)"). A chase had left the camera at the pan end stop
            # (0, 471), so "back to where it was" parked it on a corner of the living room until the next motion.
            rest = self.rest_view()
            if rest:
                bx, by, label = rest["x"], rest["y"], f"the saved view '{rest['name']}'"
            else:
                bx, by, label = x0, y0, "where it was"
            self._move(self.cam.goto, bx, by)
            # Report 2026-11-03 16:10: the slow drive home outlasts the frame reader's 4 s blind window, so the
            # picture sliding past read as "something moved since the last sweep" and re-armed the timer: a full
            # sweep every 10 min for 1.5 h with nobody there. Count our own move as lasting until the motor stops.
            self._wait_idle()
            self.own_move_t = self.last_move = self.clock()
            return f"search ({reason}): not found on {len(self.rows(y0))} rows, back to {label} ({bx}, {by})"
        except SearchHalted as h:
            return str(h)
        finally:
            self.searching = False
            self.search_t0 = None
            self.search_hits = None
