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
import json
import math
import os
import time

TRACK_WINDOW = 4.0      # seconds of detections used for the velocity estimate
LEAD = 1.0              # aim where she will be this many seconds ahead
EDGE = 0.20             # re-aim when the predicted centre is this far from the middle (frame = 1.0 wide)
MOVE_EVERY = 2.5        # seconds between tracking moves
CENTRE_RESTING = 0.22   # a resting pet this far off-centre for RESTING_WAIT s gets centred, once per rest. Report
                        # 20261105-201331: lying top right (centre offset +0.19, -0.25), half the picture floor;
                        # 0.36 only caught a pet already cut off at the edge.
RESTING_WAIT = 20.0
CENTRE_RESTING_ON = True    # owner steering 2026-11-02: no automatic moves near a resting dog until the motor noise was
                        # measured full band. Measured 2026-11-08 11:31 (tools/noise_study.py loudness): at QUIET_SPEED 250,
                        # 8-20 kHz +0.3 dB mean / +0.9 peak over the room, 1-8 kHz +6.3 / +14.8 (as every follow move).
                        # One small quiet move per rest.
CHASE_WINDOW = (1.0, 8.0)   # chase if she was last seen this many seconds ago...
CHASE_EDGE = 0.20       # ...off-centre toward an edge (or heading out of the frame), and was moving
CHASE_MISS_LIMIT = 2    # this many chases ending "not found, back" within CHASE_MISS_WINDOW s: stop chasing for that long
CHASE_MISS_WINDOW = 600.0   # Report 20261107-095133: 8 chases in 16 min, all missed: she was behind the sofa, and each
                        # chase took the camera away from where she reappeared. The week's 64 chases (2026-11-01..08) ended
                        # "not found" 67-77% of the time; a distance cut-off cannot tell this from a real exit (report
                        # 2026-11-05 17:47 needed the chase at the same distance), repeated misses can.
SEARCH_AFTER = 45.0     # out of sight this long (after having been followed) -> search
SEARCH_EVERY = 600.0    # at most one unprompted search per 10 min
SOUND_SEARCH_EVERY = 180.0  # a dog sound while she is not visible may search every 3 min
AFTER_FIND_MAX = 3      # after a search that FOUND her, a loss SEARCH_AFTER s later may search again, at most this many
AFTER_FIND_WINDOW = 600.0   # ...times in this window (reports 2026-11-02/04/05/08: she was found, walked off, and the
                        # 10-min SEARCH_EVERY wait meant no search for minutes while she was plainly home and moving)
GIVE_UP_AFTER = 2       # this many unprompted searches in a row found nothing: stop searching on a timer
                        # until something moves in the picture or a dog sound is heard (2026-11-02: the
                        # camera swept the empty flat every 10 min for an hour while she was out on a walk)
MEMORY_MAX_AGE = 3600.0  # retain search limits across nearby restarts; never restore a stale visual track
MOTION_CHASE_QUIET = 180.0  # after a chase set off by motion alone, no new motion chase for this long unless
                        # she is seen (2026-11-02 16:44-16:48: people moving about while she slept on the rug
                        # set off 6 motion chases, 18 motor moves in 4 min; the owner wants fewer, quieter moves)
REST_MISS_HOLD = 8.0  # she was seen lying still this recently: edge motion is not her, so no motion chase
                        # (report 2026-11-03 12:22: two hours asleep on the sofa, one missed frame right after the
                        # stream restarted, and a motion chase turned the camera 3 times away from her)
# A learned Follow policy (sim/train_policy.py) may set only these constants, clamped to these bounds. The watcher runs
# the constants above (no live rotation since 0.2.3); a policy override file can force the learned ones for a test.
POLICY_BOUNDS = {"LEAD": (0.0, 2.0), "EDGE": (0.10, 0.35), "MOVE_EVERY": (1.0, 5.0), "CHASE_EDGE": (0.10, 0.40),
                 "WEAK_GATE": (0.05, 0.25), "WEAK_MAX": (3.0, 15.0), "POST_MOVE_IGNORE": (0.8, 2.5),
                 "TRACK_WINDOW": (2.0, 6.0), "MOTION_CHASE_QUIET": (30.0, 600.0), "SEARCH_AFTER": (15.0, 120.0)}


SEARCH_STOPS = []      # a learned search order [(pan, tilt), ...] (sim/train_search.py); empty = the saved views
ROOM_FOLLOW = None     # roomfollow.RoomFollow (the room model's geometry), or None: Follow works in picture coordinates
METRIC_FOLLOW = False  # use ROOM_FOLLOW for Follow: predict her in metres, wait at the exit of hidden areas, frame by size
STREAM_LAG = 0.7       # s between a frame being taken and the detector seeing it (measured, sim/params.json lag_s)
# Delay-aware Follow (sim/follow_delay.py), off until it wins on held-out rooms. CAPTURE_TIME stamps her track with
# when each frame was taken (the detector's time minus STREAM_LAG) and skips the watcher's repeats of a stale box,
# which otherwise bend the velocity estimate. SETTLE_LEAD predicts her to when the aim will have finished (frame age,
# motor latency, travel at AIM_RATE steps/s) and keeps LEAD as a residual on top.
CAPTURE_TIME = False
SETTLE_LEAD = False
AIM_RATE = 217.0       # steps/s at QUIET_SPEED (sysid 2026-11-04: 216-218)
AIM_LATENCY = 0.1      # s from command to first motion (sysid: 50-120 ms)
# When a person's move pauses Follow and she was last seen lying still elsewhere, the camera stays where they left it
# and nothing brings it back: no motion, no detection, and the search rule needs a followed pet. In the simulator this
# is the largest controllable loss (sim/follow_delay.py: in view 0.52 with ~3 hand moves an hour, 0.68 with none).
RETURN_AFTER_PAUSE = False
RETURN_MAX_AGE = 1800.0  # s: a resting sighting older than this says too little about where she is now
RETURN_SEEN_BEFORE = 10.0  # s: return only if she was in view this recently when the person moved the camera
# Calm Follow (live walk 2026-11-08: 35 aims in 6 min, many undoing the previous one while she ambled around a person
# at 0.05-0.16 frame/s). Speed alone can't tell ambling from walking (a steady walk across the room is ~0.15 frame/s):
# direction can. Once the camera aimed the other way within AMBLE_WINDOW s, she is ambling: she must be predicted past
# EDGE_SLOW before it moves, and an aim brings her back to KEEP from the centre on her own side instead of re-centring
# her. A steady walk never reverses, so it is followed exactly as before.
CALM_FOLLOW = True     # live from 2026-11-08 (tools/follow_live.py measures it against the days before)
AMBLE_WINDOW = 20.0    # s
EDGE_SLOW = 0.32       # the threshold while ambling (EDGE otherwise)
KEEP = 0.15            # where an aim leaves her while ambling, from the centre (0 = re-centre)
HIDDEN_WAIT = 60.0     # s to hold at the exit of a hidden area before the usual lost-pet logic resumes
WALK_SPEED = 0.3       # m/s: above box jitter on a still dog (90th pct 0.09 m/s) and below her measured gait (median 0.52)
FAST_FOLLOW = True     # follow her between detector runs with the per-frame motion tracker (fasttrack.py, fed by
                       # the watcher): replay on recorded walks kept her within EDGE 38% of the time vs 17% (held-out
                       # half, sim/world/fastloop_replay.py). On since v0.3.2.
FAST_MIN_MOVE = 0.05   # the motion estimate replaces the detector's box only once it has moved this far from it
FAST_LOST_MAX = 8.0    # ...and stands in for a missing box at most this long after the last verified one
ROOM_MAP = None        # roommap.RoomMap (models/room_map.json from sim/world/find_map.py), or None: search as before
SOUND_FIND_USE = False  # look first where her sound came from (server/soundfind.py; the watcher sets this from its gate)
SOUND_HINT_MAX_AGE = 90.0   # s: a sound older than this says nothing about where she is now
SOUND_FIRST = 2         # at most this many views from the sound go before the room map's


def hour_of(t):
    return time.localtime(t).tm_hour


def merge_stops(first, then, fw, fh):
    """`first`, then each stop of `then` whose view is not mostly inside one already chosen (half a frame apart).
    Replayed on 56 recorded searches (sim/world/find_bench.py --replay): room-map stops then the learned ones had her
    in view at a stop in 48 (the learned order alone: 36), median 16 s to the view that shows her (learned: 20 s)."""
    out = [tuple(s[:2]) for s in first]
    for s in then:
        if all(abs(s[0] - o[0]) > fw / 2 or abs(s[1] - o[1]) > fh / 2 for o in out):
            out.append(tuple(s[:2]))
    return out


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
BACKS_MAX = 3           # mid-pan sightings the search goes back to, at most, per search (see _back_to_sighting)
AGREE = 0.15            # two boxes agree when their centres are within this share of the frame
SCAN_POLL = 0.2         # check a new frame this often while panning (the stream delivers 5 fps)
# Keeping the track through weak detections (ByteTrack's idea, arXiv 2110.06864): a moving, blurred dog often scores
# just under the verifier's cut-off. A dog-like box where the track predicts her keeps following her, but only
# for WEAK_MAX seconds since the last box the verifier accepted, so a weak chain can never start or keep a track
# on its own. Boxes the verifier scores near zero (her toys) are never passed in (server/watch.py WEAK_MIN).
WEAK_MAX = 8.0
WEAK_GATE = 0.12        # a weak box counts only this close (share of the frame) to where she is predicted, plus speed x time
POST_MOVE_IGNORE = 1.6  # after our own aim, frames this young may predate the move (stream lag + settle): don't re-aim on them
BOX_MAX_AGE = 10.0      # a delayed watcher cycle must not aim at a pet position from many seconds ago
TOP_EDGE_NUDGE = 0.18   # reveal a cropped upper body with at most two small tilts per continuous clipped track


class SearchHalted(Exception):
    """A running search must stop now (cancelled, or a person moved the camera); the message is its result."""


class Tracker:
    def __init__(self, cam, find_pet, clock, log=print, sleep=None, find_now=None, can_scan=None, tick=None, cancelled=None, memory=None):
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
        self.last_search_found = False
        self.chase_misses = collections.deque()  # times chases ended "not found, back"
        self.after_find = collections.deque()    # times of searches started by the after-find rule
        self.sound_hint = None        # {"t", "guess": [{"pose": [x, y], "score"}], ...}: where her last unseen sound came from
        self.search_plan = None       # what the last search_stops() chose and what the sound would have chosen
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
        self.search_phase = None      # during a search: "looking" (camera still, judging this view) or "moving"
        self.was_paused = False
        self.fast = None              # fasttrack.FastTrack the watcher feeds (or None); told about our own moves
        self.fast_used_t = -1e9       # a frozen motion estimate is not a new sighting on every watcher cycle
        self.top_edge_t, self.top_edge_n, self.top_edge_nudges = -1e9, 0, 0
        self.activity_search_used = False  # one household-noise re-arm per absence, not periodic music-driven sweeps
        self.memory = memory
        self._load_memory()

    # ---- helpers -------------------------------------------------------------------------
    def _load_memory(self):
        if not self.memory:
            return
        try:
            with open(self.memory) as f:
                state = json.load(f)
            if not isinstance(state, dict):
                return
            last, count = state.get("last_search"), state.get("failed_searches")
            if isinstance(last, bool) or not isinstance(last, (int, float)) or not math.isfinite(last):
                return
            if type(count) is not int or count < 0 or not 0 <= self.clock() - last <= MEMORY_MAX_AGE:
                return
            self.last_search, self.failed_searches = last, min(count, GIVE_UP_AFTER)
            self.activity_search_used = state.get("activity_search_used") is True
        except (OSError, ValueError, TypeError):
            pass

    def _save_memory(self):
        if not self.memory or self.last_search < 0:
            return
        tmp = f"{self.memory}.tmp"
        try:
            with open(tmp, "w") as f:
                json.dump({"last_search": self.last_search, "failed_searches": self.failed_searches,
                           "activity_search_used": self.activity_search_used}, f)
            os.replace(tmp, self.memory)
        except OSError:
            # Losing persistence must not stop camera control or the watcher heartbeat.
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def _move(self, fn, *a, shift=None):
        # During a search every motor command is a boundary: a cancel or a person's move since the last check stops
        # it here, before the camera moves (review 2026-11-04: cancelled at 8.5 s, aimed at 8.9 s, reported found).
        if self.searching and self.search_t0 is not None:
            halt = self._halt(getattr(self, "search_reason", "search"))
            if halt:
                raise SearchHalted(halt)
        fn(*a)
        if self.searching:
            self.search_phase = "moving"
        self.n_motor = getattr(self, "n_motor", 0) + 1        # every motor command, for the A/B (a search is many)
        self.own_move_t = self.last_move = self.clock()
        if self.fast is not None:
            if shift is None:
                self.fast.drop()
            else:
                self.fast.shift(*shift)
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
    def step(self, box, moving, sound=False, motion=None, weak=None, box_t=None, fast=None, activity_sound=False):
        """Call once per watcher cycle with the verified pet box (or None), the latest motion centroid (t, fx, fy)
        from the frame reader, and `weak`: dog-like boxes from this cycle that the verifier scored below its cut-off
        but not near zero. box_t: when the detector produced `box` (the watcher reuses its last answer between
        detector runs). fast: FastTrack.estimate() (FAST_FOLLOW): where the motion since the last verified box put
        her; it stands in for a box that has gone stale while she walks, or for a missed one. Returns what it did, or None."""
        t = self.clock()
        if self.user_paused():
            self.hist.clear()
            if not self.was_paused:
                self.was_paused = True
                a = getattr(self, "rest_anchor", None)
                self.pause_anchor = a if a is not None and t - self.last_seen_t <= RETURN_SEEN_BEFORE else None
                return f"paused for {USER_PAUSE / 60:.0f} min (a person moved the camera)"
            return None
        resumed, self.was_paused = self.was_paused, False
        if resumed and RETURN_AFTER_PAUSE and box is None:
            back = self._return_to_rest(t)
            if back:
                return back
        # Report 2026-11-03 18:17: the detector runs only every few seconds and the watcher passes its last box
        # in between. After our own aim that box shows her where she was BEFORE the turn; read as new, it looked
        # like she had jumped by the aim's size, so Follow aimed again and again (+0.19, +0.20, +0.22 in 7 s)
        # and turned the camera off her while she lay still. A box from before our last move says nothing new.
        if box is not None and box_t is not None and box_t < self.last_move + POST_MOVE_IGNORE:
            return None
        if box is not None and box_t is not None and t - box_t > BOX_MAX_AGE:
            return None
        fbox = self._fast_box(t, box, fast)
        if fbox is not None:
            return self._track(t, fbox, True, src="fast")
        if box is not None:
            self.last_verified_t = t
            self.box_t = box_t if box_t is not None else t
            if self.box_t > self.top_edge_t:
                clipped = moving and box[1] <= 0.01 and box[3] - box[1] >= 0.5
                self.top_edge_n = self.top_edge_n + 1 if clipped and self.box_t - self.top_edge_t <= BOX_MAX_AGE else int(clipped)
                self.top_edge_t = self.box_t
                if box[1] > 0.03:
                    self.top_edge_nudges = 0
            return self._track(t, box, moving, verified=True, t_obs=self.box_t)
        wbox = self._weak_match(t, weak)
        if wbox is not None:
            return self._track(t, wbox, moving)
        return self._lost(t, sound, motion, activity_sound)

    def _return_to_rest(self, t):
        """After a person's move: one quiet move back to where she was last seen lying still, if that was recent."""
        a, self.pause_anchor = getattr(self, "pause_anchor", None), None
        if a is None or t - a[0] > RETURN_MAX_AGE:
            return None
        try:
            here = self.cam.position()
        except Exception:  # noqa: BLE001
            return None
        if abs(here[0] - a[1]) + abs(here[1] - a[2]) < 40:
            return None
        self._move(self.cam.goto, a[1], a[2])
        self.last_aim_t = self.clock()
        return f"back to where she was ({a[1]:.0f}, {a[2]:.0f}) when a person moved the camera, {(t - a[0]) / 60:.0f} min ago"

    def _fast_box(self, t, box, fast):
        """A box at the motion tracker's estimate, when it should replace the detector's: it has followed her since
        the last verified box, and either that box is still the one shown (stale) while she moved FAST_MIN_MOVE away
        from it, or there is no box and she was verified less than FAST_LOST_MAX ago."""
        if not FAST_FOLLOW or fast is None:
            return None
        fx, fy, n, age, t_update, bw, bh = fast
        if n <= 0:
            return None
        if box is not None:
            cx, cy = (box[0] + box[2]) / 2 - 0.5, (box[1] + box[3]) / 2 - 0.5
            if math.hypot(fx - cx, fy - cy) < FAST_MIN_MOVE:
                return None
        elif t - self.last_verified_t > FAST_LOST_MAX or t_update <= self.fast_used_t:
            return None
        else:
            self.fast_used_t = t_update
        return [fx + 0.5 - bw / 2, fy + 0.5 - bh / 2, fx + 0.5 + bw / 2, fy + 0.5 + bh / 2]

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

    def _track(self, t, box, moving, src=None, verified=False, t_obs=None):
        cx, cy = (box[0] + box[2]) / 2 - 0.5, (box[1] + box[3]) / 2 - 0.5
        ts = ((t_obs if t_obs is not None else t) - STREAM_LAG) if CAPTURE_TIME else t
        if not (CAPTURE_TIME and self.hist and ts <= self.hist[-1][0] + 1e-6):   # a repeat of the same answer: no news
            self.hist.append((ts, cx, cy))
        while self.hist and ts - self.hist[0][0] > TRACK_WINDOW:
            self.hist.popleft()
        recent = t - self.last_seen_t < 3.0
        self.chase_dir, self.chase_steps, self.chase_origin = None, 0, None
        self.motion_chase_t = -1e9
        if self.failed_searches or verified and self.activity_search_used:
            self.failed_searches = 0
            if verified:
                self.activity_search_used = False
            self._save_memory()
        self.last_moving = bool(moving)
        self.last_seen_t, self.last_seen_c = t, (cx, cy)
        if verified:
            try:
                x, y = self.cam.position()
                fw, fh = self.cam.frame_steps()
                self.rest_anchor = (t, x - cx * fw, y - cy * fh)     # the pose that centres her where last verified
            except Exception:  # noqa: BLE001
                pass
        if len(self.hist) >= 2 and self.hist[-1][0] - self.hist[0][0] >= 0.8:
            vx, vy = self.velocity()
        elif recent:
            vx, vy = self.last_vel          # just after our own move: her speed in the room is unchanged
        else:
            vx, vy = 0.0, 0.0
        self.last_vel = (vx, vy)
        metric = self._metric_add(box) if METRIC_FOLLOW and ROOM_FOLLOW is not None else None
        if t - self.last_move < MOVE_EVERY or t - self.last_aim_t < POST_MOVE_IGNORE:
            return None
        if metric is not None:
            return self._metric_track(t, moving)
        px, py = cx + vx * LEAD, cy + vy * LEAD
        if SETTLE_LEAD:
            fw, fh = self.cam.frame_steps()
            age = max(0.0, t - (self.hist[-1][0] if self.hist else t))
            h = age + LEAD
            for _ in range(2):        # the move's length depends on where she'll be, and that on how long it takes
                px, py = cx + vx * h, cy + vy * h
                steps = max(abs(px) * 0.9 * fw, abs(py) * 0.8 * fh)
                h = age + AIM_LATENCY + steps / max(AIM_RATE, 1.0) + LEAD
            px, py = cx + vx * h, cy + vy * h
        speed = math.hypot(vx, vy)
        la = getattr(self, "last_aim_x", None)
        ambling = CALM_FOLLOW and la is not None and t - la[0] <= AMBLE_WINDOW and la[1] * px < 0
        edge = EDGE_SLOW if ambling else EDGE
        if (moving or speed > 0.05) and (abs(px) > edge or abs(py) > edge * 1.25):
            if ambling:         # leave her KEEP inside, on her own side: enough to keep her in, no swing across
                ax = math.copysign(max(0.0, abs(px) - KEEP), px) if abs(px) > edge else 0.0
                ay = math.copysign(max(0.0, abs(py) - KEEP), py) if abs(py) > edge * 1.25 else 0.0
                fx, fy = max(-0.6, min(0.6, ax)), max(-0.6, min(0.6, ay))
            else:
                fx, fy = max(-0.6, min(0.6, px * 0.9)), max(-0.6, min(0.6, py * 0.8))
            if abs(fx) >= 0.05:
                self.last_aim_x = (t, fx)
            self._move(self.cam.aim, fx, fy, shift=(fx, fy))
            self.last_aim_t = self.clock()
            self.offcentre_since = None
            return f"track{' (' + src + ')' if src else ''}: aim ({fx:+.2f}, {fy:+.2f}), speed {speed:.2f}/s"
        if verified and moving and self.top_edge_n >= 2 and self.top_edge_nudges < 2:
            # A whole-body box can stay centred while its head is above the image. Only fresh
            # verified moving boxes confirm this; weak/fast tracks and resting pets never do.
            x, y = self.cam.position()
            _, fh = self.cam.frame_steps()
            _, ymax = self.cam.limits()
            target_y = min(ymax, y + TOP_EDGE_NUDGE * fh)
            if target_y - y >= 5 and (ROOM_MAP is None or ROOM_MAP.has_floor(x, target_y)):
                fy = -(target_y - y) / fh
                self._move(self.cam.aim, 0.0, fy, shift=(0.0, fy))
                self.last_aim_t = self.clock()
                self.top_edge_nudges += 1
                self.top_edge_n = 0
                return f"frame upper edge: tilt ({fy:+.2f}), {self.top_edge_nudges}/2"
        # Lying down but cut off at the side of the picture: centre her once, but only if she is mostly
        # out of the picture. A motor next to a sleeping dog startles her (owner, 2026-11-02).
        if moving:
            self.centred_this_rest = False
        if CENTRE_RESTING_ON and not moving and not getattr(self, "centred_this_rest", False) \
                and max(abs(cx), abs(cy)) > CENTRE_RESTING:
            self.offcentre_since = self.offcentre_since or t
            if t - self.offcentre_since >= RESTING_WAIT:
                self.offcentre_since = None
                self.centred_this_rest = True
                self._move(self.cam.aim, cx * 0.9, cy * 0.8)
                return f"centre resting pet ({cx:+.2f}, {cy:+.2f})"
        else:
            self.offcentre_since = None
        return None

    # ---- Follow in metres (roomfollow.py) ---------------------------------------------------------
    def _metric_add(self, box):
        """Add this box's floor point to her metric track (timed when its frame was taken); None if not placeable."""
        from roomfollow import MetricTrack
        if getattr(self, "mtrack", None) is None or self.mtrack.rf is not ROOM_FOLLOW:
            self.mtrack = MetricTrack(ROOM_FOLLOW)
        try:
            pan, tilt = self.cam.position()
        except Exception:  # noqa: BLE001
            return None
        return self.mtrack.add(getattr(self, "box_t", self.clock()) - STREAM_LAG, pan, tilt, box)

    def _metric_track(self, t, moving):
        """Aim where she will be when the move lands (box age + LEAD), at her body's centre; into a hidden area,
        aim at the last visible point of her path once and wait there instead."""
        pan, tilt = self.cam.position()
        a = self.mtrack.aim(pan, tilt, t + LEAD)
        if a is None:
            return None
        fx, fy, hid, exit_o = a
        vx, vz = self.mtrack.velocity()
        speed = math.hypot(vx, vz)
        if hid and exit_o is not None:
            if t - getattr(self, "exit_wait_t", -1e9) < HIDDEN_WAIT:
                return None
            self.exit_wait_t = t
            ex, ey = max(-0.6, min(0.6, exit_o[0] * 0.9)), max(-0.6, min(0.6, exit_o[1] * 0.8))
            if max(abs(ex), abs(ey)) > EDGE * 0.5:
                self._move(self.cam.aim, ex, ey, shift=(ex, ey))
                self.last_aim_t = self.clock()
            return f"track (room): heading into a hidden area, waiting at its exit ({ex:+.2f}, {ey:+.2f})"
        if speed > WALK_SPEED and (abs(fx) > EDGE or abs(fy) > EDGE * 1.25):
            fx, fy = max(-0.6, min(0.6, fx * 0.9)), max(-0.6, min(0.6, fy * 0.8))
            self._move(self.cam.aim, fx, fy, shift=(fx, fy))
            self.last_aim_t = self.clock()
            self.offcentre_since = None
            return f"track (room): aim ({fx:+.2f}, {fy:+.2f}), {speed:.2f} m/s"
        return None

    def _chase_target(self, fx, fy):
        """Where a chase turn of (fx, fy) would point the camera, and whether it is worth making: not when an end
        stop takes most of the turn, nor onto a pose the room map shows has no floor (owner report 2026-11-05 14:56:
        three turns from her resting spot ran the camera to pan 3666, facing the wall beside the mount, and it
        stayed there; 5 of the day's 6 three-step chases ended at an end stop)."""
        try:
            cx, cy = self.cam.position()
            fw, fh = self.cam.frame_steps()
            xm, ym = self.cam.limits()
        except Exception:  # noqa: BLE001 - no pose: chase as before
            return None, True
        nx, ny = min(max(cx - fx * fw, 0), xm), min(max(cy - fy * fh, 0), ym)
        moved = abs(nx - cx) + abs(ny - cy)
        if moved < 0.4 * (abs(fx) * fw + abs(fy) * fh):
            return (cx, cy), False
        if ROOM_MAP is not None and not ROOM_MAP.has_floor(nx, ny):
            return (cx, cy), False
        return (cx, cy), True

    def _chases_missing(self, t):
        """True while CHASE_MISS_LIMIT chases ended "not found" within CHASE_MISS_WINDOW: she is probably behind
        something in view, so holding still is where she reappears."""
        while self.chase_misses and t - self.chase_misses[0] > CHASE_MISS_WINDOW:
            self.chase_misses.popleft()
        return len(self.chase_misses) >= CHASE_MISS_LIMIT

    def _chase(self, fx, fy, why):
        origin, ok = self._chase_target(fx, fy)
        if self.chase_steps == 0 or getattr(self, "chase_origin", None) is None:
            self.chase_origin = origin
        if not ok:
            return self._chase_back("the next turn shows no floor" if ROOM_MAP is not None else "at the end of its turn")
        self.chase_dir, self.chase_steps = (fx, fy), self.chase_steps + 1
        self._move(self.cam.aim, fx, fy)
        return f"chase: {why}, turned ({fx:+.2f}, {fy:+.2f})"

    def _chase_back(self, why):
        """A chase that found nothing goes back to where it was looking (where she was), instead of staying put."""
        o, self.chase_origin, self.chase_dir = getattr(self, "chase_origin", None), None, None
        self.chase_steps = 3
        if o is None:
            return None
        try:
            here = self.cam.position()
        except Exception:  # noqa: BLE001
            here = None
        if here is not None and abs(here[0] - o[0]) + abs(here[1] - o[1]) < 20:
            return f"chase: stopped, {why}"
        self._move(self.cam.goto, *o)
        self.chase_misses.append(self.clock())
        return f"chase: not found, {why}; back to where it was looking ({o[0]:.0f}, {o[1]:.0f})"

    def _lost(self, t, sound, motion=None, activity_sound=False):
        gone = t - self.last_seen_t
        if METRIC_FOLLOW and ROOM_FOLLOW is not None and t - getattr(self, "exit_wait_t", -1e9) < HIDDEN_WAIT:
            return None                                   # she went where no pose can see: wait at the exit
        # keep going the way she went: one turn rarely catches up with a trotting dog
        if self.chase_dir and self.chase_steps < 3 and t - self.last_move >= MOVE_EVERY:
            return self._chase(*self.chase_dir, f"still not in view, continuing ({self.chase_steps + 1}/3)")
        if self.chase_dir and self.chase_steps >= 3 and t - self.last_move >= MOVE_EVERY:
            return self._chase_back("3 turns")
        # chase on motion: she crossed too fast to be detected twice (or at all), but the frame reader
        # saw where the picture changed. Motion near an edge, just now, with no pet in view: look there.
        resting_here = gone < REST_MISS_HOLD and self.lay_still()
        if motion and t - motion[0] < 3.0 and motion[0] > self.own_move_t + 2.0 and motion[0] != self.chased_for \
                and not resting_here:
            mx, my = motion[1], motion[2]
            if max(abs(mx), abs(my) * 1.2) > 0.3 and t - self.last_move > MOVE_EVERY and \
                    t - self.motion_chase_t >= MOTION_CHASE_QUIET and not self._chases_missing(t):
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
            if max(abs(ex), abs(ey)) > CHASE_EDGE and not self._chases_missing(t):
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
        household = bool(activity_sound) and not self.activity_search_used and gone > SEARCH_AFTER
        while self.after_find and t - self.after_find[0] > AFTER_FIND_WINDOW:
            self.after_find.popleft()
        # The last search found her and she has been seen since: she is home and moving, so a new loss gets a new
        # search after SEARCH_AFTER instead of waiting out SEARCH_EVERY (bounded by AFTER_FIND_MAX per window).
        after_find = (gone > SEARCH_AFTER and self.last_search_found and self.last_seen_t > self.last_search
                      and t - self.last_search > SEARCH_AFTER and len(self.after_find) < AFTER_FIND_MAX)
        due = (gone > SEARCH_AFTER and t - self.last_search > SEARCH_EVERY and timer_ok) or \
              (sound and t - self.last_search > SOUND_SEARCH_EVERY) or \
              (household and t - self.last_search > SEARCH_EVERY) or after_find
        # Report 2026-11-02 18:31: after a restart the watcher ran 3.5 min on camera snapshots; the search
        # panned all 3 rows with no frames to check, "found nothing" past her on the rug by the TV, and
        # counted as an empty search. Sweeping blind is only motor noise: wait for the live stream.
        if due and not self.can_scan():
            return None
        if due:
            if household:
                self.activity_search_used = True
            if after_find and not (gone > SEARCH_AFTER and t - self.last_search > SEARCH_EVERY):
                self.after_find.append(t)
            reason = "household sound while out of sight" if household else (
                "dog sound while out of sight" if sound and gone <= SEARCH_AFTER else f"out of sight {gone:.0f} s")
            res = self.search(reason=reason)
            if "not found" in res:
                self.failed_searches += 1
                if self.failed_searches == GIVE_UP_AFTER:
                    res += f"; {GIVE_UP_AFTER} empty searches, waiting for movement or a sound before the next"
            elif "found her" in res:
                self.failed_searches = 0
            self._save_memory()
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
        starts. With a room map each row is trimmed to the pans whose view shows floor, and a row with no floor
        anywhere is left out (2026-11-05: 24% of recorded search frames showed no floor at all).
        tools/coverage.py draws this plan's coverage, so it lives here and not inside search()."""
        xm, _ = self.cam.limits()
        heading_right = bool(self.last_seen_c and self.last_seen_c[0] > 0)   # +fx (right) = lower x
        plan, flip = [], False
        for ry in self.rows(y0):
            span = self.floor_span(ry, xm)
            if span is None:
                self.log(f"search: row at tilt {ry} shows no floor in the room model: skipped")
                continue
            a, b = (span[0], span[1]) if heading_right else (span[1], span[0])
            if flip:
                a, b = b, a
            if plan and plan[-1][0] != a:                 # continue from the end nearest the previous row
                if abs(plan[-1][0] - b) < abs(plan[-1][0] - a):
                    a, b = b, a
            plan += [(a, ry), (b, ry)]
            flip = not flip
        return plan

    def floor_span(self, ry, xm):
        """(lowest, highest) pan of the row whose view shows floor, padded by half a frame; the whole row without
        a room map; None when no view of the row shows floor."""
        if ROOM_MAP is None:
            return (0, xm)
        fw, _ = self.cam.frame_steps()
        ok = [x for x in range(0, xm + 1, ROOM_MAP.ps) if ROOM_MAP.has_floor(x, ry)]
        if not ok:
            return None
        return (max(0, int(min(ok) - fw / 2)), min(xm, int(max(ok) + fw / 2)))

    def baseline_stops(self):
        """Where to stop and look without the sound, best first: the room map's views for this hour (most probable
        place of her first), then the learned stops or saved views that add a view the map's do not cover; views
        that show no floor are dropped."""
        base = [(x, y) for x, y in SEARCH_STOPS] or [(p["x"], p["y"]) for p in self.cam.presets()]
        if ROOM_MAP is None:
            return [{"x": x, "y": y} for x, y in base]
        fw, fh = self.cam.frame_steps()
        first = ROOM_MAP.stops(hour_of(self.clock()))
        order = [s for s in merge_stops(first, base, fw, fh) if ROOM_MAP.has_floor(*s)]
        return [{"x": x, "y": y, "map": any((x, y) == f[:2] for f in first)} for x, y in order]

    def search_stops(self):
        """baseline_stops(), with up to SOUND_FIRST views where her recent unseen sound came from put first when
        SOUND_FIND_USE is on. Either way the sound's guess is kept in search_plan, so every search measures it."""
        order = self.baseline_stops()
        hint = self.sound_hint
        fresh = hint is not None and 0 <= self.clock() - hint["t"] <= SOUND_HINT_MAX_AGE and hint.get("guess")
        self.search_plan = None
        if not fresh:
            return order
        xm, ym = self.cam.limits()
        guess = [(int(min(xm, max(0, g["pose"][0]))), int(min(ym - 40, max(40, g["pose"][1]))))
                 for g in hint["guess"][:SOUND_FIRST]]
        self.search_plan = {"sound_t": round(hint["t"], 2), "guess": [list(g) for g in guess],
                            "scores": [g["score"] for g in hint["guess"][:SOUND_FIRST]],
                            "baseline": [[p["x"], p["y"]] for p in order[:3]], "used": bool(SOUND_FIND_USE)}
        if not SOUND_FIND_USE:
            return order
        fw, fh = self.cam.frame_steps()
        merged = merge_stops(guess, [(p["x"], p["y"]) for p in order], fw, fh)
        flags = {(p["x"], p["y"]): p for p in order}
        return [{"x": x, "y": y, "sound": (x, y) in guess, "map": flags.get((x, y), {}).get("map", False)} for x, y in merged]

    def not_found_note(self):
        """After a failed search: what the room model and her history say instead of a flat 'not found'."""
        if ROOM_MAP is None:
            return ""
        h = hour_of(self.clock())
        away = self.clock() - self.last_seen_t
        places = [p for p in ROOM_MAP.places(h) if p.get("p", 0) >= 0.15]
        bits = []
        if away >= 600:
            bits.append(f"probably out of the room (not seen for {away / 60:.0f} min)")
        if places:
            p = places[0]
            where = "on the open floor" if p["place"] == "the open floor" else f"at the {p['place']}"
            hidden = f", {p['hidden']:.0%} of it out of the camera's sight" if p.get("hidden", 0) >= 0.05 else ""
            bits.append(f"{'if she is in, ' if bits else ''}at {h:02d}:00 she is most often {where} "
                        f"({p['p']:.0%}{hidden}), not confirmed")
        return ("; " + "; ".join(bits)) if bits else ""

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
        if self.searching:
            self.search_phase = "looking"
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
        self.last_search_found = True
        self.failed_searches = 0
        self.activity_search_used = False
        self._save_memory()
        return f"search ({reason}): found her {where} in {self.clock() - t0:.0f} s"

    def _scan_to(self, b, y, t0, reason):
        """Pan (slowly) to (b, y), checking every frame on the way. Returns the result text if found."""
        x_now, y_now = self.cam.position()
        if abs(x_now - b) < 5 and abs(y_now - y) < 5:
            return None
        self._move(self.cam.goto, b, y)              # returns at once; the motor keeps going
        seen_t = t_start = self.clock()
        while self.cam.busy():
            self.tick()
            halt = self._halt(reason)
            if halt:
                self.cam.stop()
                return halt
            box = self.find_now(seen_t)
            seen_t = self.clock()
            if box is not None:
                t_stop = self.clock()
                self.cam.stop()
                self._wait_idle()
                self._back_to_sighting(box, x_now, y_now, t_start, t_stop)
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

    def _back_to_sighting(self, box, x0, y0, t_start, t_stop):
        """A mid-pan sighting comes from a frame taken some time ago (its delay, detection, the stop): by the time the
        motor stops the camera has panned on, at ~215 steps/s about 0.3 frame per second, and she can be out of the
        picture (report 2026-11-04 14:45: three sure sightings, each look at the stop showed the next view and the search
        ended "not found"). Place her where the camera pointed when the frame was taken (linear motion from the move's
        start to the stop), count it as one sure sighting, and aim back at her if the camera has overshot."""
        x1, y1 = self.cam.position()
        t_seen = getattr(box, "captured", None)
        if t_seen is None:
            t_seen = t_stop
        span = max(1e-6, t_stop - t_start)
        f = min(1.0, max(0.0, (t_seen - t_start) / span))
        xc, yc = x0 + (x1 - x0) * f, y0 + (y1 - y0) * f
        fw, fh = self.cam.frame_steps()
        cx, cy = (box[0] + box[2]) / 2 - 0.5, (box[1] + box[3]) / 2 - 0.5
        rx, ry = xc - cx * fw, yc - cy * fh
        seen = getattr(self, "search_hits", None)
        if seen is not None and getattr(box, "p", 1.0) >= GLIMPSE_P:
            seen.append((rx, ry, self.clock()))
        backs = getattr(self, "search_backs", None)
        if backs is None:
            backs = self.search_backs = []
        # At most once per place and BACKS_MAX per search: a look that fails after going back must not send the scan
        # past her again and back again forever (it did, in the glimpse test, before this bound).
        if any(abs(bx - rx) <= 0.5 * fw and abs(by - ry) <= 0.5 * fh for bx, by in backs) or len(backs) >= BACKS_MAX:
            return
        if abs(rx - x1) > 0.15 * fw or abs(ry - y1) > 0.15 * fh:
            backs.append((rx, ry))
            self.log(f"search: the sighting was at ({rx:.0f}, {ry:.0f}), the camera stopped at ({x1}, {y1}): going back")
            self._move(self.cam.goto, rx, ry)
            self._wait_idle()

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
        self.last_search_found = False             # _found() sets it: any search, the owner's Find included
        self._save_memory()
        self.search_t0 = t0
        self.searching = True
        self.search_reason = reason
        self.search_hits = []
        self.search_backs = []            # sure sightings of this search, room coordinates (pan, tilt steps)
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
            stops = self.search_stops()          # room map for this hour, then the learned order or the saved views
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
                kind = ("the view her sound came from" if p.get("sound") else "the room map's stop" if p.get("map")
                        else ("the learned stop" if SEARCH_STOPS else "the saved view"))
                where = f"at {kind} ({p['x']}, {p['y']})"
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
            return (f"search ({reason}): not found on {len(self.sweep_plan(y0)) // 2} rows, back to {label} ({bx}, {by})"
                    + self.not_found_note())
        except SearchHalted as h:
            return str(h)
        finally:
            self.searching = False
            self.search_phase = None
            self.search_t0 = None
            self.search_hits = None
