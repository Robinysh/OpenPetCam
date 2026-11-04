#!/usr/bin/env python3
"""petcam-watch: watches the camera around the clock on a $5 box.

Reads the camera's 640x360 substream (decoded at 5 fps, motion checked on every frame; camera
snapshots as the fallback). Every 3 s, or within ~1 s of motion: check the picture is not frozen,
measure motion, run a tiny detector plus the trained "is it really the pet?" verifier, and keep the pet's state. Writes to SQLite
(/var/lib/petcam/petcam.db): raw observations, per-minute aggregates (kept forever, they are the
long-term memory), and events with short clips cut from MediaMTX's rolling recording.

Also listens to the audio for loud sounds / bark-like bursts, saves one snapshot a minute
for a daily timelapse, and runs nightly housekeeping. Touches /var/lib/petcam/heartbeat on
every healthy cycle for the absence alerter.
"""
import collections
import datetime as dt
import io
import json
import math
import os
import queue
import shutil
import sqlite3
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
from PIL import Image

sys.path.insert(0, "/opt/petcam")
import app as camapi  # noqa: E402  - camera session login + motor/status helpers
import petconf  # noqa: E402
from detector import Detector  # noqa: E402
from tracker import Tracker  # noqa: E402
import tracker as T  # noqa: E402

STATE = petconf.STATE
DB = f"{STATE}/petcam.db"
SNAPS = f"{STATE}/snaps"
CLIPS = f"{STATE}/clips"
TIMELAPSE = f"{STATE}/timelapse"
HIGHLIGHTS = f"{STATE}/highlights"
HEARTBEAT = f"{STATE}/heartbeat"
LIVE = f"{STATE}/live.json"          # current status for the web app
MOVED = f"{STATE}/moved_at"          # web app touches this when it moves the camera
PRIVACY = f"{STATE}/privacy"         # "1" while the camera is paused (privacy mode)
SETTINGS = f"{STATE}/settings.json"  # {"follow": bool, "talk_volume": int} written by the web app
FLAP_LOCK_FILE = f"{STATE}/daynight_lock_until"   # survives restarts, so a lock always ends
HOUSE_FILE = f"{STATE}/housekeeping_day"            # date of the last nightly housekeeping run
MODEL = "/opt/petcam/models/yolox_nano.onnx"
YAMNET_MODEL = "/opt/petcam/models/yamnet.tflite"
VERIFIER = "/opt/petcam/models/verifier.json"        # from tools/bootstrap_pet.py
FOLLOW_POLICY = "/opt/petcam/models/follow_policy.json"   # from sim/train_policy.py (learned on this camera's simulator)
FOLLOW_PPO = "/opt/petcam/models/follow_ppo.onnx"         # a recurrent PPO policy (sim/train_ppo.py), server/ppo_follow.py
POLICY_BLOCK_S = 7200          # live A/B: the learned constants and today's alternate in 2-hour blocks
POLICY_OVERRIDE = f"{STATE}/policy_override"   # "<arm> <unix expiry>": force one arm for a live smoke test, expires itself
REPORTS = f"{STATE}/reports"                         # "Something wrong?" reports from the app
CANDIDATES = f"{STATE}/candidates"                  # frames with dog-like things, for the next bootstrap
SEARCHES = f"{STATE}/searches"                      # what each search saw, to label Find right/wrong (tools/search_review.py)
SEARCH_FRAMES_MAX = 24                              # frames kept per search: the first at every stop, and every sighting
ZONES = f"{STATE}/zones.json"                       # written by the web app
ZONE_POSE_TOL = (40, 25)                            # pan/tilt steps: camera counts as "at the zone's view"
ZONE_GAP = 20                                       # s without her in the zone that ends a visit
YAMNET_CLASSES = "/opt/petcam/models/yamnet_class_map.csv"
DOG_SOUND_THRESHOLD = 0.55    # PROVISIONAL, see SoundModel; refit on the 👍/👎 labels from the app
LOUD_NON_DOG_DB = -28.0       # non-dog sounds become a moment only when this loud (a crash, an alarm)
DOG_SOUND_KINDS = set(petconf.DOG_SOUND_KINDS)

SUB_URL = "rtsp://127.0.0.1:8554/sub"   # the camera's 640x360 stream, relayed by MediaMTX
MAIN_URL = "rtsp://127.0.0.1:8554/cam"  # the 1080p main stream (MediaMTX pulls it all day for the recording)
# How old the picture is when a decoded frame reaches the watcher (2026-11-04: motor command -> first moving frame,
# minus the ~0.1 s motor start; ffmpeg low-delay flags and the MediaMTX relay changed nothing, the camera's
# substream pipeline itself holds ~0.6 s more, and MORE at higher fps: 10 fps 1.3 s, 25 fps 2.5 s).
STREAM_LAG = {"sub": 0.90, "main": 0.26, "push": 0.55, "snapshot": 1.0}
# JPEG push (2026-11-04): the camera's own ch1 JPEG snapshots are fresh (the delay of the ch1 H.264 stream is added
# after them, in its encoder/RTSP output, and no encoder setting removed it), so while live frames are wanted the camera
# streams snapshots to the watcher over one ssh channel. On the same moves: 0.95 s vs 1.40 s for the substream; JPEG
# decode costs the server less than H.264; ~0.8 Mbit/s of home upload while it runs. Falls back to the substream.
PUSH_ENABLED = True
PUSH_FPS = 5
PUSH_QUALITY = 60
PUSH_MAX_S = 900               # the camera-side loop ends by itself after this; the watcher simply starts a new one
PUSH_RETRY_S = 600             # after a failed push, use the substream this long
PUSH_MAX_BYTES = 400_000       # a ch1 JPEG is ~20-40 KB; a larger length means the framing is broken
PUSH_SHORT_S = 60              # a push session that ends sooner than this (other than by our own stop) counts as a failure
PUSH_SHORT_MAX = 3             # ...and this many in a row fall back to the substream
FRAME_PROGRESS_S = 10          # no complete frame for this long: the source is stuck (connected but silent); restart it
# Fast mode: while Follow is engaged with her (seen moving, or a search), frames come from the main stream (0.26 s
# instead of 0.9 s old) and the detector runs every FAST_CYCLE. Costs ~9.4% of a core to decode (1080p25 with the
# loop filter skipped, scaled fast_bilinear) plus ~10% for detecting at 2 Hz, so a governor turns it off whenever
# the box shows CPU steal (its burst credits are spent) and for FAST_COOLDOWN after.
# Main stream: every 25 fps frame, not decimated. ffmpeg's fps=5 filter alone added ~450 ms of lag here (2026-11-04,
# interleaved on the same moves: command -> first moving frame 365 ms at 25 fps, 809 ms through fps=5, 1112 ms on the
# substream), and scaling all 25 costs about the same as 5 (9.8% vs 9.5% of a core): decoding dominates.
MAIN_VF = "scale={w}:{h}:flags=fast_bilinear"
MAIN_FPS = 25
FAST_ENABLED = False          # off until the box has CPU to spare for it: its average use already sits at or above
                              # the burst baseline, so fast mode would only spend credits it does not earn (2026-11-04:
                              # credits at 0, 71% steal, the main stream decoded at 6.6 of 25 fps)
FAST_HOLD = 20                # s fast mode stays on after the last reason
FAST_CYCLE = 0.5              # s between watcher cycles (and detector runs) in fast mode
FAST_STEAL_MAX = 0.08         # steal share of CPU time over the last minute that switches fast mode off
FAST_COOLDOWN = 600
STREAM_HOLD = 90              # seconds the live decoder keeps running after the last reason to watch closely
FRAME_FPS = 5                 # decoded frames per second for motion (the detector runs far less often)
STREAM_STALL = 90             # seconds of no live frames (while snapshots work) before the streamer is restarted
CAM_MOVE_GRACE = 10.0         # s around a camera move in which entering/leaving the picture is not an event
EPISODE_GAP = 60.0            # dog sounds closer than this are one barking episode (one event)
TALKING = f"{STATE}/talking"      # "start until": the app is playing the owner's voice on the speaker
MIN_CYCLE = 1.0               # at most one full cycle (detector, DB row) per second, even while she moves
PERIOD = 3.0                  # seconds between snapshots (2 s kept the box over its CPU baseline)
PRESENT_SCORE = 0.30          # animal score that counts as "the pet is here" (see README for evidence)
ABSENT_SCORE = 0.15
LEAVE_AFTER = 180             # seconds without her before "left"
UNSEEN_AFTER = 30             # seconds without a detection before she counts as out of sight (not resting)
APPEAR_AFTER = 180            # seconds she must have been gone before "appeared" fires
FREEZE_AFTER = 90             # seconds of a bit-identical picture before the encoder is restarted
OFFLINE_AFTER = 180           # seconds of failed snapshots before "camera offline"
CLIP_BEFORE, CLIP_LEN = 8, 30
DETECT_EVERY = 8              # seconds between detector runs while she is active, followed or searched for
DETECT_STILL = 30             # ...and while the picture is still and nobody moved for a minute (profiled 2026-11-02:
                              # the detector was ~18% of a core on a box allowed ~10% sustained; motion still
                              # triggers a run at once, so arrivals are seen as fast as before)
LOUD_PER_HOUR = 6             # at most this many "loud sound" moments per hour (barking is not capped)
FLAP_SWITCHES = 4             # this many day/night switches within FLAP_WINDOW = flapping
FLAP_WINDOW = 600
FLAP_LOCK = 7200              # lock to night this long, then back to auto
AUDIO_LOST = 90                # s without audio samples (mic on, not paused) before the streamer is restarted
IMAGING_EVERY = 20            # seconds between reads of the camera's day/night state
BLOCKED_SHARPNESS = 60.0      # Laplacian variance (160x90 gray). Night 1: normal views >= 132, a wall
                              # at close range 14-30 (noisy dark frames up to 176, hence the 10-min rule)
BLOCKED_WINDOW = 600          # judge "view blocked" over this many seconds
BLOCKED_FRACTION = 0.8
WEAK_MIN = 0.15               # a dog-like box the verifier scored at least this (but under its cut-off) may keep an
                              # existing Follow track going (tracker.py WEAK_MAX); her toys score ~0.00 and never do
MOTION_DETECT = 1.6           # mean pixel change (0-255, 160x90 gray) that triggers a detector run; IR grain alone is 0.5-1.1
RETAIN = {"snaps_days": 14, "clips_days": 30, "clips_gb": 4.0, "obs_days": 30, "timelapse_days": 365, "highlights_days": 90, "searches_days": 21}


def infrared(jpg):
    """True for a night-vision (grayscale) frame."""
    a = np.asarray(Image.open(io.BytesIO(jpg)).convert("RGB").resize((32, 18)), dtype=np.float32)
    return bool(np.abs(a[..., 0] - a[..., 1]).mean() + np.abs(a[..., 1] - a[..., 2]).mean() < 4.0)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def now():
    return time.time()


# ---------------------------------------------------------------- storage

def db():
    con = sqlite3.connect(DB, timeout=10, isolation_level=None)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA busy_timeout=5000")
    return con


def init_db():
    con = db()
    con.executescript("""
    CREATE TABLE IF NOT EXISTS obs(ts REAL, score REAL, person REAL, box TEXT, motion REAL,
                                   present INT, moving INT);
    CREATE INDEX IF NOT EXISTS obs_ts ON obs(ts);
    CREATE TABLE IF NOT EXISTS track(ts REAL, pan INT, tilt INT, act TEXT, weak INT);
    CREATE INDEX IF NOT EXISTS track_ts ON track(ts);
    CREATE TABLE IF NOT EXISTS minute(ts INTEGER PRIMARY KEY, n INT, present INT, moving INT,
                                      person INT, motion REAL, noise_db REAL, sounds INT, snap TEXT);
    CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, ts REAL, kind TEXT, detail TEXT,
                                      clip TEXT, thumb TEXT, label TEXT, labelled_by TEXT);
    CREATE INDEX IF NOT EXISTS events_ts ON events(ts);
    """)
    for col in ("policy TEXT",        # which Follow policy ran this cycle (A/B)
                "moves INT"):         # motor commands Follow started during this cycle (a search is many)
        try:
            con.execute(f"ALTER TABLE track ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass                                                     # already there
    con.close()


def add_event(con, kind, detail=None, thumb=None, ts=None):
    ts = ts or now()
    cur = con.execute("INSERT INTO events(ts, kind, detail, thumb) VALUES(?,?,?,?)",
                      (ts, kind, json.dumps(detail or {}), thumb))
    log("event", kind, detail or "")
    return cur.lastrowid


def write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


# ---------------------------------------------------------------- camera

def snapshot(channel="ch1"):
    """The camera's own JPEG. ch1 (640x360) through the app's camera session; ch0 (1080p) for tools."""
    if channel == "ch1":
        return camapi.live_frame()
    camapi.cam_login() if not camapi._cam["cookie"] else None
    req = urllib.request.Request(f"http://{camapi.CAM_HOST}/x/{channel}.jpg", headers={"Cookie": camapi._cam["cookie"] or ""})
    return camapi.with_deadline(lambda: urllib.request.urlopen(req, timeout=6).read(2_000_000), 8)


class FrameSource:
    """Live frames from the camera's 640x360 substream, relayed by MediaMTX (path "sub") and decoded
    here at FRAME_FPS. Replaces the 300 ms snapshot CGI, whose picture is ~1 s old (profiled 2026-11-02).
    Motion is checked on every frame: real motion wakes the main loop at once instead of it sleeping
    out its PERIOD. If the stream stalls, latest() returns None and the loop falls back to snapshots,
    so the freeze / self-heal watchdogs work as before."""
    W, H = 640, 360

    def __init__(self, url=SUB_URL):
        self.url = url
        self.wake_decoder = threading.Event()
        self.frame, self.t = None, 0.0
        self.prev_small = None
        self.motion_c = None        # (t, fx, fy): where in the picture the last real motion was (-0.5..0.5)
        self.mode, self.lag = "sub", STREAM_LAG["sub"]
        self.fast_until = 0.0
        self.governor = StealGovernor()
        self.wake = threading.Event()
        self.lock = threading.Lock()

    def start(self):
        threading.Thread(target=self.run, daemon=True).start()

    # Idle mode (2026-11-02 CPU profile: decoding this stream all day was ~10% of a core on a box allowed ~10% in
    # total). The decoder runs only while someone wants live frames: motion, an active or followed pet, a search.
    # Otherwise the watcher uses the camera's own JPEG snapshot every few seconds.
    want_until = 0.0

    def want(self, secs=STREAM_HOLD):
        self.want_until = max(self.want_until, now() + secs)
        self.wake_decoder.set()

    def wanted(self):
        return now() < self.want_until

    def want_fast(self, secs=FAST_HOLD):
        """Main-stream frames for the next secs (Follow engaged), unless the governor says the box can't afford it."""
        if FAST_ENABLED:
            self.fast_until = max(self.fast_until, now() + secs)
        self.want(STREAM_HOLD)

    def fast(self):
        return now() < self.fast_until and self.governor.ok()

    push_down_until = 0.0

    def desired_mode(self):
        if self.fast():
            return "main"
        return "push" if PUSH_ENABLED and now() >= self.push_down_until else "sub"

    def push_cmd(self):
        """ssh to the camera running the snapshot loop. One loop at a time (pid file), bounded in time, and it stops
        when a write fails, so a dropped connection cannot leave it running for long. One ~20 KB file in RAM."""
        gap = max(0, int(1e6 / PUSH_FPS) - 40000)            # a snapshot takes ~40 ms on the camera
        # a stale pid (the loop was killed) may belong to another process after a reboot: kill it only if it is a push loop
        loop = ("F=/tmp/.push.jpg; O=$(cat /tmp/.push.pid 2>/dev/null); "
                "[ -n \"$O\" ] && [ \"$O\" != $$ ] && grep -q push.jpg /proc/$O/cmdline 2>/dev/null && kill $O; "
                "echo $$ > /tmp/.push.pid; "
                f"E=$(( $(cut -d. -f1 /proc/uptime) + {PUSH_MAX_S} )); "
                f"while [ $(cut -d. -f1 /proc/uptime) -lt $E ]; do prudyntctl snapshot -c 1 -q {PUSH_QUALITY} > $F || break; "
                f"echo \"J $(wc -c < $F)\" || break; cat $F || break; usleep {gap}; done; rm -f $F /tmp/.push.pid")
        key = camapi.CAM_SSH_KEY
        return ["ssh", "-T", "-i", key, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
                "-o", f"UserKnownHostsFile={os.path.dirname(key)}/known_hosts", "-o", "ConnectTimeout=5",
                "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2", f"root@{camapi.CAM_HOST}", loop]

    def read_frame(self, p, mode, n):
        """The next frame as an HxWx3 uint8 array, or None when the source ended."""
        if mode != "push":
            buf = p.stdout.read(n)
            return np.frombuffer(buf, np.uint8).reshape(self.H, self.W, 3) if len(buf) == n else None
        while True:
            line = p.stdout.readline()
            if not line:
                return None
            if line.startswith(b"J "):
                break
        try:
            size = int(line.split()[1])
        except (IndexError, ValueError):
            return None                                   # broken framing: end this session (counted as a failure)
        if not 0 < size <= PUSH_MAX_BYTES:
            return None
        b = p.stdout.read(size)
        if len(b) < size:
            return None
        try:
            im = Image.open(io.BytesIO(b)).convert("RGB")
        except Exception:  # noqa: BLE001 - a torn JPEG: skip it, keep the source
            return np.zeros((0,), np.uint8)
        if im.size != (self.W, self.H):
            im = im.resize((self.W, self.H))
        return np.asarray(im)

    def run(self):
        backoff = 2
        self.wake_decoder = getattr(self, "wake_decoder", threading.Event())
        while True:
            if not self.wanted():
                with self.lock:
                    self.frame = None
                self.wake_decoder.wait(5)
                self.wake_decoder.clear()
                continue
            mode = self.desired_mode()
            if mode == "main":       # 1080p25: every frame must be decoded; skipping the deblocking filter saves ~20%
                src = ["-skip_loop_filter", "all", "-rtsp_transport", "tcp", "-i", MAIN_URL]
                vf = MAIN_VF.format(w=self.W, h=self.H)
            else:
                # no fps filter: the substream is natively FRAME_FPS, and the filter waits for the NEXT frame to decide
                # which to emit (~one frame interval of lag, 200 ms at 5 fps)
                src, vf = ["-rtsp_transport", "tcp", "-i", self.url], f"scale={self.W}:{self.H}"
            if mode == "push":
                p = subprocess.Popen(self.push_cmd(), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            else:
                # "-threads 1" before -i limits the decoder only; the rawvideo output encoder is frame-threaded and,
                # left automatic, can hold frames too (a local reproduction measured ~200 ms at 5 fps), so limit it as well
                p = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-nostdin", "-threads:v", "1", *src,
                                      "-an", "-vf", vf, "-c:v", "rawvideo", "-threads:v", "1", "-pix_fmt", "rgb24",
                                      "-flush_packets", "1", "-f", "rawvideo", "-"],
                                     stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            started, got = now(), 0
            progress = {"t": now(), "done": False}

            def stall_watch(p=p, progress=progress, mode=mode):
                # A source can stay connected and send nothing (ssh keepalives answer, the producer is stuck): a
                # blocked read would never return. Kill the session when no complete frame arrives for a while.
                while not progress["done"]:
                    time.sleep(1)
                    if now() - progress["t"] > FRAME_PROGRESS_S + (10 if got == 0 else 0):
                        log(f"frames: no new {mode} frame for {FRAME_PROGRESS_S} s, restarting the source")
                        p.kill()
                        return
            threading.Thread(target=stall_watch, daemon=True).start()
            switched = False
            every = max(1, round(MAIN_FPS / FRAME_FPS)) if mode == "main" else 1   # motion compares frames 0.2 s apart
            k = 0
            n = self.W * self.H * 3
            stopped = False
            # A new decoder session must not compare its first frame with the last one of the previous session
            # (minutes old when it went idle): the difference reads as motion and can start a chase.
            self.prev_small = None
            try:
                while True:
                    if not self.wanted():
                        stopped = True          # idle again: stop decoding
                        break
                    if self.desired_mode() != mode:
                        switched = True         # into or out of fast mode: restart on the other stream at once
                        break
                    arr = self.read_frame(p, mode, n)
                    if arr is None:
                        break
                    if arr.size == 0:
                        continue
                    backoff = 2
                    got += 1
                    progress["t"] = now()
                    k += 1
                    if k % every:                       # newest frame for the detector; motion on every `every`-th
                        with self.lock:
                            self.frame, self.t = arr, now()
                            self.mode, self.lag = mode, STREAM_LAG[mode]
                        continue
                    s4 = arr[::4, ::4]                                          # 160x90 gray: integer sum, not float mean
                    small = (s4[..., 0].astype(np.int16) + s4[..., 1] + s4[..., 2]) // 3   # bit-identical, 18x less CPU
                    if self.prev_small is not None:
                        d = np.abs(small - self.prev_small)
                        d[:9, :] = 0                                         # the clock overlay
                        if float(d.mean()) > MOTION_DETECT and not camera_moved_at_recently():
                            ys, xs = np.nonzero(d > 25)
                            if len(xs) > 30:      # where it moved: lets follow chase a pet seen once or not at all
                                self.motion_c = (now(), float(xs.mean()) / 160 - 0.5, float(ys.mean()) / 90 - 0.5)
                            self.wake.set()
                    self.prev_small = small
                    with self.lock:
                        self.frame, self.t = arr, now()
                        self.mode, self.lag = mode, STREAM_LAG[mode]
            finally:
                progress["done"] = True
                p.kill()
                p.wait()
            if stopped or switched:
                self.push_short = 0
                continue
            if mode == "push":
                lasted = now() - started
                if got == 0 or lasted < PUSH_SHORT_S:
                    self.push_short = getattr(self, "push_short", 0) + 1
                elif lasted >= PUSH_MAX_S - 30:
                    self.push_short = 0                        # the loop's own time limit: a healthy end
                if got == 0 or self.push_short >= PUSH_SHORT_MAX:
                    log(f"frames: camera push failed ({got} frames, {self.push_short} short sessions), "
                        f"substream for {PUSH_RETRY_S // 60} min")
                    self.push_down_until = now() + PUSH_RETRY_S
                    self.push_short = 0
                else:
                    time.sleep(1)
                continue
            time.sleep(backoff)
            backoff = min(backoff * 2, 60)

    def latest(self, max_age=2.0):
        with self.lock:
            if self.frame is None or now() - self.t > max_age:
                return None
            self.got = (self.t, self.lag, self.mode)      # when the returned frame arrived, how old it was then
            return Image.fromarray(self.frame)


class VerifiedBox(list):
    """A verified pet box [x0, y0, x1, y1] that also carries the verifier's probability (.p), for the tracker."""

    def __init__(self, box, p):
        super().__init__(box)
        self.p = float(p)


class StealGovernor:
    """Is there CPU to spare for fast mode? On a burstable VM, steal time appears once the burst credits are spent;
    then fast mode would only slow everything else down. Reads /proc/stat at most every 10 s."""

    def __init__(self, path="/proc/stat"):
        self.path, self.last, self.at, self.off_until, self.share = path, None, 0.0, 0.0, 0.0
        self.hist = []

    def sample(self):
        try:
            v = [int(x) for x in open(self.path).readline().split()[1:9]]
        except (OSError, ValueError):
            return None
        return sum(v), v[7]

    def ok(self, t=None):
        t = now() if t is None else t
        if t - self.at >= 10:
            self.at = t
            s = self.sample()
            if s:
                self.hist = [h for h in self.hist if t - h[0] <= 70] + [(t, *s)]
                h0 = self.hist[0]
                if t - h0[0] >= 30 and s[0] > h0[1]:
                    self.share = (s[1] - h0[2]) / (s[0] - h0[1])
                    if self.share > FAST_STEAL_MAX and t >= self.off_until:
                        log(f"fast mode off for {FAST_COOLDOWN // 60} min: CPU steal {100 * self.share:.0f}%")
                        self.off_until = t + FAST_COOLDOWN
        return t >= self.off_until


def camera_moved_at_recently(window=4.0):
    try:
        return now() - os.path.getmtime(MOVED) < window
    except OSError:
        return False


class CamAdapter:
    """What the tracker needs from the camera, backed by the app's camera code."""
    position = staticmethod(lambda: camapi.position())
    goto = staticmethod(lambda x, y: camapi.goto(x, y, camapi.QUIET_SPEED))      # automatic moves: slow and quiet
    wait_idle = staticmethod(lambda: camapi.wait_idle(timeout=30.0))            # a quiet full pan takes ~15 s
    busy = staticmethod(lambda: camapi.motor_status()[0] != "0")
    home = staticmethod(lambda: camapi.start_pose())
    stop = staticmethod(lambda: camapi.motor_cmd("s"))
    aim = staticmethod(lambda fx, fy: camapi.aim(fx, fy, camapi.QUIET_SPEED))
    limits = staticmethod(lambda: camapi.limits())
    frame_steps = staticmethod(lambda: (camapi.PAN_STEPS_PER_FRAME, camapi.TILT_STEPS_PER_FRAME))
    presets = staticmethod(lambda: camapi.load_presets())
    moved_at = staticmethod(lambda: os.path.getmtime(MOVED))


def cam_action(path):
    """Fire-and-forget camera CGI (restart encoder, reboot)."""
    try:
        req = urllib.request.Request(f"http://{camapi.CAM_HOST}{path}",
                                     headers={"Cookie": camapi._cam["cookie"] or ""})
        camapi.with_deadline(lambda: urllib.request.urlopen(req, timeout=15).read(2000), 20)
        return True
    except Exception as e:  # noqa: BLE001
        log("camera action failed", path, e)
        return False


def imaging_state():
    """First event of the camera's heartbeat stream: day/night, IR, colour."""
    import http.client

    def call():
        c = http.client.HTTPConnection(camapi.CAM_HOST, 80, timeout=5)
        try:
            c.request("GET", "/x/json-heartbeat.cgi", headers={"Cookie": camapi._cam["cookie"] or ""})
            r = c.getresponse()
            for _ in range(20):
                line = r.fp.readline(4096).decode(errors="replace")
                if line.startswith("data:"):
                    return json.loads(line[5:])
            return {}
        finally:
            c.close()

    return camapi.with_deadline(call, 8)


# ---------------------------------------------------------------- audio

def privacy_flag():
    try:
        with open(PRIVACY) as f:
            return f.read().strip() == "1"
    except OSError:
        return False


class Verifier:
    """Second stage: is this candidate really the pet? YOLOX calls her black plush toy a dog
    (2026-11-02: 42 of 51 morning candidates at >= 0.30 were the toy). Crops are described by
    MobileNetV2 ImageNet logits; verifier.json (from tools/bootstrap_pet.py) is either
      zeroshot: P(any ImageNet dog breed) >= threshold   (untrained; used while labels are few)
      logreg:   a logistic regression trained on Claude-labelled crops
    """

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, path=VERIFIER):
        import onnxruntime as ort
        with open(path) as f:
            self.cfg = json.load(f)
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        model = os.path.join(os.path.dirname(path), self.cfg.get("feature_model", "mobilenetv2-12.onnx"))
        self.sess = ort.InferenceSession(model, opts, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.threshold = float(self.cfg["threshold"])
        if self.cfg["mode"] == "finetuned":           # tools/finetune_pet.py: one logit from a fine-tuned MobileNetV2
            self.sess = ort.InferenceSession(os.path.join(os.path.dirname(path), self.cfg["model"]), opts,
                                             providers=["CPUExecutionProvider"])
            self.inp = self.sess.get_inputs()[0].name
        if self.cfg["mode"] == "logreg":
            self.mu, self.sd = np.array(self.cfg["mean"]), np.array(self.cfg["std"])
            self.w, self.b = np.array(self.cfg["w"]), float(self.cfg["b"])

    def prob(self, img, box, margin=0.15):
        w, h = img.size
        x0, y0, x1, y1 = box
        mx, my = (x1 - x0) * margin, (y1 - y0) * margin
        crop = img.crop((int(max(0, (x0 - mx) * w)), int(max(0, (y0 - my) * h)),
                         int(min(w, (x1 + mx) * w)), int(min(h, (y1 + my) * h)))).convert("RGB")
        if crop.width < 8 or crop.height < 8:
            return 0.0
        x = np.asarray(crop.resize((224, 224), Image.BILINEAR), dtype=np.float32) / 255.0
        x = ((x - self.MEAN) / self.STD).transpose(2, 0, 1)[None]
        z = self.sess.run(None, {self.inp: x})[0].reshape(-1).astype(np.float64)
        if self.cfg["mode"] == "finetuned":
            return float(1 / (1 + np.exp(-z[0])))
        e = np.exp(z - z.max())
        e /= e.sum()
        a, b = self.cfg.get("classes", [151, 268])
        dog = float(e[a:b + 1].sum())               # zero-shot: P(any ImageNet dog breed)
        if self.cfg["mode"] != "logreg":
            return dog
        p = float(1 / (1 + np.exp(-(((z - self.mu) / self.sd) @ self.w + self.b))))
        # Optional zero-shot backup ("a confident 'this is a dog' vouches for her"). Off by default: on
        # 2026-11-02 the octopus toy reached 0.72 zero-shot in afternoon sun and passed for an hour.
        backup = self.cfg.get("zeroshot_backup")
        if backup is not None and dog >= backup:
            p = max(p, self.threshold)
        return p


class SoundModel:
    """YAMNet (Google, Apache-2.0; the model Frigate uses) on a sound episode's audio.

    Only runs on episodes the loudness gate already found (~3.3 ms per 0.975 s window on this
    box), so its average CPU is negligible. On 2026-11-02 the loudness rule called 26 morning
    sounds "barking"; YAMNet heard taps, dishes, frying, speech and music in 19 of them and
    dog sounds in 7, all within minutes of the pet appearing on camera. The 0.55 threshold is
    PROVISIONAL (no human labels yet): the app collects 👍/👎 on sound moments to fit it.
    """
    DOG_CLASSES = {69: "dog_sound", 70: "barking", 71: "barking", 72: "howling", 73: "barking",
                   74: "growling", 75: "whining"}
    WIN, HOP = 15600, 7800

    def __init__(self):
        from ai_edge_litert.interpreter import Interpreter
        import csv
        self.it = Interpreter(model_path=YAMNET_MODEL, num_threads=1)
        self.it.allocate_tensors()
        self.inp = self.it.get_input_details()[0]["index"]
        self.out = self.it.get_output_details()[0]["index"]
        with open(YAMNET_CLASSES) as f:
            self.names = [r["display_name"] for r in csv.DictReader(f)]

    def classify(self, pcm16):
        x = pcm16.astype(np.float32) / 32768.0
        if len(x) < self.WIN:
            x = np.pad(x, (0, self.WIN - len(x)))
        best = np.zeros(521, dtype=np.float32)
        for st in range(0, len(x) - self.WIN + 1, self.HOP):
            self.it.set_tensor(self.inp, x[st:st + self.WIN])
            self.it.invoke()
            best = np.maximum(best, self.it.get_tensor(self.out).reshape(-1))
        dog_i = max(self.DOG_CLASSES, key=lambda k: best[k])
        ranked = [k for k in np.argsort(best)[::-1] if self.names[k] not in ("Silence", "Inside, small room")]
        top = int(ranked[0])
        return {"dog": round(float(best[dog_i]), 2), "dog_label": self.names[dog_i],
                "dog_kind": self.DOG_CLASSES[dog_i], "top": self.names[top], "top_score": round(float(best[top]), 2)}


class Audio(threading.Thread):
    """Reads mono 16 kHz audio from the local MediaMTX relay and reports sound episodes.

    Cheap gate first: an episode is a run of 250 ms blocks at least 18 dB over the rolling noise
    floor (and above -45 dBFS), ended by 3 s of quiet; a single burst of <= 0.5 s is a click and
    is dropped. Each remaining episode is classified by YAMNet (see SoundModel); if the model is
    unavailable the old loudness rule is used. Per-minute peak levels are kept for the stats.
    """
    BLOCK = 4000  # 0.25 s at 16 kHz
    RING = 60     # keep the last 15 s of audio for classification

    def __init__(self, events):
        super().__init__(daemon=True)
        self.events = events
        self.levels = []          # recent block levels (dBFS) for the noise floor
        self.floor, self.n_blocks = -60.0, 0   # the noise floor, recomputed every 8 blocks
        self.minute_peak = -120.0
        self.minute_sounds = 0
        self.lock = threading.Lock()
        self.last_ok = 0.0
        self.model = None
        self.model_failed = False
        self.ring = collections.deque(maxlen=self.RING)

    def take_minute(self):
        with self.lock:
            peak, n = self.minute_peak, self.minute_sounds
            self.minute_peak, self.minute_sounds = -120.0, 0
        return (peak if peak > -119 else None), n

    def run(self):
        while True:
            if privacy_flag():
                time.sleep(5)              # paused: not even subscribed to the (now silent) audio
                continue
            try:
                self.listen()
            except Exception as e:  # noqa: BLE001
                log("audio:", e)
            time.sleep(10)

    def get_model(self):
        if self.model is None and not self.model_failed:
            try:
                self.model = SoundModel()
            except Exception as e:  # noqa: BLE001 - fall back to the loudness rule
                self.model_failed = True
                log("sound model unavailable, using loudness rule:", e)
        return self.model

    def listen(self):
        cmd = ["nice", "-n", "10", "ffmpeg", "-nostdin", "-loglevel", "error", "-rtsp_transport", "tcp",
               # Subscribe to the audio track only: receiving the video just to drop it cost
               # both this process and MediaMTX CPU on a box that lives at its 5% baseline.
               "-allowed_media_types", "audio", "-threads", "1",
               "-i", "rtsp://127.0.0.1:8554/cam", "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        burst, bursts, quiet, episode_blocks = 0, [], 0, 0
        try:
            while True:
                raw = p.stdout.read(self.BLOCK * 2)
                if len(raw) < self.BLOCK * 2 or privacy_flag():
                    return
                self.last_ok = now()
                pcm = np.frombuffer(raw, dtype=np.int16)
                self.ring.append(pcm)
                x = pcm.astype(np.float32)
                db_ = 20 * math.log10(max(1e-9, float(np.sqrt(np.mean(x * x))) / 32768))
                self.levels.append(db_)
                if len(self.levels) > 1200:          # 5 minutes of blocks
                    self.levels = self.levels[-1200:]
                self.n_blocks += 1
                if self.n_blocks % 8 == 1:   # every 2 s: a 1200-value percentile per 0.25 s block
                    self.floor = float(np.percentile(self.levels, 20)) if len(self.levels) > 40 else -60.0   # was ~1% of a core
                floor = self.floor
                loud = db_ > max(floor + 18, -45)
                with self.lock:
                    self.minute_peak = max(self.minute_peak, db_)
                if bursts or burst:
                    episode_blocks += 1
                if loud:
                    if not bursts and not burst:
                        episode_blocks = 1
                    burst += 1
                    quiet = 0
                    continue
                if burst:
                    bursts.append((now(), burst, db_))
                    burst = 0
                quiet += 1
                if bursts and quiet >= 12:            # a sound episode ends after 3 s of quiet
                    self.close_episode(bursts, episode_blocks, floor, db_)
                    bursts, episode_blocks = [], 0
        finally:
            p.kill()

    def close_episode(self, bursts, episode_blocks, floor, db_):
        t0 = bursts[0][0] - 0.25 * bursts[0][1]
        longest = max(b[1] for b in bursts)
        if len(bursts) == 1 and longest <= 2:
            # One burst of <= 0.5 s is a click (the IR-cut filter flipping, a door latch):
            # 53 of these in 2 h on 2026-11-02 while the day/night switch was flapping.
            return
        peak = max(self.levels[-(episode_blocks + 1):] or [db_])
        detail = {"bursts": len(bursts), "longest_s": longest / 4, "peak_db": round(peak, 1), "floor_db": round(floor, 1)}
        model = self.get_model()
        if model is not None:
            # The episode plus 0.5 s before it, without the 3 s of trailing quiet.
            n = min(len(self.ring), episode_blocks + 2)
            blocks = list(self.ring)[-n:]
            audio = np.concatenate(blocks[:max(1, len(blocks) - 10)])
            try:
                t = time.time()
                y = model.classify(audio)
                y["ms"] = round((time.time() - t) * 1000)
            except Exception as e:  # noqa: BLE001
                log("sound model failed:", e)
                y = None
            if y is not None:
                detail["yamnet"] = y
                if y["dog"] >= DOG_SOUND_THRESHOLD:
                    kind = y["dog_kind"]
                elif peak >= LOUD_NON_DOG_DB:
                    kind = "loud sound"
                else:
                    return                        # ordinary household sound: not a moment
                self.emit(kind, detail, t0)
                return
        # Fallback: the loudness rule.
        short = [b for b in bursts if b[1] <= 4]
        kind = "barking" if len(short) >= 2 else "loud sound"
        if kind == "loud sound" and longest < 3:
            return
        self.emit(kind, detail, t0)

    def emit(self, kind, detail, t0):
        with self.lock:
            self.minute_sounds += 1
        self.events.put(("sound", {"kind": kind, **detail}, t0))


# ---------------------------------------------------------------- clips

class Clips(threading.Thread):
    """Cuts short MP4 clips out of MediaMTX's rolling 1-hour recording (no re-encoding)."""

    def __init__(self):
        super().__init__(daemon=True)
        self.jobs = queue.Queue()
        self.last_clip_end = 0.0

    def want(self, event_id, ts):
        start = ts - CLIP_BEFORE
        if start < self.last_clip_end - 5:          # overlaps the previous clip: reuse it
            return False
        self.last_clip_end = start + CLIP_LEN
        self.jobs.put((event_id, start))
        return True

    def run(self):
        while True:
            event_id, start = self.jobs.get()
            wait = start + CLIP_LEN + 6 - now()       # the recording must contain the whole clip
            if wait > 0:
                time.sleep(wait)
            try:
                self.cut(event_id, start)
            except Exception as e:  # noqa: BLE001
                log("clip failed", event_id, e)

    def cut(self, event_id, start):
        day = time.strftime("%Y-%m-%d", time.localtime(start))
        os.makedirs(f"{CLIPS}/{day}", exist_ok=True)
        name = f"{day}/{time.strftime('%H%M%S', time.localtime(start))}-{event_id}.mp4"
        iso = dt.datetime.fromtimestamp(start, dt.timezone.utc).isoformat().replace("+00:00", "Z")
        url = "http://127.0.0.1:9996/get?" + urllib.parse.urlencode(
            {"path": "cam", "start": iso, "duration": CLIP_LEN, "format": "mp4"})

        def call():
            with urllib.request.urlopen(url, timeout=30) as r:
                return r.read(80_000_000)

        data = camapi.with_deadline(call, 45)
        if len(data) < 10_000:
            raise RuntimeError(f"clip too small ({len(data)} bytes)")
        with open(f"{CLIPS}/{name}", "wb") as f:
            f.write(data)
        con = db()
        con.execute("UPDATE events SET clip=? WHERE id=?", (name, event_id))
        con.close()
        log("clip", name, f"{len(data) // 1024} KB")


# ---------------------------------------------------------------- daily housekeeping

def make_timelapse(day):
    src = f"{SNAPS}/{day}"
    out = f"{TIMELAPSE}/{day}.mp4"
    if not os.path.isdir(src) or os.path.exists(out) or len(os.listdir(src)) < 30:
        return
    os.makedirs(TIMELAPSE, exist_ok=True)
    tmp = out + ".part.mp4"
    cmd = ["timeout", "-k", "10", "900", "nice", "-n", "19", "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
           "-framerate", "24", "-pattern_type", "glob", "-i", f"{src}/*.jpg",
           "-vf", "scale=640:-2", "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
           "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-threads", "1", tmp]
    subprocess.run(cmd, check=True)
    os.replace(tmp, out)
    log("timelapse", out)


def make_highlights(day):
    """Stitch up to 8 of the day's clips into one reel, stream-copied (no re-encode): the pet
    getting up first, then arrivals, then barking/sounds, played in time order."""
    out = f"{HIGHLIGHTS}/{day}.mp4"
    if os.path.exists(out):
        return
    import datetime as dt
    d = dt.date.fromisoformat(day)
    start = time.mktime(d.timetuple())                                  # local midnights: DST days
    end = time.mktime((d + dt.timedelta(days=1)).timetuple())           # are 23 or 25 hours long
    con = db()
    rows = con.execute("SELECT ts, kind, clip FROM events WHERE ts>=? AND ts<? AND clip IS NOT NULL",
                       (start, end)).fetchall()
    con.close()
    rank = {"active": 0, "barking": 1, "howling": 1, "whining": 1, "growling": 1, "dog_sound": 2, "appeared": 3}
    pick = sorted(rows, key=lambda r: (rank.get(r[1], 9), r[0]))[:8]
    pick = sorted((r for r in pick if os.path.exists(f"{CLIPS}/{r[2]}")), key=lambda r: r[0])
    if len(pick) < 2:
        return
    os.makedirs(HIGHLIGHTS, exist_ok=True)
    listing = f"{HIGHLIGHTS}/{day}.txt"
    with open(listing, "w") as f:
        for r in pick:
            f.write(f"file '{CLIPS}/{r[2]}'\n")
    tmp = out + ".part.mp4"
    cmd = ["timeout", "-k", "10", "300", "nice", "-n", "19", "ffmpeg", "-nostdin", "-loglevel", "error", "-y",
           "-f", "concat", "-safe", "0", "-i", listing, "-c", "copy", "-movflags", "+faststart", tmp]
    try:
        subprocess.run(cmd, check=True)
        os.replace(tmp, out)
        log("highlights", out, f"{len(pick)} clips")
    finally:
        for f in (listing, tmp):
            if os.path.exists(f):
                os.remove(f)


def prune_dirs(root, keep_days):
    if not os.path.isdir(root):
        return
    cutoff = time.strftime("%Y-%m-%d", time.localtime(now() - keep_days * 86400))
    for name in sorted(os.listdir(root)):
        path = f"{root}/{name}"
        if name[:10] < cutoff:
            subprocess.run(["rm", "-rf", "--", path], check=False)


def prune_clips_by_size(limit_gb):
    files = []
    for root, _, names in os.walk(CLIPS):
        for n in names:
            p = os.path.join(root, n)
            files.append((os.path.getmtime(p), os.path.getsize(p), p))
    total = sum(f[1] for f in files)
    for _, size, p in sorted(files):
        if total <= limit_gb * 1e9:
            break
        os.remove(p)
        total -= size


def housekeeping():
    yesterday = time.strftime("%Y-%m-%d", time.localtime(now() - 86400))
    try:
        make_timelapse(yesterday)
    except Exception as e:  # noqa: BLE001
        log("timelapse failed", e)
    try:
        make_highlights(yesterday)
    except Exception as e:  # noqa: BLE001
        log("highlights failed", e)
    prune_dirs(SNAPS, RETAIN["snaps_days"])
    prune_dirs(CANDIDATES, RETAIN["clips_days"])
    prune_dirs(SEARCHES, RETAIN["searches_days"])
    prune_dirs(CLIPS, RETAIN["clips_days"])
    prune_clips_by_size(RETAIN["clips_gb"])
    # Bug reports (frame + 100 s clip each), named YYYYMMDD-HHMMSS: kept 60 days.
    cutoff = time.strftime("%Y%m%d", time.localtime(now() - 60 * 86400))
    for n in os.listdir(REPORTS) if os.path.isdir(REPORTS) else []:
        if n[:8].isdigit() and n[:8] < cutoff:
            shutil.rmtree(os.path.join(REPORTS, n), ignore_errors=True)
    prune_dirs(TIMELAPSE, RETAIN["timelapse_days"])      # files named <day>.mp4: same date-prefix rule
    prune_dirs(HIGHLIGHTS, RETAIN["highlights_days"])
    con = db()
    con.execute("DELETE FROM obs WHERE ts < ?", (now() - RETAIN["obs_days"] * 86400,))
    con.execute("DELETE FROM track WHERE ts < ?", (now() - RETAIN["obs_days"] * 86400,))
    con.execute("UPDATE events SET clip=NULL WHERE clip IS NOT NULL AND ts < ?", (now() - RETAIN["clips_days"] * 86400,))
    con.close()


# ---------------------------------------------------------------- main loop

class Watch:
    def __init__(self):
        self.det = Detector(MODEL)
        try:
            self.verifier = Verifier()
            log(f"verifier: {self.verifier.cfg['mode']} threshold {self.verifier.threshold}")
        except Exception as e:  # noqa: BLE001 - no verifier yet: YOLOX score alone, as before
            self.verifier = None
            log("verifier unavailable, using detector score only:", e)
        self.last_candidate_save = 0.0
        self.events = queue.Queue()
        self.audio = Audio(self.events)
        self.last_audio_fix = 0.0
        self.audio_watch_from = now()
        self.clips = Clips()
        self.frames = FrameSource()
        self.source = None
        self.dog_sound_pending = False
        self.cam_moves = collections.deque(maxlen=20)
        self.move_box0 = None
        self.tracker = Tracker(CamAdapter(), self.find_pet, now, log=lambda m: log("follow:", m), sleep=time.sleep,
                               find_now=self.find_pet_now, can_scan=lambda: self.source != "snapshot",   # idle is fine (frames start on demand); a broken stream is blind
                               tick=self.search_tick, cancelled=self.find_cancelled)
        self.last_tick = 0.0
        self.last_motion_t = 0.0
        self.last_find_req = now()          # requests from before this process started are stale
        self.find_result = None
        self.prev_small = None
        self.prev_box = None
        self.search_rec = None              # frames a running search judged, written by save_search_evidence()
        self.last_found = []
        self.last_detect = 0.0
        self.sharp_samples = []       # (t, sharpness below threshold?)
        self.view_blocked = False
        self.mode_changes = []        # (ts, mode) when the camera's day/night mode changed
        self.last_mode = None
        try:
            with open(FLAP_LOCK_FILE) as f:
                self.flap_lock_until = float(f.read().strip() or 0)
        except (OSError, ValueError):
            self.flap_lock_until = 0.0
        self.same_since = None
        self.last_fix = 0.0
        self.fixes = 0
        self.last_ok = now()
        self.offline = False
        self.present = False
        self.present_since = None
        self.last_seen = 0.0
        self.gone_since = now()
        self.rest_since = None
        self.moving_since = None
        self.policy_arm = "base"                   # which Follow policy the last cycle ran (apply_policy)
        try:                                       # the code version that ran (deploy.sh writes git describe here)
            self.code_version = open("/opt/petcam/VERSION").read().strip()[:40]
        except OSError:
            self.code_version = "unknown"
        self.minute = None
        self.acc = None
        self.imaging = {}
        self.last_imaging = 0.0
        # Last date housekeeping ran, persisted: initialising it to "today" at start-up skipped
        # the 3 am run whenever the watcher had restarted after midnight (2026-11-02).
        try:
            with open(HOUSE_FILE) as f:
                self.last_house = f.read().strip()
        except OSError:
            self.last_house = time.strftime("%Y-%m-%d", time.localtime(now() - 86400))

    def start(self):
        init_db()
        for d in (SNAPS, CLIPS, TIMELAPSE, HIGHLIGHTS):
            os.makedirs(d, exist_ok=True)
        self.audio.start()
        self.clips.start()
        self.frames.start()
        while True:
            t0 = now()
            try:
                self.cycle()
            except Exception as e:  # noqa: BLE001
                log("cycle error:", type(e).__name__, e)
                self.on_failure()
            # Next cycle after PERIOD, or as soon as the frame reader sees motion (but not within MIN_CYCLE).
            fast = self.frames.fast() and self.source == "stream" and self.frames.mode == "main"
            time.sleep(max(0.0, (FAST_CYCLE if fast else MIN_CYCLE) - (now() - t0)))
            self.frames.wake.wait(max(0.0, (FAST_CYCLE if fast else PERIOD) - (now() - t0)))
            self.frames.wake.clear()

    def apply_policy(self, t):
        """Live A/B of a learned Follow policy: odd 2-hour blocks run its constants, even blocks today's. Fails closed:
        no file, a bad file or no valid constants -> today's constants, logged as 'base'."""
        if not hasattr(self, "policy_base"):
            self.policy_base = {k: getattr(T, k) for k in T.POLICY_BOUNDS}
            self.policy_doc, self.policy_mtime, self.policy_stops = {}, None, []
        try:
            m = os.path.getmtime(FOLLOW_POLICY)
            if m != self.policy_mtime:
                with open(FOLLOW_POLICY) as f:
                    doc = json.load(f)
                self.policy_doc = T.policy_constants(doc)
                self.policy_stops = T.policy_stops(doc)
                self.policy_mtime = m
                log("follow policy:", self.policy_doc or "none valid")
        except (OSError, ValueError):
            self.policy_doc, self.policy_mtime, self.policy_stops = {}, None, []
        if not hasattr(self, "ppo"):
            self.ppo = None
            if os.path.exists(FOLLOW_PPO):
                try:
                    from ppo_follow import PPOFollow
                    self.ppo = PPOFollow(FOLLOW_PPO, CamAdapter(), Tracker, self.find_pet, now,
                                         log=lambda m: log("follow:", m), sleep=time.sleep, find_now=self.find_pet_now)
                    log("follow policy: PPO controller loaded")
                except Exception as e:  # noqa: BLE001 - a broken model must never stop following: base runs instead
                    log("follow policy: PPO controller unavailable:", e)
        arms = ["base"] + (["learned"] if self.policy_doc or self.policy_stops else []) + (["ppo"] if self.ppo else [])
        # 2-hour blocks rotate through every policy present, shifted one block per day: with 12 blocks a day and 2 arms
        # an unshifted rotation gave each arm the SAME hours every day (review 2026-11-04), confounding policy with
        # her daily routine. Shifted, each hour of the day alternates between arms day by day.
        arm = arms[(int(t // POLICY_BLOCK_S) + int(t // 86400)) % len(arms)]
        try:
            want, until = open(POLICY_OVERRIDE).read().split()[:2]
            if want in arms and t < float(until):
                arm = want
        except (OSError, ValueError):
            pass
        for k, v in self.policy_base.items():
            setattr(T, k, self.policy_doc.get(k, v) if arm == "learned" else v)
        T.SEARCH_STOPS = self.policy_stops if arm == "learned" else []
        if arm == "ppo" and self.policy_arm != "ppo":
            self.ppo.reset()                                # a fresh memory at the start of each PPO block
        self.policy_arm = arm

    def check_stream_stall(self):
        """Video stopped but snapshots still work: after a stream-settings change (2026-11-02 10:28) the
        encoder stopped sending RTSP while ch1.jpg kept updating, which the freeze watchdog (identical
        snapshots) cannot see. Restart the streamer after STREAM_STALL s; at most once per 10 min."""
        t = now()
        down = t - (getattr(self, "stream_down_since", None) or t)
        if down < STREAM_STALL or t - self.last_fix < 600:
            return
        self.last_fix = t
        con = db()
        add_event(con, "camera_frozen", {"action": "restart streamer (video stalled, snapshots fine)", "down_s": round(down)})
        con.close()
        cam_action("/x/restart-prudynt.cgi")
        self.stream_down_since = t

    def check_audio_lost(self, privacy, imaging):
        """Sound gone while video works and the camera says its mic is on: the streamer only brings the mic
        back after a restart (2026-11-02: after pause/resume the stream carried an empty audio track for
        20 min until restarted by hand). Restart after AUDIO_LOST s; at most once per 10 min."""
        t = now()
        lost = t - max(self.audio.last_ok, self.audio_watch_from)   # no reading yet after a start is not silence
        if privacy or imaging.get("mic_enabled") == 0 or lost < AUDIO_LOST or t - self.last_audio_fix < 600:
            return
        self.last_audio_fix = t
        con = db()
        add_event(con, "camera_frozen", {"action": "restart streamer (no sound, mic on)", "down_s": round(lost)})
        con.close()
        cam_action("/x/restart-prudynt.cgi")

    def on_failure(self):
        self.heal_streamer()
        if not self.offline and now() - self.last_ok > OFFLINE_AFTER:
            self.offline = True
            con = db()
            add_event(con, "camera_offline", {"since": self.last_ok})
            con.close()
        self.publish(status="offline" if self.offline else "trouble")

    def heal_streamer(self):
        """Snapshots fail but the camera answers: the streamer (prudynt) crashed rather than froze
        (2026-11-02 06:15, after a restart: RTSP and its API port closed, web UI fine, nothing came
        back on its own). Restart it after 2 min; if two restarts don't help, reboot the camera.
        Rate-limited like the freeze watchdog."""
        t = now()
        down = t - self.last_ok
        if down < 120 or t - self.last_fix < 300:
            return
        try:
            camapi.with_deadline(lambda: urllib.request.urlopen(f"http://{camapi.CAM_HOST}/", timeout=5).read(100), 8)
        except Exception:  # noqa: BLE001 - camera itself unreachable: nothing we can restart
            return
        self.last_fix = t
        self.heal_tries = getattr(self, "heal_tries", 0) + 1
        con = db()
        if self.heal_tries <= 2:
            add_event(con, "camera_frozen", {"action": "restart streamer (not responding)", "down_s": round(down)})
            camapi.cam_login() if not camapi._cam["cookie"] else None
            cam_action("/x/restart-prudynt.cgi")
        elif t - getattr(self, "last_reboot", 0) > 3600:
            self.last_reboot = t
            self.heal_tries = 0
            add_event(con, "camera_frozen", {"action": "reboot camera", "down_s": round(down)})
            cam_action("/x/reboot.cgi")
        con.close()

    def cycle(self):
        img = self.frames.latest()
        if img is not None:
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=85)
            jpg = buf.getvalue()
            arrived, lag, mode = self.frames.got
            if self.source != "stream" or getattr(self, "frame_mode", None) != mode:
                log(f"frames: live stream ({mode})")
            self.source, self.frame_mode = "stream", mode
            self.frame_capture_t = arrived - lag
        else:                       # idle (decoder off) or stream down: the camera's own snapshot
            jpg = snapshot("ch1")
            img = Image.open(io.BytesIO(jpg))
            img.load()
            self.frame_mode, self.frame_capture_t = "snapshot", now() - STREAM_LAG["snapshot"]
            if not self.frames.wanted():
                if self.source != "idle":
                    log("frames: idle, camera snapshots")
                self.source = "idle"
                self.stream_down_since = None
            else:
                if self.source != "snapshot":
                    log("frames: camera snapshots (stream not available)")
                    self.stream_down_since = now()
                self.source = "snapshot"
                self.check_stream_stall()
        t = now()
        con = db()
        self.heal_tries = 0
        if self.offline:
            self.offline = False
            add_event(con, "camera_back", {"down_s": round(t - self.last_ok)})
        self.last_ok = t

        gray = np.asarray(img.convert("L").resize((160, 90)), dtype=np.int16)
        frozen = self.check_frozen(con, gray, t)
        self.check_blocked(con, gray, t)

        moved_recently = self.camera_moved_recently()
        prev = self.prev_small
        # A camera JPEG and a decoded stream frame differ in compression and exposure: comparing across a switch
        # reads as motion and restarted the decoder 3 s after every switch to idle (measured 2026-11-02 21:26).
        same_source = getattr(self, "prev_source", None) in (None, self.source)
        self.prev_source = self.source
        diff = None
        motion = 0.0
        if prev is not None and not moved_recently and same_source:
            diff = np.abs(gray - prev)
            diff[:9, :] = 0                               # ignore the clock overlay
            motion = float(diff.mean())
        self.prev_small = gray

        privacy = self.privacy_on()
        if privacy:
            # Paused: the picture is black. Keep the freeze watchdog and heartbeat, but don't
            # detect or log comings/goings (they would all be false "left" events).
            con.close()
            with open(HEARTBEAT, "w") as f:
                f.write(str(t))
            self.publish(status="privacy", score=0.0, moving=False, box=None, det_t=None, motion=motion, person=0.0)
            return
        # The detector only runs when the picture changed or every DETECT_EVERY seconds: a still
        # room doesn't need a fresh answer every cycle, and this box lives near its CPU baseline.
        lively = (self.moving_since is not None or t - (self.last_motion_t or 0) < 60 or self.tracker.searching
                  or t - (self.last_seen or 0) < 20 and not self.rest_since)
        if motion > MOTION_DETECT:
            self.last_motion_t = t
        if lively or motion > MOTION_DETECT:
            self.frames.want()                 # something to watch closely: live frames for the next 90 s
        # Follow engaged with her: fresher frames (main stream) and a detector run every FAST_CYCLE (CPU governed).
        engaged = (not privacy and self.setting("follow", False)
                   and (self.tracker.searching or t - (self.last_seen or 0) < 20 and not self.rest_since))
        if engaged:
            self.frames.want_fast()
        fast_now = self.frame_mode == "main" and self.frames.fast()
        # After our own camera move the last answer shows the old view: ask again once the picture has settled,
        # so Follow sees where she is now instead of waiting up to DETECT_EVERY (tracker.step ignores the old box).
        settle_t = self.tracker.last_move + T.POST_MOVE_IGNORE
        after_move = self.last_detect < settle_t <= t
        if motion > MOTION_DETECT or t - self.last_detect >= (DETECT_EVERY if lively else DETECT_STILL) or prev is None \
                or after_move or fast_now and t - self.last_detect >= 0.9 * FAST_CYCLE:
            self.last_found = self.det.detect(img, min_score=0.10)
            self.last_detect = t
            self.last_detect_capture = self.frame_capture_t
            if self.verifier:
                for f in sorted((f for f in self.last_found if f["label"] != "person"), key=lambda f: -f["score"])[:3]:
                    f["verified"] = round(self.verifier.prob(img, f["box"]), 3)
            if any(f["label"] != "person" for f in self.last_found):
                self.save_candidate(jpg, t)
        found = self.last_found
        animals = [f for f in found if f["label"] != "person"]
        if self.verifier:
            # Only candidates the verifier accepts are the pet; the toy's "dog" box is ignored.
            pets = [f for f in animals if f.get("verified", 0.0) >= self.verifier.threshold]
            best = max(pets, key=lambda f: f["verified"]) if pets else None
            score = 1.0 if best else 0.0          # presence logic below works on a yes/no here
        else:
            best = max(animals, key=lambda f: f["score"]) if animals else None
            score = best["score"] if best else 0.0
        person = max((f["score"] for f in found if f["label"] == "person"), default=0.0)

        moving = False
        if best and score >= ABSENT_SCORE:
            b = best["box"]
            if self.prev_box is not None and not moved_recently:
                c0 = ((self.prev_box[0] + self.prev_box[2]) / 2, (self.prev_box[1] + self.prev_box[3]) / 2)
                c1 = ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)
                # Relative to her size: the detector's box on a dog lying still jitters by ~0.03 between
                # runs (2026-11-02 14:00: "<pet> is moving" while she slept and the owner walked past).
                moving = math.hypot(c1[0] - c0[0], c1[1] - c0[1]) > max(0.025, 0.15 * math.hypot(b[2] - b[0], b[3] - b[1]))
            if not moving and diff is not None:
                # Pixel change inside her box: catches stretching, scratching, head lifts.
                x0, y0, x1, y1 = int(b[0] * 160), int(b[1] * 90), int(math.ceil(b[2] * 160)), int(math.ceil(b[3] * 90))
                if x1 > x0 and y1 > y0:
                    # minus the frame-wide change, so an exposure / white-balance jump isn't her moving
                    moving = float(diff[y0:y1, x0:x1].mean()) - float(diff.mean()) > 4.0
            self.prev_box = b
        else:
            self.prev_box = None

        self.update_pet(con, t, score, moving, jpg, best["box"] if best else None)
        if not privacy and self.find_requested(t):
            # "already in view" only from THIS cycle's detector run: between runs `best` repeats an older answer
            # (review 2026-11-04: a 20 s old box answered the owner's Find without any detection)
            fresh = self.last_detect == t
            self.find(con, best if fresh and best and score >= PRESENT_SCORE else None)
        elif self.setting("follow", False) and not privacy:
            self.apply_policy(t)
            try:
                weak = None
                if self.verifier and self.last_detect == t:   # only from this cycle's detector run, never a stale one
                    weak = [f["box"] for f in animals if f is not best and WEAK_MIN <= f.get("verified", 0.0) < self.verifier.threshold]
                motion_c = self.frames.motion_c if person < 0.5 else None   # a person leaving is not her
                if self.policy_arm == "ppo" and self.ppo:
                    fresh = self.last_detect == t
                    if fresh:
                        self.ppo_weak_n = len(weak or [])
                    st, px, py = camapi.motor_status()       # the counter right now (the cached pose can be 30 s old)
                    seen = best["box"] if best and score >= PRESENT_SCORE else None
                    act = self.ppo.step(t, fresh, seen if fresh else None, seen, motion_c,
                                        getattr(self, "ppo_weak_n", 0), (px, py), st != "0")
                else:
                    act = self.tracker.step(best["box"] if best and score >= PRESENT_SCORE else None, moving,
                                            sound=self.dog_sound_pending and not self.present,
                                            motion=motion_c, weak=weak,
                                            # on the substream's clock (the constants were tuned on it): a fresher
                                            # main-stream frame is judged by when it was actually taken
                                            box_t=getattr(self, "last_detect_capture", self.last_detect) + STREAM_LAG["sub"])
                try:   # per cycle: where the camera pointed and what Follow did, to fit and replay its simulator
                    pose = self.current_pose(t) or (None, None)
                    n_motor = getattr(self.tracker, "n_motor", 0)
                    moves, self.n_motor_logged = n_motor - getattr(self, "n_motor_logged", n_motor), n_motor
                    con.execute("INSERT INTO track(ts, pan, tilt, act, weak, policy, moves) VALUES(?,?,?,?,?,?,?)",
                                (t, pose[0], pose[1], (act or "")[:80] or None, len(weak or []),
                                 f"{self.policy_arm}@{self.code_version}", moves))   # replayable: arm + exact code
                except Exception as e:  # noqa: BLE001 - logging must never stop following
                    log("track log:", e)
                if act:
                    log("follow:", act)
                    if act.startswith("search"):
                        add_event(con, "search", {"result": act})
                        self.save_search_evidence(act, asked=False)
            except Exception as e:  # noqa: BLE001 - following must never stop the watcher
                log("follow failed:", e)
        self.dog_sound_pending = False
        try:
            self.check_zones(con, t, best if score >= PRESENT_SCORE else None, jpg)
        except Exception as e:  # noqa: BLE001 - zones must never stop the watcher
            log("zones:", e)
        self.drain_events(con, jpg)
        con.execute("INSERT INTO obs VALUES(?,?,?,?,?,?,?)",
                    (t, round(score, 3), round(person, 3), json.dumps(best["box"]) if best else None,
                     round(motion, 2), int(self.present), int(moving and self.present)))
        self.accumulate(con, t, jpg, motion, moving, person >= 0.5)

        if t - self.last_imaging > IMAGING_EVERY:
            try:
                self.imaging = imaging_state()
                self.check_flapping(con, t)
                # Paused, but the camera forgot (privacy is not saved across a camera reboot): pause it again.
                self.check_audio_lost(privacy, self.imaging)
                if privacy and (self.imaging.get("privacy_enabled") == 0 or self.imaging.get("mic_enabled") == 1):
                    log("privacy: camera is not paused (reboot?), re-applying picture and mic off")
                    camapi.prudynt({"privacy": {"enabled": True}, "audio": {"mic_enabled": False}})
            except Exception as e:  # noqa: BLE001
                log("imaging:", e)
            self.last_imaging = t
        today = time.strftime("%Y-%m-%d")
        if today != self.last_house and time.localtime().tm_hour >= 3:
            self.last_house = today
            with open(HOUSE_FILE, "w") as f:
                f.write(today)
            threading.Thread(target=housekeeping, daemon=True).start()
        con.close()

        if not frozen:
            with open(HEARTBEAT, "w") as f:
                f.write(str(now()))          # not the cycle's start: a search inside it can take minutes
        self.publish(status="frozen" if frozen else "ok", score=score, moving=moving,
                     box=best["box"] if best else None, det_t=self.last_detect, motion=motion, person=person)

    def check_blocked(self, con, gray, t):
        """The camera tipped over / sagged to face a wall at 00:47 on night 1 and spent 3 h staring
        at it with nobody told. A close wall or a covered lens is blurry: low Laplacian variance."""
        if self.privacy_on():                 # paused = black on purpose, not blocked
            self.sharp_samples = []
            return
        g = gray.astype(np.float32)
        lap = -4 * g[10:-1, 1:-1] + g[9:-2, 1:-1] + g[11:, 1:-1] + g[10:-1, :-2] + g[10:-1, 2:]
        low = float(lap.var()) < BLOCKED_SHARPNESS
        self.sharp_samples = [x for x in self.sharp_samples if t - x[0] < BLOCKED_WINDOW] + [(t, low)]
        if len(self.sharp_samples) < BLOCKED_WINDOW / PERIOD * 0.6:
            return
        frac = sum(1 for _, l in self.sharp_samples if l) / len(self.sharp_samples)
        if not self.view_blocked and frac >= BLOCKED_FRACTION:
            self.view_blocked = True
            add_event(con, "view_blocked", {"low_fraction": round(frac, 2)}, None, t)
        elif self.view_blocked and frac <= 0.2:
            self.view_blocked = False
            add_event(con, "view_clear", {}, None, t)

    def check_flapping(self, con, t):
        """Day/night flapping: with the camera facing something close, its own IR light reads as
        daylight, so auto mode flips every ~30 s (2026-11-02 00:47). Each flip clicks the IR-cut
        filter and swaps the picture. On FLAP_SWITCHES flips in FLAP_WINDOW while on auto, lock
        to night for FLAP_LOCK, then hand back to auto. A mode set by hand is never overridden."""
        im = self.imaging or {}
        mode = im.get("daynight_mode")
        if self.flap_lock_until and t > self.flap_lock_until:
            self.flap_lock_until = 0.0
            self.mode_changes = []
            try:
                os.remove(FLAP_LOCK_FILE)
            except OSError:
                pass
            try:
                camapi.set_imaging("auto")
                add_event(con, "daynight_auto", {"note": "back to automatic day/night"})
            except Exception as e:  # noqa: BLE001
                log("unlock day/night:", e)
            return
        if not im.get("daynight_enabled") or self.flap_lock_until:
            self.last_mode = mode
            return
        if mode and self.last_mode and mode != self.last_mode:
            self.mode_changes.append(t)
        self.last_mode = mode
        self.mode_changes = [x for x in self.mode_changes if t - x < FLAP_WINDOW]
        if len(self.mode_changes) >= FLAP_SWITCHES:
            try:
                camapi.set_imaging("night")
                self.flap_lock_until = t + FLAP_LOCK
                with open(FLAP_LOCK_FILE, "w") as f:
                    f.write(str(self.flap_lock_until))
                add_event(con, "daynight_flapping", {"switches": len(self.mode_changes), "locked_s": FLAP_LOCK})
            except Exception as e:  # noqa: BLE001
                log("lock night:", e)

    def save_candidate(self, jpg, t):
        """Frames where the detector saw something dog-like (the pet or not), max one per 30 s:
        the raw material for the next tools/bootstrap_pet.py run. Kept 30 days."""
        if t - self.last_candidate_save < 30:
            return
        self.last_candidate_save = t
        day = time.strftime("%Y-%m-%d", time.localtime(t))
        os.makedirs(f"{CANDIDATES}/{day}", exist_ok=True)
        with open(f"{CANDIDATES}/{day}/{time.strftime('%H%M%S', time.localtime(t))}.jpg", "wb") as f:
            f.write(jpg)

    # ------------------------------------------------------------ zones (bowl / water / bed)

    def load_zones(self):
        try:
            mt = os.path.getmtime(ZONES)
        except OSError:
            self.zones = []
            return self.zones
        if mt != getattr(self, "_zones_mt", None):
            try:
                with open(ZONES) as f:
                    self.zones = json.load(f)
            except (OSError, ValueError):
                self.zones = []
            self._zones_mt = mt
        return self.zones

    def current_pose(self, t):
        """Motor position, re-read every 30 s and a few seconds after anyone moves the camera."""
        moved = 0.0
        try:
            moved = os.path.getmtime(MOVED)
        except OSError:
            pass
        stale = t - getattr(self, "pose_t", 0) > 30 or (moved > getattr(self, "pose_t", 0) and t - moved > 3)
        if stale:
            try:
                _, x, y = camapi.motor_status()
                self.pose, self.pose_t = (x, y), t
            except Exception as e:  # noqa: BLE001
                log("pose:", e)
                self.pose_t = t
        return getattr(self, "pose", None)

    def check_zones(self, con, t, best, jpg):
        """Frigate-style: the bottom-centre of the pet's (verified) box inside the zone, held for the
        zone's dwell time (one missed detection tolerated), then a cooldown before the next visit.
        Zones belong to the camera view they were drawn at, so they only count at that view."""
        zones = self.load_zones()
        if not zones:
            return
        pose = self.current_pose(t)
        state = getattr(self, "zone_state", {})
        self.zone_state = state
        point = None
        if best:
            b = best["box"]
            point = ((b[0] + b[2]) / 2, b[3])
        for z in zones:
            st = state.setdefault(z["id"], {"since": None, "last_in": 0.0, "event": None, "last_visit_end": -1e12})
            at_view = pose is not None and abs(pose[0] - z["pose"][0]) <= ZONE_POSE_TOL[0] and abs(pose[1] - z["pose"][1]) <= ZONE_POSE_TOL[1]
            r = z["rect"]
            inside = at_view and point is not None and r[0] <= point[0] <= r[2] and r[1] <= point[1] <= r[3]
            if inside:
                st["last_in"] = t
                if st["since"] is None:
                    st["since"] = t
                if (st["event"] is None and t - st["since"] >= z.get("dwell_s", 6)
                        and t - st["last_visit_end"] >= z.get("cooldown_s", 300)):
                    thumb = self.save_thumb(jpg, st["since"], "visit")
                    st["event"] = add_event(con, "visit", {"zone": z["name"], "zone_kind": z["kind"], "zone_id": z["id"],
                                                            "seconds": round(t - st["since"])}, thumb, st["since"])
                    if z["kind"] in ("water", "food"):
                        self.clips.want(st["event"], st["since"])
            elif st["since"] is not None and t - st["last_in"] > ZONE_GAP:
                if st["event"] is not None:
                    secs = round(st["last_in"] - st["since"])
                    con.execute("UPDATE events SET detail=? WHERE id=?", (json.dumps(
                        {"zone": z["name"], "zone_kind": z["kind"], "zone_id": z["id"], "seconds": secs}), st["event"]))
                    st["last_visit_end"] = st["last_in"]
                st["since"], st["event"] = None, None

    def setting(self, key, default=None):
        try:
            mt = os.path.getmtime(SETTINGS)
            if mt != getattr(self, "_settings_mt", None):
                with open(SETTINGS) as f:
                    self._settings = json.load(f)
                self._settings_mt = mt
            return self._settings.get(key, default)
        except (OSError, ValueError):
            return default

    def find_pet(self, after_t, wait=2.5):
        """For the tracker's search: the verified pet box in a frame that arrived after after_t
        (the camera has settled), or None."""
        self.frames.want()
        deadline = now() + wait
        img = None
        while now() < deadline:
            with self.frames.lock:
                t = self.frames.t
                # taken after the boundary, and a frame this search has not judged yet (looks share one boundary)
                fresh = self.frames.frame is not None and t > after_t + self.frames.lag and t > getattr(self, "last_find_t", 0)
            if fresh:
                img = self.frames.latest()
                self.last_find_t = t
                break
            time.sleep(0.05)
        if img is None:
            try:
                img = Image.open(io.BytesIO(snapshot("ch1"))).convert("RGB")
            except Exception:  # noqa: BLE001
                return None
        return self.verified_box(img, settled=True)

    def verified_box(self, img, settled=False):
        found = [f for f in self.det.detect(img, min_score=0.10) if f["label"] != "person"]
        best, judged = None, []
        thr = self.verifier.threshold if self.verifier else PRESENT_SCORE
        for f in sorted(found, key=lambda f: -f["score"])[:3]:
            p = self.verifier.prob(img, f["box"]) if self.verifier else f["score"]
            judged.append({"box": [round(v, 4) for v in f["box"]], "det": round(f["score"], 3), "p": round(p, 4)})
            if p >= thr and (best is None or p > best[0]):
                best = (p, f["box"])
        if self.tracker.searching:
            self.record_search_frame(img, judged, best[1] if best else None, settled)
        return VerifiedBox(best[1], best[0]) if best else None

    # ------------------------------------------------------------ search evidence (step 0: measure Find honestly)
    # (infrared(): a night-vision frame is grayscale, so its colour channels agree)

    def record_search_frame(self, img, judged, box, settled):
        """Keep what a running search looked at: the first settled frame after every camera move (each stop)
        and every frame the verifier accepted. Without these, "found her" could only be checked against the
        verifier's own opinion. Never raises: evidence must not break a search."""
        try:
            rec = self.search_rec
            if rec is None:
                rec = self.search_rec = {"t0": now(), "frames": [], "stop_move": None}
            new_stop = settled and rec["stop_move"] != self.tracker.last_move
            if not (box is not None or new_stop) or len(rec["frames"]) >= SEARCH_FRAMES_MAX:
                return
            if new_stop:
                rec["stop_move"] = self.tracker.last_move
            buf = io.BytesIO()
            img.convert("RGB").save(buf, "JPEG", quality=80)
            pos = None
            if new_stop:                                     # where the camera stopped; mid-pan the motor is moving
                try:
                    pos = list(self.tracker.cam.position())
                except Exception:  # noqa: BLE001
                    pass
            rec["frames"].append({"t": round(now(), 2), "jpg": buf.getvalue(), "settled": settled, "judged": judged,
                                  "verified": [round(v, 4) for v in box] if box else None, "pos": pos})
        except Exception as e:  # noqa: BLE001
            log("search evidence:", e)

    def save_search_evidence(self, result, asked):
        """Write the recorded frames of the search that just ended to SEARCHES/<day>/<HHMMSS>/."""
        rec, self.search_rec = self.search_rec, None
        if not rec:
            return
        try:
            t0 = rec["t0"]
            d = f"{SEARCHES}/{time.strftime('%Y-%m-%d', time.localtime(t0))}/{time.strftime('%H%M%S', time.localtime(t0))}"
            os.makedirs(d, exist_ok=True)
            meta = []
            for i, fr in enumerate(rec["frames"]):
                name = f"{i:02d}.jpg"
                with open(f"{d}/{name}", "wb") as f:
                    f.write(fr["jpg"])
                meta.append({k: v for k, v in fr.items() if k != "jpg"} | {"file": name})
            with open(f"{d}/search.json", "w") as f:
                json.dump({"t0": round(t0, 2), "t1": round(now(), 2), "result": result, "asked": asked,
                           "found": "found her" in result, "night": infrared(rec["frames"][0]["jpg"]) if rec["frames"] else None,
                           "confirm": f"{T.CONFIRM_FRAMES} of {T.LOOK_FRAMES}", "frames": meta}, f)
        except Exception as e:  # noqa: BLE001
            log("search evidence:", e)

    def find_pet_now(self, since):
        """For the tracker's scan while panning: the newest frame if it arrived after `since`, no waiting."""
        self.frames.want()
        with self.frames.lock:
            fresh = self.frames.frame is not None and self.frames.t > since
        if not fresh:
            return None
        img = self.frames.latest()
        return self.verified_box(img, settled=False) if img is not None else None

    def privacy_on(self):
        try:
            with open(PRIVACY) as f:
                return f.read().strip() == "1"
        except OSError:
            return False

    def camera_moved_recently(self):
        try:
            return now() - os.path.getmtime(MOVED) < 4
        except OSError:
            return False

    def check_frozen(self, con, gray, t):
        """The camera burns a ticking clock into the picture, so a live picture is never
        bit-identical a few seconds apart. 90 s of identical frames means the encoder hung
        (it did at 22:12 on 2026-11-01). Restart it; if restarts keep failing, reboot."""
        same = self.prev_small is not None and np.array_equal(gray, self.prev_small)
        if not same:
            self.same_since = None
            self.fixes = 0
            return False
        self.same_since = self.same_since or t
        if t - self.same_since < FREEZE_AFTER or t - self.last_fix < 600:
            return t - self.same_since >= FREEZE_AFTER
        self.last_fix = t
        self.fixes += 1
        if self.fixes <= 2:
            add_event(con, "camera_frozen", {"action": "restart encoder", "frozen_s": round(t - self.same_since)})
            cam_action("/x/restart-prudynt.cgi")
        else:
            add_event(con, "camera_frozen", {"action": "reboot camera", "frozen_s": round(t - self.same_since)})
            cam_action("/x/reboot.cgi")
            self.fixes = 0
        self.same_since = None
        return True

    def camera_moved_near(self, t, before=2.0, after=CAM_MOVE_GRACE):
        """Did a person or the tracker move the camera around time t? Then a pet entering or leaving
        the picture is the camera's doing (2026-11-02 10:19: 'appeared' when the owner turned the
        camera onto her sleeping), not hers."""
        try:
            m = os.path.getmtime(MOVED)
        except OSError:
            return False
        if not self.cam_moves or self.cam_moves[-1] != m:
            self.cam_moves.append(m)
        return any(t - before <= x <= t + after for x in self.cam_moves)

    def update_pet(self, con, t, score, moving, jpg, box=None):
        if score >= PRESENT_SCORE:
            if not self.present:
                gone = t - self.gone_since if self.gone_since else 1e9
                self.present = True
                self.present_since = t
                self.rest_since = t
                if gone >= APPEAR_AFTER and self.camera_moved_near(t, before=CAM_MOVE_GRACE, after=0):
                    log("appeared: not an event, the camera was just moved onto her")
                elif gone >= APPEAR_AFTER:
                    thumb = self.save_thumb(jpg, t, "appeared")
                    eid = add_event(con, "appeared", {"away_s": round(gone)}, thumb, t)
                    self.clips.want(eid, t)
            self.last_seen = t
            self.gone_since = None
        elif score < ABSENT_SCORE and self.present and t - self.last_seen > LEAVE_AFTER:
            self.present = False
            self.gone_since = self.last_seen
            stayed = round(self.last_seen - self.present_since) if self.present_since else None
            if self.camera_moved_near(self.last_seen):
                log("left: not an event, the camera was moved away from her")
            else:
                add_event(con, "left", {"stayed_s": stayed}, None, self.last_seen)

        if not self.present:
            self.moving_since = None
            return
        if score < PRESENT_SCORE and t - self.last_seen > UNSEEN_AFTER:
            # Still "present" only because of the LEAVE_AFTER debounce: not in this frame. Don't call
            # an empty picture "resting", or the motion of walking out "moving" (2026-11-02 bug: the
            # app said "resting 5m" over an empty hallway for 3 minutes after she left).
            self.moving_since = None
            self.rest_since = None
            return
        if score < PRESENT_SCORE:
            return                       # one missed detection: keep the current rest/move state
        if moving:
            if not self.moving_since:
                self.moving_since, self.move_box0 = t, box
            # "Woke up / got active": moving for 6 s after resting for 5+ minutes, AND her box shifted or
            # changed shape (standing up). A head turn or an exposure jump alone is not "active"
            # (2026-11-02 10:33: fired while she stayed lying down).
            b0 = self.move_box0
            shifted = box is None or b0 is None or \
                math.hypot((box[0] + box[2] - b0[0] - b0[2]) / 2, (box[1] + box[3] - b0[1] - b0[3]) / 2) > 0.06 or \
                abs((box[2] - box[0]) * (box[3] - box[1]) / max(1e-6, (b0[2] - b0[0]) * (b0[3] - b0[1])) - 1) > 0.3
            if self.rest_since and t - self.moving_since >= 6 and shifted:
                # A real movement run ends any rest, long or short: the next still frame starts a new
                # one (2026-11-02 19:26: "resting 2m" counted from her arrival after 90 s of moving).
                if self.moving_since - self.rest_since >= 300:
                    thumb = self.save_thumb(jpg, t, "active")
                    eid = add_event(con, "active", {"rested_s": round(self.moving_since - self.rest_since)}, thumb, self.moving_since)
                    self.clips.want(eid, self.moving_since)
                self.rest_since = None
        else:
            # One still frame ends a movement run; resting starts counting from here.
            self.moving_since = None
            if self.rest_since is None:
                self.rest_since = t

    def drain_events(self, con, jpg):
        while True:
            try:
                kind, detail, ts = self.events.get_nowait()
            except queue.Empty:
                return
            if kind == "sound":
                detail["pet_visible"] = self.present
                detail["pet_in_frame"] = bool(self.last_seen) and ts - self.last_seen < 5
                name = detail.pop("kind").replace(" ", "_")
                y = detail.get("yamnet") or {}
                if name not in DOG_SOUND_KINDS:
                    y.pop("dog_label", None)         # "Growling" at dog 0.08 read as a growl
                    y.pop("dog_kind", None)
                if name == "loud_sound" and self.owner_talking(ts):
                    log("sound: dropped, it is the owner talking through the speaker")
                    continue
                if name == "loud_sound":
                    # The cap counts the database, not memory: today's restarts reset it (14 in 28 min).
                    n = con.execute("SELECT count(*) FROM events WHERE kind='loud_sound' AND ts > ?", (ts - 3600,)).fetchone()[0]
                    if n >= LOUD_PER_HOUR:
                        continue
                if name in DOG_SOUND_KINDS:
                    # One barking episode = one event: later barks within EPISODE_GAP extend it.
                    prev = con.execute("SELECT id, detail FROM events WHERE kind IN (%s) AND ts > ? ORDER BY ts DESC LIMIT 1"
                                       % ",".join("?" * len(DOG_SOUND_KINDS)), (*DOG_SOUND_KINDS, ts - EPISODE_GAP)).fetchone()
                    if prev:
                        d = json.loads(prev[1] or "{}")
                        d["episode_barks"] = d.get("episode_barks", 1) + 1
                        d["episode_last"] = round(ts, 1)
                        d["peak_db"] = max(d.get("peak_db", -99), detail.get("peak_db", -99))
                        con.execute("UPDATE events SET detail=? WHERE id=?", (json.dumps(d), prev[0]))
                        self.dog_sound_pending = True
                        continue
                thumb = self.save_thumb(jpg, ts, "sound")
                eid = add_event(con, name, detail, thumb, ts)
                if name in DOG_SOUND_KINDS:
                    self.clips.want(eid, ts)
                    self.dog_sound_pending = True           # follow: may look for her if she isn't in view

    def owner_talking(self, ts):
        """True while the app plays the owner's voice on the camera speaker (app.talk writes the window)."""
        try:
            with open(TALKING) as f:
                start, until = (float(x) for x in f.read().split())
        except (OSError, ValueError):
            return False
        return start - 1 <= ts <= until

    def save_thumb(self, jpg, t, kind):
        day = time.strftime("%Y-%m-%d", time.localtime(t))
        os.makedirs(f"{CLIPS}/{day}", exist_ok=True)
        name = f"{day}/{time.strftime('%H%M%S', time.localtime(t))}-{kind}.jpg"
        with open(f"{CLIPS}/{name}", "wb") as f:
            f.write(jpg)
        return name

    def accumulate(self, con, t, jpg, motion, moving, person):
        minute = int(t // 60 * 60)
        if self.minute is not None and minute != self.minute:
            a = self.acc
            peak, sounds = self.audio.take_minute()
            con.execute("INSERT OR REPLACE INTO minute VALUES(?,?,?,?,?,?,?,?,?)",
                        (self.minute, a["n"], a["present"], a["moving"], a["person"],
                         round(a["motion"] / max(1, a["n"]), 2), None if peak is None else round(peak, 1),
                         sounds, a["snap"]))
        if minute != self.minute:
            self.minute = minute
            day = time.strftime("%Y-%m-%d", time.localtime(t))
            os.makedirs(f"{SNAPS}/{day}", exist_ok=True)
            snap = f"{day}/{time.strftime('%H-%M', time.localtime(t))}.jpg"
            with open(f"{SNAPS}/{snap}", "wb") as f:
                f.write(jpg)
            self.acc = {"n": 0, "present": 0, "moving": 0, "person": 0, "motion": 0.0, "snap": snap}
        a = self.acc
        a["n"] += 1
        a["present"] += int(self.present)
        a["moving"] += int(self.present and moving)
        a["person"] += int(person)
        a["motion"] += motion

    def find_requested(self, t):
        """A 'Find <pet>' tap from the app (find_request holds its time); each request is served once."""
        try:
            req = float(open(f"{STATE}/find_request").read().strip() or 0)
        except (OSError, ValueError):
            return False
        if req <= self.last_find_req or t - req > 60:
            return False
        self.last_find_req = req
        return True

    def find_cancelled(self, since):
        """'Stop looking' in the app (find_cancel holds its time): true if pressed after the search started."""
        try:
            return float(open(f"{STATE}/find_cancel").read().strip() or 0) > since
        except (OSError, ValueError):
            return False

    def find(self, con, seen):
        """Run the room search because the owner asked, even with following off. Their tap is not 'a person
        moved the camera': their own recent joystick moves don't block it."""
        try:
            self.tracker.own_move_t = max(self.tracker.own_move_t, os.path.getmtime(MOVED))
        except OSError:
            pass
        try:
            act = "search (you asked): she is already in view" if seen else self.tracker.search(reason="you asked")
        except Exception as e:  # noqa: BLE001 - must never stop the watcher
            act = f"search (you asked): failed, {type(e).__name__}"
        log("find:", act)
        add_event(con, "search", {"result": act, "asked": True})
        self.save_search_evidence(act, asked=True)
        self.find_result = {"t": now(), "result": act, "found": "found her" in act or "already in view" in act}

    def search_tick(self):
        """While a search holds the loop: keep the heartbeat and the app's status fresh, at most every 3 s."""
        t = now()
        if t - self.last_tick < 3:
            return
        self.last_tick = t
        try:
            with open(HEARTBEAT, "w") as f:
                f.write(str(t))
        except OSError:
            pass
        self.publish(status="ok", searching=True, score=0.0, moving=False, box=None, det_t=self.last_detect, motion=0.0, person=0.0)

    def publish(self, status, **kw):
        live = {"t": now(), "status": status, "present": self.present and status != "privacy",
                "last_seen": self.last_seen or None, "gone_since": self.gone_since,
                "rest_since": self.rest_since if self.present else None,
                "moving_since": self.moving_since, "imaging": self.imaging,
                "audio_ok": now() - self.audio.last_ok < 30, "view_blocked": self.view_blocked,
                "find": self.find_result, **kw}
        try:
            write_json(LIVE, live)
        except OSError as e:
            log("publish:", e)


if __name__ == "__main__":
    os.makedirs(STATE, exist_ok=True)
    Watch().start()
