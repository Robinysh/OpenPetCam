#!/usr/bin/env python3
"""petcam web app: password login, pan/tilt proxy, last-hour recording list.

Standard library only. Listens on 127.0.0.1:8080 behind Caddy, which also asks
GET /auth before serving any video. Secrets come from /etc/petcam/app.env:
  APP_PASSWORD_HASH  pbkdf2_sha256$<iters>$<salt hex>$<hash hex>
  APP_SECRET         random hex, signs the session cookie
  CAM_WEB_USER / CAM_WEB_PASS   the camera's web login (for motor calls)
Non-secret settings (pet name, addresses) come from /etc/petcam/site.env via petconf.py.
"""
import base64
import hashlib
import hmac
import http.client
import json
import os
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import petconf

HERE = os.path.dirname(os.path.abspath(__file__))
SESSION_DAYS = 30
COOKIE = "petcam_session"


def load_env(path="/etc/petcam/app.env"):
    env = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip()
    except FileNotFoundError:
        pass
    return env


ENV = load_env()
SECRET = ENV.get("APP_SECRET", "").encode()
CAM_HOST = ENV.get("CAM_HOST") or petconf.CAM_HOST


def with_deadline(fn, seconds):
    """Run fn() with a real wall-clock bound; a socket timeout alone is not one."""
    box = {}

    def run():
        try:
            box["ok"] = fn()
        except Exception as e:  # noqa: BLE001 - reported to caller
            box["err"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        raise TimeoutError(f"no answer within {seconds}s")
    if "err" in box:
        raise box["err"]
    return box.get("ok")


# ---------- auth ----------

def check_password(pw):
    stored = ENV.get("APP_PASSWORD_HASH", "")
    try:
        algo, iters, salt, want = stored.split("$")
        assert algo == "pbkdf2_sha256"
    except (ValueError, AssertionError):
        return False
    got = hashlib.pbkdf2_hmac("sha256", pw.encode(), bytes.fromhex(salt), int(iters)).hex()
    return hmac.compare_digest(got, want)


def make_session():
    exp = str(int(time.time()) + SESSION_DAYS * 86400)
    sig = hmac.new(SECRET, exp.encode(), hashlib.sha256).hexdigest()
    return f"{exp}.{sig}"


def session_ok(cookie_header):
    if not SECRET or not cookie_header:
        return False
    for part in cookie_header.split(";"):
        name, _, value = part.strip().partition("=")
        if name == COOKIE and "." in value:
            exp, sig = value.split(".", 1)
            good = hmac.new(SECRET, exp.encode(), hashlib.sha256).hexdigest()
            if hmac.compare_digest(sig, good) and exp.isdigit() and int(exp) > time.time():
                return True
    return False


FAILS = {}  # ip -> [count, locked_until]
FAILS_LOCK = threading.Lock()


def login_locked(ip):
    with FAILS_LOCK:
        count, until = FAILS.get(ip, [0, 0])
        return until > time.time()


def note_failure(ip):
    with FAILS_LOCK:
        count, until = FAILS.get(ip, [0, 0])
        count += 1
        until = time.time() + 15 * 60 if count >= 5 else until
        FAILS[ip] = [0 if count >= 5 else count, until]


def note_success(ip):
    with FAILS_LOCK:
        FAILS.pop(ip, None)


# ---------- remembered devices ----------
# Each phone/laptop gets its own random 256-bit key, stored server-side only as a SHA-256 hash,
# in an HttpOnly Secure cookie that lasts 400 days and is refreshed on every visit. Any device
# can be revoked from the app. New people join with a single-use invite link (24 h).

DB_PATH = "/var/lib/petcam/petcam.db"
DEVICE_COOKIE = "petcam_device"
DEVICE_DAYS = 400   # browsers cap cookie lifetime at 400 days; refreshed on each visit


def dbc():
    import sqlite3

    con = sqlite3.connect(DB_PATH, timeout=10, isolation_level=None)
    con.execute("PRAGMA busy_timeout=5000")
    con.row_factory = sqlite3.Row
    return con


def init_auth_db():
    con = dbc()
    con.executescript("""
    PRAGMA journal_mode=WAL;
    CREATE TABLE IF NOT EXISTS devices(id TEXT PRIMARY KEY, name TEXT, token_hash TEXT, created REAL,
                                       last_seen REAL, last_ip TEXT, ua TEXT, revoked INT DEFAULT 0);
    CREATE TABLE IF NOT EXISTS invites(token_hash TEXT PRIMARY KEY, name TEXT, created REAL, expires REAL,
                                       used_at REAL, created_by TEXT);
    CREATE TABLE IF NOT EXISTS push_subs(endpoint TEXT PRIMARY KEY, device_id TEXT, sub TEXT, prefs TEXT, created REAL);
    """)
    # Human feedback on moments ("was this the pet?") = ground truth for tuning the detectors.
    cols = [r[1] for r in con.execute("PRAGMA table_info(events)")]
    if cols and "label" not in cols:
        con.execute("ALTER TABLE events ADD COLUMN label TEXT")
        con.execute("ALTER TABLE events ADD COLUMN labelled_by TEXT")
    con.close()


def label_event(event_id, label, device_id):
    if label not in ("yes", "no", None):
        raise ValueError("label must be yes, no or null")
    con = dbc()
    con.execute("UPDATE events SET label=?, labelled_by=? WHERE id=?", (label, device_id, int(event_id)))
    con.close()
    return {"ok": True, "label": label}


# ---------- push notifications (sent by alert.py, stored here) ----------

PUSH_DEFAULTS = {"camera": True, "barking": True, "daily": True, "comings": False}


def push_key():
    with open("/etc/petcam/vapid_public.txt") as f:
        return {"key": f.read().strip()}


def push_subscribe(device_id, sub, prefs):
    if not isinstance(sub, dict) or not str(sub.get("endpoint", "")).startswith("https://"):
        raise ValueError("bad subscription")
    keys = sub.get("keys") or {}
    if not keys.get("p256dh") or not keys.get("auth"):
        raise ValueError("bad subscription keys")
    p = {k: bool((prefs or {}).get(k, v)) for k, v in PUSH_DEFAULTS.items()}
    con = dbc()
    con.execute("INSERT OR REPLACE INTO push_subs VALUES(?,?,?,?,?)",
                (sub["endpoint"], device_id, json.dumps(sub), json.dumps(p), time.time()))
    con.close()
    return {"ok": True, "prefs": p}


def push_status(device_id):
    con = dbc()
    try:
        row = con.execute("SELECT prefs FROM push_subs WHERE device_id=? ORDER BY created DESC LIMIT 1", (device_id,)).fetchone()
        return {"subscribed": bool(row), "prefs": json.loads(row["prefs"]) if row else PUSH_DEFAULTS}
    finally:
        con.close()


def push_unsubscribe(device_id):
    con = dbc()
    con.execute("DELETE FROM push_subs WHERE device_id=?", (device_id,))
    con.close()
    return {"ok": True}


def push_test(device_id):
    import subprocess

    r = subprocess.run(["timeout", "-k", "5", "40", "/opt/petcam/venv/bin/python", "/opt/petcam/alert.py", "--test", device_id or ""],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError((r.stdout + r.stderr).strip()[-200:] or "push failed")
    return {"ok": True, "result": r.stdout.strip()[-200:]}


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def device_name_from_ua(ua):
    ua = ua or ""
    for key, name in (("iPhone", "iPhone"), ("iPad", "iPad"), ("Android", "Android phone"),
                      ("Macintosh", "Mac"), ("Windows", "Windows PC")):
        if key in ua:
            return name
    return "Browser"


def new_device(name, ip, ua):
    import secrets

    dev_id, token = secrets.token_hex(8), secrets.token_urlsafe(32)
    con = dbc()
    con.execute("INSERT INTO devices(id,name,token_hash,created,last_seen,last_ip,ua) VALUES(?,?,?,?,?,?,?)",
                (dev_id, (name or device_name_from_ua(ua))[:40], sha(token), time.time(), time.time(), ip, (ua or "")[:200]))
    con.close()
    return f"{dev_id}.{token}"


def device_cookie(value):
    return f"{DEVICE_COOKIE}={value}; Path=/; Max-Age={DEVICE_DAYS * 86400}; HttpOnly; Secure; SameSite=Lax"


def cookie_value(cookie_header, name):
    for part in (cookie_header or "").split(";"):
        k, _, v = part.strip().partition("=")
        if k == name:
            return v
    return None


_seen_cache = {}


def device_from_cookie(cookie_header, ip=None):
    """Returns the device row (dict) for a valid, unrevoked device cookie, else None."""
    value = cookie_value(cookie_header, DEVICE_COOKIE)
    if not value or "." not in value:
        return None
    dev_id, token = value.split(".", 1)
    con = dbc()
    try:
        row = con.execute("SELECT * FROM devices WHERE id=? AND revoked=0", (dev_id,)).fetchone()
        if not row or not hmac.compare_digest(row["token_hash"], sha(token)):
            return None
        if time.time() - _seen_cache.get(dev_id, 0) > 300:
            _seen_cache[dev_id] = time.time()
            con.execute("UPDATE devices SET last_seen=?, last_ip=? WHERE id=?", (time.time(), ip, dev_id))
        return dict(row)
    finally:
        con.close()


def create_invite(name, created_by):
    import secrets

    token = secrets.token_urlsafe(24)
    con = dbc()
    con.execute("INSERT INTO invites VALUES(?,?,?,?,NULL,?)",
                (sha(token), (name or "Guest")[:40], time.time(), time.time() + 86400, created_by))
    con.close()
    return token


def use_invite(token, ip, ua):
    """Single use: the invite is marked used in the same statement that checks it."""
    con = dbc()
    try:
        cur = con.execute("UPDATE invites SET used_at=? WHERE token_hash=? AND used_at IS NULL AND expires>?",
                          (time.time(), sha(token), time.time()))
        if cur.rowcount != 1:
            return None, None
        name = con.execute("SELECT name FROM invites WHERE token_hash=?", (sha(token),)).fetchone()["name"]
    finally:
        con.close()
    return new_device(f"{name}'s {device_name_from_ua(ua)}", ip, ua), name


def invite_valid(token):
    con = dbc()
    try:
        row = con.execute("SELECT name FROM invites WHERE token_hash=? AND used_at IS NULL AND expires>?",
                          (sha(token), time.time())).fetchone()
        return row["name"] if row else None
    finally:
        con.close()


def list_devices(current_id):
    con = dbc()
    try:
        rows = con.execute("SELECT id,name,created,last_seen,ua FROM devices WHERE revoked=0 ORDER BY last_seen DESC").fetchall()
        return [{**dict(r), "this": r["id"] == current_id} for r in rows]
    finally:
        con.close()


def revoke_device(dev_id):
    con = dbc()
    con.execute("UPDATE devices SET revoked=1 WHERE id=?", (dev_id,))
    con.execute("DELETE FROM push_subs WHERE device_id=?", (dev_id,))
    con.close()
    return {"ok": True}


def rename_device(dev_id, name):
    name = " ".join(str(name).split())[:40]
    if not name:
        raise ValueError("name required")
    con = dbc()
    con.execute("UPDATE devices SET name=? WHERE id=? AND revoked=0", (name, dev_id))
    con.close()
    return {"ok": True}


# ---------- camera ----------

_cam = {"cookie": None}
_cam_lock = threading.Lock()


def cam_login():
    """Thingino's web login: POST JSON with a base64 password, returns a session cookie."""
    body = json.dumps({
        "username": ENV.get("CAM_WEB_USER", "root"),
        "password": base64.b64encode(ENV.get("CAM_WEB_PASS", "").encode()).decode(),
        "encoding": "base64",
    }, separators=(",", ":")).encode()  # compact: the camera's JSON parser chokes on spaces
    def call():
        conn = http.client.HTTPConnection(CAM_HOST, 80, timeout=5)
        try:
            conn.request("POST", "/x/login.cgi", body=body,
                         headers={"Content-Type": "application/json", "Content-Length": str(len(body))})
            r = conn.getresponse()
            result = json.loads(r.read(65536) or b"{}")
            cookies = [c.split(";", 1)[0] for c in r.headers.get_all("Set-Cookie") or []]
            return r.status, result, "; ".join(cookies)
        finally:
            conn.close()

    status, result, cookie = with_deadline(call, 8)
    if status != 200 or not result.get("success") or not cookie:
        raise PermissionError(f"camera login refused ({status}: {result.get('error', {}).get('message', 'no session') if isinstance(result.get('error'), dict) else 'no session'})")
    _cam["cookie"] = cookie


def cam_get(path, query=None):
    url = f"http://{CAM_HOST}{path}"
    if query:
        url += "?" + urllib.parse.urlencode(query)

    def call():
        req = urllib.request.Request(url, headers={"Cookie": _cam["cookie"] or ""})
        with urllib.request.urlopen(req, timeout=5) as r:
            return json.loads(r.read(65536) or b"{}")

    with _cam_lock:
        if not _cam["cookie"]:
            cam_login()
    try:
        return with_deadline(call, 8)
    except urllib.error.HTTPError as e:
        if e.code != 401:
            raise
        with _cam_lock:  # session expired: log in again once
            cam_login()
        return with_deadline(call, 8)


_params = {"t": 0, "v": None}


def motor_params():
    if _params["v"] is None or time.time() - _params["t"] > 600:
        _params["v"] = cam_get("/x/json-motor-params.cgi")
        _params["t"] = time.time()
    return _params["v"]


def num(p, *keys, default):
    for k in keys:
        try:
            return float(p[k])
        except (KeyError, TypeError, ValueError):
            continue
    return default


# Motor geometry, measured 2026-11-01 on this camera (mounted upside down, image flipped 180°)
# by moving a known number of steps and phase-correlating frames before and after.
#
# Everything below works in REPORTED coordinates (what d=j returns, 0..steps_* per axis):
#   - one frame width  ~ 733 pan steps;  view turns RIGHT as reported x DEcreases
#   - one frame height ~ 424 tilt steps; view tilts UP   as reported y INcreases
# The camera's motor config has tilt inverted ("invert": 2): a tilt COMMAND moves the reported
# position the opposite way, for both relative (d=g) and absolute (d=x) moves. Absolute tilt
# commands therefore land mirrored around the current position (asked for y=0 from 760 it ran
# to 1030, past its 1000 limit, and then refused every tilt move). So every move here is a
# clamped target in reported coordinates, sent as a relative d=g step with the tilt sign flipped.
PAN_STEPS_PER_FRAME = float(ENV.get("PAN_STEPS_PER_FRAME", 733))
TILT_STEPS_PER_FRAME = float(ENV.get("TILT_STEPS_PER_FRAME", 424))
TILT_COMMAND_SIGN = -1 if ENV.get("TILT_COMMAND_INVERTED", "1") == "1" else 1
TILT_MARGIN = 40


def limits():
    p = motor_params() or {}
    return int(num(p, "steps_pan", "x_max", default=3700)), int(num(p, "steps_tilt", "y_max", default=1000))


def start_pose():
    p = motor_params() or {}
    xm, ym = limits()
    return int(num(p, "pos_0_x", default=xm / 2)), int(num(p, "pos_0_y", default=ym / 2))


def note_camera_moved():
    """Tell petcam-watch the picture is about to shift because WE moved it, not the pet."""
    try:
        with open("/var/lib/petcam/moved_at", "a"):
            pass
        os.utime("/var/lib/petcam/moved_at")
    except OSError:
        pass


def motor_status():
    m = cam_get("/x/json-motor.cgi", {"d": "j"}).get("message") or {}
    return str(m.get("status")), int(float(m["xpos"])), int(float(m["ypos"]))


def position():
    _, x, y = motor_status()
    return x, y


_move_lock = threading.Lock()   # one move at a time, each computed from a settled position
_homing = {"until": 0.0}


class Busy(Exception):
    pass


def wait_idle(timeout=6.0):
    """Relative moves must start from where the motors actually stopped; a position read
    mid-move (or during homing) sends the next move to the wrong place."""
    if time.time() < _homing["until"]:
        raise Busy("Recalibrating — try again in a few seconds")
    deadline = time.time() + timeout
    while True:
        status, x, y = motor_status()
        if status == "0":
            return x, y
        if time.time() > deadline:
            raise Busy("Camera is still moving")
        time.sleep(0.2)


def goto(x, y):
    """Move to reported position (x, y), clamped inside the motor range so no axis can run past
    its end stop. Sent as a relative step from the settled position (see the note above)."""
    if _joystick is not None:
        _joystick.forget()
    with _move_lock:
        return _goto_locked(x, y, wait_idle())


def _goto_locked(x, y, current):
    xm, ym = limits()
    cx, cy = current
    if not (0 <= cy <= ym) or not (0 <= cx <= xm):
        # An axis already past its end stop refuses every move on that axis; only homing resets it.
        recalibrate()
        raise Busy("The camera was past its end stop, so it is recalibrating (about 30 s)")
    # Keep tilt off its very ends: with the inverted range check below, a tilt sitting at 0
    # could never be commanded upward again.
    tx = max(0, min(xm, int(round(x))))
    ty = max(TILT_MARGIN, min(ym - TILT_MARGIN, int(round(y))))
    # The motor driver range-checks a relative tilt step as if it went the un-inverted way:
    # it accepts cmd only while cy + cmd stays in [0, ym], yet the head moves to cy - cmd.
    # (Measured: from 415 asked for 1000 it stopped at 830; from 830 asked for 0, at 660.)
    # So one step can reach only [2*cy - ym, 2*cy]; longer tilts go in a few short steps.
    for _ in range(6):
        step_y = ty
        if TILT_COMMAND_SIGN == -1:
            step_y = max(2 * cy - ym, min(2 * cy, ty))
        dx, dy = tx - cx, (step_y - cy) * TILT_COMMAND_SIGN
        if dx or dy:
            note_camera_moved()
            cam_get("/x/json-motor.cgi", {"d": "g", "x": dx, "y": dy})
        if step_y == ty:
            break
        time.sleep(0.15)
        cx, cy = wait_idle()
    return {"x": tx, "y": ty}


def aim(fx, fy):
    """Bring the point at (fx, fy) of the frame to the centre. fx/fy are offsets from the
    centre as a fraction of the frame: +fx is right of centre, +fy is below centre."""
    fx, fy = max(-0.75, min(0.75, float(fx))), max(-0.75, min(0.75, float(fy)))
    if _joystick is not None:
        _joystick.forget()
    with _move_lock:
        cx, cy = wait_idle()
        # Right of centre -> turn right -> x down. Below centre -> tilt down -> y down.
        return _goto_locked(cx - fx * PAN_STEPS_PER_FRAME, cy - fy * TILT_STEPS_PER_FRAME, (cx, cy))


def recalibrate():
    """Homing sweeps both axes to their end stops and back to the start pose. The CGI answers
    only when it is done; moves are refused until then so they can't overlap the sweep."""
    if time.time() < _homing["until"]:
        return
    _homing["until"] = time.time() + 60
    if _joystick is not None:
        _joystick.forget()
    if not _cam["cookie"]:
        with _cam_lock:
            cam_login()
    cookie = _cam["cookie"]

    def run():
        try:
            req = urllib.request.Request(f"http://{CAM_HOST}/x/json-motor.cgi?d=r", headers={"Cookie": cookie or ""})
            urllib.request.urlopen(req, timeout=60).read(1000)
            time.sleep(1)
            end = time.time() + 30
            while time.time() < end and motor_status()[0] != "0":
                time.sleep(0.5)
        except Exception as e:  # noqa: BLE001 - background, best effort
            print(f"recalibrate: {e}", flush=True)
        finally:
            _homing["until"] = 0.0

    threading.Thread(target=run, daemon=True).start()


# ---------- saved spots ----------

PRESETS_FILE = "/var/lib/petcam/presets.json"
_presets_lock = threading.Lock()


def load_presets():
    try:
        with open(PRESETS_FILE) as f:
            data = json.load(f)
        return [p for p in data if isinstance(p, dict) and {"name", "x", "y"} <= p.keys()]
    except (FileNotFoundError, ValueError):
        return []


def save_presets(items):
    tmp = PRESETS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(items, f)
    os.replace(tmp, PRESETS_FILE)


def add_preset(name):
    name = " ".join(str(name).split())[:24]
    if not name:
        raise ValueError("name required")
    x, y = position()
    with _presets_lock:
        items = [p for p in load_presets() if p["name"].lower() != name.lower()]
        if len(items) >= 12:
            raise ValueError("12 saved spots at most; delete one first")
        items.append({"name": name, "x": x, "y": y})
        save_presets(items)
    return items


def delete_preset(name):
    with _presets_lock:
        items = [p for p in load_presets() if p["name"] != name]
        save_presets(items)
    return items


def state():
    xm, ym = limits()
    x, y = position()
    return {"x": x, "y": y, "x_max": xm, "y_max": ym, "start": start_pose(),
            "frame": [PAN_STEPS_PER_FRAME, TILT_STEPS_PER_FRAME], "presets": load_presets()}


class Joystick:
    """Joystick moves, made to feel instant on a slow camera.

    The camera's motor CGI takes ~200-400 ms per call. So /api/nudge answers right away and a
    single background sender talks to the camera. Ticks that arrive while it is busy are
    ADDED to one pending move (never queued), so the camera can't fall behind the thumb, and
    a released stick stops within one camera call. The position is tracked locally from our
    own targets and re-read from the camera only when the estimate is stale.
    """
    STALE = 4.0   # seconds after which the local position estimate is re-read

    def __init__(self):
        self.cv = threading.Condition()
        self.pending = [0, 0]          # requested view-steps not yet sent (reported coords)
        self.pos = None                # (x, y) estimate
        self.pos_t = 0.0
        self.error = None
        threading.Thread(target=self.run, daemon=True).start()

    def add(self, sx, sy):
        if time.time() < _homing["until"]:
            raise Busy("Recalibrating — try again in a few seconds")
        if self.error:
            err, self.error = self.error, None
            raise err
        with self.cv:
            self.pending[0] += sx
            self.pending[1] += sy
            self.cv.notify()
        return {"ok": True, "pos": self.pos}

    def run(self):
        while True:
            with self.cv:
                while not any(self.pending):
                    self.cv.wait()
                sx, sy = self.pending
                self.pending = [0, 0]
            try:
                with _move_lock:
                    self.send(sx, sy)
            except Exception as e:  # noqa: BLE001 - surfaced on the next tick
                self.error = e if isinstance(e, Busy) else RuntimeError(f"camera: {e}")
                self.pos = None

    def send(self, sx, sy):
        if self.pos is None or time.time() - self.pos_t > self.STALE:
            _, cx, cy = motor_status()
        else:
            cx, cy = self.pos
        xm, ym = limits()
        if not (0 <= cy <= ym) or not (0 <= cx <= xm):
            recalibrate()
            raise Busy("The camera was past its end stop, so it is recalibrating (about 15 s)")
        tx = max(0, min(xm, cx + sx))
        ty = max(TILT_MARGIN, min(ym - TILT_MARGIN, cy + sy))
        if TILT_COMMAND_SIGN == -1:            # the driver's mirrored range check, see goto()
            ty = max(2 * cy - ym, min(2 * cy, ty))
        dx, dy = tx - cx, (ty - cy) * TILT_COMMAND_SIGN
        if dx or dy:
            note_camera_moved()
            cam_get("/x/json-motor.cgi", {"d": "g", "x": dx, "y": dy})
        self.pos, self.pos_t = (tx, ty), time.time()

    def forget(self):
        """Absolute moves (aim/goto/homing) invalidate the local estimate."""
        self.pos = None


_joystick = None


def nudge(vx, vy):
    """Joystick tick (~6/s while held). vx/vy in [-1, 1]; +vx = view right, +vy = view up."""
    global _joystick
    vx, vy = max(-1.0, min(1.0, float(vx))), max(-1.0, min(1.0, float(vy)))
    # Squared deflection: gentle near the centre, faster at the edge.
    sx = int(round(-vx * abs(vx) * 0.16 * PAN_STEPS_PER_FRAME))     # view right = x down
    sy = int(round(vy * abs(vy) * 0.16 * TILT_STEPS_PER_FRAME))     # view up = y up
    if not sx and not sy:
        return {"ok": True}
    if _joystick is None:
        _joystick = Joystick()
    return _joystick.add(sx, sy)


# ---------- picture modes (day / night / IR) ----------

IMAGING = {
    "auto": {"cmd": "auto", "val": 1},
    "day": {"cmd": "daynight", "val": "day"},
    "night": {"cmd": "daynight", "val": "night"},
    "ir_on": {"cmd": "ir940", "val": 1},
    "ir_off": {"cmd": "ir940", "val": 0},
    "filter_on": {"cmd": "ircut", "val": 1},
    "filter_off": {"cmd": "ircut", "val": 0},
    "color": {"cmd": "color", "val": 0},
    "bw": {"cmd": "color", "val": 1},
}


def set_imaging(action):
    if action not in IMAGING:
        raise ValueError("unknown mode")
    body = json.dumps(IMAGING[action], separators=(",", ":")).encode()  # compact: see cam_login

    def call():
        conn = http.client.HTTPConnection(CAM_HOST, 80, timeout=5)
        try:
            conn.request("POST", "/x/json-imp.cgi", body=body,
                         headers={"Content-Type": "application/json", "Content-Length": str(len(body)),
                                  "Cookie": _cam["cookie"] or ""})
            r = conn.getresponse()
            return r.status, r.read(4000).decode(errors="replace")
        finally:
            conn.close()

    with _cam_lock:
        if not _cam["cookie"]:
            cam_login()
    status, text = with_deadline(call, 8)
    if status == 401:
        with _cam_lock:
            cam_login()
        status, text = with_deadline(call, 8)
    if status != 200:
        raise RuntimeError(f"camera answered {status}")
    return {"ok": True, "action": action}


# ---------- privacy (pause the camera) ----------

_api_key = {"v": None}


def prudynt(body):
    """POST to the camera's streamer API (port 8080), authenticated with its API key."""
    def call(key):
        req = urllib.request.Request(f"http://{CAM_HOST}:8080/api/v1/config",
                                     data=json.dumps(body, separators=(",", ":")).encode(),
                                     headers={"Content-Type": "application/json", "X-API-Key": key}, method="POST")
        with urllib.request.urlopen(req, timeout=8) as r:
            return json.loads(r.read(65536) or b"{}")

    for attempt in (1, 2):
        if not _api_key["v"]:
            _api_key["v"] = cam_get("/x/api-key.cgi")["api_key"]
        try:
            return with_deadline(lambda: call(_api_key["v"]), 10)
        except urllib.error.HTTPError as e:
            if e.code in (401, 403) and attempt == 1:
                _api_key["v"] = None
                continue
            raise


def set_privacy(on):
    """Black out the picture (Thingino privacy mode). The watcher reads the flag file and stops
    detecting while it is on. Tested 2026-11-01: toggling it does not hang the encoder."""
    on = bool(on)
    r = prudynt({"privacy": {"enabled": on}})
    state = bool((r.get("privacy") or {}).get("enabled", on))
    with open(f"{STATE_DIR}/privacy", "w") as f:
        f.write("1" if state else "0")
    return {"privacy": state}


# ---------- small settings shared with the watcher ----------

SETTING_KEYS = {"follow": bool, "talk_volume": int}
SETTING_DEFAULTS = {"follow": False, "talk_volume": 75}


def get_settings():
    try:
        with open(f"{STATE_DIR}/settings.json") as f:
            data = json.load(f)
    except (OSError, ValueError):
        data = {}
    return {k: t(data.get(k, SETTING_DEFAULTS[k])) for k, t in SETTING_KEYS.items()}


def set_settings(body):
    cur = get_settings()
    for k, t in SETTING_KEYS.items():
        if k in body:
            cur[k] = t(body[k])
    cur["talk_volume"] = max(20, min(100, cur["talk_volume"]))
    tmp = f"{STATE_DIR}/settings.json.tmp"
    with open(tmp, "w") as f:
        json.dump(cur, f)
    os.replace(tmp, f"{STATE_DIR}/settings.json")
    return cur


# ---------- push-to-talk (camera speaker) ----------
# Browser recording -> cleaned 16 kHz WAV with a soft chime first -> the camera fetches it over the
# WireGuard tunnel and plays it with Thingino's `play` (prudynt audio-out FIFO). Verified
# 2026-11-02: a server-made chime was heard back by the camera's own mic.

TALK_DIR = "/var/lib/petcam/talk"
TALK_BIND = (petconf.SERVER_WG_IP, 8099)            # tunnel address only: the camera is the only client
_talk_last = {}


def camera_run(cmd, timeout=25):
    """Run a shell command on the camera through Thingino's web command runner (base64 cmd)."""
    def call():
        url = f"http://{CAM_HOST}/x/run.cgi?cmd=" + base64.b64encode(cmd.encode()).decode()
        req = urllib.request.Request(url, headers={"Cookie": _cam["cookie"] or ""})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read(20000).decode(errors="replace")
    with _cam_lock:
        if not _cam["cookie"]:
            cam_login()
    try:
        return with_deadline(call, timeout + 5)
    except urllib.error.HTTPError as e:
        if e.code != 401:
            raise
        with _cam_lock:
            cam_login()
        return with_deadline(call, timeout + 5)


def ensure_chime():
    path = f"{TALK_DIR}/chime.wav"
    if not os.path.exists(path):
        import subprocess
        subprocess.run(["timeout", "-k", "5", "20", "ffmpeg", "-nostdin", "-v", "error", "-y",
                        "-f", "lavfi", "-i", "sine=frequency=880:duration=0.18", "-f", "lavfi", "-i", "sine=frequency=660:duration=0.28",
                        "-filter_complex", "[0][1]concat=n=2:v=0:a=1,afade=t=in:d=0.02,afade=t=out:st=0.36:d=0.1,volume=0.35",
                        "-ac", "1", "-ar", "16000", path], check=True)
    return path


def talk(device_id, device_name, raw, content_type):
    import secrets
    import subprocess

    if read_live().get("status") == "privacy":
        raise Busy(f"The camera is paused. Resume it to talk to {petconf.PET}.")
    if time.time() - _talk_last.get(device_id, 0) < 5:
        raise Busy("One message every 5 seconds")
    if len(raw) < 2000:
        raise ValueError("That was too short. Hold the button while you speak.")
    _talk_last[device_id] = time.time()
    os.makedirs(TALK_DIR, exist_ok=True)
    tok = secrets.token_urlsafe(16)
    src, voice, out = f"{TALK_DIR}/{tok}.in", f"{TALK_DIR}/{tok}.voice.wav", f"{TALK_DIR}/{tok}.wav"
    with open(src, "wb") as f:
        f.write(raw)
    try:
        # Max 15 s; drop rumble below 120 Hz; even out the level so a quiet phone mic is still heard.
        subprocess.run(["timeout", "-k", "5", "30", "nice", "-n", "10", "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", src,
                        "-t", "15", "-af", "highpass=f=120,dynaudnorm=f=150:g=15,volume=0.9", "-ac", "1", "-ar", "16000", voice],
                       check=True, capture_output=True)
        subprocess.run(["timeout", "-k", "5", "20", "ffmpeg", "-nostdin", "-v", "error", "-y", "-i", ensure_chime(), "-i", voice,
                        "-filter_complex", "[0][1]concat=n=2:v=0:a=1", "-ac", "1", "-ar", "16000", out], check=True, capture_output=True)
        dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", voice],
                                   capture_output=True, text=True).stdout.strip() or 0)
        vol = int(get_settings().get("talk_volume", 75))
        reply = camera_run(f"curl -s -m 20 http://{TALK_BIND[0]}:{TALK_BIND[1]}/{tok}.wav | play -s -f wav -v {vol} 2>&1; echo rc=$?")
        if "Queued" not in reply:
            raise RuntimeError("the camera did not accept the audio")
        con = dbc()
        con.execute("INSERT INTO events(ts, kind, detail) VALUES(?,?,?)",
                    (time.time(), "you_spoke", json.dumps({"by": device_name, "seconds": round(dur, 1)})))
        con.close()
        return {"ok": True, "seconds": round(dur, 1)}
    finally:
        def cleanup():
            time.sleep(90)                      # the camera fetches within seconds
            for f in (src, voice, out):
                try:
                    os.remove(f)
                except OSError:
                    pass
        threading.Thread(target=cleanup, daemon=True).start()


class TalkFiles(BaseHTTPRequestHandler):
    """Serves /<token>.wav from TALK_DIR on the tunnel address, nothing else."""

    def do_GET(self):
        name = self.path.lstrip("/")
        if "/" in name or not name.endswith(".wav") or name == "chime.wav" or len(name) < 20:
            self.send_error(404)
            return
        try:
            with open(f"{TALK_DIR}/{name}", "rb") as f:
                data = f.read()
        except OSError:
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *a):
        pass


def serve_talk_files():
    """Bind to the WireGuard address; retry until wg0 is up (it may come up after us at boot)."""
    while True:
        try:
            ThreadingHTTPServer(TALK_BIND, TalkFiles).serve_forever()
        except OSError as e:
            print(f"talk file server: {e}; retrying in 15 s", flush=True)
            time.sleep(15)


# ---------- the pet: live status, days, diary ----------

STATE_DIR = "/var/lib/petcam"


def read_live():
    try:
        with open(f"{STATE_DIR}/live.json") as f:
            live = json.load(f)
    except (OSError, ValueError):
        live = {"status": "starting"}
    try:
        live["watch_age"] = round(time.time() - os.path.getmtime(f"{STATE_DIR}/live.json"), 1)
    except OSError:
        live["watch_age"] = None
    if live.get("watch_age") is not None and live["watch_age"] > 20:
        live["status"] = "watcher_down"
    return live


def day_bounds(day):
    """Local midnight to next local midnight. NOT start + 86400: DST days are 23 or 25 hours
    (2026-11-01 was 25 h, and the extra hour broke the timeline)."""
    import datetime as dt

    d = dt.date.fromisoformat(day) if day else dt.date.today()
    start = time.mktime(d.timetuple())
    end = time.mktime((d + dt.timedelta(days=1)).timetuple())
    return d.isoformat(), start, end


def bucket_of(ts):
    """10-minute slot (0..143) by local clock time, so a 25-hour day folds its repeated hour."""
    lt = time.localtime(ts)
    return (lt.tm_hour * 60 + lt.tm_min) // 10


def day_report(day=None):
    day, start, end = day_bounds(day)
    con = dbc()
    try:
        rows = con.execute("SELECT * FROM minute WHERE ts>=? AND ts<? ORDER BY ts", (start, end)).fetchall()
        events = con.execute("SELECT * FROM events WHERE ts>=? AND ts<? ORDER BY ts DESC LIMIT 200",
                             (start, end)).fetchall()
    finally:
        con.close()
    buckets = [{"present": 0, "moving": 0, "n": 0, "sounds": 0, "person": 0, "snap": None} for _ in range(144)]
    visible = active = 0
    rest_run = longest_rest = 0
    first = last = None
    for r in rows:
        b = buckets[bucket_of(r["ts"])]
        n = max(1, r["n"])
        pres, mov = r["present"] / n, r["moving"] / n
        b["n"] += 1
        b["present"] += pres
        b["moving"] += mov
        b["sounds"] += r["sounds"] or 0
        b["person"] += (r["person"] or 0) / n
        b["snap"] = r["snap"] or b["snap"]
        if pres >= 0.5:
            visible += 1
            first = first or r["ts"]
            last = r["ts"]
            if mov >= 0.15:
                active += 1
                rest_run = 0
            else:
                rest_run += 1
                longest_rest = max(longest_rest, rest_run)
        else:
            rest_run = 0
    for b in buckets:
        if b["n"]:
            b["present"] = round(b["present"] / b["n"], 2)
            b["moving"] = round(b["moving"] / b["n"], 2)
            b["person"] = round(b["person"] / b["n"], 2)
    ev = [{"id": e["id"], "ts": e["ts"], "kind": e["kind"], "detail": json.loads(e["detail"] or "{}"),
           "label": e["label"] if "label" in e.keys() else None,
           "clip": f"/media/clips/{e['clip']}" if e["clip"] else None,
           "thumb": f"/media/clips/{e['thumb']}" if e["thumb"] else None} for e in events]
    dog_kinds = ("barking", "howling", "whining", "growling", "dog_sound")
    sounds = sum(1 for e in ev if e["kind"] in dog_kinds + ("loud_sound",))
    barks = sum(1 for e in ev if e["kind"] in dog_kinds)
    zones = {}
    for e in ev:
        if e["kind"] == "visit":
            z = zones.setdefault(e["detail"].get("zone", "?"), {"kind": e["detail"].get("zone_kind"), "count": 0, "last": None, "seconds": 0})
            z["count"] += 1
            z["seconds"] += e["detail"].get("seconds", 0)
            z["last"] = max(z["last"] or 0, e["ts"])
    timelapse = f"/media/timelapse/{day}.mp4" if os.path.exists(f"{STATE_DIR}/timelapse/{day}.mp4") else None
    highlights = f"/media/highlights/{day}.mp4" if os.path.exists(f"{STATE_DIR}/highlights/{day}.mp4") else None
    return {"day": day, "minutes_recorded": len(rows), "visible_min": visible, "active_min": active,
            "longest_rest_min": longest_rest, "first_seen": first, "last_seen": last,
            "sounds": sounds, "barks": barks, "buckets": buckets, "events": ev, "timelapse": timelapse,
            "highlights": highlights, "zones": zones}


def diary(limit=60):
    con = dbc()
    try:
        days = [r["d"] for r in con.execute(
            "SELECT DISTINCT date(ts,'unixepoch','localtime') d FROM minute ORDER BY d DESC LIMIT ?", (limit,))]
    finally:
        con.close()
    out = []
    for d in days:
        rep = day_report(d)
        cover = next((e["thumb"] for e in rep["events"] if e["kind"] == "active" and e["thumb"]), None) or \
            next((e["thumb"] for e in rep["events"] if e["thumb"]), None)
        out.append({k: rep[k] for k in ("day", "minutes_recorded", "visible_min", "active_min", "longest_rest_min",
                                        "sounds", "barks", "timelapse", "highlights")} | {"cover": cover,
                                                                            "events": len(rep["events"])})
    return out


# ---------- zones (bowl / water / bed), drawn over a live frame ----------

ZONES_FILE = "/var/lib/petcam/zones.json"
ZONE_DWELL = {"water": 6, "food": 6, "bed": 60, "other": 10}


def list_zones():
    try:
        with open(ZONES_FILE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return []


def save_zones(zones):
    tmp = ZONES_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(zones, f)
    os.replace(tmp, ZONES_FILE)


def add_zone(name, kind, rect):
    import secrets

    kind = kind if kind in ZONE_DWELL else "other"
    name = " ".join(str(name or kind.title()).split())[:24]
    r = [max(0.0, min(1.0, float(v))) for v in rect]
    x0, y0, x1, y1 = min(r[0], r[2]), min(r[1], r[3]), max(r[0], r[2]), max(r[1], r[3])
    if x1 - x0 < 0.03 or y1 - y0 < 0.03:
        raise ValueError("Draw a bigger box")
    if time.time() < _homing["until"]:
        raise Busy("Recalibrating; try again in a few seconds")
    _, px, py = motor_status()
    zones = [z for z in list_zones() if z["name"].lower() != name.lower()]
    if len(zones) >= 12:
        raise ValueError("12 zones at most")
    zones.append({"id": secrets.token_hex(4), "name": name, "kind": kind, "rect": [x0, y0, x1, y1], "pose": [px, py],
                  "dwell_s": ZONE_DWELL[kind], "cooldown_s": 300, "created": time.time()})
    save_zones(zones)
    return zones


def delete_zone(zone_id):
    zones = [z for z in list_zones() if z["id"] != zone_id]
    save_zones(zones)
    return zones


def live_frame():
    def call():
        req = urllib.request.Request(f"http://{CAM_HOST}/x/ch1.jpg", headers={"Cookie": _cam["cookie"] or ""})
        with urllib.request.urlopen(req, timeout=8) as r:
            return r.read(2_000_000)
    with _cam_lock:
        if not _cam["cookie"]:
            cam_login()
    try:
        return with_deadline(call, 10)
    except urllib.error.HTTPError as e:
        if e.code != 401:
            raise
        with _cam_lock:
            cam_login()
        return with_deadline(call, 10)


# ---------- wellbeing: her day against her own normal ----------

def day_metrics(day):
    """Activity numbers for one day from the per-minute table (plus zone visits)."""
    import datetime as dt

    d, start, end = day_bounds(day)
    prev_night = time.mktime((dt.date.fromisoformat(d) - dt.timedelta(days=1)).timetuple()) + 22 * 3600
    con = dbc()
    try:
        rows = con.execute("SELECT ts, n, present, moving FROM minute WHERE ts>=? AND ts<? ORDER BY ts", (start, end)).fetchall()
        night = con.execute("SELECT n, present, moving FROM minute WHERE ts>=? AND ts<?", (prev_night, start + 6 * 3600)).fetchall()
        visits = con.execute("SELECT detail FROM events WHERE kind='visit' AND ts>=? AND ts<?", (start, end)).fetchall()
    finally:
        con.close()
    visible = active = run = longest = settles = 0
    prev_active = False
    for r in rows:
        n = max(1, r["n"])
        here = r["present"] / n >= 0.5
        act = here and r["moving"] / n >= 0.15
        if here:
            visible += 1
            if act:
                active += 1
                run = 0
            else:
                run += 1
                longest = max(longest, run)
                if prev_active:
                    settles += 1
        else:
            run = 0
        prev_active = act
    night_active = sum(1 for r in night if r["present"] / max(1, r["n"]) >= 0.5 and r["moving"] / max(1, r["n"]) >= 0.15)
    zone_counts = {}
    for (det,) in visits:
        k = json.loads(det or "{}").get("zone_kind", "other")
        zone_counts[k] = zone_counts.get(k, 0) + 1
    return {"day": d, "minutes": len(rows), "visible": visible, "active": active, "longest_rest": longest,
            "settle_downs": settles, "night_active": night_active, "water": zone_counts.get("water", 0),
            "food": zone_counts.get("food", 0)}


def wellbeing(day=None):
    import datetime as dt
    import statistics

    today = day_metrics(day)
    base = []
    d0 = dt.date.fromisoformat(today["day"])
    for k in range(1, 15):
        m = day_metrics((d0 - dt.timedelta(days=k)).isoformat())
        if m["minutes"] >= 600 and m["visible"] >= 30:      # a day with real data about her
            base.append(m)
    out = {"today": today, "baseline_days": len(base)}
    if len(base) < 3:
        out["verdict"] = f"Learning {petconf.POSS} normal ({len(base)} of 3 days so far)"
        return out
    med = {k: statistics.median(b[k] for b in base) for k in ("active", "visible", "longest_rest", "night_active", "water", "settle_downs")}
    out["baseline"] = med
    # active share of in-view time, so a day she spent elsewhere isn't "calm"
    rate = today["active"] / max(1, today["visible"])
    base_rate = statistics.median(b["active"] / max(1, b["visible"]) for b in base)
    if today["visible"] < 30:
        out["verdict"] = "Not much time in view today"
    elif rate > base_rate * 1.5 and today["active"] - med["active"] >= 15:
        out["verdict"] = "More active than usual"
    elif rate < base_rate * 0.6 and med["active"] - today["active"] >= 15:
        out["verdict"] = "Calmer than usual"
    else:
        out["verdict"] = "A typical day"
    notes = []
    if today["night_active"] > max(10, med["night_active"] * 2):
        notes.append(f"restless night ({today['night_active']} active min vs usual {med['night_active']:.0f})")
    if med["water"] >= 2 and today["water"] == 0 and today["visible"] >= 120:
        notes.append("no water-bowl visits seen today")
    out["notes"] = notes
    return out


def snap_at(ts):
    """Nearest per-minute snapshot path for a timestamp (snapshots are kept 14 days)."""
    t = time.localtime(float(ts))
    for delta in (0, -60, 60, -120, 120):
        tt = time.localtime(float(ts) + delta)
        rel = f"{time.strftime('%Y-%m-%d', tt)}/{time.strftime('%H-%M', tt)}.jpg"
        if os.path.exists(f"{STATE_DIR}/snaps/{rel}"):
            return {"snap": f"/media/snaps/{rel}"}
    return {"snap": None, "asked": time.strftime("%H:%M", t)}


def recordings():
    def call():
        with urllib.request.urlopen("http://127.0.0.1:9996/list?path=cam", timeout=5) as r:
            return json.loads(r.read(1_000_000) or b"[]")

    try:
        return with_deadline(call, 8)
    except urllib.error.HTTPError as e:
        if e.code == 404:  # nothing recorded yet
            return []
        raise


# ---------- HTTP ----------

def read_file(name):
    with open(os.path.join(HERE, name), "rb") as f:
        return f.read()


PUBLIC_HOST = ENV.get("PUBLIC_HOST") or petconf.PUBLIC_HOST


def page(name, **subs):
    """A page or asset with its {{PLACEHOLDERS}} filled: the pet's name and pronouns, plus subs."""
    body = read_file(name)
    for k, v in (petconf.template_vars() | subs).items():
        body = body.replace(f"{{{{{k}}}}}".encode(), v.encode())
    return body


def html_escape(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


class Handler(BaseHTTPRequestHandler):
    server_version = "petcam"
    protocol_version = "HTTP/1.1"

    def client_ip(self):
        return self.headers.get("X-Forwarded-For", self.client_address[0]).split(",")[0].strip()

    def handle_one_request(self):
        # One handler object serves every request on a keep-alive connection, and Caddy pools
        # those connections across users. Per-request auth state MUST be reset here, or one
        # signed-in request authorises the next anonymous one on the same connection
        # (found in testing 2026-11-01: /api/live and /media answered 200 without a cookie).
        self._auth = None
        super().handle_one_request()

    def device(self):
        if self._auth is None:
            self._auth = {"device": device_from_cookie(self.headers.get("Cookie"), self.client_ip())}
        return self._auth["device"]

    def legacy(self):
        return session_ok(self.headers.get("Cookie"))

    def authed(self):
        return bool(self.device()) or self.legacy()

    def send(self, code, body=b"", ctype="text/plain; charset=utf-8", headers=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (headers or {}).items():
            if isinstance(v, list):
                for item in v:
                    self.send_header(k, item)
            else:
                self.send_header(k, v)
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def send_json(self, code, obj):
        self.send(code, json.dumps(obj).encode(), "application/json")

    def redirect(self, to, headers=None):
        h = {"Location": to}
        h.update(headers or {})
        self.send(303, b"", headers=h)

    def log_message(self, fmt, *args):  # no cookies, tokens or query strings in logs
        path = self.path.split("?")[0]
        if path.startswith("/join/"):
            path = "/join/<token>"
        print(f"{self.client_ip()} {self.command} {path} {args[1] if len(args) > 1 else ''}", flush=True)

    def query(self):
        return {k: v[0] for k, v in urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query).items()}

    def session_headers(self):
        """Refresh the device cookie on every page load so a used device never expires; move
        a pre-device (password-only) session onto its own device key."""
        dev = self.device()
        value = cookie_value(self.headers.get("Cookie"), DEVICE_COOKIE)
        if dev and value:
            return {"Set-Cookie": device_cookie(value)}
        if self.legacy():
            value = new_device(None, self.client_ip(), self.headers.get("User-Agent"))
            return {"Set-Cookie": [device_cookie(value), f"{COOKIE}=; Path=/; Max-Age=0; HttpOnly; Secure; SameSite=Strict"]}
        return {}

    # ---------- GET ----------
    def do_GET(self):
        path = self.path.split("?")[0]
        if path == "/healthz":
            return self.send(200, b"ok")
        if path == "/auth":  # Caddy forward_auth for video and media
            return self.send(200 if self.authed() else 401)
        if path == "/manifest.webmanifest":
            return self.send(200, page("manifest.webmanifest"), "application/manifest+json")
        if path == "/sw.js":
            return self.send(200, page("sw.js"), "text/javascript", {"Service-Worker-Allowed": "/"})
        if path in ("/icon.png", "/apple-touch-icon.png"):
            return self.send(200, read_file("icon.png"), "image/png", {"Cache-Control": "max-age=86400"})
        if path == "/login":
            if self.authed():
                return self.redirect("/")
            return self.send(200, page("login.html"), "text/html; charset=utf-8")
        if path.startswith("/join/"):
            name = invite_valid(path[len("/join/"):])
            if not name:
                return self.send(410, page("join.html", NAME="", STATE="expired"), "text/html; charset=utf-8")
            return self.send(200, page("join.html", NAME=html_escape(name), STATE="ok"), "text/html; charset=utf-8")
        if not self.authed():
            if path.startswith("/api/"):
                return self.send_json(401, {"error": "login required"})
            return self.redirect("/login")
        if path == "/":
            return self.send(200, page("index.html"), "text/html; charset=utf-8", self.session_headers())
        q = self.query()
        if path == "/api/live":
            return self.send_json(200, read_live() | {"settings": get_settings()})
        if path == "/api/day":
            return self.camera_call(day_report, q.get("d"))
        if path == "/api/diary":
            return self.camera_call(diary)
        if path == "/api/snap":
            return self.camera_call(snap_at, q.get("ts", time.time()))
        if path == "/api/devices":
            return self.send_json(200, list_devices((self.device() or {}).get("id")))
        if path == "/api/zones":
            return self.send_json(200, list_zones())
        if path == "/api/wellbeing":
            return self.camera_call(wellbeing, q.get("d"))
        if path == "/api/frame.jpg":
            try:
                return self.send(200, live_frame(), "image/jpeg")
            except Exception as e:  # noqa: BLE001
                return self.send_json(502, {"error": f"camera unreachable: {e}"})
        if path == "/api/push":
            return self.camera_call(lambda: push_status((self.device() or {}).get("id")) | push_key())
        if path == "/api/recordings":
            try:
                return self.send_json(200, recordings())
            except Exception as e:  # noqa: BLE001
                return self.send_json(502, {"error": str(e)})
        if path == "/api/state":
            return self.camera_call(state)
        return self.send(404, b"not found")

    def camera_call(self, fn, *args):
        try:
            return self.send_json(200, fn(*args))
        except Busy as e:
            return self.send_json(409, {"error": str(e)})
        except (ValueError, TypeError, KeyError) as e:
            return self.send_json(400, {"error": str(e)})
        except urllib.error.HTTPError as e:
            return self.send_json(502, {"error": f"camera answered {e.code}"})
        except Exception as e:  # noqa: BLE001
            return self.send_json(502, {"error": f"camera unreachable: {e}"})

    # ---------- POST ----------
    def do_POST(self):
        path = self.path.split("?")[0]
        limit = 2_000_000 if path == "/api/talk" else 10_000
        length = int(self.headers.get("Content-Length") or 0)
        if length > limit:
            self.rfile.read(min(length, limit + 1))
            return self.send_json(413, {"error": "too large"})
        raw = self.rfile.read(length)
        ip, ua = self.client_ip(), self.headers.get("User-Agent")
        if path == "/login":
            if login_locked(ip):
                return self.send(429, page("login.html").replace(b"<!--ERR-->", b"Too many attempts. Try again in 15 minutes."), "text/html; charset=utf-8")
            form = urllib.parse.parse_qs(raw.decode(errors="replace"))
            if check_password(form.get("password", [""])[0]):
                note_success(ip)
                value = new_device(form.get("device", [""])[0].strip() or None, ip, ua)
                return self.redirect("/", {"Set-Cookie": device_cookie(value)})
            note_failure(ip)
            time.sleep(1)
            return self.send(401, page("login.html").replace(b"<!--ERR-->", b"Wrong password."), "text/html; charset=utf-8")
        if path.startswith("/join/"):
            value, _ = use_invite(path[len("/join/"):], ip, ua)
            if not value:
                return self.send(410, page("join.html", NAME="", STATE="expired"), "text/html; charset=utf-8")
            return self.redirect("/", {"Set-Cookie": device_cookie(value)})
        if path == "/logout":
            dev = self.device()
            if dev:
                revoke_device(dev["id"])
            return self.redirect("/login", {"Set-Cookie": [f"{DEVICE_COOKIE}=; Path=/; Max-Age=0; HttpOnly; Secure; SameSite=Lax",
                                                           f"{COOKIE}=; Path=/; Max-Age=0; HttpOnly; Secure; SameSite=Strict"]})
        if not self.authed():
            return self.send_json(401, {"error": "login required"})
        if path == "/api/talk":
            ctype = self.headers.get("Content-Type", "").split(";")[0]
            if not ctype.startswith("audio/"):
                return self.send_json(415, {"error": "audio only"})
            dev = self.device() or {}
            return self.camera_call(lambda: talk(dev.get("id", "legacy"), dev.get("name", "Someone"), raw, ctype))
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            return self.send_json(415, {"error": "json only"})
        try:
            body = json.loads(raw or b"{}")
            assert isinstance(body, dict)
        except (ValueError, AssertionError):
            return self.send_json(400, {"error": "bad json"})
        me = (self.device() or {}).get("id")
        routes = {
            "/api/aim": lambda: aim(body.get("fx", 0), body.get("fy", 0)),
            "/api/nudge": lambda: nudge(body.get("vx", 0), body.get("vy", 0)),
            "/api/goto": lambda: goto(*start_pose()) if body.get("to") == "start" else goto(float(body["x"]), float(body["y"])),
            "/api/recalibrate": lambda: (recalibrate(), {"ok": True})[1],
            "/api/presets": lambda: add_preset(body.get("name", "")),
            "/api/presets/delete": lambda: delete_preset(str(body.get("name", ""))),
            "/api/imaging": lambda: set_imaging(str(body.get("action", ""))),
            "/api/privacy": lambda: set_privacy(body.get("on")),
            "/api/settings": lambda: set_settings(body),
            "/api/events/label": lambda: label_event(body.get("id"), body.get("label"), me),
            "/api/zones": lambda: add_zone(body.get("name"), body.get("kind"), body.get("rect", [])),
            "/api/zones/delete": lambda: delete_zone(str(body.get("id", ""))),
            "/api/invites": lambda: {"url": f"https://{PUBLIC_HOST}/join/{create_invite(body.get('name'), me)}",
                                     "expires_in_h": 24},
            "/api/devices/revoke": lambda: revoke_device(str(body.get("id", ""))),
            "/api/devices/rename": lambda: rename_device(str(body.get("id", "")), body.get("name", "")),
            "/api/push/subscribe": lambda: push_subscribe(me, body.get("sub"), body.get("prefs")),
            "/api/push/unsubscribe": lambda: push_unsubscribe(me),
            "/api/push/test": lambda: push_test(me),
        }
        if path in routes:
            return self.camera_call(routes[path])
        return self.send(404, b"not found")


if __name__ == "__main__":
    if not SECRET or not ENV.get("APP_PASSWORD_HASH"):
        print("WARNING: APP_SECRET or APP_PASSWORD_HASH missing; nobody can log in yet", flush=True)
    init_auth_db()
    threading.Thread(target=serve_talk_files, daemon=True).start()
    ThreadingHTTPServer(("127.0.0.1", 8080), Handler).serve_forever()
