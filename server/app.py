#!/usr/bin/env python3
"""petcam web app: login and devices, camera control, the pet's day, live status stream, talk, reports.

Standard library only. Listens on 127.0.0.1:8080 behind Caddy, which also asks
GET /auth before serving any video. Secrets come from /etc/petcam/app.env:
  APP_PASSWORD_HASH  pbkdf2_sha256$<iters>$<salt hex>$<hash hex>
  APP_SECRET         random hex, signs the session cookie
  CAM_WEB_USER / CAM_WEB_PASS   the camera's web login (for motor calls)
Non-secret settings (pet name, addresses) come from /etc/petcam/site.env via petconf.py.
"""
import base64
import contextlib
import fcntl
import hashlib
import hmac
import http.client
import json
import os
import socket
import sys
import threading
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import petconf
import roomapi
import requestboard

STATE_DIR = petconf.STATE

HERE = os.path.dirname(os.path.abspath(__file__))
SESSION_DAYS = 30
COOKIE = "petcam_session"


ENV = petconf.load_env("/etc/petcam/app.env", unquote=False)   # secrets: values taken verbatim
SECRET = ENV.get("APP_SECRET", "").encode()
CAM_HOST = ENV.get("CAM_HOST") or petconf.CAM_HOST


class AppServer(ThreadingHTTPServer):
    """A phone or Caddy hanging up mid-request is normal, not a crash: don't print a traceback
    for it (qa.py counts tracebacks as crashes). Any other exception is still reported in full."""
    def handle_error(self, request, client_address):
        if isinstance(sys.exc_info()[1], (ConnectionResetError, BrokenPipeError, ConnectionAbortedError)):
            return
        super().handle_error(request, client_address)


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
        return FAILS.get(ip, [0, 0])[1] > time.time()


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

DB_PATH = f"{STATE_DIR}/petcam.db"
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
    CREATE TABLE IF NOT EXISTS events(id INTEGER PRIMARY KEY, ts REAL, kind TEXT, detail TEXT, clip TEXT, thumb TEXT,
                                      label TEXT, labelled_by TEXT);
    CREATE INDEX IF NOT EXISTS events_ts ON events(ts);
    CREATE TABLE IF NOT EXISTS usage(ts REAL, device TEXT, kind TEXT, name TEXT, ms REAL, ok INT, detail TEXT);
    CREATE INDEX IF NOT EXISTS usage_ts ON usage(ts);
    """)
    # Human feedback on moments ("was this the pet?") = ground truth for tuning the detectors.
    cols = [r[1] for r in con.execute("PRAGMA table_info(events)")]
    if cols and "label" not in cols:
        con.execute("ALTER TABLE events ADD COLUMN label TEXT")
        con.execute("ALTER TABLE events ADD COLUMN labelled_by TEXT")
    con.close()


# ---------- usage telemetry: how the app is really used, so agents can improve it (tools/usage.py) ----------
# Local only (the 512 MB box can't host an OpenTelemetry + Grafana stack, and this data stays at home).
# kind "api": every /api call, timed, by device. kind "ui": events the server can't see, sent by the page.
USAGE_SKIP = {"/api/live", "/api/live/stream", "/api/monitoring", "/api/reports/frame", "/api/usage", "/api/clientlog"}
USAGE_DAYS = 90
_usage_q = None


def usage(device, kind, name, ms=None, ok=True, detail=None, ts=None):
    global _usage_q
    import queue
    if _usage_q is None:
        _usage_q = queue.Queue(maxsize=5000)
        threading.Thread(target=_usage_writer, daemon=True).start()
    try:
        _usage_q.put_nowait((ts or time.time(), device or "", kind, str(name)[:60], None if ms is None else round(float(ms), 1),
                             int(bool(ok)), json.dumps(detail, separators=(",", ":"))[:600] if detail else None))
    except queue.Full:
        pass


def _usage_writer():
    """Batched inserts every 2 s: a slow disk never delays a request."""
    import queue
    while True:
        rows = [_usage_q.get()]
        time.sleep(2)
        while True:
            try:
                rows.append(_usage_q.get_nowait())
            except queue.Empty:
                break
        try:
            con = dbc()
            con.executemany("INSERT INTO usage VALUES(?,?,?,?,?,?,?)", rows)
            con.execute("DELETE FROM usage WHERE ts < ?", (time.time() - USAGE_DAYS * 86400,))
            con.close()
        except Exception as e:  # noqa: BLE001 - telemetry must never break the app
            print("usage write failed:", e, flush=True)


UI_EVENT_OK = set("abcdefghijklmnopqrstuvwxyz_.0123456789")


def record_ui_events(device, body):
    """POST /api/usage {events: [{t, name, ms?, detail?}]} from the page. Capped and validated."""
    now = time.time()
    evs = body.get("events") if isinstance(body.get("events"), list) else []
    n = 0
    for e in evs[:100]:
        if not isinstance(e, dict):
            continue
        name = str(e.get("name", ""))[:40]
        if not name or not set(name) <= UI_EVENT_OK:
            continue
        t = e.get("t") if isinstance(e.get("t"), (int, float)) and abs(e["t"] - now) < 3600 else now
        ms = e.get("ms") if isinstance(e.get("ms"), (int, float)) else None
        detail = e.get("detail") if isinstance(e.get("detail"), dict) else None
        usage(device, "ui", name, ms, True, detail, t)
        n += 1
    return {"ok": True, "n": n}


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

    if not device_id:
        raise ValueError("Sign in on this device first (Settings) to test its notifications.")

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
    if _params["v"] is None or time.time() - _params["t"] > 3600:   # fixed motor config: hourly is plenty
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
    if not petconf.PTZ:
        return 1, 1
    p = motor_params() or {}
    return int(num(p, "steps_pan", "x_max", default=3700)), int(num(p, "steps_tilt", "y_max", default=1000))


def start_pose():
    if not petconf.PTZ:
        return 0, 0
    p = motor_params() or {}
    xm, ym = limits()
    return int(num(p, "pos_0_x", default=xm / 2)), int(num(p, "pos_0_y", default=ym / 2))


MOVED_FILE = f"{STATE_DIR}/moved_at"


def note_camera_moved():
    """Tell petcam-watch the picture is about to shift because WE moved it, not the pet."""
    try:
        with open(MOVED_FILE, "a"):
            pass
        os.utime(MOVED_FILE)
    except OSError:
        pass


# ---------- fast camera channel: one long-lived SSH shell instead of a CGI per call ----------
# Each Thingino CGI forks a shell, runs auth.sh and several helpers on a saturated single-core CPU:
# ~300 ms per motor call. `motors` itself takes ~80 ms. One persistent dropbear session runs it
# directly. Any failure falls back to the CGI, so this is purely an optimisation (profiled 2026-11-02).
CAM_SSH_KEY = f"{STATE_DIR}/ssh/id_ed25519"   # made by install.sh; authorised on the camera by ensure_cam_ssh()


class CamShell:
    MARK = "__petcam_done__"

    def __init__(self):
        self.p = None
        self.lock = threading.Lock()
        self.down_until = 0.0

    def _start(self):
        import subprocess
        self.p = subprocess.Popen(
            ["ssh", "-T", "-i", CAM_SSH_KEY, "-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
             "-o", f"UserKnownHostsFile={os.path.dirname(CAM_SSH_KEY)}/known_hosts", "-o", "ConnectTimeout=5",
             "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2", f"root@{CAM_HOST}", "sh"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1)

    def run(self, cmd, timeout=5.0):
        """Run one shell command on the camera; returns stdout. Raises on any failure."""
        if time.time() < self.down_until or not os.path.exists(CAM_SSH_KEY):
            raise OSError("camera ssh unavailable")
        with self.lock:
            try:
                if self.p is None or self.p.poll() is not None:
                    self._start()
                self.p.stdin.write(f"{cmd} 2>/dev/null; echo {self.MARK}$?\n")
                self.p.stdin.flush()
                out, box = [], {}

                def read():
                    for line in self.p.stdout:
                        if line.startswith(self.MARK):
                            box["rc"] = int(line[len(self.MARK):].strip() or 1)
                            return
                        out.append(line)
                t = threading.Thread(target=read, daemon=True)
                t.start()
                t.join(timeout)
                if "rc" not in box:
                    raise TimeoutError(f"camera ssh: no answer to {cmd.split()[0]}")
                if box["rc"] != 0:
                    raise OSError(f"camera ssh: {cmd.split()[0]} exited {box['rc']}")
                return "".join(out)
            except Exception:
                if self.p is not None:
                    self.p.kill()
                self.p = None
                self.down_until = time.time() + 30          # use the CGI for a while
                raise


_camsh = CamShell()


QUIET_SPEED = 250   # steps/s for automatic moves (follow, search). Measured with the camera mic 2026-11-02:
                    # 900 (default) +14.9 dB over the room, 600 +16.7, 400 +16.1, 250 +10.4. People's joystick stays at 900.


def motor_cmd(d, x=0, y=0, speed=None):
    """A `motors` move: over the fast channel, or the CGI if that is down (the CGI can't set a speed)."""
    if not petconf.PTZ:
        if d == "s":
            return          # a stop with nothing to stop is not an error
        require_ptz()
    if d != "s":
        require_camera_active()
    try:
        sp = f" -s {int(speed)}" if speed else ""
        _camsh.run(f"motors -d {d}{sp} -x {int(x)} -y {int(y)} >/dev/null; true")
    except Exception:  # noqa: BLE001
        cam_get("/x/json-motor.cgi", {"d": d, "x": int(x), "y": int(y)})


def motor_status():
    if not petconf.PTZ:
        return "0", 0, 0    # one fixed view, settled: zones, poses and the watcher's tables keep working
    try:
        m = json.loads(_camsh.run("motors -j"))
    except Exception:  # noqa: BLE001
        m = cam_get("/x/json-motor.cgi", {"d": "j"}).get("message") or {}
    return str(m.get("status")), int(float(m["xpos"])), int(float(m["ypos"]))


def position():
    _, x, y = motor_status()
    return x, y


_move_lock = threading.Lock()   # one move at a time, each computed from a settled position
_homing = {"until": 0.0}


class Busy(Exception):
    pass


class NoMotors(Busy):
    """This install's camera has no pan/tilt motors (PTZ=0). A Busy, so every route answers 409 with the message."""


def require_ptz():
    if not petconf.PTZ:
        raise NoMotors("This camera has no motors")


def wait_idle(timeout=20.0):   # slow (quiet) automatic moves take up to ~15 s
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


def goto(x, y, speed=None):
    """Move to reported position (x, y), clamped inside the motor range so no axis can run past
    its end stop. Sent as a relative step from the settled position (see the note above)."""
    require_ptz()
    require_camera_active()
    if _joystick is not None:
        _joystick.forget()
    with _move_lock:
        return _goto_locked(x, y, wait_idle(), speed)


def _goto_locked(x, y, current, speed=None):
    xm, ym = limits()
    cx, cy = current
    if not (0 <= cy <= ym) or not (0 <= cx <= xm):
        # An axis already past its end stop refuses every move on that axis; only homing resets it.
        recalibrate()
        raise Busy("The camera was past its end stop, so it is recalibrating (about 15 s)")
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
            motor_cmd("g", dx, dy, speed)
        if step_y == ty:
            break
        time.sleep(0.15)
        cx, cy = wait_idle()
    return {"x": tx, "y": ty}


def aim(fx, fy, speed=None):
    """Bring the point at (fx, fy) of the frame to the centre. fx/fy are offsets from the
    centre as a fraction of the frame: +fx is right of centre, +fy is below centre."""
    require_ptz()
    require_camera_active()
    fx, fy = max(-0.75, min(0.75, float(fx))), max(-0.75, min(0.75, float(fy)))
    if _joystick is not None:
        _joystick.forget()
    with _move_lock:
        cx, cy = wait_idle()
        # Right of centre -> turn right -> x down. Below centre -> tilt down -> y down.
        return _goto_locked(cx - fx * PAN_STEPS_PER_FRAME, cy - fy * TILT_STEPS_PER_FRAME, (cx, cy), speed)


def recalibrate():
    """Homing sweeps both axes to their end stops and back to the start pose. The CGI answers
    only when it is done; moves are refused until then so they can't overlap the sweep."""
    require_ptz()
    require_camera_active()
    if time.time() < _homing["until"]:
        return
    if not _cam["cookie"]:
        with _cam_lock:
            cam_login()                 # before blocking moves: a failed login must not block them for 60 s
    _homing["until"] = time.time() + 60
    if _joystick is not None:
        _joystick.forget()
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

PRESETS_FILE = f"{STATE_DIR}/presets.json"
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
    require_ptz()
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


_view_insights_lock = threading.Lock()
_view_insights_cache = {}
_view_insights_errors = {}
_view_insights_fingerprint = None
_view_insights_worker = None
_view_insights_active_key = None
_VIEW_INSIGHTS_TTL = 300
_VIEW_INSIGHTS_RETRY = 60
_VIEW_INSIGHTS_ERROR = "View statistics could not be refreshed. Please try again shortly."


def _compute_view_insights(days, presets):
    """Full-window aggregation, called only by the single background worker."""
    import viewstats

    end = time.time()
    start = end - days * 86400
    con = dbc()
    try:
        # Stream the complete requested window, with bounded pose lookahead.
        obs = con.execute("SELECT ts, score, box FROM obs WHERE ts>=? AND ts<=? ORDER BY ts", (start, end))
        track = con.execute("SELECT ts, pan, tilt, act, moves FROM track WHERE ts>=? AND ts<=? ORDER BY ts",
                            (start - 33, end + 33))
        result = viewstats.summarize_sorted(obs, track, presets, PAN_STEPS_PER_FRAME, TILT_STEPS_PER_FRAME)
    finally:
        con.close()
    result.update(days=days, since=start, until=end)
    return result


def _refresh_view_insights(key, days, presets):
    global _view_insights_worker, _view_insights_active_key
    try:
        result = _compute_view_insights(days, presets)
    except Exception:
        # Neither SQL paths nor exception details belong in this public status.
        with _view_insights_lock:
            if key[1] == _view_insights_fingerprint:
                _view_insights_errors[key] = time.monotonic()
    else:
        with _view_insights_lock:
            if key[1] == _view_insights_fingerprint:
                _view_insights_cache[key] = (time.monotonic(), result)
                _view_insights_errors.pop(key, None)
    finally:
        with _view_insights_lock:
            _view_insights_worker = None
            _view_insights_active_key = None


def view_insights(days=7):
    """Return cached data or prompt progress; never wait for a full history scan."""
    global _view_insights_fingerprint, _view_insights_worker, _view_insights_active_key
    days = int(days)
    if not 1 <= days <= 7:
        raise ValueError("days must be between 1 and 7")
    presets = load_presets()
    fingerprint = json.dumps(presets, sort_keys=True)
    key = (days, fingerprint)
    with _view_insights_lock:
        if fingerprint != _view_insights_fingerprint:
            _view_insights_fingerprint = fingerprint
            _view_insights_cache.clear()
            _view_insights_errors.clear()
        now = time.monotonic()
        cached = _view_insights_cache.get(key)
        if cached and now - cached[0] < _VIEW_INSIGHTS_TTL:
            return dict(cached[1], status="ready", refreshing=False)
        failed = _view_insights_errors.get(key)
        if failed is not None and now - failed < _VIEW_INSIGHTS_RETRY:
            if cached:
                return dict(cached[1], status="ready", refreshing=False,
                            refresh_error=_VIEW_INSIGHTS_ERROR, retry_after_s=_VIEW_INSIGHTS_RETRY)
            return {"status": "error", "error": _VIEW_INSIGHTS_ERROR,
                    "retry_after_s": _VIEW_INSIGHTS_RETRY, "days": days}
        if _view_insights_worker is None:
            _view_insights_active_key = key
            _view_insights_worker = threading.Thread(target=_refresh_view_insights,
                                                     args=(key, days, presets), daemon=True,
                                                     name="view-insights")
            try:
                _view_insights_worker.start()
            except RuntimeError:
                _view_insights_worker = None
                _view_insights_active_key = None
                _view_insights_errors[key] = now
                if cached:
                    return dict(cached[1], status="ready", refreshing=False,
                                refresh_error=_VIEW_INSIGHTS_ERROR, retry_after_s=_VIEW_INSIGHTS_RETRY)
                return {"status": "error", "error": _VIEW_INSIGHTS_ERROR,
                        "retry_after_s": _VIEW_INSIGHTS_RETRY, "days": days}
        if cached:
            # Another requested window may own the worker. Keep polling stale
            # data so this window can refresh once that single worker is free.
            return dict(cached[1], status="ready", refreshing=True, retry_after_s=5)
        return {"status": "pending", "retry_after_s": 5, "days": days}


def state():
    xm, ym = limits()
    x, y = position()
    return {"x": x, "y": y, "x_max": xm, "y_max": ym, "start": start_pose(), "ptz": petconf.PTZ,
            "frame": [PAN_STEPS_PER_FRAME, TILT_STEPS_PER_FRAME], "presets": load_presets() if petconf.PTZ else []}


class Joystick:
    """Joystick moves, made to feel instant on a slow camera.

    The camera's motor CGI takes ~200-400 ms per call. So /api/nudge answers right away and a
    single background sender talks to the camera. Ticks that arrive while it is busy are
    ADDED to one pending move (never queued), so the camera can't fall behind the thumb, and
    a released stick stops within one camera call. The position is tracked locally from our
    own targets and re-read from the camera only when the estimate is stale.
    """
    STALE = 120.0  # safety net: re-read the position at least this often
    # The position is otherwise trusted until something else moves the camera (the watcher's
    # follow mode, aim/goto, homing): those touch MOVED_FILE or call forget(). Skipping the
    # re-read saves one ~300 ms camera CGI on the first touch after a pause (profiled 2026-11-02).

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
        try:
            moved_elsewhere = os.path.getmtime(MOVED_FILE) > self.pos_t
        except OSError:
            moved_elsewhere = False
        if self.pos is None or moved_elsewhere or time.time() - self.pos_t > self.STALE:
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
            motor_cmd("g", dx, dy)
        self.pos, self.pos_t = (tx, ty), time.time()

    def forget(self):
        """Absolute moves (aim/goto/homing) invalidate the local estimate."""
        self.pos = None


_joystick = None


def nudge(vx, vy):
    """Joystick tick (~6/s while held). vx/vy in [-1, 1]; +vx = view right, +vy = view up."""
    global _joystick
    require_ptz()
    require_camera_active()
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
    if not petconf.THINGINO:
        raise Busy("This camera has no day/night controls")
    require_camera_active()
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


_privacy_lock = threading.Lock()
PRIVACY_RESUME_WAIT = 30
PRIVACY_CONFIRM_FRESH = 60


class PrivacyError(RuntimeError):
    """Requested privacy stays latched when the camera cannot confirm a transition."""


def _privacy_intent():
    try:
        with open(f"{STATE_DIR}/privacy") as f:
            requested = f.read().strip() != "0"
            # Read the value and version from the same inode across an atomic replacement.
            return requested, os.fstat(f.fileno()).st_mtime_ns / 1_000_000_000
    except FileNotFoundError:
        return False, None
    except OSError:
        return True, None


def privacy_requested():
    return _privacy_intent()[0]


def _privacy_intent_changed_at():
    return _privacy_intent()[1]


def require_camera_active():
    if privacy_requested():
        raise Busy("The camera is paused. Resume it first.")


def _privacy_owner(fd):
    # CLI maintenance may run as root; keep new files readable by the app and watcher.
    if os.geteuid() == 0:
        owner = os.stat(STATE_DIR)
        os.fchown(fd, owner.st_uid, owner.st_gid)


def _privacy_atomic(name, text):
    path = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=STATE_DIR, prefix=".privacy-", delete=False) as f:
            path = f.name
            os.fchmod(f.fileno(), 0o640)
            _privacy_owner(f.fileno())
            f.write(text)
            f.flush()
            os.fsync(f.fileno())
        os.replace(path, f"{STATE_DIR}/{name}")
        path = None
        directory = os.open(STATE_DIR, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if path is not None:
            os.unlink(path)


@contextlib.contextmanager
def _privacy_operation():
    # The HTTP app and watcher are separate processes. Both must serialize with resume.
    with _privacy_lock:
        fd = os.open(f"{STATE_DIR}/privacy-operation.lock", os.O_RDWR | os.O_CREAT, 0o640)
        try:
            _privacy_owner(fd)
            fcntl.flock(fd, fcntl.LOCK_EX)
            yield
        finally:
            fcntl.flock(fd, fcntl.LOCK_UN)
            os.close(fd)


def privacy_status():
    requested, intent_changed_at = _privacy_intent()
    try:
        with open(f"{STATE_DIR}/privacy-state.json") as f:
            state = json.load(f)
        if not isinstance(state, dict):
            state = {}
    except (OSError, ValueError):
        state = {}
    phase = "checking" if requested else "active"
    result = {"requested": requested, "confirmed": False, "mode": "unknown", "phase": phase,
              "target": requested, "boot_persistent": False, "last_verified_at": None,
              "intent_changed_at": intent_changed_at,
              "message": "Privacy requested; checking the camera." if requested else "Camera privacy is off."}
    if state.get("requested") is requested:
        result.update({k: state[k] for k in result if k in state and k not in ("requested", "intent_changed_at")})
        result["confirmed"] = state.get("confirmed") is True
    verified = result["last_verified_at"]
    fresh = isinstance(verified, (float, int)) and 0 <= time.time() - verified <= PRIVACY_CONFIRM_FRESH
    if requested and result["confirmed"] and not fresh:
        result.update(confirmed=False, phase="checking", message="Privacy requested; checking the camera.")
    return result


def _privacy_save(target, confirmed, mode, phase, message, boot_persistent=False):
    state = {"requested": privacy_requested(), "target": target, "confirmed": confirmed,
             "mode": mode, "phase": phase, "message": message, "boot_persistent": boot_persistent,
             "last_verified_at": time.time() if confirmed else None}
    _privacy_atomic("privacy-state.json", json.dumps(state, separators=(",", ":")) + "\n")
    return state | {"intent_changed_at": _privacy_intent_changed_at()}


def _privacy_guard(action, retry_cached=False):
    assert action in ("status", "pause", "resume")
    # An existing broken package is not absence and must never silently downgrade.
    command = ("if test -x /etc/opc-privacy/control; then /etc/opc-privacy/control " + action +
               "; elif test -e /etc/opc-privacy || test -L /etc/opc-privacy; then "
               "printf '%s\\n' '{\"installed\":true,\"verified\":false}'; "
               "else printf '%s\\n' '{\"installed\":false}'; fi")
    cached_unavailable = time.time() < _camsh.down_until
    try:
        output = _camsh.run(command, timeout=12 if action == "status" else 35)
    except OSError as error:
        # A camera reboot can finish before this channel's old 30-second cooldown.
        # Only an explicit transition may replace that cached failure, and only
        # for a read-only status probe after the camera's web port is reachable.
        if not (retry_cached and action == "status" and cached_unavailable
                and str(error) == "camera ssh unavailable" and os.path.isfile(CAM_SSH_KEY)):
            raise
        with socket.create_connection((CAM_HOST, 80), timeout=2):
            pass
        with _camsh.lock:
            if time.time() < _camsh.down_until:
                if _camsh.p is not None:
                    try:
                        _camsh.p.kill()
                    except OSError:
                        pass
                    _camsh.p = None
                _camsh.down_until = 0.0
        output = _camsh.run(command, timeout=12)  # One fresh status attempt; never retry a mutation.
    response = json.loads(output)
    if not isinstance(response, dict):
        raise PrivacyError("The camera privacy guard did not return a valid status.")
    if response.get("installed") is False:
        if os.path.exists(f"{STATE_DIR}/privacy-guard-required"):
            raise PrivacyError("The installed camera privacy guard is unavailable.")
        return None
    if not os.path.exists(f"{STATE_DIR}/privacy-guard-required"):
        _privacy_atomic("privacy-guard-required", "1\n")
    return response


def _privacy_verify_guard(response, paused):
    mode = "capture_stopped" if paused else "capture_running"
    if not isinstance(response, dict) or response.get("verified") is not True \
            or response.get("paused") is not paused or response.get("privacy") is not paused \
            or response.get("running") is not (not paused) or response.get("mode") != mode \
            or response.get("boot_persistent") is not True:
        raise PrivacyError("The camera has not confirmed the requested capture state.")


def _privacy_verify_config(response, paused):
    if not isinstance(response, dict):
        raise PrivacyError("The camera did not return its picture and microphone settings.")
    picture = response.get("privacy")
    audio = response.get("audio")
    if not isinstance(picture, dict) or not isinstance(audio, dict) \
            or type(picture.get("enabled")) not in (bool, int) \
            or type(audio.get("mic_enabled")) not in (bool, int) \
            or picture["enabled"] not in (0, 1) or audio["mic_enabled"] not in (0, 1) \
            or bool(picture["enabled"]) is not paused or bool(audio["mic_enabled"]) is paused:
        raise PrivacyError("The camera has not confirmed its picture and microphone settings.")


def _privacy_wait_active():
    deadline = time.monotonic() + PRIVACY_RESUME_WAIT
    while True:
        try:
            response = prudynt({"privacy": {"enabled": None}, "audio": {"mic_enabled": None}})
            _privacy_verify_config(response, False)
            return
        except Exception:  # noqa: BLE001 - startup often precedes API readiness
            if time.monotonic() >= deadline:
                raise PrivacyError("The camera has not confirmed that picture and microphone are enabled.") from None
            time.sleep(min(1, max(0, deadline - time.monotonic())))


MTX_API = "http://127.0.0.1:9997"
MTX_PATHS = ("cam", "sub")      # primary first. MediaMTX refuses a config in which the secondary ("sub") has no rpiCamera
                                # primary, so pause goes sub then cam, and resume cam then sub.


def _mtx(method, path, body=None):
    """MediaMTX control API (localhost only; enabled for CAMERA=rpi through camera.env)."""
    data = json.dumps(body, separators=(",", ":")).encode() if body is not None else None
    req = urllib.request.Request(MTX_API + path, data=data, method=method, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read(65536) or b"{}")


def _mtx_capture_running():
    """True while the camera is open. A path switched to 'publisher' has released it and has no source."""
    return any(_mtx("GET", f"/v3/paths/get/{n}").get("source") for n in MTX_PATHS)


def _privacy_transition_rpi(on, refresh=False):
    """Privacy for a camera on this machine: pause releases the camera in MediaMTX, so there is no picture,
    no recording and no live view. A MediaMTX restart would reopen it, so the watcher re-confirms every cycle
    (ensure_privacy_paused, refresh=True)."""
    try:
        if on:
            if not (refresh and not _mtx_capture_running()):
                for n in reversed(MTX_PATHS):
                    _mtx("PATCH", f"/v3/config/paths/patch/{n}", {"source": "publisher"})
                deadline = time.monotonic() + 10
                while _mtx_capture_running():
                    if time.monotonic() >= deadline:
                        raise PrivacyError("The camera has not confirmed that capture stopped.")
                    time.sleep(0.2)
            state = _privacy_save(True, True, "capture_stopped", "paused",
                                  "Camera capture is stopped.", False)
        else:
            for n in MTX_PATHS:
                _mtx("PATCH", f"/v3/config/paths/patch/{n}", {"source": "rpiCamera"})
            deadline = time.monotonic() + PRIVACY_RESUME_WAIT
            while not all(_mtx("GET", f"/v3/paths/get/{n}").get("ready") for n in MTX_PATHS):
                if time.monotonic() >= deadline:
                    raise PrivacyError("The camera has not confirmed that capture is running.")
                time.sleep(0.5)
            _privacy_atomic("privacy", "0\n")
            state = _privacy_save(False, True, "capture_running", "active",
                                  "Camera capture is enabled. Reconnecting to the stream.", False)
        return {"privacy": on, "mic": False, "privacy_status": state}
    except Exception:  # noqa: BLE001 - every failure retains the local acquisition gate
        if not privacy_requested():
            _privacy_atomic("privacy", "1\n")
        restored = False
        if not on:
            try:
                for n in reversed(MTX_PATHS):
                    _mtx("PATCH", f"/v3/config/paths/patch/{n}", {"source": "publisher"})
                restored = not _mtx_capture_running()
            except Exception:  # noqa: BLE001
                pass
        message = "The camera did not resume. Privacy is confirmed again." if restored else \
                  "Privacy is requested, but the camera has not confirmed that capture is paused."
        _privacy_save(on, restored, "capture_stopped" if restored else "unknown", "error", message, False)
        raise PrivacyError(message) from None


def _privacy_transition(on, refresh=False):
    if not petconf.THINGINO:
        if not refresh:
            _privacy_atomic("privacy", "1\n")
            _privacy_save(on, False, "unknown", "pausing" if on else "resuming",
                          "Stopping camera capture." if on else "Starting camera capture; privacy remains requested.")
        return _privacy_transition_rpi(on, refresh)
    if not refresh:
        # Suspend server acquisition before any camera I/O. Resume keeps the latch until verified.
        _privacy_atomic("privacy", "1\n")
        _privacy_save(on, False, "unknown", "pausing" if on else "resuming",
                      "Stopping camera capture." if on else "Starting camera capture; privacy remains requested.")
    guard = None
    attempted_resume = False
    try:
        guard = _privacy_guard("status", retry_cached=not refresh)
        if guard is not None:
            if on:
                response = guard
                if refresh:
                    try:
                        _privacy_verify_guard(response, True)
                    except PrivacyError:
                        response = _privacy_guard("pause")
                else:
                    response = _privacy_guard("pause")
                _privacy_verify_guard(response, True)
                state = _privacy_save(True, True, "capture_stopped", "paused",
                                      "Camera capture is stopped. The camera remains powered.", True)
            else:
                attempted_resume = True
                response = guard
                try:
                    _privacy_verify_guard(response, False)
                except PrivacyError:
                    response = _privacy_guard("resume")
                _privacy_verify_guard(response, False)
                _privacy_wait_active()
                _privacy_atomic("privacy", "0\n")
                state = _privacy_save(False, True, "capture_running", "active",
                                      "Camera capture is enabled. Reconnecting to the stream.", True)
        else:
            attempted_resume = not on
            response = prudynt({"privacy": {"enabled": on}, "audio": {"mic_enabled": not on}})
            _privacy_verify_config(response, on)
            if not on:
                try:
                    cam_get("/x/restart-prudynt.cgi")
                except ValueError:
                    pass  # This CGI can return non-JSON after the restart; readback is still mandatory.
                _privacy_wait_active()
                _privacy_atomic("privacy", "0\n")
            state = _privacy_save(on, True, "blackout", "paused" if on else "active",
                                  "Picture is blacked out and microphone is disabled. Capture remains running." if on
                                  else "Camera capture is enabled. Reconnecting to the stream.")
        return {"privacy": on, "mic": not on, "privacy_status": state}
    except Exception:  # noqa: BLE001 - every failure retains the local acquisition gate
        if not privacy_requested():
            _privacy_atomic("privacy", "1\n")
        restored = False
        if attempted_resume:
            try:
                if guard is not None:
                    _privacy_verify_guard(_privacy_guard("pause"), True)
                else:
                    _privacy_verify_config(prudynt({"privacy": {"enabled": True}, "audio": {"mic_enabled": False}}), True)
                restored = True
            except Exception:
                pass
        mode = "capture_stopped" if restored and guard is not None else "blackout" if restored else "unknown"
        message = "The camera did not resume. Privacy is confirmed again." if restored else \
                  "Privacy is requested, but the camera has not confirmed that capture is paused."
        _privacy_save(on, restored, mode, "error", message, restored and guard is not None)
        raise PrivacyError(message) from None


def set_privacy(on):
    """Persist intent first; confirm camera shutdown or conservative legacy suppression explicitly."""
    if type(on) is not bool:
        raise ValueError("on must be true or false")
    with _privacy_operation():
        try:
            return _privacy_transition(on)
        finally:
            if on:      # after the intent is on record, so a snapshot still in flight is refused rather than kept
                _report_held.clear()


def ensure_privacy_paused():
    """Idempotent watcher reconciliation; a worker queued before resume cannot re-pause afterward."""
    with _privacy_operation():
        if privacy_requested():
            return _privacy_transition(True, refresh=True)["privacy_status"]
        return privacy_status()


def request_find():
    """'Find <pet>': ask the watcher to run its room search now (owner, 2026-11-02). The watcher answers in
    live.json ('find': {t, result, found}) and as a 'search' event."""
    require_ptz()
    require_camera_active()
    t = time.time()
    with open(f"{STATE_DIR}/find_request", "w") as f:
        f.write(str(t))
    return {"requested": t}


def cancel_find():
    """'Stop looking' (owner, report 2026-11-03): ask the watcher to end the running search where the camera is."""
    t = time.time()
    with open(f"{STATE_DIR}/find_cancel", "w") as f:
        f.write(str(t))
    return {"cancelled": t}


# ---------- small settings shared with the watcher ----------

SETTING_KEYS = {"follow": bool, "talk_volume": int, "still_hold": bool}
SETTING_DEFAULTS = {"follow": False, "talk_volume": 75, "still_hold": True}   # still_hold: watch.Watch.still_hold


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

TALK_DIR = f"{STATE_DIR}/talk"
TALK_BIND = (petconf.SERVER_WG_IP, 8099)            # tunnel address only: the camera is the only client
_talk_last = {}
_talk_intents = {}


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

    if not petconf.caps()["talk"]:
        raise Busy("This camera has no speaker")
    talk_intent = _privacy_intent()
    if talk_intent[0]:
        raise Busy("The camera is paused. Resume it first.")
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
        if _privacy_intent() != talk_intent:
            raise Busy("Camera privacy changed. Record the message again.")
        _talk_intents[tok] = talk_intent
        reply = camera_run(f"curl -s -m 20 http://{TALK_BIND[0]}:{TALK_BIND[1]}/{tok}.wav | play -s -f wav -v {vol} 2>&1; echo rc=$?")
        if "Queued" not in reply:
            raise RuntimeError("the camera did not accept the audio")
        try:   # the watcher drops "loud sound" events while the owner's voice plays (chime ~1 s + the message)
            with open(f"{STATE_DIR}/talking", "w") as f:
                f.write(f"{time.time()} {time.time() + dur + 4}")
        except OSError:
            pass
        con = dbc()
        con.execute("INSERT INTO events(ts, kind, detail) VALUES(?,?,?)",
                    (time.time(), "you_spoke", json.dumps({"by": device_name, "seconds": round(dur, 1)})))
        con.close()
        return {"ok": True, "seconds": round(dur, 1)}
    finally:
        def cleanup():
            time.sleep(90)                      # the camera fetches within seconds
            _talk_intents.pop(tok, None)
            for f in (src, voice, out):
                try:
                    os.remove(f)
                except OSError:
                    pass
        threading.Thread(target=cleanup, daemon=True).start()


class TalkFiles(BaseHTTPRequestHandler):
    """Serves /<token>.wav from TALK_DIR on the tunnel address, nothing else."""

    def do_GET(self):
        if privacy_requested():
            self.send_error(409, "The camera is paused.")
            return
        name = self.path.lstrip("/")
        if "/" in name or not name.endswith(".wav") or name == "chime.wav" or len(name) < 20:
            self.send_error(404)
            return
        intent = _talk_intents.get(name[:-4])
        if intent is None or _privacy_intent() != intent:
            self.send_error(409, "Camera privacy changed. Record the message again.")
            return
        try:
            with open(f"{TALK_DIR}/{name}", "rb") as f:
                data = f.read()
        except OSError:
            self.send_error(404)
            return
        if _privacy_intent() != intent:
            self.send_error(409, "Camera privacy changed. Record the message again.")
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
            AppServer(TALK_BIND, TalkFiles).serve_forever()
        except OSError as e:
            print(f"talk file server: {e}; retrying in 15 s", flush=True)
            time.sleep(15)


# ---------- the pet: live status, days, diary ----------



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
    privacy = privacy_status()
    live["privacy"] = privacy
    if privacy["requested"]:
        # The local latch wins immediately, even while the watcher still has an old frame.
        live.update(status="privacy", present=False, score=0.0, box=None, det_t=None,
                    frame_t=None, frame_arrived=None, frame_motion=0.0, motion=0.0,
                    person=0.0, moving=False, moving_since=None, rest_since=None,
                    audio_ok=False, last_bark=None, find=None, imaging={})
    elif live.get("status") == "privacy" and privacy["phase"] == "active" and privacy["confirmed"]:
        live.update(status="starting", present=False, box=None, det_t=None, audio_ok=False)
    return live


_streams = {"n": 0}
STREAM_MAX = 8          # concurrent live streams (each is one server thread)
STREAM_LIFETIME = 600   # seconds; EventSource reconnects by itself


def live_payload():
    return read_live() | {"settings": get_settings()}


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


def activity_rows(con, start, end):
    """Correct legacy leave-debounce totals from complete retained observations.

    Keep original minute records intact, and do not replace a minute when only
    part of its raw evidence survives retention. Other minute fields stay intact.
    """
    rows = [dict(row) for row in con.execute("SELECT * FROM minute WHERE ts>=? AND ts<? ORDER BY ts", (start, end))]
    evidence = {r[0]: r[1:] for r in con.execute(
        "SELECT CAST(ts/60 AS INTEGER)*60, count(*), sum(score>=0.15), "
        "sum(score>=0.15 AND moving>0) FROM obs WHERE ts>=? AND ts<? GROUP BY CAST(ts/60 AS INTEGER)", (start, end))}
    for row in rows:
        raw = evidence.get(row["ts"])
        if raw and raw[0] >= row["n"] and raw[0] > 0:
            row["present"] = (raw[1] or 0) / raw[0] * row["n"]
            row["moving"] = (raw[2] or 0) / raw[0] * row["n"]
    return rows


def day_report(day=None):
    day, start, end = day_bounds(day)
    con = dbc()
    try:
        rows = activity_rows(con, start, end)
        events = con.execute("SELECT * FROM events WHERE ts>=? AND ts<? ORDER BY ts DESC LIMIT 200",
                             (start, end)).fetchall()
        # Totals over the whole day: the feed above is capped at 200, the tiles must not be.
        kind_counts = dict(con.execute("SELECT kind, count(*) FROM events WHERE ts>=? AND ts<? GROUP BY kind",
                                       (start, end)).fetchall())
        visits = con.execute("SELECT ts, detail FROM events WHERE kind='visit' AND ts>=? AND ts<?", (start, end)).fetchall()
    finally:
        con.close()
    buckets = [{"present": 0, "moving": 0, "n": 0, "sounds": 0, "person": 0, "snap": None} for _ in range(144)]
    visible = active = 0
    rest_run = longest_rest = 0
    first = last = None
    previous_ts = None
    for r in rows:
        if previous_ts is not None and r["ts"] - previous_ts > 60:
            rest_run = 0
        previous_ts = r["ts"]
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
    ev = []
    for e in events:
        detail = json.loads(e["detail"] or "{}")
        warning = detail.get("clip_warning") if isinstance(detail, dict) else None
        ev.append({"id": e["id"], "ts": e["ts"], "kind": e["kind"], "detail": detail,
                   "label": e["label"] if "label" in e.keys() else None,
                   "clip": f"/media/clips/{e['clip']}" if e["clip"] and not (warning and detail.get("clip_unusable", True)) else None,
                   "clip_warning": warning,
                   "thumb": f"/media/clips/{e['thumb']}" if e["thumb"] else None})
    dog_kinds = petconf.DOG_SOUND_KINDS
    sounds = sum(kind_counts.get(k, 0) for k in dog_kinds + ("loud_sound",))
    barks = sum(kind_counts.get(k, 0) for k in dog_kinds)
    zones = {}
    for ts, detail in visits:
        d = json.loads(detail or "{}")
        z = zones.setdefault(d.get("zone", "?"), {"kind": d.get("zone_kind"), "count": 0, "last": None, "seconds": 0})
        z["count"] += 1
        z["seconds"] += d.get("seconds", 0)
        z["last"] = max(z["last"] or 0, ts)
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

ZONES_FILE = f"{STATE_DIR}/zones.json"
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


RTSP_SUB = "rtsp://127.0.0.1:8554/sub"     # MediaMTX's 640x360 path (an MJPEG secondary stream for CAMERA=rpi)


def _mediamtx_frame(url=RTSP_SUB):
    """One JPEG from MediaMTX. MJPEG comes out as is (-c:v copy); anything else is encoded once."""
    import subprocess
    codec = "copy" if url == RTSP_SUB else "mjpeg"
    r = subprocess.run(["ffmpeg", "-v", "error", "-rtsp_transport", "tcp", "-i", url, "-frames:v", "1",
                        "-c:v", codec, "-f", "image2pipe", "-"], capture_output=True, timeout=10)
    if r.returncode != 0 or not r.stdout:
        raise OSError("no picture from the camera stream")
    return r.stdout


def live_frame():
    require_camera_active()
    if not petconf.THINGINO:
        picture = _mediamtx_frame()
        require_camera_active()
        return picture
    def call():
        req = urllib.request.Request(f"http://{CAM_HOST}/x/ch1.jpg", headers={"Cookie": _cam["cookie"] or ""})
        with urllib.request.urlopen(req, timeout=8) as r:
            return r.read(2_000_000)
    with _cam_lock:
        if not _cam["cookie"]:
            cam_login()
    try:
        picture = with_deadline(call, 10)
    except urllib.error.HTTPError as e:
        if e.code != 401:
            raise
        with _cam_lock:
            cam_login()
        picture = with_deadline(call, 10)
    require_camera_active()  # Do not return an in-flight JPEG after pause was requested.
    return picture


# ---------- wellbeing: her day against her own normal ----------

def day_metrics(day):
    """Activity numbers for one day from the per-minute table (plus zone visits)."""
    import datetime as dt

    d, start, end = day_bounds(day)
    prev_night = time.mktime((dt.date.fromisoformat(d) - dt.timedelta(days=1)).timetuple()) + 22 * 3600
    con = dbc()
    try:
        rows = activity_rows(con, start, end)
        night = activity_rows(con, prev_night, start + 6 * 3600)
        visits = con.execute("SELECT detail FROM events WHERE kind='visit' AND ts>=? AND ts<?", (start, end)).fetchall()
    finally:
        con.close()
    visible = active = run = longest = settles = 0
    prev_active = False
    previous_ts = None
    for r in rows:
        if previous_ts is not None and r["ts"] - previous_ts > 60:
            run = 0
            prev_active = False
        previous_ts = r["ts"]
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


# ---------- bug reports: everything needed to debug a moment, captured when it happens ----------

REPORTS_DIR = f"{STATE_DIR}/reports"
CLIENT_ERRORS = f"{REPORTS_DIR}/client-errors.jsonl"
_report_last = {}          # device id -> ts of its last report (rate limit)
_report_held = {}          # device id -> (ts, jpeg): the picture when its report form opened
REPORT_HOLD_S = 600        # older than this, the form was abandoned rather than still being typed
_clientlog = {"window": 0.0, "n": 0}


def _db_rows(sql, args):
    con = dbc()
    try:
        return [dict(r) for r in con.execute(sql, args)]
    finally:
        con.close()


def _report_clip(d, t):
    """Request -90/+10 s, retry short/unavailable recordings, and retain explicit coverage provenance."""
    from clipmedia import collect_recording, write_receipt
    if os.path.exists(f"{d}/clip.mp4"):
        return                 # preserve an existing report's original evidence
    time.sleep(max(0.0, t + 16 - time.time()))
    start = t - 90
    iso = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(start)) + f".{int((start % 1) * 1e6):06d}Z"
    url = "http://127.0.0.1:9996/get?" + urllib.parse.urlencode({"path": "cam", "start": iso, "duration": 100, "format": "mp4"})

    def call():
        with urllib.request.urlopen(url, timeout=30) as r:
            data = r.read(60_000_001)
            if len(data) > 60_000_000:
                raise ValueError("report recording exceeds the bounded response size")
            return data
    try:
        receipt = collect_recording(lambda: with_deadline(call, 60), f"{d}/clip.source.mp4", 100, f"{d}/clip-capture.json")
        receipt.update(requested_start_unix=start, requested_end_unix=t + 10,
                       post_report_tail_verified=False)
        write_receipt(f"{d}/clip-capture.json", receipt)
        if receipt["source_file"]:
            os.link(f"{d}/clip.source.mp4", f"{d}/clip.mp4")
        if not receipt["duration_complete"]:
            raise RuntimeError(f"partial or unavailable report recording: video {receipt['observed_video_duration_s']} of 100s; "
                               f"audio {receipt['observed_audio_duration_s']}s, complete={receipt['audio_window_complete']}; any recovered source is preserved")
    except Exception as e:  # noqa: BLE001 - the report is still useful without video
        with open(f"{d}/clip-error.txt", "w") as f:
            f.write(str(e))


def hold_report_frame(device_id):
    """The owner opened the report form: keep the picture they are describing. Typing takes long enough for a
    running search to reach another view, and a report's picture has to match its note."""
    _report_held.pop(device_id, None)
    _report_held[device_id] = (time.time(), live_frame())
    return {"ok": True}


def _save_report_frames(d, rep, device_id, t):
    """frame.jpg: the picture from when the form opened, or the current one if there is none; frame_sent.jpg: the
    current one as well, when they differ. A paused camera keeps neither."""
    held = _report_held.pop(device_id, None)
    if held and t - held[0] > REPORT_HOLD_S:
        held = None
    sent = None
    try:
        sent = live_frame()
    except Busy as e:
        rep["frame_error"] = str(e)
        held = None
    except Exception as e:  # noqa: BLE001
        rep["frame_error"] = str(e)
    try:
        frame = held[1] if held else sent
        if frame is not None:
            with open(f"{d}/frame.jpg", "wb") as f:
                f.write(frame)
            rep["frame_t"] = held[0] if held else t
        if held and sent is not None:
            with open(f"{d}/frame_sent.jpg", "wb") as f:
                f.write(sent)
    except OSError as e:
        rep["frame_error"] = str(e)


def file_report(device_id, device_name, body):
    t = time.time()
    if t - _report_last.get(device_id, 0) < 20:
        raise Busy("You just sent one. Try again in a few seconds.")
    _report_last[device_id] = t
    rid = time.strftime("%Y%m%d-%H%M%S", time.localtime(t))
    d = f"{REPORTS_DIR}/{rid}"
    os.makedirs(d, exist_ok=True)
    client = body.get("client") if isinstance(body.get("client"), dict) else {}
    client_json = json.dumps(client)[:20_000]
    rep = {
        "id": rid, "ts": t, "local_time": time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(t)),
        "by": device_name, "device": device_id, "note": str(body.get("note") or "")[:2000],
        "client": json.loads(client_json) if len(client_json) < 20_000 else {"truncated": client_json},
        "live": read_live(), "settings": get_settings(), "zones": list_zones(),
        "obs_15min": _db_rows("select * from obs where ts > ? order by ts", (t - 900,)),
        "events_3h": _db_rows("select id, ts, kind, detail, label from events where ts > ? order by ts", (t - 10800,)),
    }
    _save_report_frames(d, rep, device_id, t)
    with open(f"{d}/report.json", "w") as f:
        json.dump(rep, f, indent=1, default=str)
    threading.Thread(target=_report_clip, args=(d, t), daemon=True).start()
    return {"id": rid}


# ---------- owner marks: "she is HERE" (or "she isn't in this picture") ----------
# The detector's complete misses never reach training data: watch.py saves a candidate frame only when it proposed
# an animal. A mark is the owner's own label for the current picture: where the pet is, or that she is absent.
# Labels only: a mark never changes presence or moves the camera. Bounded: one per device every MARK_MIN_GAP_S,
# at most MARK_MAX_PER_DAY a day; watch.py prunes the folder after RETAIN["marks_days"].
MARKS_DIR = f"{STATE_DIR}/marks"
MARK_MIN_GAP_S = 5
MARK_MAX_PER_DAY = 200
_mark_last = {}            # device id -> ts of its last mark


def mark_pet(device_id, device_name, body):
    """Save the live picture with the owner's tap on it. body: {"fx", "fy"} as /api/aim (-0.5..0.5 from the
    centre of the picture), or {"absent": true}. Returns the saved mark's id."""
    absent = body.get("absent") is True
    point = None
    if not absent:
        try:
            fx, fy = float(body.get("fx")), float(body.get("fy"))
        except (TypeError, ValueError):
            raise ValueError("Tap where she is in the picture")
        if not (abs(fx) <= 0.5 and abs(fy) <= 0.5):
            raise ValueError("That tap is outside the picture")
        point = [round(fx + 0.5, 4), round(fy + 0.5, 4)]     # 0..1 from the top-left corner, like detector boxes
    t = time.time()
    if t - _mark_last.get(device_id, 0) < MARK_MIN_GAP_S:
        raise Busy("Saved a moment ago. Try again in a few seconds.")
    day = time.strftime("%Y-%m-%d", time.localtime(t))
    d = f"{MARKS_DIR}/{day}"
    try:
        today = sum(1 for n in os.listdir(d) if n.endswith(".json"))
    except OSError:
        today = 0
    if today >= MARK_MAX_PER_DAY:
        raise Busy(f"That's {MARK_MAX_PER_DAY} marks today, enough to learn from. Thank you!")
    frame = live_frame()                 # raises while the camera is paused: no picture, no mark
    _mark_last[device_id] = t
    try:
        pose = list(position())
    except Exception:  # noqa: BLE001 - the label is still useful without the motor position
        pose = None
    live = read_live()
    mid = time.strftime("%H%M%S", time.localtime(t)) + f"-{int((t % 1) * 1000):03d}"
    os.makedirs(d, exist_ok=True)
    with open(f"{d}/{mid}.jpg", "wb") as f:
        f.write(frame)
    rec = {"id": f"{day}/{mid}", "ts": t, "by": device_name, "device": device_id,
           "label": "absent" if absent else "pet", "point": point, "pose": pose,
           # what the watcher believed at that moment: a "pet" mark over present=False is a detector miss
           "watch": {k: live.get(k) for k in ("present", "status", "score", "box", "det_t", "last_seen")}}
    with open(f"{d}/{mid}.json", "w") as f:
        json.dump(rec, f)
    missed = not absent and not live.get("present")
    print(f"mark: {rec['label']} at {point} by {device_name}{' (detector missed her)' if missed else ''}", flush=True)
    return {"id": rec["id"], "label": rec["label"], "missed": missed}


def list_reports(limit=20):
    """Compatibility list; paginated clients use report_page to reach older requests."""
    return report_page(limit)["reports"]


def report_page(limit=20, before=None):
    """Read all request ages through stable ID cursors; resolution is separate from replies."""
    import re
    limit = int(limit)
    if not 1 <= limit <= 100:
        raise ValueError("report limit must be 1..100")
    if before is not None and not re.fullmatch(r"\d{8}-\d{6}", before):
        raise ValueError("invalid report cursor")
    out = []
    try:
        ids = sorted((n for n in os.listdir(REPORTS_DIR) if re.fullmatch(r"\d{8}-\d{6}", n)), reverse=True)
    except OSError:
        ids = []
    remaining = [rid for rid in ids if before is None or rid < before]
    selected = remaining[:limit]
    for rid in selected:
        try:
            out.append(requestboard.load(REPORTS_DIR, rid))
        except (OSError, ValueError, TypeError):
            continue
    return {"reports": out, "total": len(ids),
            "next_before": selected[-1] if len(remaining) > len(selected) else None}


def version_info():
    """The installed version and the daily update check's outcome (server/update.py writes update.json)."""
    try:
        d = json.load(open(os.path.join(STATE_DIR, "update.json")))
    except (OSError, ValueError):
        d = {}
    try:
        d["current"] = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "VERSION")).read().strip()
    except OSError:
        d.setdefault("current", "")
    return {k: d.get(k) for k in ("current", "latest", "result", "checked", "updated_at", "url")}


def agent_dashboard(thread=None):
    """The super-agent's state for the owners' dashboard (/agent): goals, open threads, decisions, heartbeats,
    and the steering file the agent obeys (pushed here by tools/superagent.py on every tick)."""
    import agentdb
    import re
    with agentdb.db() as c:
        d = agentdb.cmd_dashboard(c, {})
        st = c.execute("SELECT title, summary FROM threads WHERE id='steer'").fetchone()
        if isinstance(thread, str) and re.fullmatch(r'goal:[a-zA-Z0-9_.-]{1,100}', thread):
            selected = agentdb.cmd_recall(c, {"ref": thread})
            d["selected_thread"] = selected if "id" in selected else None
    d["north"], d["steer"] = (st["title"], st["summary"]) if st else ("", "")
    return d


def monitoring_goals():
    """Existing agent goals as read-only monitors; viewing them never starts work."""
    import agentdb
    import goalmonitor
    return goalmonitor.read(agentdb.DB)


def log_client_error(body, ua):
    """Uncaught errors from the page (iPhone-only bugs are otherwise invisible). Capped: 30/min, 1 MB."""
    now = time.time()
    if now - _clientlog["window"] > 60:
        _clientlog.update(window=now, n=0)
    _clientlog["n"] += 1
    if _clientlog["n"] > 30:
        return
    os.makedirs(REPORTS_DIR, exist_ok=True)
    if os.path.exists(CLIENT_ERRORS) and os.path.getsize(CLIENT_ERRORS) > 1_000_000:
        os.replace(CLIENT_ERRORS, CLIENT_ERRORS + ".1")
    row = {"ts": now, "ua": (ua or "")[:200]} | {k: str(body.get(k, ""))[:2000] for k in ("msg", "src", "line", "col", "stack", "where")}
    with open(CLIENT_ERRORS, "a") as f:
        f.write(json.dumps(row) + "\n")


SNAP_STEP_REACH_MIN = 180   # how far one step looks past minutes with no picture (a search or privacy time)


def snap_at(ts, step=0):
    """Per-minute snapshot for a timestamp (snapshots are kept 14 days), with the minute it really shows.
    step 0: the nearest within 2 minutes. step +1/-1: the next saved picture after/before ts, so stepping
    crosses minutes with no picture (the watcher saves none during a search) instead of sticking on one."""
    t = time.localtime(float(ts))
    minute = int(float(ts) // 60 * 60)
    step = {"1": 1, "-1": -1}.get(str(step), 0)
    deltas = [step * 60 * n for n in range(1, SNAP_STEP_REACH_MIN + 1)] if step else (0, -60, 60, -120, 120)
    for delta in deltas:
        tt = time.localtime(minute + delta)
        rel = f"{time.strftime('%Y-%m-%d', tt)}/{time.strftime('%H-%M', tt)}.jpg"
        if os.path.exists(f"{STATE_DIR}/snaps/{rel}"):
            return {"snap": f"/media/snaps/{rel}", "ts": minute + delta}
    return {"snap": None, "asked": time.strftime("%H:%M", t)}


def read_file(name):
    with open(os.path.join(HERE, name), "rb") as f:
        return f.read()


PUBLIC_HOST = ENV.get("PUBLIC_HOST") or petconf.PUBLIC_HOST


def build_id():
    """Changes whenever the app's code changes (every deploy restarts the app). Open phones compare it with the
    build they loaded and reload themselves: a Home Screen app on iOS can stay resident for days otherwise."""
    h = hashlib.sha1()
    for n in sorted(os.listdir(HERE)):
        if n.endswith((".html", ".js", ".css", ".py")):
            with open(os.path.join(HERE, n), "rb") as f:
                h.update(n.encode() + f.read())
    return h.hexdigest()[:12]


BUILD = build_id()


def page(name, **subs):
    """A page or asset with its {{PLACEHOLDERS}} filled: the pet's name and pronouns, plus subs."""
    body = read_file(name)
    for k, v in (petconf.template_vars() | {"BUILD": BUILD} | subs).items():
        body = body.replace(f"{{{{{k}}}}}".encode(), v.encode())
    return body


html_escape = petconf.html_escape


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

    def parse_request(self):
        # Start the response-time clock once the request line and headers have arrived: on a
        # keep-alive connection handle_one_request first WAITS for the next request, and timing
        # from there counted the idle gap between touches as response time (4.2 s for a nudge).
        ok = super().parse_request()
        self._t0 = time.perf_counter()
        return ok

    def device(self):
        if self._auth is None:
            self._auth = {"device": device_from_cookie(self.headers.get("Cookie"), self.client_ip())}
        return self._auth["device"]

    def legacy(self):
        return session_ok(self.headers.get("Cookie"))

    def authed(self):
        return bool(self.device()) or self.legacy()

    def send(self, code, body=b"", ctype="text/plain; charset=utf-8", headers=None):
        path = self.path.split("?")[0]
        if path.startswith("/api/") and path not in USAGE_SKIP:
            dev = ((self._auth or {}).get("device") or {}).get("id") if isinstance(self._auth, dict) else None
            usage(dev, "api", f"{self.command} {path}", (time.perf_counter() - getattr(self, "_t0", time.perf_counter())) * 1000,
                  code < 400, {"status": code} if code >= 400 else None)
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

    def live_stream(self):
        """Server-sent events: push live.json to the phone the moment the watcher rewrites it,
        instead of the phone polling every 2 s (profiled 2026-11-02: about 1 s saved on average)."""
        if _streams["n"] >= STREAM_MAX:
            return self.send_json(503, {"error": "too many live streams"})
        _streams["n"] += 1
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.end_headers()
            path, last_mt, last_beat, end = f"{STATE_DIR}/live.json", None, 0.0, time.time() + STREAM_LIFETIME
            while time.time() < end:
                try:
                    mt = os.path.getmtime(path)
                except OSError:
                    mt = None
                now_ = time.time()
                if mt != last_mt or now_ - last_beat > 5:      # also every 5 s, so "watcher down" still shows
                    last_mt, last_beat = mt, now_
                    self.wfile.write(b"data: " + json.dumps(live_payload()).encode() + b"\n\n")
                    self.wfile.flush()
                time.sleep(0.1)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass
        finally:
            _streams["n"] -= 1
            self.close_connection = True

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
        petconf.refresh()
        if path == "/healthz":
            return self.send(200, b"ok")
        if path == "/auth":  # Caddy forward_auth for video and media
            return self.send(200 if self.authed() else 401)
        if path == "/manifest.webmanifest":
            return self.send(200, page("manifest.webmanifest"), "application/manifest+json")
        if path == "/sw.js":
            return self.send(200, page("sw.js"), "text/javascript", {"Service-Worker-Allowed": "/"})
        if path in ("/icon.png", "/icon-maskable.png", "/badge.png", "/apple-touch-icon.png", "/favicon.png", "/logo.png", "/logo-light.png"):
            return self.send(200, read_file(path[1:]), "image/png", {"Cache-Control": "max-age=86400"})
        if path == "/favicon.svg":
            return self.send(200, read_file("favicon.svg"), "image/svg+xml", {"Cache-Control": "max-age=86400"})
        if path in ("/themes.css", "/theme-controller.js", "/theme-data.js", "/theme-app.js"):   # needed on /login too
            ctype = "text/css; charset=utf-8" if path.endswith(".css") else "text/javascript; charset=utf-8"
            return self.send(200, read_file(path[1:]), ctype, {"Cache-Control": "max-age=300"})
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
            return self.send_json(200, live_payload())
        if path == "/api/live/stream":
            return self.live_stream()
        if path == "/api/day":
            return self.camera_call(day_report, q.get("d"))
        if path == "/api/diary":
            return self.camera_call(diary)
        if path == "/api/snap":
            return self.camera_call(snap_at, q.get("ts", time.time()), q.get("step", 0))
        if path == "/api/devices":
            return self.send_json(200, list_devices((self.device() or {}).get("id")))
        if path == "/api/zones":
            return self.send_json(200, list_zones())
        if path == "/api/views/insights":
            return self.camera_call(view_insights, q.get("days", 7))
        if path == "/api/reports":
            if q.get("page") == "1":
                return self.camera_call(report_page, q.get("limit", 20), q.get("before"))
            return self.send_json(200, list_reports())
        if path == "/api/reports/frame":
            try:
                return self.send(200, requestboard.frame(REPORTS_DIR, q.get("id")), "image/jpeg")
            except ValueError as error:
                return self.send_json(400, {"error": str(error)})
            except OSError:
                return self.send_json(404, {"error": "The saved report image is unavailable."})
        if path == "/api/monitoring":
            return self.send_json(200, monitoring_goals())
        if path in ("/requests.js", "/requests.css"):
            ctype = "text/css; charset=utf-8" if path.endswith(".css") else "text/javascript; charset=utf-8"
            return self.send(200, read_file(path[1:]), ctype)
        if path == "/api/agent":
            return self.send_json(200, agent_dashboard(q.get("thread")))
        if path == "/api/version":
            return self.send_json(200, version_info())
        if path == "/api/build":
            return self.send_json(200, {"build": BUILD})
        if path == "/agent":
            return self.send(200, page("agent.html"), "text/html; charset=utf-8")
        if path == "/api/wellbeing":
            return self.camera_call(wellbeing, q.get("d"))
        if path == "/api/frame.jpg":
            try:
                return self.send(200, live_frame(), "image/jpeg")
            except Busy as e:
                return self.send_json(409, {"error": str(e), "privacy": privacy_status()})
            except Exception as e:  # noqa: BLE001
                return self.send_json(502, {"error": f"camera unreachable: {e}"})
        if path == "/api/push":
            return self.camera_call(lambda: push_status((self.device() or {}).get("id")) | push_key())
        if path == "/api/state":
            return self.camera_call(state)
        # ---- Room tab: your room model and where the pet was on its floor (signed-in devices only) ----
        if path in ("/room.js", "/vendor/three.module.min.js", "/vendor/OrbitControls.js"):
            return self.send(200, read_file(path[1:]), "text/javascript; charset=utf-8", {"Cache-Control": "max-age=300"})
        if path == "/api/room":
            return self.send_json(200, roomapi.info())
        if path == "/api/room/tex":
            f = roomapi.texture(q.get("f", ""))
            if not f:
                return self.send(404, b"not found")
            with open(f, "rb") as fh:
                body = fh.read()
            ctype = "image/png" if f.endswith(".png") else "application/octet-stream" if f.endswith(".u16") else "image/jpeg"
            return self.send(200, body, ctype, {"Cache-Control": "private, max-age=3600"})
        if path == "/api/room/positions":
            now_t = time.time()
            try:
                until = min(float(q.get("until", now_t)), now_t)
                since = max(float(q.get("since", until - 86400)), until - 3 * 86400)    # at most 3 days per request
            except ValueError:
                return self.send_json(400, {"error": "since/until must be unix times"})
            con = dbc()
            try:
                return self.send_json(200, {"since": since, "until": until, "rows": roomapi.positions(con, since, until),
                                            "poses": roomapi.camera_poses(con, since, until)})
            finally:
                con.close()
        return self.send(404, b"not found")

    def camera_call(self, fn, *args):
        try:
            return self.send_json(200, fn(*args))
        except PrivacyError as e:
            return self.send_json(503, {"error": str(e), "privacy": privacy_status()})
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
        petconf.refresh()
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
        if path == "/api/reports/move":
            try:
                row = requestboard.move(REPORTS_DIR, body.get("id"), body.get("state"),
                                        body.get("revision"), (self.device() or {}).get("name", "Owner"),
                                        body.get("reason", ""))
                return self.send_json(200, row)
            except requestboard.Conflict as e:
                return self.send_json(409, {"error": str(e)})
            except FileNotFoundError:
                return self.send_json(404, {"error": "This request no longer exists."})
            except (ValueError, TypeError) as e:
                return self.send_json(400, {"error": str(e)})
        if path == "/api/pet":     # the pet's name and pronoun, set from the app (first open or Settings)
            try:
                return self.send_json(200, petconf.save_pet(body.get("name"), body.get("pronoun")))
            except ValueError as e:
                return self.send_json(400, {"error": str(e)})
        if path == "/api/usage":
            return self.send_json(200, record_ui_events(me, body))
        if path == "/api/clientlog":
            log_client_error(body, self.headers.get("User-Agent"))
            return self.send_json(200, {"ok": True})
        routes = {
            "/api/report": lambda: file_report(me or "legacy", (self.device() or {}).get("name", "Someone"), body),
            "/api/report/hold": lambda: hold_report_frame(me or "legacy"),
            "/api/mark": lambda: mark_pet(me or "legacy", (self.device() or {}).get("name", "Someone"), body),
            "/api/find": lambda: request_find(),
            "/api/find/cancel": lambda: cancel_find(),
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


def ensure_cam_ssh():
    """Authorise this server's key on the camera once (through the web command runner), so the fast
    channel works. Idempotent; failures just leave the CGI path in use."""
    try:
        _camsh.run("true")
        motor_params()
        return
    except Exception:  # noqa: BLE001
        pass
    try:
        pub = open(CAM_SSH_KEY + ".pub").read().strip()
        assert pub.startswith("ssh-ed25519 ") and "'" not in pub
        camera_run(f"mkdir -p /root/.ssh && chmod 700 /root/.ssh && touch /root/.ssh/authorized_keys && "
                   f"(grep -qF '{pub}' /root/.ssh/authorized_keys || echo '{pub}' >> /root/.ssh/authorized_keys) && "
                   f"chmod 600 /root/.ssh/authorized_keys")
        _camsh.down_until = 0
        _camsh.run("true")
        print("camera ssh: fast channel ready", flush=True)
        try:
            motor_params()       # warm the caches the app reads on every open (/api/state was ~1 s cold)
        except Exception:  # noqa: BLE001
            pass
    except Exception as e:  # noqa: BLE001
        print(f"camera ssh: not available, using the web CGIs ({e})", flush=True)


if __name__ == "__main__":
    if not SECRET or not ENV.get("APP_PASSWORD_HASH"):
        print("WARNING: APP_SECRET or APP_PASSWORD_HASH missing; nobody can log in yet", flush=True)
    init_auth_db()
    if petconf.THINGINO:          # the Thingino camera's SSH channel and its talk-back WAV server
        threading.Thread(target=serve_talk_files, daemon=True).start()
        threading.Thread(target=ensure_cam_ssh, daemon=True).start()
    AppServer(("127.0.0.1", 8080), Handler).serve_forever()
