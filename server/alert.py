#!/usr/bin/env python3
"""petcam-alert: the absence alerter, run by a systemd timer every 3 minutes (separate from
petcam-watch on purpose, so a dead watcher still gets reported).

Checks: watcher heartbeat, video relay + recording freshness, and new events written by the
watcher (camera offline/back, picture frozen, barking, comings and goings). Sends iPhone/web
push notifications to every subscribed device whose preferences allow that kind.

  alert.py            run the checks once (what the timer does)
  alert.py --test ID  send a test notification to device ID
"""
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time

sys.path.insert(0, "/opt/petcam")
import petconf  # noqa: E402

PET, APP = petconf.PET, petconf.APP_NAME

STATE = "/var/lib/petcam"
DB = f"{STATE}/petcam.db"
ALERT_STATE = f"{STATE}/alert_state.json"
HEARTBEAT = f"{STATE}/heartbeat"
RECORDINGS = "/var/lib/mediamtx/recordings/cam"
VAPID_KEY = "/etc/petcam/vapid_private.pem"
CONTACT = "mailto:petcam@localhost"
BARK_EVERY = 30 * 60          # at most one barking alert per 30 minutes
DISTRESS_EVERY = 10 * 60      # whining / howling / growling: at most one alert per 10 minutes
SUMMARY_HOUR = 19             # daily summary at 7 pm local
WATCH_STALE = 120             # watcher heartbeat older than this = watcher down / camera unreachable
AUTOPILOT_HEARTBEAT = f"{STATE}/autopilot_ok"  # touched by tools/autopilot.py on the owner's computer
AUTOPILOT_STALE = 36 * 3600   # nightly job: alert after a day and a half without one
RECORDING_STALE = 180         # newest recording segment older than this = video relay stuck


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def db():
    con = sqlite3.connect(DB, timeout=10, isolation_level=None)
    con.execute("PRAGMA busy_timeout=5000")
    con.row_factory = sqlite3.Row
    return con


def with_deadline(fn, seconds):
    box = {}

    def run():
        try:
            box["ok"] = fn()
        except Exception as e:  # noqa: BLE001
            box["err"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(seconds)
    if t.is_alive():
        raise TimeoutError(f"no answer within {seconds}s")
    if "err" in box:
        raise box["err"]
    return box.get("ok")


def send(title, body, kind, tag=None, only_device=None):
    con = db()
    try:
        rows = con.execute("SELECT * FROM push_subs").fetchall()
    except sqlite3.OperationalError:          # table not created yet (app not started)
        rows = []
    targets = [r for r in rows
               if (only_device is None or r["device_id"] == only_device)
               and (kind == "test" or json.loads(r["prefs"] or "{}").get(kind, False))]
    if not targets:
        con.close()
        log(f"'{title}': no subscribed device wants '{kind}'")
        return 0
    # Imported only when there is someone to notify: pywebpush + cryptography cost ~8 s of CPU
    # to load on this box, which every 2 minutes would eat its whole CPU baseline.
    if "/opt/petcam/venv/lib" not in " ".join(sys.path):      # run by system python3 normally
        import glob
        sys.path[:0] = glob.glob("/opt/petcam/venv/lib/python3*/site-packages")
    from pywebpush import WebPushException, webpush

    sent = 0
    for r in targets:
        payload = json.dumps({"title": title, "body": body, "tag": tag or kind, "url": "/"})
        try:
            with_deadline(lambda: webpush(json.loads(r["sub"]), payload, vapid_private_key=VAPID_KEY,
                                          vapid_claims={"sub": CONTACT}, timeout=10, ttl=3600), 20)
            sent += 1
        except WebPushException as e:
            code = getattr(e.response, "status_code", None)
            log("push failed", code, str(e)[:120])
            if code in (404, 410):            # subscription gone (app removed / permission revoked)
                con.execute("DELETE FROM push_subs WHERE endpoint=?", (r["endpoint"],))
        except Exception as e:  # noqa: BLE001
            log("push error", type(e).__name__, str(e)[:120])
    con.close()
    log(f"sent '{title}' to {sent} device(s)")
    return sent


def load_state():
    try:
        with open(ALERT_STATE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_state(st):
    tmp = ALERT_STATE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f)
    os.replace(tmp, ALERT_STATE)


def age(path):
    try:
        return time.time() - os.path.getmtime(path)
    except OSError:
        return None


def newest_recording_age():
    """Seconds since the newest recording segment was written. petcam is in the mediamtx group
    (install.sh) so it can list the folder; MediaMTX's own /list API re-parses every segment
    and cost ~2.7 s of CPU per call on this box."""
    try:
        names = os.listdir(RECORDINGS)
    except OSError as e:
        log("recordings:", e)
        return None
    ages = [age(os.path.join(RECORDINGS, n)) for n in names]
    return min((a for a in ages if a is not None), default=None)


def incident(st, key, bad, down_msg, up_msg):
    """Alert once when a problem starts, once when it clears."""
    inc = st.setdefault("incidents", {})
    if bad and key not in inc:
        inc[key] = time.time()
        send(*down_msg, kind="camera", tag=key)
    elif not bad and key in inc:
        mins = round((time.time() - inc.pop(key)) / 60)
        send(up_msg[0], f"{up_msg[1]} (down about {mins} min)", kind="camera", tag=key)


def check():
    st = load_state()
    # 1. Watcher alive? (also covers "camera unreachable": the watcher only heartbeats on a good frame)
    hb = age(HEARTBEAT)
    incident(st, "watch", hb is None or hb > WATCH_STALE,
             (f"{APP}: not getting pictures", "The camera or the watcher stopped responding. It will retry automatically."),
             (f"{APP} is back", "Pictures are coming in again"))
    # 2. Video relay alive and recording?
    relay = subprocess.run(["systemctl", "is-active", "--quiet", "mediamtx"]).returncode == 0
    rec = newest_recording_age()
    incident(st, "video", not relay or rec is None or rec > RECORDING_STALE,
             (f"{APP}: video stopped", "Live video and recording are not running."),
             (f"{APP} video is back", "Live video and recording are running again"))
    # 3. The nightly self-improvement loop (runs on the owner's computer). Only once it has ever run.
    ap = age(AUTOPILOT_HEARTBEAT)
    if ap is not None:
        incident(st, "autopilot", ap > AUTOPILOT_STALE,
                 (f"{APP}: nightly check-up hasn't run", "The computer that retrains the pet detector hasn't checked in "
                  "for a day and a half. Is it asleep or off? Everything else keeps working."),
                 (f"{APP}: nightly check-up is back", "It ran again"))
    # 4. New events from the watcher. Some are stamped earlier than they are written ("left" uses
    #    the last sighting, sounds their start), and SQLite reuses ids after deletes, so neither
    #    an id nor a time watermark is safe: re-read the last 30 min, skip (id, ts) already handled.
    con = db()
    first_run = "seen_events" not in st
    seen = set(map(tuple, st.get("seen_events", [])))
    rows = con.execute("SELECT * FROM events WHERE ts > ? ORDER BY ts", (time.time() - 1800,)).fetchall()
    con.close()
    for e in rows:
        key = (e["id"], round(e["ts"], 3))
        if key in seen:
            continue
        seen.add(key)
        if first_run:                                  # first run: don't replay history
            continue
        d = json.loads(e["detail"] or "{}")
        when = time.strftime("%-I:%M %p", time.localtime(e["ts"]))
        if e["kind"] in ("barking", "dog_sound") and time.time() - st.get("last_bark", 0) > BARK_EVERY:
            st["last_bark"] = time.time()
            send(f"{PET} is barking", f"Barking at {when}. Tap to look.", kind="barking", tag="bark")
        elif e["kind"] in ("whining", "howling", "growling") and time.time() - st.get("last_distress", 0) > DISTRESS_EVERY:
            # Whining and howling can mean distress (e.g. separation), so they alert more readily.
            st["last_distress"] = time.time()
            send(f"{PET} is {e['kind']}", f"Heard at {when}. Tap to look and listen.", kind="barking", tag="distress")
        elif e["kind"] == "camera_frozen" and str(d.get("action", "")).startswith("restart streamer"):
            send(f"{APP}: video stopped, restarting it", "The camera's video service stopped responding; it is being restarted.", kind="camera", tag="heal")
        elif e["kind"] == "camera_frozen" and d.get("action") == "reboot camera":
            send(f"{APP} rebooted the camera", "The picture froze and restarting the encoder didn't help.", kind="camera")
        elif e["kind"] == "view_blocked":
            send(f"{APP} can't see the room", "The picture has been blurry or blocked for 10 minutes. "
                 "The camera may have tipped over or be facing a wall.", kind="camera", tag="view")
        elif e["kind"] == "view_clear":
            send(f"{APP} can see again", "The view looks normal again.", kind="camera", tag="view")
        elif e["kind"] == "appeared":
            send(f"{PET} is in view", f"Came into view at {when}", kind="comings", tag="comings")
        elif e["kind"] == "left":
            send(f"{PET} left the view", f"Last seen at {when}", kind="comings", tag="comings")
    st["seen_events"] = sorted(seen, key=lambda k: k[1])[-300:]
    daily_summary(st)
    st["last_run"] = time.time()
    save_state(st)


def fmt_min(m):
    m = int(round(m))
    return f"{m} min" if m < 60 else (f"{m // 60}h {m % 60}m" if m % 60 else f"{m // 60}h")


def daily_summary(st):
    """Once a day at 7 pm local: the pet's day against its own normal (same numbers as the app)."""
    lt = time.localtime()
    today = time.strftime("%Y-%m-%d", lt)
    if lt.tm_hour < SUMMARY_HOUR or st.get("summary_day") == today:
        return
    st["summary_day"] = today
    sys.path.insert(0, "/opt/petcam")
    import app                                   # stdlib only; reads the same SQLite as the web app
    w = app.wellbeing(today)
    m = w["today"]
    if m["minutes"] < 60:
        return
    parts = [f"In view {fmt_min(m['visible'])}", f"active {fmt_min(m['active'])}", f"longest nap {fmt_min(m['longest_rest'])}"]
    if m["water"] or m["food"]:
        parts.append(f"water ×{m['water']}, food ×{m['food']}")
    body = w["verdict"] + ". " + " · ".join(parts)
    if w.get("notes"):
        body += ". Note: " + "; ".join(w["notes"])
    send(f"{PET}'s day", body, kind="daily", tag="daily")

if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--test":
        n = send(APP, "Notifications are on. You'll hear from me if something needs you.", kind="test",
                 only_device=sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] else None)
        if not n:
            print("no subscribed device received it")
            sys.exit(1)
        print(f"sent to {n} device(s)")
    elif len(sys.argv) >= 4 and sys.argv[1] == "--notify":
        # Used by tools/autopilot.py: "--notify TITLE BODY [kind]"; kind defaults to camera.
        n = send(sys.argv[2][:80], sys.argv[3][:300], kind=sys.argv[4] if len(sys.argv) > 4 else "camera", tag="autopilot")
        print(f"sent to {n} device(s)")
    else:
        check()
