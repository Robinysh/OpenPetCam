#!/usr/bin/env python3
"""A sweep the server runs on its own at a set time (no laptop needed): the camera visits a list of poses, takes a
1080p main-stream frame at each and returns to where it was. Used after a re-mount to photograph the room again from
the new lens position (stereo against the earlier sweep; sim/world/remount.py), at night (infrared) and in daylight.

    scheduled_sweep.py schedule TAG "YYYY-MM-DD HH:MM" poses.json   write the plan and a one-shot timer (root)
    scheduled_sweep.py run TAG                                       the timer runs this (root, systemd, hard timeout)
    scheduled_sweep.py check                                         called by alert.py: overdue or failed sweeps

Unattended: every camera call has a timeout and the unit has TimeoutStartSec; the run writes done.json (the heartbeat)
or failed.json; alert.py alerts once when a sweep is failed, or is still missing an hour after its time.
Skips (and says so) when privacy is on or another capture holds /var/lib/petcam/capture.lock.
Pictures stay on the server under /var/lib/petcam/sweeps/<tag>/ (they show the home)."""
import json
import os
import subprocess
import sys
import time

STATE = "/var/lib/petcam"
ROOT = os.path.join(STATE, "sweeps")
LOCK = os.path.join(STATE, "capture.lock")
LOCK_STALE_S = 1800
CAM = ["ssh", "-T", "-i", f"{STATE}/ssh/id_ed25519", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
       "-o", f"UserKnownHostsFile={STATE}/ssh/known_hosts", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=2"]
GRACE_S = 3600                     # a sweep with no result this long after its time is overdue


def sh(cmd, timeout, **kw):
    return subprocess.run(cmd, capture_output=True, timeout=timeout, **kw)


def cam_host():
    import petconf                                   # the camera address the app uses (CAM_HOST in site.env)
    return f"root@{petconf.SITE.get('CAM_HOST') or petconf.CAM_HOST}"


def site_tz():
    import petconf
    return petconf.SITE.get("TZ") or "UTC"


def plan_path(tag):
    return os.path.join(ROOT, tag, "plan.json")


def schedule(tag, when, poses_file):
    d = os.path.join(ROOT, tag)
    os.makedirs(d, exist_ok=True)
    os.chmod(ROOT, 0o755); os.chmod(d, 0o755)                         # alert.py (user petcam) reads the results
    poses = json.load(open(poses_file))
    from datetime import datetime
    from zoneinfo import ZoneInfo
    tz = site_tz()                                                     # the home's time, not the server's clock
    at = datetime.strptime(when, "%Y-%m-%d %H:%M").replace(tzinfo=ZoneInfo(tz)).timestamp()
    json.dump({"tag": tag, "at": at, "when": when, "poses": poses}, open(plan_path(tag), "w"), indent=1)
    unit = f"petcam-sweep-{tag}"
    open(f"/etc/systemd/system/{unit}.timer", "w").write(
        f"[Unit]\nDescription=petcam scheduled sweep {tag}\n\n[Timer]\nOnCalendar={when} {tz}\nUnit=petcam-sweep@{tag}.service\n"
        "Persistent=false\nAccuracySec=1min\n\n[Install]\nWantedBy=timers.target\n")
    sh(["systemctl", "daemon-reload"], 30)
    sh(["systemctl", "enable", "--now", f"{unit}.timer"], 30)
    print(f"scheduled {tag} at {when} ({len(poses)} poses)")


def _goto(x, y):
    code = ("import sys; sys.path.insert(0, '/opt/petcam'); import app; app.goto(%d, %d, app.QUIET_SPEED); "
            "app.wait_idle(timeout=40); print(app.motor_status())" % (int(x), int(y)))
    r = sh(["sudo", "-u", "petcam", "/opt/petcam/venv/bin/python", "-c", code], 90, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"goto {x},{y}: {r.stderr[-200:]}")
    st = eval(r.stdout.strip().splitlines()[-1])                     # ('0', x, y)
    return int(st[1]), int(st[2])


def _snapshot(path):
    r = sh(CAM + [cam_host(), "prudyntctl snapshot -c 0 -q 88"], 40)
    if r.returncode != 0 or len(r.stdout) < 10000:
        raise RuntimeError(f"snapshot: {r.stderr[-200:]!r}")
    open(path, "wb").write(r.stdout)


def _motors():
    r = sh(CAM + [cam_host(), "motors -j"], 20, text=True)
    m = json.loads(r.stdout)
    return int(m["xpos"]), int(m["ypos"])


def run(tag):
    d = os.path.join(ROOT, tag)
    plan = json.load(open(plan_path(tag)))
    result = {"tag": tag, "started": time.time(), "poses": []}

    def finish(name, **extra):
        result.update(extra, finished=time.time())
        json.dump(result, open(os.path.join(d, name), "w"), indent=1)

    pv = os.path.join(STATE, "privacy")
    if os.path.exists(pv) and open(pv).read().strip() == "1":
        return finish("failed.json", reason="privacy is on: nothing captured")
    if os.path.exists(LOCK) and time.time() - os.path.getmtime(LOCK) < LOCK_STALE_S:
        return finish("failed.json", reason=f"another capture holds {LOCK}: {open(LOCK).read()[:120]}")
    open(LOCK, "w").write(f"scheduled_sweep {tag} pid {os.getpid()}")
    try:
        start = _motors()
        last_tilt = None
        for k, p in enumerate(plan["poses"]):
            open(os.path.join(STATE, "moved_at"), "a").close()
            os.utime(os.path.join(STATE, "moved_at"), None)                 # Follow stays paused during the sweep
            os.utime(LOCK, None)
            if p["tilt"] != last_tilt:                                      # same approach side for every row (gear play)
                _goto(max(0, p["pan"] - 150), p["tilt"])
                last_tilt = p["tilt"]
            px, py = _goto(p["pan"], p["tilt"])
            f = f"{k:02d}_{px}_{py}.jpg"
            time.sleep(0.6)                                                 # settle
            _snapshot(os.path.join(d, f))
            result["poses"].append({"file": f, "pan": px, "tilt": py, "t": time.time(), "ref": p.get("ref")})
        try:
            _goto(*start)
        except Exception as e:  # noqa: BLE001
            result["return_error"] = str(e)
        finish("done.json")
    except Exception as e:  # noqa: BLE001
        finish("failed.json", reason=f"{type(e).__name__}: {e}")
    finally:
        try:
            os.remove(LOCK)
        except OSError:
            pass


def check(now=None):
    """[(tag, problem)] for alert.py: a sweep that failed, or has no result GRACE_S after its time."""
    now = time.time() if now is None else now
    out = []
    try:
        tags = sorted(os.listdir(ROOT))
    except OSError:
        return out
    for tag in tags:
        d = os.path.join(ROOT, tag)
        try:
            plan = json.load(open(os.path.join(d, "plan.json")))
        except (OSError, ValueError):
            continue
        if os.path.exists(os.path.join(d, "failed.json")):
            try:
                why = json.load(open(os.path.join(d, "failed.json"))).get("reason", "")
            except (OSError, ValueError):
                why = ""
            out.append((tag, f"failed: {why}"))
        elif not os.path.exists(os.path.join(d, "done.json")) and now > plan["at"] + GRACE_S and now < plan["at"] + 7 * 86400:
            out.append((tag, f"no result {int((now - plan['at']) / 60)} min after {plan.get('when')}"))
    return out


if __name__ == "__main__":
    sys.path.insert(0, "/opt/petcam")
    a = sys.argv[1:]
    if a and a[0] == "schedule":
        schedule(a[1], a[2], a[3])
    elif a and a[0] == "run":
        run(a[1])
    elif a and a[0] == "check":
        print(json.dumps(check()))
    else:
        sys.exit(__doc__)
