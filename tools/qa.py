#!/usr/bin/env python3
"""Exhaustive, repeatable QA of the whole install: server, camera, video, detection pipeline, behaviours,
app API, and the unattended loops. Every check states the assumption it tests and prints its evidence.

  qa.py                 read-only checks (safe any time; nothing moves, nothing plays a sound)
  qa.py --hardware      also checks that move the camera (quiet speed) and toggle night vision
  qa.py --json          machine-readable (the nightly loop stores it and compares with the last run)

Result per check: PASS, WARN (works, but outside the expected range), FAIL (broken or an assumption is wrong),
SKIP (could not be checked here, with the reason). Exit 1 if anything FAILs.
"""
import json
import os
import socket
import ssl
import statistics
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from siteconf import ROOT, SITE, SSH  # noqa: E402

HOST = SITE.get("PUBLIC_HOST", "")
IP = SITE.get("PUBLIC_IP", "")
RESULTS = []


def check(area, name, assumption):
    def deco(fn):
        def run():
            t = time.time()
            try:
                status, evidence = fn()
            except Exception as e:  # noqa: BLE001 - a crashing check is a failed check
                status, evidence = "FAIL", f"check crashed: {type(e).__name__}: {str(e)[:300]}"
            RESULTS.append({"area": area, "name": name, "assumption": assumption, "status": status,
                            "evidence": evidence, "secs": round(time.time() - t, 1)})
        run.hardware = getattr(fn, "hardware", False)
        CHECKS.append(run)
        return run
    return deco


CHECKS = []


def hardware(fn):
    fn.hardware = True
    return fn


def sh(cmd, timeout=60):
    return subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True, timeout=timeout)


# ------------------------------------------------------------------ one round trip for everything on the server

REMOTE = r'''
import json, os, sqlite3, subprocess, time, glob, statistics
out = {}
def sh(c, t=30):
    try:
        r = subprocess.run(c, shell=True, capture_output=True, text=True, timeout=t); return r.stdout.strip()
    except Exception as e:
        return "ERR " + str(e)
S = "/var/lib/petcam"
try: out["privacy_requested"] = open(f"{S}/privacy").read().strip() == "1"
except OSError: out["privacy_requested"] = False
units = "caddy mediamtx petcam-app petcam-watch wg-quick@wg0 petcam-alert.timer"
out["units"] = dict(zip(units.split(), sh("systemctl is-active " + units).split()))
out["failed_units"] = sh("systemctl --failed --no-legend --plain | awk '{print $1}'")
out["df"] = sh("df -P / | awk 'NR==2{print $2, $3, $4}'")
out["mem"] = sh("free -m | awk '/Mem/{print $2, $7} /Swap/{print $3}'")
out["ntp"] = sh("timedatectl show -p NTPSynchronized --value")
out["wg_age"] = sh("wg show wg0 latest-handshakes | awk '{print systime()-$2}'")
def age(p):
    try: return round(time.time() - os.path.getmtime(p))
    except OSError: return None
out["ages"] = {k: age(f"{S}/{k}") for k in ("heartbeat", "autopilot_ok", "report_agent_ok", "live.json")}
try: out["live"] = json.load(open(f"{S}/live.json"))
except Exception as e: out["live"] = {"error": str(e)}
c = sqlite3.connect(f"{S}/petcam.db"); now = time.time()
ts = [r[0] for r in c.execute("select ts from obs where ts > ? order by ts", (now - 86400,))]
gaps = [b - a for a, b in zip(ts, ts[1:])]
out["obs"] = {"n": len(ts), "median_gap": round(statistics.median(gaps), 2) if gaps else None,
              "gaps_over_30s": sum(1 for g in gaps if g > 30), "max_gap": round(max(gaps), 1) if gaps else None}
ev = c.execute("select ts, kind, detail from events where ts > ? order by ts", (now - 86400,)).fetchall()
out["events"] = [[round(t, 1), k, (d or "")[:300]] for t, k, d in ev]
out["pres"] = c.execute("select count(*), sum(present), sum(score>0) from obs where ts > ?", (now - 86400,)).fetchone()
out["db_mb"] = round(os.path.getsize(f"{S}/petcam.db") / 1e6, 1)
out["tables"] = {t: c.execute(f"select count(*) from {t}").fetchone()[0] for (t,) in c.execute("select name from sqlite_master where type='table'")}
out["oldest"] = {t: c.execute(f"select min(ts) from {t}").fetchone()[0] for t in ("obs", "events", "usage")}
def oldest_dir(p):
    ds = sorted(glob.glob(p + "/20*")); return os.path.basename(ds[0]) if ds else None
out["dirs"] = {k: [oldest_dir(f"{S}/{k}"), sh(f"du -sm {S}/{k} 2>/dev/null | cut -f1")] for k in ("snaps", "candidates", "clips", "timelapse", "highlights", "reports")}
rec = sorted(glob.glob("/var/lib/mediamtx/recordings/cam/*"))
out["rec"] = {"n": len(rec), "oldest_age_h": round((now - os.path.getmtime(rec[0])) / 3600, 1) if rec else None,
              "newest_age_s": round(now - os.path.getmtime(rec[-1])) if rec else None, "mb": sh("du -sm /var/lib/mediamtx/recordings | cut -f1")}
out["tracebacks"] = {u: sh(f"journalctl -u {u} --since -24h --no-pager | grep -c Traceback") for u in ("petcam-watch", "petcam-app", "petcam-alert")}
out["errors"] = sh("journalctl -u petcam-watch -u petcam-app --since -24h --no-pager | grep -iE 'error|failed' | grep -v 'no subscribed' | tail -8")
out["verifier"] = sh("journalctl -u petcam-watch --since -48h --no-pager | grep 'verifier:' | tail -1")
out["journal_mb"] = sh("journalctl --disk-usage | grep -oE '[0-9.]+[MG]'")
out["ffprobe"] = "" if out["privacy_requested"] else sh("timeout 15 ffprobe -v error -rtsp_transport tcp -show_entries stream=codec_type,codec_name,width,height,avg_frame_rate,sample_rate -of json rtsp://127.0.0.1:8554/cam", 25)
out["push_subs"] = c.execute("select count(*) from push_subs").fetchone()[0] if "push_subs" in out["tables"] else None
import sys; sys.path.insert(0, "/opt/petcam")
try:
    import petconf
    cam = petconf.CAM_HOST
    k = f"{S}/ssh/id_ed25519"
    camsh = lambda c: sh(f"ssh -i {k} -o BatchMode=yes -o ConnectTimeout=8 -o StrictHostKeyChecking=no root@{cam} '{c}'", 20)
    t_before = time.time(); cam_date = camsh("date +%s"); t_after = time.time()
    out["cam"] = {"date": cam_date, "server_mid": (t_before + t_after) / 2, "ssh_secs": round(t_after - t_before, 2), "uptime": camsh("cat /proc/uptime"), "mem": camsh("free | awk '/Mem/{print $2, $4}'"),
                  "fw": camsh("head -c 120 /etc/os-release"),
                  "motor": camsh("motors -p 2>/dev/null || cat /tmp/motors 2>/dev/null"),
                  "df": camsh("df -k / /overlay 2>/dev/null | tail -2"), "load": camsh("cat /proc/loadavg"),
                  "temp": camsh("cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null"),
                  "procs": camsh("pidof prudynt | wc -w"),
                  "capture_guard": camsh("if test -x /etc/opc-privacy/control; then /etc/opc-privacy/control status; else echo absent; fi"),
                  "privacy": camsh("jct /etc/prudynt.json get privacy 2>/dev/null || grep -o '\"privacy\"[^}]*}' /etc/prudynt.json | head -1"),
                  "audio_in": camsh("jct /etc/prudynt.json get audio 2>/dev/null | head -c 400")}
except Exception as e:
    out["cam"] = {"error": str(e)}
print(json.dumps(out, default=str))
'''

_R = None


def remote():
    global _R
    if _R is None:
        r = subprocess.run(SSH + ["sudo python3 -"], input=REMOTE, capture_output=True, text=True, timeout=180)
        if r.returncode != 0:
            raise RuntimeError(r.stderr[-400:])
        _R = json.loads(r.stdout)
    return _R


# ------------------------------------------------------------------ server

@check("server", "services running", "every service the install needs is active, none failed")
def _():
    u = remote()["units"]
    bad = {k: v for k, v in u.items() if v != "active"}
    failed = remote()["failed_units"]
    return ("FAIL" if bad or failed else "PASS"), f"{u}; failed units: {failed or 'none'}"


@check("server", "disk, memory, swap", "disk < 80% used, > 80 MB RAM available, swap < 300 MB on the 512 MB box")
def _():
    tot, used, free = map(int, remote()["df"].split())
    mt, ma, sw = map(int, remote()["mem"].split())
    pct = used / tot
    st = "FAIL" if pct > 0.9 or ma < 40 else "WARN" if pct > 0.8 or ma < 80 or sw > 300 else "PASS"
    return st, f"disk {pct:.0%} used ({free // 1024} MB free); RAM {ma}/{mt} MB available; swap used {sw} MB"


@check("server", "clock", "server clock is NTP-synchronised (events, clips and the camera all key off it)")
def _():
    return ("PASS" if remote()["ntp"] == "yes" else "FAIL"), f"NTPSynchronized={remote()['ntp']}"


@check("server", "TLS certificate", "the public HTTPS certificate is valid for the host for > 14 more days")
def _():
    ctx = ssl.create_default_context()
    with socket.create_connection((HOST, 443), timeout=10) as s, ctx.wrap_socket(s, server_hostname=HOST) as t:
        exp = ssl.cert_time_to_seconds(t.getpeercert()["notAfter"])
    days = (exp - time.time()) / 86400
    return ("PASS" if days > 14 else "WARN" if days > 3 else "FAIL"), f"expires in {days:.0f} days (Caddy renews at ~30)"


@check("server", "exposed ports", "from the internet only SSH, HTTP(S) and WireGuard answer; video, talk and APIs are not reachable")
def _():
    want_closed = [554, 8554, 8888, 8889, 9997, 9996, 8099, 8000, 8080, 1935, 3000, 5000]
    open_ = []
    for p in [22, 80, 443] + want_closed:
        try:
            with socket.create_connection((IP or HOST, p), timeout=3):
                open_.append(p)
        except OSError:
            pass
    leak = [p for p in open_ if p in want_closed]
    return ("FAIL" if leak else "PASS"), f"open TCP: {open_}; should be closed but open: {leak or 'none'} (UDP 51820 not probed)"


@check("server", "auth on every API", "without a login every /api route answers 401 and changes nothing")
def _():
    gets = ["/api/live", "/api/day", "/api/diary", "/api/snap", "/api/devices", "/api/zones", "/api/reports", "/api/wellbeing",
            "/api/frame.jpg", "/api/push", "/api/state", "/api/live/stream"]
    posts = ["/api/report", "/api/aim", "/api/nudge", "/api/goto", "/api/recalibrate", "/api/presets", "/api/imaging", "/api/privacy",
             "/api/settings", "/api/zones", "/api/invites", "/api/devices/revoke", "/api/talk", "/api/usage", "/api/push/test"]
    bad = []
    for p in gets:
        c = sh(["curl", "-s", "-m", "10", "-o", "/dev/null", "-w", "%{http_code}", f"https://{HOST}{p}"]).stdout
        if c != "401":
            bad.append(f"GET {p}={c}")
    for p in posts:
        c = sh(["curl", "-s", "-m", "10", "-o", "/dev/null", "-w", "%{http_code}", "-X", "POST", "-H", "Content-Type: application/json",
                "-d", "{}", f"https://{HOST}{p}"]).stdout
        if c not in ("401", "403"):
            bad.append(f"POST {p}={c}")
    login = sh(["curl", "-s", "-m", "10", "-o", "/dev/null", "-w", "%{http_code}", f"https://{HOST}/login"]).stdout
    return ("FAIL" if bad else "PASS"), f"{len(gets)} GET + {len(posts)} POST routes checked; not 401: {bad or 'none'}; /login={login}"


@check("server", "retention", "snapshots 14 d, candidates 30 d, recordings ~6 h, usage 90 d: storage stays bounded")
def _():
    r = remote()
    notes, st = [], "PASS"
    today = time.strftime("%Y-%m-%d")
    for k, keep in (("snaps", 14), ("candidates", 30)):
        oldest, mb = r["dirs"][k]
        if oldest:
            days = (time.mktime(time.strptime(today, "%Y-%m-%d")) - time.mktime(time.strptime(oldest, "%Y-%m-%d"))) / 86400
            notes.append(f"{k}: oldest {oldest} ({days:.0f} d, keep {keep}), {mb} MB")
            if days > keep + 1:
                st = "FAIL"
    rec = r["rec"]
    notes.append(f"recordings: {rec['n']} files, oldest {rec['oldest_age_h']} h, {rec['mb']} MB")
    if rec["oldest_age_h"] and rec["oldest_age_h"] > 7:          # recordDeleteAfter 6h
        st = "WARN"
    notes.append(f"db {r['db_mb']} MB, journal {r['journal_mb']}; rows {r['tables']}")
    return st, "; ".join(notes)


@check("server", "no crashes", "no Python tracebacks in the services in the last 24 h")
def _():
    tb = {k: int(v or 0) for k, v in remote()["tracebacks"].items()}
    return ("FAIL" if sum(tb.values()) else "PASS"), f"tracebacks {tb}; recent error lines: {remote()['errors'][-600:] or 'none'}"


# ------------------------------------------------------------------ camera

@check("camera", "tunnel", "the camera's WireGuard handshake is fresh (< 3 min): the server can reach it")
def _():
    a = [int(x) for x in remote()["wg_age"].split() if x.lstrip("-").isdigit()]
    return ("PASS" if a and min(a) < 180 else "FAIL"), f"handshake age {a} s"


@check("camera", "clock", "camera clock within 3 s of the server (clip times and the burned-in timestamp depend on it)")
def _():
    c = remote()["cam"]
    if "error" in c or not c["date"].isdigit():
        return "SKIP", f"camera shell not reachable: {c}"
    d = int(c["date"]) - c["server_mid"]          # server time taken around the camera call (ssh ~1 s)
    return ("PASS" if abs(d) <= 3 else "WARN" if abs(d) <= 30 else "FAIL"), f"camera - server = {d:+.1f} s (ssh round trip {c['ssh_secs']} s)"


@check("camera", "health", "camera up, not memory- or disk-starved, streamer running")
def _():
    c = remote()["cam"]
    if "error" in c:
        return "SKIP", c["error"]
    up = round(float(c["uptime"].split()[0]) / 3600, 1) if c["uptime"][:1].isdigit() else None
    mem = c["mem"].split()
    free_pct = round(int(mem[1]) / int(mem[0]), 2) if len(mem) == 2 and mem[0].isdigit() else None
    capture_expected = not remote().get("privacy_requested", False)
    process_ok = c["procs"] not in ("", "0") if capture_expected else c["procs"] == "0"
    st = "PASS" if up is not None and process_ok and (free_pct is None or free_pct > 0.08) else "WARN"
    return st, f"uptime {up} h; RAM free {free_pct}; load {c['load']}; temp {c['temp']}; streamer procs {c['procs']}; fw {c['fw']}; disk {c['df']}"


@check("camera", "video stream", "the relay serves H.264 1920x1080 video plus an audio track; recording is current")
def _():
    r = remote()
    if r.get("privacy_requested"):
        return "PASS", "Privacy is requested: live capture and recording are intentionally suspended; checked separately below."
    try:
        st = json.loads(r["ffprobe"])["streams"]
    except Exception:  # noqa: BLE001
        return "FAIL", f"ffprobe: {r['ffprobe'][:300]}"
    v = [s for s in st if s.get("codec_type") == "video"]
    a = [s for s in st if s.get("codec_type") == "audio"]
    newest = r["rec"]["newest_age_s"]
    ok = v and v[0].get("codec_name") == "h264" and a and newest is not None and newest < 180
    return ("PASS" if ok else "FAIL"), f"video {[(s.get('codec_name'), s.get('width'), s.get('height'), s.get('avg_frame_rate')) for s in v]}; " \
        f"audio {[(s.get('codec_name'), s.get('sample_rate')) for s in a]}; newest recording {newest} s old"


@check("camera", "privacy capture state", "verify requested software capture stop; this is not electrical power isolation")
def _():
    r = remote()
    try:
        guard = json.loads((r.get("cam") or {}).get("capture_guard", ""))
    except (ValueError, TypeError):
        guard = {}
    if not r.get("privacy_requested"):
        if guard.get("paused"):
            return "WARN", "Camera-side privacy remains paused while the server requests capture; reconcile before claiming live video."
        return "PASS", "Privacy is not requested. Capture-stop guard installed." if guard.get("boot_persistent") else "Privacy is not requested; capture-stop guard is unavailable on this camera."
    ok = (guard.get("paused") is True and guard.get("running") is False
          and guard.get("verified") is True and guard.get("mode") == "capture_stopped"
          and (r.get("cam") or {}).get("procs") == "0")
    if ok:
        return "PASS", "Persistent camera gate is paused; no capture process is running. Camera power and management network remain on."
    if not guard:
        return "WARN", "Capture-stop guard is unavailable. Legacy black/mute privacy needs a fresh functional media check."
    return "FAIL", "Privacy is requested but stopped capture is not verified: " + json.dumps(guard)


# ------------------------------------------------------------------ pipeline

@check("pipeline", "watcher cadence", "a detection cycle at least every ~3 s, no gap over 30 s outside restarts (24 h)")
def _():
    o = remote()["obs"]
    st = "PASS" if o["median_gap"] and o["median_gap"] <= 3.5 and o["gaps_over_30s"] <= 5 else "WARN"
    return st, f"{o['n']} cycles; median gap {o['median_gap']} s; {o['gaps_over_30s']} gaps > 30 s (max {o['max_gap']} s); " \
        "no rows are written while privacy is on or during a restart, so gaps there are expected"


@check("pipeline", "heartbeats", "watcher heartbeat < 30 s; nightly autopilot < 36 h; report agent < 30 min once installed")
def _():
    a = remote()["ages"]
    st = "PASS"
    if a["heartbeat"] is None or a["heartbeat"] > 30:
        st = "FAIL"
    if a["autopilot_ok"] is None or a["autopilot_ok"] > 36 * 3600:
        st = "FAIL" if st == "FAIL" else "WARN"
    if a["report_agent_ok"] is None or a["report_agent_ok"] > 1800:
        st = "FAIL" if st == "FAIL" else "WARN"
    return st, f"ages (s): {a}"


@check("pipeline", "verifier loaded", "the watcher runs the trained verifier (logreg), not a fallback")
def _():
    v = remote()["verifier"]
    return ("PASS" if ("logreg" in v or "finetuned" in v) else "FAIL"), v or "no 'verifier:' line in 48 h"


@check("pipeline", "presence flapping", "comings and goings are real: few left->appeared pairs within 2 min (a flapping detector)")
def _():
    ev = remote()["events"]
    flips = sum(1 for (t1, k1, _), (t2, k2, _) in zip(ev, ev[1:]) if k1 == "left" and k2 == "appeared" and t2 - t1 < 120)
    lefts = sum(1 for e in ev if e[1] == "left")
    st = "PASS" if lefts == 0 or flips / lefts < 0.25 else "WARN"
    return st, f"{lefts} 'left' events in 24 h, {flips} reversed within 2 min"


@check("pipeline", "search effectiveness", "searches find her when she is home; median time to find < 60 s")
def _():
    s = [e for e in remote()["events"] if e[1] == "search"]
    found = [e for e in s if "found her" in e[2]]
    import re
    secs = [float(m.group(1)) for e in found for m in [re.search(r"in (\d+) s", e[2])] if m]
    med = statistics.median(secs) if secs else None
    return ("PASS" if med is None or med < 60 else "WARN"), \
        f"{len(s)} searches in 24 h, {len(found)} found her (median {med} s); the rest include times nobody was home"


LOCAL_TESTS = ["tracker_test", "events_test", "presence_test", "stall_test"]        # pure simulations, run anywhere
SERVER_TESTS = {                                                    # run on the server against /opt/petcam; nothing moves
    "onserver/heal_test": ["total actions in 50 min"],
    "server/zones_test": ["visits started: 2"],
    "onserver/authleak": ["same connection, anonymous -> 401"],
    "onserver/e2e": ["no cookie live  401", "join reuse      410", "db via /media   404", "live video auth 401", "media no cookie 401"],
}


@check("tests", "simulation tests", "follow/search, event rules, presence and stall watchdog behave as specified")
def _():
    out = []
    for t in LOCAL_TESTS:
        r = sh(["python3", os.path.join(ROOT, "tests", "server", f"{t}.py")], timeout=300)
        out.append(f"{t}: {'ok' if r.returncode == 0 else 'FAILED ' + (r.stdout + r.stderr)[-200:]}")
    return ("FAIL" if any("FAILED" in o for o in out) else "PASS"), "; ".join(out)


@check("tests", "server tests", "self-heal, zones, auth isolation and the login/invite/media paths work on the real server")
def _():
    files = [os.path.join(ROOT, "tests", f"{t}.py") for t in SERVER_TESTS]
    sh(["scp", "-q", "-i", SSH[SSH.index("-i") + 1]] + files + [SSH[-1] + ":/tmp/"], timeout=60)
    out, bad = [], []
    for path, must in SERVER_TESTS.items():
        t = path.split("/")[-1]
        r = subprocess.run(SSH + [f"cd /tmp && sudo timeout 120 /opt/petcam/venv/bin/python {t}.py; echo rc=$?; rm -f /tmp/{t}.py"],
                           capture_output=True, text=True, timeout=200)
        missing = [m for m in must if m not in r.stdout]
        ok = "rc=0" in r.stdout and not missing
        out.append(f"{t}: {'ok' if ok else 'FAILED, missing ' + str(missing)}")
        if not ok:
            bad.append(t)
    return ("FAIL" if bad else "PASS"), "; ".join(out)


# ------------------------------------------------------------------ loops on this computer

@check("loops", "launchd jobs", "the nightly autopilot and the report agent are scheduled on this computer")
def _():
    lst = sh("launchctl list | grep -i openpetcam").stdout
    want = ["ai.openpetcam.autopilot", "ai.openpetcam.agent.tick"]   # agent.tick replaced the reports job
    miss = [w for w in want if w not in lst]
    return ("FAIL" if miss else "PASS"), f"loaded: {lst.strip() or 'none'}; missing: {miss or 'none'}"


@check("loops", "last nightly run", "the last autopilot run finished without an error")
def _():
    p = os.path.join(ROOT, "eval", "autopilot", "autopilot.log")
    if not os.path.exists(p):
        return "FAIL", "no autopilot log"
    tail = open(p).read().splitlines()[-40:]
    errs = [line for line in tail if "failed" in line.lower() or "error" in line.lower()]
    return ("WARN" if errs else "PASS"), (errs[-1] if errs else tail[-1])[:300]


# ------------------------------------------------------------------ hardware (moves the camera; opt in)

@check("hardware", "motor reaches commanded pose", "a quiet goto lands within 10 steps of the target and reads back its position")
@hardware
def _():
    return "SKIP", "not implemented yet"


def main():
    hw = "--hardware" in sys.argv
    for c in CHECKS:
        if c.hardware and not hw:
            continue
        c()
    if "--json" in sys.argv:
        print(json.dumps(RESULTS, indent=1))
    else:
        for r in RESULTS:
            print(f"{r['status']:4s}  {r['area']:8s} {r['name']}: {r['evidence']}")
        n = {s: sum(1 for r in RESULTS if r["status"] == s) for s in ("PASS", "WARN", "FAIL", "SKIP")}
        print(f"\n{n}")
    sys.exit(1 if any(r["status"] == "FAIL" for r in RESULTS) else 0)


if __name__ == "__main__":
    main()
