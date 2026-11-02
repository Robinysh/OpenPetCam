#!/usr/bin/env python3
"""Pull "Something wrong?" reports from the server and turn each into a debuggable case file.

  triage.py list            reports on the server (newest first), marking ones already pulled
  triage.py pull [ID ...]   pull the given reports, or every one not pulled yet, into eval/reports/<ID>/
  triage.py errors          the page's uncaught JavaScript errors (last 50)

Each pulled report gets:
  report.json   what the app showed, the watcher's live state, last 15 min of detections, last 3 h of events
  frame.jpg     the camera picture at the moment of the report
  clip.mp4      the 90 s before it (+10 s after), when the recording had it
  journal.txt   watcher / app / video relay / alert logs from 15 min before to 5 min after
  summary.md    the above, condensed and in the install's local time: start here

eval/ is git-ignored: reports contain pictures of your home. Never paste them into a public issue.
"""
import datetime as dt
import json
import os
import subprocess
import sys
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from siteconf import ROOT, SITE, SSH  # noqa: E402

OUT = os.path.join(ROOT, "eval", "reports")
REMOTE = "/var/lib/petcam/reports"
TZ = ZoneInfo(SITE.get("TZ", "UTC"))


def ssh(cmd, check=True):
    return subprocess.run(SSH + [cmd], check=check, capture_output=True, text=True, timeout=120).stdout


def local(ts):
    return dt.datetime.fromtimestamp(float(ts), TZ).strftime("%H:%M:%S")


def remote_ids():
    out = ssh(f"sudo ls -1 {REMOTE} 2>/dev/null || true")
    return sorted((n for n in out.split() if n[:8].isdigit()), reverse=True)


def cmd_list():
    pulled = set(os.listdir(OUT)) if os.path.isdir(OUT) else set()
    ids = remote_ids()
    if not ids:
        print("no reports on the server")
        return
    notes = ssh("sudo python3 -c \"import json,sys\nfor i in sys.argv[1:]:\n"
                f" r=json.load(open('{REMOTE}/'+i+'/report.json')); print(i, '|', r.get('by'), '|', "
                "(r.get('client') or {}).get('shown_status'), '|', r.get('note','').replace(chr(10),' ')[:100])\" "
                + " ".join(ids[:30]), check=False)
    for line in notes.splitlines():
        rid = line.split(" ", 1)[0]
        print(("  pulled " if rid in pulled else "  NEW    ") + line)


def runs(obs):
    """Compress per-3s observations into runs of the same (present, detected, moving) state."""
    out, cur = [], None
    for o in obs:
        key = (o["present"], "seen" if (o.get("score") or 0) >= 0.3 else "not seen", "moving" if o["moving"] else "still")
        if cur and cur[0] == key:
            cur[2] = o["ts"]
            cur[3] = max(cur[3], o.get("motion") or 0)
        else:
            cur = [key, o["ts"], o["ts"], o.get("motion") or 0]
            out.append(cur)
    return [f"- {local(a)}–{local(b)}  present={k[0]} detector:{k[1]} {k[2]} (max motion {m:.1f})" for k, a, b, m in out]


def summarize(d):
    r = json.load(open(os.path.join(d, "report.json")))
    c = r.get("client") or {}
    lv = r.get("live") or {}
    lines = [f"# Report {r['id']}", "",
             f"- When: {r.get('local_time')} (all times below: {TZ.key}; the server's system clock is UTC)",
             f"- By: {r.get('by')}",
             f"- Note: {r.get('note') or '(none)'}",
             f"- App showed: **{c.get('shown_status')}** {c.get('shown_health') or ''}",
             f"- Watcher state: present={lv.get('present')} status={lv.get('status')} "
             f"last_seen={local(lv['last_seen']) if lv.get('last_seen') else None} "
             f"rest_since={local(lv['rest_since']) if lv.get('rest_since') else None} "
             f"moving_since={local(lv['moving_since']) if lv.get('moving_since') else None} score={lv.get('score')}",
             f"- Phone: video={c.get('video')} webrtc={c.get('webrtc')} tab={c.get('tab')}",
             f"- Files: frame.jpg{', clip.mp4' if os.path.exists(os.path.join(d, 'clip.mp4')) else ' (no clip)'}, journal.txt, report.json",
             "", "## Presence, last 15 min (runs of identical state)", ""]
    lines += runs(r.get("obs_15min") or [])
    lines += ["", "## Events, last 3 h", ""]
    for e in r.get("events_3h") or []:
        lines.append(f"- {local(e['ts'])} {e['kind']} {(e.get('detail') or '')[:120]}{' label=' + e['label'] if e.get('label') else ''}")
    if c.get("recent_errors"):
        lines += ["", "## Page errors before the report", ""] + [f"- {x}" for x in c["recent_errors"]]
    with open(os.path.join(d, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


def cmd_pull(ids):
    os.makedirs(OUT, exist_ok=True)
    ids = ids or [i for i in remote_ids() if not os.path.exists(os.path.join(OUT, i, "summary.md"))]
    for rid in ids:
        d = os.path.join(OUT, rid)
        os.makedirs(d, exist_ok=True)
        tar = subprocess.run(SSH + [f"sudo tar -C {REMOTE} -cf - {rid}"], capture_output=True, check=True, timeout=300).stdout
        subprocess.run(["tar", "-C", OUT, "-xf", "-"], input=tar, check=True)
        r = json.load(open(os.path.join(d, "report.json")))
        since = dt.datetime.fromtimestamp(r["ts"] - 900, dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        until = dt.datetime.fromtimestamp(r["ts"] + 300, dt.timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        j = ssh(f"sudo journalctl -u petcam-watch -u petcam-app -u mediamtx -u petcam-alert --since '{since} UTC' "
                f"--until '{until} UTC' --no-pager -o short-iso", check=False)
        with open(os.path.join(d, "journal.txt"), "w") as f:
            f.write(j)
        summarize(d)
        print(f"pulled {rid} -> {os.path.relpath(d, ROOT)}/summary.md")
    if not ids:
        print("nothing new")


def cmd_errors():
    out = ssh(f"sudo tail -n 50 {REMOTE}/client-errors.jsonl 2>/dev/null || true")
    for line in out.splitlines():
        e = json.loads(line)
        print(local(e["ts"]), e.get("where"), e.get("msg"), e.get("src"), e.get("line"), (e.get("ua") or "")[:60])
    if not out.strip():
        print("no page errors logged")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "list"
    {"list": lambda: cmd_list(), "pull": lambda: cmd_pull(sys.argv[2:]), "errors": lambda: cmd_errors()}[cmd]()
