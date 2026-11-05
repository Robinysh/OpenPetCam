#!/usr/bin/env python3
"""Export the watcher's records for the simulator's fits and held-out checks (read-only on the server).

    python3 tools/export_logs.py [days] [out_dir]   ->  out_dir/rl_export.json (obs, track, events) + rl_follow.log
obs: ts, score, person, box, motion, present, moving.  track: ts, pan, tilt, act, weak.  events: ts, kind, detail.
rl_follow.log: the watcher's journal lines with 'follow:' (journalctl -o short-unix), as sim/fit_from_logs.py reads.
Everything lands in eval/ (git-ignored): it is the inside of a home."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import siteconf  # noqa: E402


def ssh(cmd, timeout=600):
    r = subprocess.run(siteconf.SSH + [cmd], capture_output=True, text=True, timeout=timeout)
    if r.returncode:
        raise SystemExit(f"server command failed: {r.stderr.strip()[:300]}")
    return r.stdout


def query(sql):
    out = ssh(f"sudo sqlite3 -json /var/lib/petcam/petcam.db \"{sql}\"")
    return json.loads(out or "[]")


def main(days=5.0, out=os.path.join(HERE, "..", "eval", "rl")):
    os.makedirs(out, exist_ok=True)
    since = f"strftime('%s','now') - {float(days) * 86400:.0f}"
    obs = query(f"select ts, score, person, box, motion, present, moving from obs where ts > {since} order by ts")
    track = query(f"select ts, pan, tilt, act, weak from track where ts > {since} order by ts")
    events = query(f"select ts, kind, detail from events where ts > {since} order by ts")
    d = {"obs": [[r["ts"], r["score"], r["person"], r["box"], r["motion"], r["present"], r["moving"]] for r in obs],
         "track": [[r["ts"], r["pan"], r["tilt"], r["act"], r["weak"]] for r in track],
         "events": [[r["ts"], r["kind"], r["detail"]] for r in events]}
    json.dump(d, open(os.path.join(out, "rl_export.json"), "w"))
    log = ssh(f"sudo journalctl -u petcam-watch -o short-unix --since '-{int(float(days) * 24)}h' --no-pager | grep 'follow:'", timeout=900)
    open(os.path.join(out, "rl_follow.log"), "w").write(log)
    print(f"{len(obs)} obs, {len(track)} track, {len(events)} events, {log.count(chr(10))} follow lines -> {out}")


if __name__ == "__main__":
    a = sys.argv[1:]
    main(*(a[:1] or [5]), *(a[1:2]))
