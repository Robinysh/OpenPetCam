"""Pull what the server already records into eval/world/: the track (pose), obs (detections) and minute (sound level,
snapshot) tables, and the minute snapshots. Read-only on the server; nothing moves, nothing plays."""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))


def _ssh():
    import siteconf
    return siteconf.SSH


def sql_json(query, timeout=120):
    out = subprocess.run(_ssh() + [f"sudo sqlite3 -json /var/lib/petcam/petcam.db \"{query}\""], capture_output=True,
                         text=True, timeout=timeout).stdout
    return json.loads(out or "[]")


def refresh(world):
    tables = {"track.json": "select ts,pan,tilt,moves from track",
              "obs.json": "select ts,score,box,present,moving from obs where box is not null and box<>''",
              "minute.json": "select * from minute"}
    for name, q in tables.items():
        rows = sql_json(q)
        json.dump(rows, open(os.path.join(world, name), "w"))
        print(f"{name}: {len(rows)} rows", flush=True)
    have = set(os.listdir(os.path.join(world, "snaps"))) if os.path.isdir(os.path.join(world, "snaps")) else set()
    days = subprocess.run(_ssh() + ["ls /var/lib/petcam/snaps"], capture_output=True, text=True, timeout=60).stdout.split()
    new = [d for d in days if d not in have] + sorted(d for d in days if d in have)[-1:]       # new days + today again
    if new:
        tar = subprocess.run(_ssh() + ["sudo tar c -C /var/lib/petcam/snaps " + " ".join(new)], capture_output=True, timeout=600)
        os.makedirs(os.path.join(world, "snaps"), exist_ok=True)
        subprocess.run(["tar", "x", "-C", os.path.join(world, "snaps")], input=tar.stdout, check=True)
        print(f"snaps: {', '.join(new)}", flush=True)
