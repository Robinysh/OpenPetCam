#!/usr/bin/env python3
"""Self-calibrating Follow: learn THIS camera and THIS pet, build their simulator, train a policy for them, install it.

    python3 sim/selfcal.py [--sysid] [--owner-present] [--deploy]

  1 export     the camera's own logs from its server (obs, track, events, Follow actions; numbers only)
  2 fit        sensing, cycle timing and the pet's room map from those logs      (sim/fit_from_logs.py)
  3 sysid      optional: motor delay/speed/queueing, frame size, backlash        (sim/sysid_camera.py; pet not resting)
  4 calibrate  the unobservable parameters, through the observation model        (sim/calibrate.py)
  5 gate 1     the real controller in the sim must reproduce the real metrics (held-out seeds)  else STOP
  6 train      direct policy search on randomised rooms                          (sim/train_policy.py)
  7 gate 2     the policy must beat today's controller on held-out rooms, without more moves   else STOP
  8 install    models/follow_policy.json; with --deploy, ./deploy.sh. The watcher then A/Bs it in 2-hour blocks
               (gate 3, live): python3 sim/ab_report.py after a few days decides whether it stays.
Each stage writes its evidence into eval/rl/ (git-ignored); the fitted parameters go to sim/params.json.
"""
import json
import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "eval", "rl")
sys.path.insert(0, os.path.join(ROOT, "tools"))


def run(*cmd, timeout=7200):
    print("+", " ".join(cmd), flush=True)
    r = subprocess.run([sys.executable, *cmd], cwd=ROOT, timeout=timeout)
    return r.returncode


EXPORT = r'''
import sqlite3, json, time
c = sqlite3.connect("/var/lib/petcam/petcam.db")
t0 = time.time() - 7 * 86400
cols = [r[1] for r in c.execute("pragma table_info(track)")]
json.dump({"obs": c.execute("select ts,score,person,box,motion,present,moving from obs where ts>? order by ts", (t0,)).fetchall(),
           "track": c.execute("select ts,pan,tilt,act,weak from track where ts>? order by ts", (t0,)).fetchall(),
           "events": c.execute("select ts,kind,detail from events where ts>? and kind in ('search','left','appeared','active')", (t0,)).fetchall()},
          open("/tmp/selfcal_export.json", "w"))
'''


def export():
    from siteconf import KEY, SSH
    os.makedirs(OUT, exist_ok=True)
    host = SSH[-1]
    subprocess.run(SSH + ["sudo python3 -"], input=EXPORT, text=True, check=True, timeout=300)
    subprocess.run(SSH + ["sudo journalctl -u petcam-watch --since '-7 days' --no-pager -o short-unix | grep 'follow: ' "
                          "> /tmp/selfcal_follow.log; sudo chmod 644 /tmp/selfcal_export.json /tmp/selfcal_follow.log"],
                   check=True, timeout=300)
    for f in ("selfcal_export.json", "selfcal_follow.log"):
        subprocess.run(["scp", "-q", "-i", KEY, f"{host}:/tmp/{f}", os.path.join(OUT, f)], check=True, timeout=300)
    return os.path.join(OUT, "selfcal_export.json"), os.path.join(OUT, "selfcal_follow.log")


def gate1_heldout():
    sys.path.insert(0, HERE)
    import gate1
    from room import Room, load_params
    P = load_params()
    T = gate1.load_tracker()
    rows = []
    for s in range(100, 108):
        r = Room(seed=s, free=P["free"])
        obs, acts = gate1.run_watcher(r, T, 4)
        rows.append((gate1.metrics(obs, acts), gate1.truth(r)))
    rep = gate1.compare(rows, P["real_metrics"])
    inside = sum(v["sim_min"] <= (v["real"] or 0) <= v["sim_max"] for v in rep.values())
    json.dump(rep, open(os.path.join(OUT, "gate1.json"), "w"), indent=1)
    print(f"gate 1: {inside}/{len(rep)} real metrics inside the simulator's range on held-out seeds")
    return inside >= len(rep) - 1          # at most one metric may sit just outside (reported, not hidden)


def main(argv):
    exp, follow = export()
    if run("sim/fit_from_logs.py", exp, follow):
        sys.exit("fit failed")
    if "--sysid" in argv:
        if run("sim/sysid_camera.py", *(["--owner-present"] if "--owner-present" in argv else [])):
            print("sysid skipped or failed; keeping the previous motor/geometry measurements")
    if run("sim/calibrate.py", "120"):
        sys.exit("calibration failed")
    if not gate1_heldout():
        sys.exit("gate 1 failed: the simulator does not reproduce this camera's logs; nothing was trained")
    code = run("sim/train_policy.py", "200")
    if code != 0 or not os.path.exists(os.path.join(HERE, "policy.json")):
        sys.exit("gate 2 failed: the learned policy did not beat today's controller on held-out rooms; nothing installed")
    shutil.copy(os.path.join(HERE, "policy.json"), os.path.join(ROOT, "models", "follow_policy.json"))
    print("installed models/follow_policy.json")
    if "--deploy" in argv:
        subprocess.run(["./deploy.sh"], cwd=ROOT, check=True, timeout=1800)
        print("deployed; the watcher alternates it with today's constants in 2-hour blocks. Judge it with sim/ab_report.py")


if __name__ == "__main__":
    main(sys.argv[1:])
