#!/usr/bin/env python3
"""Learn this camera's behaviour (system identification) and write it into sim/params.json, so the Follow simulator
is built from THIS camera rather than the reference one.

    python3 sim/sysid_camera.py [--owner-present] [--dry-run]

Measures, over the server's existing SSH channel to the camera (Thingino `motors`, `prudyntctl snapshot`):
  - pan/tilt range (motors -j / the app's limits), time from command to first motion, speed at the quiet (automatic)
    and joystick speeds, and whether a new command replaces or queues behind the move in progress;
  - how many motor steps one frame width/height is: a snapshot, a 200-step pan, another snapshot, and the image shift
    from phase correlation (steps per frame = 200 / shift as a fraction of the frame). Same for tilt.
Safety: refuses while the pet is resting in view (the owner's rule: no automatic motor moves near a resting pet)
unless --owner-present; pauses Follow first (the 'camera moved' mark); every move is undone; restores the motor's
stored speed.
"""
import io
import json
import os
import re
import subprocess
import sys
import time

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
from siteconf import SSH  # noqa: E402

CAM = ("sudo ssh -T -i /var/lib/petcam/ssh/id_ed25519 -o BatchMode=yes "
       "-o UserKnownHostsFile=/var/lib/petcam/ssh/known_hosts root@10.77.0.2")


def server(cmd, timeout=60, input_=None, binary=False):
    r = subprocess.run(SSH + [cmd], input=input_, capture_output=True, timeout=timeout, text=not binary)
    if r.returncode != 0:
        raise RuntimeError(f"server: {cmd[:60]}: {r.stderr[-200:]}")
    return r.stdout


def camera(script, timeout=90, binary=False):
    return server(f"{CAM} sh", timeout=timeout, input_=script if not binary else script.encode(), binary=binary)


def snapshot():
    raw = server(f"{CAM} prudyntctl snapshot -c 1", timeout=30, binary=True)
    return np.asarray(Image.open(io.BytesIO(raw)).convert("L"), dtype=np.float32)


def shift(a, b):
    """Translation of b relative to a, in pixels (phase correlation, Hann-windowed)."""
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    A, B = np.fft.fft2(a * win), np.fft.fft2(b * win)
    R = A * np.conj(B)
    r = np.fft.ifft2(R / (np.abs(R) + 1e-9)).real
    dy, dx = np.unravel_index(np.argmax(r), r.shape)
    if dy > a.shape[0] // 2:
        dy -= a.shape[0]
    if dx > a.shape[1] // 2:
        dx -= a.shape[1]
    return float(dx), float(dy), float(r.max() / (r.mean() + 1e-9))


IDLE = ("n=0; while [ \"$(motors -j | tr -d '{}\\\"' | tr ',' '\\n' | grep '^status:' | cut -d: -f2)\" != 0 ]"
        " && [ $n -lt 300 ]; do n=$((n+1)); sleep 0.1; done")   # wait until the motor is idle

POLL = r'''
up() { cut -d' ' -f1 /proc/uptime; }
pos() { motors -j 2>/dev/null | tr -d '{}"' | tr ',' ' '; }
poll() { end=$(awk "BEGIN{print $(up)+$1}"); while awk "BEGIN{exit !($(up) < $end)}"; do echo "$(up) $(pos)"; done; }
idle() { n=0; while [ "$(motors -j 2>/dev/null | tr -d '{}"' | tr ',' '\n' | grep '^status:' | cut -d: -f2)" != "0" ] && [ $n -lt 300 ]; do n=$((n+1)); sleep 0.1; done; }
'''


def parse(text):
    seg, data, cmds = None, {}, {}
    for line in text.splitlines():
        if line.startswith("== "):
            seg = line[3:].split(":")[0]
            data[seg], cmds[seg] = [], {}
            continue
        m = re.match(r"(CMD2?) ([\d.]+)", line)
        if m and seg:
            cmds[seg][m.group(1)] = float(m.group(2))
            continue
        m = re.match(r"([\d.]+) status:(\d) xpos:(-?\d+) ypos:(-?\d+)", line)
        if m and seg:
            data[seg].append((float(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))))
    return data, cmds


def hold_follow():
    """Keep Follow paused for the whole test: the 'camera moved' mark pauses it for 3 minutes, the test takes longer
    (2026-11-04: an expired pause let the tracker move the camera mid-measurement and corrupt it)."""
    server("sudo touch /var/lib/petcam/moved_at")


def centre(cam=None, pan=1850, tilt=500):
    """Start from mid-range so no test move can reach an end stop (a run from pan 2700 hit the 3700 limit)."""
    cam = cam or camera
    m = json.loads(cam("motors -j"))
    x, y = int(float(m["xpos"])), int(float(m["ypos"]))
    # tilt commands are inverted on this mount (docs/camera-thingino.md: invert 2, commands only)
    inv = -1 if str(m.get("invert", "")) == "2" else 1
    cam(f"motors -d g -s 250 -x {pan - x} -y {inv * (tilt - y)} >/dev/null; sleep 0.3; " + IDLE + "; sleep 0.5")
    m = json.loads(cam("motors -j"))
    return int(float(m["xpos"])), int(float(m["ypos"]))


def motor_runs(cam=None):
    cam = cam or camera
    script = POLL + r'''
echo "START $(pos)"
echo "== A: +400 at 250"; echo "CMD $(up)"; motors -d g -s 250 -x 400 -y 0 >/dev/null; poll 3.5; idle
echo "== B: +600 at 250 then reverse"; echo "CMD $(up)"; motors -d g -s 250 -x 600 -y 0 >/dev/null; poll 0.8; echo "CMD2 $(up)"; motors -d g -s 250 -x -600 -y 0 >/dev/null; poll 5; idle
echo "== C: -400 at 900"; echo "CMD $(up)"; motors -d g -s 900 -x -400 -y 0 >/dev/null; poll 2.5
motors -d g -s 250 -x 0 -y 0 >/dev/null; echo "END $(pos)"
'''
    data, cmds = parse(cam(script))
    out = {}
    for seg, rows in data.items():
        t0 = cmds[seg]["CMD"]
        first = next((t for t, st, x, _ in rows if st == 1 or x != rows[0][2]), None)
        moving = [r for r in rows if r[1] == 1 and first and r[0] > first + 0.3]
        sp = abs((moving[-1][2] - moving[0][2]) / (moving[-1][0] - moving[0][0])) if len(moving) > 2 else None
        out[seg] = {"first_motion_ms": round((first - t0) * 1000) if first else None, "speed": round(sp) if sp else None}
        if seg == "B" and "CMD2" in cmds[seg]:
            peak = max(rows, key=lambda r: r[2])
            out[seg]["queues"] = peak[0] > cmds[seg]["CMD2"] + 0.5
    return out


def frame_steps(axis, steps=200, trials=3, cam=None, snap=None):
    cam, snap = cam or camera, snap or snapshot
    """Steps per frame width/height and backlash, from image shifts (phase correlation). Each trial measures a move that
    continues the previous direction and one that reverses it: the gears take up slack on a reversal, the step counter
    does not notice, and the picture moves less (2026-11-04 on the reference camera: 740 steps/frame, 27 steps lost)."""
    def move(n):
        if cam is camera:
            hold_follow()
        d = f"-x {n} -y 0" if axis == "pan" else f"-x 0 -y {n}"
        cam(f"motors -d g -s 250 {d} >/dev/null; sleep 0.3; " + IDLE + "; sleep 0.8")
    k = 0 if axis == "pan" else 1
    same, rev, conf = [], [], []
    for _ in range(trials):
        move(150)
        a = snap()
        move(steps)
        b = snap()
        s1 = shift(a, b)
        move(-(steps + 150))
        move(-150)
        a = snap()
        move(steps)
        b = snap()
        s2 = shift(a, b)
        move(-(steps - 150))
        same.append(abs(s1[k]))
        rev.append(abs(s2[k]))
        conf += [s1[2], s2[2]]
    size = snap().shape[1 - k]
    ps = float(np.median(same))
    return {"steps_per_frame": round(steps * size / ps, 1) if ps > 2 else None,
            "backlash_steps": round(steps * (1 - float(np.median(rev)) / ps), 1) if ps > 2 else None,
            "shift_px_same": same, "shift_px_reversed": rev, "min_confidence": round(min(conf), 1)}


def main(argv):
    owner = "--owner-present" in argv
    live = json.loads(server("sudo cat /var/lib/petcam/live.json"))
    privacy = server("sudo cat /var/lib/petcam/privacy 2>/dev/null || echo 0").strip()
    if privacy == "1":
        sys.exit("privacy is on: no camera tests")
    if live.get("present") and live.get("rest_since") and not owner:
        sys.exit("the pet is resting in view: no motor tests now (rerun with --owner-present if someone is with her)")
    if "--dry-run" in argv:
        print("would pause Follow and run the motor + frame-size tests")
        return
    hold_follow()                                              # pauses Follow and Find (re-asserted before each move)
    start = centre()
    print("centred at", start, flush=True)
    motor = motor_runs()
    hold_follow()
    centre()
    geo = {"pan": frame_steps("pan"), "tilt": frame_steps("tilt")}
    centre(pan=start[0], tilt=start[1])                        # leave it where the test started (mid-range)
    lim = json.loads(camera("motors -j"))
    p_path = os.path.join(HERE, "params.json")
    P = json.load(open(p_path)) if os.path.exists(p_path) else {}
    P["motor"] = {"measured": time.strftime("%Y-%m-%d"), "first_motion_ms": [motor[s]["first_motion_ms"] for s in motor],
                  "speed_steps_s": {"250": motor["A"]["speed"], "900": motor["C"]["speed"]},
                  "second_command": "queues behind the move in progress" if motor["B"].get("queues") else "replaces it",
                  "raw": motor}
    g = P.setdefault("geometry", {})
    for ax in ("pan", "tilt"):
        m = geo[ax]
        # trust a measurement only when both trials agree within 5% (a busy room breaks phase correlation)
        ok = m["steps_per_frame"] and len(m["shift_px_same"]) > 1 and \
            max(m["shift_px_same"]) - min(m["shift_px_same"]) <= 0.05 * max(m["shift_px_same"])
        if ok:
            g[f"{ax}_frame"] = m["steps_per_frame"]
            g[f"{ax}_backlash"] = m["backlash_steps"]
        m["accepted"] = bool(ok)
    g["frame_measurement"] = geo
    g["pose_at_test"] = [int(float(lim.get("xpos", 0))), int(float(lim.get("ypos", 0)))]
    json.dump(P, open(p_path, "w"), indent=1)
    print(json.dumps({"motor": P["motor"], "geometry": g}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1:])
