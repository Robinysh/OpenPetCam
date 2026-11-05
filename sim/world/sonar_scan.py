"""On the server: sound scan for the FWI experiment. Holds Follow, plays the probe from the camera speaker and records
the camera's RAW microphone tap (48 kHz s16, before Opus) for each direction in a priority-ordered plan, streaming the
tap to the server (nothing accumulates in the camera's /tmp). Hard deadline: no sweep starts after DEADLINE (unix).
    sudo python3 sonar_scan.py DEADLINE OUTDIR [START_INDEX] [PLAN.json]"""
import json, os, subprocess, sys, time
sys.path.insert(0, "/opt/petcam")
import app

DEADLINE, OUT = float(sys.argv[1]), sys.argv[2]
START = int(sys.argv[3]) if len(sys.argv) > 3 else 0
PLAN_FILE = sys.argv[4] if len(sys.argv) > 4 else None       # a sim/world/sonar_plan.py file (randomised, seeded)
CAM = open("/tmp/camcmd").read().split()
os.makedirs(OUT, exist_ok=True)
LEVEL, DOWN, UP = 699, 520, 880                     # level tilt (calibrated), ~15 deg down, ~15 deg up
PANS = [1850, 2210, 1490, 2570, 1130, 2930, 770, 3290, 410, 3650, 50]   # outward from the room centre, 30 deg apart
plan = [("silence", 1850, LEVEL, 0), ("repeat", 1850, LEVEL, 40)]
for row in (LEVEL, DOWN, UP):
    for i, p in enumerate(PANS):
        plan.append(("scan", p, row, 40))
        if row == DOWN and i == 5:
            plan.append(("repeat", 1850, LEVEL, 40))
plan += [("repeat", 1850, LEVEL, 40), ("level", 1850, LEVEL, 30), ("silence", 1850, LEVEL, 0)]
meta = {"seed": None, "design": "fixed rows (2026-11-04 session)"}
if PLAN_FILE:
    meta = json.load(open(PLAN_FILE))
    plan = [tuple(e) for e in meta["plan"]]
json.dump({"seed": meta.get("seed"), "design": meta.get("design"), "plan": plan}, open(os.path.join(OUT, "plan.json"), "w"))
app.cam_login()
start = app.motor_status()[1:]


def hold():
    open("/var/lib/petcam/moved_at", "w").close()


log = open(os.path.join(OUT, "log.jsonl"), "a")
try:
    for k, (kind, pan, tilt, vol) in enumerate(plan):
        if k < START:
            continue
        if time.time() > DEADLINE - 20:
            print("deadline reached before", k, flush=True)
            break
        hold()
        app.goto(pan, tilt, 250); time.sleep(0.4); app.wait_idle(30); time.sleep(0.6)
        pos = app.motor_status()[1:]
        hold()
        now = time.time()
        if vol:
            open("/var/lib/petcam/talking", "w").write(f"{now} {now + 16}")
        fn = os.path.join(OUT, f"{k:02d}_{kind}_{pan}_{tilt}_v{vol}.pcm")
        with open(fn, "wb") as f:
            rec = subprocess.Popen(CAM + ["timeout 13 cat /run/prudynt/audio_mic.pcm"], stdout=f, stderr=subprocess.DEVNULL)
            time.sleep(1.5)
            t_play = time.time()
            if vol:
                with open("/tmp/chirps.wav", "rb") as w:
                    r = subprocess.run(CAM + [f"play -s -f wav -v {vol} 2>&1"], stdin=w, capture_output=True, text=True, timeout=30)
            rec.wait(timeout=30)
            # play only QUEUES the file for prudynt: delete it after the recording, never before (2026-11-04 bug)
            subprocess.run(CAM + ["rm -f /tmp/play-stdin*"], capture_output=True, timeout=20)
        size = os.path.getsize(fn)
        rowj = {"k": k, "kind": kind, "pan": pan, "tilt": tilt, "pos": list(pos), "vol": vol, "t_rec": now, "t_play": t_play,
                "bytes": size, "seconds": round(size / 96000, 2)}
        log.write(json.dumps(rowj) + "\n"); log.flush()
        print(k, kind, pos, vol, rowj["seconds"], "s", flush=True)
        time.sleep(0.5)
finally:
    hold()
    app.goto(*start, 250)
