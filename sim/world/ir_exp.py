"""On the server: active vs passive near-IR at one pose. Restores the IR-cut filter and LEDs whatever happens."""
import json, os, subprocess, sys, time
sys.path.insert(0, "/opt/petcam")
import app
CAM = open("/tmp/camcmd").read().split()
OUT = sys.argv[1]; pan, tilt = int(sys.argv[2]), int(sys.argv[3])
os.makedirs(OUT, exist_ok=True)
cam = lambda c: subprocess.run(CAM + [c], capture_output=True, timeout=30).stdout
app.cam_login(); start = app.motor_status()[1:]
hold = lambda: open("/var/lib/petcam/moved_at", "w").close()
log = []
try:
    hold(); app.goto(pan, tilt, 250); time.sleep(0.4); app.wait_idle(30); time.sleep(1.5)
    for state, filt, led in (("A_visible", "on", "off"), ("B_passive", "off", "off"), ("C_active", "off", "on"), ("B2_passive", "off", "off")):
        hold(); cam(f"ircut {filt} >/dev/null 2>&1; light ir940 {led} >/dev/null 2>&1")
        for k, wait in enumerate((1.0, 2.0)):
            time.sleep(wait)
            jpg = cam("prudyntctl snapshot -c 0 -q 92")
            fn = f"{state}_{k}.jpg"; open(os.path.join(OUT, fn), "wb").write(jpg)
            log.append({"file": fn, "state": state, "ircut": filt, "led": led, "t": time.time(), "bytes": len(jpg)})
            print(fn, len(jpg), flush=True)
finally:
    cam("light ir940 off >/dev/null 2>&1; ircut on >/dev/null 2>&1")
    json.dump({"pan": pan, "tilt": tilt, "pos": list(app.motor_status()[1:]), "frames": log}, open(os.path.join(OUT, "log.json"), "w"))
    hold(); app.goto(*start, 250)
    print("restored; ircut:", cam("ircut status 2>&1").decode(errors="replace").strip()[-30:], flush=True)
