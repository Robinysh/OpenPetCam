"""Latency profile of every stage between the pet moving and the app knowing it, and between a
joystick touch and the motor moving. Runs on the server as the petcam user:

  sudo -u petcam /opt/petcam/venv/bin/python profile_latency.py [--n 15] [--motor] [--video]

--motor   nudges the camera one small step and back (it moves!)
--video   blanks the picture for ~3 s (privacy mode) to time camera -> server video latency

Prints median / p90 per stage. Keep N small: the box has a ~5% CPU baseline.
"""
import io
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.request

sys.path.insert(0, "/opt/petcam")
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

import app as camapi  # noqa: E402
import watch  # noqa: E402
from detector import Detector  # noqa: E402

N = int(sys.argv[sys.argv.index("--n") + 1]) if "--n" in sys.argv else 15


def timed(fn, n=N):
    out, ret = [], None
    for _ in range(n):
        t0 = time.perf_counter()
        ret = fn()
        out.append((time.perf_counter() - t0) * 1000)
    return out, ret


def show(name, ms):
    ms = sorted(ms)
    p90 = ms[min(len(ms) - 1, int(len(ms) * 0.9))]
    print(f"{name:42s} median {statistics.median(ms):7.1f} ms   p90 {p90:7.1f} ms   (n={len(ms)})")


def steal():
    with open("/proc/stat") as f:
        v = list(map(int, f.readline().split()[1:]))
    return v[7], sum(v)


s0 = steal()
print("== watcher pipeline (one cycle)")
ms, jpg = timed(lambda: watch.snapshot("ch1"))
show("snapshot ch1.jpg from camera (WireGuard)", ms)
print(f"   snapshot size {len(jpg)} bytes")
ms, img = timed(lambda: Image.open(io.BytesIO(jpg)).convert("RGB"))
show("JPEG decode 640x360", ms)
ms, _ = timed(lambda: np.asarray(img.convert("L").resize((160, 90), Image.BILINEAR), dtype=np.float32))
show("motion: gray 160x90", ms)
det = Detector(watch.MODEL)
timed(lambda: det.detect(img, min_score=0.10), n=2)          # warm-up
ms, found = timed(lambda: det.detect(img, min_score=0.10))
show("YOLOX-nano detect (1 thread)", ms)
ver = watch.Verifier(watch.VERIFIER)
box = next((f["box"] for f in found if f["label"] != "person"), [0.3, 0.3, 0.6, 0.6])
timed(lambda: ver.prob(img, box), n=2)
ms, _ = timed(lambda: ver.prob(img, box))
show("verifier per candidate (MobileNetV2)", ms)
ms, _ = timed(lambda: watch.snapshot("ch0"), n=5)
show("snapshot ch0.jpg (main stream, for reference)", ms)

print("\n== web app")
def get(path):
    with urllib.request.urlopen(f"http://127.0.0.1:8080{path}", timeout=5) as r:
        return r.read()
ms, _ = timed(lambda: get("/healthz"))
show("app /healthz (local)", ms)
ms, _ = timed(lambda: camapi.read_live())
show("read_live() (what /api/live returns)", ms)
ms, _ = timed(lambda: camapi.motor_status(), n=8)
show("camera motor status CGI", ms)

if "--motor" in sys.argv:
    print("\n== motor (moves the camera one small step and back)")
    camapi.wait_idle()
    res = []
    for sign in (1, -1, 1, -1):
        _, x0, _ = camapi.motor_status()
        t0 = time.perf_counter()
        camapi.nudge(0.35 * sign, 0)                  # what the joystick sends
        t_ack = time.perf_counter()
        t_move = None
        while time.perf_counter() - t0 < 3:
            st, x, _ = camapi.motor_status()
            if x != x0:
                t_move = time.perf_counter()
                break
        camapi.wait_idle()
        t_done = time.perf_counter()
        res.append(((t_ack - t0) * 1000, ((t_move or t_done) - t0) * 1000, (t_done - t0) * 1000))
    show("nudge(): API returns", [r[0] for r in res])
    show("nudge(): motor position first changes", [r[1] for r in res])
    show("nudge(): motor idle again", [r[2] for r in res])

if "--video" in sys.argv:
    print("\n== video: camera -> server (privacy on, see when frames go black, privacy off)")
    proc = subprocess.Popen(["ffmpeg", "-loglevel", "error", "-rtsp_transport", "tcp", "-i", "rtsp://127.0.0.1:8554/cam",
                             "-an", "-vf", "scale=64:36,format=gray", "-f", "rawvideo", "-"], stdout=subprocess.PIPE)
    frames = []
    def reader():
        while True:
            b = proc.stdout.read(64 * 36)
            if len(b) < 64 * 36:
                return
            frames.append((time.time(), float(np.frombuffer(b, np.uint8).mean())))
    threading.Thread(target=reader, daemon=True).start()
    t_wait = time.time()
    while len(frames) < 30 and time.time() - t_wait < 15:
        time.sleep(0.2)
    print(f"   first decoded frame after {(frames[0][0] - t_wait) * 1000:.0f} ms (waits for a keyframe)" if frames else "   no frames")
    base = statistics.median(m for _, m in frames[-50:])
    t_on = time.time(); camapi.set_privacy(True)
    t_api = time.time()
    snap_dark = None
    while time.time() - t_on < 8:
        if snap_dark is None:
            try:
                m = float(np.asarray(Image.open(io.BytesIO(watch.snapshot("ch1"))).convert("L")).mean())
                if m < base * 0.3:
                    snap_dark = time.time()
            except Exception:  # noqa: BLE001
                pass
        time.sleep(0.05)
    dark = next((t for t, m in frames if t > t_on and m < base * 0.3), None)
    camapi.set_privacy(False)
    proc.terminate()
    print(f"   baseline brightness {base:.0f}; privacy API call took {(t_api - t_on) * 1000:.0f} ms")
    print(f"   RTSP via MediaMTX shows black after {((dark or float('nan')) - t_on) * 1000:.0f} ms")
    print(f"   ch1.jpg snapshot shows black after  {((snap_dark or float('nan')) - t_on) * 1000:.0f} ms")

s1 = steal()
print(f"\nCPU steal during the run: {100 * (s1[0] - s0[0]) / max(1, s1[1] - s0[1]):.1f}%  (load: {os.getloadavg()[0]:.2f})")
print(json.dumps({"period_s": watch.PERIOD, "detect_every_s": watch.DETECT_EVERY, "motion_detect": watch.MOTION_DETECT,
                  "leave_after_s": watch.LEAVE_AFTER, "ui_poll_s": 2.0}))
