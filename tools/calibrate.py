"""Measure how far the picture shifts per motor step, per axis, and in which direction.

Moves the camera by a known number of steps, grabs a frame before and after from the LAN
RTSP substream, and finds the shift with phase correlation. Returns the camera to where it
started. Prints steps-per-frame-width/height and the sign of the image motion.
"""
import subprocess
import sys
import time

import numpy as np
from PIL import Image

from siteconf import SITE, SSH  # noqa: E402

# Camera on your home network (run this from home). The default Thingino RTSP login is assumed.
RTSP = f"rtsp://thingino:thingino@{SITE.get('CAMERA_LAN_IP', 'thingino.local')}:554/ch1"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp"


def frame(name):
    path = f"{OUT}/{name}.png"
    # Skip a second of frames so we don't read a stale buffered frame.
    subprocess.run(["timeout", "-k", "3", "20", "ffmpeg", "-v", "error", "-rtsp_transport", "tcp", "-i", RTSP,
                    "-ss", "1.5", "-frames:v", "1", "-y", path], check=True)
    return np.asarray(Image.open(path).convert("L"), dtype=np.float64)


def move(dx, dy):
    code = ("import sys; sys.path.insert(0,'/opt/petcam'); import app; "
            f"m=app.cam_get('/x/json-motor.cgi', {{'d':'g','x':{dx},'y':{dy}}}).get('message') or {{}}; "
            "print(m.get('xpos'), m.get('ypos'))")
    subprocess.run(SSH + [f"sudo -u petcam python3 -c \"{code}\""], check=True, capture_output=True)
    time.sleep(4)


def shift(a, b):
    """Return (dx, dy) in pixels such that b ≈ a shifted by (dx, dy)."""
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    fa, fb = np.fft.fft2((a - a.mean()) * win), np.fft.fft2((b - b.mean()) * win)
    r = fb * np.conj(fa)
    r /= np.abs(r) + 1e-9
    corr = np.abs(np.fft.ifft2(r))
    py, px = np.unravel_index(np.argmax(corr), corr.shape)
    h, w = a.shape
    dy = py - h if py > h // 2 else py
    dx = px - w if px > w // 2 else px
    peak = corr.max() / corr.mean()
    return dx, dy, peak


def _measure():
    STEPS = {"pan": 150, "tilt": 100}
    f0 = frame("cal0")
    h, w = f0.shape
    print(f"frame {w}x{h}")
    
    move(STEPS["pan"], 0)
    f1 = frame("cal_pan")
    move(-STEPS["pan"], 0)
    dx, dy, peak = shift(f0, f1)
    print(f"pan  +{STEPS['pan']} steps -> image shift dx={dx}px dy={dy}px (peak {peak:.0f})")
    if dx:
        print(f"  steps per frame width  ~ {STEPS['pan'] * w / abs(dx):.0f}; image moves {'right' if dx > 0 else 'left'} for +x")
    
    f0b = frame("cal0b")
    move(0, STEPS["tilt"])
    f2 = frame("cal_tilt")
    move(0, -STEPS["tilt"])
    dx, dy, peak = shift(f0b, f2)
    print(f"tilt +{STEPS['tilt']} steps -> image shift dx={dx}px dy={dy}px (peak {peak:.0f})")
    if dy:
        print(f"  steps per frame height ~ {STEPS['tilt'] * h / abs(dy):.0f}; image moves {'down' if dy > 0 else 'up'} for +y")

if __name__ == "__main__":
    _measure()
