"""FRAME_WIDTH / DETECT_WIDTH scale the watcher's picture and detector input; the defaults reproduce 640x360 with a
256x416 detector and a (384, 640) second look. Run: python3 tests/server/frame_size_test.py (no camera or server)."""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def probe(env_lines):
    env = Path(tempfile.mkdtemp()) / "site.env"
    env.write_text("PET_NAME=T\n" + env_lines)
    code = (f"import sys; sys.path.insert(0, '{ROOT}/server'); import petconf, app, watch; "
            "print(watch.FrameSource.W, watch.FrameSource.H, watch.DETECT_SHAPE, watch.NATIVE_SHAPE, watch.FrameSource.W // 160)")
    return subprocess.run([sys.executable, "-I", "-c", code], env={**os.environ, "PETCAM_SITE_ENV": str(env)},
                          capture_output=True, text=True, check=True).stdout.strip()


assert probe("") == "640 360 (256, 416) (384, 640) 4", probe("")                       # unchanged defaults
assert probe("FRAME_WIDTH=1280\nDETECT_WIDTH=1024\n") == "1280 720 (576, 1024) (736, 1280) 8"
assert probe("FRAME_WIDTH=1000\n") == "640 360 (256, 416) (384, 640) 4"                 # not a multiple of 160: ignored
assert probe("FRAME_WIDTH=abc\nDETECT_WIDTH=100\n") == "640 360 (256, 416) (384, 640) 4"
print("ok")
