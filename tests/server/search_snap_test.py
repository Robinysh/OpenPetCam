"""While a search holds the loop, the minute's picture is still saved (report 20261106-184614), without overwriting one.
  python3 tests/server/search_snap_test.py
"""
import os
import sys
import tempfile
import time
import types

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "server"))
import watch  # noqa: E402

d = tempfile.mkdtemp()
watch.SNAPS = d
watch.log = lambda *a: None
w = watch.Watch.__new__(watch.Watch)
w.frames = types.SimpleNamespace(latest=lambda: Image.new("RGB", (64, 36), (90, 90, 90)))
t = time.time()
w.search_snap(t)
day = time.strftime("%Y-%m-%d", time.localtime(t))
p = os.path.join(d, day, time.strftime("%H-%M", time.localtime(t)) + ".jpg")
assert os.path.exists(p)
open(p, "wb").write(b"kept")
w.search_snap(t + 1)
assert open(p, "rb").read() == b"kept"            # the cycle's own picture is never replaced
w.frames = types.SimpleNamespace(latest=lambda: None)
w.search_snap(t + 61)                              # no frame: nothing, no error
print("search snapshot ok")
