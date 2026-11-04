"""Toggle Thingino privacy mode on/off via the app and check the picture recovers. Runs on the server."""
import hashlib
import sys
import time

sys.path.insert(0, "/opt/petcam")
import app  # noqa: E402


def snap(tag):
    b = app.live_frame()
    with open(f"/tmp/priv_{tag}.jpg", "wb") as f:
        f.write(b)
    return hashlib.md5(b).hexdigest()[:8], len(b)


print("privacy now:", app.prudynt({"privacy": {"enabled": None}}))
print("before :", snap("before"))
print("on     :", app.set_privacy(True))
time.sleep(5)
print("during :", snap("during"))
print("off    :", app.set_privacy(False))
time.sleep(5)
a = snap("after1")
time.sleep(3)
b = snap("after2")
print("after  :", a, b, "-> picture", "LIVE (changing)" if a[0] != b[0] else "IDENTICAL (frozen?)")
