"""The status pill must not claim she is away when the watcher hasn't reported for a while.

Report 20261102-200430: the dog stood in the middle of the picture, but the watcher's last word was
27 s old (status watcher_down: the server was starved and one cycle ran long), and the pill said
"the dog isn't in view". A stale watcher knows nothing about where she is now.

Runs petText() from server/index.html in node. Usage: python3 tests/server/pill_test.py
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time

HTML = os.path.join(os.path.dirname(__file__), "..", "..", "server", "index.html")


def extract(src, name):
    m = re.search(r"^function " + name + r"\(.*?^}\n", src, re.S | re.M)
    assert m, f"{name} not found in index.html"
    return m.group(0)


def pet_text(live):
    src = open(HTML).read()
    js = "\n".join([
        'const PET = {"name": "Biscuit"};',
        extract(src, "fmtDur"), extract(src, "ago"), extract(src, "petText"),
        f"console.log(JSON.stringify(petText({json.dumps(live)})));",
    ])
    out = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)[0]


def main():
    if not shutil.which("node"):
        print("pill test skipped: node not installed")
        return
    # The report's live state: watcher silent for 26.8 s, not present, never seen.
    stale = {"status": "watcher_down", "present": False, "last_seen": None, "watch_age": 26.8}
    text = pet_text(stale)
    assert "isn't in view" not in text and "Not in view" not in text, \
        f"stale watcher must not say she's away: {text!r}"
    # Same with an old sighting.
    text = pet_text(stale | {"last_seen": 1.0})
    assert "Not in view" not in text, f"stale watcher must not say she's away: {text!r}"
    # A working watcher that doesn't see her still says so.
    assert pet_text({"status": "ok", "present": False, "last_seen": None}) == "Biscuit isn't in view"
    # Report 20261103-074105: Follow's automatic search swept the room while the pill said
    # "Not in view · seen 53 min ago", so the owner thought the camera was moving on its own.
    # The watcher publishes searching=True during a sweep; the pill must say so.
    text = pet_text({"status": "ok", "present": False, "last_seen": time.time() - 600, "searching": True,
                     "settings": {"follow": True}})
    assert "Follow" in text and "looking" in text.lower(), f"a follow search must be visible: {text!r}"
    # A search holds the camera still while it judges a view: "Looking for" over a motionless picture
    # read as a stuck camera. The pill says it is checking this view; a moving camera keeps the old text.
    asked = {"status": "ok", "present": False, "last_seen": None, "searching": True}
    assert pet_text(asked | {"search_phase": "looking"}) == "Checking this view for Biscuit…"
    assert pet_text(asked | {"search_phase": "moving"}) == "Looking for Biscuit…"
    text = pet_text(asked | {"search_phase": "looking", "settings": {"follow": True}})
    assert text == "Follow: checking this view for Biscuit…", text
    # Report 20261104-172936: she barked (dog sound 0.74) right after settling, and the pill kept
    # saying "resting 1m". A dog sound in the last minute must show; an old one must not.
    t = time.time()
    resting = {"status": "ok", "present": True, "last_seen": t - 5, "rest_since": t - 60}
    text = pet_text(resting | {"last_bark": {"t": t - 3, "kind": "dog_sound"}})
    assert "barking" in text and "resting" not in text, f"a fresh bark must show: {text!r}"
    text = pet_text(resting | {"last_bark": {"t": t - 30, "kind": "howling"}})
    assert "howling" in text and "s ago" in text, f"a bark 30 s ago must show: {text!r}"
    text = pet_text(resting | {"last_bark": {"t": t - 300, "kind": "barking"}})
    assert "resting" in text, f"an old bark must not hide 'resting': {text!r}"
    # Heard but not in view: still worth saying.
    text = pet_text({"status": "ok", "present": False, "last_seen": t - 600, "last_bark": {"t": t - 2, "kind": "barking"}})
    assert "barking" in text, f"a bark out of view must show: {text!r}"
    print("pill ok")


if __name__ == "__main__":
    sys.exit(main())
