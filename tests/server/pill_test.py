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

HTML = os.path.join(os.path.dirname(__file__), "..", "..", "server", "index.html")


def extract(src, name):
    m = re.search(r"^function " + name + r"\(.*?^}\n", src, re.S | re.M)
    assert m, f"{name} not found in index.html"
    return m.group(0)


def pet_text(live):
    src = open(HTML).read()
    js = "\n".join([
        'const PET = {"name": "the dog"};',
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
    assert pet_text({"status": "ok", "present": False, "last_seen": None}) == "the dog isn't in view"
    print("pill ok")


if __name__ == "__main__":
    sys.exit(main())
