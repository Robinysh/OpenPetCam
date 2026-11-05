"""The saved-view chip for the view the camera is at must be highlighted.

Report 20261103-111546: the camera was at the corridor view (a search had parked there and Follow had nudged
it a little since), but no chip in the saved-views row was marked, so the owner couldn't tell which view
was on screen. The app had no notion of "the current saved view" at all.

Runs currentSpot() from server/index.html in node. Usage: python3 tests/server/spot_highlight_test.py
"""
import json
import os
import re
import shutil
import subprocess

HTML = os.path.join(os.path.dirname(__file__), "..", "..", "server", "index.html")
PRESETS = [{"name": "Corridor", "x": 1019, "y": 534}, {"name": "Bed", "x": 355, "y": 450},
           {"name": "Window", "x": 1718, "y": 426}]
FRAME = [733, 424]


def current_spot(x, y, presets=PRESETS):
    src = open(HTML).read()
    m = re.search(r"^function currentSpot\(.*?^}\n", src, re.S | re.M)
    assert m, "currentSpot() not found in index.html: the app can't tell which saved view it is at"
    cam = {"x": x, "y": y, "frame": FRAME, "presets": presets}
    js = m.group(0) + f"\nconsole.log(JSON.stringify([currentSpot({json.dumps(cam)})]));"
    out = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)[0]


def main():
    if not shutil.which("node"):
        print("spot highlight test skipped: node not installed")
        return
    assert current_spot(1019, 534) == "Corridor"                     # exactly there
    assert current_spot(1080, 560) == "Corridor", "Follow's small nudges keep it the same view"
    assert current_spot(1276, 513) == "Corridor", "the reported tracking turn still shows most of the saved view"
    assert current_spot(360, 440) == "Bed"
    assert current_spot(1400, 500) is None, "half a frame away from every view is no saved view"
    assert current_spot(1019, 300) is None, "tilted well away is not the view"
    assert current_spot(1312, 704) is None, "small overlap on both axes is not mostly the saved view"
    assert current_spot(1019, 534, []) is None
    # Two views close together: the nearer one wins, never both.
    two = [{"name": "A", "x": 1000, "y": 500}, {"name": "B", "x": 1100, "y": 500}]
    assert current_spot(1080, 500, two) == "B"
    # With offsets on two axes, prefer the view with more image area left, as view statistics do.
    overlap = [{"name": "A", "x": 780, "y": 500}, {"name": "B", "x": 853, "y": 415}]
    assert current_spot(1000, 500, overlap) == "A"
    src = open(HTML).read()
    assert re.search(r'setAttribute\("aria-pressed", p\.name === here\)', src), "renderSpots must mark the chip"
    assert '.chip.spot[aria-pressed="true"]' in src, "a marked chip needs a visible style"
    print("spot highlight ok")


if __name__ == "__main__":
    main()
