"""Step 0 of measuring Find honestly: a search keeps the frame at every stop and every sighting, with each box
and verifier score, so "found her" can be labelled right or wrong later (tools/search_review.py).
  python3 tests/server/search_evidence_test.py
"""
import json
import os
import sys
import tempfile
import types

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "server"))
sys.path.insert(0, "/opt/petcam")
import watch  # noqa: E402
from PIL import Image  # noqa: E402

watch.SEARCHES = tempfile.mkdtemp()
w = watch.Watch.__new__(watch.Watch)
w.search_rec = None
w.tracker = types.SimpleNamespace(searching=True, last_move=1.0, cam=types.SimpleNamespace(position=lambda: (1200, 500)))
day, night = Image.new("RGB", (640, 360), (180, 140, 90)), Image.new("RGB", (640, 360), (90, 90, 90))
J = [{"box": [0.4, 0.4, 0.5, 0.5], "det": 0.4, "p": 0.97}]

w.record_search_frame(night, [], None, settled=True)            # first stop, nothing there: kept
w.record_search_frame(night, [], None, settled=True)            # same stop, still nothing: not kept
w.record_search_frame(night, [], None, settled=False)           # mid-pan, nothing: not kept
w.record_search_frame(night, J, J[0]["box"], settled=False)     # mid-pan sighting: kept
w.tracker.last_move = 2.0                                        # the camera moved to a new stop
w.record_search_frame(night, J, J[0]["box"], settled=True)      # new stop with a sighting: kept, with its position
assert len(w.search_rec["frames"]) == 3, [f["settled"] for f in w.search_rec["frames"]]
for _ in range(40):
    w.record_search_frame(night, J, J[0]["box"], settled=False)
assert len(w.search_rec["frames"]) == watch.SEARCH_FRAMES_MAX, "frame cap not enforced"

w.save_search_evidence("search (you asked): found her at (1200, 500) in 9 s", asked=True)
assert w.search_rec is None
(d,) = [os.path.join(dp) for dp, _, fs in os.walk(watch.SEARCHES) if "search.json" in fs]
meta = json.load(open(os.path.join(d, "search.json")))
assert meta["found"] and meta["asked"] and meta["night"] is True, meta
assert meta["frames"][0]["pos"] == [1200, 500] and meta["frames"][1]["pos"] is None and meta["frames"][2]["pos"] == [1200, 500]
assert meta["frames"][1]["verified"] == J[0]["box"] and meta["frames"][1]["judged"][0]["p"] == 0.97
assert len([f for f in os.listdir(d) if f.endswith(".jpg")]) == watch.SEARCH_FRAMES_MAX
assert meta["confirm"] == "2 of 4"
w.save_search_evidence("search: nothing recorded", asked=False)  # nothing recorded: no folder, no error
w.search_rec = {"t0": 1.0, "frames": [{"jpg": b"not a jpeg", "t": 0, "settled": True, "judged": [], "verified": None, "pos": None}]}
w.save_search_evidence("search: broken frame", asked=False)      # bad data is logged, never raised
print(f"search evidence: stops and sightings kept ({watch.SEARCH_FRAMES_MAX} max), night flagged, scores saved")
