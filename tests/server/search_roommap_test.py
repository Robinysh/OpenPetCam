"""Search uses the room map: the likeliest views for this hour first, no stops or sweep rows where the view shows no
floor, and a failed search says what the history suggests instead of a flat "not found" (2026-11-05).

Evidence (sim/world/find_bench.py --replay on 56 recorded searches with a known position, map built without the
searched day): the room map's stops then the learned ones had her in view at a stop in 48 (learned order: 36),
median 16 s to that view (20 s); 24% of recorded search frames showed no floor at all.
  python3 tests/server/search_roommap_test.py
"""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "server"))
import tracker as T  # noqa: E402
from roommap import RoomMap  # noqa: E402
from tracker_test import World, tracker_for  # noqa: E402  (runs the tracker suite once on import)


def doc(stops, no_floor=(), rows_without_floor=()):
    poses = []
    for pan in range(0, 3701, 100):
        for tilt in range(0, 1001, 100):
            bare = (pan, tilt) in no_floor or tilt in rows_without_floor
            poses.append({"pan": pan, "tilt": tilt, "floor_share": 0.0 if bare else 0.3, "tiles": [] if bare else [3, 4, 5]})
    return {"version": 1, "pan_step": 100, "tilt_step": 100, "min_floor": 0.02, "poses": poses,
            "stops": {str(h): stops for h in range(24)},
            "places": {str(h): [{"place": "bowls", "p": 0.6, "hidden": 0.1}] for h in range(24)}}


class Saved(World):
    def presets(self):
        return [{"x": 400, "y": 500}, {"x": 3200, "y": 500}]


try:
    # 1. The map's stop for this hour comes first, ahead of the saved views, and she is found there.
    T.ROOM_MAP = RoomMap(doc([[2500, 500, 0.7]]))
    w = Saved(X=1850, pet=2500)
    tr = tracker_for(w)
    res = tr.search("test")
    assert "found her at the room map's stop (2500, 500)" in res, res
    assert w.moves[0] == ("goto", 2500), w.moves
    print("map stop first:", res)

    # 2. A saved view whose view shows no floor is never visited; one the map's stop already covers is merged away.
    T.ROOM_MAP = RoomMap(doc([[3100, 500, 0.7]], no_floor={(400, 500)}))
    tr = tracker_for(Saved())
    st = tr.search_stops()
    assert (400, 500) not in [(s["x"], s["y"]) for s in st], st          # no floor
    assert (3200, 500) not in [(s["x"], s["y"]) for s in st], st         # inside the map stop's view
    assert st[0] == {"x": 3100, "y": 500, "map": True}, st
    print("stops:", st)

    # 3. Sweep rows: a row with no floor is skipped; without a map the plan is unchanged.
    w = World()
    tr = tracker_for(w)
    rows = tr.rows(500)
    T.ROOM_MAP = None
    plain = tr.sweep_plan(500)
    T.ROOM_MAP = RoomMap(doc([], rows_without_floor={rows[0] // 100 * 100}))
    trimmed = tr.sweep_plan(500)
    assert len(plain) == 2 * len(rows) and len(trimmed) == len(plain) - 2, (plain, trimmed)
    assert all(y != rows[0] for _, y in trimmed), trimmed
    print("sweep without the floorless row:", trimmed)

    # 4. Not found anywhere (she is not in the room): the result names the likeliest place and how long she has been
    #    unseen, and still starts with "not found" (the app and the evidence recorder read that).
    T.ROOM_MAP = RoomMap(doc([[2500, 500, 0.7]]))
    w = Saved(X=1850, pet=99999)
    tr = tracker_for(w)
    tr.last_seen_t = w.t - 1800
    res = tr.search("test")
    assert "not found" in res and "probably out of the room" in res and "bowls (60%" in res and "not confirmed" in res, res
    print("not found:", res)

    # 5. A missing, old or broken file is no map (never an exception): search as before.
    d = tempfile.mkdtemp()
    for name, text in (("bad.json", "{"), ("old.json", json.dumps({"version": 0})), ("half.json", json.dumps({"version": 1}))):
        p = os.path.join(d, name)
        open(p, "w").write(text)
        assert RoomMap.load(p) is None, name
    assert RoomMap.load(os.path.join(d, "missing.json")) is None
    m = RoomMap(doc([]))
    assert m.has_floor(-5000, -5000) and len(m.tiles(-5000, -5000)) == 9      # unknown pose: never skipped
    print("search room map ok")
finally:
    T.ROOM_MAP = None
