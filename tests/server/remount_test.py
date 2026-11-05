"""After a re-mount: logged positions keep the correction in force when they were logged, and the room map's floor
grid is looked up through the pose shift (sim/world/remount.py)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "server"))
import roomapi  # noqa: E402
from roommap import RoomMap  # noqa: E402


def test_offset_at():
    doc = {"mount_offset_deg": [8.7, 7.7, 0], "mount_offsets": [[0, 2.2, -4.8, 0], [1000.0, 8.7, 7.7, 0]]}
    assert roomapi.offset_at(doc, 500) == (2.2, -4.8, 0)
    assert roomapi.offset_at(doc, 1000.0) == (8.7, 7.7, 0)
    assert roomapi.offset_at(doc, 5000) == (8.7, 7.7, 0)
    assert roomapi.offset_at({"mount_offset_deg": [1, 2, 0]}, 5) == (1, 2, 0)     # no history: the single value
    assert roomapi.offset_at({}, 5) == (0, 0, 0)


def test_room_map_pose_shift():
    doc = {"version": 1, "pan_step": 90, "tilt_step": 60, "min_floor": 0.02, "stops": {}, "places": {},
           "poses": [{"pan": 1800, "tilt": 420, "floor_share": 0.5, "tiles": [4]}, {"pan": 3600, "tilt": 540, "floor_share": 0.0, "tiles": []}]}
    m = RoomMap(doc)
    assert m.floor_share(1800, 420) == 0.5
    doc["pose_shift"] = [160, 79]
    m = RoomMap(doc)
    assert m.floor_share(1960, 499) == 0.5          # the same direction, now 160 / 79 steps further
    assert m.floor_share(3760, 619) == 0.0 and not m.has_floor(3760, 619)


if __name__ == "__main__":
    for n, fn in list(globals().items()):
        if n.startswith("test_"):
            fn()
            print("ok", n)
