"""The Room tab's data: your room bundle (models/room/, written by sim/world/app_bundle.py from your room model) and
where the pet was on its floor, from the watcher's own records (obs: detection boxes, track: camera poses).

A detection's floor point is its box's bottom centre (her feet) along the calibrated ray at that moment's pose, met
with the floor plane; the same rule as the room model's presence layer, so the app and the model agree. A detection
re-logged unchanged (a cached box) extends the previous one's dwell instead of counting again. Everything here is the
inside of a home: it is served only to signed-in devices (app.py), and the bundle never leaves your server."""
import bisect
import json
import os

from roomcam import Camera, floor_point

ROOM_DIR = os.environ.get("PETCAM_ROOM_DIR", "/opt/petcam/models/room")
MAX_POSE_GAP = 5.0          # s: a detection needs a known camera pose this close in time
_cache = {"mtime": None, "doc": None, "cam": None}


def bundle():
    """(room document, camera) or (None, None) when no room bundle is installed or it is unreadable."""
    p = os.path.join(ROOM_DIR, "room.json")
    try:
        m = os.path.getmtime(p)
    except OSError:
        _cache.update(mtime=None, doc=None, cam=None)
        return None, None
    if m != _cache["mtime"]:
        try:
            with open(p) as f:
                doc = json.load(f)
            cam = Camera(doc["calib"]) if doc.get("version") == 1 else None
        except (OSError, ValueError, KeyError, TypeError):
            doc, cam = None, None
        _cache.update(mtime=m, doc=doc if cam else None, cam=cam)
    return _cache["doc"], _cache["cam"]


def info():
    doc, _ = bundle()
    return {"available": doc is not None, **(doc or {})}


def texture(name):
    """Path of one of the bundle's textures (only names the bundle lists), or None."""
    doc, _ = bundle()
    if not doc or name not in (doc.get("textures") or {}).values():
        return None
    p = os.path.join(ROOM_DIR, os.path.basename(name))
    return p if os.path.isfile(p) else None


def poses(con, since, until):
    rows = con.execute("SELECT ts, pan, tilt FROM track WHERE ts >= ? AND ts <= ? ORDER BY ts",
                       (since - 120, until + 120)).fetchall()
    return [r[0] for r in rows], [(r[1], r[2]) for r in rows]


def pose_at(tt, pp, ts):
    """The camera's pose at ts: the nearest track row within MAX_POSE_GAP, and only while it was not moving (its
    neighbours share the pose); None otherwise."""
    if not tt:
        return None
    k = bisect.bisect_left(tt, ts)
    cand = [i for i in (k - 1, k) if 0 <= i < len(tt)]
    i = min(cand, key=lambda i: abs(tt[i] - ts))
    if abs(tt[i] - ts) > MAX_POSE_GAP:
        return None
    if pp[max(i - 1, 0)] != pp[min(i + 1, len(pp) - 1)]:
        return None
    return pp[i]


def positions(con, since, until):
    """Where she was between since and until: rows [t, x, z, moving, dwell_s, cut_off], oldest first."""
    doc, cam = bundle()
    if not doc:
        return []
    h = float(doc["floor_height_m"])
    off = tuple(doc.get("mount_offset_deg") or (0, 0, 0))
    tt, pp = poses(con, since, until)
    out, last_box, last_ts = [], None, None
    for ts, box, present, moving in con.execute(
            "SELECT ts, box, present, moving FROM obs WHERE ts >= ? AND ts <= ? AND box IS NOT NULL AND box <> '' "
            "ORDER BY ts", (since, until)):
        if not present:
            last_box = None
            continue
        if box == last_box:                                     # a cached detection re-logged
            if out and ts - last_ts < 30:
                out[-1][4] = round(ts - out[-1][0] + 1.5, 1)
            last_ts = ts
            continue
        last_box, last_ts = box, ts
        pose = pose_at(tt, pp, ts)
        if not pose:
            continue
        try:
            b = json.loads(box)
        except ValueError:
            continue
        p = floor_point(cam, pose[0], pose[1], b, h, off)
        if p is None:
            continue
        out.append([round(ts, 1), round(p[0], 3), round(p[1], 3), int(moving or 0), 1.5, int(b[3] > 0.97)])
    return out
