"""What the room model tells a search (built offline by sim/world/find_map.py, deployed as models/room_map.json).

  stops(hour)          the views most likely to show her at this hour, best first (dwell by hour x what each
                       view can see unblocked)
  floor_share(x, y)    share of the view at motor pose (x, y) that is floor; a view without floor cannot show a dog
  tiles(x, y)          which of the 3 x 3 half-size tiles of that view contain floor (zoomed detection runs on them)
  places(hour)         the likeliest named places at this hour, for an honest "not found" message
Every lookup takes the nearest grid pose. A missing or broken file means no map: the search runs as before."""
import json

TILES = [(tx * 0.25, ty * 0.25, tx * 0.25 + 0.5, ty * 0.25 + 0.5) for ty in range(3) for tx in range(3)]


class RoomMap:
    def __init__(self, doc):
        self.doc = doc
        self.ps, self.ts = int(doc["pan_step"]), int(doc["tilt_step"])
        # after a re-mount the grid was measured in the old mount's poses: look up (x - dx, y - dy) (sim/world/remount.py)
        self.dx, self.dy = (int(v) for v in (doc.get("pose_shift") or (0, 0)))
        self.min_floor = float(doc.get("min_floor", 0.02))
        self.grid = {(int(p["pan"]), int(p["tilt"])): p for p in doc["poses"]}
        self.stops_by_hour = {int(h): [(int(s[0]), int(s[1]), float(s[2])) for s in v] for h, v in doc["stops"].items()}
        self.places_by_hour = {int(h): v for h, v in doc.get("places", {}).items()}

    @classmethod
    def load(cls, path):
        """The map, or None when the file is missing or unreadable (never raises: no map = search as before)."""
        try:
            with open(path) as f:
                doc = json.load(f)
            if doc.get("version") != 1:
                return None
            return cls(doc)
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _pose(self, x, y):
        x, y = x - self.dx, y - self.dy
        k = (int(round(x / self.ps)) * self.ps, int(round(y / self.ts)) * self.ts)
        return self.grid.get(k)

    def floor_share(self, x, y):
        p = self._pose(x, y)
        return None if p is None else float(p["floor_share"])

    def has_floor(self, x, y):
        f = self.floor_share(x, y)
        return f is None or f >= self.min_floor          # unknown pose: do not skip it

    def tiles(self, x, y):
        """Tile boxes (normalised x0, y0, x1, y1) worth zooming at this pose; all nine when the pose is unknown."""
        p = self._pose(x, y)
        return TILES if p is None else [TILES[i] for i in p["tiles"]]

    def stops(self, hour):
        return list(self.stops_by_hour.get(int(hour) % 24, []))

    def places(self, hour):
        return list(self.places_by_hour.get(int(hour) % 24, []))

