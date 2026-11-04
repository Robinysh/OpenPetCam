"""tools/coverage.py: the map must show the sweep the tracker really drives, and its numbers must be right.
    python3 tests/server/coverage_test.py"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "tools"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "server"))
import coverage as C  # noqa: E402
import tracker as T  # noqa: E402

XM, YM, FW, FH, HOME, PRESET = 3700, 1000, 733.0, 424.0, (1850, 500), (1019, 534)


class EmptyRoom:
    """A camera in a room with no dog: records every goto. Moves finish at once."""
    def __init__(self, pos):
        self.pos, self.gotos, self.t = pos, [], 0.0
    def position(self): return self.pos
    def limits(self): return XM, YM
    def frame_steps(self): return FW, FH
    def home(self): return HOME
    def presets(self): return [{"x": PRESET[0], "y": PRESET[1], "name": "Corridor"}]
    def moved_at(self): return -1e9
    def busy(self): return False
    def wait_idle(self): return self.pos
    def stop(self): pass
    def goto(self, x, y):
        self.gotos.append((x, y)); self.pos = (x, y)
    def aim(self, fx, fy): pass


def test_plan_is_what_search_drives():
    for start in ((2600, 300), PRESET):
        cam = EmptyRoom(start)
        clock = [0.0]
        def tick_clock():
            clock[0] += 0.05
            return clock[0]
        tr = T.Tracker(cam, find_pet=lambda t: None, clock=tick_clock, log=lambda *a: None)
        tr.last_move = -1e9
        tr.search("test")
        driven = [g for g in cam.gotos[:-1]]                       # the last goto parks at the rest view
        planned = [s["b"] for s in C.plan(XM, YM, FW, FH, HOME, [PRESET], start) if s["kind"] != "look"]
        assert driven == planned, (start, driven, planned)


def test_numbers():
    segs = C.plan(XM, YM, FW, FH, HOME, [PRESET], PRESET)
    st = C.stats(segs, XM, YM, FW, FH)
    # rows at 500 +- 0.55 * 424 see tilt 55..945 of the reachable -172..1172: two bands are never looked at
    assert len(st["unseen_tilt"]) == 2 and abs(st["unseen_tilt"][0][1] - 55) < 15 and abs(st["unseen_tilt"][1][0] - 945) < 15, st
    assert 0.6 < st["covered"] < 0.7, st
    wide = C.stats(C.plan(XM, YM, FW, FH, HOME, [PRESET], PRESET, row_step=1.0), XM, YM, FW, FH)
    assert wide["covered"] > st["covered"] + 0.2 and wide["twice"] < st["twice"], wide
    gap = C.stats(C.plan(XM, YM, FW, FH, HOME, [PRESET], PRESET, row_step=1.15), XM, YM, FW, FH)
    assert any(a > 0 and b < YM for a, b in gap["unseen_tilt"]), gap    # rows too far apart: a gap between them
    assert T.ROW_STEP == 0.55                                          # a what-if never leaks into the tracker


def test_svg():
    out = C.svg(C.plan(XM, YM, FW, FH, HOME, [PRESET], PRESET), XM, YM, FW, FH, frames=[(1019, 534, "a&b.jpg")])
    assert out.startswith("<svg") and out.endswith("</svg>") and 'href="a&amp;b.jpg"' in out


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
