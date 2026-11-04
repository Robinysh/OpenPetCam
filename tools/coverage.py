#!/usr/bin/env python3
"""Draw what a Follow/Find search actually looks at: a view-overlap map of the sweep, plus how much of the reachable
pan/tilt range it covers, how much it looks at twice, and how far and how long the motor runs. Offline, no camera.

    python3 tools/coverage.py --home 1850,500 --preset 1200,500 --out eval/coverage.svg
    python3 tools/coverage.py ... --search eval/searches/<day>/<HHMMSS>      # put that search's stop frames on the map
    python3 tools/coverage.py ... --sweep                                    # compare other row spacings

The plan is the tracker's own (server/tracker.py rows() and sweep_plan()), so the map changes when the code does.
Geometry: the camera at (x, y) sees x +- PAN_STEPS_PER_FRAME/2 and y +- TILT_STEPS_PER_FRAME/2 motor steps. A pan
checks every frame on the way, so a pan from a to b sees the whole strip between them. Times use the quiet motor
speed and ignore acceleration. The map is in motor steps, not a picture of the room: pass --search to see the room.
"""
import argparse
import html
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "server"))
import tracker as T  # noqa: E402

QUIET_SPEED = 250.0      # steps/s, server/app.py QUIET_SPEED
LOOK_S = T.SETTLE + T.LOOK_FRAMES * 0.2   # a stop: settle, then up to LOOK_FRAMES frames at 5 fps
MARGIN = 40              # the app's tilt margin (server/app.py TILT_MARGIN)


class PlanCam:
    """Just enough camera for Tracker.rows() / sweep_plan()."""
    def __init__(self, xm, ym, fw, fh, home, presets, pos):
        self.xm, self.ym, self.fw, self.fh, self._home, self._presets, self.pos = xm, ym, fw, fh, home, presets, pos
    def limits(self): return self.xm, self.ym
    def frame_steps(self): return self.fw, self.fh
    def home(self): return self._home
    def presets(self): return [{"x": x, "y": y} for x, y in self._presets]
    def position(self): return self.pos


def plan(xm, ym, fw, fh, home, presets, start, row_step=None, heading_right=False):
    """The search's moves in order as segments {"kind": "look"|"stop"|"pan", "a": (x, y), "b": (x, y)}.
    Mirrors Tracker.search(): look where it is, visit the saved views it doesn't already see, then the zigzag."""
    old = T.ROW_STEP
    if row_step is not None:
        T.ROW_STEP = row_step
    try:
        cam = PlanCam(xm, ym, fw, fh, home, presets, start)
        tr = T.Tracker(cam, find_pet=lambda t: None, clock=lambda: 0.0, log=lambda *a: None)
        tr.last_seen_c = (1.0, 0.0) if heading_right else None
        targets = tr.sweep_plan(start[1])
    finally:
        T.ROW_STEP = old
    segs, here = [{"kind": "look", "a": start, "b": start}], start
    for x, y in presets:
        if abs(x - start[0]) < fw * 0.4 and abs(y - start[1]) < fw * 0.25:
            continue
        segs.append({"kind": "stop", "a": here, "b": (x, y)})
        here = (x, y)
    for x, y in targets:
        if abs(here[0] - x) < 5 and abs(here[1] - y) < 5:
            continue
        segs.append({"kind": "pan", "a": here, "b": (x, y)})
        here = (x, y)
    return segs


def footprints(segs, fw, fh):
    """Rectangles (x0, x1, y0, y1) the search saw. A stop sees one frame at its end. A pan sees the strip it swept; a
    pan that also changes tilt (the step between rows) is counted at its end only, which is conservative."""
    out = []
    for s in segs:
        (ax, ay), (bx, by) = s["a"], s["b"]
        if s["kind"] == "pan" and abs(ay - by) < 1:
            out.append((min(ax, bx) - fw / 2, max(ax, bx) + fw / 2, by - fh / 2, by + fh / 2))
        else:
            out.append((bx - fw / 2, bx + fw / 2, by - fh / 2, by + fh / 2))
    return out


def domain(xm, ym, fw, fh):
    """Everything the camera can show: every pan, every tilt within the app's margin, plus half a frame each side."""
    return (-fw / 2, xm + fw / 2, MARGIN - fh / 2, ym - MARGIN + fh / 2)


def stats(segs, xm, ym, fw, fh, cell=10.0):
    """covered: share of the reachable range seen at least once; twice: share seen 2+ times (overlap);
    unseen_tilt: tilt bands (steps) no part of the sweep looks at; travel: motor steps; seconds: estimated duration."""
    dx0, dx1, dy0, dy1 = domain(xm, ym, fw, fh)
    rects = footprints(segs, fw, fh)
    nx, ny = int(math.ceil((dx1 - dx0) / cell)), int(math.ceil((dy1 - dy0) / cell))
    once = twice = 0
    row_seen = [False] * ny
    for j in range(ny):
        y = dy0 + (j + 0.5) * cell
        for i in range(nx):
            x = dx0 + (i + 0.5) * cell
            n = sum(1 for r in rects if r[0] <= x <= r[1] and r[2] <= y <= r[3])
            once += n >= 1
            twice += n >= 2
            row_seen[j] = row_seen[j] or n >= 1
    bands, j = [], 0
    while j < ny:
        if not row_seen[j]:
            k = j
            while k < ny and not row_seen[k]:
                k += 1
            bands.append((round(dy0 + j * cell), round(dy0 + k * cell)))
            j = k
        else:
            j += 1
    travel = sum(abs(s["b"][0] - s["a"][0]) + abs(s["b"][1] - s["a"][1]) for s in segs)
    seconds = sum(max(abs(s["b"][0] - s["a"][0]), abs(s["b"][1] - s["a"][1])) / QUIET_SPEED + LOOK_S for s in segs)
    total = nx * ny
    return {"covered": once / total, "twice": twice / total, "unseen_tilt": bands, "travel": round(travel),
            "seconds": round(seconds, 1), "moves": sum(1 for s in segs if s["kind"] != "look")}


def svg(segs, xm, ym, fw, fh, frames=(), scale=0.25):
    """The map: grey = reachable, each view a translucent blue box (darker = seen more often), the path in orange,
    numbered in order. frames: [(x, y, href)] stop frames from a recorded search, drawn where they were taken.
    Tilt is drawn with larger y lower on the page (check on your camera which way is the floor)."""
    dx0, dx1, dy0, dy1 = domain(xm, ym, fw, fh)
    W, H = (dx1 - dx0) * scale, (dy1 - dy0) * scale
    X = lambda x: (x - dx0) * scale  # noqa: E731
    Y = lambda y: (y - dy0) * scale  # noqa: E731
    p = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W + 20:.0f}" height="{H + 60:.0f}" font-family="sans-serif" '
         f'font-size="11"><g transform="translate(10,10)">',
         f'<rect x="0" y="0" width="{W:.1f}" height="{H:.1f}" fill="#eee" stroke="#999"/>']
    for x, y, href in frames:
        p.append(f'<image href="{html.escape(href)}" x="{X(x - fw / 2):.1f}" y="{Y(y - fh / 2):.1f}" '
                 f'width="{fw * scale:.1f}" height="{fh * scale:.1f}" opacity="0.85" preserveAspectRatio="none"/>')
    for r in footprints(segs, fw, fh):
        p.append(f'<rect x="{X(r[0]):.1f}" y="{Y(r[2]):.1f}" width="{(r[1] - r[0]) * scale:.1f}" '
                 f'height="{(r[3] - r[2]) * scale:.1f}" fill="#2a6fdb" fill-opacity="0.18" stroke="#2a6fdb" stroke-opacity="0.5"/>')
    for n, s in enumerate(segs):
        (ax, ay), (bx, by) = s["a"], s["b"]
        p.append(f'<line x1="{X(ax):.1f}" y1="{Y(ay):.1f}" x2="{X(bx):.1f}" y2="{Y(by):.1f}" stroke="#e07a00" stroke-width="2"/>')
        p.append(f'<circle cx="{X(bx):.1f}" cy="{Y(by):.1f}" r="7" fill="#e07a00"/>'
                 f'<text x="{X(bx):.1f}" y="{Y(by) + 4:.1f}" text-anchor="middle" fill="#fff">{n}</text>')
    st = stats(segs, xm, ym, fw, fh)
    p.append(f'<text x="0" y="{H + 20:.0f}">covered {st["covered"]:.0%} of the reachable range, {st["twice"]:.0%} seen twice; '
             f'{st["moves"]} moves, {st["travel"]} steps, about {st["seconds"]:.0f} s at the quiet speed</text>')
    p.append(f'<text x="0" y="{H + 36:.0f}" fill="#666">pan 0..{xm} left to right, tilt {MARGIN}..{ym - MARGIN} top to '
             f'bottom; 0 = where the search started</text></g></svg>')
    return "\n".join(p)


def search_frames(d, out_path):
    """Stop frames of a recorded search (eval/searches/<day>/<HHMMSS>/search.json) as (x, y, href relative to the SVG)."""
    with open(os.path.join(d, "search.json")) as f:
        meta = json.load(f)
    base = os.path.dirname(os.path.abspath(out_path))
    return [(fr["pos"][0], fr["pos"][1], os.path.relpath(os.path.join(d, fr["file"]), base))
            for fr in meta.get("frames", []) if fr.get("pos")]


def xy(s):
    a, b = s.split(",")
    return float(a), float(b)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--xm", type=float, default=3700, help="pan steps (camera's steps_pan)")
    ap.add_argument("--ym", type=float, default=1000, help="tilt steps (camera's steps_tilt)")
    ap.add_argument("--fw", type=float, default=float(os.environ.get("PAN_STEPS_PER_FRAME", 733)))
    ap.add_argument("--fh", type=float, default=float(os.environ.get("TILT_STEPS_PER_FRAME", 424)))
    ap.add_argument("--home", type=xy, default=None, help="the camera's start pose x,y (default: the middle)")
    ap.add_argument("--preset", type=xy, action="append", default=[], help="a saved view x,y (repeatable, in order)")
    ap.add_argument("--start", type=xy, default=None, help="where the search starts (default: the first preset or home)")
    ap.add_argument("--row-step", type=float, default=None, help=f"what-if row spacing in frame heights (now {T.ROW_STEP})")
    ap.add_argument("--sweep", action="store_true", help="table of coverage and time for other row spacings")
    ap.add_argument("--search", help="a pulled search folder: draw its stop frames on the map")
    ap.add_argument("--out", default="coverage.svg")
    a = ap.parse_args(argv)
    home = a.home or (a.xm / 2, a.ym / 2)
    start = a.start or (a.preset[0] if a.preset else home)
    segs = plan(a.xm, a.ym, a.fw, a.fh, home, a.preset, start, a.row_step)
    frames = search_frames(a.search, a.out) if a.search else ()
    with open(a.out, "w") as f:
        f.write(svg(segs, a.xm, a.ym, a.fw, a.fh, frames))
    st = stats(segs, a.xm, a.ym, a.fw, a.fh)
    print(f"{a.out}: covered {st['covered']:.0%}, seen twice {st['twice']:.0%}, {st['moves']} moves, "
          f"{st['travel']} steps, ~{st['seconds']} s; tilt never seen: {st['unseen_tilt'] or 'none'}")
    if a.sweep:
        print("row_step  covered  twice  steps  seconds  tilt never seen")
        for rs in (0.4, 0.55, 0.7, 0.85, 1.0, 1.15):
            s = stats(plan(a.xm, a.ym, a.fw, a.fh, home, a.preset, start, rs), a.xm, a.ym, a.fw, a.fh)
            print(f"{rs:8.2f}  {s['covered']:7.0%}  {s['twice']:5.0%}  {s['travel']:5d}  {s['seconds']:7.1f}  {s['unseen_tilt']}")


if __name__ == "__main__":
    main()
