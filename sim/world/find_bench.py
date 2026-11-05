"""Find, measured on real searches with the room model (step 1: truth, visibility, detection rate).

For every recorded search (server searches/<day>/<id>/search.json: frames with pose and verdict):
  truth      where the dog was: the confirmed box of a successful search, else her next detection if it comes within
             60 s of the search ending (she was resting there through the search); otherwise unknown
  visible    for each frame: is her floor cell inside that frame AND unblocked in the room model (fused surface and
             object boxes; a ray reaching her must not hit anything nearer)
  outcome    found / missed_in_view (visible in >= 1 settled frame, never detected) / hidden (never visible to any frame
             of the search) / unknown
  p_detect   per settled frame where she was visible: did the detector see her (day / night separately)
This is the measurement every later policy is judged against, so it uses only real frames and real outcomes.
    python3 -m sim.world.find_bench eval/world_remap1"""
import glob
import json
import math
import os
import sys

import numpy as np

from .geometry import floor_hit


def load_episodes(world):
    E = []
    for f in sorted(glob.glob(os.path.join(world, "raw", "find", "*", "*", "search.json"))):
        j = json.load(open(f))
        E.append({"id": os.path.relpath(os.path.dirname(f), os.path.join(world, "raw", "find")), **j})
    return E


class Room:
    """Visibility of a floor point from a pose, from the fused surfels (range per direction, boxes included)."""

    def __init__(self, world, ctx):
        z = np.load(os.path.join(world, "model", "surfels.npz"))
        self.R = z["range_m"]
        self.H, self.W = self.R.shape
        self.ctx, self.cam = ctx, ctx.camera
        self.h = ctx.room.floor_height

    def direction_range(self, d):
        az = math.degrees(math.atan2(d[0], d[2])); el = math.degrees(math.asin(max(-1, min(1, d[1]))))
        i, j = int((az + 180) / 360 * self.W) % self.W, min(self.H - 1, int((90 - el) / 180 * self.H))
        r = self.R[max(0, j - 1):j + 2, max(0, i - 1):i + 2]
        r = r[np.isfinite(r)]
        return float(np.median(r)) if len(r) else float("nan")

    def visible(self, q, pan, tilt, off=(0, 0, 0)):
        """q: floor point (x, z). Inside the frame and nothing in front of it (10% tolerance)."""
        p = np.array([q[0], -self.h, q[1]])
        r = float(np.linalg.norm(p))
        d = p / r
        u, v, ok = self.cam.project(pan, tilt, d[None], 1920, 1080, off=off)
        if not ok[0] or not (0.04 < u[0] / 1920 < 0.96 and 0.04 < v[0] / 1080 < 0.96):
            return False
        rr = self.direction_range(d)
        return not (np.isfinite(rr) and rr < r * 0.9)

    def floor_point(self, pan, tilt, box, off=(0, 0, 0)):
        x0, y0, x1, y1 = box
        d = self.cam.rays(pan, tilt, (x0 + x1) / 2 * 1920, y1 * 1080, off=off)
        p = floor_hit(d, self.h)
        return (float(p[0]), float(p[2])) if np.all(np.isfinite(p)) else None


def truth_and_visibility(world):
    from .registration import at, corrections
    from .room import fuse
    from .sensors.acoustic import ActiveSonar
    from .sensors.base import Context
    from .sensors.visible import snapshots
    ctx = Context(world)
    ctx.room = fuse(ctx, ActiveSonar(ctx).layers()[0])
    room = Room(world, ctx)
    corr = corrections(ctx, snapshots(ctx), os.path.join(ctx.model, "pano_visible_sweep.jpg"))
    m = json.load(open(os.path.join(ctx.model, "manifest.json")))
    dog = sorted(next(x for x in m["layers"] if x["id"] == "presence.dog")["data"]["rows"], key=lambda r: r[0])
    dt = np.array([r[0] for r in dog])
    out, frames_vis = [], []
    for e in load_episodes(world):
        fr = [f for f in e.get("frames", []) if f.get("pos")]
        q, src = None, None
        if e.get("found"):
            fv = [f for f in e["frames"] if f.get("verified")]
            pf = next((f for f in reversed(e["frames"]) if f.get("pos")), None)
            if fv and pf:
                q = room.floor_point(*pf["pos"], fv[-1]["verified"], off=at(corr, fv[-1]["t"]))
                src = "found box"
        else:
            k = int(np.searchsorted(dt, e["t1"]))
            gap = dog[k][0] - e["t1"] if k < len(dog) else 1e9
            prev_gap = e["t0"] - dog[k - 1][0] if k > 0 else 1e9
            if gap < 300 and not dog[k][4] and not dog[k][3]:
                q, src = (dog[k][1], dog[k][2]), f"next detection {gap:.0f} s after, resting"
            elif gap > 1800 and prev_gap > 600:
                src = "probably out (no detection for 30 min after, 10 min before)"
        vis = []
        for f in fr:
            if q is None:
                vis.append(None); continue
            v = room.visible(q, *f["pos"], off=at(corr, f["t"]))
            vis.append(v)
            if f.get("settled") and v:
                frames_vis.append({"night": bool(e.get("night")), "detected": f.get("verified") is not None,
                                   "judged": len(f.get("judged") or [])})
        if q is None:
            outcome = "probably_out" if src and src.startswith("probably out") else "unknown"
        elif e.get("found"):
            outcome = "found"
        elif any(v for v, f in zip(vis, fr) if f.get("settled")):
            outcome = "missed_in_view"
        else:
            outcome = "hidden"
        out.append({"id": e["id"], "t0": e["t0"], "dur": e["t1"] - e["t0"], "found": bool(e.get("found")), "night": bool(e.get("night")),
                    "truth": q, "truth_from": src, "visible_frames": int(sum(1 for v in vis if v)), "frames": len(fr), "outcome": outcome})
    pd = {}
    for night in (False, True):
        s = [f for f in frames_vis if f["night"] == night]
        if s:
            pd["night" if night else "day"] = {"frames": len(s), "p_detect": round(sum(f["detected"] for f in s) / len(s), 3)}
    res = {"episodes": out, "p_detect": pd}
    json.dump(res, open(os.path.join(world, "raw", "find", "bench_truth.json"), "w"), indent=1, default=float)
    return res


def check_labels(world, res):
    """The owner's / reviewer's outcome labels (eval/searches/outcomes.jsonl) against this truth."""
    p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(world))), "eval", "searches", "outcomes.jsonl")
    if not os.path.exists(p):
        p = os.path.join(os.path.dirname(os.path.abspath(world)), "searches", "outcomes.jsonl")
    if not os.path.exists(p):
        return None
    lab = {json.loads(l)["search"]: json.loads(l)["outcome"] for l in open(p)}
    rows = [(lab[e["id"]], e["outcome"]) for e in res["episodes"] if e["id"] in lab]
    return rows


SPEED = 250.0                # QUIET_SPEED, steps/s (server/app.py)


def move_s(a, b):
    return max(abs(a[0] - b[0]), abs(a[1] - b[1])) / SPEED


def merge_stops(first, then, fw=686, fh=429):
    """`first`, then each stop of `then` whose view is not mostly inside one already chosen (half a frame)."""
    out = list(first)
    for s in then:
        if all(abs(s[0] - o[0]) > fw / 2 or abs(s[1] - o[1]) > fh / 2 for o in out):
            out.append(tuple(s))
    return out


def replay(world, look_s=None):
    """Counterfactual replay, no A/B: for every recorded search where we know where she was, which stop of each
    order first has her in view, and how long it takes to get there (move at quiet speed + settle + look). The
    stop order of the room map is built WITHOUT the searched day's detections. The look constant is fitted so the
    model reproduces the real time between stops of the current order. Caveat: a 'found' search is biased toward
    the current order (it found her where it looked); the unbiased cases are the 'resting, next detection' ones."""
    from .find_map import build, local_hour
    from .registration import at, corrections
    from .room import fuse
    from .sensors.acoustic import ActiveSonar
    from .sensors.base import Context
    from .sensors.visible import snapshots
    ctx = Context(world)
    ctx.room = fuse(ctx, ActiveSonar(ctx).layers()[0])
    room = Room(world, ctx)
    corr = corrections(ctx, snapshots(ctx), os.path.join(ctx.model, "pano_visible_sweep.jpg"))
    truth = {e["id"]: e for e in json.load(open(os.path.join(world, "raw", "find", "bench_truth.json")))["episodes"]}
    pol = json.load(open(os.path.join(world, "raw", "find", "follow_policy.json")))
    learned = [tuple(x) for x in pol["search_stops"]]
    E = {e["id"]: e for e in load_episodes(world)}
    # fit the per-stop look time on the real searches: gap between settled stops minus the modelled move
    gaps = []
    for e in E.values():
        st = [f for f in e["frames"] if f.get("settled") and f.get("pos")]
        for a, b in zip(st, st[1:]):
            gaps.append(b["t"] - a["t"] - move_s(a["pos"], b["pos"]))
    look_s = look_s if look_s is not None else float(np.median(gaps))
    maps = {}
    rows = []
    for eid, tr in truth.items():
        if tr["truth"] is None or eid not in E:
            continue
        e = E[eid]
        day = eid.split("/")[0]
        if day not in maps:
            maps[day] = build(world, exclude_day=day)
        mp = maps[day]
        start = next((f["pos"] for f in e["frames"] if f.get("pos")), None)
        if start is None:
            continue
        off = at(corr, e["t0"])
        here = room.visible(tr["truth"], *start, off=off)
        new = [(s[0], s[1]) for s in mp["stops"][local_hour(e["t0"])]]
        combined = merge_stops(new, learned)
        res = {"id": eid, "src": tr["truth_from"], "found": tr["found"], "in_view_at_start": bool(here)}
        for name, order in (("learned", learned), ("room_map", new), ("map_then_learned", combined)):
            t, pos, hit = 0.0, start, None
            for k, s in enumerate(order):
                t += move_s(pos, s) + look_s
                pos = s
                if room.visible(tr["truth"], *s, off=off):
                    hit = (k + 1, round(t, 1))
                    break
            res[name] = hit
        real = None                                   # real: first recorded frame with her in view
        for f in e["frames"]:
            if f.get("pos") and room.visible(tr["truth"], *f["pos"], off=at(corr, f["t"])):
                real = round(f["t"] - e["t0"], 1)
                break
        res["real_first_in_view_s"] = real
        rows.append(res)
    out = {"look_s": round(look_s, 2), "n": len(rows), "rows": rows}
    for name in ("learned", "room_map", "map_then_learned"):
        pending = [r for r in rows if not r["in_view_at_start"]]
        hits = [r[name] for r in pending if r[name]]
        out[name] = {"searched": len(pending), "in_view_at_a_stop": len(hits),
                     "median_s": float(np.median([h[1] for h in hits])) if hits else None,
                     "median_stops": float(np.median([h[0] for h in hits])) if hits else None,
                     "stops_in_order": len(learned) if name == "learned" else None}
    json.dump(out, open(os.path.join(world, "raw", "find", "replay.json"), "w"), indent=1, default=float)
    return out


if __name__ == "__main__":
    import collections
    w = sys.argv[1]
    if "--replay" in sys.argv:
        r = replay(w)
        print({k: v for k, v in r.items() if k != "rows"})
        sys.exit(0)
    res = truth_and_visibility(w)
    E = res["episodes"]
    print(len(E), "searches; outcomes:", dict(collections.Counter(e["outcome"] for e in E)))
    print("detection per settled frame where she was visible:", res["p_detect"])
    lab = check_labels(w, res)
    if lab:
        print("against hand labels:", collections.Counter(lab).most_common())
