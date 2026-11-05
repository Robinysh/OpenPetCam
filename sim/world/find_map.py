"""The room model, reduced to what a search needs: where she is likely to be at this hour, which camera views can
actually see those places, and which parts of each view show floor.

For a grid of camera poses (pan, tilt in motor steps) it writes:
  floor_share   share of the frame showing floor (class 1 of the fused surfels, weighted by solid angle); a view
                with almost no floor (a wall, the ceiling, the underside of the counter) cannot show a dog
  tiles         which of the 3 x 3 half-size tiles of the frame contain floor: the server runs its detector on
                those tiles zoomed (a curled-up dog is too few pixels in the whole 640 x 360 frame)
  stops[h]      for each hour of the day, up to K stops chosen greedily by the probability of her position they
                add (dwell by hour, smoothed, seen only where the model says nothing nearer blocks the view)
  places[h]     the most likely named places for that hour and their probability, so a failed search can say
                "probably at the bowls, not confirmed" instead of a flat "not found"
Probabilities come from the presence layer (every detection of her, weighted by how long she stayed). With
`exclude_day` the map is built without that day's detections, so a replay of that day is judged on a map that
did not see it (find_bench --replay).
    python3 -m sim.world.find_map eval/world_remap1 [out.json]
Known limit (2026-11-05): searches are read in raw pan/tilt. After a re-mount (sim/world/remount.py) a rebuild mixes
poses from two mounts; until searches are converted through models/room/room.json's mount_offsets, rebuild only from
searches after the latest re-mount, or re-run remount.py on the rebuilt map."""
import json
import os
import sys
import time

import numpy as np

from .geometry import az_el, directions

CELL = 0.10                  # floor grid, metres
BLUR_M = 0.20                # position uncertainty of one detection (floor point from a box bottom)
HOUR_BLUR = 1.0              # hours: her habits at 14:00 inform 13:00 and 15:00
PRIOR_MIX = 0.3              # share of the all-day distribution mixed into each hour (hours with few detections)
PAN_STEP, TILT_STEP = 90, 60
FRAME = (1920, 1080)
EDGE = 0.04                  # a cell this close to the frame edge does not count as seen (a dog there is cut off)
MIN_FLOOR = 0.02             # views with less floor than this are skipped
MIN_TILE_FLOOR = 0.03        # a tile is worth zooming when this share of it is floor
K_STOPS = 8
COVER = 0.90                 # stop adding stops once this share of the visible probability is covered


def local_hour(t):
    return time.localtime(t).tm_hour


def load(world):
    from .sensors.base import Context
    ctx = Context(world)
    m = json.load(open(os.path.join(ctx.model, "manifest.json")))
    z = np.load(os.path.join(ctx.model, "surfels.npz"))
    dog = next(x for x in m["layers"] if x["id"] == "presence.dog")["data"]["rows"]
    return ctx, m, z, dog


def dwell_grid(dog, lo, shape, exclude_day=None):
    """Hourly probability of her position on the floor grid: [24, nz, nx], each hour summing to 1."""
    from scipy.ndimage import gaussian_filter
    G = np.zeros((24,) + shape)
    for t, x, z, cut, moving, dwell in dog:
        if exclude_day and time.strftime("%Y-%m-%d", time.localtime(t)) == exclude_day:
            continue
        i, j = int((x - lo[0]) / CELL), int((z - lo[1]) / CELL)
        if 0 <= j < shape[0] and 0 <= i < shape[1]:
            G[local_hour(t), j, i] += max(float(dwell or 0), 1.0)
    G = gaussian_filter(G, (HOUR_BLUR, BLUR_M / CELL, BLUR_M / CELL), mode=("wrap", "constant", "constant"))
    allday = G.sum(0)
    allday /= max(allday.sum(), 1e-12)
    out = np.empty_like(G)
    for h in range(24):
        s = G[h].sum()
        out[h] = (1 - PRIOR_MIX) * (G[h] / s if s > 0 else allday) + PRIOR_MIX * allday
    return out


def room_range(R, d):
    """Surfel range along floor-cell directions (nan where the model has nothing)."""
    H, W = R.shape
    az, el = az_el(d)
    i = ((az + 180) / 360 * W).astype(int) % W
    j = np.clip(((90 - el) / 180 * H).astype(int), 0, H - 1)
    return R[j, i]


def build(world, exclude_day=None):
    ctx, m, z, dog = load(world)
    cam = ctx.camera
    h = m["room"]["floor_height_m"]
    xs, zs = np.array([r[1] for r in dog]), np.array([r[2] for r in dog])
    lo = (float(np.percentile(xs, 0.5)) - 1.0, float(np.percentile(zs, 0.5)) - 1.0)
    hi = (float(np.percentile(xs, 99.5)) + 1.0, float(np.percentile(zs, 99.5)) + 1.0)
    shape = (int((hi[1] - lo[1]) / CELL) + 1, int((hi[0] - lo[0]) / CELL) + 1)
    P = dwell_grid(dog, lo, shape, exclude_day)                              # [24, nz, nx]
    gz, gx = np.mgrid[0:shape[0], 0:shape[1]]
    pts = np.stack([lo[0] + (gx + 0.5) * CELL, np.full(gx.shape, -h), lo[1] + (gz + 0.5) * CELL], -1).reshape(-1, 3)
    r = np.linalg.norm(pts, axis=1)
    d = pts / r[:, None]
    rr = room_range(z["range_m"], d)
    unblocked = ~(np.isfinite(rr) & (rr < r * 0.9))                          # nothing nearer than the floor point
    Pf = P.reshape(24, -1)
    # floor share per pose: surfel directions in the frame, weighted by solid angle (cos el)
    cls = z["class"]
    Hs, Ws = cls.shape
    D = directions(Ws, Hs).reshape(-1, 3)
    _, el = az_el(D)
    wgt = np.cos(np.radians(el))
    floor = (cls.reshape(-1) == 1)
    seen = (cls.reshape(-1) > 0)
    xm, ym = 3700, 1000
    poses, info = [], {}
    for pan in range(0, xm + 1, PAN_STEP):
        for tilt in range(0, ym + 1, TILT_STEP):
            u, v, ok = cam.project(pan, tilt, D, *FRAME)
            sk = ok & seen
            fs = float((wgt * (sk & floor)).sum() / max((wgt * sk).sum(), 1e-9)) if sk.any() else 0.0
            tiles = []
            for ty in range(3):
                for tx in range(3):
                    inside = sk & (u >= tx * 0.25 * FRAME[0]) & (u < (tx * 0.25 + 0.5) * FRAME[0]) & \
                        (v >= ty * 0.25 * FRAME[1]) & (v < (ty * 0.25 + 0.5) * FRAME[1])
                    share = float((wgt * (inside & floor)).sum() / max((wgt * inside).sum(), 1e-9)) if inside.any() else 0.0
                    if share >= MIN_TILE_FLOOR:
                        tiles.append(ty * 3 + tx)
            cu, cv, cok = cam.project(pan, tilt, d, *FRAME)
            vis = cok & unblocked & (cu > EDGE * FRAME[0]) & (cu < (1 - EDGE) * FRAME[0]) & \
                (cv > EDGE * FRAME[1]) & (cv < (1 - EDGE) * FRAME[1])
            info[(pan, tilt)] = vis
            poses.append({"pan": pan, "tilt": tilt, "floor_share": round(fs, 3), "tiles": tiles})
    # greedy stops per hour: the pose adding the most not-yet-covered probability, among views with floor
    hidden = [round(float(Pf[hh][~unblocked].sum()), 3) for hh in range(24)]
    stops, places = {}, {}
    usable = [p for p in poses if p["floor_share"] >= MIN_FLOOR]
    for hh in range(24):
        left = Pf[hh].copy()
        visible_total = float(Pf[hh][unblocked].sum())
        chosen, covered = [], 0.0
        for _ in range(K_STOPS):
            best = max(usable, key=lambda p: float(left[info[(p["pan"], p["tilt"])]].sum()))
            gain = float(left[info[(best["pan"], best["tilt"])]].sum())
            if gain < 0.01:
                break
            chosen.append([best["pan"], best["tilt"], round(gain, 3)])
            left[info[(best["pan"], best["tilt"])]] = 0
            covered += gain
            if covered >= COVER * visible_total:
                break
        stops[hh] = chosen
        places[hh] = top_places(m["room"]["objects"], d, Pf[hh], unblocked)
    return {"version": 1, "built": time.strftime("%Y-%m-%d %H:%M"), "exclude_day": exclude_day,
            "calib_sha": ctx.twin.get("calib_sha") if hasattr(ctx, "twin") and isinstance(ctx.twin, dict) else None,
            "pan_step": PAN_STEP, "tilt_step": TILT_STEP, "min_floor": MIN_FLOOR,
            "poses": poses, "stops": stops, "places": places, "hidden_share": hidden,
            "_grid": {"lo": lo, "shape": shape, "P": Pf, "unblocked": unblocked, "dirs": d, "info": info}}


def top_places(objects, d, p, unblocked, n=3):
    """Name the probability mass by the nearest solid object (angular distance to its window), within 25 degrees;
    the rest is 'the open floor'. Thin objects (a TV screen) are not places a dog lies at."""
    az, el = az_el(d)
    best = np.full(len(p), 25.0)
    names = np.array(["the open floor"] * len(p), dtype=object)
    for o in objects:
        if o.get("kind") == "thin":
            continue
        a0, a1 = o["az"]; e0, e1 = o["el"]
        da = np.maximum(0, np.maximum(a0 - az, az - a1)); de = np.maximum(0, np.maximum(e0 - el, el - e1))
        dist = np.hypot(da, de)
        closer = dist < best
        best[closer] = dist[closer]
        names[closer] = o["label"].split(" (")[0]
    out = {}
    for nm in set(names):
        msk = names == nm
        out[nm] = (float(p[msk].sum()), float(p[msk & ~unblocked].sum()))
    rows = sorted(out.items(), key=lambda kv: -kv[1][0])[:n]
    return [{"place": k, "p": round(v[0], 3), "hidden": round(v[1], 3)} for k, v in rows]


def save(res, out):
    json.dump({k: v for k, v in res.items() if not k.startswith("_")}, open(out, "w"), indent=0)


if __name__ == "__main__":
    a = sys.argv[1:]
    res = build(a[0])
    out = a[1] if len(a) > 1 else os.path.join(a[0], "model", "room_map.json")
    save(res, out)
    fs = [p["floor_share"] for p in res["poses"]]
    print(f"{len(fs)} poses; {sum(f < MIN_FLOOR for f in fs)} show no floor; written {out}")
    for hh in (3, 9, 14, 20):
        print(hh, "stops", res["stops"][hh], "places", res["places"][hh])
