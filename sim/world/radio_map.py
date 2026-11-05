"""The camera's WiFi link by head direction: an electromagnetic layer of the room model.

The camera's antenna turns with its head, so the link's signal strength (RSSI, 2.4 GHz, sampled twice a second through
the server) depends on where the head points (RF-2: camera pose alone explains about two thirds of RSSI). Each RSSI
sample is tied to the head direction from the calibrated camera at that moment's pose, and the map is
    RSSI(az, el) = a + b cos(az - bearing) + c sin(el)
fitted to all samples. It is drawn only where the head actually pointed, and only if it repeats: interleaved 5-minute
blocks (A even, B odd) must give the same per-direction means (split-half correlation against a shuffle null) and a
consistent bearing. The bearing is the direction of the strongest link, usually toward the access point, but a strong
reflection can also produce it: it is recorded as unconfirmed until someone says where the router is.
    python3 -m sim.world.radio_map eval/world      -> model/surfel_radio.png, radio layer in the manifest"""
import json
import os
import sys

import numpy as np

from .geometry import az_el, directions
from .rti import load_rssi

BLOCK_S, MIN_BIN = 300.0, 30


def samples(world, rssi_paths=None):
    from .sensors.base import Context
    ctx = Context(world)
    t, v = load_rssi(rssi_paths or [os.path.join(world, "rssi.log")])
    tr = json.load(open(os.path.join(world, "track.json")))
    tt = np.array([x["ts"] for x in tr]); pan = np.array([x["pan"] for x in tr]); tilt = np.array([x["tilt"] for x in tr])
    k = np.clip(np.searchsorted(tt, t), 1, len(tt) - 1); kk = np.where(np.abs(tt[k] - t) < np.abs(tt[k - 1] - t), k, k - 1)
    ok = np.abs(tt[kk] - t) <= 5
    P, Q, V, T = pan[kk][ok], tilt[kk][ok], v[ok], t[ok]
    reg = os.path.join(world, "raw", "registration.json")
    smooth = (json.load(open(reg)).get("smooth") or []) if os.path.exists(reg) else []
    cache, az, el = {}, [], []
    for p, q, ts in zip(P, Q, T):
        off = (0.0, 0.0, 0.0)
        if smooth:
            j = int(np.argmin([abs(s[0] - ts) for s in smooth])) if len(smooth) < 200 else 0
            if abs(smooth[j][0] - ts) <= 7200:
                off = (smooth[j][1], smooth[j][2], 0.0)
        key = (int(p), int(q), off)
        if key not in cache:
            d = ctx.camera.rays(p, q, 960, 540, off=off)
            cache[key] = (float(np.degrees(np.arctan2(d[0], d[2]))), float(np.degrees(np.arcsin(np.clip(d[1], -1, 1)))))
        a, e = cache[key]
        az.append(a); el.append(e)
    return np.radians(az), np.radians(el), V, T, P, Q


def fit(az, el, v):
    X = np.c_[np.ones(len(v)), np.cos(az), np.sin(az), np.sin(el)]
    c, *_ = np.linalg.lstsq(X, v, rcond=None)
    pred = X @ c
    return c, float(1 - ((v - pred) ** 2).sum() / max(((v - v.mean()) ** 2).sum(), 1e-9))


def build(world, rssi_paths=None, seed=0):
    from PIL import Image
    az, el, V, T, P, Q = samples(world, rssi_paths)
    blocks = ((T - T.min()) // BLOCK_S).astype(int)
    half = blocks % 2
    key = (P // 150).astype(int) * 100 + (Q // 150).astype(int)

    def means(m):
        return {k: V[m & (key == k)].mean() for k in np.unique(key[m]) if (m & (key == k)).sum() >= MIN_BIN}
    A, B = means(half == 0), means(half == 1)
    shared = sorted(set(A) & set(B))
    a = np.array([A[k] for k in shared]); b = np.array([B[k] for k in shared])
    rng = np.random.default_rng(seed)
    r = float(np.corrcoef(a, b)[0, 1]) if len(shared) >= 5 else float("nan")
    null = np.array([np.corrcoef(a, rng.permutation(b))[0, 1] for _ in range(5000)]) if len(shared) >= 5 else np.array([])
    c, r2 = fit(az, el, V)
    bearing = float(np.degrees(np.arctan2(c[2], c[1])))
    halves = [float(np.degrees(np.arctan2(*fit(az[half == h], el[half == h], V[half == h])[0][[2, 1]]))) for h in (0, 1)]
    ths = []
    ub = np.unique(blocks)
    for _ in range(500):
        idx = np.concatenate([np.where(blocks == x)[0] for x in rng.choice(ub, len(ub))])
        cc, _ = fit(az[idx], el[idx], V[idx])
        ths.append(float(np.degrees(np.arctan2(cc[2], cc[1]))))
    dd = (np.array(ths) - bearing + 180) % 360 - 180
    res = {"samples": int(len(V)), "hours": round(float((T.max() - T.min()) / 3600), 2), "direction_bins": len(shared),
           "split_half_r": round(r, 3), "shuffle_null_p95": round(float(np.percentile(null, 95)), 3) if len(null) else None,
           "p": round(float((null >= r).mean()), 4) if len(null) else None,
           "fit": {"mean_dbm": round(float(c[0]), 1), "swing_db": round(float(np.hypot(c[1], c[2])), 1),
                   "elevation_db": round(float(c[3]), 1), "r2": round(r2, 3)},
           "bearing_deg": round(bearing, 1), "bearing_halves_deg": [round(h, 1) for h in halves],
           "bearing_95ci_deg": [round(bearing + float(np.percentile(dd, 2.5)), 1), round(bearing + float(np.percentile(dd, 97.5)), 1)],
           "covered_azimuth_deg": [round(float(np.degrees(az.min()))), round(float(np.degrees(az.max())))],
           "covered_elevation_deg": [round(float(np.degrees(el.min()))), round(float(np.degrees(el.max())))],
           "meaning": "direction of the strongest link from the camera's head antenna; usually toward the access point, "
                      "unconfirmed (a reflection can produce it)"}
    res["supported"] = bool(res["p"] is not None and res["p"] < 0.05 and abs(((halves[0] - halves[1]) + 180) % 360 - 180) < 45)
    model = os.path.join(world, "model")
    mp = os.path.join(model, "manifest.json")
    m = json.load(open(mp))
    L = {x["id"]: x for x in m["layers"]}
    if res["supported"]:
        # paint every direction of the measured surface with the predicted RSSI for a head pointing there; only inside
        # the span of head directions the data covers (elsewhere: no data, black)
        H, W = 384, 768
        D = directions(W, H)
        A2, E2 = az_el(D)
        ar, er = np.radians(A2), np.radians(E2)
        pred = c[0] + c[1] * np.cos(ar) + c[2] * np.sin(ar) + c[3] * np.sin(er)
        cov = (E2 >= res["covered_elevation_deg"][0] - 5) & (E2 <= res["covered_elevation_deg"][1] + 5)
        lo, hi = float(np.percentile(V, 5)), float(np.percentile(V, 95))
        x = np.clip((pred - lo) / max(hi - lo, 1e-6), 0, 1)
        rgb = np.stack([0.25 + 0.7 * x, 0.15 + 0.55 * x ** 1.3, 0.55 - 0.35 * x], -1)        # weak = dim violet, strong = warm
        img = np.where(cov[..., None], (rgb * 255).astype(np.uint8), 0).astype(np.uint8)
        Image.fromarray(img).save(os.path.join(model, "surfel_radio.png"))
        if "fused.surfels" in L:
            L["fused.surfels"]["data"].setdefault("textures", {})["radio"] = "surfel_radio.png"
        res["texture_range_dbm"] = [round(lo, 1), round(hi, 1)]
    if "radio.rssi" in L:
        L["radio.rssi"]["data"]["by_direction"] = res
    json.dump(m, open(mp, "w"))
    json.dump(res, open(os.path.join(world, "raw", "radio", "radio_map.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    a = sys.argv[1:]
    print(json.dumps(build(a[0], a[1:] or None), indent=1))
