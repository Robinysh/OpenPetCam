"""Radio shadow test: does the camera's WiFi signal change with WHERE the pet is (the pet as the moving probe of a
radio tomographic image), beyond what time alone explains?

Statistic: the share of RSSI variance explained by the floor cell (CELL m) she is in, from detections within MATCH_S of
each RSSI sample (eta squared), WITHIN a fixed camera pose: the camera follows her, and its antenna turns with the head,
so RSSI against her position alone measures the antenna, not her shadow (first run, 2026-11-05: 0.89 raw vs a null of
0.50, but pose alone explained 0.67 and within a pose her cell explained 0.09 against a null of 0.28). Null: the same with the RSSI series circularly shifted in time by >= MIN_SHIFT_S, which
keeps its own drift and autocorrelation but breaks any link to her position. A spatial radio map is only drawn when
the true value beats the null's 95th percentile; otherwise the result is reported and nothing is drawn.
    python3 -m sim.world.rti eval/world [rssi.log ...]"""
import json
import os
import re
import sys

import numpy as np

CELL, MATCH_S, MIN_SHIFT_S, MIN_PER_CELL = 0.5, 2.0, 600.0, 20


def load_rssi(paths):
    t, v = [], []
    for p in paths:
        for line in open(p):
            m = re.match(r"(\d+\.\d+) > RSSI=(-?\d+)", line) or re.match(r"(\d+\.\d+) RSSI=(-?\d+)", line)
            if m:
                t.append(float(m.group(1))); v.append(float(m.group(2)))
    o = np.argsort(t)
    return np.array(t)[o], np.array(v)[o]


def eta2(cells, y):
    ok = cells >= 0
    c, y = cells[ok], y[ok]
    if len(y) < 50:
        return np.nan, 0
    keep = np.isin(c, [k for k in np.unique(c) if (c == k).sum() >= MIN_PER_CELL])
    c, y = c[keep], y[keep]
    if len(np.unique(c)) < 3:
        return np.nan, 0
    mu = y.mean()
    between = sum((c == k).sum() * (y[c == k].mean() - mu) ** 2 for k in np.unique(c))
    return float(between / max(((y - mu) ** 2).sum(), 1e-9)), int(len(np.unique(c)))


def run(world, rssi_paths=None, n_null=500, seed=0):
    m = json.load(open(os.path.join(world, "model", "manifest.json")))
    dog = np.array([r[:3] for r in next(x for x in m["layers"] if x["id"] == "presence.dog")["data"]["rows"]])
    dog = dog[np.argsort(dog[:, 0])]
    t, v = load_rssi(rssi_paths or [os.path.join(world, "rssi.log")])
    span = (t.min(), t.max()) if len(t) else (0, 0)
    # camera pose at each RSSI sample (track.json); RSSI is taken relative to its own pose's mean
    tr = json.load(open(os.path.join(world, "track.json")))
    tt = np.array([x["ts"] for x in tr]); pp = np.array([(x["pan"] // 150) * 100 + x["tilt"] // 150 for x in tr])
    kk = np.clip(np.searchsorted(tt, t), 1, len(tt) - 1); kk = np.where(np.abs(tt[kk] - t) < np.abs(tt[kk - 1] - t), kk, kk - 1)
    pose = np.where(np.abs(tt[kk] - t) <= 5, pp[kk], -1)
    pose_eta, n_pose = eta2(pose, v)
    r = v.astype(float).copy()
    for q in np.unique(pose[pose >= 0]):
        r[pose == q] -= v[pose == q].mean()

    def cells_for(tt):
        k = np.clip(np.searchsorted(dog[:, 0], tt), 1, len(dog) - 1)
        near = np.where(np.abs(dog[k, 0] - tt) < np.abs(dog[k - 1, 0] - tt), k, k - 1)
        ok = np.abs(dog[near, 0] - tt) <= MATCH_S
        ix = np.floor(dog[near, 1] / CELL).astype(int) + 1000
        iz = np.floor(dog[near, 2] / CELL).astype(int) + 1000
        return np.where(ok, ix * 10000 + iz, -1)

    c = np.where(pose >= 0, cells_for(t), -1)
    raw, _ = eta2(cells_for(t), v)
    true, ncell = eta2(c, r)
    rng = np.random.default_rng(seed)
    dur = span[1] - span[0]
    null = []
    if dur > 2 * MIN_SHIFT_S:
        for _ in range(n_null):
            s = rng.uniform(MIN_SHIFT_S, dur - MIN_SHIFT_S)
            tt = span[0] + (t - span[0] + s) % dur                 # RSSI moved in time, wrapped around
            null.append(eta2(np.where(pose >= 0, cells_for(tt), -1), r)[0])
    null = np.array([x for x in null if np.isfinite(x)])
    res = {"rssi_samples": int(len(t)), "hours": round(dur / 3600, 2), "matched_samples": int((c >= 0).sum()), "cells": ncell,
           "raw_eta2_without_pose_control": None if not np.isfinite(raw) else round(raw, 4),
           "pose_alone_eta2": None if not np.isfinite(pose_eta) else round(pose_eta, 4), "poses": n_pose,
           "eta2_within_pose": None if not np.isfinite(true) else round(true, 4),
           "null_mean": round(float(null.mean()), 4) if len(null) else None,
           "null_p95": round(float(np.percentile(null, 95)), 4) if len(null) else None,
           "p": round(float((null >= true).mean()), 4) if len(null) and np.isfinite(true) else None}
    res["supported"] = bool(res["p"] is not None and res["p"] < 0.05)
    os.makedirs(os.path.join(world, "raw", "radio"), exist_ok=True)
    json.dump(res, open(os.path.join(world, "raw", "radio", "rti.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    a = sys.argv[1:]
    print(run(a[0], a[1:] or None))
