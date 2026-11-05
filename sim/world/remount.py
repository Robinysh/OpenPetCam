"""The camera was re-mounted (same height, slightly turned): carry every stored pose over to the new mount.

A pan/tilt pose is only meaningful together with the mount's rotation (Camera.R(..., off)). After a re-mount the
registration of new snapshots to the panorama (registration.measure) gives the new correction; this tool
  1. appends it to the dated mount history (raw/mount_history.json; the app bundle carries it as mount_offsets, so
     positions logged before the move keep the old correction and later ones get the new),
  2. re-aims stored poses so each points where it did before: the room map's hourly stops (and a median pose shift
     for its floor grid), the learned search stops (models/follow_policy.json) and, given --presets, the saved views.
Each pose is re-aimed numerically with the server's camera model: the pose whose optical axis under the new
correction is closest to the old pose's axis under the old one.
    python3 -m sim.world.remount eval/world SINCE_TS PITCH YAW [--before PITCH YAW] [--presets presets.json]"""
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "server"))
from roomcam import Camera  # noqa: E402


def reaim(cam, x, y, old, new):
    """(x', y') under `new` looking where (x, y) looked under `old`, and the remaining angle (deg)."""
    want = cam.R(x, y, old)[:, 2]
    f = lambda p: float(np.arccos(np.clip(cam.R(p[0], p[1], new)[:, 2] @ want, -1, 1)))     # noqa: E731
    best = None
    for dx in (-200, 0, 200):
        for dy in (-120, 0, 120):
            r = minimize(f, [x + dx, y + dy], method="Nelder-Mead", options={"xatol": 0.5, "fatol": 1e-7, "maxiter": 400})
            if best is None or r.fun < best.fun:
                best = r
    return float(best.x[0]), float(best.x[1]), float(np.degrees(best.fun))


def main(world, since, pitch, yaw, presets=None, before=None):
    """before: the correction measured just before the move with the same method and panorama (the like-for-like
    baseline for re-aiming); the history keeps the bundle's value for logged positions before the move."""
    doc = json.load(open(os.path.join(ROOT, "models", "room", "room.json")))
    cam = Camera(doc["calib"])
    hist_p = os.path.join(world, "raw", "mount_history.json")
    hist = json.load(open(hist_p)) if os.path.exists(hist_p) else [{"since": None, "off": list(doc.get("mount_offset_deg") or [0, 0, 0]),
                                                                    "note": "correction in the bundle before the first recorded re-mount"}]
    old = tuple(before) if before else tuple(hist[-1]["off"])
    new = (float(pitch), float(yaw), 0.0)
    hist.append({"since": float(since), "off": list(new), "note": "re-mount (new tape), registered from snapshots after it"})
    json.dump(hist, open(hist_p, "w"), indent=1)
    xm = 3700
    clamp = lambda x, y: (int(round(min(max(x, 0), xm))), int(round(min(max(y, 0), 1000))))     # noqa: E731
    report = {"reaim_from": old, "history_before": hist[-2]["off"], "new": new, "since": since}
    rm_p = os.path.join(ROOT, "models", "room_map.json")
    rm = json.load(open(rm_p))
    shifts, worst = [], 0.0
    for h, stops in rm["stops"].items():
        out = []
        for s in stops:
            x, y, err = reaim(cam, s[0], s[1], old, new)
            worst = max(worst, err); shifts.append((x - s[0], y - s[1]))
            out.append([*clamp(x, y), s[2]])
        rm["stops"][h] = out
    sh = np.median(np.array(shifts), 0)
    prev = rm.get("pose_shift") or [0, 0]
    rm["pose_shift"] = [int(round(prev[0] + sh[0])), int(round(prev[1] + sh[1]))]   # grid lookups: (x - dx, y - dy)
    rm.setdefault("remounts", []).append({"since": since, "old": old, "new": new})
    json.dump(rm, open(rm_p, "w"))
    report["room_map"] = {"stops_reaimed": len(shifts), "median_shift_steps": [round(float(v)) for v in sh], "worst_residual_deg": round(worst, 3)}
    fp = os.path.join(ROOT, "models", "follow_policy.json")
    if os.path.exists(fp):
        pol = json.load(open(fp))
        if pol.get("search_stops"):
            pol["search_stops"] = [list(clamp(*reaim(cam, s[0], s[1], old, new)[:2])) for s in pol["search_stops"]]
            json.dump(pol, open(fp, "w"), indent=1)
            report["follow_policy_stops"] = pol["search_stops"]
    if presets:
        pr = json.load(open(presets))
        items = pr if isinstance(pr, list) else pr.get("items", [])
        for it in items:
            it["x"], it["y"] = clamp(*reaim(cam, it["x"], it["y"], old, new)[:2])
        json.dump(pr, open(presets, "w"))
        report["presets"] = [(it.get("name"), it["x"], it["y"]) for it in items]
    # the bundle: dated offsets for history, the latest for live use
    doc["mount_offsets"] = [[h["since"] if h["since"] is not None else 0, *h["off"]] for h in hist]
    doc["mount_offset_deg"] = list(new)
    json.dump(doc, open(os.path.join(ROOT, "models", "room", "room.json"), "w"))
    return report


if __name__ == "__main__":
    a = sys.argv[1:]
    pres = a[a.index("--presets") + 1] if "--presets" in a else None
    bef = (float(a[a.index("--before") + 1]), float(a[a.index("--before") + 2]), 0.0) if "--before" in a else None
    print(json.dumps(main(a[0], float(a[1]), float(a[2]), float(a[3]), pres, bef)))
