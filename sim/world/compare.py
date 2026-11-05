"""Did a model change help where it was meant to? Compares two built worlds on the plan's target directions and
everywhere, with the uncertainty map (partly the model's own say) and two checks the new frames did not produce:
dog positions landing beyond the surface (only time blocks the sightline carving did NOT use), and the unsnapped floor against the sound-set floor plane.
    python3 -m sim.world.compare eval/world eval/world_remap1 [target.npy]"""
import json
import math
import os
import sys

import numpy as np


def stats(world, target):
    m = json.load(open(os.path.join(world, "model", "manifest.json")))
    z = np.load(os.path.join(world, "model", "surfels.npz"))
    unc, R, fr = z["uncertainty"], z["range_m"], z["floor_residual_m"]
    H, W = R.shape
    h = m["room"]["floor_height_m"]
    rows = next(x for x in m["layers"] if x["id"] == "presence.dog")["data"]["rows"]
    beyond = {"target": [0, 0], "all": [0, 0]}
    from .objects import is_carving_row
    for t, x, zz, cut, mv, dw in rows:
        if cut or is_carving_row(t):             # held out: only time blocks the carving never used
            continue
        az = math.degrees(math.atan2(x, zz)); rh = math.hypot(x, zz); el = -math.degrees(math.atan2(h, rh))
        i, j = int((az + 180) / 360 * W) % W, min(H - 1, int((90 - el) / 180 * H))
        r = R[j, i]
        if not np.isfinite(r):
            continue
        b = math.hypot(rh, h) > r * 1.15
        for k in ("all",) + (("target",) if target[j, i] else ()):
            beyond[k][0] += int(b); beyond[k][1] += 1
    da3 = next((x for x in m["layers"] if x["id"] == "visible.da3"), None)
    def med(a, msk):
        v = a[msk & np.isfinite(a)]
        return float(np.median(v)) if len(v) else float("nan")
    return {"uncertainty_target": round(med(unc, target), 4), "uncertainty_all": round(med(unc, np.ones_like(target)), 4),
            "seen_target": round(float(np.isfinite(R[target]).mean()), 3),
            "floor_check_cm_target": round(100 * med(np.abs(fr), target), 1), "floor_check_cm_all": round(100 * med(np.abs(fr), np.ones_like(target)), 1),
            "dog_beyond_target": f"{beyond['target'][0]}/{beyond['target'][1]}", "dog_beyond_all": f"{beyond['all'][0]}/{beyond['all'][1]}",
            "da3_overlap_pct": round(100 * da3["data"]["eval"]["overlap_disagreement"], 2) if da3 else None}


if __name__ == "__main__":
    a = sys.argv[1:]
    target = np.load(a[2] if len(a) > 2 else os.path.join(a[0], "raw", "nbv_target.npy"))
    for w in a[:2]:
        print(f"{os.path.basename(w):14s}", stats(w, target))
