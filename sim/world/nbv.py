"""Where is the model weakest, and which camera views would improve it most (next best views).

Value of a candidate pose = sum over the directions its frame sees of uncertainty x importance (unseen directions count
as uncertainty 0.5). Views are picked greedily; after each pick the uncertainty inside its footprint is halved, so the
next pick goes elsewhere (diminishing returns). The plan is written for `remap_capture.py` and the regions the plan
targets are recorded, so the improvement can be checked there afterwards with evidence the new frames did not produce.
    python3 -m sim.world.nbv eval/world [K]   -> eval/world/raw/nbv_plan.json"""
import json
import os
import sys

import numpy as np

from .geometry import az_el, directions


def plan(world, K=12):
    from .sensors.base import Context
    ctx = Context(world)
    z = np.load(os.path.join(ctx.model, "surfels.npz"))
    unc, imp = z["uncertainty"].astype(np.float64), z["importance"].astype(np.float64)
    H, W = unc.shape
    value = np.where(np.isfinite(unc), unc, 0.5) * imp
    D = directions(W, H)
    az, el = az_el(D)
    elmask = (el > -60) & (el < 45)                                  # what the head can ever see (sweep range)
    value = np.where(elmask, value, 0)
    cands = [(p, t) for p in range(0, 3701, 120) for t in range(60, 961, 75)]
    cam = ctx.camera
    foot = {}
    for p, t in cands:
        u, v, ok = cam.project(p, t, D, 1920, 1080)
        foot[(p, t)] = ok
    picks, residual = [], value.copy()
    for _ in range(K):
        best = max(cands, key=lambda c: float(residual[foot[c]].sum()))
        gain = float(residual[foot[best]].sum())
        if gain <= 0:
            break
        fp = foot[best]
        picks.append({"pan": best[0], "tilt": best[1], "gain": round(gain, 1),
                      "centre_az": round(float(np.median(az[fp])), 1), "centre_el": round(float(np.median(el[fp])), 1),
                      "mean_uncertainty_in_view": round(float(np.nanmean(np.where(np.isfinite(unc[fp]), unc[fp], np.nan))), 3)})
        residual[fp] *= 0.5
    target = np.zeros((H, W), bool)
    for pk in picks:
        target |= foot[(pk["pan"], pk["tilt"])]
    out = {"k": len(picks), "picks": picks, "total_value": round(float(value.sum()), 1),
           "value_covered_share": round(float(1 - residual.sum() / max(value.sum(), 1e-9)), 3)}
    np.save(os.path.join(ctx.path("raw"), "nbv_target.npy"), target)
    json.dump(out, open(ctx.path("raw", "nbv_plan.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    r = plan(a[0], int(a[1]) if len(a) > 1 else 12)
    for p in r["picks"]:
        print(p)
    print("value covered", r["value_covered_share"])
