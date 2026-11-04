#!/usr/bin/env python3
"""Fit the simulator's unobservable parameters (her true walking speed, how long she rests or is away, how often
people set off motion) so the REAL controller, run in the sim, produces the same logged statistics as the real
install (moment matching through the observation model; Ramos et al. BayesSim, simplified). CMA-ES, common random
numbers (the same seeds for every candidate).

    python3 sim/calibrate.py [evals]     ->  writes sim/params.json "free" + the fit report
"""
import json
import math
import os
import sys

import cma

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate1  # noqa: E402
from room import load_params  # noqa: E402

# (name, low, high, log-scale)
SPACE = [("walk_speed_med", 30, 400, True), ("rest_med_s", 30, 1800, True), ("away_prob", 0.05, 0.6, False),
         ("away_med_s", 120, 3600, True), ("people_motion_per_h", 0.5, 30, True), ("p_hidden", 0.0, 0.9, False),
         ("q_hard", 0.0, 0.8, False), ("p_hard", 0.02, 0.9, True)]
TARGETS = [("in_view_share_while_home", None, 0.05), ("losses_per_home_hour", None, 2.0),
           ("seen_again_within_s", 30, 0.08), ("seen_again_within_s", 60, 0.08),
           ("follow_actions_per_hour", "chase", 1.5), ("follow_actions_per_hour", "search", 1.0),
           ("follow_actions_per_hour", "track", 1.5), ("search_found_share", None, 0.08)]
HOURS, SEEDS = 4, 4


def decode(z):
    out = {}
    for (k, lo, hi, lg), v in zip(SPACE, z):
        u = 1 / (1 + math.exp(-v))                                    # unbounded search space -> (lo, hi)
        out[k] = math.exp(math.log(lo) + u * (math.log(hi) - math.log(lo))) if lg else lo + u * (hi - lo)
    return out


def get(m, k):
    v = m.get(k[0])
    if k[1] is None:
        return v if v is not None else 0.0                 # e.g. search_found_share of a run with no search
    return (v or {}).get(k[1], (v or {}).get(str(k[1]), 0.0))


def loss(free, real, params=None):
    fixed = {k: v for k, v in ((params or {}).get("free") or {}).items() if k not in dict((s[0], 0) for s in SPACE)}
    rows = gate1.simulate(HOURS, SEEDS, free=dict(fixed, **free), params=params)
    err = 0.0
    for k in TARGETS:
        sim = sum(get(m, k) for m, _ in rows) / len(rows)
        err += ((sim - get(real, k)) / k[2]) ** 2
    return err, rows


def main(evals=120, path=None, write=True):
    path = path or os.path.join(HERE, "params.json")
    P = load_params(path)
    real = P["real_metrics"]
    es = cma.CMAEvolutionStrategy([0.0] * len(SPACE), 1.0, {"seed": 1, "maxfevals": evals, "verbose": -9, "popsize": 8})
    best = (1e18, None)
    while not es.stop():
        Z = es.ask()
        L = []
        for z in Z:
            e, _ = loss(decode(z), real, P)
            L.append(e)
            if e < best[0]:
                best = (e, decode(z))
        es.tell(Z, L)
        print(f"evals {es.countevals:>4}  best loss {best[0]:.2f}  {json.dumps({k: round(v, 3) for k, v in best[1].items()})}", flush=True)
    e, rows = loss(best[1], real, P)
    report = gate1.compare(rows, real)
    P["free"] = dict(P.get("free") or {}, **{k: round(v, 4) for k, v in best[1].items()})   # keep fitted ones (owner moves)
    P["calibration"] = {"loss": round(e, 3), "hours": HOURS, "seeds": SEEDS, "evals": int(es.countevals), "fit": report}
    if write:
        json.dump(P, open(path, "w"), indent=1)
    for k, v in report.items():
        print(f"{k:<36} real {v['real']!s:<7} sim {v['sim_mean']:<7} [{v['sim_min']}, {v['sim_max']}]")
    return P


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 120)
