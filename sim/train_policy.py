#!/usr/bin/env python3
"""Train a Follow policy for THIS camera and pet on its calibrated simulator, and judge it on rooms it never saw.

The policy is the real controller (server/tracker.py) with its tuning constants learned by direct policy search
(CMA-ES), the right size for a policy that must run free on a $5 server and stay readable. Training rooms are drawn
around the calibrated parameters (domain randomisation); every candidate is scored on the same bank of rooms (common
random numbers) with a robust objective (the mean of the worst 25% of rooms) under a hard budget: no more motor moves
per hour than today's controller.

    python3 sim/train_policy.py [evals]   ->  sim/policy.json (constants + held-out report) if it wins gate 2
"""
import json
import math
import os
import sys

import cma
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate1  # noqa: E402
from room import Room, load_params  # noqa: E402

# constant: (low, high). Bounds are where the controller stays sane; the server clamps to the same ranges.
SPACE = gate1.load_tracker().POLICY_BOUNDS   # one source of truth: server/tracker.py
HOURS = 2.0


def rooms(n, seed0, P, spread=True):
    rng = np.random.default_rng(seed0)
    out = []
    for i in range(n):
        f = dict(P["free"])
        if spread:
            for k in ("walk_speed_med", "rest_med_s", "away_med_s", "people_motion_per_h"):
                f[k] = f[k] * float(rng.uniform(0.6, 1.6))
            f["away_prob"] = float(np.clip(f["away_prob"] * rng.uniform(0.6, 1.4), 0.02, 0.8))
            f["lag_s"] = float(rng.uniform(0.5, 1.1))
        out.append((seed0 + i, f))
    return out


def evaluate(overrides, bank, T):
    res = []
    for seed, free in bank:
        r = Room(seed=seed, free=free)
        base = {k: getattr(T, k) for k in overrides}
        try:
            gate1.run_watcher(r, T, HOURS, overrides)
        finally:
            for k, v in base.items():
                setattr(T, k, v)
        t = gate1.truth(r)
        res.append((t["true_in_view_share_while_home"], t["moves_per_hour"]))
    return res


def score(res, move_budget):
    iv = sorted(x for x, _ in res)
    worst = iv[: max(1, len(iv) // 4)]
    moves = sum(m for _, m in res) / len(res)
    return sum(worst) / len(worst) - 0.02 * max(0.0, moves - move_budget), sum(iv) / len(iv), moves


def decode(z):
    return {k: lo + (hi - lo) / (1 + math.exp(-v)) for (k, (lo, hi)), v in zip(SPACE.items(), z)}


def main(evals=160, n_train=24, n_test=32):
    P = load_params()
    T = gate1.load_tracker()
    train, test = rooms(n_train, 10_000, P), rooms(n_test, 90_000, P)      # disjoint seeds and parameter draws
    base_train = evaluate({}, train, T)
    b_score, b_iv, b_moves = score(base_train, 1e9)
    print(f"baseline (today's constants) on training rooms: robust {b_score:.3f}, mean in-view {b_iv:.3f}, {b_moves:.1f} moves/h", flush=True)
    # start the search at today's constants
    z0 = []
    for k, (lo, hi) in SPACE.items():
        u = min(0.98, max(0.02, (getattr(T, k) - lo) / (hi - lo)))
        z0.append(math.log(u / (1 - u)))
    es = cma.CMAEvolutionStrategy(z0, 0.6, {"seed": 3, "maxfevals": evals, "popsize": 10, "verbose": -9})
    best = (b_score, {})
    while not es.stop():
        Z = es.ask()
        L = []
        for z in Z:
            o = decode(z)
            s, iv, mv = score(evaluate(o, train, T), b_moves)
            L.append(-s)
            if s > best[0]:
                best = (s, o)
        es.tell(Z, L)
        print(f"evals {es.countevals:>4}  best robust {best[0]:.3f}", flush=True)
    # gate 2: held-out rooms, never seen in training
    base_test = evaluate({}, test, T)
    cand_test = evaluate(best[1], test, T)
    bs, biv, bmv = score(base_test, 1e9)
    cs, civ, cmv = score(cand_test, 1e9)
    wins = sum(c[0] > b[0] for c, b in zip(cand_test, base_test))
    report = {"train_rooms": n_train, "test_rooms": n_test, "hours_per_room": HOURS,
              "baseline": {"robust": round(bs, 3), "in_view": round(biv, 3), "moves_per_h": round(bmv, 1)},
              "candidate": {"robust": round(cs, 3), "in_view": round(civ, 3), "moves_per_h": round(cmv, 1)},
              "rooms_won": f"{wins}/{n_test}"}
    passed = cs > bs and civ > biv and cmv <= bmv * 1.05
    report["gate2"] = "pass" if passed else "fail"
    print(json.dumps(report, indent=1))
    out = {"constants": {k: round(v, 4) for k, v in best[1].items()}, "report": report,
           "params_source": P.get("source"), "calibration_loss": (P.get("calibration") or {}).get("loss")}
    json.dump(out, open(os.path.join(HERE, "policy.json" if passed else "policy.rejected.json"), "w"), indent=1)
    return passed


if __name__ == "__main__":
    sys.exit(0 if main(int(sys.argv[1]) if len(sys.argv) > 1 else 160) else 1)
