#!/usr/bin/env python3
"""Delay-aware Follow vs the deployed Follow, before any reinforcement learning.

The question (outside review 2026-11-08): how much of Follow's lost view is left once the controller accounts for
observation age and for the time its own aim takes? Two switches in server/tracker.py, off by default:
  CAPTURE_TIME  her track is stamped with each frame's capture time and the watcher's repeats of a stale box are skipped
  SETTLE_LEAD   she is predicted to when the aim will have finished; LEAD becomes a residual on top
Every arm runs the real controller on the same simulated rooms (common random numbers), scored on what only the
simulator knows: the true share of time she is in view while home, motor moves per hour, and how often she leaves the
view while home ("exits": in view for at least 3 s, then out of view, per home hour).

  python3 sim/follow_delay.py grid             # the deployed constants with each switch, 32 held-out rooms
  python3 sim/follow_delay.py home             # the same arms in the 3D room model (sim/home.py), paired seeds
  python3 sim/follow_delay.py tune [evals]     # re-tune both arms the same way on training rooms, judge on held-out
                                               # (the candidate arm: SETTLE_LEAD + RETURN_AFTER_PAUSE; CAPTURE_TIME lost)
Results: eval/rl/follow_delay_<mode>.json. Training rooms and held-out rooms are disjoint (sim/train_policy.py).
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import gate1  # noqa: E402
import train_policy as tp  # noqa: E402
from room import Room, load_params  # noqa: E402

FLAGS = ("CAPTURE_TIME", "SETTLE_LEAD")


def deployed():
    return dict(json.load(open(os.path.join(ROOT, "models", "follow_policy.json")))["constants"])


def exits_per_home_hour(room, min_in=3.0):
    n, run, home = 0, 0.0, 0.0
    prev = None
    for _, dt, v, h in room.truth:
        if h:
            home += dt
        if prev is not None and prev[0] and prev[1] and h and not v and run >= min_in:
            n += 1
        run = run + dt if (v and h) else 0.0
        prev = (v, h)
    return n / max(home / 3600, 1e-9)


def run_bank(overrides, bank, T):
    rows = []
    for seed, free in bank:
        r = Room(seed=seed, free=free)
        base = {k: getattr(T, k) for k in overrides}
        try:
            gate1.run_watcher(r, T, tp.HOURS, overrides)
        finally:
            for k, v in base.items():
                setattr(T, k, v)
        t = gate1.truth(r)
        home = sum(dt for dt, st, _, _ in r.truth_detail if st != "away") or 1e-9
        lost = lambda pred: round(sum(dt for dt, st, hid, iv in r.truth_detail if st != "away" and pred(st, hid, iv)) / home, 4)  # noqa: E731
        rows.append({"seed": seed, "in_view": t["true_in_view_share_while_home"], "moves": t["moves_per_hour"],
                     "exits": round(exits_per_home_hour(r), 2),
                     "lost_hidden": lost(lambda st, hid, iv: hid),                          # behind something: no aim helps
                     "lost_frame_rest": lost(lambda st, hid, iv: not hid and not iv and st == "rest"),
                     "lost_frame_move": lost(lambda st, hid, iv: not hid and not iv and st == "move")})
    return rows


def summary(rows):
    iv = sorted(r["in_view"] for r in rows)
    worst = iv[: max(1, len(iv) // 4)]
    return {"in_view": round(float(np.mean(iv)), 3), "worst_quarter": round(float(np.mean(worst)), 3),
            "moves_per_h": round(float(np.mean([r["moves"] for r in rows])), 1),
            "exits_per_home_h": round(float(np.mean([r["exits"] for r in rows])), 2),
            **{k: round(float(np.mean([r[k] for r in rows])), 3) for k in ("lost_hidden", "lost_frame_rest", "lost_frame_move")}}


def paired(a, b, key, n=4000):
    d = np.array([y[key] - x[key] for x, y in zip(a, b)])
    rng = np.random.default_rng(0)
    bs = [float(np.mean(rng.choice(d, len(d)))) for _ in range(n)]
    return {"mean": round(float(d.mean()), 3), "ci95": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)],
            "rooms_better": int(sum((v > 0) if key == "in_view" else (v < 0) for v in d))}


def grid(n_test=32):
    P, T = load_params(), gate1.load_tracker()
    test = tp.rooms(n_test, 90_000, P)
    base = deployed()
    arms = {"deployed": {}, "capture_time": {"CAPTURE_TIME": True}, "settle_lead": {"SETTLE_LEAD": True},
            "both": {"CAPTURE_TIME": True, "SETTLE_LEAD": True}}
    for lead in (0.0, 0.25):
        arms[f"both_lead{lead}"] = {"CAPTURE_TIME": True, "SETTLE_LEAD": True, "LEAD": lead}
    out = {}
    for name, extra in arms.items():
        rows = run_bank(dict(base, **extra), test, T)
        out[name] = {"summary": summary(rows), "rows": rows}
        print(f"{name:16s}", out[name]["summary"], flush=True)
    for name in arms:
        if name != "deployed":
            out[name]["vs_deployed"] = {k: paired(out["deployed"]["rows"], out[name]["rows"], k) for k in ("in_view", "moves", "exits")}
            print(f"{name:16s} vs deployed:", json.dumps(out[name]["vs_deployed"]), flush=True)
    json.dump(out, open(os.path.join(ROOT, "eval", "rl", "follow_delay_grid.json"), "w"), indent=1)
    return out


def tune(evals=160, n_train=24, n_test=32):
    """Re-tune the same constants for each arm with the same search (CMA-ES, robust objective, move budget = the
    deployed constants' moves on the training rooms), then compare the two tuned arms on held-out rooms."""
    import cma
    P, T = load_params(), gate1.load_tracker()
    train, test = tp.rooms(n_train, 10_000, P), tp.rooms(n_test, 90_000, P)
    base = deployed()
    budget = summary(run_bank(base, train, T))["moves_per_h"]
    space = tp.SPACE
    results = {}
    for arm, flags in (("deployed_retuned", {}), ("delay_aware", {"SETTLE_LEAD": True, "RETURN_AFTER_PAUSE": True})):
        z0 = []
        for k, (lo, hi) in space.items():
            u = min(0.98, max(0.02, (base.get(k, getattr(T, k)) - lo) / (hi - lo)))
            z0.append(math.log(u / (1 - u)))
        es = cma.CMAEvolutionStrategy(z0, 0.5, {"seed": 7, "maxfevals": evals, "popsize": 10, "verbose": -9})
        best = (-1e9, dict(base))
        while not es.stop():
            Z = es.ask()
            L = []
            for z in Z:
                o = tp.decode(z)
                s = summary(run_bank(dict(o, **flags), train, T))
                sc = s["worst_quarter"] - 0.02 * max(0.0, s["moves_per_h"] - budget)
                L.append(-sc)
                if sc > best[0]:
                    best = (sc, o)
            es.tell(Z, L)
        results[arm] = {"constants": {k: round(v, 4) for k, v in best[1].items()}, "flags": flags, "train_score": round(best[0], 3)}
        print(arm, "train score", round(best[0], 3), flush=True)
    rows = {"deployed": run_bank(base, test, T)}
    for arm, r in results.items():
        rows[arm] = run_bank(dict(r["constants"], **r["flags"]), test, T)
    out = {"move_budget_train": budget, "arms": results,
           "held_out": {k: summary(v) for k, v in rows.items()},
           "delay_aware_vs_deployed": {k: paired(rows["deployed"], rows["delay_aware"], k) for k in ("in_view", "moves", "exits")},
           "delay_aware_vs_retuned": {k: paired(rows["deployed_retuned"], rows["delay_aware"], k) for k in ("in_view", "moves", "exits")},
           "rows": rows}
    print(json.dumps({k: out[k] for k in ("held_out", "delay_aware_vs_deployed", "delay_aware_vs_retuned")}, indent=1), flush=True)
    json.dump(out, open(os.path.join(ROOT, "eval", "rl", "follow_delay_tune.json"), "w"), indent=1)
    return out


def home(seeds=24, hours=4.0):
    """The same arms in the 3D room model (sim/home.py HomeRoom: her real stays and walks on the measured floor,
    geometric occlusion), paired by seed and start hour as sim/follow_eval.py does."""
    from zoneinfo import ZoneInfo
    from home import HomeData, HomeRoom
    try:
        sys.path.append(os.path.join(ROOT, "tools"))
        from siteconf import SITE
        tz = ZoneInfo(SITE.get("TZ", "UTC"))
    except Exception:  # noqa: BLE001
        tz = ZoneInfo("UTC")
    P = load_params()
    free = P.get("free") or {}
    Room.HARD_CORRELATED = True
    data = HomeData.from_world(os.path.join(ROOT, "eval", "world"), os.path.join(ROOT, "eval", "rl", "rl_export.json"),
                               tz=tz, walks=os.path.join(ROOT, "eval", "rl", "walks.json"))
    hours_of_day = [st["hour"] for st in data.stays]
    T = gate1.load_tracker()
    base = deployed()
    arms = {"deployed": {}, "settle_lead": {"SETTLE_LEAD": True}, "return_after_pause": {"RETURN_AFTER_PAUSE": True},
            "settle+return": {"SETTLE_LEAD": True, "RETURN_AFTER_PAUSE": True}}
    rows = {}
    for name, extra in arms.items():
        ov = dict(base, **extra)
        out = []
        for s in range(seeds):
            room = HomeRoom(seed=1000 + s, params=P, free=free, home=data, start_hour=hours_of_day[s * 7919 % len(hours_of_day)])
            saved = {k: getattr(T, k) for k in ov}
            try:
                gate1.run_watcher(room, T, hours, ov)
            finally:
                for k, v in saved.items():
                    setattr(T, k, v)
            t = gate1.truth(room)
            out.append({"seed": s, "in_view": t["true_in_view_share_while_home"], "moves": t["moves_per_hour"],
                        "exits": round(exits_per_home_hour(room), 2)})
        rows[name] = out
        print(f"{name:20s}", {k: round(float(np.mean([r[k] for r in out])), 3) for k in ("in_view", "moves", "exits")}, flush=True)
    res = {"seeds": seeds, "hours": hours, "rows": rows,
           "vs_deployed": {n: {k: paired(rows["deployed"], rows[n], k) for k in ("in_view", "moves", "exits")} for n in arms if n != "deployed"}}
    for n, v in res["vs_deployed"].items():
        print(f"{n:20s} vs deployed:", json.dumps(v), flush=True)
    json.dump(res, open(os.path.join(ROOT, "eval", "rl", "follow_delay_home.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "grid"
    if mode == "home":
        home()
    elif mode == "tune":
        tune(int(sys.argv[2]) if len(sys.argv) > 2 else 160)
    else:
        grid()
