#!/usr/bin/env python3
"""Learn where to look first (the Find search) for THIS camera and pet, from where she is actually found.

    python3 sim/train_search.py   ->  adds "search_stops" to models/follow_policy.json if it wins on held-out trials

Prior: the 2D map of her logged sightings (sim/params.json pet.room_map, pan x tilt), mixed with 20% uniform over the
room, because sightings are biased toward where the camera already looks. Plan: greedy over a grid of stops: the next
stop is the one that adds the most not-yet-covered probability per second (motor travel at the measured speed,
settle, the look). Judged by the REAL tracker search (server/tracker.py) in the calibrated simulator, on held-out trials
where she lies somewhere out of view: time to find, found within 60 s, wrong finds; against today's order (the saved
views). The zigzag after the stops stays as the fallback either way.
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
from room import Room, load_params  # noqa: E402

UNIFORM = 0.2
LOOK_S = 2.4            # settle + a 2-of-4 confirmed look, measured in the sim's search below


def prior(P, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    rm = P["pet"]["room_map"]
    g = P["geometry"]
    w = np.array([r[2] for r in rm], float)
    w /= w.sum()
    pts = []
    for _ in range(n):
        rp = (P.get("scene") or {}).get("rest_places") or []
        u = rng.random()
        if u < UNIFORM:
            pts.append((rng.uniform(0, g["pan_max"]), rng.uniform(60, g["tilt_max"] - 60)))
        elif rp and u < UNIFORM + float(P.get("free", {}).get("rest_place_share", 0.2)):
            r = rp[int(rng.integers(len(rp)))]
            pts.append((rng.uniform(*r["pan"]), rng.uniform(*r["tilt"])))
        else:
            r = rm[int(rng.choice(len(rm), p=w))]
            pts.append((r[0] + rng.uniform(-g["pan_frame"] / 4, g["pan_frame"] / 4),
                        r[1] + rng.uniform(-g["tilt_frame"] / 4, g["tilt_frame"] / 4)))
    return np.array(pts)


def plan(P, start=(1019, 534), most=8):
    g = P["geometry"]
    fw, fh, speed = g["pan_frame"], g["tilt_frame"], float(P["motor"]["speed_steps_s"]["250"])
    pts = prior(P)
    covered = np.zeros(len(pts), bool)
    cands = [(x, y) for x in np.arange(fw / 2, g["pan_max"] - fw / 2 + 1, fw / 2) for y in (318, 530, 742)]
    cands.append((g["pan_max"] - fw / 2, 530))
    blank = (P.get("scene") or {}).get("blank") or []        # bare walls and ceiling (Claude's map): nothing to find
    cands = [c for c in cands if not any(b["pan"][0] <= c[0] <= b["pan"][1] and b["tilt"][0] <= c[1] <= b["tilt"][1]
                                         for b in blank)]
    stops, pos = [], start
    for _ in range(most):
        best = None
        for c in cands:
            inside = (np.abs(pts[:, 0] - c[0]) < 0.45 * fw) & (np.abs(pts[:, 1] - c[1]) < 0.45 * fh) & ~covered
            gain = inside.mean()
            cost = math.hypot(c[0] - pos[0], c[1] - pos[1]) / speed + LOOK_S
            if gain > 0 and (best is None or gain / cost > best[0]):
                best = (gain / cost, c, inside)
        if best is None:
            break
        _, c, inside = best
        covered |= inside
        stops.append((int(c[0]), int(c[1])))
        pos = c
        if covered.mean() > 0.97:
            break
    return stops, float(covered.mean())


def trials(P, n, seed0):
    """Rooms where she lies still somewhere out of view and the camera waits at the rest view."""
    out = []
    rng = np.random.default_rng(seed0)
    for i in range(n):
        free = dict(P["free"], p_false_accept=P["free"].get("p_false_accept", 0.002))
        out.append((seed0 + i, free, int(rng.integers(1 << 30))))
    return out


def run_search(T, stops, bank, P):
    res = []
    for seed, free, s2 in bank:
        r = Room(seed=seed, free=free, params=P)
        rng = np.random.default_rng(s2)
        for _ in range(50):                                 # place her out of view, lying still for the whole trial
            r.dog_state, (r.px, r.py) = "rest", r._spot()
            if not r.in_view():
                break
        r.state_until = r.t + 1e6
        r.advance(1.0)
        T.SEARCH_STOPS = list(stops)
        tr = T.Tracker(r, r.find_pet, clock=lambda: r.t, log=lambda m: None, sleep=r.sleep, find_now=r.find_now)
        t0 = r.t
        out = tr.search("eval")
        found = bool(out and "found her" in out)
        right = found and r.in_view()
        res.append({"found": found, "right": right, "wrong": found and not right, "s": r.t - t0})
        del rng
    T.SEARCH_STOPS = []
    return res


def summary(res):
    ok = [x["s"] for x in res if x["right"]]
    n = len(res)
    return {"found_right": round(len(ok) / n, 3), "within_60s": round(sum(1 for s in ok if s <= 60) / n, 3),
            "median_s_when_found": round(float(np.median(ok)), 1) if ok else None,
            "wrong_finds": round(sum(x["wrong"] for x in res) / n, 3)}


def main():
    P = load_params()
    T = gate1.load_tracker()
    stops, cov = plan(P)
    print("learned stops:", stops, f"(prior coverage {cov:.2f})")
    test = trials(P, 120, 70_000)                          # held-out: seeds never used to build the prior's map
    base = summary(run_search(T, [], test, P))
    learned = summary(run_search(T, stops, test, P))
    print("today's search (saved views):", base)
    print("learned search order:        ", learned)
    better = (learned["within_60s"] >= base["within_60s"] and learned["found_right"] >= base["found_right"] - 0.02
              and learned["wrong_finds"] <= base["wrong_finds"] + 0.01
              and (learned["median_s_when_found"] or 1e9) < (base["median_s_when_found"] or 1e9))
    report = {"stops": stops, "prior_coverage": round(cov, 3), "test_trials": len(test), "base": base, "learned": learned,
              "gate": "pass" if better else "fail"}
    print(json.dumps(report, indent=1))
    if better:
        path = os.path.join(ROOT, "models", "follow_policy.json")
        doc = json.load(open(path)) if os.path.exists(path) else {}
        doc["search_stops"] = stops
        doc.setdefault("report", {})["search"] = report
        json.dump(doc, open(path, "w"), indent=1)
        print("added search_stops to models/follow_policy.json")
    return better


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
