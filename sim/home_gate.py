#!/usr/bin/env python3
"""Is the room-model simulator (sim/home.py) more truthful than the schematic one (sim/room.py)? Held out by day.

For every day with positions (the pose log starts 2026-11-03), both simulators are fitted on the OTHER days only:
the same logged-statistics fit (fit_from_logs), the same calibration budget (CMA-ES, same seeds), the same real
controller (server/tracker.py). HomeRoom takes her stays, absences and where she leaves and returns from the training
days' detections, and calibrates only what is not observable (walking speed, hard poses, people motion); Room
calibrates its behaviour parameters as before. Each then predicts the held-out day's real logged statistics with 8
new seeds. Score per day: how many real metrics fall inside the simulator's range, and the mean error in units of
the calibration tolerance (calibrate.TARGETS). The room model is worth having for simulation only if HomeRoom wins.
    python3 sim/home_gate.py [world] [export] [follow] [evals]   ->  eval/rl/home_gate.json"""
import functools
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "eval", "rl")
sys.path.insert(0, HERE)
sys.path.append(os.path.join(ROOT, "tools"))
import calibrate  # noqa: E402
import fit_from_logs  # noqa: E402
import gate1  # noqa: E402
import validate_real as VR  # noqa: E402
from home import HomeData, HomeRoom  # noqa: E402
from room import Room, load_params  # noqa: E402

ROOM_SPACE = list(calibrate.SPACE)
HOME_SPACE = [("people_motion_per_h", 0.5, 30, True),
              ("q_hard", 0.0, 0.8, False), ("p_hard", 0.02, 0.9, True)]


def sharpness(rep):
    """Mean width of the simulator's 8-seed range, in calibration tolerances: a wide range makes 'inside' easy."""
    tol = {"/".join(map(str, (k[0],) + ((k[1],) if k[1] is not None else ()))): k[2] for k in calibrate.TARGETS}
    w = [(v["sim_max"] - v["sim_min"]) / tol[n] for n, v in rep.items() if n in tol]
    return round(sum(w) / max(len(w), 1), 3)


def score(rep):
    tol = {"/".join(map(str, (k[0],) + ((k[1],) if k[1] is not None else ()))): k[2] for k in calibrate.TARGETS}
    inside, errs = 0, []
    for name, v in rep.items():
        real = v["real"] or 0.0
        inside += v["sim_min"] <= real <= v["sim_max"]
        if name in tol:
            errs.append(abs(v["sim_mean"] - real) / tol[name])
    return inside, round(sum(errs) / max(len(errs), 1), 3)


def clean_follow(follow):
    """The real follow log without the 2026-11-04 PPO smoke test's 201 fake searches ('search (ppo)'), relabelled
    search_smoketest in the database but still in the journal: they are not behaviour either simulator should match."""
    out = follow + ".clean"
    lines = [ln for ln in open(follow) if "search (ppo)" not in ln]
    open(out, "w").writelines(lines)
    return out


def main(world=os.path.join(ROOT, "eval", "world"), export=os.path.join(OUT, "rl_export.json"),
         follow=os.path.join(OUT, "rl_follow.log"), evals=60, walks=os.path.join(OUT, "walks.json")):
    from zoneinfo import ZoneInfo
    try:
        from siteconf import SITE
        tz = ZoneInfo(SITE.get("TZ", "UTC"))
    except Exception:  # noqa: BLE001
        tz = ZoneInfo("UTC")
    evals = int(evals)
    follow = clean_follow(follow)
    Room.HARD_CORRELATED = True                       # v3: pose-consistent detection, for both simulators
    d = json.load(open(export))
    pose_days = sorted({VR.day_of(r[0], tz) for r in d["track"]})
    obs_days = {}
    for r in d["obs"]:
        obs_days[VR.day_of(r[0], tz)] = obs_days.get(VR.day_of(r[0], tz), 0) + 1
    days = [x for x in pose_days if obs_days.get(x, 0) > 6 * 3600 / 3.5]
    print("held-out days (poses logged, >= 6 h of cycles):", days, flush=True)
    results = {}
    for test_day in days:
        parts = VR.split(export, follow, tz, test_day)
        train_days = {x for x in days if x != test_day}
        p_train, p_test = os.path.join(OUT, "hg_params_train.json"), os.path.join(OUT, "hg_params_test.json")
        for p in (p_train, p_test):
            shutil.copy(os.path.join(HERE, "params.json"), p)
        fit_from_logs.main(parts["train"][0], parts["train"][1], p_train)
        fit_from_logs.main(parts["test"][0], parts["test"][1], p_test)
        real = load_params(p_test)["real_metrics"]
        owner = load_params(p_test)["free"].get("owner_moves_per_h", 0.0)
        day = {"train_days": sorted(train_days)}
        for name, room, space in (("schematic", Room, ROOM_SPACE), ("room_model", None, HOME_SPACE)):
            if room is None:
                data = HomeData.from_world(world, export, train_days=train_days, tz=tz,
                                           walks=walks if os.path.exists(walks) else None)
                room = functools.partial(HomeRoom, home=data)
                day["home_data"] = data.summary()
            gate1.ROOM, calibrate.SPACE = room, space
            shutil.copy(p_train, p_train + f".{name}")
            P = calibrate.main(evals, p_train + f".{name}", write=True)
            free = dict(P["free"], owner_moves_per_h=owner)
            rows = gate1.simulate(4, 8, free=free, params=P, seed0=500)
            rep = gate1.compare(rows, real)
            inside, err = score(rep)
            day[name] = {"inside": f"{inside}/{len(rep)}", "mean_error_in_tolerances": err, "range_width_in_tolerances": sharpness(rep),
                         "free": P["free"], "metrics": rep}
            print(f"{test_day} {name}: mean error {err} tolerances, {inside}/{len(rep)} inside a range {sharpness(rep)} tolerances wide", flush=True)
        gate1.ROOM, calibrate.SPACE = Room, ROOM_SPACE
        results[test_day] = day
        json.dump(results, open(os.path.join(OUT, "home_gate.json"), "w"), indent=1, default=str)
    tot = {n: [sum(int(r[n]["inside"].split("/")[0]) for r in results.values()), sum(int(r[n]["inside"].split("/")[1]) for r in results.values()),
               round(sum(r[n]["mean_error_in_tolerances"] for r in results.values()) / max(len(results), 1), 3)] for n in ("schematic", "room_model")}
    print("TOTAL", json.dumps(tot), flush=True)
    results["_total"] = tot
    json.dump(results, open(os.path.join(OUT, "home_gate.json"), "w"), indent=1, default=str)
    return results


if __name__ == "__main__":
    main(*sys.argv[1:])
