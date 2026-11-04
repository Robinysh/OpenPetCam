#!/usr/bin/env python3
"""Does the simulator predict REAL behaviour it was not fitted on? Leave one real day out: fit and calibrate the
simulator on the other days only, then predict the held-out day's logged statistics with the real controller.

    python3 sim/validate_real.py [export.json] [follow.log]     -> eval/rl/validate_real.json

A metric passes when the held-out day's real value lies inside the simulator's range over 8 seeds. This is the
check that new random seeds cannot give: the generator against reality, on data it never saw.
"""
import datetime as dt
import json
import os
import re
import shutil
import sys
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.path.join(ROOT, "eval", "rl")
sys.path.insert(0, HERE)
sys.path.append(os.path.join(ROOT, "tools"))   # after sim/: tools/ has its own calibrate.py
import calibrate  # noqa: E402
import fit_from_logs  # noqa: E402
import gate1  # noqa: E402
from room import load_params  # noqa: E402


def day_of(ts, tz):
    return dt.datetime.fromtimestamp(ts, tz).strftime("%Y-%m-%d")


def split(export, follow, tz, test_day):
    d = json.load(open(export))
    lines = open(follow).read().splitlines() if os.path.exists(follow) else []
    keep = lambda ts, test: (day_of(ts, tz) == test_day) == test      # noqa: E731
    parts = {}
    for test in (False, True):
        sub = {k: [r for r in v if keep(r[0], test)] for k, v in d.items()}
        fl = [ln for ln in lines if re.match(r"[\d.]+", ln) and keep(float(ln.split()[0]), test)]
        tag = "test" if test else "train"
        e, f = os.path.join(OUT, f"vr_{tag}_export.json"), os.path.join(OUT, f"vr_{tag}_follow.log")
        json.dump(sub, open(e, "w"))
        open(f, "w").write("\n".join(fl) + "\n")
        parts[tag] = (e, f, len(sub["obs"]))
    return parts


def main(export=os.path.join(OUT, "rl_export.json"), follow=os.path.join(OUT, "rl_follow.log")):
    try:
        from siteconf import SITE
        tz = ZoneInfo(SITE.get("TZ", "UTC"))
    except Exception:  # noqa: BLE001
        tz = ZoneInfo("UTC")
    obs = json.load(open(export))["obs"]
    by_day = {}
    for r in obs:
        by_day[day_of(r[0], tz)] = by_day.get(day_of(r[0], tz), 0) + 1
    days = [d for d, n in sorted(by_day.items()) if n > 8 * 3600 / 3.5]        # at least ~8 hours of cycles
    print("days with enough data:", days, flush=True)
    results = {}
    for test_day in days:
        parts = split(export, follow, tz, test_day)
        if parts["train"][2] < 8 * 3600 / 3.5:
            continue
        p_train = os.path.join(OUT, "vr_params_train.json")
        p_test = os.path.join(OUT, "vr_params_test.json")
        for p in (p_train, p_test):                       # start from the measured camera (motor, geometry, backlash)
            shutil.copy(os.path.join(HERE, "params.json"), p)
        fit_from_logs.main(parts["train"][0], parts["train"][1], p_train)
        fit_from_logs.main(parts["test"][0], parts["test"][1], p_test)
        P = calibrate.main(100, p_train, write=True)
        real_test = load_params(p_test)["real_metrics"]
        free = dict(P["free"], owner_moves_per_h=load_params(p_test)["free"].get("owner_moves_per_h", 0.0))
        rows = gate1.simulate(4, 8, free=free, params=P, seed0=500)   # that day's measured joystick use, nothing else
        rep = gate1.compare(rows, real_test)
        inside = sum(v["sim_min"] <= (v["real"] or 0) <= v["sim_max"] for v in rep.values())
        results[test_day] = {"inside": f"{inside}/{len(rep)}", "metrics": rep}
        print(f"held-out real day {test_day}: {inside}/{len(rep)} real metrics inside the simulator's range", flush=True)
    json.dump(results, open(os.path.join(OUT, "validate_real.json"), "w"), indent=1)
    return results


if __name__ == "__main__":
    main(*sys.argv[1:])
