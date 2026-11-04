#!/usr/bin/env python3
"""Calibrate one simulator per real day (her behaviour differs a lot between days: 1.8 vs 7.8 tracking aims an hour on
2026-11-02/03), so training can randomise ACROSS days instead of around one average. Writes sim/params.json "free_days".

    python3 sim/per_day.py [export.json] [follow.log]
"""
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
import validate_real as VR  # noqa: E402
from room import load_params  # noqa: E402


def main(export=os.path.join(OUT, "rl_export.json"), follow=os.path.join(OUT, "rl_follow.log")):
    from zoneinfo import ZoneInfo
    try:
        from siteconf import SITE
        tz = ZoneInfo(SITE.get("TZ", "UTC"))
    except Exception:  # noqa: BLE001
        tz = ZoneInfo("UTC")
    obs = json.load(open(export))["obs"]
    days = sorted({VR.day_of(r[0], tz) for r in obs})
    days = [d for d in days if sum(1 for r in obs if VR.day_of(r[0], tz) == d) > 8 * 3600 / 3.5]
    fits = {}
    for d in days:
        parts = VR.split(export, follow, tz, d)            # "test" part = that day alone
        p = os.path.join(OUT, f"day_{d}.json")
        shutil.copy(os.path.join(HERE, "params.json"), p)
        fit_from_logs.main(parts["test"][0], parts["test"][1], p)
        P = calibrate.main(100, p, write=True)
        fits[d] = {"free": P["free"], "loss": P["calibration"]["loss"], "real": P["real_metrics"]}
        print(f"day {d}: calibration loss {P['calibration']['loss']}", flush=True)
    path = os.path.join(HERE, "params.json")
    P = load_params(path)
    P["free_days"] = {d: f["free"] for d, f in fits.items()}
    json.dump(P, open(path, "w"), indent=1)
    json.dump(fits, open(os.path.join(OUT, "per_day.json"), "w"), indent=1)
    return fits


if __name__ == "__main__":
    main(*sys.argv[1:])
