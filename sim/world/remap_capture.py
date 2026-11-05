"""Capture the views the next-best-view planner asked for (raw/nbv_plan.json), silently, and build a new sweep folder
= the reference sweep's frames + these, so the whole pipeline (matches, calibration, DA3, fusion) reruns on it.
    python3 -m sim.world.remap_capture eval/world eval/twin/day eval/twin/day_remap1 [--owner-present]"""
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import sweep as SW  # noqa: E402  (sim/sweep.py: goto, snapshot_main, hold_follow)


def main(world, ref_twin, out_twin):
    plan = json.load(open(os.path.join(world, "raw", "nbv_plan.json")))["picks"]
    S = SW.S
    if S.server("sudo cat /var/lib/petcam/privacy 2>/dev/null || echo 0").strip() == "1":
        sys.exit("privacy is on")
    os.makedirs(out_twin, exist_ok=True)
    ref = json.load(open(os.path.join(ref_twin, "poses.json")))
    for p in ref["poses"]:
        dst = os.path.join(out_twin, p["file"])
        if not os.path.exists(dst):
            shutil.copyfile(os.path.join(ref_twin, p["file"]), dst)
    poses = list(ref["poses"])
    n = len(poses)
    start = json.loads(S.camera("motors -j"))
    try:
        for k, pk in enumerate(plan):
            S.hold_follow()
            SW.goto(max(0, pk["pan"] - 150), pk["tilt"])          # same approach side as the sweep (gear play)
            px, py = SW.goto(pk["pan"], pk["tilt"])
            img = SW.snapshot_main()
            f = f"{n + k:02d}_{px}_{py}_nbv.jpg"
            img.save(os.path.join(out_twin, f), quality=90)
            poses.append({"file": f, "pan": px, "tilt": py, "t": time.time(), "nbv": True})
            print(f"{k + 1}/{len(plan)} pan {px} tilt {py}", flush=True)
    finally:
        S.hold_follow()
        try:
            SW.goto(int(start["xpos"]), int(start["ypos"]))
        except Exception as e:
            print("return to start failed:", e, flush=True)
    json.dump(dict(ref, tag=os.path.basename(out_twin), poses=poses), open(os.path.join(out_twin, "poses.json"), "w"), indent=1)
    print("wrote", out_twin, len(poses), "frames")


if __name__ == "__main__":
    main(*sys.argv[1:4])
