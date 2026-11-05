"""Feature matches for calibration from ALIKED + LightGlue (BSD/Apache, ~50 MB of weights): far more matches than SIFT
on white walls and repeating doors. Run in the model environment, writes eval/world/raw/matches_lightglue.npz, which
camera.calibrate() uses in place of SIFT when present.
    ~/.venvs/da3/bin/python -m sim.world.matches eval/twin/day eval/world"""
import json
import os
import sys

import numpy as np


def main(twin, world):
    import torch
    from lightglue import ALIKED, LightGlue
    from lightglue.utils import load_image, rbd
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    ext = ALIKED(max_num_keypoints=2048).eval().to(dev)
    lg = LightGlue(features="aliked").eval().to(dev)
    P = json.load(open(os.path.join(twin, "poses.json")))["poses"]
    feats = {}
    S = 0.5                                                   # matched at 960 px wide, stored in full-res pixels
    for p in P:
        img = load_image(os.path.join(twin, p["file"]), resize=960).to(dev)
        img[:, :20, :350] = 0                                 # the timestamp overlay
        feats[p["file"]] = ext.extract(img)
    out = {}
    for i in range(len(P)):
        for j in range(i + 1, len(P)):
            if abs(P[i]["pan"] - P[j]["pan"]) > 400 or abs(P[i]["tilt"] - P[j]["tilt"]) > 220:
                continue
            fa, fb = feats[P[i]["file"]], feats[P[j]["file"]]
            r = rbd(lg({"image0": fa, "image1": fb}))
            mm = r["matches"].cpu().numpy()
            if len(mm) < 12:
                continue
            ka, kb = rbd(fa)["keypoints"].cpu().numpy(), rbd(fb)["keypoints"].cpu().numpy()
            out[f"{i}_{j}"] = np.concatenate([ka[mm[:, 0]], kb[mm[:, 1]]], 1) / S
        print(f"{i + 1}/{len(P)}", flush=True)
    np.savez_compressed(os.path.join(world, "raw", "matches_lightglue.npz"), **out)
    print(len(out), "pairs matched")


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main(*sys.argv[1:3])
