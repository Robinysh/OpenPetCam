"""More detector runs for the same CPU: what cheaper detector inputs cost and what they miss, on recorded frames.

Variants (each followed by the same verifier and threshold as the watcher):
  square    today: the 640x360 frame letterboxed into the model's fixed 416x416 (44% of the input is grey padding)
  rect      the same pixels at 256x416, no padding (needs models/yolox_nano_dyn.onnx, tools/make_dyn_detector.py)
  crop256 / crop192   a square window around where she was ANCHOR_S earlier (2.5x her box, at least 1/3 of the
            frame height), run at 256 or 192 pixels: what a tracker-guided detector would look at
  top1      'rect', but the verifier judges only the best candidate (it costs ~19 ms per candidate, more than the
            detector itself when three are checked)
Positives: frames where the full-frame detector + verifier found her (walks_dets.json). They were chosen BY the square
variant, so its recall on them is 1 by construction; the others are scored as agreement with it. Negatives: frames
with no detection within 3 s; anything verified there is a false alarm (for the crops, the window is placed where
she was seen last in that segment). Time per variant is measured on this machine; the box is slower, the ratios hold.
    python3 -m sim.world.detcost eval/rl/video eval/rl/walks_dets.json [n_pos] [n_neg]  -> eval/rl/detcost.json"""
import json
import os
import sys
import time

import numpy as np

from .walks import frames, segment_start

ANCHOR_S = 3.0
FPS = 5


def main(video_dir, dets_path, n_pos=400, n_neg=300, seed=0):
    from PIL import Image
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, os.path.join(root, "server"))
    from detector import Detector, detect_crop
    src = open(os.path.join(root, "server", "watch.py")).read()
    ns = {}
    exec(compile("import json, os\nimport numpy as np\nfrom PIL import Image\nVERIFIER=''\n" +
                 src[src.index("class Verifier:"):src.index("class SoundModel:")], "verifier", "exec"), ns)
    V = ns["Verifier"](os.path.join(root, "models", "verifier.json"))
    sq = Detector(os.path.join(root, "models", "yolox_nano.onnx"), size=416)
    dyn = Detector(os.path.join(root, "models", "yolox_nano_dyn.onnx"), shape=(256, 416))
    dets = json.load(open(dets_path))
    rng = np.random.default_rng(seed)
    segs = sorted(f for f in os.listdir(video_dir) if f.endswith(".mp4"))
    by_seg = {f: [] for f in segs}
    for x in dets:
        for f in segs:
            s0 = segment_start(f)
            if s0 <= x["t"] < s0 + 600:
                by_seg[f].append(x)
                break
    timing = {k: [] for k in ("square", "rect", "crop256", "crop192", "verify")}

    def timed(k, fn):
        t = time.perf_counter(); r = fn(); timing[k].append(time.perf_counter() - t); return r

    def verified(img, found, top=3):
        best = None
        for f in sorted([f for f in found if f["label"] != "person"], key=lambda f: -f["score"])[:top]:
            p = timed("verify", lambda: V.prob(img, f["box"]))
            if p >= V.threshold and (best is None or p > best[0]):
                best = (p, f["box"])
        return best

    def window(b):
        cx, cy = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
        side = max(2.5 * max((b[2] - b[0]) * 640, (b[3] - b[1]) * 360), 120)
        side = min(side, 360)
        x0 = min(max(0, cx * 640 - side / 2), 640 - side); y0 = min(max(0, cy * 360 - side / 2), 360 - side)
        return (x0 / 640, y0 / 360, (x0 + side) / 640, (y0 + side) / 360)

    rows = []
    for f in segs:
        s0 = segment_start(f)
        D = by_seg[f]
        if not D:
            continue
        idx = {int(round((x["t"] - s0) * FPS)): x for x in D}
        ts = np.array(sorted(x["t"] for x in D))
        pos = [k for k in idx if (k - ANCHOR_S * FPS) in idx]
        pos = set(rng.choice(pos, min(len(pos), n_pos // len(segs) + 1), replace=False).tolist()) if pos else set()
        neg_c = [k for k in range(0, 600 * FPS, 3) if not len(ts) or np.min(np.abs(ts - (s0 + k / FPS))) > 3]
        neg = set(rng.choice(neg_c, min(len(neg_c), n_neg // len(segs) + 1), replace=False).tolist()) if neg_c else set()
        for rel, arr in frames(os.path.join(video_dir, f)):
            k = int(round(rel * FPS))
            if k not in pos and k not in neg:
                continue
            img = Image.fromarray(arr)
            if k in pos:
                anchor = idx[k - int(ANCHOR_S * FPS)]["box"]
            else:
                prev = [x for x in D if x["t"] < s0 + rel]
                anchor = (prev[-1] if prev else D[0])["box"]
            w = window(anchor)
            fs = timed("square", lambda: sq.detect(img, min_score=0.10))
            fr = timed("rect", lambda: dyn.detect(img, min_score=0.10))
            c2 = timed("crop256", lambda: detect_crop(dyn, img, w, 256))
            c1 = timed("crop192", lambda: detect_crop(dyn, img, w, 192))
            rows.append({"seg": f, "k": k, "pos": k in pos, "candidates": sum(x["label"] != "person" for x in fr),
                         "square": verified(img, fs) is not None, "rect": verified(img, fr) is not None,
                         "top1": verified(img, fr, top=1) is not None,
                         "crop256": verified(img, c2) is not None, "crop192": verified(img, c1) is not None})
        print(f, len(rows), flush=True)
    P = [r for r in rows if r["pos"]]; N = [r for r in rows if not r["pos"]]
    res = {"positives": len(P), "negatives": len(N),
           "ms_per_run_here": {k: round(1000 * float(np.median(v)), 1) for k, v in timing.items() if v},
           "candidates_per_frame": {"positives_mean": round(float(np.mean([r["candidates"] for r in P])), 2) if P else None,
                                    "negatives_mean": round(float(np.mean([r["candidates"] for r in N])), 2) if N else None}}
    for v in ("square", "rect", "top1", "crop256", "crop192"):
        res[v] = {"found_on_positives": round(float(np.mean([r[v] for r in P])), 3) if P else None,
                  "false_alarm_on_negatives": round(float(np.mean([r[v] for r in N])), 3) if N else None}
    json.dump({**res, "rows": rows}, open(os.path.join(os.path.dirname(dets_path), "detcost.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    a = sys.argv[1:]
    print(json.dumps(main(a[0], a[1], *(int(x) for x in a[2:4])), indent=1))
