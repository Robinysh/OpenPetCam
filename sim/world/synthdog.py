"""Insert the pet into camera views, and prove the composites are faithful before using them.

Cutout: a frame where she is verified, and a background for the same pose from the same recording at moments she was
not in that part of the picture (median of BG_N frames, so lighting matches). Her matte is where the frame differs
from that background inside her box (colour distance over a threshold, cleaned by a small opening and the largest
component).
Realism check (the gate for everything else): paste each cutout back onto its own background at exactly where she
was, and run the same detector and verifier on the composite and on the original frame. The composites must be
detected at the same rate as the originals (paired, with a bootstrap interval) before synthetic insertions are used
to test the detector anywhere else.
    python3 -m sim.world.synthdog eval/world eval/rl/rl_export.json eval/rl/video [n]  -> eval/rl/synthdog_check.json"""
import json
import os
import subprocess
import sys

import numpy as np

W, H = 640, 360
BG_N, BG_GAP_S = 9, 20.0


def frame_at(path, rel_s):
    out = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, rel_s):.2f}", "-i", path, "-frames:v", "1", "-vf",
                          f"scale={W}:{H}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], capture_output=True, timeout=60).stdout
    return np.frombuffer(out, np.uint8).reshape(H, W, 3) if len(out) == W * H * 3 else None


def matte(img, bg, box, thr=0.12):
    from scipy import ndimage
    x0, y0, x1, y1 = (int(box[0] * W), int(box[1] * H), int(np.ceil(box[2] * W)), int(np.ceil(box[3] * H)))
    m = np.zeros((H, W), bool)
    diff = np.linalg.norm(img[y0:y1, x0:x1].astype(float) - bg[y0:y1, x0:x1].astype(float), axis=-1) / 255
    sub = ndimage.binary_opening(diff > thr, iterations=1)
    lab, n = ndimage.label(sub)
    if n:
        sizes = ndimage.sum(sub, lab, range(1, n + 1))
        sub = lab == (1 + int(np.argmax(sizes)))
        sub = ndimage.binary_fill_holes(ndimage.binary_closing(sub, iterations=2))
    m[y0:y1, x0:x1] = sub
    return m


def main(world, export, video_dir, n=120, seed=0):
    from PIL import Image
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, os.path.join(root, "server"))
    from detector import Detector
    from .walks import pose_interp, segment_start
    src = open(os.path.join(root, "server", "watch.py")).read()
    ns = {}
    exec(compile("import json, os\nimport numpy as np\nfrom PIL import Image\nVERIFIER=''\n" +
                 src[src.index("class Verifier:"):src.index("class SoundModel:")], "verifier", "exec"), ns)
    V = ns["Verifier"](os.path.join(root, "models", "verifier.json"))
    det = Detector(os.path.join(root, "models", "yolox_nano.onnx"), size=416)

    def verified(img):
        best = None
        for f in sorted([f for f in det.detect(Image.fromarray(img), min_score=0.10) if f["label"] != "person"], key=lambda f: -f["score"])[:3]:
            p = V.prob(Image.fromarray(img), f["box"])
            if p >= V.threshold and (best is None or p > best[0]):
                best = (p, f["box"])
        return best

    d = json.load(open(export))
    tr = sorted(d["track"]); tt = np.array([r[0] for r in tr]); pan = np.array([r[1] for r in tr], float); tilt = np.array([r[2] for r in tr], float)
    dets = json.load(open(os.path.join(os.path.dirname(export), "walks_dets.json")))
    segs = {f: segment_start(f) for f in os.listdir(video_dir) if f.endswith(".mp4")}
    seg_of = lambda t: next((f for f, s in sorted(segs.items(), key=lambda x: x[1]) if s <= t < s + 600), None)   # noqa: E731
    rng = np.random.default_rng(seed)
    pick = [dets[i] for i in rng.choice(len(dets), min(n * 3, len(dets)), replace=False)]
    rows = []
    for x in pick:
        if len(rows) >= n:
            break
        t, box = x["t"], x["box"]
        f = seg_of(t)
        pose = pose_interp(tt, pan, tilt, t)
        if f is None or pose is None:
            continue
        # background: frames of the same segment at the same pose (within 3 steps), >= BG_GAP_S away, without her in this box
        cands = [y for y in dets if abs(y["t"] - t) >= BG_GAP_S and seg_of(y["t"]) == f]
        same = []
        for y in rng.permutation(len(cands))[:60] if cands else []:
            y = cands[int(y)]
            p2 = pose_interp(tt, pan, tilt, y["t"])
            if p2 and abs(p2[0] - pose[0]) <= 3 and abs(p2[1] - pose[1]) <= 3:
                b = y["box"]
                overlap = max(0, min(b[2], box[2]) - max(b[0], box[0])) * max(0, min(b[3], box[3]) - max(b[1], box[1]))
                if overlap <= 0.1 * (box[2] - box[0]) * (box[3] - box[1]):
                    same.append(y["t"])
            if len(same) >= BG_N:
                break
        if len(same) < 3:
            continue
        img = frame_at(os.path.join(video_dir, f), t - segs[f])
        bgs = [frame_at(os.path.join(video_dir, f), s - segs[f]) for s in same]
        bgs = [b for b in bgs if b is not None]
        if img is None or len(bgs) < 3:
            continue
        bg = np.median(np.stack(bgs), 0).astype(np.uint8)
        m = matte(img, bg, box)
        if m.sum() < 0.15 * (box[2] - box[0]) * W * (box[3] - box[1]) * H:
            continue                                         # matte failed (she blends into the background): skip, counted
        comp = bg.copy(); comp[m] = img[m]
        empty = verified(bg)
        rows.append({"t": t, "orig": verified(img) is not None, "composite": verified(comp) is not None,
                     "background_alone": empty is not None, "matte_px": int(m.sum())})
    o = np.array([r["orig"] for r in rows]); c = np.array([r["composite"] for r in rows]); e = np.array([r["background_alone"] for r in rows])
    bs = [float(c[i].mean() - o[i].mean()) for i in (rng.integers(0, len(rows), len(rows)) for _ in range(2000))] if rows else []
    res = {"n": len(rows), "original_detected": round(float(o.mean()), 3) if rows else None,
           "composite_detected": round(float(c.mean()), 3) if rows else None,
           "background_alone_detected": round(float(e.mean()), 3) if rows else None,
           "difference_95ci": [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)] if bs else None,
           "agree_per_frame": round(float((o == c).mean()), 3) if rows else None, "rows": rows}
    res["faithful"] = bool(bs and np.percentile(bs, 2.5) <= 0 <= np.percentile(bs, 97.5) and res["background_alone_detected"] < 0.1)
    json.dump(res, open(os.path.join(os.path.dirname(export), "synthdog_check.json"), "w"), indent=1)
    return {k: v for k, v in res.items() if k != "rows"}


if __name__ == "__main__":
    a = sys.argv[1:]
    print(json.dumps(main(a[0], a[1], a[2], int(a[3]) if len(a) > 3 else 120), indent=1))
