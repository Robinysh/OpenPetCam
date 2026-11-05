"""Between detector runs: can cheap per-frame signals keep her position?

Ground truth: the 5 fps detector + verifier boxes on the rolling recording (walks_dets.json, sim/world/walks.py).
The live watcher decodes 5 fps and computes motion on every frame, but runs the detector only every DETECT_EVERY s
(CPU budget). For each moment she is detected while moving, take the last detection GAP s earlier as what the
controller knew, and score predictors of her box centre now (share of frame width; 0.20 = Follow's re-aim edge):
  hold     the last box (today, between detections)
  motion   centroid of frame-difference pixels (160x90 grey, the watcher's own motion image), gated around her last
           position, updated frame by frame from the last detection
  flow     the last box shifted by the median optical flow inside it, frame by frame (OpenCV Farneback at 160x90)
Only frames where the camera did not move are used (motion is suppressed while the motor runs).
    python3 -m sim.world.gaptrack eval/rl/video eval/rl/walks_dets.json [gap_s ...]  -> eval/rl/gaptrack.json"""
import json
import os
import subprocess
import sys
import time

import numpy as np

from .walks import segment_start

FPS, W, H = 5, 160, 90
DIFF_T = 18          # per-pixel grey change (0-255) counted as motion


def grey(path):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-vf", f"fps={FPS},scale={W}:{H}", "-f", "rawvideo",
                          "-pix_fmt", "gray", "-"], stdout=subprocess.PIPE)
    out = []
    while True:
        b = p.stdout.read(W * H)
        if len(b) < W * H:
            break
        out.append(np.frombuffer(b, np.uint8).reshape(H, W))
    p.wait()
    return np.stack(out) if out else None


def run(video_dir, dets_path, gaps=(2.0, 4.0, 8.0)):
    import cv2
    dets = json.load(open(dets_path))
    rows = {g: [] for g in gaps}
    exits = []
    t0 = time.time()
    for name in sorted(f for f in os.listdir(video_dir) if f.endswith(".mp4")):
        s0 = segment_start(name)
        G = grey(os.path.join(video_dir, name))
        if G is None:
            continue
        n = len(G)
        box = [None] * n
        for x in dets:
            k = int(round((x["t"] - s0) * FPS))
            if 0 <= k < n:
                box[k] = x["box"]
        diff = np.abs(G[1:].astype(np.int16) - G[:-1].astype(np.int16)) > DIFF_T          # diff[k-1]: frame k vs k-1
        frac = diff.reshape(len(diff), -1).mean(1)
        cam_moved = np.r_[False, frac > 0.35]                                                # whole picture changes: our motor
        flows = {}

        def flow(k):
            if k not in flows:
                flows[k] = cv2.calcOpticalFlowFarneback(G[k - 1], G[k], None, 0.5, 2, 9, 2, 5, 1.1, 0)
            return flows[k]
        c = lambda b: ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)                                   # noqa: E731
        for k in range(n):
            if box[k] is None:
                continue
            for g in gaps:
                j = k - int(g * FPS)
                if j < 1 or box[j] is None or cam_moved[j:k + 1].any():
                    continue
                truth = np.array(c(box[k])); last = np.array(c(box[j]))
                if np.hypot(*(truth - last)) < 0.03:              # she did not move: every method is trivially right
                    continue
                m = last.copy(); f = last.copy(); bw, bh = box[j][2] - box[j][0], box[j][3] - box[j][1]
                for q in range(j + 1, k + 1):
                    # motion: centroid of changed pixels within ~one body of the current estimate
                    x0, x1 = int(max(0, (m[0] - bw) * W)), int(min(W, (m[0] + bw) * W) + 1)
                    y0, y1 = int(max(0, (m[1] - bh) * H)), int(min(H, (m[1] + bh) * H) + 1)
                    sub = diff[q - 1][y0:y1, x0:x1]
                    if sub.sum() >= 6:
                        ys, xs = np.nonzero(sub)
                        m = np.array([(x0 + xs.mean()) / W, (y0 + ys.mean()) / H])
                    # flow: median flow inside the current box estimate
                    fx0, fx1 = int(max(0, (f[0] - bw / 2) * W)), int(min(W, (f[0] + bw / 2) * W) + 1)
                    fy0, fy1 = int(max(0, (f[1] - bh / 2) * H)), int(min(H, (f[1] + bh / 2) * H) + 1)
                    if fx1 > fx0 and fy1 > fy0:
                        v = flow(q)[fy0:fy1, fx0:fx1].reshape(-1, 2)
                        mag = np.hypot(v[:, 0], v[:, 1])
                        if (mag > 0.3).sum() >= 4:
                            f = f + np.median(v[mag > 0.3], 0) / [W, H]
                    m = m.clip(0, 1); f = f.clip(0, 1)
                e = lambda p: float(np.hypot(*(p - truth)))       # noqa: E731
                rows[g].append({"seg": name, "k": k, "hold": e(last), "motion": e(m), "flow": e(f)})
        ks = [k for k in range(n) if box[k] is not None]
        for a, b in zip(ks, ks[1:] + [n + 100]):
            if b - a < 3 * FPS or a < FPS + 1 or cam_moved[a - FPS:a + 1].any():
                continue
            bx = box[a]; side = -1 if bx[0] < 0.03 else (1 if bx[2] > 0.97 else 0)
            if side == 0:
                continue
            fx0, fx1 = int(bx[0] * W), int(bx[2] * W) + 1; fy0, fy1 = int(bx[1] * H), int(bx[3] * H) + 1
            v = np.concatenate([flow(q)[fy0:fy1, fx0:fx1, 0].ravel() for q in range(a - FPS + 1, a + 1)])
            # motion after the last sighting: which side of the picture keeps changing for the next 2 s
            after = diff[a:min(n - 1, a + 2 * FPS)].sum(0) if a < n - 1 else None
            mside = 0 if after is None or after.sum() < 6 else (1 if (after[:, W // 2:].sum() > after[:, :W // 2].sum()) else -1)
            exits.append({"seg": name, "k": a, "side": side, "flow_side": int(np.sign(np.median(v))) if len(v) else 0,
                          "motion_after_side": mside})
        print(name, {g: len(v) for g, v in rows.items()}, f"{time.time() - t0:.0f}s", flush=True)
    rng = np.random.default_rng(0)
    res = {}
    for g, R in rows.items():
        if not R:
            continue
        segs = sorted({r["seg"] for r in R})
        out = {"n": len(R), "segments": len(segs)}
        for meth in ("hold", "motion", "flow"):
            v = np.array([r[meth] for r in R])
            out[meth] = {"median": round(float(np.median(v)), 3), "share_within_edge_0.20": round(float((v < 0.20).mean()), 3)}
        for meth in ("motion", "flow"):
            # paired, resampled by segment (frames within a walk are not independent)
            bys = {s: np.array([r["hold"] - r[meth] for r in R if r["seg"] == s]) for s in segs}
            bs = [float(np.mean(np.concatenate([bys[s] for s in rng.choice(segs, len(segs))]))) for _ in range(2000)]
            out[meth]["hold_minus_" + meth + "_mean"] = round(float(np.mean([r["hold"] - r[meth] for r in R])), 3)
            out[meth]["ci95"] = [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
        big = [r for r in R if r["hold"] >= 0.15]                 # she really walked across the picture
        if big:
            out["walked_0.15+"] = {"n": len(big), **{m: {"median": round(float(np.median([r[m] for r in big])), 3),
                                   "share_within_edge_0.20": round(float(np.mean([r[m] < 0.20 for r in big])), 3)} for m in ("hold", "motion", "flow")}}
        res[str(g)] = out
    res["exits"] = exits
    json.dump(res, open(os.path.join(os.path.dirname(dets_path), "gaptrack.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    a = sys.argv[1:]
    print(json.dumps(run(a[0], a[1], tuple(float(x) for x in a[2:]) or (2.0, 4.0, 8.0)), indent=1))
