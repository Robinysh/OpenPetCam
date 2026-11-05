"""Replay: does the per-frame motion tracker (server/fasttrack.py) know where she is better than today's Follow?

Ground truth: the 5 fps detector + verifier boxes on the rolling recording (walks_dets.json). Detector runs are
replayed with the LIVE timing: gaps drawn from the measured intervals between fresh boxes while she moved (the
watcher's own obs log), each answer arriving LAG s after its frame was taken, and missing when the 5 fps truth has
no box in that frame. Beliefs about her centre, scored on every frame that has a truth box and no camera motion:
  today    the last box delivered (what the watcher hands Follow between detector runs)
  today_v  the last box moved by the velocity of the last two delivered boxes, over its age (Follow's own prediction)
  fast     FastTrack anchored at every delivered box and fed every frame's motion
Scores: error (share of frame width), share within EDGE (Follow re-aims beyond it), and reaction: each time the truth
leaves the central zone (|offset| > EDGE), seconds until the belief is also beyond EDGE on that side (censored at
10 s). Paired by recording segment, over DRAWS draws of detector timing.
    python3 -m sim.world.fastloop_replay eval/rl/video eval/rl/walks_dets.json eval/rl/rl_export.json -> eval/rl/fastloop_replay.json"""
import json
import os
import sys

import numpy as np

from .gaptrack import grey
from .walks import segment_start

FPS, LAG, EDGE, DRAWS, CENSOR = 5, 0.9, 0.20, 10, 10.0


def live_gaps(export):
    o = [r for r in json.load(open(export))["obs"] if r[3] and r[6]]
    t = [r[0] for r in o]; b = [r[3] for r in o]
    ch = [t[i] for i in range(1, len(b)) if b[i] != b[i - 1] and t[i] - t[i - 1] < 30]
    g = np.diff(ch)
    return g[(g > 0.4) & (g < 60)]


_CACHE = {}


def main(video_dir, dets_path, export, seed=0, params=None, segs_only=None, write=True):
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, os.path.join(root, "server"))
    import fasttrack
    from fasttrack import FastTrack
    for k_, v_ in (params or {}).items():
        setattr(fasttrack, k_, v_)
    gaps = live_gaps(export)
    dets = json.load(open(dets_path))
    rng = np.random.default_rng(seed)
    rows, reacts = [], []
    for name in sorted(f for f in os.listdir(video_dir) if f.endswith(".mp4")):
        if segs_only is not None and name not in segs_only:
            continue
        s0 = segment_start(name)
        if name not in _CACHE:
            _CACHE[name] = grey(os.path.join(video_dir, name))
        G = _CACHE[name]
        if G is None:
            continue
        n = len(G)
        truth = [None] * n
        for x in dets:
            k = int(round((x["t"] - s0) * FPS))
            if 0 <= k < n:
                truth[k] = x["box"]
        if sum(b is not None for b in truth) < 20:
            continue
        frac = np.r_[0, (np.abs(G[1:].astype(np.int16) - G[:-1]) > 18).reshape(n - 1, -1).mean(1)]
        moving_cam = frac > 0.35
        if ("m", name) not in _CACHE:
            _CACHE[("m", name)] = [None] + [FastTrack.mask(G[k], G[k - 1]) for k in range(1, n)]
        masks = _CACHE[("m", name)]
        c = lambda b: ((b[0] + b[2]) / 2 - 0.5, (b[1] + b[3]) / 2 - 0.5)          # noqa: E731
        for draw in range(DRAWS):
            ft = FastTrack()
            t_next = rng.uniform(0, 3)
            delivered = []                                 # (arrival_t, capture_t, box)
            pending = []
            beliefs = {"today": [None] * n, "today_v": [None] * n, "fast": [None] * n}
            for k in range(n):
                t = k / FPS
                if t >= t_next:                            # a detector run on the frame taken LAG s ago
                    kc = int(round((t - LAG) * FPS))
                    if 0 <= kc < n and truth[kc] is not None and not moving_cam[max(0, kc - 2):kc + 1].any():
                        pending.append((t, kc / FPS, truth[kc]))
                    t_next = t + float(rng.choice(gaps))
                if k:
                    ft.frame(t, masks[k], camera_moving=bool(moving_cam[k]))
                for p in [p for p in pending if p[0] <= t]:
                    pending.remove(p); delivered.append(p); ft.anchor(p[1], p[2])
                if not delivered:
                    continue
                a, tc, b = delivered[-1]
                beliefs["today"][k] = c(b)
                if len(delivered) >= 2 and 0.8 <= tc - delivered[-2][1] <= 8:
                    (x1, y1), (x0, y0) = c(b), c(delivered[-2][2]); dt = tc - delivered[-2][1]
                    age = min(t - tc, 4.0)
                    beliefs["today_v"][k] = (x1 + (x1 - x0) / dt * age, y1 + (y1 - y0) / dt * age)
                else:
                    beliefs["today_v"][k] = c(b)
                e = ft.estimate(t)
                beliefs["fast"][k] = (e[0], e[1]) if e else c(b)
            for k in range(n):
                if truth[k] is None or beliefs["today"][k] is None or moving_cam[max(0, k - 2):k + 1].any():
                    continue
                tx, ty = c(truth[k])
                r = {"seg": name, "draw": draw, "walked": abs(beliefs["today"][k][0] - tx) >= 0.15}
                for m, B in beliefs.items():
                    r[m] = float(np.hypot(B[k][0] - tx, B[k][1] - ty))
                rows.append(r)
            # reaction: truth crosses out of the centre zone (from inside, with truth on both sides of the crossing)
            for k in range(1, n):
                if truth[k] is None or truth[k - 1] is None or beliefs["today"][k] is None:
                    continue
                x0_, x1_ = c(truth[k - 1])[0], c(truth[k])[0]
                if abs(x0_) <= EDGE < abs(x1_) and not moving_cam[k]:
                    side = np.sign(x1_)
                    rr = {"seg": name, "draw": draw}
                    for m, B in beliefs.items():
                        d = CENSOR
                        for j in range(k, min(n, k + int(CENSOR * FPS))):
                            if moving_cam[j]:
                                break
                            if B[j] is not None and np.sign(B[j][0]) == side and abs(B[j][0]) > EDGE:
                                d = (j - k) / FPS
                                break
                        rr[m] = d
                    reacts.append(rr)
        if write:
            print(name, len(rows), len(reacts), flush=True)
    segs = sorted({r["seg"] for r in rows})

    def summ(R, keys, stat):
        out = {}
        for m in keys:
            out[m] = round(float(stat(np.array([r[m] for r in R]))), 3)
        bys = {s: [r for r in R if r["seg"] == s] for s in segs}
        bs = []
        for _ in range(2000):
            pick = [r for s in rng.choice(segs, len(segs)) for r in bys[s]]
            if pick:
                bs.append(float(np.mean([r["today"] - r["fast"] for r in pick])))
        out["today_minus_fast_mean"] = round(float(np.mean([r["today"] - r["fast"] for r in R])), 3)
        out["ci95"] = [round(float(np.percentile(bs, 2.5)), 3), round(float(np.percentile(bs, 97.5)), 3)]
        out["n"] = len(R)
        return out
    W_ = [r for r in rows if r["walked"]]
    res = {"segments": len(segs), "draws": DRAWS, "live_gap_s": {"median": round(float(np.median(gaps)), 2),
                                                                  "p90": round(float(np.percentile(gaps, 90)), 1)},
           "all_frames_error_median": summ(rows, ("today", "today_v", "fast"), np.median),
           "all_frames_within_edge": {m: round(float(np.mean([r[m] < EDGE for r in rows])), 3) for m in ("today", "today_v", "fast")},
           "walked_error_median": summ(W_, ("today", "today_v", "fast"), np.median) if W_ else None,
           "walked_within_edge": {m: round(float(np.mean([r[m] < EDGE for r in W_])), 3) for m in ("today", "today_v", "fast")} if W_ else None,
           "reaction_s": {"n": len(reacts), **{m: {"median": round(float(np.median([r[m] for r in reacts])), 2),
                                                   "censored_share": round(float(np.mean([r[m] >= CENSOR for r in reacts])), 3)}
                                               for m in ("today", "today_v", "fast")}} if reacts else None}
    if reacts:
        bys = {s: [r for r in reacts if r["seg"] == s] for s in sorted({r["seg"] for r in reacts})}
        ss = list(bys)
        bs = [float(np.mean([r["today"] - r["fast"] for s in rng.choice(ss, len(ss)) for r in bys[s]])) for _ in range(2000)]
        res["reaction_s"]["today_minus_fast_mean"] = round(float(np.mean([r["today"] - r["fast"] for r in reacts])), 2)
        res["reaction_s"]["ci95"] = [round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)]
        res["reaction_s"]["segments"] = len(ss)
    res["params"] = params or {}
    if write:
        json.dump(res, open(os.path.join(os.path.dirname(dets_path), "fastloop_replay.json"), "w"), indent=1)
    return res


if __name__ == "__main__":
    a = sys.argv[1:]
    print(json.dumps(main(a[0], a[1], a[2]), indent=1))
