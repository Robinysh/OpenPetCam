"""Her real walks, from continuous video: what the watcher's 3-12 s detector cadence cannot see.

The watcher judges a frame every few seconds, and a walking dog crosses the frame in less, so walks leave almost no
trace in its records (moving detections there are mostly shifting in place, median 0.15 m/s). The server's rolling
recording is continuous video. Here every recorded segment is decoded at FPS, judged by the same detector and pet
verifier as the server, and each verified box is placed on the floor by the room model's camera: the pose from the
watcher's track (interpolated while the motor moves), the mount's measured drift, the floor height from sound.
The recording clock is checked against the watcher's own detections before anything is trusted (CLOCK check).
Consecutive placed detections link into tracks; a walk is a track that moves; its speed is path length over time.
    python3 -m sim.world.walks eval/world eval/rl/rl_export.json eval/rl/video  -> eval/rl/walks.json"""
import json
import math
import os
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

FPS, W, H = 5, 640, 360
LINK_S, LINK_M = 1.0, 1.0           # consecutive detections join a track within this gap and jump
MIN_TRACK_S = 1.0


def segment_start(name):
    return datetime.strptime(name[:19], "%Y-%m-%d_%H-%M-%S").replace(tzinfo=timezone.utc).timestamp()


def frames(path):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-vf", f"fps={FPS},scale={W}:{H}", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    k = 0
    while True:
        buf = p.stdout.read(W * H * 3)
        if len(buf) < W * H * 3:
            break
        yield k / FPS, np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        k += 1
    p.wait()


def pose_interp(tt, pan, tilt, ts, gap=4.0):
    """Pan/tilt at ts, linearly between the watcher's track rows (the motor moves at constant speed); None far from any."""
    k = int(np.searchsorted(tt, ts))
    if k <= 0 or k >= len(tt):
        return None
    a, b = k - 1, k
    if tt[b] - tt[a] > gap:
        return (pan[a], tilt[a]) if ts - tt[a] < gap else None
    f = (ts - tt[a]) / max(tt[b] - tt[a], 1e-6)
    return (pan[a] + f * (pan[b] - pan[a]), tilt[a] + f * (tilt[b] - tilt[a]))


def main(world, export, video_dir, out=None):
    from PIL import Image
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    sys.path.insert(0, os.path.join(root, "server"))
    import roomcam
    from detector import Detector
    src = open(os.path.join(root, "server", "watch.py")).read()
    ns = {}
    exec(compile("import json, os\nimport numpy as np\nfrom PIL import Image\nVERIFIER=''\n" +
                 src[src.index("class Verifier:"):src.index("class SoundModel:")], "verifier", "exec"), ns)
    V = ns["Verifier"](os.path.join(root, "models", "verifier.json"))
    det = Detector(os.path.join(root, "models", "yolox_nano.onnx"), size=416)
    calib = json.load(open(os.path.join(world, "calib.json")))
    cam = roomcam.Camera({k: calib[k] for k in ("focal_px", "k1", "k2", "deg_per_step_pan", "deg_per_step_tilt", "level_tilt", "roll_deg")})
    m = json.load(open(os.path.join(world, "model", "manifest.json")))
    floor_h = m["room"]["floor_height_m"]
    reg = os.path.join(world, "raw", "registration.json")
    smooth = (json.load(open(reg)).get("smooth") or []) if os.path.exists(reg) else []
    off = (smooth[-1][1], smooth[-1][2], 0.0) if smooth else (0.0, 0.0, 0.0)     # the latest drift (these are recent days)
    d = json.load(open(export))
    tr = sorted(d["track"])
    tt = np.array([r[0] for r in tr]); pan = np.array([r[1] for r in tr], float); tilt = np.array([r[2] for r in tr], float)
    obs_seen = np.array(sorted(r[0] for r in d["obs"] if r[5] and r[1] and r[1] >= 0.3))
    dets = []
    names = sorted(f for f in os.listdir(video_dir) if f.endswith(".mp4"))
    for i, name in enumerate(names):
        t0 = segment_start(name)
        n = 0
        for rel, rgb in frames(os.path.join(video_dir, name)):
            img = Image.fromarray(rgb)
            best = None
            for f in sorted([f for f in det.detect(img, min_score=0.10) if f["label"] != "person"], key=lambda f: -f["score"])[:3]:
                p = V.prob(img, f["box"])
                if p >= V.threshold and (best is None or p > best[0]):
                    best = (p, f["box"])
            n += 1
            if best:
                dets.append({"t": t0 + rel, "box": [round(x, 4) for x in best[1]], "p": round(best[0], 3)})
        print(f"{i + 1}/{len(names)} {name}: {n} frames, {sum(1 for x in dets if t0 <= x['t'] < t0 + 700)} with her", flush=True)
    json.dump(dets, open(os.path.join(os.path.dirname(export), "walks_dets.json"), "w"))   # per-frame boxes (synthdog.py)
    # CLOCK check: the recording's timestamps against the watcher's own sightings (cross-correlate presence per second)
    if dets and len(obs_seen):
        lo, hi = int(min(x["t"] for x in dets)), int(max(x["t"] for x in dets))
        a = np.zeros(hi - lo + 1); b = np.zeros(hi - lo + 1)
        for x in dets:
            a[int(x["t"]) - lo] = 1
        for t in obs_seen[(obs_seen >= lo) & (obs_seen <= hi)]:
            b[int(t) - lo] = 1
        lags = range(-20, 21)
        sc = [float(np.dot(a[max(0, -L):len(a) - max(0, L)], b[max(0, L):len(b) - max(0, -L)])) for L in lags]
        clock_lag = list(lags)[int(np.argmax(sc))]
    else:
        clock_lag = 0
    # place on the floor (shift by the measured clock lag), link into tracks
    placed = []
    for x in dets:
        ts = x["t"] + clock_lag
        pose = pose_interp(tt, pan, tilt, ts)
        if pose is None or x["box"][3] > 0.97:
            continue
        p = roomcam.floor_point(cam, pose[0], pose[1], x["box"], floor_h, off)
        if p:
            placed.append((ts, p[0], p[1]))
    tracks, cur = [], []
    for q in placed:
        if cur and (q[0] - cur[-1][0] > LINK_S or math.hypot(q[1] - cur[-1][1], q[2] - cur[-1][2]) > LINK_M):
            tracks.append(cur); cur = []
        cur.append(q)
    if cur:
        tracks.append(cur)
    walks = []
    for tk in tracks:
        dur = tk[-1][0] - tk[0][0]
        if dur < MIN_TRACK_S:
            continue
        path = sum(math.hypot(b[1] - a[1], b[2] - a[2]) for a, b in zip(tk, tk[1:]))
        net = math.hypot(tk[-1][1] - tk[0][1], tk[-1][2] - tk[0][2])
        # a walk moves: net displacement over 0.5 m; speed from 1 s windows, so box jitter on a still dog is not speed
        sp = []
        j = 0
        for k in range(len(tk)):
            while tk[k][0] - tk[j][0] > 1.0:
                j += 1
            if tk[k][0] - tk[j][0] >= 0.8:
                sp.append(math.hypot(tk[k][1] - tk[j][1], tk[k][2] - tk[j][2]) / (tk[k][0] - tk[j][0]))
        walks.append({"t0": tk[0][0], "dur_s": round(dur, 1), "n": len(tk), "path_m": round(path, 2), "net_m": round(net, 2),
                      "moving": net >= 0.5, "speed_1s_median": round(float(np.median(sp)), 3) if sp else None,
                      "points": [[round(a, 2), round(b, 3), round(c, 3)] for a, b, c in tk]})
    mv = [w for w in walks if w["moving"] and w["speed_1s_median"]]
    res = {"segments": len(names), "frames_with_her": len(dets), "placed": len(placed), "clock_lag_s": clock_lag,
           "tracks": len(walks), "walks": len(mv),
           "walk_speed_m_s": {"median": round(float(np.median([w["speed_1s_median"] for w in mv])), 2) if mv else None,
                              "p25_p75": [round(float(x), 2) for x in np.percentile([w["speed_1s_median"] for w in mv], [25, 75])] if mv else None},
           "walk_path_m_median": round(float(np.median([w["path_m"] for w in mv])), 2) if mv else None,
           "items": walks}
    json.dump(res, open(out or os.path.join(os.path.dirname(export), "walks.json"), "w"))
    return {k: v for k, v in res.items() if k != "items"}


if __name__ == "__main__":
    a = sys.argv[1:]
    print(json.dumps(main(*a), indent=1))
