"""Insert the pet into camera views, and prove the composites are faithful before using them.

Cutout: a frame where she is verified, and a background of the same view from the same recording at moments she was
not there: frames the detector and verifier did not pass (walks_dets.json lists every 5 fps frame they did), or
passed with her box elsewhere (overlapping hers by at most 10%), showing the same view (found by the pictures, not the pose log: the colour outside her box must
match, at 160x90 and again after alignment), different inside her box (she moved away: a sleeping dog the detector missed would
otherwise become 'background'), re-checked by the judge, shifted onto the frame (align), and the median of up to BG_N
of them (so lighting matches and anything passing is voted out). Her matte is where the frame differs from that background inside her box (colour distance over a threshold,
cleaned by a small opening and the largest component).
Realism check (the gate for everything else): paste each cutout back onto its own background at exactly where she
was, and run the same detector and verifier on the composite and on the original frame (a sighting counts when a
verified box overlaps hers, IoU >= 0.3). The composites must be detected at the same rate as the originals (paired,
with a bootstrap interval covering 0) and the backgrounds alone must almost never be (< 0.1) before synthetic
insertions are used to test the detector anywhere else. Rows of one resting episode are near-duplicates, so the interval
is also computed resampling whole (segment, pose) clusters, and both must cover 0, with at least MIN_N rows from
MIN_CLUSTERS episodes.
Also measured, not gated: the same cutout pasted onto the synthetic render of that pose (synthview.py, dog-free
panorama; pose registered to the pet-free background first, see register), which is what the report's step 4b shows.
Transplant check (reported, and used by the report to say whether a cutout carries her on its own): each cutout on
the pet-free background of a different pose, against the same matte shape filled with her own background.
Mattes are a colour difference refined by GrabCut (refine); each row records the matte's fill of her box, solidity
and leak (matte_quality).
Every usable cutout is saved (eval/rl/synthdog_cutouts/, RGBA + matte, with its row) for the report.
Backgrounds are audited by eye (apply_audit): the judge cannot vouch for frames chosen because it missed her.
    python3 -m sim.world.synthdog eval/world eval/rl/rl_export.json eval/rl/video [n]  -> eval/rl/synthdog_check.json
    python3 -m sim.world.synthdog --audit eval/rl/synthdog_check.json eval/rl/synthdog_bg_audit.json"""
import json
import os
import subprocess
import sys

import numpy as np

W, H = 640, 360
FPS = 5                      # the grid walks.py judged (walks_dets.json)
BG_N, BG_GAP_S = 9, 20.0
CLEAR_S = 1.0                # a background frame is at least this far from any verified frame
BG_FPS = 0.5                 # background candidates: every recording decoded at this rate
OUT_MAX = 0.06               # 'the same view': mean colour difference outside her box (same view 0.01-0.03, another > 0.14)
IN_MIN = 0.06                # and inside her box at least this (she has moved away; under it she is still lying there)
SPACING_S = 5.0              # sampled sightings at least this far apart (consecutive frames are near-duplicates)
IOU_MIN = 0.3
MIN_N, MIN_CLUSTERS = 30, 5    # the least evidence the gate accepts (rows, independent episodes)
REG_MIN = 0.25             # render-to-frame match needed before the render check uses a pose (see register)


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


def refine(img, bg, box, m, thr=0.12):
    """GrabCut from the difference matte: outside the (slightly grown) box is background, strong differences are
    probably her, the rest of the box probably not. Fur against a wood floor of the same colour leaves holes in a
    plain difference; colour models over the whole box close them. Keeps the largest component, holes filled."""
    import cv2
    from scipy import ndimage
    x0, y0, x1, y1 = (int(box[0] * W), int(box[1] * H), int(np.ceil(box[2] * W)), int(np.ceil(box[3] * H)))
    g = 6
    X0, Y0, X1, Y1 = max(0, x0 - g), max(0, y0 - g), min(W, x1 + g), min(H, y1 + g)
    gm = np.full((Y1 - Y0, X1 - X0), cv2.GC_BGD, np.uint8)
    gm[y0 - Y0:y1 - Y0, x0 - X0:x1 - X0] = cv2.GC_PR_BGD
    diff = np.linalg.norm(img[Y0:Y1, X0:X1].astype(float) - bg[Y0:Y1, X0:X1].astype(float), axis=-1) / 255
    inbox = gm == cv2.GC_PR_BGD
    gm[inbox & m[Y0:Y1, X0:X1]] = cv2.GC_PR_FGD
    gm[inbox & ndimage.binary_erosion(m[Y0:Y1, X0:X1], iterations=3) & (diff > 2 * thr)] = cv2.GC_FGD
    if not (gm == cv2.GC_FGD).any() and not (gm == cv2.GC_PR_FGD).any():
        return m
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    try:
        cv2.grabCut(np.ascontiguousarray(img[Y0:Y1, X0:X1, ::-1]), gm, None, bgd, fgd, 4, cv2.GC_INIT_WITH_MASK)
    except cv2.error:
        return m
    sub = (gm == cv2.GC_FGD) | (gm == cv2.GC_PR_FGD)
    lab, n = ndimage.label(sub)
    if n:
        sizes = ndimage.sum(sub, lab, range(1, n + 1))
        sub = ndimage.binary_fill_holes(lab == (1 + int(np.argmax(sizes))))
    out = np.zeros((H, W), bool)
    out[Y0:Y1, X0:X1] = sub
    return out


def matte_quality(m, box, img=None, bg=None, thr=0.12):
    """fill: matte share of her box (a matte filling the box is copying background); solidity: matte over its convex
    hull (shreds are low); leak: share of the matte where the frame equals the background (diff under thr/2: things
    that were there without her, e.g. a bowl in front of her)."""
    import cv2
    a = (box[2] - box[0]) * W * (box[3] - box[1]) * H
    pts = np.argwhere(m)[:, ::-1].astype(np.int32)
    hull = cv2.contourArea(cv2.convexHull(pts)) if len(pts) >= 3 else 0
    q = {"fill": round(float(m.sum() / max(a, 1)), 3), "solidity": round(float(m.sum() / max(hull, 1)), 3)}
    if img is not None and m.any():
        diff = np.linalg.norm(img[m].astype(float) - bg[m].astype(float), axis=-1) / 255
        q["leak"] = round(float((diff < thr / 2).mean()), 3)
    return q


def align(b, img, box):
    """Shift a background frame onto img (phase correlation of greys, her box blanked): poses within 3 motor steps
    still differ by a few pixels, and high-contrast edges (a bowl, a cable) then show up as 'her' in the difference."""
    import cv2
    g1 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY).astype(np.float32); g2 = cv2.cvtColor(b, cv2.COLOR_RGB2GRAY).astype(np.float32)
    x0, y0, x1, y1 = int(box[0] * W), int(box[1] * H), int(np.ceil(box[2] * W)), int(np.ceil(box[3] * H))
    for g in (g1, g2):
        g[y0:y1, x0:x1] = g.mean()
        g[: int(0.06 * H), : int(0.36 * W)] = g.mean()      # the timestamp overlay does not move
    (dx, dy), resp = cv2.phaseCorrelate(g2, g1, cv2.createHanningWindow((W, H), cv2.CV_32F))
    if resp < 0.05 or abs(dx) > 12 or abs(dy) > 12:
        return b
    return cv2.warpAffine(b, np.float32([[1, 0, dx], [0, 1, dy]]), (W, H), borderMode=cv2.BORDER_REPLICATE)


def _decode(path, fps):
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-vf", f"fps={fps},scale={W}:{H}", "-f", "rawvideo",
                          "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
    k = 0
    while True:
        buf = p.stdout.read(W * H * 3)
        if len(buf) < W * H * 3:
            break
        yield k / fps, np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        k += 1
    p.wait()


def _small(img, w=160, h=90):
    import cv2
    return cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)


def _diffs(img, b, box, w=W, h=H):
    """Mean colour difference (0..1) outside her box (and the timestamp) and inside it."""
    x0, y0, x1, y1 = int(box[0] * w), int(box[1] * h), int(np.ceil(box[2] * w)), int(np.ceil(box[3] * h))
    d = np.abs(img.astype(np.float32) - b.astype(np.float32)).mean(-1) / 255
    m = np.ones((h, w), bool); m[y0:y1, x0:x1] = False; m[: int(0.06 * h), : int(0.36 * w)] = False
    return float(d[m].mean()), float(d[y0:y1, x0:x1].mean())


def _overlap(a, b):
    """Share of box b covered by box a."""
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / max(1e-9, (b[2] - b[0]) * (b[3] - b[1]))


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0])); iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    u = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - ix * iy
    return ix * iy / u if u > 0 else 0.0


def load_judge(root=None):
    """The server's detector and pet verifier (read from server/, not modified). Returns judge(img) -> [(p, box)], every
    verified box among the detector's three best non-person boxes, best first."""
    from PIL import Image
    root = root or os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if os.path.join(root, "server") not in sys.path:
        sys.path.insert(0, os.path.join(root, "server"))
    from detector import Detector
    src = open(os.path.join(root, "server", "watch.py")).read()
    ns = {}
    exec(compile("import json, os\nimport numpy as np\nfrom PIL import Image\nVERIFIER=''\n" +
                 src[src.index("class Verifier:"):src.index("class SoundModel:")], "verifier", "exec"), ns)
    V = ns["Verifier"](os.path.join(root, "models", "verifier.json"))
    det = Detector(os.path.join(root, "models", "yolox_nano.onnx"), size=416)

    def judge(img, min_score=0.10):
        im = Image.fromarray(img) if isinstance(img, np.ndarray) else img
        out = []
        for f in sorted([f for f in det.detect(im, min_score=min_score) if f["label"] != "person"], key=lambda f: -f["score"])[:3]:
            p = V.prob(im, f["box"])
            if p >= V.threshold:
                out.append((float(p), [float(b) for b in f["box"]]))
        return sorted(out, key=lambda x: -x[0])
    judge.threshold = V.threshold
    return judge


def sees(found, box):
    return any(iou(b, box) >= IOU_MIN for _p, b in found)


def register(sv, real, pan, tilt, off, span=(300, 150), step=10):
    """The render pose that best matches a real (pet-free) frame: the logged pose is only a start, because the mount
    moved after the panorama's sweep and the drift correction ends before the recordings. Normalised correlation of
    blurred gradient magnitudes (synthview._grad), coarse grid at 160x90 then +-step at 320x180.
    Returns (pan, tilt, score, grid median score) - a score near the grid median means no real match."""
    from PIL import Image
    from . import synthview
    ctx, pano = sv

    def score(p, t, w, h, ref):
        g = synthview._grad(synthview.render(ctx, pano, p, t, off=off, w=w, h=h))
        return float((g * ref).mean())
    refs = {}
    for w, h in ((160, 90), (320, 180)):
        r = np.asarray(Image.fromarray(real).resize((w, h)), np.float32) / 255
        r[: int(0.06 * h), : int(0.36 * w)] = r.reshape(-1, 3).mean(0)          # the timestamp overlay
        refs[w] = synthview._grad(r)
    grid = [(pan + dp, tilt + dt) for dp in range(-span[0], span[0] + 1, step) for dt in range(-span[1], span[1] + 1, step)]
    sc = [score(p, t, 160, 90, refs[160]) for p, t in grid]
    p0, t0 = grid[int(np.argmax(sc))]
    fine = [(p0 + dp, t0 + dt) for dp in range(-step, step + 1, step // 2) for dt in range(-step, step + 1, step // 2)]
    fs = [score(p, t, 320, 180, refs[320]) for p, t in fine]
    k = int(np.argmax(fs))
    return fine[k][0], fine[k][1], fs[k], float(np.median(sc))


def cluster_key(r):
    return f"{r['seg']}|{round(r['pan'])}|{round(r['tilt'])}"


def summarise(rows, rng=None):
    """The gate's numbers for a set of rows (see the module docstring)."""
    rng = rng if rng is not None else np.random.default_rng(0)
    tr_rows = [r for r in rows if "transplant" in r]
    o = np.array([r["orig"] for r in rows]); c = np.array([r["composite"] for r in rows]); e = np.array([r["background_alone"] for r in rows])
    bs = [float(c[i].mean() - o[i].mean()) for i in (rng.integers(0, len(rows), len(rows)) for _ in range(2000))] if rows else []
    rr = [r for r in rows if "render_composite" in r]
    ro = np.array([r["orig"] for r in rr]); rc = np.array([r["render_composite"] for r in rr])
    bs_r = [float(rc[i].mean() - ro[i].mean()) for i in (rng.integers(0, len(rr), len(rr)) for _ in range(2000))] if rr else []
    # rows from one resting episode are not independent: also resample whole (segment, pose) clusters
    cl = {}
    for k, r in enumerate(rows):
        cl.setdefault(cluster_key(r), []).append(k)
    groups = list(cl.values())
    bs_c = []
    for _ in range(2000) if rows else []:
        idx = [k for g in rng.integers(0, len(groups), len(groups)) for k in groups[int(g)]]
        bs_c.append(float(c[idx].mean() - o[idx].mean()))
    rnd = lambda v: round(float(v), 3)                       # noqa: E731
    res = {"n": len(rows),
           "original_detected": rnd(o.mean()) if rows else None,
           "composite_detected": rnd(c.mean()) if rows else None,
           "background_alone_detected": rnd(e.mean()) if rows else None,
           "difference_95ci": [rnd(np.percentile(bs, 2.5)), rnd(np.percentile(bs, 97.5))] if bs else None,
           "agree_per_frame": rnd((o == c).mean()) if rows else None,
           "distinct_poses": len({(round(r["pan"]), round(r["tilt"])) for r in rows}), "clusters": len(groups),
           "difference_95ci_by_cluster": [rnd(np.percentile(bs_c, 2.5)), rnd(np.percentile(bs_c, 97.5))] if bs_c else None,
           "moving_rows": sum(1 for r in rows if r["moving"]),
           "transplant": {"n": len(tr_rows),
                          "detected": rnd(np.mean([r["transplant"] for r in tr_rows])) if tr_rows else None,
                          "null_detected": rnd(np.mean([r["transplant_null"] for r in tr_rows])) if tr_rows else None},
           "matte_fill_median": rnd(np.median([r["fill"] for r in rows])) if rows else None,
           "matte_solidity_median": rnd(np.median([r["solidity"] for r in rows])) if rows else None,
           "matte_leak_median": rnd(np.median([r.get("leak", 0) for r in rows])) if rows else None,
           "on_render": {"n": len(rr), "original_detected": rnd(ro.mean()) if rr else None,
                         "composite_detected": rnd(rc.mean()) if rr else None,
                         "render_alone_detected": rnd(np.mean([r["render_alone"] for r in rr])) if rr else None,
                         "difference_95ci": [rnd(np.percentile(bs_r, 2.5)), rnd(np.percentile(bs_r, 97.5))] if bs_r else None}}
    # an interval covering 0 means nothing when it is 0.5 wide: the gate also needs MIN_N rows from MIN_CLUSTERS episodes
    enough = len(rows) >= MIN_N and len(groups) >= MIN_CLUSTERS
    res["enough_rows"] = enough
    res["faithful"] = bool(enough and bs and np.percentile(bs, 2.5) <= 0 <= np.percentile(bs, 97.5)
                           and np.percentile(bs_c, 2.5) <= 0 <= np.percentile(bs_c, 97.5) and res["background_alone_detected"] < 0.1)
    res["faithful_on_render"] = bool(len(rr) >= MIN_N and bs_r and np.percentile(bs_r, 2.5) <= 0 <= np.percentile(bs_r, 97.5)
                                     and res["on_render"]["render_alone_detected"] < 0.1)
    return res


def apply_audit(check, audit):
    """The judge cannot vouch for backgrounds it chose itself (they are, by construction, frames it missed her in), so
    a person looks at one background per cluster (eval/rl/synthdog_cutouts/bg_*.jpg) and writes
    {"seg|pan|tilt": {"pet_in_background": true|false, "note": ...}} to synthdog_bg_audit.json. The gate is then
    re-run on the clusters judged pet-free by eye AND by the judge; the all-rows numbers stay under "all_rows"."""
    res = json.load(open(check))
    lab = {k: v for k, v in json.load(open(audit)).items() if not k.startswith("_")}
    rows = res["rows"]
    keep = [r for r in rows if lab.get(cluster_key(r), {}).get("pet_in_background") is False and not r["background_alone"]]
    unlabelled = sorted({cluster_key(r) for r in rows if cluster_key(r) not in lab})
    all_rows = summarise(rows)
    res = dict(res)
    res.update(summarise(keep))
    by_eye = [lab[cluster_key(r)]["pet_in_background"] for r in rows if cluster_key(r) in lab]
    res["all_rows"] = all_rows
    res["audited"] = {"clusters_labelled": len(lab), "clusters_unlabelled": unlabelled,
                      "rows_with_pet_in_background_by_eye": round(float(np.mean(by_eye)), 3) if by_eye else None,
                      "rows_kept": len(keep), "rows": len(rows)}
    res["clean_cluster_keys"] = sorted({cluster_key(r) for r in keep})
    json.dump(res, open(check, "w"), indent=1)
    return {k: v for k, v in res.items() if k != "rows"}


def main(world, export, video_dir, n=120, seed=0):
    from PIL import Image
    from .walks import pose_interp, segment_start
    judge = load_judge()
    d = json.load(open(export))
    tr = sorted(d["track"]); tt = np.array([r[0] for r in tr]); pan = np.array([r[1] for r in tr], float); tilt = np.array([r[2] for r in tr], float)
    rl = os.path.dirname(export)
    dets = sorted(json.load(open(os.path.join(rl, "walks_dets.json"))), key=lambda x: x["t"])
    det_t = np.array([x["t"] for x in dets])
    wk = os.path.join(rl, "walks.json")
    lag = json.load(open(wk)).get("clock_lag_s", 0) if os.path.exists(wk) else 0      # recording clock vs the pose log
    moving_t = []
    if os.path.exists(wk):
        for it in json.load(open(wk))["items"]:
            if it["moving"]:
                moving_t.append((it["points"][0][0] - lag, it["points"][-1][0] - lag))
    segs = {}
    for f in sorted(os.listdir(video_dir)):
        if f.endswith(".mp4"):
            dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                        os.path.join(video_dir, f)], capture_output=True, text=True).stdout or 0)
            segs[f] = (segment_start(f), dur)

    def seg_of(t):
        return next((f for f, (s, du) in segs.items() if s <= t < s + du), None)

    rng = np.random.default_rng(seed)
    order, chosen = rng.permutation(len(dets)), []
    for i in order:                                          # sightings spaced apart: near-duplicates inflate n
        if all(abs(dets[i]["t"] - dets[j]["t"]) >= SPACING_S for j in chosen):
            chosen.append(int(i))
        if len(chosen) >= 6 * n:
            break
    by_seg = {}
    for i in chosen:
        f = seg_of(dets[i]["t"])
        if f:
            by_seg.setdefault(f, []).append(i)
    per_seg = max(8, -(-n // max(1, len(by_seg))))           # spread over recordings: one long rest must not be all of n
    cut_dir = os.path.join(rl, "synthdog_cutouts"); os.makedirs(cut_dir, exist_ok=True)
    reg = os.path.join(world, "raw", "registration.json")
    smooth = (json.load(open(reg)).get("smooth") or []) if os.path.exists(reg) else []
    off = (smooth[-1][1], smooth[-1][2], 0.0) if smooth else (0.0, 0.0, 0.0)
    sv = None
    try:
        from . import synthview
        ctx, pano = synthview.load(world, dog_free=True)
        sv = (ctx, pano)
    except FileNotFoundError:
        pass
    reg_cache, keep, bg_saved, bg_rej = {}, [], {}, {}
    skipped = {"no_segment": len(chosen) - sum(len(v) for v in by_seg.values()), "few_backgrounds": 0, "decode": 0, "matte_small": 0,
               "segment_quota": 0}
    rows = []
    for f in sorted(by_seg, key=lambda f: rng.random()):
        if len(rows) >= n:
            break
        s0 = segs[f][0]
        # clear frames of this recording: decoded once at BG_FPS, no verified sighting within CLEAR_S
        # candidate frames of this recording, decoded once at BG_FPS, each with where she was verified (None: nowhere
        # within CLEAR_S). She is in view most of the time (the camera follows her), so a frame where she lies
        # ELSEWHERE in the same view also serves: only her box region of the background is new information.
        cl_t, cl_img, cl_box = [], [], []
        for rel, rgb in _decode(os.path.join(video_dir, f), BG_FPS):
            t = s0 + rel
            k = int(np.searchsorted(det_t, t))
            js = [j for j in (k - 1, k) if 0 <= j < len(det_t)]
            j = min(js, key=lambda j: abs(t - det_t[j])) if js else None
            if j is None or abs(t - det_t[j]) >= CLEAR_S:
                cl_t.append(t); cl_img.append(rgb); cl_box.append(None)
            elif abs(t - det_t[j]) <= 0.25:
                cl_t.append(t); cl_img.append(rgb); cl_box.append(dets[j]["box"])
        cl_t = np.array(cl_t)
        cl_small = [_small(b) for b in cl_img]
        got = 0
        for i in by_seg[f]:
            if len(rows) >= n:
                break
            if got >= per_seg:
                skipped["segment_quota"] += 1
                continue
            x = dets[i]
            t, box = x["t"], x["box"]
            pose = pose_interp(tt, pan, tilt, t + lag) or (-1.0, -1.0)     # the logged pose (for the render; may be stale)
            img = frame_at(os.path.join(video_dir, f), t - s0)
            if img is None:
                skipped["decode"] += 1
                continue
            # the same VIEW, found by the pictures, not the pose log (it is stale after the mount moved): phase
            # correlation of small greys outside her box, then the full frame must match outside her box and differ
            # inside it (she is not still lying there), and the judge must not see her
            ref = _small(img)
            far = [j for j in np.where(np.abs(cl_t - t) >= BG_GAP_S)[0] if cl_box[j] is None or _overlap(cl_box[j], box) <= 0.1]
            sc = [int(j) for j in far if _diffs(ref, cl_small[j], box, 160, 90)[0] <= OUT_MAX]
            if len(sc) < 3:
                skipped["few_backgrounds"] += 1; skipped["few_same_view"] = skipped.get("few_same_view", 0) + 1
                continue
            sc = [sc[k] for k in np.linspace(0, len(sc) - 1, min(len(sc), 3 * BG_N)).astype(int)]   # spread in time
            bgs, rejected = [], {"view": 0, "still_there": 0, "judge": 0}
            for j in sc:
                b = align(cl_img[j], img, box)
                out_d, in_d = _diffs(img, b, box)
                if out_d > OUT_MAX:
                    rejected["view"] += 1
                    continue
                if in_d < IN_MIN:
                    rejected["still_there"] += 1
                    continue
                if any(_overlap(fb, box) > 0.1 for _p, fb in judge(b)):   # the judge sees her (or a look-alike) in her box
                    rejected["judge"] += 1
                    continue
                bgs.append(b)
                if len(bgs) >= BG_N:
                    break
            for k_, v_ in rejected.items():
                bg_rej[k_] = bg_rej.get(k_, 0) + v_
            if len(bgs) < 3:
                skipped["few_backgrounds"] += 1
                continue
            bg = np.median(np.stack(bgs), 0).astype(np.uint8)
            m = refine(img, bg, box, matte(img, bg, box))
            if m.sum() < 0.15 * (box[2] - box[0]) * W * (box[3] - box[1]) * H:
                skipped["matte_small"] += 1                      # matte failed (she blends into the background)
                continue
            comp = bg.copy(); comp[m] = img[m]
            j_orig, j_comp, j_bg = judge(img), judge(comp), judge(bg)
            row = {"t": t, "seg": f, "pan": pose[0], "tilt": pose[1], "box": box, "n_bg": len(bgs), "bg_rejected": rejected,
                   "orig": sees(j_orig, box), "composite": sees(j_comp, box), "background_alone": bool(j_bg),
                   "background_alone_at_box": sees(j_bg, box), "matte_px": int(m.sum()),
                   "moving": any(a <= t <= b for a, b in moving_t), **matte_quality(m, box, img, bg)}
            ck = (f, round(pose[0]), round(pose[1]))
            keep.append((img, bg, m, box, ck))
            if ck not in bg_saved:                                # one background per cluster, for a look by eye: the
                bg_saved[ck] = f"bg_{len(bg_saved):02d}.jpg"      # judge cannot vouch for backgrounds it chose itself
                try:
                    Image.fromarray(bg).save(os.path.join(cut_dir, bg_saved[ck]), quality=90)
                except OSError:
                    pass
            row["bg_image"] = bg_saved[ck]
            if sv is not None and 300 <= pose[1] <= 800:          # the same cutout on the synthetic render of this pose
                key = (f, round(pose[0] / 5), round(pose[1] / 5))
                if key not in reg_cache:
                    reg_cache[key] = register(sv, bg, pose[0], pose[1], off)
                rp, rt, rs, rnull = reg_cache[key]
                row.update({"render_pan": rp, "render_tilt": rt, "render_match": round(rs, 3), "render_match_null": round(rnull, 3)})
                if rs >= REG_MIN and rs >= 2 * rnull:              # registered: paste where she was in the real frame
                    syn = (synthview.render(sv[0], sv[1], rp, rt, off=off, smooth=True) * 255).astype(np.uint8)
                    cr = syn.copy(); cr[m] = img[m]
                    row["render_composite"] = sees(judge(cr), box)
                    row["render_alone"] = bool(judge(syn))
            k = len(rows)
            x0, y0, x1, y1 = int(box[0] * W), int(box[1] * H), int(np.ceil(box[2] * W)), int(np.ceil(box[3] * H))
            rgba = np.dstack([img[y0:y1, x0:x1], (m[y0:y1, x0:x1] * 255).astype(np.uint8)])
            try:
                Image.fromarray(rgba, "RGBA").save(os.path.join(cut_dir, f"cut_{k:03d}.png"))
                row["cutout"] = f"cut_{k:03d}.png"
            except OSError as ex:                                # e.g. a full disk: the row still counts for the check
                print(f"cutout not saved: {ex}", flush=True)
            rows.append(row); got += 1
            print(f"{len(rows)}/{n} t={t:.1f} orig={row['orig']} comp={row['composite']} bg={row['background_alone']} "
                  f"render={row.get('render_composite')} match={row.get('render_match')} n_bg={len(bgs)} rej={rejected}", flush=True)
    # transplant: each cutout onto the pet-free background of a DIFFERENT cluster (real frames, so no render-domain
    # gap), at the same place; null: the same matte shape filled with her own background's pixels. The cutout carries
    # her when the first is detected and the second is not.
    for r, (img, bg, m, box, ck) in zip(rows, keep):
        others = [k for k in keep if k[4] != ck]
        if not others:
            continue
        fbg = others[int(rng.integers(len(others)))][1]
        tp = fbg.copy(); tp[m] = img[m]
        nl = fbg.copy(); nl[m] = bg[m]
        r["transplant"] = sees(judge(tp), box)
        r["transplant_null"] = sees(judge(nl), box)
    res = summarise(rows, rng)
    res.update({"skipped": skipped, "background_frames_rejected": bg_rej, "sightings_considered": len(chosen), "rows": rows})
    last_reg = smooth[-1][0] if smooth else None
    rr = [r for r in rows if "render_composite" in r]
    if rr and last_reg and min(r["t"] for r in rr) > last_reg:
        res["on_render"]["note"] = ("the recordings postdate the world's last drift correction by "
                                    f"{(min(r['t'] for r in rr) - last_reg) / 3600:.1f} h; if the mount moved in between (AUDIT GEO-5), "
                                    "renders do not match these frames and this number measures that, not the cutouts")
    out = os.path.join(rl, "synthdog_check.json")
    json.dump(res, open(out, "w"), indent=1)
    audit = os.path.join(rl, "synthdog_bg_audit.json")
    if os.path.exists(audit):
        res = apply_audit(out, audit)
    return {k: v for k, v in res.items() if k != "rows"}


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "--audit":                              # re-score an existing check with the eye audit
        print(json.dumps(apply_audit(a[1], a[2]), indent=1))
    else:
        print(json.dumps(main(a[0], a[1], a[2], int(a[3]) if len(a) > 3 else 120), indent=1))
