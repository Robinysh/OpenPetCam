"""Sound as a check and a correction for the fused geometry.

Forward model (monostatic, the speaker and mic at the lens, both turning with the head):
    E(r) = sum over surface elements  area * facing^m * axis^p / range^2   binned at range r,
where facing = cos(angle between the element's normal and the ray back to the lens) (m large = mirror-like),
axis = cos(angle between the ray and the head direction) clipped at 0 (p = how directional the head is), then blurred by
the measurement's own kernel (the envelope width of an ideal echo through the same deconvolution). Geometry: the fused
range panorama plus the object boxes, i.e. everything vision and DA3 put in the room.

Refinement: over scale (all ranges x s), offset (a range bias: half the speaker-mic spacing and the direct-arrival
pick), m and p, the parameters that maximise the correlation between predicted and measured envelopes over 0.12-2 m.
Each recording is scored with parameters fitted to the other four (leave-one-out), so a better number is not just a
better fit. Strong measured echoes the best model leaves unexplained are listed: they are what the geometry is missing."""
import itertools
import math

import numpy as np

from .geometry import directions

C = 343.0


def scene_ranges(rng, objects, W, H):
    """The fused range panorama with the object boxes ray-cast into it (nearest hit wins)."""
    D = directions(W, H).reshape(-1, 3)
    R = rng.reshape(-1).copy()
    R[R <= 0] = np.inf
    for o in objects:
        cx, cy, cz = o["center"]
        sx, sy, sz = (np.array(o["size"]) / 2)
        yaw = math.radians(o["yaw_deg"])
        # box frame: depth along the window's azimuth (fwd), width across it (side)
        fwd = np.array([math.sin(yaw), 0, math.cos(yaw)]); side = np.array([math.cos(yaw), 0, -math.sin(yaw)]); up = np.array([0, 1.0, 0])
        Bm = np.stack([side, up, fwd])                      # rows: box axes in world
        o_ = Bm @ np.array([cx, cy, cz]); d_ = D @ Bm.T     # origin (lens) and rays in box frame, box centred at o_
        with np.errstate(divide="ignore", invalid="ignore"):
            t1 = (o_ - np.array([sx, sy, sz])) / d_
            t2 = (o_ + np.array([sx, sy, sz])) / d_
        tmin = np.nanmax(np.minimum(t1, t2), 1); tmax = np.nanmin(np.maximum(t1, t2), 1)
        hit = (tmax >= tmin) & (tmin > 0.05)
        R = np.where(hit & (tmin < R), tmin, R)
    R[~np.isfinite(R)] = 0
    return R.reshape(H, W)


def elements(R, W, H):
    """Per pixel: range, area, and the cosine between its surface normal and the ray back to the lens."""
    D = directions(W, H)
    P = D * R[..., None]
    dx = np.roll(P, -1, 1) - np.roll(P, 1, 1)
    dy = np.roll(P, -1, 0) - np.roll(P, 1, 0)
    n = np.cross(dx, dy)
    nn = np.linalg.norm(n, axis=-1, keepdims=True)
    n = n / np.where(nn > 0, nn, 1)
    facing = np.abs(np.sum(n * D, -1))
    el = np.radians(90 - (np.arange(H) + 0.5) / H * 180)
    dOmega = (2 * math.pi / W) * (math.pi / H) * np.cos(el)[:, None]
    area = dOmega * R ** 2 / np.maximum(facing, 0.05)
    ok = (R > 0.05) & (nn[..., 0] > 0) & np.isfinite(facing)
    # a range jump (an object's silhouette) is not a facet
    jump = (np.abs(np.roll(R, -1, 1) - np.roll(R, 1, 1)) > 0.25 * R) | (np.abs(np.roll(R, -1, 0) - np.roll(R, 1, 0)) > 0.25 * R)
    ok &= ~jump
    return D[ok], R[ok], area[ok], facing[ok]


def speaker_axis(head):
    """The speaker grille is on the back of the PAN body (FCC filing photos): it points horizontally, opposite
    the lens azimuth, and does not tilt. (World frame of camera.py: y up.)"""
    s = -np.array([head[0], 0.0, head[2]])
    return s / (np.linalg.norm(s) + 1e-9)


def predict(els, head, r_axis, scale, offset, m, p, kernel, ks=0.0):
    """p: microphone directivity about the lens axis (the mic's place is not known: p = 0 means omni on the body);
    ks: speaker directivity about speaker_axis(head)."""
    D, R, A, F = els
    # head directivity exp(k cos theta): measured from the floor echo across tilt rows (k ~ 1.4 on the reference camera); the old
    # cos^p clipped at 90 deg could not express a smooth front-to-back difference and every fit chose p = 0
    w = A * F ** m * np.exp(p * (D @ head) + ks * (D @ speaker_axis(head))) / np.maximum(R * scale, 0.05) ** 4 * (R * scale) ** 2
    rr = R * scale + offset
    E, _ = np.histogram(rr, bins=len(r_axis), range=(r_axis[0], r_axis[-1]), weights=w)
    E = np.convolve(E, kernel, "same")
    return E / (E.max() + 1e-12)


def kernel_from_probe(dr):
    """The envelope of an ideal single echo through the same tapered 2.1-7.4 kHz deconvolution, on the range grid."""
    fs = 48000
    t = np.arange(-0.004, 0.004, 1 / fs)
    f0, f1 = 2100, 7400
    bw = f1 - f0
    env = np.abs(np.sinc(bw * t)) ** 1.6                    # a Tukey-tapered band is close to sinc^1.6 in envelope
    r = C * t / 2
    grid = np.arange(-0.03, 0.03 + 1e-9, dr)
    k = np.interp(grid, r, env)
    return k / k.sum()


def fit(ctx, echo_items, rng, objects, W, H, cam, grid=None, folds=None):
    R0 = scene_ranges(rng, objects, W, H)
    els = elements(R0, W, H)
    dr = 0.004
    r_axis = np.arange(0.30, 2.0, dr)      # past the direct arrival's tail, which any model 'fits'
    kern = kernel_from_probe(dr)
    meas, heads = [], []
    for it in echo_items:
        ra, en = np.array(it["range_m"]), np.array(it["envelope"])
        meas.append(np.interp(r_axis, ra, en))
        heads.append(cam.R(it["pan"], it["tilt"])[:, 2])
    grid = grid or list(itertools.product([0.75, 0.8, 0.85, 0.9, 0.95, 1.0, 1.05, 1.1, 1.2], [-0.08, -0.05, -0.02, 0.0, 0.02, 0.05],
                                          [1, 4, 16], [0, 1, 3]))

    def peaks_only(x):
        # divide out the smooth fall-off with range (every room has one); what is left are the individual echoes
        k = int(0.2 / dr) | 1
        trend = np.convolve(x, np.ones(k) / k, "same") + 1e-6
        return x / trend - 1

    meas = [peaks_only(m_) for m_ in meas]

    def score(params, idx):
        s = []
        for i in idx:
            e = peaks_only(predict(els, heads[i], r_axis, *params[:4], kern, *(params[4:5] or (0.0,))))
            s.append(np.corrcoef(e, meas[i])[0, 1])
        return float(np.nanmean(s))

    base = (1.0, 0.0, 4, 1)
    n = len(meas)
    allidx = list(range(n))
    best_all = max(grid, key=lambda g: score(g, allidx))
    loo = []
    k = folds or n                                     # k-fold cross-validation (leave-one-out when folds is None)
    fold_of = [i % k for i in range(n)]
    for f in range(k):
        test = [i for i in allidx if fold_of[i] == f]
        g = max(grid, key=lambda g: score(g, [j for j in allidx if fold_of[j] != f]))
        for i in test:
            loo.append({"file": echo_items[i]["file"], "baseline_r": round(score(base, [i]), 3),
                        "held_out_r": round(score(g, [i]), 3), "params": g})
    # what the best model leaves unexplained: strong measured peaks with little predicted energy within 3 cm
    resid = []
    for i, it in enumerate(echo_items):
        e = predict(els, heads[i], r_axis, *best_all[:4], kern, *(best_all[4:5] or (0.0,)))
        for rpk, spk in it["echoes"]:
            if spk < 0.15 or not (0.12 <= rpk < 2.0):
                continue
            k = int((rpk - r_axis[0]) / dr)
            if e[max(k - 8, 0): k + 8].max() < 0.25 * spk * e.max():
                resid.append({"file": it["file"], "range_m": rpk, "strength": spk, "az": it["az"]})
        it["model_profile"] = {"range_m": np.round(r_axis[::2], 3).tolist(), "envelope": np.round(e[::2], 3).tolist()}
    # null test: the same fit against the room turned 180 deg about the lens. If sound really sees this geometry,
    # the true room must beat the turned one by a clear margin.
    els_true = els
    els = elements(np.roll(R0, W // 2, 1), W, H)
    null_best = max(grid, key=lambda g: score(g, allidx))
    null_r = score(null_best, allidx)
    els = els_true
    return {"null_rotated_r": round(null_r, 3), "best": dict(zip(["scale", "offset_m", "gloss_m", "mic_directivity", "speaker_directivity"], best_all)),
            "baseline": dict(zip(["scale", "offset_m", "gloss_m", "directivity_p"], base)),
            "mean_r_baseline": round(score(base, allidx), 3), "mean_r_best_in_sample": round(score(best_all, allidx), 3),
            "mean_r_held_out": round(float(np.mean([l["held_out_r"] for l in loo])), 3), "leave_one_out": loo,
            "unexplained": resid}
