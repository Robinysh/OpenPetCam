"""Wave simulation of the camera's near field, compared with every sound recording, direction by direction.

Geometry (metres, world frame of camera.py: lens at the origin, y up):
  room       every fused surfel nearer than the domain edge becomes a rigid voxel (+1 voxel thick), plus the floor
             plane at the sound-measured height; the camera hangs from the counter slab, so the slab underside sits
             at the top of the camera's base ring (FCC ZCB-D1: shell 73.4 x 101.9 mm, lens ~87 mm from the base face)
  camera     body: a 73.4 mm cylinder about the pan axis from the slab down to the body seam; head: a ~58 mm ball on
             the tilt axis; the lens on the ball's front, the pan axis ~29 mm behind the lens (horizontal)
  speaker    on the BACK of the pan body at the grille's height (FCC manual label and photos)
  mic        on the head, beside the lens: ~12 mm from it toward the base (the camera hangs inverted)
For each recording the camera is posed at its logged pan/tilt (the room stays put), a band-limited pulse is emitted at
the speaker and pressure is recorded at the mic. Simulated and measured responses are both band-limited to the same
band, aligned on their direct arrival, and compared by the envelope correlation over 0.12-1.5 m of one-way range.
Acceptance (AUDIT ACO-5): the matched-direction score must beat a direction-shuffle null.
    ~/.venvs/da3/bin/python -m sim.world.nearfield eval/world [--max N] [--scan <scan folder prefix>]
    ... --face-sweep 0.25,0.33,0.41 --face-az 130,230 --face-el -70,15   (distance of one unseen face, set by sound)"""
import json
import math
import os
import sys

import numpy as np

from .fdtd import C, Grid

DX = 0.0076                 # 10 points per wavelength at 4.5 kHz
F_LO, F_HI = 2100.0, 4500.0
X_HALF, Y_TOP, Y_BOT = 0.80, 0.12, -0.92
LENS_FROM_BASE, BASE_R, BODY_TO_SEAM = 0.087, 0.0367, 0.076
BALL_R, LENS_FROM_PAN_AXIS = 0.029, 0.029
SPK_FROM_BASE, MIC_FROM_LENS = 0.039, 0.012


def band_pulse(t, f_lo=F_LO, f_hi=F_HI):
    """Gaussian-windowed cosine spanning the band (zero mean)."""
    fc, bw = (f_lo + f_hi) / 2, (f_hi - f_lo)
    sig = 1 / (math.pi * bw / 2.0)
    t0 = 4 * sig
    return np.cos(2 * math.pi * fc * (t - t0)) * np.exp(-((t - t0) / sig) ** 2)


def grid_shape():
    nx = int(2 * X_HALF / DX); ny = int((Y_TOP - Y_BOT) / DX); nz = nx
    return nx, ny, nz


def to_idx(p):
    return (int(round((p[0] + X_HALF) / DX)), int(round((p[1] - Y_BOT) / DX)), int(round((p[2] + X_HALF) / DX)))


def room_solid(surfel_range, D, floor_h):
    nx, ny, nz = grid_shape()
    s = np.zeros((nx, ny, nz), bool)
    P = (D * surfel_range[..., None]).reshape(-1, 3)
    P = P[np.isfinite(P).all(1)]
    ii = np.round((P[:, 0] + X_HALF) / DX).astype(int); jj = np.round((P[:, 1] - Y_BOT) / DX).astype(int)
    kk = np.round((P[:, 2] + X_HALF) / DX).astype(int)
    ok = (ii >= 1) & (ii < nx - 1) & (jj >= 1) & (jj < ny - 1) & (kk >= 1) & (kk < nz - 1)
    for di in (0, 1):                                   # two voxels thick, so no wall leaks between voxels
        s[np.clip(ii[ok] + di, 0, nx - 1), jj[ok], kk[ok]] = True
    fj = int(round((-floor_h - Y_BOT) / DX))
    s[:, :max(fj, 1), :] = True                          # floor and below
    slab = int(round((LENS_FROM_BASE - Y_BOT) / DX))
    return s, slab


def camera_solid(shape, head, slab_j):
    """Body cylinder + head ball for a head direction (unit vector in world). Returns solid mask and speaker/mic points."""
    nx, ny, nz = shape
    fwd = np.array([head[0], 0, head[2]]); fwd /= np.linalg.norm(fwd) + 1e-9
    ball_c = -BALL_R * head                                # the lens sits on the ball's surface along the view direction
    xs = (np.arange(nx) * DX - X_HALF)[:, None, None]; ys = (np.arange(ny) * DX + Y_BOT)[None, :, None]
    zs = (np.arange(nz) * DX - X_HALF)[None, None, :]
    pa = np.array([ball_c[0], 0, ball_c[2]])
    rr = np.sqrt((xs - pa[0]) ** 2 + (zs - pa[2]) ** 2)
    y_base = LENS_FROM_BASE                               # slab underside: the base face, 87 mm above the lens
    y_seam = y_base - BODY_TO_SEAM
    body = (rr <= BASE_R) & (ys <= y_base + DX) & (ys >= y_seam)
    ball = (xs - ball_c[0]) ** 2 + (ys - ball_c[1]) ** 2 + (zs - ball_c[2]) ** 2 <= BALL_R ** 2
    solid = body | ball
    back = -fwd
    spk = np.array([pa[0], y_base - SPK_FROM_BASE, pa[2]]) + back * (BASE_R + 1.5 * DX)
    up_head = np.cross(np.cross(head, [0, 1.0, 0]), head)
    up_head = up_head / (np.linalg.norm(up_head) + 1e-9)
    if up_head[1] < 0:
        up_head = -up_head
    mic = head * 1.5 * DX + up_head * MIC_FROM_LENS
    return solid, spk, mic


def measured(item, raw_dir):
    """The measured response for a recording, band-limited like the simulation, aligned on the direct arrival."""
    from scipy import signal
    from .sensors.acoustic import FS, responses
    probe = signal.resample_poly(np.load(os.path.join(raw_dir, "chirp2_16k.npy")), 3, 1)
    x = np.frombuffer(open(os.path.join(raw_dir, item["file"]), "rb").read(), np.int16).astype(float) / 32768
    hs = responses(x, probe, band=(F_LO, F_HI))
    n = min(len(h) for h in hs)
    h = np.array([a[:n] for a in hs]).mean(0)
    env = np.abs(signal.hilbert(h))
    return np.arange(len(env)) / FS, env / env.max()


def simulate(room, head, slab_j, T_s):
    shape = room.shape
    cam, spk, mic = camera_solid(shape, head, slab_j)
    solid = room | cam

    def free(p, away):
        # a transducer modelled inside a solid records nothing: step outward until the cell is air (and report it)
        for k in range(12):
            q = to_idx(p + away * k * DX)
            if not solid[q]:
                return q, k
        return to_idx(p), -1
    fwd = np.array([head[0], 0, head[2]]); fwd /= np.linalg.norm(fwd) + 1e-9
    si, ks = free(spk, -fwd)
    mi, km = free(mic, head)
    simulate.nudges = (ks, km)
    g = Grid(shape, DX, solid=solid, sponge=14)
    nT = int(T_s / g.dt)
    t = np.arange(nT) * g.dt
    s = band_pulse(t)
    rec = np.zeros(nT, np.float32)
    for it in range(nT):
        g.step(si, float(s[it]))
        rec[it] = float(g.p[mi])
    from scipy import signal
    env = np.abs(signal.hilbert(rec))
    return t, env / (env.max() + 1e-12)


def align_profile(t, env, r_axis):
    """Envelope vs one-way range after the direct arrival (its peak)."""
    i0 = int(np.argmax(env[: int(0.002 / (t[1] - t[0]))]))       # direct arrival within the first 2 ms
    rr = (t - t[i0]) * C / 2
    return np.interp(r_axis, rr, env, left=np.nan, right=np.nan)


def face_override(R, D, d, az_win=(130, 230), el_win=(-70, 15)):
    """Replace the fused surface in one direction window (azimuth, elevation in degrees) by a vertical plane facing the
    lens at distance d: the one geometry parameter sound is asked to set, typically a surface the sweep cannot see
    (a cabinet face right behind or below the lens); everything else stays as vision built it."""
    from .geometry import az_el
    az, el = az_el(D)
    a = (az % 360)
    win = (a >= az_win[0]) & (a <= az_win[1]) & (el >= el_win[0]) & (el <= el_win[1])
    a_c = math.radians((az_win[0] + az_win[1]) / 2)
    nrm = -np.array([math.sin(a_c), 0, math.cos(a_c)])                    # plane facing the lens, at the window's centre
    den = D @ nrm
    t = np.where(den > 0.05, d / np.where(den > 0.05, den, 1), np.nan)
    out = R.copy()
    out[win] = t[win]
    return out


def run(world, max_n=None, face_d=None, tag="", scan=None, face_win=((130, 230), (-70, 15))):
    """scan: only recordings whose file starts with this prefix (a scan folder name); default all raw-tap scans."""
    from .sensors.base import Context
    from .geometry import directions
    ctx = Context(world)
    m = json.load(open(os.path.join(ctx.model, "manifest.json")))
    L = {l["id"]: l for l in m["layers"]}
    sf = np.load(os.path.join(ctx.model, "surfels.npz"))
    R = sf["range_m"]; H, W = R.shape
    D = directions(W, H)
    if face_d is not None:
        R = face_override(R, D, face_d, *face_win)
    room, slab_j = room_solid(np.where(np.isfinite(R) & (R < 1.6), R, np.nan), D, m["room"]["floor_height_m"])
    items = [i for i in L["acoustic.echoes"]["data"]["items"] if i.get("raw_tap") and (scan is None or i["file"].startswith(scan))]
    if max_n:
        items = items[:max_n]
    r_axis = np.arange(0.12, 1.5, 0.004)
    raw_dir = ctx.path("raw", "acoustic")
    sims, meas, meta = [], [], []
    for k, it in enumerate(items):
        head = ctx.camera.R(it["pan"], it["tilt"])[:, 2]
        ts, es = simulate(room, head, slab_j, 0.0105)
        tm, em = measured(it, raw_dir)
        sims.append(align_profile(ts, es, r_axis)); meas.append(align_profile(tm, em, r_axis))
        meta.append({"file": it["file"], "pan": it["pan"], "tilt": it["tilt"], "nudged_cells": list(simulate.nudges)})
        print(f"{k + 1}/{len(items)} {it['file']}", flush=True)
    S, M = np.array(sims), np.array(meas)
    ok = np.isfinite(S).all(0) & np.isfinite(M).all(0)
    S, M = S[:, ok], M[:, ok]

    def detrend(x):
        k = 25
        tr = np.array([np.convolve(r, np.ones(k) / k, "same") for r in x]) + 1e-6
        return x / tr - 1
    Sd, Md = detrend(S), detrend(M)
    Cm = np.array([[np.corrcoef(Sd[a], Md[b])[0, 1] for a in range(len(S))] for b in range(len(M))])   # rows: recording
    true = float(np.mean(np.diag(Cm)))
    rng = np.random.default_rng(0)
    null = [float(np.mean(Cm[np.arange(len(Cm)), rng.permutation(len(Cm))])) for _ in range(2000)]
    half = np.arange(len(items)) % 2 == 0
    res = {"n": len(items), "dx_m": DX, "face_d": face_d, "scan": scan,
           "matched_r_even": round(float(np.mean(np.diag(Cm)[half])), 4), "matched_r_odd": round(float(np.mean(np.diag(Cm)[~half])), 4), "band_hz": [F_LO, F_HI], "matched_r": round(true, 3),
           "shuffled_mean": round(float(np.mean(null)), 3), "shuffled_p95": round(float(np.percentile(null, 95)), 3),
           "p": round(float(np.mean(np.array(null) >= true)), 4),
           "per_recording": [dict(mm, r=round(float(Cm[i, i]), 3)) for i, mm in enumerate(meta)]}
    json.dump(res, open(ctx.path("raw", "acoustic", f"nearfield_fit{tag}.json"), "w"), indent=1)
    np.savez_compressed(ctx.path("raw", "acoustic", f"nearfield_profiles{tag}.npz"), sim=S, meas=M, r=r_axis[ok])
    print({k: v for k, v in res.items() if k != "per_recording"})
    return res


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    a = sys.argv[1:]
    opt = lambda k, d=None: a[a.index(k) + 1] if k in a else d
    scan = opt("--scan")
    if "--face-sweep" in a:              # --face-sweep "0.25,0.33,0.41" [--face-az 130,230] [--face-el -70,15]
        win = (tuple(float(v) for v in opt("--face-az", "130,230").split(",")), tuple(float(v) for v in opt("--face-el", "-70,15").split(",")))
        for d in (float(v) for v in opt("--face-sweep").split(",")):
            run(a[0], None, d, tag=f"_face{int(d * 100)}", scan=scan, face_win=win)
    else:
        run(a[0], int(opt("--max")) if "--max" in a else None, scan=scan)
