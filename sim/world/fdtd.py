"""3D acoustic wave-equation solver for the room's near field (FDTD, staggered pressure-velocity grid).

Why a wave solver: the first-order ray model of the echoes predicts nothing direction-specific (true vs shuffled
direction pairing scored at chance), because what shapes this camera's echoes is the near field: whatever it is mounted
against, nearby furniture faces, a speaker on the BACK of the pan body and a mic elsewhere on the head. Diffraction and near-field geometry are wave effects.

Scheme (carried over from cybersweat's solver discipline): a staggered (Yee) grid, pressure at cell centres and
velocity on faces (a collocated grid amplifies dispersion errors); explicit leapfrog with the 3D stability bound
c*dt/dx <= 1/sqrt(3) and a safety factor; a graded sponge layer to absorb outgoing waves; rigid walls as velocity-zero
faces from a solid mask (a reflection coefficient per face is the next step). Verified against closed form before it
is trusted: `verify_free_field()` checks wave speed and 1/r spreading of a point source.

    python3 -m sim.world.fdtd verify      (runs on the Mac's GPU through PyTorch MPS, or CPU)
"""
import math
import sys

import numpy as np

C = 343.0
RHO = 1.2


def ricker(t, f0):
    a = (math.pi * f0 * (t - 1.2 / f0)) ** 2
    return (1 - 2 * a) * np.exp(-a)


class Grid:
    def __init__(self, shape, dx, solid=None, sponge=20, safety=0.9, device=None):
        import torch
        self.torch = torch
        self.dev = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.shape, self.dx = tuple(shape), dx
        self.dt = safety * dx / (C * math.sqrt(3))                   # 3D CFL bound with a safety factor
        nx, ny, nz = shape
        f = torch.float32
        self.p = torch.zeros(shape, dtype=f, device=self.dev)
        self.vx = torch.zeros((nx + 1, ny, nz), dtype=f, device=self.dev)
        self.vy = torch.zeros((nx, ny + 1, nz), dtype=f, device=self.dev)
        self.vz = torch.zeros((nx, ny, nz + 1), dtype=f, device=self.dev)
        # graded sponge: damping rises quadratically into the outer `sponge` cells
        def ramp(n):
            d = np.zeros(n)
            for i in range(sponge):
                d[i] = d[n - 1 - i] = ((sponge - i) / sponge) ** 2
            return d
        damp = 0.35 * np.maximum.reduce(np.meshgrid(ramp(nx), ramp(ny), ramp(nz), indexing="ij"))
        self.damp = torch.tensor(1.0 - damp, dtype=f, device=self.dev)
        self.solid = None
        if solid is not None:
            s = torch.tensor(solid, dtype=torch.bool, device=self.dev)
            self.solid = s
            # a face is rigid if either neighbouring cell is solid
            self.mx = torch.ones_like(self.vx); self.mx[1:-1][s[1:] | s[:-1]] = 0
            self.my = torch.ones_like(self.vy); self.my[:, 1:-1][s[:, 1:] | s[:, :-1]] = 0
            self.mz = torch.ones_like(self.vz); self.mz[:, :, 1:-1][s[:, :, 1:] | s[:, :, :-1]] = 0

    def step(self, src=None, amp=0.0):
        k = self.dt / (RHO * self.dx)
        p = self.p
        self.vx[1:-1] -= k * (p[1:] - p[:-1])
        self.vy[:, 1:-1] -= k * (p[:, 1:] - p[:, :-1])
        self.vz[:, :, 1:-1] -= k * (p[:, :, 1:] - p[:, :, :-1])
        if self.solid is not None:
            self.vx *= self.mx; self.vy *= self.my; self.vz *= self.mz
        kp = RHO * C * C * self.dt / self.dx
        div = (self.vx[1:] - self.vx[:-1]) + (self.vy[:, 1:] - self.vy[:, :-1]) + (self.vz[:, :, 1:] - self.vz[:, :, :-1])
        self.p -= kp * div
        if src is not None:
            self.p[src] += amp                                           # soft (additive) pressure source
        if self.solid is not None:
            self.p[self.solid] = 0
        self.p *= self.damp


def verify_free_field(f0=2000.0, ppw=10, size_m=1.6):
    """Point source in free space vs the closed form: arrival time (wave speed) and 1/r amplitude at several radii."""
    lam = C / (2.5 * f0)                                                # shortest significant wavelength of a Ricker
    dx = lam / ppw
    n = int(size_m / dx)
    g = Grid((n, n, n), dx, sponge=max(12, n // 10))
    c0 = (n // 2, n // 2, n // 2)
    radii = [0.2, 0.35, 0.5]
    probes = [(c0[0] + int(round(r / dx)), c0[1], c0[2]) for r in radii]
    T = int((max(radii) / C + 3.0 / f0) / g.dt)
    rec = np.zeros((len(radii), T))
    t = np.arange(T) * g.dt
    s = ricker(t, f0)
    for it in range(T):
        g.step(c0, float(s[it]))
        for k, pr in enumerate(probes):
            rec[k, it] = float(g.p[pr])
    # closed form for a monopole: the pressure keeps its SHAPE with range, p(r, t) ~ q(t - r/c) / r. So the travel time
    # between two radii is the cross-correlation lag of their traces (sub-sample by a parabola through the peak, as in
    # cybersweat's dsp::tdoa), and the amplitude ratio is the ratio of their norms.
    def lag(a, b):
        cc = np.correlate(b, a, "full")
        i = int(np.argmax(cc))
        y0, y1, y2 = cc[i - 1], cc[i], cc[i + 1]
        return (i - (len(a) - 1)) + 0.5 * (y0 - y2) / (y0 - 2 * y1 + y2)
    rr = [(pr[0] - c0[0]) * dx for pr in probes]
    rows = []
    for k in range(1, len(radii)):
        tt = lag(rec[0], rec[k]) * g.dt
        speed = (rr[k] - rr[0]) / tt
        rows.append({"r": round(rr[k], 3), "speed_err_pct": round(100 * (speed - C) / C, 3),
                     "amp_ratio_vs_1_over_r": round(float(np.linalg.norm(rec[k]) / np.linalg.norm(rec[0]) / (rr[0] / rr[k])), 3)})
    return {"dx_mm": round(1000 * dx, 2), "cells": n ** 3, "steps": T, "device": g.dev, "rows": rows}


def verify_convergence():
    """Second-order scheme: the wave-speed error should fall ~4x when points per wavelength double."""
    # the box must keep every probe outside the sponge (a probe inside it is damped: amplitude ratios of 0.5-0.7)
    return {ppw: verify_free_field(ppw=ppw, size_m=1.6)["rows"][-1] for ppw in (6, 8, 12)}


def verify_rigid_wall(f0=2000.0, ppw=10, d=0.30):
    """A rigid plane at distance d from a source/receiver pair: the echo must arrive at the image-source delay 2d/c
    (relative to the direct pulse at the receiver) with amplitude ~ r_direct / r_image (perfect reflection)."""
    lam = C / (2.5 * f0); dx = lam / ppw
    n = int(1.4 / dx)
    solid = np.zeros((n, n, n), bool)
    wall_i = n // 2 + int(round(d / dx))
    solid[wall_i:] = True
    g = Grid((n, n, n), dx, solid=solid, sponge=max(12, n // 10))
    src = (n // 2, n // 2, n // 2)
    rcv = (n // 2, n // 2 + int(round(0.05 / dx)), n // 2)          # 5 cm beside the source, like a mic beside a speaker
    T = int((2 * d / C + 0.004) / g.dt)
    t = np.arange(T) * g.dt; s = ricker(t, f0); rec = np.zeros(T)
    for it in range(T):
        g.step(src, float(s[it])); rec[it] = float(g.p[rcv])
    # free-field reference with the same grid, no wall
    g2 = Grid((n, n, n), dx, sponge=max(12, n // 10)); ref = np.zeros(T)
    for it in range(T):
        g2.step(src, float(s[it])); ref[it] = float(g2.p[rcv])
    echo = rec - ref
    i_d, i_e = int(np.argmax(np.abs(ref))), int(np.argmax(np.abs(echo)))
    d_eff = (wall_i - src[0]) * dx - 0.5 * dx                       # the reflecting face sits half a cell before the solid
    b = (rcv[1] - src[1]) * dx
    r_img = math.hypot(2 * d_eff, b)
    pred_delay = (r_img - b) / C
    return {"dx_mm": round(1000 * dx, 2), "wall_m": round(d_eff, 3), "echo_delay_ms": round(1000 * (t[i_e] - t[i_d]), 3),
            "image_source_delay_ms": round(1000 * pred_delay, 3),
            "echo_amp_ratio": round(float(abs(echo[i_e]) / abs(ref[i_d])), 3), "image_source_amp_ratio": round(b / r_img, 3)}


if __name__ == "__main__":
    if sys.argv[1:2] == ["verify"]:
        import time
        t0 = time.time()
        print("free field:", verify_free_field(), f"{time.time() - t0:.1f} s")
        print("convergence (speed error % at 8 and 16 points per wavelength):", verify_convergence())
        print("rigid wall:", verify_rigid_wall(), f"{time.time() - t0:.1f} s total")
