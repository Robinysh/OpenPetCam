"""The fused room: geometry nobody measured with a tape, from two sensors that each see half of it.

Vision gives directions precisely and distances poorly; sound gives distances and no direction. So: the floor height
comes from the acoustic floor echo (cross-checked against the owner's tape measure when given), and each wall is placed
where its floor junction, read from the calibrated panorama, meets that floor. A mounting surface is included only
when its clearance from the reference lens is explicitly declared with a source. A nearby acoustic peak is retained
as an unclassified observation, never promoted into that surface. predict_echoes() runs the forward model (first-order specular, monostatic)
for every sonar direction so the viewer can show measured against predicted.

Ceiling height is not observed (the sweep tops out below it): CEILING_M is an assumption and is labelled as one."""
import json

import numpy as np

from .geometry import dir_from, floor_hit

CEILING_M = 2.6            # assumed floor-to-ceiling, not measured
WALL_TOP_M = 2.6


class Room:
    def __init__(self, picks, floor_height, overhead_m, provenance):
        self.picks, self.floor_height, self.overhead_m, self.provenance = picks, floor_height, overhead_m, provenance
        self.walls = [self._wall(s) for s in picks["surfaces"]] + [self._acoustic_wall(s) for s in picks.get("acoustic_surfaces", [])]

    def _wall(self, s):
        az, el = np.array(s["pts"], float).T
        p = floor_hit(dir_from(az, el), self.floor_height)[:, [0, 2]]
        c = p.mean(0)
        u = np.linalg.svd(p - c)[2][0] if len(p) > 1 else np.array([1.0, 0])
        t = (p - c) @ u
        a, b = c + u * t.min(), c + u * t.max()
        if np.linalg.norm(b - a) < 0.3:                      # a pilaster or a short run: give it some width
            a, b = c - u * 0.15, c + u * 0.15
        n = np.array([-u[1], u[0]])
        dist = float(abs(c @ n))
        return {"label": s["label"], "a": a.round(3).tolist(), "b": b.round(3).tolist(), "distance_m": round(dist, 3),
                "floor_pts": p.round(3).tolist(), "fit_rms_m": round(float(np.sqrt(np.mean(((p - c) @ n) ** 2))), 3)}

    def _acoustic_wall(self, s):
        """A vertical surface whose range came from sound: perpendicular to the middle of its azimuth span."""
        a0, a1 = s["az"]
        mid = np.radians((a0 + a1) / 2)
        n = np.array([np.sin(mid), np.cos(mid)])
        u = np.array([n[1], -n[0]])
        c = n * s["range_m"]
        half = s["range_m"] * np.tan(np.radians((a1 - a0) / 2))
        a, b = c - u * half, c + u * half
        return {"label": s["label"], "a": a.round(3).tolist(), "b": b.round(3).tolist(), "distance_m": s["range_m"],
                "floor_pts": [], "fit_rms_m": 0.0, "source": s["source"]}

    def outline(self, step=1.0):
        """The floor's edge as seen from the lens: for each azimuth, the nearest modelled wall (None where no wall)."""
        out = []
        segs = [(np.array(w["a"]), np.array(w["b"]), w["label"]) for w in self.walls]
        for az in np.arange(-180, 180, step):
            d = np.array([np.sin(np.radians(az)), np.cos(np.radians(az))])
            best = None
            for a, b, lab in segs:
                e = b - a
                M = np.array([d, -e]).T
                if abs(np.linalg.det(M)) < 1e-9:
                    continue
                t, s = np.linalg.solve(M, a)
                if t > 0 and -0.02 <= s <= 1.02 and (best is None or t < best[0]):
                    best = (float(t), lab)
            out.append([float(az), round(best[0], 3) if best else None, best[1] if best else None])
        return out

    def planes(self):
        """Every surface as (label, point, unit normal, bounds check) for the forward model."""
        h = self.floor_height
        out = [("floor", np.array([0, -h, 0.0]), np.array([0, 1.0, 0]), lambda q: True)]
        if self.overhead_m:
            out.append(("countertop overhang", np.array([0, self.overhead_m, 0.0]), np.array([0, -1.0, 0]),
                        lambda q: True))
        for w in self.walls:
            a, b = np.array(w["a"]), np.array(w["b"])
            u = (b - a) / np.linalg.norm(b - a)
            n2 = np.array([-u[1], u[0]])
            n = np.array([n2[0], 0, n2[1]])
            p0 = np.array([a[0], 0, a[1]])
            L = np.linalg.norm(b - a)

            def inside(q, a=a, u=u, L=L):
                s = (np.array([q[0], q[2]]) - a) @ u
                return -0.05 <= s <= L + 0.05 and -h <= q[1] <= WALL_TOP_M - h
            out.append((w["label"], p0, n, inside))
        return out

    def predict_echoes(self, head_dir):
        """First-order specular echoes for a colocated speaker/mic at the lens: one-way range = distance to the plane
        along its normal, if the foot point lies on the surface. Weight: how far off the head's axis it is."""
        res = []
        for label, p0, n, inside in self.planes():
            s = float(p0 @ n)
            dist, foot = abs(s), n * s                    # nearest point of the plane to the lens
            if not inside(foot):
                continue
            off = float(np.degrees(np.arccos(np.clip(foot @ head_dir / (np.linalg.norm(foot) + 1e-9), -1, 1))))
            res.append({"label": label, "range_m": round(dist, 3), "off_axis_deg": round(off, 1)})
        return sorted(res, key=lambda r: r["range_m"])

    def to_json(self):
        return {"floor_height_m": round(self.floor_height, 3), "overhead_m": self.overhead_m, "ceiling_m": CEILING_M,
                "ceiling_assumed": True, "walls": self.walls, "outline": self.outline(), "objects": self.picks.get("objects", []),
                "provenance": self.provenance}


def acoustic_estimates(echo_layer):
    """Return the floor candidate and an unclassified near-echo proxy, plus their samples.

    The legacy near proxy is delay after an aligned response feature, converted
    with c*dt/2. Its nearest accepted peak does not identify a mounting surface
    or equal the lens-to-surface clearance of a separated speaker/microphone.
    """
    floor, near = [], []
    for it in echo_layer.data["items"]:
        e = it["echoes"]
        f = [r for r, s in e if 0.70 <= r <= 0.95]
        if f:
            floor.append(max(((r, s) for r, s in e if 0.70 <= r <= 0.95), key=lambda x: x[1])[0])
        nn = [r for r, s in e if r < 0.35 and s > 0.15]
        if nn:
            near.append(min(nn))
    return (float(np.median(floor)) if floor else None, float(np.median(near)) if near else None, floor, near)


def fuse(ctx, echo_layer, tape_m=None, *, mount_clearance_m=None, mount_clearance_source=None):
    """Build the room, keeping acoustic features separate from mechanical geometry.

    ``mount_clearance_m`` is an explicitly declared +Y distance from the fixed
    reference level lens origin to its mounting surface, not from the moving
    lens at each pose. Its source is mandatory. Without it the mounting plane
    stays unknown; the nearest echo is diagnostic evidence only.
    """
    import hashlib
    import os
    if mount_clearance_m is not None:
        if isinstance(mount_clearance_m, (bool, np.bool_)):
            raise ValueError("mount clearance must be a finite positive distance, not a boolean")
        mount_clearance_m = float(mount_clearance_m)
        if not np.isfinite(mount_clearance_m) or mount_clearance_m <= 0:
            raise ValueError("mount clearance must be a finite positive distance")
        if not isinstance(mount_clearance_source, str) or not mount_clearance_source.strip():
            raise ValueError("a source is required for declared mount geometry")
    picks = json.load(open(ctx.path("picks.json")))
    rb = ctx.path("picks_rebased.json")
    cur_sha = hashlib.sha256(open(ctx.path("calib.json"), "rb").read()).hexdigest()[:16]
    if os.path.exists(rb) and json.load(open(rb)).get("rebased", {}).get("to_calib_sha") == cur_sha:
        picks = json.load(open(rb))                    # marks moved into the current calibration (rebase.py)
    fh, near, floors, nears = acoustic_estimates(echo_layer)
    if tape_m:
        h, src = tape_m, f"tape measure {tape_m} m (acoustic {fh})"
    elif fh:
        h, src = fh, f"acoustic floor echo, median of {[round(x, 2) for x in floors]} m"
    else:
        h, src = 0.85, "assumed 0.85 m"
    prov = {"floor_height": src,
            "overhead": (f"declared reference-lens-to-mount clearance {mount_clearance_m} m; {mount_clearance_source.strip()}"
                         if mount_clearance_m is not None else "unknown: no mechanical reference-lens-to-mount datum supplied"),
            "overhead_status": "declared_mount_geometry" if mount_clearance_m is not None else "unknown_mount_geometry",
            "mount_clearance_datum": "fixed reference level lens origin; +Y toward mounting surface",
            "unclassified_near_echo_proxy_m": round(near, 3) if near is not None else None,
            "near_echo_proxy_samples_m": nears,
            "near_echo_proxy_status": "unclassified acoustic feature; not mounting-surface geometry",
            "walls": picks["note"], "ceiling": f"assumed {CEILING_M} m, not observed"}
    return Room(picks, h, mount_clearance_m, prov)
