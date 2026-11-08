"""The initial acoustic room model: the measured room simplified to walls, floor and ceiling.

Where it sits in the pipeline (README "Build your own"):
    sweep -> calibration -> learned depth (DA3) -> pano-depth (depthpano.py): metres from the floor echo or the
    tape, written unsnapped as model/range_pano_metric.f32 -> **acoustic_room.json** (this module; app_bundle.py
    copies it into the Room bundle) -> wave solver, echograms, geometry inversion.
Every acoustic experiment starts from this file. Inversions refine its parameters; they never redraw it by hand.

Method (deterministic, no learned parts):
1. Points: one range per viewing direction, kept only where the camera actually imaged (not blind-filled),
   farther than `near_m` (learned depth inflates the near field) and below `max_el_deg`.
2. Footprint (plan, `cell_m` grid): floor points, closed and hole-filled, united with the plan region the lens
   sees out to the walls (per azimuth, the far points standing >= `wall_min_m` above the floor, i.e. above
   furniture). The component containing the camera.
3. Walls: the footprint outline snapped to the room's two dominant perpendicular directions (rectilinear
   simplification). Learned depth scaled by the floor echo under-reads with distance (about 0.86x at 2-3 m on
   the floor, 0.75x on walls at 2-4 m in the first home), so where a wall's foot is seen at least 15 deg below
   the horizon (1 px of a 768-wide panorama is ~2.5% of the distance there) its distance comes from the floor-wall junction instead (floor height / tan(depression): exact
   geometry, no depth scale). `distance_source` says which was used. Each wall is `imaged` if measured wall points cover >= 30% of it, otherwise an
   `occlusion` boundary (where the view ends, e.g. behind furniture or round a corner). Both bound the air;
   only imaged walls are echo surfaces with measured positions.
4. Floor: the metric floor height. Ceiling: the bundle's value, flagged as assumed unless measured.
5. Furniture: the bundle's object boxes, turned parallel/perpendicular to the walls (measured yaw kept).
Not modelled here: the camera's own mount (a shelf or counter right above the lens). Give it as a `mount` block.

Frame: the room model's (left-handed; the Room tab mirrors x for display). The lens is the origin, y up.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

SCHEMA = "openpetcam.acoustic-room.v1"


def _points(R, imaged, near_m, max_el_deg):
    H, W = R.shape
    az = np.radians((np.arange(W) + .5) / W * 360 - 180)
    el = np.radians(90 - (np.arange(H) + .5) / H * 180)
    A, E = np.meshgrid(az, el)
    ok = imaged & np.isfinite(R) & (R > near_m) & (np.degrees(E) < max_el_deg)
    return (np.cos(E) * np.sin(A) * R)[ok], (np.sin(E) * R)[ok], (np.cos(E) * np.cos(A) * R)[ok]


def _footprint(x, y, z, floor_y, cell, wall_min_m):
    import cv2
    from scipy import ndimage
    lo = np.array([x.min(), z.min()]) - 0.5
    n = np.ceil((np.array([x.max(), z.max()]) + 0.5 - lo) / cell).astype(int) + 1
    ij = lambda X, Z: (np.clip(((X - lo[0]) / cell).astype(int), 0, n[0] - 1), np.clip(((Z - lo[1]) / cell).astype(int), 0, n[1] - 1))
    floor = y < floor_y + 0.06
    fp = np.zeros(n, np.uint8); fp[ij(x[floor], z[floor])] = 1
    fp = cv2.morphologyEx(fp, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))).astype(bool)
    hi = y > floor_y + wall_min_m
    nb = 720
    ab = ((np.arctan2(x[hi], z[hi]) + np.pi) / (2 * np.pi) * nb).astype(int) % nb
    rh = np.hypot(x[hi], z[hi])
    reach = np.zeros(nb)
    order = np.lexsort((rh, ab)); ab_s, rh_s = ab[order], rh[order]
    starts = np.searchsorted(ab_s, np.arange(nb)); ends = np.searchsorted(ab_s, np.arange(nb), side="right")
    for b in np.where(ends - starts >= 3)[0]:
        seg = rh_s[starts[b]:ends[b]]
        reach[b] = seg[int(0.95 * (len(seg) - 1))]
    gx, gz = np.meshgrid(lo[0] + cell * (np.arange(n[0]) + .5), lo[1] + cell * (np.arange(n[1]) + .5), indexing="ij")
    gb = ((np.arctan2(gx, gz) + np.pi) / (2 * np.pi) * nb).astype(int) % nb
    fp |= np.hypot(gx, gz) < reach[gb]
    fp |= np.hypot(gx, gz) < 0.5                       # the lens's own surroundings (inside the near-field cut)
    fp = cv2.morphologyEx(fp.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)).astype(bool)
    fp = ndimage.binary_fill_holes(fp)
    lab, _ = ndimage.label(fp)
    fp = lab == lab[ij(np.array([0.0]), np.array([0.0]))][0]
    return fp, lo, cell, (x[hi], z[hi])


def _dominant_angle(contour_xz, smooth_m=0.10):
    """Room axis from the smoothed outline: a raster contour's own 1-cell steps sit at 0/45/90 degrees."""
    import cv2
    p = cv2.approxPolyDP(contour_xz.astype(np.float32).reshape(-1, 1, 2), smooth_m, True)[:, 0, :].astype(float)
    d = np.diff(np.vstack([p, p[:1]]), axis=0)
    L = np.hypot(d[:, 0], d[:, 1]); th = np.arctan2(d[:, 1], d[:, 0])
    c = np.sum(L * np.exp(4j * th))                    # 4x: directions modulo 90 degrees
    return float(np.angle(c) / 4)


def _refine_angle(wall_xz, theta, span_deg=4.0, step_deg=0.05, bin_m=0.04):
    """Sharpen the room axis: the angle at which the wall points pile up on the fewest lines along both axes."""
    best, best_s = theta, -1.0
    for a in np.radians(np.arange(-span_deg, span_deg + 1e-9, step_deg)) + theta:
        c, s = math.cos(a), math.sin(a)
        u = wall_xz @ np.array([c, s]); v = wall_xz @ np.array([-s, c])
        sc = sum(float(np.sum(np.bincount(((w - w.min()) / bin_m).astype(int)).astype(float) ** 2)) for w in (u, v))
        if sc > best_s:
            best, best_s = a, sc
    return float(best)


def _rectilinear(contour_xz, theta, tol_m, min_m):
    """Snap a closed outline to the two directions theta, theta+90 deg. Returns polygon vertices (x, z)."""
    import cv2
    c, s = math.cos(theta), math.sin(theta)
    Rm = np.array([[c, s], [-s, c]])                   # world -> (u, v) aligned with the room
    q = contour_xz @ Rm.T
    poly = cv2.approxPolyDP(q.astype(np.float32).reshape(-1, 1, 2), tol_m, True)[:, 0, :].astype(float)
    # each edge: axis 0 = runs along u (constant v), axis 1 = runs along v (constant u)
    edges = []
    for i in range(len(poly)):
        a, b = poly[i], poly[(i + 1) % len(poly)]
        ax = 0 if abs(b[0] - a[0]) >= abs(b[1] - a[1]) else 1
        edges.append([ax, (a[1] + b[1]) / 2 if ax == 0 else (a[0] + b[0]) / 2, float(np.hypot(*(b - a))), b.copy()])
    # merge neighbours on the same axis and nearly the same line; drop slivers into their neighbours
    changed = True
    while changed and len(edges) > 4:
        changed = False
        for i in range(len(edges)):
            j = (i + 1) % len(edges)
            e, f = edges[i], edges[j]
            if e[0] == f[0] and abs(e[1] - f[1]) < tol_m * 1.5:
                w = e[2] + f[2]
                edges[i] = [e[0], (e[1] * e[2] + f[1] * f[2]) / w, w, f[3]]; del edges[j]; changed = True; break
        if changed:
            continue
        k = int(np.argmin([e[2] for e in edges]))
        if edges[k][2] < min_m:
            p, nx = edges[k - 1], edges[(k + 1) % len(edges)]
            if p[0] == nx[0]:                          # a jog: fold it into the longer neighbour line
                keep = p if p[2] >= nx[2] else nx
                p[1] = nx[1] = keep[1]
            edges[k - 1][3] = edges[k][3]                 # the previous edge now ends where the dropped one did
            del edges[k]; changed = True
    # a jog (two runs on the same axis at different offsets, e.g. where a room narrows into a corridor) gets
    # the short perpendicular wall that joins them, at the outline vertex between them; never drop a run
    out = []
    for i, e in enumerate(edges):
        out.append(e)
        f = edges[(i + 1) % len(edges)]
        if e[0] == f[0]:
            v = e[3]
            out.append([1 - e[0], float(v[0] if e[0] == 0 else v[1]), abs(e[1] - f[1]), v])
    return out, Rm


def _snap_to_points(edges, Rm, wall_xz, band_m=0.4, min_pts=40):
    """Move each wall line onto the median of the measured wall points beside it (within band_m, inside
    its span). Lines with too few points stay where the outline put them."""
    q = wall_xz @ Rm.T
    V = _vertices(edges, Rm) @ Rm.T
    for i, e in enumerate(edges):
        a, b = V[i - 1], V[i]                        # this edge runs between the vertex before and after it
        lo_s, hi_s = sorted((a[0], b[0]) if e[0] == 0 else (a[1], b[1]))
        along = q[:, 0] if e[0] == 0 else q[:, 1]
        across = q[:, 1] if e[0] == 0 else q[:, 0]
        sel = (along > lo_s + 0.1) & (along < hi_s - 0.1) & (np.abs(across - e[1]) < band_m)
        if sel.sum() >= min_pts:
            e[1] = float(np.median(across[sel]))
    return edges


def _junctions(R, imaged, floor_height_m, min_el_deg=15.0):
    """Per azimuth column, the horizontal distance to where the floor meets a vertical surface, from the
    junction's elevation and the floor height alone: h / tan(depression). Surface orientation (floor faces up,
    walls sideways) is read from the range's normals, which barely depend on the depth scale, so this is
    free of learned-depth scale errors. Columns whose junction is shallower than min_el_deg are dropped
    (one pixel there is worth tens of centimetres). Returns azimuths (rad) and distances (nan = none)."""
    H, W = R.shape
    az = np.radians((np.arange(W) + .5) / W * 360 - 180)
    el = np.radians(90 - (np.arange(H) + .5) / H * 180)
    A, E = np.meshgrid(az, el)
    D = np.stack([np.cos(E) * np.sin(A), np.sin(E), np.cos(E) * np.cos(A)], -1)
    P = D * np.where(np.isfinite(R), R, 0)[..., None]
    n = np.cross(np.roll(P, -1, 1) - np.roll(P, 1, 1), np.roll(P, -1, 0) - np.roll(P, 1, 0))
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    n[np.sum(n * D, -1) > 0] *= -1
    ny = n[..., 1]
    ok = imaged & np.isfinite(R) & (R > 0)
    dist = np.full(W, np.nan)
    for i in range(W):
        last_floor = None
        for j in range(H - 1, H // 2, -1):                 # from straight down up to the horizon
            if not ok[j, i]:
                continue
            if ny[j, i] > 0.75:
                last_floor = j
            elif last_floor is not None and ny[j, i] < 0.3:
                run = sum(1 for k in range(j, max(j - 8, 0), -1) if ok[k, i] and abs(ny[k, i]) < 0.35)
                if run >= 6:
                    # the junction lies between the last floor row and the first vertical row above it
                    dep = -float(np.degrees(0.5 * (el[last_floor] + el[j])))
                    if dep >= min_el_deg:
                        dist[i] = floor_height_m / math.tan(math.radians(dep))
                    break
    return az, dist


def _place_by_junction(edges, Rm, jaz, jdist, min_n=8):
    """Move each wall line to the floor-junction distance where enough of its foot is seen. Junctions nearer
    than 0.9x the depth-based line are furniture feet in front of it and are ignored. Returns, per edge,
    the source of its distance."""
    q = np.stack([np.sin(jaz) * jdist, np.cos(jaz) * jdist], 1) @ Rm.T
    good = np.isfinite(jdist)
    V = _vertices(edges, Rm) @ Rm.T
    src = []
    for i, e in enumerate(edges):
        a, b = V[i - 1], V[i]
        ax = e[0]                                          # 0: line v = e[1]; 1: line u = e[1]
        along, across = (q[:, 0], q[:, 1]) if ax == 0 else (q[:, 1], q[:, 0])
        lo_s, hi_s = sorted((a[0], b[0]) if ax == 0 else (a[1], b[1]))
        with np.errstate(invalid="ignore", divide="ignore"):
            hit = along * e[1] / across                    # where the junction's ray meets the line
            sel = good & (np.sign(across) == np.sign(e[1])) & (hit > lo_s + 0.1) & (hit < hi_s - 0.1) \
                & (np.abs(across) / np.hypot(along, across) > 0.3) & (np.abs(across) >= 0.9 * abs(e[1]))
        d = np.sort(np.abs(across[sel]))
        best = (0, None)
        for k in range(len(d)):                            # densest 15 cm window of junction distances
            m = int(np.searchsorted(d, d[k] + 0.15, side="right")) - k
            if m > best[0]:
                best = (m, d[k:k + m])
        # one clear foot line: enough samples, and most of them in the same 15 cm
        if best[0] >= min_n and best[0] >= 0.6 * len(d):
            e[1] = float(np.sign(e[1]) * np.median(best[1]))
            src.append({"source": "floor junction", "samples": int(best[0])})
        else:
            src.append({"source": "depth", "samples": int(best[0])})
    return edges, src


def _vertices(edges, Rm):
    # consecutive edges must alternate axes; vertices are their intersections
    out = []
    for i in range(len(edges)):
        e, f = edges[i], edges[(i + 1) % len(edges)]
        if e[0] == f[0]:
            continue
        u = f[1] if e[0] == 0 else e[1]
        v = e[1] if e[0] == 0 else f[1]
        out.append([u, v])
    return np.array(out) @ Rm                          # back to world


def _snap_object(o, room_axis_deg):
    """Furniture stands parallel or perpendicular to the walls: snap the box yaw to the room's axes.
    A box's size[0] runs along (cos yaw, -sin yaw) in (x, z) (acoustic_model.scene_ranges), so aligned
    means yaw = -room_axis (mod 90). The measured yaw is kept."""
    o = dict(o)
    if "yaw_deg" in o:
        y = float(o["yaw_deg"])
        o["yaw_measured_deg"] = y
        o["yaw_deg"] = round(-room_axis_deg + 90.0 * round((y + room_axis_deg) / 90.0), 2)
    return o


def build(R, imaged, floor_height_m, ceiling_m, *, objects=(), ceiling_measured=False, near_m=0.6, max_el_deg=38.0,
          cell_m=0.05, wall_min_m=1.1, tol_m=0.15, min_wall_m=0.4, snap_objects=True, provenance=None):
    """R: metric range panorama (H x W, metres; equirectangular, azimuth -180..180 left to right, elevation
    +90 at the top). imaged: H x W bool, True where the camera actually saw (not filled)."""
    import cv2
    floor_y = -float(floor_height_m)
    x, y, z = _points(np.asarray(R, float), np.asarray(imaged, bool), near_m, max_el_deg)
    if len(x) < 1000:
        raise ValueError("too few imaged range samples to build a room")
    fp, lo, cell, (wx, wz) = _footprint(x, y, z, floor_y, cell_m, wall_min_m)
    cs, _ = cv2.findContours(fp.astype(np.uint8).T.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cont = max(cs, key=len)[:, 0, :].astype(float)
    cxz = np.stack([lo[0] + (cont[:, 0] + .5) * cell, lo[1] + (cont[:, 1] + .5) * cell], 1)
    theta = _refine_angle(np.stack([wx, wz], 1), _dominant_angle(cxz))
    edges, Rm = _rectilinear(cxz, theta, tol_m, min_wall_m)
    edges = _snap_to_points(edges, Rm, np.stack([wx, wz], 1))
    depth_coord = [e[1] for e in edges]
    jaz, jdist = _junctions(np.asarray(R, float), np.asarray(imaged, bool), floor_height_m)
    V_depth = _vertices(edges, Rm)                         # where the measured wall points are (for `imaged`)
    edges, wsrc = _place_by_junction([list(e) for e in edges], Rm, jaz, jdist)
    V, theta_deg = _vertices(edges, Rm), float(np.degrees(theta))
    area = 0.5 * float(np.sum(V[:, 0] * np.roll(V[:, 1], -1) - np.roll(V[:, 0], -1) * V[:, 1]))
    # outline fit: distance from each footprint contour sample to the polygon
    def seg_dist(P, a, b):
        u = b - a; t = np.clip(((P - a) @ u) / (u @ u), 0, 1); return np.hypot(*(P - (a + t[:, None] * u)).T)
    dist = np.min([seg_dist(cxz, V[i], V[(i + 1) % len(V)]) for i in range(len(V))], axis=0)
    W2 = np.stack([wx, wz], 1)
    walls = []
    for i in range(len(V)):
        a, b = V[i], V[(i + 1) % len(V)]
        L = float(np.hypot(*(b - a)))
        u = (b - a) / L
        nrm = np.array([-u[1], u[0]]) * (1 if area > 0 else -1)          # into the room
        ad, bd = V_depth[i], V_depth[(i + 1) % len(V)]
        Ld = float(np.linalg.norm(bd - ad)); ud = (bd - ad) / Ld
        along = (W2 - ad) @ ud; off = np.abs((W2 - ad) @ np.array([-ud[1], ud[0]]))
        hit = along[(off < 0.12) & (along >= 0) & (along <= Ld)]
        cover = float(len(np.unique(np.floor(hit / 0.1))) * 0.1 / Ld) if len(hit) else 0.0
        d = float(-(nrm @ a))
        walls.append({"id": f"wall_{i}", "a": [round(float(a[0]), 3), round(float(a[1]), 3)], "b": [round(float(b[0]), 3), round(float(b[1]), 3)],
                      "length_m": round(L, 3), "normal": [round(float(nrm[0]), 4), 0.0, round(float(nrm[1]), 4)],
                      "distance_m": round(abs(d), 3), "imaged_fraction": round(min(cover, 1.0), 2),
                      "kind": "imaged" if cover >= 0.3 else "occlusion",
                      "distance_source": wsrc[(i + 1) % len(wsrc)]["source"],
                      "junction_samples": wsrc[(i + 1) % len(wsrc)]["samples"],
                      "depth_distance_m": round(abs(depth_coord[(i + 1) % len(wsrc)]), 3)})
    # self-calibration: on walls placed by their junction (beyond 1.2 m), how short does depth read? Walls still
    # placed by depth get a distance range from that ratio, measured in this room (none if no such wall).
    ratios = [w["depth_distance_m"] / w["distance_m"] for w in walls
              if w["distance_source"] == "floor junction" and w["distance_m"] > 1.2]
    depth_ratio = float(np.median(ratios)) if ratios else None
    for w in walls:
        if w["distance_source"] == "depth":
            hi = w["distance_m"] / depth_ratio if depth_ratio and depth_ratio < 1 else w["distance_m"]
            w["distance_range_m"] = [w["distance_m"], round(hi, 3)]
        else:
            w["distance_range_m"] = [w["distance_m"], w["distance_m"]]
    doc = {"schema": SCHEMA, "frame": "room model frame: lens at origin, y up, left-handed (the Room tab mirrors x)",
           "floor": {"y": round(floor_y, 3), "source": "metric floor height (sound floor echo or tape)"},
           "ceiling": {"y": round(floor_y + float(ceiling_m), 3), "height_m": float(ceiling_m),
                       "source": "measured" if ceiling_measured else "assumed (never imaged)"},
           "walls": walls, "footprint": [[round(float(p[0]), 3), round(float(p[1]), 3)] for p in V],
           "footprint_area_m2": round(abs(area), 2), "room_axis_deg": round(theta_deg, 2),
           "depth_ratio_on_walls": None if depth_ratio is None else round(depth_ratio, 3),
           "outline_fit_m": {"median": round(float(np.median(dist)), 3), "p90": round(float(np.percentile(dist, 90)), 3)},
           "objects": [_snap_object(o, theta_deg) if snap_objects else dict(o) for o in objects],
           "mount": None,
           "method": {"near_m": near_m, "max_el_deg": max_el_deg, "cell_m": cell_m, "wall_min_m": wall_min_m,
                      "tol_m": tol_m, "min_wall_m": min_wall_m, "imaged_samples": int(len(x))},
           "limits": ["walls are the measured outline snapped to two perpendicular directions",
                      "occlusion walls bound what was seen, not a measured surface",
                      "the near field (< near_m) and the camera's mount are not modelled; add a mount block",
                      "ceiling height is assumed unless marked measured"],
           "provenance": provenance or {}}
    return doc


def from_world(world):
    """Build from a world folder after `sim.world.build` (the pano-depth pipeline): the clean metric range
    (model/range_pano_metric.f32: DA3 in metres, before any snapping to modelled walls), imaged = directions at
    least one DA3 frame saw (model/depth_evidence.npz), the floor height the build measured (sound or tape),
    and the furniture boxes it fitted."""
    model = os.path.join(world, "model")
    m = json.load(open(os.path.join(model, "manifest.json")))
    fused = next((l for l in m["layers"] if l["id"] == "fused.surface"), {}).get("data") or {}
    W, H = int(fused["width"]), int(fused["height"])
    path = os.path.join(model, fused.get("metric_file") or "range_pano_metric.f32")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: rebuild with `python3 -m sim.world.build {world}` (pano-depth writes it)")
    R = np.fromfile(path, np.float32).reshape(H, W)
    views = np.load(os.path.join(model, "depth_evidence.npz"))["views"]
    room = m["room"]
    return build(R, (views > 0) & (R > 0), room["floor_height_m"], room["ceiling_m"], objects=fused.get("objects", []),
                 provenance={"world": os.path.abspath(world), "built": m.get("built"), "range": os.path.basename(path),
                             "imaged": "depth_evidence.npz views > 0", "floor": room.get("floor_source") or "manifest room.floor_height_m"})


def from_bundle(bundle):
    """Build from a Room bundle (models/room): room.json + room_range.u16 + room_photo.jpg (imaged mask)."""
    from PIL import Image
    info = json.load(open(os.path.join(bundle, "room.json")))
    W, H = info["range_shape"]
    R = np.fromfile(os.path.join(bundle, info["textures"].get("range", "room_range.u16")), "<u2").reshape(H, W) / 1000.0
    ph = np.asarray(Image.open(os.path.join(bundle, info["textures"]["photo"])).convert("L").resize((W, H), Image.NEAREST))
    imaged = (ph > 8) & (R > 0)
    return build(R, imaged, info["floor_height_m"], info["ceiling_m"], objects=info.get("objects", []),
                 provenance={"bundle_built": info.get("built"), "range": "room_range.u16", "imaged_mask": "room_photo.jpg coverage"})


def main(argv):
    """python3 -m sim.world.acoustic_room eval/<world>   (preferred: the clean metric range)
       python3 -m sim.world.acoustic_room models/room    (a Room bundle: its range is already snapped/cut)"""
    src = argv[1] if len(argv) > 1 else "models/room"
    if os.path.exists(os.path.join(src, "model", "manifest.json")):
        doc = from_world(src); out = os.path.join(src, "model", "acoustic_room.json")
    else:
        doc = from_bundle(src); out = os.path.join(src, "acoustic_room.json")
    json.dump(doc, open(out, "w"), indent=1)
    im = sum(w["kind"] == "imaged" for w in doc["walls"])
    print(f"wrote {out}: {len(doc['walls'])} walls ({im} imaged, {len(doc['walls']) - im} occlusion), "
          f"{doc['footprint_area_m2']} m2, room axis {doc['room_axis_deg']} deg, outline fit median {doc['outline_fit_m']['median']} m, "
          f"floor y {doc['floor']['y']}, ceiling y {doc['ceiling']['y']} ({doc['ceiling']['source']})")


if __name__ == "__main__":
    main(sys.argv)
