"""Objects as boxes: the window (azimuth and elevation extent) is read from the calibrated panorama by eye; the box's
footprint, height and orientation come from the fused DA3 ranges inside that window, in metres from the sound floor echo.

kind: solid = from the floor to the top of its points; thin = a screen (5 cm deep); slab = only the height band its
points occupy (a stool seat: the legs are not modelled). The box is oriented to face the lens (its depth axis runs along
the window's mean azimuth), which is how the camera sees it, not necessarily how it stands in the room."""
import math

import numpy as np

from .geometry import dir_from


CAN_LIE_ON = ("sofa", "bed", "rug", "blanket")


CARVE_BLOCK_S = 7200          # carving uses the dog's sightlines from EVEN 2-hour blocks; odd blocks are held out to test it


def is_carving_row(t):
    return int(t // CARVE_BLOCK_S) % 2 == 0


def dog_floor_points(ctx):
    """Where the dog has stood (floor x, z), from the presence sensor, in the carving half of the time blocks only."""
    try:
        from .sensors.presence import Presence
        rows = Presence(ctx).layers()[0].data["rows"]
        return np.array([[r[1], r[2]] for r in rows if not r[3] and is_carving_row(r[0])])
    except Exception:
        return np.zeros((0, 2))


LIMITS = {"sofa": (2.4, 1.1), "chair": (1.0, 1.0), "bowls": (0.7, 0.5), "stand": (2.0, 0.7), "stool": (0.6, 0.6)}
MIN_WIDTH = {"sofa": 0.80, "chair": 0.55, "stand": 0.35}            # a prior: the lens sees only the near faces
DEPTH_PRIOR = {"sofa": 0.90, "chair": 0.75, "stand": 0.45}           # front-to-back, for boxes placed by their floor contact


def floor_contact(rng, i0, i1, j0, j1, W, H, h, tol=0.10):
    """Where an object meets the floor, column by column: scanning up from the bottom of its window, floor pixels have
    the floor plane's range (h / sin(depression)); the first pixel that does not is the object's base. Returns the
    contact distances (horizontal, m) of the columns that have one. Learned depth only classifies floor vs not; the
    distance comes from the sound-set floor plane and the calibrated angle, like the dog's position."""
    out = []
    for i in range(i0, i1):
        floor_seen = False
        for j in range(min(j1 + 25, H - 1), j0 - 1, -1):            # from below the window's bottom, upward
            el = 90 - (j + 0.5) / H * 180
            if el >= -1:
                break
            r = rng[j, i]
            fr = h / math.sin(math.radians(-el))
            if r > 0 and abs(r - fr) < tol * fr:
                floor_seen = True
                continue
            if floor_seen and r > 0:
                out.append(h / math.tan(math.radians(-(el - 0.5 * 180 / H))))
                break
    return np.array(out)


def _limits(label):
    for k, v in LIMITS.items():
        if k in label:
            return v
    return (1.5, 1.0)


def build(ctx, rng, W, H):
    """Boxes from the fused range inside each marked window.
    1. Objects are fitted nearest-first and each claims its pixels: a closer object in an overlapping window (the
       armchair in front of the sofa) can no longer be mistaken for the farther one.
    2. The footprint is the minimum-area rectangle of the object's near points on the floor plane (cv2.minAreaRect),
       so a sofa seen end-on is long away from the lens, not a box turned to face it; sizes are capped per kind.
    3. kind: solid = floor to top; thin = a 5 cm screen; slab = only the height band its points occupy (stool seats).
    4. Free space: a solid box (not one she lies on) is shortened along its long axis until no dog position is inside."""
    import cv2
    out, cut = [], np.zeros((H, W), bool)
    h = ctx.room.floor_height
    dog = dog_floor_points(ctx)
    win = []
    for o in ctx.room.picks.get("objects", []):
        if "az" not in o:
            continue
        (a0, a1), (e0, e1) = o["az"], o["el"]
        i0, i1 = int((a0 + 180) / 360 * W), int((a1 + 180) / 360 * W)
        j0, j1 = int((90 - e1) / 180 * H), int((90 - e0) / 180 * H)
        r = rng[j0:j1, i0:i1]
        med = float(np.median(r[r > 0])) if (r > 0).any() else 99
        win.append((med, o, (i0, i1, j0, j1)))
    claimed = np.zeros((H, W), bool)
    for med, o, (i0, i1, j0, j1) in sorted(win, key=lambda x: x[0]):
        r = rng[j0:j1, i0:i1]
        free = ~claimed[j0:j1, i0:i1]
        ok = (r > 0) & free
        if ok.sum() < 10:
            continue
        ok &= r <= np.percentile(r[ok], 40) * 1.25                   # the near cluster of what is left
        jj, ii = np.mgrid[j0:j1, i0:i1]
        az = (ii + 0.5) / W * 360 - 180
        el = 90 - (jj + 0.5) / H * 180
        P = dir_from(az[ok], el[ok]) * r[ok][:, None]
        xz = P[:, [0, 2]].astype(np.float32)
        (cx, cz), (w1, w2), ang = cv2.minAreaRect(xz)
        a = math.radians(ang)
        ax1 = np.array([math.cos(a), math.sin(a)])                     # (x, z) direction of the side of length w1
        if w2 > w1:                                                    # depth axis = the long side
            w1, w2 = w2, w1
            ax1 = np.array([-ax1[1], ax1[0]])
        ax_override = None
        Lmax, Wmax = _limits(o["label"])
        dp = next((v for k, v in DEPTH_PRIOR.items() if k in o["label"]), None)
        placed_by = "learned depth"
        contact_front = None
        if o["kind"] == "solid" and dp:
            cd = floor_contact(rng, i0, i1, j0, j1, W, H, h)
            if len(cd) >= 0.3 * (i1 - i0):
                front = float(np.median(cd))
                azm = math.radians((o["az"][0] + o["az"][1]) / 2)
                radial = np.array([math.sin(azm), math.cos(azm)])
                span = math.radians(o["az"][1] - o["az"][0])
                wtan = 2 * front * math.tan(span / 2)
                cx, cz = radial * (front + dp / 2)
                # the box faces the lens: depth axis radial; width tangential from the angular span at the contact distance
                w1, w2, ax_override = dp, wtan, radial
                placed_by = f"floor contact ({len(cd)} columns, median {front:.2f} m)"
                contact_front = front
        if ax_override is not None:
            ax1 = ax_override
        length, width = float(min(max(w1, 0.25), Lmax)), float(min(max(w2, 0.15), Wmax))
        if o["kind"] == "thin":
            width = 0.05
        c = np.array([cx, cz])
        mw = next((v for k, v in MIN_WIDTH.items() if k in o["label"]), None)
        if mw and width < mw:
            # widen AWAY from the lens: the measured near face stays where the points put it
            side = np.array([-ax1[1], ax1[0]])
            away = side if (side @ c) > 0 else -side
            c = c + away * (mw - width) / 2
            width = mw
        y = P[:, 1]
        top = float(np.percentile(y, 97))
        if contact_front is not None:
            # same geometry as the contact: the top edge's elevation at the contact distance
            top = contact_front * math.tan(math.radians(o["el"][1]))
        bottom = -h if o["kind"] == "solid" else float(np.percentile(y, 3))
        if o["kind"] == "slab":
            top = max(top, bottom + 0.06)
        carved = 0
        if o["kind"] == "solid" and len(dog) and not any(k in o["label"] for k in CAN_LIE_ON):
            d = dog - c
            u, v = d @ ax1, d @ np.array([-ax1[1], ax1[0]])
            inside = (np.abs(u) < length / 2) & (np.abs(v) < width / 2)
            if inside.any():
                carved = int(inside.sum())
                length = max(2 * float(np.min(np.abs(u[inside]))) - 0.04, 0.15)
        yaw = math.atan2(ax1[0], ax1[1])                               # box depth axis = (sin yaw, cos yaw) in (x, z)
        # sightline carving: every ray along which the camera saw the dog's feet is clear to her; a box may not block it
        lowered = 0
        if len(dog):
            fwd_b = np.array([math.sin(yaw), math.cos(yaw)]); side_b = np.array([math.cos(yaw), -math.sin(yaw)])
            ss = np.linspace(0.02, 0.98, 49)
            for q in dog:
                if np.hypot(*q) < 0.2:
                    continue
                pts = ss[:, None] * q[None]                          # floor-plane track of the ray
                yy = -h * ss                                         # the ray's height above... (y of the sightline)
                d = pts - c
                inb = (np.abs(d @ side_b) < width / 2) & (np.abs(d @ fwd_b) < length / 2) & (yy < top) & (yy > bottom)
                if inb.any():
                    top = float(yy[inb].max()) - 0.01
                    lowered += 1
            top = max(top, bottom + 0.05)
        out.append({"label": o["label"], "kind": o["kind"], "center": [round(float(c[0]), 3), round((top + bottom) / 2, 3), round(float(c[1]), 3)],
                    "size": [round(width, 3), round(top - bottom, 3), round(length, 3)], "yaw_deg": round(math.degrees(yaw), 1),
                    "front_m": round(contact_front if contact_front is not None else float(np.percentile(np.linalg.norm(P, axis=1), 10)), 2), "points": int(ok.sum()),
                    "carved_by_dog_positions": carved, "lowered_by_sightlines": lowered, "placed_by": placed_by})
        if contact_front is not None:
            # placed by its floor contact: everything the learned depth puts nearer than the box's back inside the
            # window is this object at the wrong distance (it formed slanted sheets toward the lens); the box owns it
            back = contact_front + dp + 0.3
            inside = (r > 0) & (r < back)
            cut[j0:j1, i0:i1] |= inside
            claimed[j0:j1, i0:i1] |= inside
    return out, cut
