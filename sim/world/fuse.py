"""Deep integration: one record per viewing direction (a surfel) carrying every modality's evidence about that patch
of room, with how strongly each one supports it. The camera only rotates, so a direction is the natural key: every
sensor that can be pointed (visible, near-IR, sound) or placed (the dog on the floor) maps onto the same grid.

Attributes (equirectangular W x H, written as model/surfels.npz and model/surfel_*.png for the viewer):
  range_m, normal (3), class (0 unseen, 1 floor, 2 wall/other, 3 object), object_id
  rgb (sweep), light (24 hourly brightness values: how the light on this patch moves through the day)
  nir_excess       near-IR minus visible (material cue)
  nir_reflectance  LED light x r^2 / cos(incidence), normalised: the patch's near-IR reflectance (active minus passive)
  echo_coverage    how many sound directions point within 35 deg of this patch (speaker on the back: axis reversed)
  echo_support     the measured echo envelope at this patch's range in those directions, relative to the envelope's
                   median: > 1 means sound sees something at this distance in this direction
  dwell_min        minutes the dog spent on the floor cell this ray lands on (floor surfels only)
  checks: floor_residual (range vs the sound-scaled floor plane, floor surfels), dog_free (dog stood here: free floor)
Every attribute is NaN where its modality has nothing to say; nothing is filled in silently."""
import math
import os

import numpy as np
from PIL import Image

from .geometry import directions


def _sample_pano(path, W, H):
    if not os.path.exists(path):
        return None
    a = np.asarray(Image.open(path).convert("RGB").resize((W, H), Image.BILINEAR), np.float32)
    a[a.sum(-1) < 6] = np.nan                                  # black = unseen
    return a


def normals_from_range(R, D):
    P = D * R[..., None]
    dx = np.roll(P, -1, 1) - np.roll(P, 1, 1)
    dy = np.roll(P, -1, 0) - np.roll(P, 1, 0)
    n = np.cross(dx, dy)
    n /= np.linalg.norm(n, axis=-1, keepdims=True) + 1e-12
    flip = np.sum(n * D, -1) > 0                               # normals face the lens
    n[flip] *= -1
    return n


def build(ctx, manifest):
    L = {l["id"]: l for l in manifest["layers"]}
    fl = L["fused.surface"]["data"]
    W, H = fl["width"], fl["height"]
    R = np.fromfile(os.path.join(ctx.model, fl["file"]), np.float32).reshape(H, W).copy()
    D = directions(W, H)
    h = manifest["room"]["floor_height_m"]
    out = {}
    # objects: ray-cast the boxes into the range so every direction has one surface
    from .acoustic_model import scene_ranges
    Rs = scene_ranges(R, fl.get("objects", []), W, H)
    obj_id = np.full((H, W), -1, np.int16)
    for k, o in enumerate(fl.get("objects", [])):
        only = scene_ranges(np.zeros_like(R), [o], W, H)
        hit = (only > 0) & (np.abs(only - Rs) < 1e-4)
        obj_id[hit] = k
    R = np.where(Rs > 0, Rs, np.nan)
    out["range_m"] = R
    out["object_id"] = obj_id
    n = normals_from_range(np.nan_to_num(R, nan=0.0), D)
    out["normal"] = np.where(np.isfinite(R)[..., None], n, np.nan)
    P = D * R[..., None]
    y = P[..., 1]
    cls = np.zeros((H, W), np.uint8)
    seen = np.isfinite(R)
    floor = seen & (np.abs(y + h) < 0.05) & (n[..., 1] > 0.8)
    cls[seen] = 2
    cls[floor] = 1
    cls[obj_id >= 0] = 3
    out["class"] = cls
    # the check must use the depth BEFORE it was snapped onto the floor plane, or it measures nothing (0.0 cm)
    raw = ctx._cache.get("depthpano_raw")
    if raw is not None and raw.shape == (H, W):
        yr = (D * raw[..., None])[..., 1]
        out["floor_residual_m"] = np.where(floor & np.isfinite(yr), yr + h, np.nan)
    else:
        out["floor_residual_m"] = np.full((H, W), np.nan)
    # visible light
    out["rgb"] = _sample_pano(os.path.join(ctx.model, "pano_visible_sweep.jpg"), W, H)
    light = np.full((H, W, 24), np.nan, np.float32)
    for hh, item in (L.get("visible.hourly", {}).get("data", {}).get("hours", {}) or {}).items():
        a = _sample_pano(os.path.join(ctx.model, item["image"]), W, H)
        if a is not None:
            light[..., int(hh)] = a.mean(-1)
    out["light"] = light
    # near-IR: material and reflectance (from the overlay panoramas; their encoding is a diverging map around grey 110)
    def overlay_value(name):
        a = _sample_pano(os.path.join(ctx.model, name), W, H)
        if a is None:
            return None
        return ((a[..., 0] - 110) / 145 - (a[..., 1] - 110) / 60) / 2   # inverse of the encoding: + more, - less
    nx = overlay_value("pano_nir_excess.jpg")
    if nx is not None:
        out["nir_excess"] = nx
    led = overlay_value("pano_led_light.jpg")
    if led is not None:
        cosi = np.clip(-np.sum(n * D, -1), 0.15, 1)
        refl = led * R ** 2 / cosi
        med = np.nanmedian(refl[np.isfinite(refl) & (refl > 0)]) if np.isfinite(refl).any() else 1
        out["nir_reflectance"] = refl / (med or 1)
    # sound: coverage and support per direction (speaker on the back of the pan body; the mic hears the room both ways)
    items = L["acoustic.echoes"]["data"]["items"]
    cov = np.zeros((H, W), np.float32)
    sup = np.zeros((H, W), np.float32)
    for it in items:
        head = ctx.camera.R(it["pan"], it["tilt"])[:, 2]
        spk = -np.array([head[0], 0, head[2]]); spk /= np.linalg.norm(spk) + 1e-9
        ra, en = np.array(it["range_m"]), np.array(it["envelope"])
        base = np.median(en[ra > 0.3]) if (ra > 0.3).any() else 1
        for axis in (head, spk):
            inb = (D @ axis) > math.cos(math.radians(35))
            m = inb & seen & (R < ra.max())
            cov[m] += 1
            sup[m] += np.interp(R[m], ra, en) / (base + 1e-9)
    out["echo_coverage"] = np.where(cov > 0, cov, np.nan)
    out["echo_support"] = np.where(cov > 0, sup / np.maximum(cov, 1), np.nan)
    # the dog: dwell per floor cell where each floor ray lands, and free-floor evidence
    rows = L.get("presence.dog", {}).get("data", {}).get("rows", [])
    cell = 0.1
    grid = {}
    for t, x, z, cut, mv, dw in rows:
        if cut:
            continue
        grid[(int(math.floor(x / cell)), int(math.floor(z / cell)))] = grid.get((int(math.floor(x / cell)), int(math.floor(z / cell))), 0) + dw / 60
    dwell = np.full((H, W), np.nan, np.float32)
    fx, fz = P[..., 0], P[..., 2]
    ii = np.floor(np.nan_to_num(fx) / cell).astype(int); jj = np.floor(np.nan_to_num(fz) / cell).astype(int)
    for (a, b), v in grid.items():
        dwell[(ii == a) & (jj == b) & floor] = v
    out["dwell_min"] = dwell
    out["dog_free"] = np.where(floor, np.isfinite(dwell).astype(np.float32), np.nan)
    np.savez_compressed(os.path.join(ctx.model, "surfels.npz"), **{k: v for k, v in out.items() if v is not None})
    # viewer textures (equirectangular, black = no data)
    from .colormaps import classes, div, seq
    ranges = {}                                          # each texture's value range, for the app's colour legends

    def tex(name, val, lo, hi, cmap):
        ranges[name] = [float(lo), float(hi)]
        v = np.clip((val - lo) / (hi - lo), 0, 1)
        rgb = np.zeros((H, W, 3), np.uint8)
        ok = np.isfinite(val)
        rgb[ok] = (cmap(v[ok]) * 255).astype(np.uint8)
        rgb[ok] = np.maximum(rgb[ok], 2)
        Image.fromarray(rgb).save(os.path.join(ctx.model, f"surfel_{name}.png"))
    nn = out["normal"]
    rgbn = np.where(np.isfinite(nn), (nn * 0.5 + 0.5) * 255, 0).astype(np.uint8)
    Image.fromarray(rgbn).save(os.path.join(ctx.model, "surfel_normals.png"))
    cls.astype(np.uint8).tofile(os.path.join(ctx.model, "surfel_class.u8"))   # raw class per direction (viewer dollhouse)
    tex("class", np.where(cls > 0, cls.astype(float), np.nan), 1, 3, classes)
    if "nir_reflectance" in out:
        tex("nir_reflectance", np.log(np.clip(out["nir_reflectance"], 1e-3, None)), -1.5, 1.5, div)
    if "nir_excess" in out:
        tex("nir_excess", out["nir_excess"], -1, 1, div)
    tex("echo_support", out["echo_support"], 0, 3, seq)
    tex("echo_coverage", out["echo_coverage"], 0, max(2, float(np.nanmax(out["echo_coverage"])) if np.isfinite(out["echo_coverage"]).any() else 2), seq)
    tex("dwell", np.log1p(out["dwell_min"]), 0, max(1.0, float(np.nanmax(np.log1p(out["dwell_min"]))) if np.isfinite(out["dwell_min"]).any() else 1), seq)
    tex("floor_residual", out["floor_residual_m"], -0.15, 0.15, div)
    day = np.nanmax(light, -1) - np.nanmin(light, -1)
    tex("light_range", day, 0, 120, seq)
    unc, driver, term_names, importance, terms = uncertainty(ctx, out, W, H)
    out["uncertainty"] = unc
    out["uncertainty_driver"] = driver
    out["importance"] = importance
    tex("uncertainty", np.log10(np.clip(unc, 1e-3, None)), -2, 0, seq)
    np.savez_compressed(os.path.join(ctx.model, "surfels.npz"), **{k: v for k, v in out.items() if v is not None})
    # half-resolution probe table for the browser: one Float32 record per direction
    ph, pw = H // 2, W // 2
    sub = lambda a: a[::2, ::2]
    chans = [("range_m", out["range_m"]), ("nx", nn[..., 0]), ("ny", nn[..., 1]), ("nz", nn[..., 2]),
             ("class", np.where(cls > 0, cls, np.nan).astype(np.float32)), ("object_id", np.where(obj_id >= 0, obj_id, np.nan).astype(np.float32)),
             ("nir_excess", out.get("nir_excess", np.full((H, W), np.nan))), ("nir_reflectance", out.get("nir_reflectance", np.full((H, W), np.nan))),
             ("echo_coverage", out["echo_coverage"]), ("echo_support", out["echo_support"]), ("dwell_min", out["dwell_min"]),
             ("floor_residual_m", out["floor_residual_m"]), ("uncertainty", unc),
             ("uncertainty_driver", np.where(np.isfinite(unc), driver, np.nan).astype(np.float32))] + [(f"light_{k:02d}", light[..., k]) for k in range(24)]
    probe = np.stack([sub(np.asarray(c, np.float32)) for _n, c in chans], -1).astype(np.float32)
    probe.tofile(os.path.join(ctx.model, "surfels_probe.f32"))
    summary = {
        "texture_ranges": ranges,
        "grid": [W, H], "seen_share": round(float(seen.mean()), 3),
        "class_share": {k: round(float((cls == v).mean()), 3) for k, v in (("floor", 1), ("wall_other", 2), ("object", 3))},
        "floor_residual_cm_median_abs": round(100 * float(np.nanmedian(np.abs(out["floor_residual_m"]))), 1),
        "echo_covered_share": round(float(np.isfinite(out["echo_coverage"]).mean() / max(seen.mean(), 1e-9)), 3),
        "nir_reflectance_share": round(float(np.isfinite(out.get("nir_reflectance", np.full((1,), np.nan))).mean() / max(seen.mean(), 1e-9)), 3) if "nir_reflectance" in out else 0,
        "floor_with_dog": round(float(np.nansum(out["dog_free"]) / max(floor.sum(), 1)), 3),
        "attributes": sorted(k for k, v in out.items() if v is not None),
        "uncertainty_terms": term_names,
        "uncertainty_median": round(float(np.nanmedian(unc)), 4),
        "uncertainty_weighted_mean": round(float(np.nansum(np.nan_to_num(unc) * importance) / np.nansum(importance * np.isfinite(unc))), 4),
        "driver_share": {term_names[k]: round(float(np.mean(driver[np.isfinite(unc)] == k)), 3) for k in range(len(term_names))},
        "class_file": "surfel_class.u8", "probe": {"file": "surfels_probe.f32", "width": pw, "height": ph, "channels": [n for n, _c in chans]},
        "textures": {"uncertainty": "surfel_uncertainty.png", "class": "surfel_class.png", "normals": "surfel_normals.png", "echo_support": "surfel_echo_support.png",
                     "echo_coverage": "surfel_echo_coverage.png", "dwell": "surfel_dwell.png",
                     "floor_residual": "surfel_floor_residual.png", "light_range": "surfel_light_range.png",
                     **({"nir_reflectance": "surfel_nir_reflectance.png"} if "nir_reflectance" in out else {}),
                     **({"nir_excess": "surfel_nir_excess.png"} if "nir_excess" in out else {})},
    }
    return summary, out


def uncertainty(ctx, out, W, H):
    """Per-surfel relative range uncertainty (sigma_r / r) from independent pieces of evidence, plus what drives it.
    Terms (each a relative error, combined in quadrature):
      frames      spread of the DA3 frames that saw the direction (std/median); 0.25 if only one frame saw it
      grazing     a surface seen edge-on: 0.05 / |cos(incidence)| - 0.05
      edge        next to a depth jump (>15% between neighbours): 0.15
      floor       floor surfels: |residual| / range against the sound-set floor plane (independent of the frames)
      conflict    the dog was seen standing beyond this surface along the same ray: 0.5 (the surface cannot be there)
    Unseen directions get NaN. Importance (for planning what to look at): floor near where the dog goes counts most."""
    import math
    ev = os.path.join(ctx.model, "depth_evidence.npz")
    R = out["range_m"]
    D = directions(W, H)
    terms = {}
    if os.path.exists(ev):
        z = np.load(ev)
        n, sp = z["views"], z["spread"]
        terms["frames"] = np.where(n > 1, np.nan_to_num(sp, nan=0.25), 0.25).astype(np.float32)
    nn = out["normal"]
    cosi = np.abs(np.sum(np.nan_to_num(nn) * D, -1))
    terms["grazing"] = np.clip(0.05 / np.maximum(cosi, 0.05) - 0.05, 0, 1).astype(np.float32)
    Rz = np.nan_to_num(R, nan=0)
    jump = np.zeros_like(Rz, bool)
    for ax, sh in ((0, 1), (0, -1), (1, 1), (1, -1)):
        o = np.roll(Rz, sh, ax)
        jump |= (Rz > 0) & (o > 0) & (np.abs(o - Rz) > 0.15 * Rz)
    terms["edge"] = np.where(jump, 0.15, 0).astype(np.float32)
    fr = out.get("floor_residual_m")
    terms["floor"] = np.where(np.isfinite(fr), np.abs(np.nan_to_num(fr)) / np.maximum(Rz, 0.1), 0).astype(np.float32)
    # conflicts: dog positions beyond the surface along their ray
    conflict = np.zeros((H, W), np.float32)
    h = ctx.room.floor_height if ctx.room else 0.81
    try:
        from .sensors.presence import Presence
        for t, x, zz, cut, mv, dw in Presence(ctx).layers()[0].data["rows"]:
            if cut:
                continue
            az = math.degrees(math.atan2(x, zz)); rh = math.hypot(x, zz); el = -math.degrees(math.atan2(h, rh))
            i, j = int((az + 180) / 360 * W) % W, min(H - 1, int((90 - el) / 180 * H))
            if Rz[j, i] > 0 and math.hypot(rh, h) > Rz[j, i] * 1.15:
                conflict[max(0, j - 1):j + 2, max(0, i - 1):(i + 2)] = 0.5
    except Exception:
        pass
    terms["conflict"] = conflict
    total = np.sqrt(sum(t.astype(np.float64) ** 2 for t in terms.values())).astype(np.float32)
    total = np.where(np.isfinite(R), total, np.nan)
    driver = np.argmax(np.stack([terms[k] for k in terms]), 0).astype(np.int16)
    # importance: floor (and anything below 0.5 m above it) counts 3x, places the dog goes 5x, the rest 1x
    P = D * Rz[..., None]
    low = np.isfinite(R) & (P[..., 1] < -h + 0.5)
    imp = np.where(low, 3.0, 1.0) * np.where(np.isfinite(out.get("dwell_min", np.full((H, W), np.nan))), 5.0 / 3.0, 1.0)
    imp = np.where(np.isfinite(R), imp, 2.0)                     # unseen directions matter too: they are the unknown
    return total, driver, list(terms), imp.astype(np.float32), terms
