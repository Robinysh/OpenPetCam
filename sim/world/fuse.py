"""Deep integration: one record per viewing direction (a surfel) carrying every modality's evidence about that patch
of room, with how strongly each one supports it. The camera only rotates, so a direction is the natural key: every
sensor that can be pointed (visible, near-IR, sound) or placed (the dog on the floor) maps onto the same grid.

Attributes (equirectangular W x H, written as model/surfels.npz and model/surfel_*.png for the viewer):
  range_m, normal (3), class (0 unseen, 1 floor, 2 wall/other, 3 object), object_id
  rgb (sweep), light (24 hourly brightness values: how the light on this patch moves through the day)
  nir_excess       qualitative filter-switch code-value difference, not a material identity
  nir_led_difference qualitative on/off difference in encoded JPEG levels
  nir_led_signal   controlled, linearized LED-added response (never reconstructed from a display image)
  nir_response     approximate distance/incidence-corrected response; unknown beam/shading remain
  echo_coverage    how many sound directions point within 35 deg of this patch (speaker on the back: axis reversed)
  echo_support     the measured echo envelope at this patch's range in those directions, relative to the envelope's
                   median: > 1 means sound sees something at this distance in this direction
  dwell_min        minutes the dog spent on the floor cell this ray lands on (floor surfels only)
  checks: floor_residual (range vs the sound-scaled floor plane, floor surfels), dog_free (dog stood here: free floor)
Every attribute is NaN where its modality has nothing to say; nothing is filled in silently."""
import hashlib
import json
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


def infrared_fields(ctx, layers, ranges, normals, directions_world):
    """Fuse only numeric IR measurements whose acquisition evidence qualifies.

    Old JPEG overlays remain display-only; their colours cannot be inverted to
    recover a radiometric measurement. No strict reflectivity is inferred from
    an uncalibrated LED beam or camera shading.
    """
    from .nir import geometry_response
    h, w = ranges.shape
    source = layers.get("infrared.led", {}).get("data", {})
    info = {"status": "unavailable", "reflectivity_valid": False, "response_valid": False,
            "quantitative_observations": 0, "legacy_observations": int(source.get("legacy_observations", 0)),
            "qualitative_observations": int(source.get("qualitative_observations", source.get("legacy_observations", 0))),
            "source": None, "limitations": list(source.get("limitations", []))}
    out = {}
    name = source.get("numeric_file")
    if source.get("schema_version") != 2 or not name:
        if "infrared.led" in layers or os.path.exists(os.path.join(ctx.model, "pano_led_light.jpg")):
            info.update(status="qualitative_only")
            info["limitations"].append("legacy display images lack qualified numeric measurements")
        return out, info
    try:
        if os.path.basename(name) != name:
            raise ValueError("IR measurement must be a model-local filename")
        if not source.get("numeric_sha256") or not source.get("calibration_sha256"):
            raise ValueError("IR numeric measurement or calibration provenance is missing")
        with open(os.path.join(ctx.model, name), "rb") as measurement:
            if hashlib.sha256(measurement.read()).hexdigest() != source["numeric_sha256"]:
                raise ValueError("IR numeric measurement hash mismatch")
        calibration_sha = hashlib.sha256(json.dumps(ctx.camera.c, sort_keys=True).encode()).hexdigest()
        if calibration_sha != source["calibration_sha256"]:
            raise ValueError("IR panorama uses a different camera calibration")
        with np.load(os.path.join(ctx.model, name), allow_pickle=False) as z:
            if int(z["schema_version"]) != 2:
                raise ValueError("unknown IR numeric schema")
            shape = z["led_signal"].shape
            if len(shape) != 2 or shape != (source.get("height"), source.get("width")):
                raise ValueError("IR numeric grid does not match metadata")
            sy = np.minimum(np.arange(h) * shape[0] // h, shape[0] - 1)
            sx = np.minimum(np.arange(w) * shape[1] // w, shape[1] - 1)
            def sample(key):
                a = z[key]
                if a.shape != shape:
                    raise ValueError(f"IR {key} shape mismatch")
                return a[sy[:, None], sx[None, :]]
            if "infrared.nir_excess" in layers and "filter_delta" in z:
                filt = sample("filter_delta")
                if np.isfinite(filt).any():
                    out["nir_excess"] = filt
            if info["qualitative_observations"] > 0 and "encoded_delta" in z:
                delta = sample("encoded_delta")
                if np.isfinite(delta).any():
                    out["nir_led_difference"] = delta
            if source.get("quantitative_observations", 0) <= 0 or source.get("response_valid") is not True:
                info["status"] = "qualitative_only"
                return out, info
            signal, valid = sample("led_signal"), sample("valid").astype(bool)
            valid &= np.isfinite(signal) & (signal > 0)
            out["nir_led_signal"] = np.where(valid, signal, np.nan)
    except (OSError, ValueError, KeyError) as error:
        info["limitations"].append(str(error))
        return out, info
    # Depth boundaries make finite-difference normals unreliable. Reject them,
    # rather than hiding grazing/edge uncertainty with a cosine floor.
    geometry_valid = np.isfinite(ranges) & (ranges > 0)
    for axis, shift in ((0, 1), (0, -1), (1, 1), (1, -1)):
        neighbor = np.roll(ranges, shift, axis)
        geometry_valid &= np.isfinite(neighbor) & (neighbor > 0) & (np.abs(neighbor - ranges) <= .15 * ranges)
    geometry_valid[[0, -1]] = False
    spread = None
    evidence = os.path.join(ctx.model, "depth_evidence.npz")
    if os.path.exists(evidence):
        with np.load(evidence) as depth:
            if depth["spread"].shape == ranges.shape and depth["views"].shape == ranges.shape:
                spread = depth["spread"].copy()
                geometry_valid &= depth["views"] >= 2
            else:
                geometry_valid[:] = False
                info["limitations"].append("depth evidence grid mismatch")
    else:
        info["limitations"].append("range uncertainty is unavailable")
    response = geometry_response(signal, ranges, normals, directions_world,
                                 valid=valid & geometry_valid, range_spread=spread)
    if response["valid"].any():
        out["nir_response"] = response["response"]
    info.update(status="geometry_corrected_response" if "nir_response" in out else "controlled_linear_signal",
                response_valid="nir_response" in out, quantitative_observations=int(source["quantitative_observations"]),
                source=name, response_normalization=response["normalization"])
    info["limitations"] += response["assumptions"]
    return out, info


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
    infrared, infrared_info = infrared_fields(ctx, L, R, n, D)
    out.update(infrared)
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
    if "nir_response" in out:
        tex("nir_response", np.log(out["nir_response"]), -1.5, 1.5, div)
    if "nir_led_signal" in out:
        tex("nir_led_signal", out["nir_led_signal"], -.1, .1, div)
    if "nir_led_difference" in out:
        tex("nir_led_difference", out["nir_led_difference"], -16, 16, div)
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
             ("nir_response", out.get("nir_response", np.full((H, W), np.nan))), ("nir_led_signal", out.get("nir_led_signal", np.full((H, W), np.nan))),
             ("nir_led_difference", out.get("nir_led_difference", np.full((H, W), np.nan))),
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
        "infrared": infrared_info,
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
                     **({"nir_response": "surfel_nir_response.png"} if "nir_response" in out else {}),
                     **({"nir_led_signal": "surfel_nir_led_signal.png"} if "nir_led_signal" in out else {}),
                     **({"nir_led_difference": "surfel_nir_led_difference.png"} if "nir_led_difference" in out else {}),
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
