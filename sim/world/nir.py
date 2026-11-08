"""Qualified active near-IR differences and approximate surface-response geometry.

Processed night pictures measure reflected visible/NIR illumination, not heat.
Unknown camera response or automatic exposure cannot be repaired by normalizing
against distant surfaces. Display images are never measurement inputs.
"""
import numpy as np


_FLAGS = ("controls_verified", "exposure_locked", "gain_locked", "processing_locked",
          "same_night_mode", "ircut_unchanged", "pose_fixed")
_READBACK = ("exposure_us", "analog_gain", "digital_gain", "processing_signature",
             "night_mode", "ircut", "pan", "tilt")


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and np.isfinite(value)


def capture_quality(metadata):
    """Require verified acquisition controls and a known radiometric response."""
    reasons = []
    if metadata.get("status") not in (None, "complete"):
        reasons.append("capture did not complete successfully")
    reasons.extend(str(r) for r in metadata.get("quality_failures", []))
    if metadata.get("schema_version") != 2 or metadata.get("protocol") != "nir-led-off-on-off":
        reasons.append("legacy or unknown acquisition protocol")
    reasons += [f"{key} is unverified" for key in _FLAGS if metadata.get(key) is not True]
    readbacks = metadata.get("controls_readbacks") or []
    if len(readbacks) < 3 or any(not isinstance(r, dict) for r in readbacks):
        reasons.append("missing per-frame control readbacks")
    else:
        for key in _READBACK:
            values = [r.get(key) for r in readbacks]
            if any(v is None for v in values) or any(v != values[0] for v in values[1:]):
                reasons.append(f"{key} missing or changed during capture")
        if any(r.get("verified") is not True for r in readbacks):
            reasons.append("control readback is not verified")
        for key in ("exposure_us", "analog_gain", "digital_gain"):
            if any(not _number(r.get(key)) or r[key] <= 0 for r in readbacks):
                reasons.append(f"invalid {key}")
        for key in ("pan", "tilt"):
            if any(not _number(r.get(key)) for r in readbacks):
                reasons.append(f"invalid {key}")
        led_states = [r.get("led") for r in readbacks]
        changes = [v for i, v in enumerate(led_states) if i == 0 or v != led_states[i - 1]]
        if changes != ["off", "on", "off"]:
            reasons.append("LED off/on/off readbacks are unverified")
        if any(r.get("night_mode") != "night" or r.get("ircut") != "out" for r in readbacks):
            reasons.append("night mode/filter is not fixed in the NIR-sensitive state")
        if any(not isinstance(r.get("processing_signature"), str) or not r["processing_signature"] for r in readbacks):
            reasons.append("missing image-processing state")
    rad = metadata.get("radiometry") or {}
    if rad.get("verified") is not True:
        reasons.append("camera response has not been verified")
    model = rad.get("response_model")
    if model not in ("linear", "lut"):
        reasons.append("unknown pixel-to-light response")
    black, white = rad.get("black_level", 0.), rad.get("white_level", 255.)
    if not _number(black) or not _number(white) or white <= black:
        reasons.append("invalid radiometric code range")
    if any(not _number(rad.get(key, default)) or rad.get(key, default) < 0 for key, default in (("noise_floor", 1 / 255), ("max_off_drift", .02))):
        reasons.append("invalid radiometric noise or drift limit")
    if model == "lut":
        try:
            lut = np.asarray(rad.get("lut", []), float)
        except (ValueError, TypeError):
            lut = np.empty(0)
        if lut.ndim != 1 or len(lut) < 2 or not np.isfinite(lut).all() or np.any(np.diff(lut) < 0) or not (lut[-1] > lut[0]):
            reasons.append("missing or invalid inverse-response lookup table")
    return {"quantitative": not reasons, "status": "controlled_linear_signal" if not reasons else "qualitative_only",
            "reasons": list(dict.fromkeys(reasons)),
            "limitations": ["off/on/off detects some ambient drift and motion; on-only motion and specular highlights can remain",
                            "reflected illumination is not thermal imaging or a material identity"]}


def _gray(a):
    a = np.asarray(a, float)
    return a.mean(axis=-1) if a.ndim == 3 else a


def _linearize(a, rad):
    black, white = rad.get("black_level", 0.), rad.get("white_level", 255.)
    code = (np.asarray(a, float) - black) / (white - black)
    if rad["response_model"] == "lut":
        lut = np.asarray(rad["lut"], float)
        code = np.interp(code, np.linspace(0, 1, len(lut)), lut)
    # Linearize channels BEFORE averaging; applying a transfer curve to an RGB
    # average is a different, generally incorrect measurement.
    return _gray(code)


def difference_signal(off_before, on, off_after, metadata, *, times=None, motion_mask=None):
    """Subtract time-interpolated off frames in verified linear response units.

    Unknown radiometry still yields encoded_delta for a qualitative display.
    Negative/low-signal, saturated and unstable pixels never become reflectivity.
    """
    images = [np.asarray(a, float) for a in (off_before, on, off_after)]
    if any(a.shape != images[0].shape for a in images):
        raise ValueError("IR images must have identical dimensions")
    if images[0].ndim not in (2, 3) or (images[0].ndim == 3 and images[0].shape[-1] not in (1, 3)):
        raise ValueError("IR images must be grayscale or channel-last images")
    t = np.asarray(times if times is not None else (0., 1., 2.), float)
    if t.shape != (3,) or not np.isfinite(t).all() or not t[0] < t[1] < t[2]:
        raise ValueError("on frame must lie strictly between the two off times")
    weight = (t[1] - t[0]) / (t[2] - t[0])
    a, b, c = [_gray(im) for im in images]
    encoded = b - ((1 - weight) * a + weight * c)
    quality = capture_quality(metadata)
    missing = np.full(a.shape, np.nan)
    result = {"encoded_delta": encoded, "signal": missing.copy(), "valid": np.zeros(a.shape, bool),
              "noise": missing.copy(), "quality": quality, "off_interpolation_weight": float(weight)}
    if not quality["quantitative"]:
        return result
    rad = metadata["radiometry"]
    before, active, after = [_linearize(im, rad) for im in images]
    signal = active - ((1 - weight) * before + weight * after)
    drift = np.abs(after - before)
    noise_floor = float(rad.get("noise_floor", 1 / 255))
    noise = np.maximum(noise_floor, drift / 2)
    black, white = rad.get("black_level", 0.), rad.get("white_level", 255.)
    margin = (white - black) / 255
    finite = np.isfinite(np.stack(images)).all(axis=0)
    finite &= (np.minimum.reduce(images) >= black) & (np.maximum.reduce(images) <= white)
    saturated = np.maximum.reduce(images) >= white - margin
    if finite.ndim == 3:
        finite, saturated = finite.all(axis=-1), saturated.any(axis=-1)
    valid = finite & ~saturated & (drift <= float(rad.get("max_off_drift", .02)))
    valid &= np.isfinite(signal) & (signal > 3 * noise)
    if motion_mask is not None:
        valid &= ~np.asarray(motion_mask, bool)
    # Preserve signed linear values for diagnosis, accompanied by a separate
    # validity mask; downstream estimators must use that mask.
    result.update(signal=signal, noise=noise, valid=valid,
                  drift=drift, saturated=saturated)
    quality["valid_share"] = float(valid.mean())
    return result


def geometry_response(signal, range_m, normals, directions, *, valid=None, beam=None,
                      light_position=None, min_cos=.2, range_spread=None,
                      max_relative_range_spread=.25):
    """Correct illumination distance/incidence and normalize to the valid median.

    With unknown beam/sensor shading this is a response PROXY, not material
    reflectivity. A caller may supply their calibrated combined illumination
    and camera sensitivity map as beam. No extra camera-leg inverse square is
    applied to resolved-surface radiance. Grazing points are rejected, not clamped.
    """
    signal, ranges = np.broadcast_arrays(np.asarray(signal, float), np.asarray(range_m, float))
    d, n = np.asarray(directions, float), np.asarray(normals, float)
    if d.shape != signal.shape + (3,) or n.shape != d.shape:
        raise ValueError("directions and normals must match the signal grid with three components")
    dn, nn = np.linalg.norm(d, axis=-1), np.linalg.norm(n, axis=-1)
    good = np.isfinite(signal) & (signal > 0) & np.isfinite(ranges) & (ranges > 0)
    good &= np.isfinite(d).all(axis=-1) & np.isfinite(n).all(axis=-1) & (dn > 0) & (nn > 0)
    d = np.divide(d, dn[..., None], out=np.zeros_like(d), where=dn[..., None] > 0)
    n = np.divide(n, nn[..., None], out=np.zeros_like(n), where=nn[..., None] > 0)
    light = np.zeros(3) if light_position is None else np.asarray(light_position, float)
    toward = light - ranges[..., None] * d
    distance = np.linalg.norm(toward, axis=-1)
    cosine = np.sum(n * np.divide(toward, distance[..., None], out=np.zeros_like(toward), where=distance[..., None] > 0), axis=-1)
    good &= np.isfinite(distance) & (distance > 0) & (cosine >= min_cos)
    illumination = np.ones_like(signal) if beam is None else np.broadcast_to(np.asarray(beam, float), signal.shape)
    good &= np.isfinite(illumination) & (illumination > 0)
    if valid is not None:
        good &= np.broadcast_to(np.asarray(valid, bool), signal.shape)
    if range_spread is not None:
        spread = np.broadcast_to(np.asarray(range_spread, float), signal.shape)
        good &= np.isfinite(spread) & (spread >= 0) & (spread <= max_relative_range_spread)
    values = np.full(signal.shape, np.nan)
    np.divide(signal * distance ** 2, cosine * illumination, out=values, where=good)
    good &= np.isfinite(values) & (values > 0)
    scale = float(np.median(values[good])) if good.any() else None
    response = np.where(good, values / scale, np.nan) if scale else np.full(signal.shape, np.nan)
    return {"response": response, "valid": good, "normalization": scale,
            "light_distance_m": distance, "incidence_cos": cosine,
            "kind": "beam_compensated_response" if beam is not None else "geometry_corrected_response",
            "assumptions": ["approximate range and surface normals", "matte direct illumination, no unmodelled shadows",
                            *( ["unknown LED beam and camera shading remain in the response"] if beam is None else [] ),
                            *( ["light treated as coincident with camera"] if light_position is None else [] )]}
