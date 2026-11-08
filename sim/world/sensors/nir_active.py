"""Active near-IR with explicit acquisition quality and numeric measurements.

Legacy filter/LED JPEG experiments remain qualitative. Only verified controls
and linearized samples enter the numeric LED signal used by room fusion.
"""
import hashlib
import json
import os

import numpy as np
from PIL import Image

from ..geometry import directions
from ..layers import Band, Layer
from ..nir import difference_signal
from .base import Sensor


def _image(root, name):
    with Image.open(os.path.join(root, name)) as image:
        return np.asarray(image.convert("RGB"), float)


def _groups(root, log):
    """Consecutive fixed-LED bursts, or the historical uncontrolled triplet."""
    groups = []
    for frame in log.get("frames", []):
        if "phase" not in frame:
            continue
        key = (frame["phase"], frame.get("led"))
        if not groups or groups[-1]["key"] != key:
            groups.append({"key": key, "led": frame.get("led"), "frames": []})
        groups[-1]["frames"].append(frame)
    if groups:
        result = []
        for group in groups:
            frames = group["frames"]
            failures = []
            if len(frames) % 2 == 0:
                failures.append("even encoded burst cannot be median-linearized without radiometric bias")
            for frame in frames:
                for key in ("controls", "controls_after"):
                    if frame.get(key, {}).get("led") != frame["led"]:
                        failures.append("per-frame LED readback missing or changed")
                path = os.path.join(root, frame["file"])
                with open(path, "rb") as source:
                    digest = hashlib.sha256(source.read()).hexdigest()
                if digest != frame.get("sha256"):
                    failures.append("captured image hash missing or mismatched")
            images = np.stack([_image(root, f["file"]) for f in frames])
            result.append({"image": np.median(images, axis=0),
                           "t": float(np.median([f["t"] for f in frames])),
                           "led": group["led"],
                           "controls": [state for f in frames for state in (f.get("controls", {}), f.get("controls_after", {}))],
                           "quality_failures": list(dict.fromkeys(failures)),
                           "motion": np.max(np.ptp(images, axis=0), axis=-1) > 8.})
        return result
    lookup = {f["file"]: f for f in log.get("frames", [])}
    result = []
    for i, name in enumerate(("B_passive_1.jpg", "C_active_1.jpg", "B2_passive_1.jpg")):
        image = _image(root, name)
        result.append({"image": image, "t": float(lookup.get(name, {}).get("t", i)),
                       "led": "on" if i == 1 else "off", "controls": [],
                       "quality_failures": ["legacy acquisition has no verified radiometry"],
                       "motion": np.zeros(image.shape[:2], bool)})
    return result


def _signature(metadata):
    """Never merge observations on different radiometric/exposure scales."""
    state = metadata["controls_readbacks"][0]
    keys = ("exposure_us", "analog_gain", "digital_gain", "processing_signature")
    return json.dumps({"controls": {k: state[k] for k in keys}, "radiometry": metadata["radiometry"]}, sort_keys=True)


def _overlay(arr, path, span):
    seen = np.isfinite(arr)
    value = np.clip(np.nan_to_num(arr) / span, -1, 1)
    pos, neg = np.clip(value, 0, 1), np.clip(-value, 0, 1)
    rgb = np.stack([110 + 145 * pos - 80 * neg, 110 - 80 * pos + 60 * neg,
                    110 + 120 * pos + 100 * neg], axis=-1).clip(1, 255).astype(np.uint8)
    rgb[~seen] = 0
    Image.fromarray(rgb).save(path, quality=92)


class ActivePassiveNIR(Sensor):
    name = "infrared.active"
    band = Band("infrared", 800e-9, 1000e-9,
                "reflected night-mode illumination; driver label ir940, spectrum not measured")
    needs_room = True

    def layers(self):
        ctx = self.ctx
        root = ctx.path("raw", "ir")
        if not os.path.isdir(root):
            return []
        pw, ph = 1024, 512
        directions_world = directions(pw, ph)
        acc = {k: np.zeros((ph, pw)) for k in ("signal", "noise", "encoded", "filter")}
        counts = {k: np.zeros((ph, pw), np.uint16) for k in ("signal", "encoded", "filter")}
        stats, common_signature, input_logs = [], None, set()
        for exp in sorted(os.listdir(root)):
            folder = os.path.join(root, exp)
            logpath = os.path.join(folder, "log.json")
            if not os.path.isfile(logpath):
                continue
            try:
                with open(logpath, "rb") as source:
                    logbytes = source.read()
                log_sha = hashlib.sha256(logbytes).hexdigest()
                if log_sha in input_logs:
                    stats.append({"experiment": exp, "rejected": "duplicate capture log; not an independent observation"})
                    continue
                input_logs.add(log_sha)
                log = json.loads(logbytes)
                groups = _groups(folder, log)
                pan, tilt = log.get("pos", [log.get("pan"), log.get("tilt")])
                if pan is None or tilt is None or not groups:
                    raise ValueError("capture pose or frames missing")
                h, w = groups[0]["image"].shape[:2]
                u, v, visible = ctx.camera.project(pan, tilt, directions_world, w, h,
                                                  off=log.get("pose_offset_deg", (0, 0, 0)))
                visible &= ~((v < .06 * h) & (u < .36 * w))
                ui, vi = np.clip(u.astype(int), 0, w - 1), np.clip(v.astype(int), 0, h - 1)
                observations = []
                for i in range(1, len(groups) - 1):
                    before, on, after = groups[i - 1:i + 2]
                    if (before["led"], on["led"], after["led"]) != ("off", "on", "off"):
                        continue
                    metadata = dict(log)
                    metadata["controls_readbacks"] = before["controls"] + on["controls"] + after["controls"]
                    metadata["quality_failures"] = before["quality_failures"] + on["quality_failures"] + after["quality_failures"]
                    result = difference_signal(before["image"], on["image"], after["image"], metadata,
                                               times=(before["t"], on["t"], after["t"]),
                                               motion_mask=before["motion"] | on["motion"] | after["motion"])
                    qualitative = result["encoded_delta"][vi, ui]
                    good = visible & np.isfinite(qualitative)
                    acc["encoded"][good] += qualitative[good]
                    counts["encoded"][good] += 1
                    quality = result["quality"]
                    if quality["quantitative"]:
                        signature = _signature(metadata)
                        if common_signature is None:
                            common_signature = signature
                        if signature != common_signature:
                            quality = dict(quality, quantitative=False, status="incompatible_radiometric_scale",
                                           reasons=["exposure/gain/processing/response differs from reference capture"])
                        else:
                            valid = visible & result["valid"][vi, ui]
                            acc["signal"][valid] += result["signal"][vi, ui][valid]
                            acc["noise"][valid] += result["noise"][vi, ui][valid]
                            counts["signal"][valid] += 1
                    observations.append({"on_time": on["t"], "quality": quality,
                                         "median_encoded_added": float(np.median(qualitative[good])) if good.any() else None,
                                         "median_off_change": float(np.median(np.abs(after["image"] - before["image"])))})
                # Filter switching changes spectrum and often processing, too.
                # Keep the old A/B observation without a material interpretation.
                if all(os.path.isfile(os.path.join(folder, name)) for name in ("A_visible_1.jpg", "B_passive_1.jpg")):
                    filt = (_image(folder, "B_passive_1.jpg") - _image(folder, "A_visible_1.jpg")).mean(-1)[vi, ui] / 255
                    good = visible & np.isfinite(filt)
                    acc["filter"][good] += filt[good]
                    counts["filter"][good] += 1
                stats.append({"experiment": exp, "pan": pan, "tilt": tilt, "observations": observations,
                              "log_sha256": log_sha})
            except (OSError, ValueError, KeyError, IndexError) as error:
                stats.append({"experiment": exp, "rejected": str(error)})
        if not stats:
            return []

        def mean(key, count):
            return np.divide(acc[key], counts[count], out=np.full((ph, pw), np.nan), where=counts[count] > 0)
        signal, noise = mean("signal", "signal"), mean("noise", "signal")
        encoded, filt = mean("encoded", "encoded"), mean("filter", "filter")
        quantitative = int(sum(o["quality"]["quantitative"] for s in stats for o in s.get("observations", [])))
        legacy = int(sum(not o["quality"]["quantitative"] for s in stats for o in s.get("observations", [])))
        np.savez_compressed(os.path.join(ctx.model, "nir_signal.npz"), schema_version=np.array(2),
                            led_signal=signal.astype(np.float32), noise=noise.astype(np.float32),
                            valid=counts["signal"] > 0, observations=counts["signal"],
                            encoded_delta=encoded.astype(np.float32), filter_delta=filt.astype(np.float32))
        with open(os.path.join(ctx.model, "nir_signal.npz"), "rb") as source:
            numeric_sha = hashlib.sha256(source.read()).hexdigest()
        info = {"schema_version": 2, "width": pw, "height": ph, "numeric_file": "nir_signal.npz",
                "numeric_sha256": numeric_sha,
                "calibration_sha256": hashlib.sha256(json.dumps(ctx.camera.c, sort_keys=True).encode()).hexdigest(),
                "quantitative_observations": quantitative, "legacy_observations": legacy,
                "qualitative_observations": legacy,
                "reflectivity_valid": False, "response_valid": bool(np.isfinite(signal).any()),
                "status": "controlled_linear_signal" if quantitative else "qualitative_only",
                "radiometric_signature": common_signature,
                "limitations": ["uncalibrated LED beam, source offset and camera shading",
                                "geometry correction is an approximate response proxy; LED power assumed stable",
                                "reflected illumination, not temperature or material identity"], "experiments": stats}
        with open(os.path.join(ctx.model, "nir_signal.json"), "w") as destination:
            json.dump(info, destination, indent=2)
        _overlay(encoded, os.path.join(ctx.model, "pano_led_light.jpg"), 16.)
        layers = [Layer("infrared.led", self.name, self.band, "panorama",
                        "LED brightness change (qualitative preview)",
                        "LEDs on minus time-interpolated off frames in the same night/filter state. Numeric measurements "
                        "require verified fixed controls and linearized response; legacy JPEG differences are display-only.",
                        None, None, {**info, "image": "pano_led_light.jpg", "overlay": True,
                                     "display_qualitative": True, "display_units": "encoded JPEG levels"})]
        if counts["filter"].any():
            _overlay(filt, os.path.join(ctx.model, "pano_nir_excess.jpg"), .2)
            layers.append(Layer("infrared.nir_excess", self.name, self.band, "panorama", "Filter response (qualitative)",
                                "Historical filter-out minus filter-in JPEG code values. Exposure, processing and spectrum "
                                "differ; this is not isolated near-IR reflectivity or a material classifier.",
                                None, None, {"image": "pano_nir_excess.jpg", "overlay": True, "qualitative": True}))
        return layers
