#!/usr/bin/env python3
"""Export a cropped, qualitative off/on/off subtraction example.

    python3 tools/blog/build_ir_example.py --capture PRIVATE_CAPTURE \
        --output NEW_PRIVATE_OUTPUT --crop X0 Y0 X1 Y1 --sample X Y

Coordinates use the original decoded images; the crop's right/bottom edges are
exclusive. Public metadata reports only crop-relative sample coordinates and
elapsed times. Review the five PNGs before publishing selected crops. Original
JPEGs are never changed or exported. NumPy and Pillow are the only dependencies.
"""
import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


FRAMES = (
    ("off_before", "B_passive_1.jpg", "off", "ir-pet-off-before.png"),
    ("on", "C_active_1.jpg", "on", "ir-pet-on.png"),
    ("off_after", "B2_passive_1.jpg", "off", "ir-pet-off-after.png"),
)
SPAN = 16.0


def digest(data):
    return hashlib.sha256(data).hexdigest()


def div(values):
    """The analytical Room palette from sim/world/colormaps.py, standalone.

    Input is scaled to 0..1; .5 is zero/grey, low values blue and high magenta.
    Both diagnostic maps use this exact same fixed scale, without normalization.
    """
    v = np.asarray(values, float)
    return np.stack([
        np.where(v > .5, .43 + 1.14 * (v - .5), .43 - .6 * (.5 - v)),
        np.where(v > .5, .43 - .5 * (v - .5), .43 + .2 * (.5 - v)),
        np.where(v > .5, .43 + .9 * (v - .5), .43 + .9 * (.5 - v)),
    ], axis=-1).clip(0, 1)


def colourize(values):
    return (div(np.clip((values + SPAN) / (2 * SPAN), 0, 1)) * 255).astype(np.uint8)


def statistics(values):
    return {
        "minimum": float(np.min(values)), "maximum": float(np.max(values)),
        "mean": float(np.mean(values)), "median": float(np.median(values)),
        "median_absolute": float(np.median(np.abs(values))),
        "p05": float(np.quantile(values, .05)),
        "p95": float(np.quantile(values, .95)),
        "clipped_display_fraction": float(np.mean(np.abs(values) > SPAN)),
    }


def build_example(capture, output, crop, sample):
    capture, output = Path(capture), Path(output)
    log_bytes = (capture / "log.json").read_bytes()
    log = json.loads(log_bytes)
    x0, y0, x1, y1 = crop
    sx, sy = sample
    if not (0 <= x0 < x1 and 0 <= y0 < y1 and x0 <= sx < x1 and y0 <= sy < y1):
        raise ValueError("Use an ordered, nonempty crop containing the sample pixel.")
    images, times, source_hashes = {}, [], {}
    dimensions = None
    for phase, filename, led, _ in FRAMES:
        records = [f for f in log.get("frames", []) if f.get("file") == filename]
        if len(records) != 1:
            raise ValueError(f"Exactly one log record is required for {filename}.")
        record = records[0]
        if record.get("led") != led or record.get("ircut") != "off":
            raise ValueError("The recorded triplet must have off/on/off LEDs and the filter out throughout.")
        timestamp = record.get("t")
        if isinstance(timestamp, bool) or not isinstance(timestamp, (int, float)) or not math.isfinite(timestamp):
            raise ValueError("The triplet requires finite numeric capture timestamps.")
        times.append(timestamp)
        raw = (capture / filename).read_bytes()
        source_hashes[phase] = digest(raw)
        with Image.open(io.BytesIO(raw)) as source:
            if dimensions is None:
                dimensions = source.size
            if source.size != dimensions or x1 > source.width or y1 > source.height:
                raise ValueError("Images must have matching dimensions and contain the crop.")
            # Convert/decode once, then copy only the requested pixels. No
            # brightness, colour, registration or contrast adjustment is applied.
            images[phase] = np.asarray(source.convert("RGB"))[y0:y1, x0:x1].copy()
    if not times[0] < times[1] < times[2]:
        raise ValueError("The on observation must lie strictly between the off observations.")
    weight = (times[1] - times[0]) / (times[2] - times[0])
    grey = {key: image.astype(float).mean(axis=-1) for key, image in images.items()}
    ambient = (1 - weight) * grey["off_before"] + weight * grey["off_after"]
    residual = grey["on"] - ambient
    drift = grey["off_after"] - grey["off_before"]
    px, py = sx - x0, sy - y0
    sample_grey = {key: float(image[py, px]) for key, image in grey.items()}
    metadata = {
        "schema_version": 1,
        "description": "A cropped recorded LED off/on/off sequence and its signed JPEG-level diagnostics",
        "quality": "qualitative_only", "quantitative": False,
        "units": "encoded JPEG pixel levels, averaged across decoded RGB channels",
        "source_sha256": source_hashes,
        "source_log_sha256": digest(log_bytes),
        "source_dimensions": list(dimensions), "crop_dimensions": [x1 - x0, y1 - y0],
        "elapsed_seconds": {key: float(t - times[0]) for (key, _, _, _), t in zip(FRAMES, times)},
        "interpolation": {"off_before_weight": 1 - weight, "off_after_weight": weight},
        "formula": {
            "grey": "(R + G + B) / 3 in encoded JPEG levels",
            "estimated_ambient": "off_before_weight * off_before + off_after_weight * off_after",
            "residual": "on - estimated_ambient",
            "drift": "off_after - off_before",
        },
        "sample": {
            "pixel_xy": [px, py], "coordinate_system": "zero-based crop pixels",
            "rgb": {key: image[py, px].tolist() for key, image in images.items()},
            "grey": sample_grey, "estimated_ambient": float(ambient[py, px]),
            "residual": float(residual[py, px]), "drift": float(drift[py, px]),
        },
        "crop_statistics": {
            "pixels": int(residual.size), "residual": statistics(residual), "drift": statistics(drift),
            "any_input_channel_at_least_254_fraction": float(np.mean(
                np.maximum.reduce(list(images.values())).max(axis=-1) >= 254)),
        },
        "legend": {
            "minimum": -SPAN, "maximum": SPAN, "zero": 0,
            "low_label": "-16 or less", "high_label": "+16 or more",
            "units": "JPEG pixel levels", "clipped_at_endpoints": True,
            "colours": ["#%02x%02x%02x" % tuple(row) for row in colourize(np.linspace(-SPAN, SPAN, 5))],
            "interpretation": "Blue is negative, grey zero, magenta positive; identical scale for residual and drift",
        },
        "limitations": [
            "Logged LED/filter states do not establish verified hardware controls or a calibrated camera response",
            "Subject movement, ambient drift, clipping, sensor noise and automatic image processing can affect the difference",
            "The arithmetic uses encoded JPEG values, without linearization or exposure normalization",
            "Residual colours are qualitative diagnostics, not infrared reflectivity or temperature",
        ],
    }
    output.mkdir(parents=True, exist_ok=False)
    for phase, _, _, filename in FRAMES:
        Image.fromarray(images[phase]).save(output / filename, format="PNG")
    Image.fromarray(colourize(residual)).save(output / "ir-pet-residual.png", format="PNG")
    Image.fromarray(colourize(drift)).save(output / "ir-pet-drift.png", format="PNG")
    metadata["output_sha256"] = {p.name: digest(p.read_bytes()) for p in sorted(output.glob("*.png"))}
    (output / "ir-pet-example.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="A fresh private directory; never overwritten")
    parser.add_argument("--crop", type=int, nargs=4, metavar=("X0", "Y0", "X1", "Y1"), required=True)
    parser.add_argument("--sample", type=int, nargs=2, metavar=("X", "Y"), required=True)
    args = parser.parse_args()
    result = build_example(args.capture, args.output, args.crop, args.sample)
    print(json.dumps({"files": list(result["output_sha256"]), "sample": result["sample"]}, indent=2))


if __name__ == "__main__":
    main()
