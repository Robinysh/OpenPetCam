#!/usr/bin/env python3
"""Build the room's world model from every sensor and write it for the viewer.

    python3 -m sim.world.build eval/world [--tape 0.86] [--collect]
        -> eval/world/model/{manifest.json, *.jpg, frames/, index.html}
    then: python3 -m http.server -d eval/world/model 8765   and open http://localhost:8765

Order matters only once: sound runs before the room is fused (the floor height comes from the floor echo), and the
presence sensor runs after it (it needs that floor)."""
import argparse
import json
import os
import shutil
import time

from . import sensors
from .room import fuse
from .sensors.base import Context

HERE = os.path.dirname(os.path.abspath(__file__))


def inputs_fingerprint(ctx):
    """sha256 of every input file the model depends on, plus the code revision: a model is reproducible only if its
    inputs are named exactly (cybersweat's calibration-identity rule)."""
    import hashlib
    import subprocess
    def h(p):
        d = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(1 << 20), b""):
                d.update(b)
        return d.hexdigest()[:16]
    out = {}
    for rel in ("calib.json", "picks.json", "raw/matches_lightglue.npz", "raw/geometry/da3.npz", "raw/registration.json",
                "rssi.log", "minute.json", "track.json", "obs.json"):
        p = ctx.path(rel)
        if os.path.exists(p):
            out[rel] = h(p)
    for root in ("raw/acoustic", "raw/ir"):
        d = ctx.path(root)
        if os.path.isdir(d):
            for dp, _dn, fn in os.walk(d):
                for f in sorted(fn):
                    out[os.path.relpath(os.path.join(dp, f), ctx.world)] = h(os.path.join(dp, f))
    try:
        rev = subprocess.run(["git", "-C", HERE, "describe", "--always", "--dirty"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        rev = "unknown"
    return {"files": out, "code": rev}


def build(world, tape=None, collect=False, acoustic_fit=False):
    if collect:
        from .sources import refresh
        refresh(world)
    ctx = Context(world)
    layers, report = [], []
    order = [s for s in sensors.ALL if not s.needs_room] + [s for s in sensors.ALL if s.needs_room]
    for cls in order:
        if cls.needs_room and ctx.room is None:
            echo = next(l for l in layers if l.id == "acoustic.echoes")
            ctx.room = fuse(ctx, echo, tape)
        s = cls(ctx)
        t = time.time()
        if collect:
            s.collect()
        got = s.layers()
        layers += got
        report.append(f"{cls.__name__:16s} {s.band.kind:9s} {len(got)} layers  {time.time() - t:5.1f} s")
        print(report[-1], flush=True)
    echo_l = next(l for l in layers if l.id == "acoustic.echoes")
    for it in echo_l.data["items"]:
        it["predicted"] = ctx.room.predict_echoes(ctx.camera.R(it["pan"], it["tilt"])[:, 2])
    fit_path = ctx.path("raw", "acoustic", "fit.json")
    fused_l = next((l for l in layers if l.id == "fused.surface"), None)
    if acoustic_fit and fused_l:
        import numpy as np
        from .acoustic_model import fit as acoustic_fit_run
        rngp = np.fromfile(os.path.join(ctx.model, fused_l.data["file"]), np.float32).reshape(fused_l.data["height"], fused_l.data["width"])
        res = acoustic_fit_run(ctx, echo_l.data["items"], rngp, fused_l.data["objects"], fused_l.data["width"], fused_l.data["height"], ctx.camera)
        json.dump(res, open(fit_path, "w"), default=float)
    if os.path.exists(fit_path):
        from .layers import Layer
        res = json.load(open(fit_path))
        passed = res["mean_r_held_out"] > res["null_rotated_r"] + 0.1
        layers.append(Layer("acoustic.fit", "acoustic.active", echo_l.band, "note", "Sound against the fused geometry",
                            f"Forward model (specular facets of the fused surface and boxes, head directivity, measurement "
                            f"kernel) fitted to the echo peaks past 0.3 m: held-out r = {res['mean_r_held_out']}, but the room "
                            f"turned 180 deg scores {res['null_rotated_r']}. " + ("Passes the null: parameters applied." if passed else
                            "Fails the null, so sound does not yet discriminate this geometry and its parameters are NOT applied; "
                            "sound sets the floor height and the two near surfaces only."), None, None,
                            {"z": {"held-out r": res["mean_r_held_out"], "rotated-room r": res["null_rotated_r"],
                                   "baseline r": res["mean_r_baseline"]}, "fit": res}))
    cam = ctx.camera.c
    manifest_inputs = inputs_fingerprint(ctx)
    manifest = {"built": time.time(), "inputs": manifest_inputs, "camera": {k: cam[k] for k in ("focal_px", "k1", "k2", "deg_per_step_pan",
                                                                    "deg_per_step_tilt", "level_tilt", "roll_deg", "median_px")},
                "room": ctx.room.to_json(), "layers": [l.to_json() for l in layers],
                "bands": sorted({l.band.kind for l in layers})}
    # deep integration: every modality onto one record per viewing direction (fuse.py)
    from .fuse import build as fuse_surfels
    from .layers import Band, Layer
    if any(l.id == "fused.surface" for l in layers):
        summ, _ = fuse_surfels(ctx, json.loads(json.dumps(manifest, default=float)))
        manifest["layers"].append(Layer("fused.surfels", "fused", Band("visible", 400e-9, 1000e-9, "all modalities per direction"),
                                        "surfels", "Every modality, per patch of room",
                                        "One record per viewing direction carrying shape (fused depth, normals, class, object), "
                                        "visible colour and its 24-hour light curve, near-IR material and reflectance, sound "
                                        "coverage and support, the dog's dwell and free floor, and the floor check. Empty where "
                                        "a modality has nothing to say.", None, None, summ).to_json())
    json.dump(manifest, open(os.path.join(ctx.model, "manifest.json"), "w"), default=float)
    for f in os.listdir(os.path.join(HERE, "viewer")):
        shutil.copyfile(os.path.join(HERE, "viewer", f), os.path.join(ctx.model, f))
    print(f"wrote {ctx.model}: {len(layers)} layers in bands {manifest['bands']}")
    return manifest


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("world")
    ap.add_argument("--tape", type=float, help="lens height above the floor from a tape measure, metres")
    ap.add_argument("--collect", action="store_true", help="also collect fresh raw data where a sensor can")
    ap.add_argument("--acoustic-fit", action="store_true", help="refit sound against the fused geometry (~3 min)")
    a = ap.parse_args()
    build(a.world, a.tape, a.collect, a.acoustic_fit)
