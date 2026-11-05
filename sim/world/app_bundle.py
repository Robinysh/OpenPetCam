"""Your room model, reduced to what the app's Room tab draws: the calibrated camera, the measured shape (the fused
surface: one range per direction), the floor outline, the walls and furniture boxes (for naming places), the latest mount-drift correction and four textures (day photo, night, where the model is unsure,
near-IR materials), resized for a phone.

The bundle is the inside of your home. It goes to YOUR server's models/ folder (deploy.sh copies it, install.sh
installs it, the app serves it to signed-in devices only); it is git-ignored and never belongs in the repository.
    python3 -m sim.world.app_bundle eval/world [models/room]"""
import json
import os
import sys
import time

TEX_W = 2048                 # equirectangular width for the phone (2048 x 1024)


def legends_for(model, surf, radio_info):
    """A colour legend for every view, sampled from the same colour functions that painted its texture."""
    import numpy as np
    from .blindfill import SOURCE_RGB
    from .colormaps import classes, div, gradient, radio, seq, swatches
    rng = dict(surf.get("texture_ranges") or {})
    if not rng:                                          # models built before texture_ranges: the ranges fuse.py uses
        z = np.load(os.path.join(model, "surfels.npz"))
        ec = z["echo_coverage"]; dw = z["dwell_min"]
        rng = {"uncertainty": [-2, 0], "nir_reflectance": [-1.5, 1.5], "nir_excess": [-1, 1], "echo_support": [0, 3],
               "echo_coverage": [0, max(2.0, float(np.nanmax(ec)) if np.isfinite(ec).any() else 2.0)],
               "dwell": [0, max(1.0, float(np.nanmax(np.log1p(dw))) if np.isfinite(dw).any() else 1.0)],
               "floor_residual": [-0.15, 0.15], "light_range": [0, 120]}
    lg = {
        "uncertainty": gradient(seq, f"{10 ** rng['uncertainty'][0]:.0%} error", f"{10 ** rng['uncertainty'][1]:.0%}", "relative range error"),
        "class": swatches([(classes(np.array([0.0]))[0], "floor"), (classes(np.array([0.5]))[0], "wall, other"), (classes(np.array([1.0]))[0], "object")], "surface class"),
        "normals": swatches([((1.0, 0.5, 0.5), "faces right"), ((0.5, 1.0, 0.5), "faces up"), ((0.5, 0.5, 1.0), "faces forward")], "surface direction (colour = x, y, z)"),
        "floor_residual": gradient(div, f"{rng['floor_residual'][0] * 100:+.0f} cm", f"{rng['floor_residual'][1] * 100:+.0f} cm", "learned depth vs sound floor"),
        "nir_excess": gradient(div, "less near-IR", "more near-IR", "near-infrared minus visible"),
        "nir_reflectance": gradient(div, f"x{np.exp(rng['nir_reflectance'][0]):.2f}", f"x{np.exp(rng['nir_reflectance'][1]):.1f}", "IR reflectance vs typical"),
        "light_range": gradient(seq, "steady", f"{rng['light_range'][1]:.0f} levels", "daylight change over the day"),
        "echo_support": gradient(seq, "0", f"{rng['echo_support'][1]:.0f}", "echoes agreeing with the shape"),
        "echo_coverage": gradient(seq, "0", f"{rng['echo_coverage'][1]:.0f} directions", "head directions that heard it"),
        "dwell": gradient(seq, "0 min", f"{np.expm1(rng['dwell'][1]):.0f} min", "time spent, all days"),
        "source": swatches([(np.array(SOURCE_RGB[1]) / 255, "seen by the camera"), (np.array(SOURCE_RGB[2]) / 255, "above: measured by sound"),
                            (np.array(SOURCE_RGB[3]) / 255, "below: floor, assumed")], "where each part comes from"),
    }
    if radio_info and radio_info.get("texture_range_dbm"):
        lo, hi = radio_info["texture_range_dbm"]
        lg["radio"] = gradient(radio, f"{lo:.0f} dBm", f"{hi:.0f} dBm", "WiFi link strength when the camera's antenna points there")
    return lg


def build(world, out):
    from PIL import Image
    model = os.path.join(world, "model")
    m = json.load(open(os.path.join(model, "manifest.json")))
    L = {x["id"]: x for x in m["layers"]}
    calib = json.load(open(os.path.join(world, "calib.json")))
    room = m["room"]
    fused = (L.get("fused.surface") or {}).get("data") or {}
    os.makedirs(out, exist_ok=True)
    tex = {}
    sf = ((L.get("fused.surfels") or {}).get("data") or {}).get("textures") or {}
    views = [("photo", "pano_visible_sweep.jpg"), ("night", "pano_infrared.jpg")] + sorted(sf.items())
    for key, name in views:                          # every colour view the room model has, for the app's Room sheet
        src = os.path.join(model, name)
        if not os.path.exists(src):
            continue
        im = Image.open(src).convert("RGB")
        photo = key in ("photo", "night")
        if photo:
            im = im.resize((TEX_W, TEX_W // 2), Image.LANCZOS)
        dst = f"room_{key}.jpg" if photo else f"room_{key}.png"
        im.save(os.path.join(out, dst), **({"quality": 85} if photo else {}))
        tex[key] = dst
    # the measured shape: one range per direction (the fused surface), millimetres as uint16 (0 = nothing measured)
    import numpy as np
    meta = fused
    rf = meta.get("filled_file") or meta.get("file")         # blind spots filled by sound where it holds (blindfill.py)
    if rf and os.path.exists(os.path.join(model, rf)):
        R = np.fromfile(os.path.join(model, rf), np.float32).reshape(meta["height"], meta["width"])
        mm = np.where(np.isfinite(R) & (R > 0.05) & (R < 60), np.round(R * 1000), 0).astype("<u2")
        mm.tofile(os.path.join(out, "room_range.u16"))
        tex["range"] = "room_range.u16"
        shape = [int(meta["width"]), int(meta["height"])]
    else:
        shape = None
    # the mount's latest drift correction (registration of recent snapshots to the panorama)
    off = (0.0, 0.0, 0.0)
    rp = os.path.join(world, "raw", "registration.json")
    if os.path.exists(rp):
        smooth = (json.load(open(rp)) or {}).get("smooth") or []
        if smooth:
            off = (float(smooth[-1][1]), float(smooth[-1][2]), 0.0)
    # re-mounts (sim/world/remount.py): dated corrections, so positions logged before a move keep the old one
    mh = os.path.join(world, "raw", "mount_history.json")
    offsets = None
    if os.path.exists(mh):
        hist = json.load(open(mh))
        offsets = [[h["since"] if h["since"] is not None else 0, *h["off"]] for h in hist]
        off = tuple(hist[-1]["off"])
    legends = legends_for(model, (L.get("fused.surfels") or {}).get("data") or {}, ((L.get("radio.rssi") or {}).get("data") or {}).get("by_direction"))
    outline = []
    for row in room["outline"]:                        # [azimuth deg, range m or None, label]: labels stay out
        outline.append([row[0], row[1]])
    doc = {"version": 1, "built": time.strftime("%Y-%m-%d %H:%M"),
           "calib": {k: calib[k] for k in ("focal_px", "k1", "k2", "deg_per_step_pan", "deg_per_step_tilt", "level_tilt", "roll_deg")},
           "floor_height_m": room["floor_height_m"], "ceiling_m": room["ceiling_m"], "overhead_m": room.get("overhead_m"),
           "outline": outline, "walls": [{"a": w["a"], "b": w["b"]} for w in room["walls"]],
           "objects": [{"label": o["label"], "center": o["center"], "size": o["size"], "yaw_deg": o["yaw_deg"],
                        "kind": o.get("kind", "solid")} for o in fused.get("objects", [])],
           "mount_offset_deg": off, "mount_offsets": offsets, "textures": tex, "range_shape": shape, "blind_fill": meta.get("blind_fill"),
           "radio": ((L.get("radio.rssi") or {}).get("data") or {}).get("by_direction"),
           "legends": legends}
    json.dump(doc, open(os.path.join(out, "room.json"), "w"))
    return doc


if __name__ == "__main__":
    a = sys.argv[1:]
    out = a[1] if len(a) > 1 else "models/room"
    if os.path.isdir(out) and not os.path.exists(os.path.join(out, "room.json")) and os.listdir(out):
        sys.exit(f"{out} exists and is not a room bundle; not writing into it")
    d = build(a[0], out)
    size = sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out))
    print(f"room bundle: {len(d['walls'])} walls, {len(d['objects'])} objects, textures {sorted(d['textures'])}, "
          f"{size / 1e6:.1f} MB in {out}")
