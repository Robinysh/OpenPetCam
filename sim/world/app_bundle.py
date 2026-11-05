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
    if meta.get("file") and os.path.exists(os.path.join(model, meta["file"])):
        R = np.fromfile(os.path.join(model, meta["file"]), np.float32).reshape(meta["height"], meta["width"])
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
    outline = []
    for row in room["outline"]:                        # [azimuth deg, range m or None, label]: labels stay out
        outline.append([row[0], row[1]])
    doc = {"version": 1, "built": time.strftime("%Y-%m-%d %H:%M"),
           "calib": {k: calib[k] for k in ("focal_px", "k1", "k2", "deg_per_step_pan", "deg_per_step_tilt", "level_tilt", "roll_deg")},
           "floor_height_m": room["floor_height_m"], "ceiling_m": room["ceiling_m"], "overhead_m": room.get("overhead_m"),
           "outline": outline, "walls": [{"a": w["a"], "b": w["b"]} for w in room["walls"]],
           "objects": [{"label": o["label"], "center": o["center"], "size": o["size"], "yaw_deg": o["yaw_deg"],
                        "kind": o.get("kind", "solid")} for o in fused.get("objects", [])],
           "mount_offset_deg": off, "textures": tex, "range_shape": shape}
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
