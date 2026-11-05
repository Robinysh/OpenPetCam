"""Your room model, reduced to what the app's Room tab draws: the calibrated camera, the floor outline, the walls, the
furniture boxes, the latest mount-drift correction and four textures (day photo, night, where the model is unsure,
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
    for key, name in (("photo", "pano_visible_sweep.jpg"), ("night", "pano_infrared.jpg"),
                      ("unsure", "surfel_uncertainty.png"), ("materials", "pano_nir_excess.jpg")):
        src = os.path.join(model, name)
        if not os.path.exists(src):
            continue
        im = Image.open(src)
        im = im.convert("RGB").resize((TEX_W, TEX_W // 2), Image.LANCZOS if key != "unsure" else Image.NEAREST)
        dst = f"room_{key}.png" if key == "unsure" else f"room_{key}.jpg"
        im.save(os.path.join(out, dst), **({} if dst.endswith(".png") else {"quality": 85}))
        tex[key] = dst
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
           "mount_offset_deg": off, "textures": tex}
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
