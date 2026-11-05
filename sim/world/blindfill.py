"""Fill the camera's blind spots with what sound measured, and only where it has been shown to hold.

The sweep sees from about -60 to +40 degrees. Two blind regions are surfaces whose distance the camera's own sound
measures (AUDIT procedure ACO-7):
  above   the surface the camera hangs under. Every recording's nearest strong echo is its distance; a flat surface
          above the lens returns the same delay whatever the head direction (checked: the spread of that echo over
          head directions), and vision, near the top of its sweep, sees the same surface at that height around a
          hole about 0.2 m wide above the lens (checked: vision height vs the echo). Filled as a plane at the echo's
          height in every direction above +40 degrees that vision left empty, and below that only out to where vision
          itself saw the surface extend (per azimuth).
  below   straight down, where the sweep does not reach: the floor at the echo's height (the floor echo, which the
          pet's positions on the floor confirm). Filled as floor, flagged 'assumed': sound gives the floor's height,
          not that nothing stands under the camera.
Everything else sound cannot place: the strongest or first echo beyond 0.3 m does not predict vision's range along the
head axis better than shuffled directions, and the near cabinet echo is not direction-specific. Those stay empty.
Returns the range panorama with fills, and a source map: 0 none, 1 vision, 2 sound (above), 3 sound (floor, assumed).
"""
import numpy as np

ABOVE_EL, BELOW_EL = 40.0, -55.0       # degrees: only beyond what the sweep sees


def fill(R, overhead_m, floor_m):
    H, W = R.shape
    el = (90 - (np.arange(H) + 0.5) / H * 180)[:, None] * np.ones((1, W))
    seen = np.isfinite(R) & (R > 0)
    out, src = np.where(seen, R, np.nan).astype(np.float32), np.where(seen, 1, 0).astype(np.uint8)
    s = np.sin(np.radians(el))
    if overhead_m and overhead_m > 0:
        # how far out vision itself saw the surface above, per azimuth column: unseen directions between the sweep's
        # top and +40 deg are filled only where the plane would be hit within that reach (never beyond what vision
        # confirms the surface extends to)
        r = np.where(seen, R, np.nan)
        y, hz = r * s, r * np.cos(np.radians(el))
        under = seen & (el > 8) & (el < 45) & (y > 0.05) & (y < 0.4) & (r < 1.5)
        reach = np.where(under, hz, 0).max(0)
        k = 9                                                        # smooth over ~4 deg of azimuth (wraps around)
        reach = np.max([np.roll(reach, d) for d in range(-k, k + 1)], 0)
        hit = overhead_m / np.tan(np.radians(np.maximum(el, 1)))     # horizontal distance where the ray meets the plane
        m = ~seen & ((el > ABOVE_EL) | ((el > 5) & (hit <= reach[None, :])))
        out[m], src[m] = overhead_m / s[m], 2
    if floor_m and floor_m > 0:
        m = ~seen & (el < BELOW_EL)
        out[m], src[m] = floor_m / -s[m], 3
    return out, src


def checks(items, R):
    """The two checks the fills rest on, from the raw-tap recordings and the vision range (ACO-7)."""
    H, W = R.shape
    near = [min([e for e in i["echoes"] if 0.05 < e[0] < 0.3], key=lambda e: e[0], default=None) for i in items]
    r1 = np.array([n[0] for n in near if n])
    hs = []
    for j in range(H):
        e = 90 - (j + 0.5) / H * 180
        if not 8 < e < 45:
            continue
        r = R[j]
        ok = np.isfinite(r) & (r > 0) & (r < 1.5)
        y = r[ok] * np.sin(np.radians(e))
        hs.extend(y[(y > 0.05) & (y < 0.4)])
    hs = np.array(hs)
    return {"echo_recordings": len(items), "echo_with_near_surface": int(len(r1)),
            "echo_height_m": [round(float(np.median(r1)), 3), [round(float(x), 3) for x in np.percentile(r1, [25, 75])]] if len(r1) else None,
            "vision_above_lens_directions": int(len(hs)),
            "vision_height_m": [round(float(np.median(hs)), 3), [round(float(x), 3) for x in np.percentile(hs, [25, 75])]] if len(hs) else None}


SOURCE_RGB = {0: (0, 0, 0), 1: (70, 80, 92), 2: (82, 214, 178), 3: (150, 120, 220)}   # none / vision / sound / assumed floor


def source_image(src):
    rgb = np.zeros(src.shape + (3,), np.uint8)
    for k, c in SOURCE_RGB.items():
        rgb[src == k] = c
    return rgb


def apply(world):
    """Write model/range_pano_filled.f32 and model/surfel_source.png, and record them (and the checks) in the
    manifest's fused-surface layer, so the viewer and the app bundle draw the same filled surface."""
    import json
    import os
    from PIL import Image
    model = os.path.join(world, "model")
    mp = os.path.join(model, "manifest.json")
    m = json.load(open(mp))
    L = {x["id"]: x for x in m["layers"]}
    fs = L["fused.surface"]["data"]
    R = np.fromfile(os.path.join(model, fs["file"]), np.float32).reshape(fs["height"], fs["width"])
    out, src = fill(R, m["room"].get("overhead_m"), m["room"]["floor_height_m"])
    out.tofile(os.path.join(model, "range_pano_filled.f32"))
    Image.fromarray(source_image(src)).save(os.path.join(model, "surfel_source.png"))
    items = [i for i in L["acoustic.echoes"]["data"]["items"] if i.get("raw_tap")]
    fs["filled_file"] = "range_pano_filled.f32"
    fs["blind_fill"] = {"share": {"vision": round(float((src == 1).mean()), 3), "sound_above": round(float((src == 2).mean()), 3),
                                  "floor_assumed": round(float((src == 3).mean()), 3)}, "checks": checks(items, R)}
    if "fused.surfels" in L:
        L["fused.surfels"]["data"].setdefault("textures", {})["source"] = "surfel_source.png"
    json.dump(m, open(mp, "w"))
    return fs["blind_fill"]


if __name__ == "__main__":
    import sys
    print(apply(sys.argv[1]))
