#!/usr/bin/env python3
"""Claude reads the camera's own panorama (sim/sweep.py) and maps the scene the simulator needs: where the pet leaves
the camera's reach (exits), furniture she can lie behind and be hidden (occluders), and places she rests. Pixel boxes
are converted to motor coordinates with the panorama's own geometry and written to sim/params.json "scene" under
generic names (exit 1, occluder 2, ...): params.json is in the public repository, so Claude's descriptions of your
rooms stay in your twin folder (scene_claude_raw.json, git-ignored).

    python3 sim/scene_from_claude.py [eval/twin/day]

The picture goes to Claude (claude -p, the owner's subscription), as the labelling tools already do. Nothing else leaves
the machine. The scene is checked before use: every region must lie inside the panorama, and the simulator must
still pass gate 1 with it.
"""
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
from room import load_params  # noqa: E402

PROMPT = """Read the image file {path} (use your Read tool). It is a panorama stitched from a fixed pan/tilt pet camera in
an apartment: {w} x {h} pixels, the full range the camera can turn to. A dog lives here. I am building a simulator of
what this camera can see, so I need the scene mapped. Return ONLY a JSON object, no prose, with pixel boxes [x0, y0, x1, y1]
in this image's coordinates:
{{"exits": [{{"name": "...", "box": [...]}}],          // doorways/corridor ends where the dog can leave the camera's view of the room
  "occluders": [{{"name": "...", "box": [...], "hides": "what is behind/under it"}}],  // furniture the dog could be hidden behind or under (sofa, table, stools, island)
  "rest_places": [{{"name": "...", "box": [...]}}],     // where a dog would lie: dog bed, rug, sofa, near bowls
  "blank": [{{"name": "...", "box": [...]}}]}}          // views with nothing to see (bare wall, ceiling) that a search can skip
Be precise; leave a category empty rather than guess. Ignore stitching ghosts at frame seams."""


def ask_claude(path, w, h):
    cmd = ["claude", "-p", PROMPT.format(path=path, w=w, h=h), "--output-format", "text",
           "--allowedTools", "Read"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=300,
                       env=dict(os.environ, CLAUDE_CODE_HOOKS_ENABLED="false"))
    if r.returncode != 0:
        raise RuntimeError(f"claude -p failed: {r.stderr[-300:]}")
    m = re.search(r"\{.*\}", r.stdout, re.S)
    if not m:
        raise RuntimeError(f"no JSON in Claude's answer: {r.stdout[:300]}")
    return json.loads(m.group(0))


def to_motor(box, meta, w, h):
    """Panorama pixels -> (pan, tilt) ranges. sweep.mosaic places a frame at x0 = (3700 - pan) * sx, y0 = (1000 - tilt) * sy,
    the frame's top-left; a pixel at x therefore views pan = 3700 - x / sx + pan_frame / 2 (the frame centre)."""
    fw, fh = meta["frame_px"]
    scale = w / ((3700 + meta["pan_frame"]) * fw / meta["pan_frame"])     # the mosaic's scale, recovered from its width
    sx, sy = fw / meta["pan_frame"] * scale, fh / meta["tilt_frame"] * scale
    x0, y0, x1, y1 = box
    pan = sorted((3700 - x0 / sx + meta["pan_frame"] / 2, 3700 - x1 / sx + meta["pan_frame"] / 2))
    tilt = sorted((1000 - y0 / sy + meta["tilt_frame"] / 2, 1000 - y1 / sy + meta["tilt_frame"] / 2))
    return [round(max(0, pan[0])), round(min(3700, pan[1])), round(max(0, tilt[0])), round(min(1000, tilt[1]))]


def main(twin=os.path.join(ROOT, "eval", "twin", "day")):
    from PIL import Image
    pano = os.path.join(twin, "panorama.jpg")
    meta = json.load(open(os.path.join(twin, "poses.json")))
    w, h = Image.open(pano).size
    raw = ask_claude(os.path.abspath(pano), w, h)
    json.dump(raw, open(os.path.join(twin, "scene_claude_raw.json"), "w"), indent=1)
    scene, bad = {}, []
    for kind, word in (("exits", "exit"), ("occluders", "occluder"), ("rest_places", "rest place"), ("blank", "blank view")):
        scene[kind] = []
        for r in raw.get(kind) or []:
            b = r.get("box") or []
            if len(b) != 4 or not (0 <= b[0] < b[2] <= w and 0 <= b[1] < b[3] <= h):
                bad.append((kind, r))
                continue
            pan0, pan1, tilt0, tilt1 = to_motor(b, meta, w, h)
            scene[kind].append({"name": f"{word} {len(scene[kind]) + 1}", "pan": [pan0, pan1], "tilt": [tilt0, tilt1],
                                **({"hides": r["hides"]} if r.get("hides") else {})})
    path = os.path.join(HERE, "params.json")
    P = load_params(path)
    P["scene"] = dict(scene, source="claude -p on the camera's own panorama (sim/scene_from_claude.py)",
                      rejected=len(bad))
    json.dump(P, open(path, "w"), indent=1)
    print(json.dumps(scene, indent=1))
    print(f"{sum(len(v) for v in scene.values())} regions kept, {len(bad)} rejected (outside the panorama)")
    return scene


if __name__ == "__main__":
    main(*sys.argv[1:])
