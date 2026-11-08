#!/usr/bin/env python3
"""Render the current qualitative IR layers through the real Room renderer.

Run tools/room_dev.py first, then pass a fresh private --output directory. Review
the movies before copying them into site/media/blog/film. Only the measured IR
textures and range grid are loaded: no photos, sightings, names or audio. This
does not rebuild or recut the original film.
"""
import argparse
import hashlib
import html
import json
import math
from pathlib import Path
import subprocess
from urllib.parse import urlparse
from urllib.request import urlopen

from playwright.sync_api import sync_playwright


SCENES = (
    ("room-led-difference", "nir_led_difference", "LED brightness change"),
    ("room-filter-response", "nir_excess", "Filter response"),
)
CSS = """
:root { --bg:#10141e; --accent:#7aa7ff; --ink:#e9edf5; }
* { box-sizing:border-box; }
html,body { margin:0; width:720px; height:640px; overflow:hidden;
  background:var(--bg); color:var(--ink); font-family:Arial,sans-serif; }
.heading { position:absolute; left:32px; top:25px; z-index:2; }
.eyebrow { color:#9ba9bc; font-size:12px; letter-spacing:1.8px; }
h1 { margin:9px 0 0; font-size:29px; font-weight:500; }
#room > :not(.room-stage) { display:none!important; }
.room-stage { position:absolute; inset:99px 0 100px; }
.room-stage canvas { display:block; width:100%; height:100%; }
footer { position:absolute; left:32px; right:32px; bottom:22px; }
.key { display:flex; align-items:center; gap:12px; font-size:13px; color:#c1cbd9; }
.bar { flex:1; height:7px; border-radius:4px; }
.note { margin-top:16px; font-size:13px; color:#9ba9bc; }
"""


def sha(data):
    return hashlib.sha256(data).hexdigest()


def get(url):
    with urlopen(url, timeout=15) as response:
        return response.read()


def frame_html(title, legend):
    gradient = ",".join(legend["stops"])
    return f"""<!doctype html><meta charset="utf-8"><style>{CSS}</style>
<div class="heading"><div class="eyebrow">REFLECTED INFRARED · QUALITATIVE</div>
<h1>{html.escape(title)}</h1></div><div id="room"></div>
<footer><div class="key"><span>{html.escape(legend['lo'])}</span>
<div class="bar" style="background:linear-gradient(90deg,{gradient})"></div>
<span>{html.escape(legend['hi'])}</span></div>
<div class="note">One stationary view · unmeasured surfaces stay grey</div></footer>
<script type="module">
import {{ mountRoom }} from '/room.js';
window.__room = await mountRoom(document.querySelector('#room'), {{name:'your pet'}});
window.__room.show();
</script>"""


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8790/")
    parser.add_argument("--seconds", type=float, default=6)
    parser.add_argument("--preview", action="store_true", help="Only first/middle/last PNGs")
    parser.add_argument("--azimuth", type=float, default=-25)
    parser.add_argument("--elevation", type=float, default=38)
    parser.add_argument("--distance", type=float, default=.75)
    parser.add_argument("--orbit", type=float, default=28)
    args = parser.parse_args()
    origin = urlparse(args.url)
    if origin.scheme != "http" or origin.hostname not in ("127.0.0.1", "localhost"):
        raise SystemExit("Use the local Room preview, never a remote camera endpoint.")
    if not 1 <= args.seconds <= 15:
        raise SystemExit("Choose a clip length from 1 to 15 seconds.")
    base = f"{origin.scheme}://{origin.netloc}"
    info = json.loads(get(base + "/api/room"))
    quality = info.get("infrared", {})
    if not info.get("available") or quality.get("qualitative_observations") != 1:
        raise SystemExit("This single-view sequence requires exactly one qualitative IR observation.")
    if quality.get("quantitative_observations") or quality.get("reflectivity_valid"):
        raise SystemExit("This exporter describes qualitative observations only.")
    for _, key, _ in SCENES:
        if key not in info.get("textures", {}) or key not in info.get("legends", {}):
            raise SystemExit(f"Current bundle is missing {key} or its legend.")
    args.output.mkdir(parents=True, exist_ok=False)
    fps, frames = 24, round(24 * args.seconds)
    receipt = {"source": "Current Room renderer and qualitative IR products; no original-film footage",
               "display": "Measured colour patch over untextured room geometry; unseen surfaces grey",
               "animation": "A camera orbit around fixed measurements, not acquisition time",
               "renderer_sha256": sha(get(base + "/room.js")),
               "source_range_sha256": sha(get(base + "/api/room/tex?f=" + info["textures"]["range"])),
               "qualitative_observations": quality["qualitative_observations"],
               "quantitative_observations": 0, "reflectivity_valid": False,
               "width": 720, "height": 640, "fps": fps, "scenes": []}
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(args=["--use-gl=angle", "--enable-webgl", "--ignore-gpu-blocklist"])
        for name, key, title in SCENES:
            page = browser.new_page(viewport={"width": 720, "height": 640}, device_scale_factor=1, color_scheme="dark")
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            selected = json.loads(json.dumps(info))
            selected["textures"] = {k: v for k, v in info["textures"].items() if k in ("range", key)}
            selected["objects"], selected["radio"] = [], None
            legend = dict(info["legends"][key])
            if key == "nir_excess":
                legend.update(lo="lower response", hi="higher response")
            if key == "nir_led_difference":
                assert legend["lo"] == "-16 DN" and legend["hi"] == "+16 DN"
            page.route("**/api/room", lambda route: route.fulfill(json=selected))
            page.route("**/api/room/positions**", lambda route: route.fulfill(json={"rows": [], "poses": []}))
            page.route("**/blog-ir-render", lambda route: route.fulfill(content_type="text/html", body=frame_html(title, legend)))
            page.add_init_script("""localStorage.clear();
                localStorage.setItem('room.look','photo');
                localStorage.setItem('room.over','none');
                localStorage.setItem('room.panel','0');""")
            page.goto(base + "/blog-ir-render")
            page.wait_for_function("window.__room?.cam?.userData.sized === true")
            texture = selected["textures"][key]
            with page.expect_response(lambda r: "/api/room/tex?" in r.url and texture in r.url and r.status == 200):
                page.locator(f'[data-over="{key}"]').evaluate("button => button.click()")
            page.wait_for_function("""() => window.__room.world.children.some(o =>
                o.material?.uniforms?.over?.value?.image?.complete)""")
            preset = page.evaluate("""() => { const r=window.__room, c=r.controls;
                c.enableDamping=false; c.update();
                return [...c.target.toArray(),r.cam.position.distanceTo(c.target)]; }""")
            target, distance = preset[:3], preset[3] * args.distance
            folder = args.output / name
            folder.mkdir()
            indices = (0, frames // 2, frames - 1) if args.preview else range(frames)
            for k in indices:
                fraction = k / (frames - 1)
                ease = .5 - .5 * math.cos(math.pi * fraction)
                az = math.radians(args.azimuth + args.orbit * (ease - .5))
                el = math.radians(args.elevation)
                position = [target[0] + distance * math.cos(el) * math.sin(az),
                            target[1] + distance * math.sin(el),
                            target[2] - distance * math.cos(el) * math.cos(az)]
                page.evaluate("""async ([position,target]) => { const r=window.__room;
                    r.cam.position.fromArray(position); r.controls.target.fromArray(target);
                    r.controls.update(); r.redraw();
                    await new Promise(resolve=>requestAnimationFrame(()=>requestAnimationFrame(resolve)));
                }""", [position, target])
                page.screenshot(path=str(folder / f"f{k:04d}.png"), animations="disabled")
            assert not errors, errors
            scene = {"name": name, "field": key, "legend": legend,
                     "source_texture_sha256": sha(get(base + "/api/room/tex?f=" + texture)),
                     "frames": frames, "duration_seconds": frames / fps,
                     "camera": {"azimuth": args.azimuth, "elevation": args.elevation,
                                "distance_factor": args.distance, "orbit": args.orbit}}
            if not args.preview:
                movie, poster = args.output / f"{name}.mp4", args.output / f"{name}.jpg"
                subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-framerate", str(fps),
                                "-i", str(folder / "f%04d.png"), "-map", "0:v:0", "-an", "-c:v", "libx264",
                                "-crf", "22", "-preset", "medium", "-threads", "2", "-pix_fmt", "yuv420p",
                                "-map_metadata", "-1", "-movflags", "+faststart", str(movie)], check=True)
                subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str((frames//2)/fps),
                                "-i", str(movie), "-map", "0:v:0", "-frames:v", "1", "-q:v", "2",
                                "-update", "1", "-map_metadata", "-1", str(poster)], check=True)
                scene.update(video_sha256=sha(movie.read_bytes()), poster_sha256=sha(poster.read_bytes()),
                             video_bytes=movie.stat().st_size, poster_bytes=poster.stat().st_size)
            receipt["scenes"].append(scene)
            page.close()
            print(name, "preview" if args.preview else f"{frames} frames", flush=True)
        browser.close()
    (args.output / "ir-provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
