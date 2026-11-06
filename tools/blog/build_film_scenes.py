#!/usr/bin/env python3
"""Reframe the original film's caption-free scenes for the article.

Pass the original caption-free picture master with --source. Its hash is pinned
so this cannot accidentally publish a different recording or a private room pack.
Only the frames used in the published film are included. No source measurements,
room geometry, timestamps or audio are exported.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tempfile


MASTER_SHA256 = "14b419be6b33ac2fda1089664a54f0fd03bba00cae2dbb8ba897eafd9b3a62f2"
PUBLIC_SHA256 = "1a3ca8945d7bce5f127ca61093b440cb2d9b8fdba151b93bf262195cd88dc5a6"
ROOT = Path(__file__).resolve().parents[2]
BG = "0x0e121a"
SCENES = [
    ("speaker-sweep", "c01", 0, 191, "portrait"),
    ("floor-return", "c02", 191, 400, "portrait"),
    ("photo-depth", "d01", 400, 660, "depth-pair"),
    ("depth-panorama", "d02", 660, 752, "panorama"),
    ("room-reveal", "p06", 752, 864, "room"),
    ("nir-contrast", "p17", 864, 991, "room"),
    ("infrared-response", "p18", 991, 1077, "room"),
    ("day-sightings", "p20", 1077, 1153, "room"),
    ("week-observations", "p21", 1153, 1229, "room"),
    ("recorded-search", "p16", 1280, 1355, "room"),
]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def probe(path):
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_streams", "-of", "json", str(path)
    ]))["streams"]


def graph(start, end, layout):
    head = f"[0:v]trim=start_frame={start}:end_frame={end},setpts=PTS-STARTPTS"
    if layout == "depth-pair":
        # Preserve the original photo/depth wipe and its legend, closing the
        # empty band reserved for narration captions in the vertical film.
        return (head + ",split[a][b];[a]crop=1080:1120:0:0[top];"
                "[b]crop=1080:140:0:1540[key];"
                "[top][key]vstack,scale=720:840,setsar=1[v]"), (720, 840)
    if layout == "panorama":
        return (head + f",crop=1080:950:0:0,scale=720:634,"
                f"pad=720:840:0:0:color={BG},setsar=1[v]"), (720, 840)
    if layout == "portrait":
        return head + ",scale=540:960,setsar=1[v]", (540, 960)
    return head + ",crop=1080:960:0:280,scale=720:640,setsar=1[v]", (720, 640)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "site/media/blog/film")
    args = parser.parse_args()
    if digest(args.source) != MASTER_SHA256:
        raise SystemExit("Source is not the approved caption-free master; refusing to export.")
    source = probe(args.source)
    assert len(source) == 1 and source[0]["codec_type"] == "video"
    assert (source[0]["width"], source[0]["height"]) == (1080, 1920)
    assert source[0]["r_frame_rate"] == "24/1" and int(source[0]["nb_frames"]) == 1447
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = {
        "source": "Original caption-free picture master",
        "source_sha256": MASTER_SHA256,
        "published_film": "https://crimeacs.github.io/OpenPetCam/media/openpetcam-4d-room.mp4",
        "published_film_sha256": PUBLIC_SHA256,
        "fps": 24,
        "frame_ranges": "Zero-based, start inclusive, end exclusive; original playback speed.",
        "time_scene": "Recorded observation heat, last-sighting marker and camera pose over a static photo texture. Clip progress is not a civil clock or measured daylight replay.",
        "scenes": [],
    }
    with tempfile.TemporaryDirectory(prefix="room-film-") as scratch:
        scratch = Path(scratch)
        for name, original, start, end, layout in SCENES:
            filters, size = graph(start, end, layout)
            movie, poster = scratch / f"{name}.mp4", scratch / f"{name}.jpg"
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-i", str(args.source),
                "-filter_complex", filters, "-map", "[v]", "-an", "-c:v", "libx264",
                "-crf", "23", "-preset", "medium", "-threads", "2", "-pix_fmt", "yuv420p",
                "-movflags", "+faststart", "-map_metadata", "-1", str(movie)
            ], check=True)
            stream = probe(movie)
            assert len(stream) == 1 and stream[0]["codec_type"] == "video"
            stream = stream[0]
            assert stream["codec_name"] == "h264" and stream["pix_fmt"] == "yuv420p"
            assert (stream["width"], stream["height"]) == size
            assert int(stream["nb_frames"]) == end - start
            # A completed explanation remains readable before playback or without JS.
            subprocess.run([
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", str((end-start-2)/24),
                "-i", str(movie), "-map", "0:v:0", "-frames:v", "1", "-q:v", "3",
                "-update", "1", str(poster)
            ], check=True)
            assert poster.stat().st_size > 0
            for path in (movie, poster):
                shutil.copyfile(path, args.output / path.name)
            receipt["scenes"].append({
                "name": name, "original_scene": original, "start_frame": start, "end_frame": end,
                "frames": end-start, "duration_seconds": (end-start)/24,
                "width": size[0], "height": size[1], "filter": filters,
                "video_sha256": digest(movie), "poster_sha256": digest(poster),
                "video_bytes": movie.stat().st_size, "poster_bytes": poster.stat().st_size,
            })
            print(f"{name}: {end-start} frames, {movie.stat().st_size:,} bytes", flush=True)
    (args.output / "provenance.json").write_text(json.dumps(receipt, indent=2) + "\n")


if __name__ == "__main__":
    main()
