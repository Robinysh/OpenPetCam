"""Create checked browser MP4 copies without changing the source recording."""
import json
import hashlib
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time


def write_receipt(path, value):
    path = Path(path)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, prefix=".capture-", delete=False) as out:
        json.dump(value, out, indent=2)
        temp = out.name
    os.replace(temp, path)


def collect_recording(fetch, source, expected_duration, receipt_path):
    """Try three bounded caller-supplied fetches; keep every media attempt and its provenance.

    A sufficient duration does not establish an absolute timestamp mapping. The
    requested MediaMTX range remains an explicit assumption, even for a full file.
    """
    source = Path(source)
    expected_duration = float(expected_duration)
    if not math.isfinite(expected_duration) or expected_duration <= 0:
        raise ValueError("invalid requested recording duration")
    if source.exists():
        raise FileExistsError("recording sources must be preserved")
    source.parent.mkdir(parents=True, exist_ok=True)
    attempts, best = [], None
    for number, delay in enumerate((0, 4, 12), 1):
        if delay:
            time.sleep(delay)
        attempt = {"number": number, "at": time.time()}
        try:
            data = fetch()
            if not isinstance(data, bytes) or len(data) < 10000:
                raise ValueError("recording is empty or too small")
            name = source.with_name(f"{source.stem}.attempt-{number}{source.suffix}")
            with name.open("xb") as out:
                out.write(data)
            attempt.update(file=name.name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest())
            media = probe(name)
            duration = float(media["format"]["duration"])
            if not math.isfinite(duration) or duration <= 0 or not any(s["codec_type"] == "video" for s in media["streams"]):
                raise ValueError("recording has no valid video duration")
            complete = expected_duration - .5 <= duration <= expected_duration + 2.5
            attempt.update(duration_s=duration, duration_complete=complete)
            if best is None or complete or duration > best["duration_s"]:
                best = dict(attempt)
            attempts.append(attempt)
            if complete:
                break
        except Exception as exc:  # noqa: BLE001 - each failed bounded attempt stays auditable
            attempt["error"] = str(exc)
            attempts.append(attempt)
    result = {"schema": "openpetcam.clip-capture.v1", "attempts": attempts,
              "requested_duration_s": expected_duration, "duration_complete": bool(best and best["duration_complete"]),
              "observed_duration_s": best["duration_s"] if best else None,
              "absolute_window_verified": False, "range_provenance": "requested MediaMTX range; absolute timestamps not independently verified",
              "source_file": source.name if best else None, "source_sha256": best["sha256"] if best else None}
    if best:
        os.link(source.parent / best["file"], source)
    write_receipt(receipt_path, result)
    return result


def probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                        "format=duration:stream=codec_type,codec_name,width,height,pix_fmt",
                        "-of", "json", str(path)], capture_output=True, text=True, timeout=20, check=True)
    return json.loads(r.stdout)


def browser_copy(source, output):
    """Copy H.264 video, encode AAC audio, check the result, then publish atomically."""
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve() or output.exists():
        raise FileExistsError("source and existing outputs must be preserved")
    before = probe(source)
    video = next((s for s in before["streams"] if s["codec_type"] == "video"), None)
    if video is None or video["codec_name"] != "h264":
        raise ValueError("browser clip requires an H.264 source video")
    audio = any(s["codec_type"] == "audio" for s in before["streams"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".clip-", dir=output.parent) as tmp:
        candidate = Path(tmp) / "browser.mp4"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-n", "-i", str(source),
                        "-map", "0:v:0", "-map", "0:a:0?", "-c:v", "copy", "-c:a", "aac",
                        "-b:a", "96k", "-threads", "1", "-movflags", "+faststart", str(candidate)],
                       capture_output=True, timeout=90, check=True)
        after = probe(candidate)
        streams = after["streams"]
        if not any(s["codec_type"] == "video" and s["codec_name"] == "h264" for s in streams):
            raise ValueError("browser copy lost H.264 video")
        if audio and not any(s["codec_type"] == "audio" and s["codec_name"] == "aac" for s in streams):
            raise ValueError("browser copy lost AAC audio")
        old_duration = float(before["format"]["duration"])
        new_duration = float(after["format"]["duration"])
        if old_duration <= 0 or abs(new_duration - old_duration) > .25:
            raise ValueError("browser copy duration differs from source")
        os.chmod(candidate, 0o644)
        os.link(candidate, output)  # atomic and refuses a concurrently created destination
    return {"source": before, "output": after, "video_reencoded": False}
