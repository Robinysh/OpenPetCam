"""Create checked browser MP4 copies without changing the source recording."""
import json
import hashlib
import math
import os
from pathlib import Path
import subprocess
import tempfile
import time

VIDEO_TOLERANCE_S = .25


def _finite(value, label):
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"video timeline has no valid {label}") from None
    if not math.isfinite(number):
        raise ValueError(f"video timeline has non-finite {label}")
    return number


def _video_metadata(media):
    video = next((s for s in media.get("streams", []) if s.get("codec_type") == "video"), None)
    if video is None:
        raise ValueError("recording has no video stream")
    container = _finite(media.get("format", {}).get("duration"), "container duration")
    duration = _finite(video.get("duration"), "stream duration")
    start = _finite(video.get("start_time"), "stream start")
    container_start = _finite(media.get("format", {}).get("start_time", 0), "container start")
    if container <= 0 or duration <= 0:
        raise ValueError("video timeline duration must be positive")
    if abs(start - container_start) > VIDEO_TOLERANCE_S or abs(start + duration - container_start - container) > VIDEO_TOLERANCE_S:
        raise ValueError("video timeline does not cover the container window")
    return {"container_duration_s": container, "video_duration_s": duration,
            "video_start_s": start, "video_end_s": start + duration}


def validate_video_timeline(media):
    """Require actual non-discarded video packets across the declared window, not audio duration.

    This verifies relative media coverage only. It cannot establish the absolute
    capture time or recover frames missing from the recording.
    """
    result = _video_metadata(media)
    packets = media.get("video_timeline") or {}
    if packets.get("usable_packets", 0) < 2 or packets.get("unique_pts", 0) < 2 or packets.get("invalid_packets", 0):
        raise ValueError("video timeline lacks valid distinct presentation packets")
    first = _finite(packets.get("first_pts_s"), "first packet time")
    last = _finite(packets.get("last_end_s"), "last packet end")
    covered = _finite(packets.get("covered_duration_s"), "packet coverage")
    gap = _finite(packets.get("largest_gap_s"), "packet gap")
    if first - result["video_start_s"] > VIDEO_TOLERANCE_S or result["video_end_s"] - last > VIDEO_TOLERANCE_S \
            or covered < result["video_duration_s"] - VIDEO_TOLERANCE_S or gap > VIDEO_TOLERANCE_S:
        raise ValueError("video packets do not cover the declared video window")
    return dict(result, packet_coverage_s=covered, largest_packet_gap_s=gap)


def audio_window_status(media):
    """Audio is optional; when present, report its own stream window independently.

    Start/end metadata coverage is not proof of gap-free audio packets. Absent
    audio is explicit None, so a deliberate video-only source remains usable.
    """
    audio = next((s for s in media.get("streams", []) if s.get("codec_type") == "audio"), None)
    result = {"audio_present": audio is not None, "audio_window_complete": None,
              "coverage_basis": "stream start/end metadata; internal packet continuity not verified"}
    if audio is None:
        return result
    result["audio_window_complete"] = False
    try:
        duration = _finite(audio.get("duration"), "audio stream duration")
        start = _finite(audio.get("start_time"), "audio stream start")
        total = _finite(media.get("format", {}).get("duration"), "container duration")
        origin = _finite(media.get("format", {}).get("start_time", 0), "container start")
        result.update(audio_duration_s=duration, audio_start_s=start, audio_end_s=start + duration)
        result["audio_window_complete"] = bool(duration > 0 and total > 0 and abs(start - origin) <= VIDEO_TOLERANCE_S
                                               and abs(start + duration - origin - total) <= VIDEO_TOLERANCE_S)
    except ValueError as exc:
        result["error"] = str(exc)
    return result


def _probe_json(args, max_bytes):
    # Spool bounded diagnostic output to disk rather than accumulating unbounded
    # packet JSON in the watcher. Both output size and wall time are checked.
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(["ffprobe", "-v", "error", *args], stdout=stdout, stderr=stderr)
        deadline = time.monotonic() + 20
        try:
            while True:
                if os.fstat(stdout.fileno()).st_size > max_bytes or os.fstat(stderr.fileno()).st_size > 256_000:
                    raise ValueError("video probe exceeded bounded diagnostic output")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("video probe exceeded 20 seconds")
                try:
                    code = process.wait(timeout=min(.05, remaining))
                    break
                except subprocess.TimeoutExpired:
                    pass
            if code:
                raise ValueError(f"video probe failed with exit code {code}")
            stdout.seek(0)
            data = stdout.read(max_bytes + 1)
            if len(data) > max_bytes:
                raise ValueError("video probe exceeded bounded diagnostic output")
            return json.loads(data)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)


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
            video = next((s for s in media.get("streams", []) if s.get("codec_type") == "video"), {})
            attempt.update(container_duration_s=media.get("format", {}).get("duration"),
                           video_duration_s=video.get("duration"), video_window_complete=False)
            audio = audio_window_status(media)
            attempt.update(audio)
            timeline = validate_video_timeline(media)
            duration = timeline["video_duration_s"]
            video_complete = expected_duration - .5 <= duration <= expected_duration + 2.5
            complete = video_complete and audio["audio_window_complete"] is not False
            attempt.update(duration_s=duration, duration_complete=complete, video_window_complete=video_complete,
                           video_timeline=timeline)
            quality = (complete, duration, audio.get("audio_duration_s", 0))
            if best is None or quality > (best["duration_complete"], best["duration_s"], best.get("audio_duration_s", 0)):
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
              "video_window_complete": bool(best and best["video_window_complete"]),
              "observed_video_duration_s": best["duration_s"] if best else None,
              "audio_present": best["audio_present"] if best else None,
              "audio_window_complete": best["audio_window_complete"] if best else None,
              "observed_audio_duration_s": best.get("audio_duration_s") if best else None,
              "absolute_window_verified": False, "range_provenance": "requested MediaMTX range; absolute timestamps not independently verified",
              "source_file": source.name if best else None, "source_sha256": best["sha256"] if best else None}
    if best:
        os.link(source.parent / best["file"], source)
    write_receipt(receipt_path, result)
    return result


def probe(path):
    media = _probe_json(["-show_entries", "format=duration,start_time:stream=codec_type,codec_name,width,height,pix_fmt,duration,start_time,nb_frames",
                         "-of", "json", str(path)], 64_000)
    try:
        timeline = _video_metadata(media)
    except ValueError as exc:
        # Reject obvious malformed/short video before scanning its packets or converting.
        media["video_timeline_error"] = str(exc)
        return media
    data = _probe_json(["-select_streams", "v:0", "-show_packets", "-show_entries",
                        "packet=pts_time,duration_time,flags", "-of", "json", str(path)], 8_000_000)
    intervals, invalid = [], 0
    packets = data.get("packets", [])
    for packet in packets:
        if "D" in packet.get("flags", ""):
            continue  # MP4 edit-list preroll/discard packets do not provide visible footage
        try:
            start = _finite(packet.get("pts_time"), "packet presentation time")
            duration = _finite(packet.get("duration_time"), "packet duration")
            if duration <= 0:
                raise ValueError("video packet duration must be positive")
            end = min(timeline["video_end_s"], start + duration)
            start = max(timeline["video_start_s"], start)
            if end > start:
                intervals.append((start, end))
        except ValueError:
            invalid += 1
    covered = gap = 0.0
    edge = timeline["video_start_s"]
    for start, end in sorted(intervals):
        gap = max(gap, start - edge)
        covered += max(0.0, end - max(edge, start))
        edge = max(edge, end)
    gap = max(gap, timeline["video_end_s"] - edge)
    media["video_timeline"] = {"total_packets": len(packets), "usable_packets": len(intervals),
                               "invalid_packets": invalid, "unique_pts": len({start for start, _ in intervals}),
                               "first_pts_s": min((start for start, _ in intervals), default=None),
                               "last_end_s": max((end for _, end in intervals), default=None),
                               "covered_duration_s": covered, "largest_gap_s": gap}
    return media


def browser_copy(source, output):
    """Copy H.264 video, encode AAC audio, check the result, then publish atomically."""
    source, output = Path(source), Path(output)
    if source.resolve() == output.resolve() or output.exists():
        raise FileExistsError("source and existing outputs must be preserved")
    before = probe(source)
    video = next((s for s in before["streams"] if s["codec_type"] == "video"), None)
    if video is None or video["codec_name"] != "h264":
        raise ValueError("browser clip requires an H.264 source video")
    old_timeline = validate_video_timeline(before)
    audio = any(s["codec_type"] == "audio" for s in before["streams"])
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".clip-", dir=output.parent) as tmp:
        candidate = Path(tmp) / "browser.mp4"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-n", "-i", str(source),
                        "-map", "0:v:0", "-map", "0:a:0?", "-c:v", "copy", "-c:a", "aac",
                        "-b:a", "96k", "-threads", "1", "-movflags", "+faststart", str(candidate)],
                       capture_output=True, timeout=90, check=True)
        after = probe(candidate)
        new_timeline = validate_video_timeline(after)
        streams = after["streams"]
        if not any(s["codec_type"] == "video" and s["codec_name"] == "h264" for s in streams):
            raise ValueError("browser copy lost H.264 video")
        if audio and not any(s["codec_type"] == "audio" and s["codec_name"] == "aac" for s in streams):
            raise ValueError("browser copy lost AAC audio")
        if abs(new_timeline["container_duration_s"] - old_timeline["container_duration_s"]) > VIDEO_TOLERANCE_S \
                or abs(new_timeline["video_duration_s"] - old_timeline["video_duration_s"]) > VIDEO_TOLERANCE_S:
            raise ValueError("browser copy duration differs from source")
        os.chmod(candidate, 0o644)
        os.link(candidate, output)  # atomic and refuses a concurrently created destination
    source_audio, output_audio = audio_window_status(before), audio_window_status(after)
    return {"source": before, "output": after, "video_reencoded": False,
            "source_audio": source_audio, "output_audio": output_audio,
            "full_av_window_complete": source_audio["audio_window_complete"] is not False
                                       and output_audio["audio_window_complete"] is not False}
