"""Build an offline, hash-addressed corpus of observed room audio.

Inputs are original local recordings, never browser AAC derivatives or dry
source labels. Decoded float samples retain their native rate and signed scale.
Timestamp discontinuities split usable intervals; no resampling, denoising,
normalization, stretching or silence insertion repairs a missing interval.
"""
import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            value.update(block)
    return value.hexdigest()


def _json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def _number(value, label):
    try:
        result = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"missing or invalid {label}") from None
    if not math.isfinite(result):
        raise ValueError(f"nonfinite {label}")
    return result


def _run(command, *, output_limit=32_000_000, timeout=120, monitored_file=None, file_limit=None):
    """Keep local codec diagnostics, runtime, and output size bounded."""
    with tempfile.TemporaryFile() as stdout, tempfile.TemporaryFile() as stderr:
        process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
        deadline = time.monotonic() + timeout
        try:
            while True:
                if stdout.tell() > output_limit or stderr.tell() > 256_000:
                    raise ValueError("codec diagnostics exceed limit")
                if monitored_file and monitored_file.exists() and monitored_file.stat().st_size > file_limit:
                    raise ValueError("decoded audio exceeds limit")
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("audio codec timeout")
                try:
                    code = process.wait(timeout=min(.05, remaining))
                    break
                except subprocess.TimeoutExpired:
                    pass
            stdout.seek(0); data = stdout.read(output_limit + 1)
            stderr.seek(0); errors = stderr.read(256_001)
            if code or errors.strip():
                raise ValueError(f"audio codec failed or reported decode errors (exit {code})")
            if len(data) > output_limit:
                raise ValueError("codec diagnostics exceed limit")
            return data
        finally:
            if process.poll() is None:
                process.kill(); process.wait(timeout=5)


def inspect_audio(path):
    """Decode only the first audio stream for frame counts and timing evidence."""
    data = _run(["ffprobe", "-v", "error", "-threads", "1", "-select_streams", "a:0",
                 "-show_streams", "-show_packets", "-show_frames", "-show_entries",
                 "stream=index,codec_name,codec_type,sample_rate,channels,time_base,start_time,duration:"
                 "packet=pts_time,duration_time,flags:frame=pts_time,best_effort_timestamp_time,nb_samples,sample_rate,channels",
                 "-of", "json", str(path)])
    return json.loads(data)


def analyze_timeline(probed):
    streams = probed.get("streams", [])
    if len(streams) != 1 or streams[0].get("codec_type") != "audio":
        raise ValueError("recording must have one selected audio stream")
    stream = streams[0]
    rate = _number(stream.get("sample_rate"), "audio sample rate")
    if rate != int(rate) or not 8000 <= rate <= 192000 or stream.get("channels") != 1:
        raise ValueError("native mono audio at a supported rate is required; no silent downmix")
    rate = int(rate)
    tolerance = 2 / rate + 1e-6  # two samples plus ffprobe decimal serialization
    rows = probed.get("packets_and_frames", [])
    packets = [x for x in rows if x.get("type") == "packet"]
    frames = [x for x in rows if x.get("type") == "frame"]
    if not frames or not packets:
        raise ValueError("missing audio packet/frame timing evidence")
    packet_discontinuities = []
    packet_previous_end = None
    for index, packet in enumerate(packets):
        stamp = _number(packet.get("pts_time"), "audio packet timestamp")
        duration = _number(packet.get("duration_time"), "audio packet duration")
        if duration <= 0:
            raise ValueError("audio packet duration must be positive")
        if packet_previous_end is not None and abs(stamp - packet_previous_end) > tolerance:
            packet_discontinuities.append({"packet_index": index, "next_pts_s": stamp,
                                            "gap_s": stamp - packet_previous_end})
        packet_previous_end = stamp + duration
    segments, discontinuities, frame_index = [], [], []
    total, previous_end, segment_start, segment_pts = 0, None, 0, None
    for index, frame in enumerate(frames):
        stamp = _number(frame.get("pts_time", frame.get("best_effort_timestamp_time")), "decoded frame timestamp")
        count = _number(frame.get("nb_samples"), "decoded frame samples")
        if count <= 0 or count != int(count) or frame.get("channels", 1) != 1 \
                or int(frame.get("sample_rate", rate)) != rate:
            raise ValueError("invalid or changing decoded audio format")
        count = int(count)
        if previous_end is not None and abs(stamp - previous_end) > tolerance:
            segments.append({"start_sample": segment_start, "stop_sample": total, "container_start_s": segment_pts})
            discontinuities.append({"frame_index": index, "sample_boundary": total, "next_pts_s": stamp,
                                    "gap_s": stamp - previous_end})
            segment_start, segment_pts = total, stamp
        elif segment_pts is None:
            segment_pts = stamp
        frame_index.append({"start_sample": total, "stop_sample": total + count, "container_pts_s": stamp})
        total += count
        previous_end = stamp + count / rate
    segments.append({"start_sample": segment_start, "stop_sample": total, "container_start_s": segment_pts})
    if total > rate * 600:
        raise ValueError("audio exceeds the ten-minute per-recording bound")
    # A packet-level discontinuity can be masked by decoder timestamp repair.
    # Keep the samples, but do not certify any uninterrupted packet timeline.
    return {"sample_rate_hz": rate, "channels": 1, "decoded_samples": total,
            "decoded_duration_s": total / rate, "codec_name": stream.get("codec_name"),
            "timestamp_tolerance_s": tolerance, "packet_count": len(packets), "frame_count": len(frames),
            "packet_discontinuities": packet_discontinuities, "frame_discontinuities": discontinuities,
            "segments": segments, "frame_sample_map": frame_index,
            "segments_timing_eligible": not packet_discontinuities,
            "container_timeline_continuous": not (discontinuities or packet_discontinuities),
            "audio_sample_time": "sample_index / native_sample_rate; gaps are not filled",
            "absolute_time_calibrated": False,
            "video_time_mapping_verified": False,
            "continuity_limit": "Timestamp checks do not prove ADC or transport continuity; concealed codec loss may be undetectable."}


def _path(root, raw):
    if not isinstance(raw, str) or not raw or "://" in raw:
        raise ValueError("only local source paths are accepted")
    path = Path(raw)
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("source is absent or outside inventory repository_root")
    return path


def _checked_source(root, item):
    path = _path(root, item.get("file"))
    if ".browser." in path.name or ".validated-browser." in path.name:
        raise ValueError("browser derivatives are not original localization sources")
    if path.stat().st_size > 128_000_000:
        raise ValueError("source exceeds bounded clip size")
    digest = sha256(path)
    if item.get("sha256") != digest:
        raise ValueError("source SHA256 differs from inventory")
    return path, digest


def leakage_groups(records):
    """Conservative groups: duplicate hashes and overlapping event-time bounds.

    Host timestamps cannot establish actual acoustic/video alignment. The
    wider day groups, including unknown-time data in one group, are the default
    split recommendation. Neither grouping makes source position labels.
    """
    parent = list(range(len(records)))
    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]; index = parent[index]
        return index
    def merge(a, b):
        parent[find(a)] = find(b)
    hashes, windows = {}, []
    for index, row in enumerate(records):
        key = row.get("source_sha256")
        if key in hashes: merge(index, hashes[key])
        if key: hashes[key] = index
        duration = row.get("decoded_duration_s") or 30.
        row["event_time_bounds"] = []
        for event in row.get("events", []):
            stamp = event.get("event_ts")
            if not isinstance(stamp, (int, float)) or not math.isfinite(stamp): continue
            # Enclose an entire clip on either side; event range is a grouping
            # heuristic only, with one minute extra clock uncertainty.
            bound = [stamp - duration - 60, stamp + duration + 60]
            row["event_time_bounds"].append(bound); windows.append((bound[0], bound[1], index))
    end, previous = -math.inf, None
    for start, stop, index in sorted(windows):
        if start <= end: merge(index, previous)
        if stop > end: end, previous = stop, index
    def labels(field):
        members = {}
        for index, row in enumerate(records): members.setdefault(find(index), []).append(row["source_sha256"])
        names = {key: hashlib.sha256("\n".join(sorted(set(values))).encode()).hexdigest()[:16] for key, values in members.items()}
        for index, row in enumerate(records): row[field] = names[find(index)]
    labels("overlap_group")
    days = {}
    for index, row in enumerate(records):
        keys = set()
        for lower, upper in row["event_time_bounds"]:
            keys.update((dt.datetime.fromtimestamp(t, dt.timezone.utc).date().isoformat() for t in (lower, upper)))
        if not keys: keys = {"unknown_time"}
        row["split_eligible"] = "unknown_time" not in keys
        for key in keys:
            if key in days: merge(index, days[key])
            else: days[key] = index
    labels("recommended_split_group")


def build_corpus(inventory_path, output_dir):
    inventory_path, output_dir = Path(inventory_path).resolve(), Path(output_dir).resolve()
    inventory = json.loads(inventory_path.read_text())
    root = Path(inventory.get("repository_root", inventory_path.parent)).resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    media_dir, reference_dir = output_dir / "audio", output_dir / "references"
    media_dir.mkdir(); reference_dir.mkdir()
    report = {"schema": "openpetcam.localization-corpus.v1", "inventory_file": str(inventory_path),
              "inventory_sha256": sha256(inventory_path), "complete": False,
              "builder_sha256": sha256(__file__), "records": [], "failures": [], "probe_references": [],
              "signal_policy": "native-rate mono float32; signed scale preserved; no filters, gain, normalization, denoise, resample, padding or timestamp repair",
              "source_kind": "observed_room_recordings; never dry source recordings",
              "label_policy": "Unknown sound source position and microphone pose remain null. Event IDs/times are bookkeeping, not calibrated localization truth.",
              "split_policy": "Use recommended_split_group and exclude split_eligible=false records from train/validation comparisons; unknown times cannot prove disjointness. Duplicates/estimated overlaps must never cross splits. Day grouping is conservative because host/video times are uncalibrated.",
              "versions": {name: _run([name, "-version"], output_limit=100_000).decode().splitlines()[0] for name in ("ffmpeg", "ffprobe")}}
    seen = {}
    for item in inventory.get("passive_event_originals", []):
        try:
            source, digest = _checked_source(root, item)
            if digest in seen:
                row = seen[digest]
                row["source_aliases"].append(str(source));row["events"].extend(item.get("events", []))
                continue
            probed = inspect_audio(source)
            timeline = analyze_timeline(probed)
            output = media_dir / f"{digest}.f32le"
            command = ["ffmpeg", "-nostdin", "-v", "error", "-n", "-err_detect", "explode", "-i", str(source),
                       "-map", "0:a:0", "-vn", "-sn", "-dn", "-c:a", "pcm_f32le", "-threads", "1", "-f", "f32le", str(output)]
            _run(command, monitored_file=output, file_limit=timeline["decoded_samples"] * 4 + 4096)
            if output.stat().st_size != timeline["decoded_samples"] * 4:
                raise ValueError("decoded byte/sample count differs from independently probed audio frames")
            if sha256(source) != digest:
                raise ValueError("source changed while decoding")
            signal = np.memmap(output, dtype="<f4", mode="r")
            if not np.isfinite(signal).all():
                raise ValueError("decoded samples are nonfinite")
            evidence_path = media_dir / f"{digest}.timeline.json"
            _json(evidence_path, {"probe": probed, "timeline": timeline})
            row = {"record_id": digest[:20], "source_sha256": digest, "source_aliases": [str(source)],
                   "source_provenance": item.get("source_provenance"), "events": list(item.get("events", [])),
                   "file": str(output.relative_to(output_dir)), "sha256": sha256(output), "format": "f32le",
                   "source_kind": "observed_room_recording", "source_position_m": None, "microphone_pose": None,
                   "sample_rate_hz": timeline["sample_rate_hz"], "channels": 1,
                   "decoded_samples": len(signal), "decoded_duration_s": timeline["decoded_duration_s"],
                   "codec_name": timeline["codec_name"], "segments": timeline["segments"],
                   "container_timeline_continuous": timeline["container_timeline_continuous"],
                   "segments_timing_eligible": timeline["segments_timing_eligible"],
                   "frame_discontinuities": len(timeline["frame_discontinuities"]),
                   "packet_discontinuities": len(timeline["packet_discontinuities"]),
                   "timeline_file": str(evidence_path.relative_to(output_dir)), "timeline_sha256": sha256(evidence_path),
                   "absolute_time_calibrated": False, "video_time_mapping_verified": False,
                   "rms": float(np.sqrt(np.mean(signal.astype(np.float64) ** 2))),
                   "peak_absolute": float(np.max(np.abs(signal))),
                   "above_unity_samples": int(np.count_nonzero(np.abs(signal) > 1)),
                   "amplitude_note": "Decoded full-scale units, not calibrated SPL; codec overshoots above unity preserved, not proof of analog clipping.",
                   "status": "decoded" if timeline["container_timeline_continuous"] else "decoded_with_timestamp_discontinuities"}
            report["records"].append(row);seen[digest] = row
        except (ValueError, OSError, TimeoutError, KeyError, json.JSONDecodeError) as exc:
            report["failures"].append({"file": item.get("file"), "source_sha256": item.get("sha256"),
                                       "events": item.get("events", []), "error": str(exc), "status": "rejected"})
    for item in inventory.get("raw_probe_references", []):
        try:
            source, digest = _checked_source(root, item)
            target = reference_dir / (digest + source.suffix)
            if not target.exists():
                with source.open("rb") as inp, target.open("xb") as out: shutil.copyfileobj(inp, out)
            if sha256(target) != digest or sha256(source) != digest:
                raise ValueError("probe reference hash changed")
            report["probe_references"].append({**item, "original_file": str(source), "file": str(target.relative_to(output_dir)),
                                               "kind": "digital_playback_reference_not_free_field_source"})
        except (ValueError, OSError) as exc:
            report["failures"].append({"file": item.get("file"), "error": str(exc), "kind": "probe_reference", "status": "rejected"})
    leakage_groups(report["records"])
    report["summary"] = {"input_originals": len(inventory.get("passive_event_originals", [])),
                          "unique_decoded_recordings": len(report["records"]), "failures": len(report["failures"]),
                          "status_counts": dict(Counter(row["status"] for row in report["records"])),
                          "decoded_seconds": sum(row["decoded_duration_s"] for row in report["records"]),
                          "overlap_groups": len({row["overlap_group"] for row in report["records"]}),
                          "recommended_split_groups": len({row["recommended_split_group"] for row in report["records"]}),
                          "probe_reference_entries": len(report["probe_references"])}
    report["complete"] = True  # inventory processed; failures remain explicit
    _json(output_dir / "manifest.json", report)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(build_corpus(args.inventory, args.output)["summary"], indent=2))
