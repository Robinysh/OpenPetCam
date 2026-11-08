"""Bounded, unvalidated room-fingerprint diagnostics on original local audio.

No source or historical microphone coordinates are inferred as measured truth.
Signed native samples remain read-only; the derived power spectra are restricted
to the frozen dictionary's frequency support before feature smoothing.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from .localization import rank_candidates, ripple_features, spectrum


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def native_window_samples(rate, reference_rate, reference_nfft):
    """Use an exact native .512s-or-longer grid; never pad/resample a short cut."""
    if any(not isinstance(v, int) or isinstance(v, bool) or v <= 0
           for v in (rate, reference_rate, reference_nfft)):
        raise ValueError("positive integer sample grids required")
    if reference_nfft / reference_rate < .512:
        raise ValueError("dictionary must have at least .512 seconds of spectral support")
    count, remainder = divmod(reference_nfft * rate, reference_rate)
    if remainder or count < 128:
        raise ValueError("native grid cannot match frozen frequency spacing without resampling")
    return count


def eligible_windows(row, count):
    """Return full nonoverlapping windows inside declared continuous segments."""
    if not row.get("segments_timing_eligible"):
        return []
    total, previous = row["decoded_samples"], 0
    result = []
    for segment_index, segment in enumerate(row.get("segments", [])):
        start, stop = segment["start_sample"], segment["stop_sample"]
        if (not isinstance(start, int) or not isinstance(stop, int)
                or start < previous or start < 0 or stop <= start or stop > total
                or not math.isfinite(segment["container_start_s"])):
            raise ValueError("invalid or overlapping declared audio segments")
        previous = stop
        for lower in range(start, stop - count + 1, count):
            result.append({"segment_index": segment_index, "start_sample": lower,
                           "stop_sample": lower + count,
                           "container_start_s": segment["container_start_s"] + (lower - start) / row["sample_rate_hz"]})
    return result


def grouping(row):
    """Keep corpus duplicate/overlap/day grouping; never form independent splits."""
    required = ("source_sha256", "overlap_group", "recommended_split_group", "split_eligible")
    if any(key not in row for key in required):
        raise ValueError("corpus leakage-group metadata is required")
    return {key: row[key] for key in required}


def run(corpus_path, benchmark_dir, output_dir, *, max_records=200, windows_per_record=3,
        max_windows=512, deadline_s=1200):
    if not (1 <= max_records <= 1000 and 1 <= windows_per_record <= 8
            and 1 <= max_windows <= 1024 and 10 <= deadline_s <= 3600):
        raise ValueError("diagnostic bounds exceeded")
    corpus_path, benchmark_dir, output_dir = map(lambda p: Path(p).resolve(),
                                                (corpus_path, benchmark_dir, output_dir))
    manifest = json.loads(corpus_path.read_text())
    protocol = json.loads((benchmark_dir / "protocol.json").read_text())
    selected = json.loads((benchmark_dir / "selected.json").read_text())
    if not manifest.get("complete"):
        raise ValueError("complete corpus manifest required")
    config = selected["config"]
    reference_rate, reference_nfft = protocol["sample_rate"], protocol["nfft"]
    dictionary_file = benchmark_dir / "dictionary.npz"
    with np.load(dictionary_file, allow_pickle=False) as archive:
        positions = archive["positions_m"].copy()
        model_features = archive[config["name"]].copy()
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    frozen = {"schema": "openpetcam.localization-real-diagnostic.v1",
              "created": time.time(), "module_sha256": sha(__file__),
              "corpus_manifest": str(corpus_path), "corpus_manifest_sha256": sha(corpus_path),
              "benchmark_dir": str(benchmark_dir),
              "benchmark_hashes": {name: sha(benchmark_dir / name) for name in
                                   ("protocol.json", "selected.json", "dictionary.npz")},
              "config": config, "max_records": max_records, "windows_per_record": windows_per_record,
              "max_windows": max_windows, "deadline_s": deadline_s,
              "window_policy": "Nonoverlapping complete native windows inside declared timestamp-continuous segments; choose highest RMS windows per recording, deterministic sample-index ties.",
              "sample_policy": "Read-only original signed native float32 scale; no gain, alignment, filtering, padding, denoising or resampling.",
              "spectrum_policy": "Native Welch nfft=reference_nfft*native_rate/reference_rate, preserving physical frequency spacing; restrict power/frequencies to dictionary Nyquist before ripple smoothing.",
              "source_position_m": None, "microphone_pose": None,
              "real_accuracy_measured": False, "acceptance_calibrated_on_real_data": False,
              "warning": "Candidate model coordinates assume the dictionary receiver pose. Historical microphone pose and emitting source are unknown. Scores, synthetic-rule passes and spatial score sets are unvalidated diagnostics, not real localization accuracy or confidence."}
    write_json(output_dir / "protocol.json", frozen)
    outcomes, exclusions, seen_groups = [], [], {}
    rows = manifest.get("records", [])
    completed_records = 0
    with (output_dir / "rankings.jsonl").open("x") as stream:
        for row in rows[:max_records]:
            if time.monotonic() - started > deadline_s or len(outcomes) >= max_windows:
                break
            record_id = row["record_id"]
            try:
                groups = grouping(row)
                key = row["source_sha256"]
                if key in seen_groups:
                    if seen_groups[key] != groups:
                        raise ValueError("duplicate source has inconsistent corpus leakage groups")
                    exclusions.append({"record_id": record_id, "reason": "duplicate source already diagnosed"})
                    completed_records += 1
                    continue
                seen_groups[key] = groups
                if row.get("source_kind") != "observed_room_recording" or row.get("format") != "f32le" or row.get("channels") != 1:
                    raise ValueError("original observed mono float32 room audio required")
                rate = row["sample_rate_hz"]
                count = native_window_samples(rate, reference_rate, reference_nfft)
                windows = eligible_windows(row, count)
                if not windows:
                    raise ValueError("no full >=.512s timing-eligible continuous window")
                path = (corpus_path.parent / row["file"]).resolve()
                if not path.is_relative_to(corpus_path.parent) or path.stat().st_size != row["decoded_samples"] * 4 or sha(path) != row["sha256"]:
                    raise ValueError("corpus sample path, length or hash mismatch")
                samples = np.memmap(path, dtype="<f4", mode="r")
                # Energy only chooses diagnostic windows; it supplies no source label.
                for window in windows:
                    cut = samples[window["start_sample"]:window["stop_sample"]]
                    if not np.isfinite(cut).all():
                        raise ValueError("nonfinite native samples")
                    window["native_rms"] = float(np.sqrt(np.mean(np.square(cut, dtype=np.float64))))
                windows.sort(key=lambda w: (-w["native_rms"], w["start_sample"]))
                for window in windows[:min(windows_per_record, max_windows - len(outcomes))]:
                    if time.monotonic() - started > deadline_s:
                        break
                    cut = samples[window["start_sample"]:window["stop_sample"]]
                    frequencies, power = spectrum(cut, rate, nfft=count)
                    keep = frequencies <= reference_rate / 2
                    frequencies, power = frequencies[keep], power[keep]
                    if len(frequencies) != reference_nfft // 2 + 1:
                        raise ValueError("native spectrum does not match frozen dictionary frequency grid")
                    feature, ripple_rms = ripple_features(power, frequencies,
                        **{k: v for k, v in config.items() if k != "name"})
                    rule = selected["acceptance_rule"]
                    rank = rank_candidates(feature, model_features, positions,
                        min_margin=rule["min_margin"], min_similarity=rule["min_similarity"])
                    rank.pop("scores")
                    rank["passes_synthetic_rule"] = rank.pop("accepted")
                    rank["accepted"] = False
                    rank["candidate_model_position_m"] = rank.pop("position_m")
                    result = {"record_id": record_id, **groups, **window, **rank,
                              "source_position_m": None, "microphone_pose": None,
                              "native_sample_rate_hz": rate, "native_window_samples": count,
                              "window_duration_s": count / rate, "analysis_nfft": count,
                              "dictionary_nyquist_hz": reference_rate / 2,
                              "resampling": {"applied": False, "native_rate_hz": rate},
                              "signed_native_window_sha256": hashlib.sha256(cut.tobytes()).hexdigest(),
                              "native_peak_absolute": float(np.max(np.abs(cut))),
                              "feature_ripple_rms": float(ripple_rms),
                              "dictionary_assumed_receiver_m": protocol.get("receiver_m"),
                              "real_accuracy_measured": False}
                    outcomes.append(result)
                    stream.write(json.dumps(result, allow_nan=False) + "\n"); stream.flush()
                if sha(path) != row["sha256"]:
                    raise ValueError("corpus changed during read-only diagnosis")
            except (ValueError, KeyError, OSError) as error:
                exclusions.append({"record_id": record_id, "reason": str(error)})
            completed_records += 1
            progress = {"pid": os.getpid(), "records_done": completed_records,
                        "records_total": min(len(rows), max_records), "windows": len(outcomes),
                        "elapsed_s": time.monotonic() - started, "updated": time.time()}
            temporary = output_dir / ".progress.tmp"
            temporary.write_text(json.dumps(progress, indent=2)); temporary.replace(output_dir / "progress.json")
    summary = {"complete": completed_records == min(len(rows), max_records),
               "records_inspected": completed_records, "records_total": len(rows),
               "windows": len(outcomes), "exclusions": exclusions,
               "statuses": dict(Counter(row["status"] for row in outcomes)),
               "native_sample_rates_hz": sorted(set(row["native_sample_rate_hz"] for row in outcomes)),
               "overlap_groups": len(set(row["overlap_group"] for row in outcomes)),
               "recommended_split_groups": len(set(row["recommended_split_group"] for row in outcomes)),
               "passes_synthetic_rule": sum(row["passes_synthetic_rule"] for row in outcomes),
               "accepted_real_locations": 0, "real_accuracy_measured": False,
               "elapsed_s": time.monotonic() - started, "limits": frozen["warning"]}
    write_json(output_dir / "summary.json", summary)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-records", type=int, default=200)
    parser.add_argument("--windows-per-record", type=int, default=3)
    parser.add_argument("--max-windows", type=int, default=512)
    parser.add_argument("--deadline-s", type=float, default=1200)
    arguments = parser.parse_args()
    print(json.dumps(run(arguments.corpus, arguments.benchmark, arguments.output,
                         max_records=arguments.max_records, windows_per_record=arguments.windows_per_record,
                         max_windows=arguments.max_windows, deadline_s=arguments.deadline_s), indent=2))
