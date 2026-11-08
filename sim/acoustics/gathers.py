"""Audit captured probe packets and assemble conditional, signed shot gathers.

This module never moves hardware, aligns one room response to another, stacks
before QC, or promotes an acoustic correlation peak to a physical emission time.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
import tempfile

import numpy as np
import scipy
from scipy import signal

from .capture import assets
from .device.decode import export_capture
from .probes import load_probe_packet
from .shot_qc import QCConfig, ShotWindow, assess_capture, fit_packet_timing


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _pcm(path):
    raw = Path(path).read_bytes()
    if not raw or len(raw) % 2:
        raise ValueError("empty or odd-length s16le PCM")
    return np.frombuffer(raw, "<i2").astype(float) / 32768.


def _local(root, name):
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("capture path escapes session directory")
    return path


def _verified_reference(root, row, manifest, packet, mic_count):
    """Reconstruct the reference proof from immutable framed device bytes."""
    acq = row["acquisition"]
    required = ("reference_file", "reference_sha256", "framed_file", "framed_sha256",
                "device_metadata_file", "device_metadata_sha256")
    if any(not acq.get(k) for k in required):
        raise ValueError("paired reference requires framed device proof, not bare sample grid flags")
    if not all(acq.get(k) is True for k in ("reference_sample_grid_verified", "reference_continuity_verified", "active_ao_reference_verified")):
        raise ValueError("paired reference lacks verified common sample grid/continuity/active AO validation")
    framed = _local(root, acq["framed_file"])
    metadata_path = _local(root, acq["device_metadata_file"])
    if _sha(framed) != acq["framed_sha256"] or _sha(metadata_path) != acq["device_metadata_sha256"]:
        raise ValueError("framed device or metadata hash mismatch")
    archived = json.loads(metadata_path.read_text())
    if archived.get("schema") != "openpetcam.device.capture.v1" or archived.get("complete") is not True:
        raise ValueError("invalid or incomplete device metadata")
    expected = {"framed_sha256": acq["framed_sha256"], "mic_sha256": row["pcm_sha256"],
                "reference_sha256": acq["reference_sha256"]}
    if any(archived.get(k) != v for k, v in expected.items()):
        raise ValueError("device metadata does not identify these captured microphone/reference bytes")
    ref_path = _local(root, acq["reference_file"])
    if _sha(ref_path) != acq["reference_sha256"]:
        raise ValueError("reference PCM hash mismatch")
    source_path = _local(root, Path(archived["source_file"]).name)
    if _sha(source_path) != archived.get("source_sha256"):
        raise ValueError("submitted AO source hash mismatch")
    with tempfile.TemporaryDirectory(prefix="openpetcam-device-verify-") as tmp:
        verified = export_capture(framed, tmp, prefix="verify")
        if any(verified.get(k) != v for k, v in expected.items()) or verified["source_sha256"] != archived.get("source_sha256"):
            raise ValueError("framed reconstruction differs from archived PCM")
        if not all(verified.get(k) is True for k in ("mic_continuity_verified", "reference_sample_grid_verified", "reference_continuity_verified")):
            raise ValueError("framed stream did not independently verify paired reference continuity")
        if not verified.get("active_ao_reference_check", {}).get("verified"):
            raise ValueError("active AO-to-reference validation failed")
        if verified["common_mic_sample_interval"] != archived.get("common_mic_sample_interval"):
            raise ValueError("device common interval changed")
    submitted = source_path.read_bytes()
    expected_pcm = np.rint(packet * 32768).astype("<i2").tobytes()
    padding = submitted[len(expected_pcm):]
    if (verified.get("source_rate_hz") != manifest["sample_rate_hz"]
            or not submitted.startswith(expected_pcm)
            or len(padding) > round(.1 * manifest["sample_rate_hz"]) * 2 or any(padding)):
        raise ValueError("submitted AO PCM is not the exact declared playback packet plus bounded zero padding")
    reference = _pcm(ref_path)
    if len(reference) != mic_count:
        raise ValueError("paired reference sample count differs from microphone")
    proof = {"framed_sha256": expected["framed_sha256"], "device_metadata_sha256": acq["device_metadata_sha256"],
             "independently_redecoded": True, "active_ao_reference_check": verified["active_ao_reference_check"],
             "source_packet_bit_identical": True, "zero_padding_source_samples": len(padding) // 2,
             "physical_DAC_timing_verified": False}
    return reference, proof


def _pilot_pair(x, pilot, events, fs, source_fs, *, max_rate_error_ppm):
    """Find two pilots jointly using their source separation, never shot peaks."""
    p = signal.resample_poly(pilot, fs, source_fs)
    if len(x) < len(p):
        raise ValueError("capture shorter than pilot")
    corr = signal.correlate(x, p, "valid", method="fft")
    energy = np.maximum(signal.correlate(x * x, np.ones(len(p)), "valid", method="fft"), 0)
    score = np.divide(np.abs(corr), np.sqrt((p @ p) * energy), out=np.zeros_like(corr), where=energy > 1e-18)
    peaks, _ = signal.find_peaks(score, distance=max(1, len(p) // 2), height=.15)
    peaks = peaks[np.argsort(score[peaks])[-256:]]
    expected = (events[-1]["sample_start"] - events[0]["sample_start"]) * fs / source_fs
    tolerance = max(3., expected * max_rate_error_ppm / 1e6)
    choices = []
    for first in peaks:
        for last in peaks:
            if abs((last - first) - expected) <= tolerance:
                choices.append((float(score[first] * score[last]), int(first), int(last)))
    if not choices:
        raise ValueError("no two pilot detections satisfy the packet separation/rate bound")
    choices.sort(reverse=True)
    if len(choices) > 1 and choices[1][0] >= .98 * choices[0][0]:
        raise ValueError("ambiguous pilot pair")
    best = choices[0]
    return [best[1], best[2]], {"joint_score": best[0], "candidate_pairs": len(choices),
                               "expected_separation_samples": expected,
                               "separation_tolerance_samples": tolerance,
                               "selection_uses_shot_waveforms": False}


def _band_power(x, fs, bands):
    # Integrate the actual acquisition-window energy. A Hann window would
    # favor frequencies occurring near the sweep's midpoint, differently for
    # linear and exponential probes, and bias the comparison by several dB.
    f, power = signal.periodogram(x, fs=fs, window="boxcar", detrend=False, scaling="density")
    df = f[1] - f[0]
    return [float(power[(f >= low) & (f < high)].sum() * df) for low, high in bands]


def _responses(x, reference, probe, manifest, timing, fs, shots, *, tail_s, pre_s=.025):
    """Keep raw cuts and deconvolve using the same source map for every shot.

    With SDK reference, divide by the paired actual output reference segment.
    Otherwise the nominal archived waveform is placed using the pilot-only map;
    this produces an explicitly conditional acoustic origin across poses.
    """
    slope = timing["slope"]
    probe_n = int(np.ceil(len(probe) * slope))
    pre, tail = round(pre_s * fs), round(tail_s * fs)
    length = pre + probe_n + tail
    nfft = 1 << (2 * length - 1).bit_length()
    f = np.fft.rfftfreq(nfft, 1 / fs)
    low, high = manifest["probe"]["band_hz"]
    use = (f > low) & (f < high)
    band = np.zeros_like(f)
    band[use] = signal.windows.tukey(int(use.sum()), .2)
    # Preserve a full probe duration of negative lag, including ESS precursors.
    negative = round(len(probe) * fs / manifest["sample_rate_hz"])
    time = np.arange(-negative, tail) / fs
    nominal = signal.resample_poly(probe, fs, manifest["sample_rate_hz"])
    raw, refs, hs, ranges, fractions = [], [], [], [], []
    for e in shots:
        origin = timing["intercept"] + slope * e["sample_start"]
        lo = int(np.floor(origin)) - pre
        hi = lo + length
        if lo < 0 or hi > len(x):
            raise ValueError("truncated common-map shot response")
        y = x[lo:hi].copy()
        if reference is not None:
            source = reference[lo:hi].copy()
            S = np.fft.rfft(source, nfft)
        else:
            # Approximate fractional rate using one capture-wide map, never
            # individual measured shot arrivals. Archive this assumption.
            ratio = slope / (fs / manifest["sample_rate_hz"])
            src = np.interp(np.arange(probe_n) / ratio, np.arange(len(nominal)), nominal, left=0, right=0)
            source = np.zeros(length)
            source[pre:pre + len(src)] = src
            frac = origin - np.floor(origin)
            S = np.fft.rfft(source, nfft) * np.exp(-2j * np.pi * f * frac / fs)
        if not np.any(np.abs(S) > 0):
            raise ValueError("zero source/reference waveform")
        inverse = band * S.conj() / (np.abs(S) ** 2 + 1e-3 * np.max(np.abs(S)) ** 2)
        h = np.fft.irfft(np.fft.rfft(y, nfft) * inverse, nfft)
        raw.append(y); refs.append(source); hs.append(np.r_[h[-negative:], h[:tail]])
        ranges.append([lo, hi]); fractions.append(float(origin - np.floor(origin)))
    return {"raw": np.array(raw), "reference": np.array(refs), "responses": np.array(hs),
            "time_s": time, "adc_sample_ranges": np.array(ranges),
            "fractional_source_origin_samples": np.array(fractions)}


def _analyze_capture(root, row, entry, packet_path, *, config, tail_s, max_rate_error_ppm):
    if row.get("status") != "recorded":
        raise ValueError("capture acquisition did not finish with recorded status")
    for key, value in entry.items():
        if row.get(key) != value:
            raise ValueError(f"capture differs from plan: {key}")
    if (not row.get("pose_stable") or row.get("actual_pose_before") != [entry["pan"], entry["tilt"]]
            or row.get("actual_pose_after") != [entry["pan"], entry["tilt"]]):
        raise ValueError("actual source/receiver pose changed or differs from plan")
    packet, probe, pilot, manifest = load_probe_packet(packet_path)
    if row.get("packet_sha256") != manifest["packet"]["sha256"]:
        raise ValueError("capture playback packet hash mismatch")
    path = _local(root, row["file"])
    if _sha(path) != row.get("pcm_sha256"):
        raise ValueError("capture PCM hash mismatch")
    acquisition = row["acquisition"]
    fs = acquisition["sample_rate_hz"]
    if fs != 48000:
        raise ValueError("capture sample rate must be declared 48000 Hz")
    x = _pcm(path)
    if acquisition.get("samples", len(x)) != len(x):
        raise ValueError("capture sample inventory mismatch")
    reference, reference_proof = None, None
    if acquisition.get("reference_file") or acquisition.get("framed_file"):
        reference, reference_proof = _verified_reference(root, row, manifest, packet, len(x))
    source_fs = manifest["sample_rate_hz"]
    pilots = [e for e in manifest["events"] if e["type"] == "pilot"]
    shots = [e for e in manifest["events"] if e["type"] == "shot"]
    anchor_signal = x if reference is None else reference
    peaks, pair = _pilot_pair(anchor_signal, pilot, pilots, fs, source_fs, max_rate_error_ppm=max_rate_error_ppm)
    quiet_stop = min(peaks[0] - round(.1 * fs), round(.25 * fs))
    if quiet_stop < round(.05 * fs):
        raise ValueError("insufficient verified pre-pilot quiet window")
    width = round(.002 * fs)
    windows = [ShotWindow(e["event_id"], e["sample_start"], p - width, p + width + 1, 0, quiet_stop)
               for e, p in zip(pilots, peaks)]
    pilot_qc = assess_capture(anchor_signal, sample_rate_hz=fs, source_rate_hz=source_fs,
                              probe=pilot, shots=windows, response_tail_s=tail_s, config=config)
    if not all(q["accepted"] for q in pilot_qc["shots"]):
        raise ValueError("pilot QC failed: " + json.dumps([q["rejection_reasons"] for q in pilot_qc["shots"]]))
    timing = fit_packet_timing(pilot_qc["shots"], source_rate_hz=source_fs, sample_rate_hz=fs)
    if abs(timing["relative_rate_error_ppm"]) > max_rate_error_ppm:
        raise ValueError("pilot-derived sample-rate ratio exceeds configured bound")
    if timing["intercept"] < 0 or timing["intercept"] + timing["slope"] * len(packet) > len(x):
        raise ValueError("truncated packet under the pilot-derived sample map")
    width = round(.02 * fs)
    windows = []
    for e in shots:
        center = round(timing["intercept"] + timing["slope"] * e["sample_start"])
        windows.append(ShotWindow(e["event_id"], e["sample_start"], center - width, center + width + 1, 0, quiet_stop))
    # Different pilot/shot families can have different acoustic peak bias. The
    # mic residual is diagnostic, not a three-sample clock-calibration test.
    qc_config = replace(config, max_timing_residual_samples=float(width + 1))
    qc = assess_capture(x, sample_rate_hz=fs, source_rate_hz=source_fs, probe=probe,
                        shots=windows, response_tail_s=tail_s, config=qc_config, timing_map=timing)
    for q in qc["shots"]:
        q["timing_residual_interpretation"] = "uncalibrated pilot/shot wavelet bias and propagation; not pure clock error"
    arrays = _responses(x, reference, probe, manifest, timing, fs, shots, tail_s=tail_s)
    noise_power = _band_power(x[:quiet_stop], fs, config.bands_hz)
    for q, raw, e in zip(qc["shots"], arrays["raw"], shots):
        pre = round(.025 * fs)
        power = _band_power(raw[pre:pre + round(len(probe) * timing["slope"])], fs, config.bands_hz)
        q["band_snr_db"] = [{"band_hz": list(b), "total_signal_power": v, "noise_power": n,
                              "snr_db": float(10 * np.log10(max(v, 1e-20) / max(n, 1e-20)))}
                             for b, v, n in zip(config.bands_hz, power, noise_power)]
        q["repeat_id"] = e["repeat_id"]
    paired = reference is not None
    return {"capture_id": entry["capture_id"], "status": "accepted" if qc["accepted"] else "rejected",
            "entry": entry, "file": row["file"], "pcm_sha256": row["pcm_sha256"],
            "pilot_pair": pair, "pilot_qc": pilot_qc, "qc": qc, "timing_map": timing,
            "timing_status": "sdk_paired_reference_continuity_checked" if paired else "local_acoustic_pilot_origin_conditional",
            "cross_pose_phase_calibrated": False, "physical_dac_timing_calibrated": False,
            "source_reference": "paired SDK output reference" if paired else "archived nominal source mapped by acoustic pilots",
            "reference_proof": reference_proof,
            "deconvolution": {"regularization": 1e-3, "negative_lag_s": -float(arrays["time_s"][0]),
                              "positive_lag_s": tail_s, "band_hz": manifest["probe"]["band_hz"],
                              "per_shot_peak_alignment": False, "amplitude_normalization": False},
            "probe": manifest["probe"], "comparison_group": manifest["comparison_group"],
            "cost": {"packet_duration_s": len(packet) / source_fs, "capture_duration_s": len(x) / fs,
                     "digital_probe_energy_full_scale_squared_seconds": float(probe @ probe / source_fs),
                     "repetitions": len(shots)}, "arrays": arrays}


def _group_checks(plan):
    groups = {}
    for e in plan["entries"]:
        key = (e["gather_id"], e["block"])
        signature = (e["source_id"], e["pan"], tuple(e["source_m_assumed"]), tuple(e["source_axis_assumed"]))
        if key in groups and groups[key] != signature:
            raise ValueError("gather changes source position or orientation within a block")
        groups[key] = signature


def _comparisons(captures, plan, manifests):
    pairs, by_id = {}, {q["capture_id"]: q for q in captures}
    # Preseed from the immutable plan, so two failed families never disappear.
    for e in plan["entries"]:
        if e["role"] != "probe_comparison":
            continue
        m = manifests[e["packet_id"]]
        key = (e["block"], e["source_id"], e["receiver_id"], e["volume"], m["comparison_group"])
        pairs.setdefault(key, []).append((by_id[e["capture_id"]], m))
    results = []
    for key, group in pairs.items():
        families = [m["probe"]["family"] for q, m in group]
        complete = sorted(families) == ["exponential", "linear"] and all(q["status"] == "accepted" for q, m in group)
        rows = []
        for q, m in group:
            if "qc" not in q:
                rows.append({"capture_id": q["capture_id"], "family": m["probe"]["family"],
                             "status": q["status"], "error": q.get("error"), "metrics_available": False})
                continue
            snr = np.array([[v["snr_db"] for v in s["band_snr_db"]] for s in q["qc"]["shots"]])
            rows.append({"capture_id": q["capture_id"], "family": q["probe"]["family"], "status": q["status"],
                         "mean_band_snr_db": snr.mean(axis=0).tolist(),
                         "repeat_coherence": q["qc"]["repeat_coherence"], "cost": q["cost"],
                         "actual_source_rms": q["probe"]["actual_rms"], "actual_source_peak": q["probe"]["actual_peak"]})
        results.append({"block": key[0], "source_id": key[1], "receiver_id": key[2], "volume": key[3],
                        "comparison_group": key[4], "complete_accepted_pair": complete, "captures": rows,
                        "winner": None, "interpretation": "descriptive paired measurements; no automatic waveform selection"})
    return results


def _session_integrity(root, plan, records, log_errors):
    reasons = []
    try:
        session = json.loads((root / "session.json").read_text())
        if session.get("schema") != "openpetcam_capture_v1":
            reasons.append("unsupported_session_schema")
        if session.get("status") != "captured_requires_qc_and_timing_calibration":
            reasons.append("session_did_not_complete_successfully")
        if session.get("completed") != len(plan["entries"]):
            reasons.append("session_completion_count_mismatch")
        if session.get("plan_sha256") != _sha(root / "plan.json") or session.get("suite_sha256") != _sha(root / "suite/suite.json"):
            reasons.append("session_asset_hash_mismatch")
        before = json.loads((root / "device_before.json").read_text())
        after = json.loads((root / "device_after.json").read_text())
        if not before or before != after:
            reasons.append("device_configuration_snapshot_mismatch")
    except (OSError, ValueError, TypeError) as exc:
        session = {}
        reasons.append("missing_or_invalid_session_metadata: " + str(exc))
    expected = {e["capture_id"] for e in plan["entries"]}
    if log_errors or set(records) != expected or any(len(v) != 1 for v in records.values()):
        reasons.append("capture_log_inventory_incomplete_or_inconsistent")
    return {"accepted": not reasons, "rejection_reasons": reasons,
            "recorded_session_status": session.get("status"),
            "configuration_evidence": "matching archived settings snapshots; not proof of runtime DSP state"}


def analyze_session(session_dir, output_dir, *, config=None, tail_s=.15, max_rate_error_ppm=500.):
    """Write inventory JSON and unstacked signed gather matrices to a new folder."""
    root, out = Path(session_dir).resolve(), Path(output_dir).resolve()
    config = config or QCConfig()
    if not np.isfinite([tail_s, max_rate_error_ppm]).all() or tail_s <= 0 or max_rate_error_ppm <= 0:
        raise ValueError("positive finite tail and rate bound required")
    plan, packet_assets = assets(root / "plan.json", root / "suite/suite.json")
    _group_checks(plan)
    packets = {key: value[1].parent / "packet.json" for key, value in packet_assets.items()}
    manifests = {key: value[0] for key, value in packet_assets.items()}
    records, log_errors = {}, []
    log_path = root / "captures.jsonl"
    for line_no, line in enumerate(log_path.read_text().splitlines() if log_path.exists() else [], 1):
        try:
            row = json.loads(line)
            records.setdefault(row["capture_id"], []).append(row)
        except (ValueError, KeyError, TypeError) as exc:
            log_errors.append({"line": line_no, "error": str(exc)})
    session_validation = _session_integrity(root, plan, records, log_errors)
    out.mkdir(parents=True, exist_ok=False)
    captures, arrays, gathers = [], {}, {}
    known = {e["capture_id"] for e in plan["entries"]}
    for entry in plan["entries"]:
        cid = entry["capture_id"]
        try:
            rows = records.get(cid, [])
            if len(rows) != 1:
                raise ValueError("missing capture" if not rows else "duplicate capture log entries")
            q = _analyze_capture(root, rows[0], entry, packets[entry["packet_id"]], config=config,
                                 tail_s=tail_s, max_rate_error_ppm=max_rate_error_ppm)
            q["signal_qc_accepted"] = q["status"] == "accepted"
            if q["signal_qc_accepted"] and not session_validation["accepted"]:
                q["status"] = "rejected_session"
            aa = q.pop("arrays")
            q["array_keys"] = {key: cid + "_" + key for key in aa}
            for name, value in aa.items():
                arrays[cid + "_" + name] = value
            key = (entry["gather_id"], entry["packet_id"], entry["volume"])
            g = gathers.setdefault(key, {"gather_id": entry["gather_id"], "packet_id": entry["packet_id"],
                                         "source_id": entry["source_id"], "volume": entry["volume"], "rows": []})
            for i, shot in enumerate(q["qc"]["shots"]):
                g["rows"].append({"capture_id": cid, "shot_id": shot["event_id"], "repeat_id": shot["repeat_id"],
                                  "receiver_id": entry["receiver_id"], "actual_pose": rows[0]["actual_pose_before"],
                                  "source_m_assumed": entry["source_m_assumed"], "receiver_m_assumed": entry["receiver_m_assumed"],
                                  "source_axis_assumed": entry["source_axis_assumed"], "receiver_axis_assumed": entry["receiver_axis_assumed"],
                                  "role": entry["role"], "capture_accepted": q["status"] == "accepted",
                                  "shot_accepted": shot["accepted"], "array_key": cid + "_responses", "array_row": i})
        except (ValueError, KeyError, TypeError, OSError) as exc:
            q = {"capture_id": cid, "status": "failed", "entry": entry, "error": f"{type(exc).__name__}: {exc}"}
        captures.append(q)
    for key, g in gathers.items():
        # Equal-length rows belong to one packet. Rejected rows remain visible
        # in the inventory; no fabricated zero traces fill missing receivers.
        rows = [r for r in g["rows"] if r["capture_accepted"] and r["shot_accepted"]]
        name = f"gather_{len([k for k in arrays if k.startswith('gather_')]):04d}"
        if rows:
            arrays[name] = np.stack([arrays[r["array_key"]][r["array_row"]] for r in rows])
            g["matrix_key"] = name
        g["matrix_rows"] = rows
        g["stacked"] = False
    np.savez_compressed(out / "shots.npz", **arrays)
    accepted = sum(q["status"] == "accepted" for q in captures)
    report = {"schema": "openpetcam_shot_gathers_v1", "session": str(root),
              "complete": session_validation["accepted"] and accepted == len(captures) and not log_errors and not (set(records) - known),
              "session_validation": session_validation,
              "accepted_captures": accepted, "planned_captures": len(captures), "captures": captures,
              "log_errors": log_errors, "unexpected_capture_ids": sorted(set(records) - known),
              "gathers": list(gathers.values()), "comparisons": _comparisons(captures, plan, manifests),
              "arrays_file": "shots.npz", "arrays_sha256": _sha(out / "shots.npz"),
              "analysis_config": {"tail_s": tail_s, "max_rate_error_ppm": max_rate_error_ppm,
                                  "pilot_search_uses_evaluated_shots": False},
              "versions": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__},
              "source_hashes": {str(p.name): _sha(p) for p in Path(__file__).parent.glob("*.py")},
              "plan_sha256": _sha(root / "plan.json"), "suite_sha256": _sha(root / "suite/suite.json"),
              "inversion_allowed": False, "physical_dac_timing_calibrated": False,
              "limitations": ["FIFO pilot origins are local acoustic references and can remove cross-pose propagation phase.",
                              "A paired SDK reference preserves its checked sample grid; physical DAC timing still requires calibration.",
                              "Source/receiver acoustic centres and directional responses remain assumptions.",
                              "No per-shot normalization, room-waveform alignment, pre-QC stacking, or automatic winner selection.",
                              "Per-band SNR uses total signal-window power divided by pre-pilot noise power.",
                              "Negative-lag responses are retained; ordinary spectral energy is not a nonlinear harmonic diagnosis."]}
    _json(out / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session", type=Path)
    parser.add_argument("output", type=Path, help="new analysis directory")
    parser.add_argument("--max-rate-error-ppm", type=float, default=500.)
    args = parser.parse_args()
    result = analyze_session(args.session, args.output, max_rate_error_ppm=args.max_rate_error_ppm)
    print(json.dumps({"accepted": result["accepted_captures"], "planned": result["planned_captures"],
                      "complete": result["complete"], "inversion_allowed": False}))


if __name__ == "__main__":
    main()
