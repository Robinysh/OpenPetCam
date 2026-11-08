"""Describe noisy-room volume trials without changing source timing or scale.

Input is a verified gather analysis, not a new room-geometry inversion. The
commanded volume, observed SDK-reference gain, absolute microphone level and
reference-normalized response remain distinct quantities throughout.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .shot_qc import mapped_window


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _db_ratio(value, reference):
    return float(20 * np.log10(value / reference)) if value > 0 and reference > 0 else None


def _summary(values):
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)], float)
    return {"count": len(a), "median": float(np.median(a)) if len(a) else None,
            "min": float(a.min()) if len(a) else None, "max": float(a.max()) if len(a) else None}


def validate_volume_plan(plan):
    if plan.get("mode") != "volume" or not plan.get("entries"):
        raise ValueError("a nonempty volume acquisition plan is required")
    entries = plan["entries"]
    levels = sorted(plan.get("volumes", []))
    reference = plan.get("reference_volume", 30)
    if not levels or len(set(levels)) != len(levels) or reference not in levels:
        raise ValueError("unique levels including the reference volume required")
    fixed = {(e["packet_id"], e["source_id"], e["receiver_id"], e["pan"], e["tilt"]) for e in entries}
    if len(fixed) != 1 or len({e["capture_id"] for e in entries}) != len(entries):
        raise ValueError("volume trial must retain one exact packet and source/receiver pose")
    blocks = sorted({e["block"] for e in entries})
    for b in blocks:
        ee = [e for e in entries if e["block"] == b]
        if (ee[0]["role"] != "reference_before" or ee[-1]["role"] != "reference_after"
                or ee[0]["volume"] != reference or ee[-1]["volume"] != reference
                or any(e["role"] != "volume_trial" for e in ee[1:-1])
                or sorted(e["volume"] for e in ee[1:-1]) != levels):
            raise ValueError("every block needs one trial per level and bracketing reference-volume anchors")
    return levels, reference, blocks


def measure_capture(capture, arrays, *, response_window_s=(0., .015)):
    """Measure already verified arrays with one frozen time window and no alignment."""
    q = capture
    proof = q.get("reference_proof") or {}
    check = proof.get("active_ao_reference_check") or {}
    if (q.get("status") != "accepted" or not proof.get("independently_redecoded")
            or not proof.get("source_packet_bit_identical") or not check.get("verified")
            or q.get("timing_status") != "sdk_paired_reference_continuity_checked"):
        raise ValueError("accepted framed, actively verified SDK-reference capture required")
    keys = q["array_keys"]
    raw, reference, h, time = (np.asarray(arrays[keys[k]], float) for k in ("raw", "reference", "responses", "time_s"))
    ranges = np.asarray(arrays[keys["adc_sample_ranges"]], int)
    shots = q["qc"]["shots"]
    if (raw.ndim != 2 or reference.shape != raw.shape or h.ndim != 2 or h.shape[0] != raw.shape[0]
            or time.shape != (h.shape[1],) or len(shots) != len(raw) or ranges.shape != (len(raw), 2)
            or not all(np.isfinite(v).all() for v in (raw, reference, h, time))):
        raise ValueError("invalid or inconsistent shot array inventory")
    fs = q["qc"]["sample_rate_hz"]
    if fs != 48000 or len(time) < 2 or not np.allclose(np.diff(time), 1 / fs):
        raise ValueError("expected unchanged 48 kHz response time coordinates")
    keep = (time >= response_window_s[0]) & (time <= response_window_s[1])
    if keep.sum() < 32:
        raise ValueError("fixed response comparison window is too short")
    gain = float(check["global_gain"])
    if not np.isfinite(gain) or gain == 0:
        raise ValueError("invalid observed SDK reference gain")
    n = round(q["probe"]["samples"] * q["timing_map"]["slope"])
    rows = []
    for j, shot in enumerate(shots):
        if not shot["accepted"]:
            raise ValueError("rejected shot cannot enter a volume estimate")
        origin = q["timing_map"]["intercept"] + q["timing_map"]["slope"] * shot["source_start_sample"]
        start = origin - ranges[j, 0]
        mic = mapped_window(raw[j], start, n)
        ref = mapped_window(reference[j], start, n)
        mic_rms, ref_rms = float(np.sqrt(np.mean(mic ** 2))), float(np.sqrt(np.mean(ref ** 2)))
        response_norm = float(np.linalg.norm(h[j, keep]))
        rows.append({"shot_id": shot["event_id"], "mic_probe_window_rms": mic_rms,
                     "sdk_reference_probe_window_rms": ref_rms,
                     "noise_rms": shot["noise_rms"], "normalized_correlation": shot["normalized_correlation"],
                     "nominal_source_matched_amplitude": shot["matched_filter_amplitude"],
                     "clipped_samples": shot["clipped_samples"],
                     "reference_normalized_response_norm": response_norm,
                     "command_source_relative_response_norm": response_norm * abs(gain),
                     "mic_to_sdk_reference_total_rms_ratio": mic_rms / ref_rms if ref_rms else None,
                     "band_snr_db": shot["band_snr_db"]})
    def mean(key):
        return float(np.mean([r[key] for r in rows]))
    snr = np.array([[v["snr_db"] for v in r["band_snr_db"]] for r in rows])
    spectrum_bands = [v["band_hz"] for v in rows[0]["band_snr_db"]]
    metric = {key: mean(key) for key in ("mic_probe_window_rms", "sdk_reference_probe_window_rms", "noise_rms",
                                        "normalized_correlation", "nominal_source_matched_amplitude",
                                        "reference_normalized_response_norm", "command_source_relative_response_norm")}
    coherent = h[:, keep].mean(axis=0)
    coherent_norm = float(np.linalg.norm(coherent))
    residual_norm = float(np.sqrt(np.mean(np.sum((h[:, keep] - coherent) ** 2, axis=1))))
    metric.update(sdk_reference_gain_from_submitted_ao=gain, mean_band_snr_db=snr.mean(axis=0).tolist(),
                  band_hz=spectrum_bands, clipped_samples=sum(r["clipped_samples"] for r in rows),
                  repeat_coherence=q["qc"].get("repeat_coherence"), shots=rows,
                  coherent_reference_normalized_response_norm=coherent_norm,
                  coherent_command_source_relative_response_norm=coherent_norm * abs(gain),
                  repeat_response_residual_norm=residual_norm,
                  repeat_residual_to_coherent_norm=residual_norm / coherent_norm if coherent_norm else None,
                  mean_mic_band_rms=[float(np.mean([np.sqrt(r["band_snr_db"][i]["total_signal_power"]) for r in rows])) for i in range(len(spectrum_bands))],
                  mean_noise_band_rms=[float(np.mean([np.sqrt(r["band_snr_db"][i]["noise_power"]) for r in rows])) for i in range(len(spectrum_bands))],
                  response_window_s=list(response_window_s), response_trace_mean=coherent.tolist())
    return metric


def summarize_volume(plan, captures):
    """Pair randomized volume trials within blocks; anchors diagnose drift."""
    levels, reference, blocks = validate_volume_plan(plan)
    by_id = {q["capture_id"]: q for q in captures}
    if set(by_id) != {e["capture_id"] for e in plan["entries"]} or len(by_id) != len(captures):
        raise ValueError("volume analysis inventory must include every planned capture exactly once")
    metrics = ("mic_probe_window_rms", "sdk_reference_probe_window_rms", "noise_rms", "normalized_correlation",
               "nominal_source_matched_amplitude", "reference_normalized_response_norm", "command_source_relative_response_norm",
               "coherent_reference_normalized_response_norm", "coherent_command_source_relative_response_norm",
               "repeat_response_residual_norm", "sdk_reference_gain_from_submitted_ao")
    level_rows = []
    for level in levels:
        all_rows = [q for q in captures if q["entry"]["role"] == "volume_trial" and q["entry"]["volume"] == level]
        accepted = [q for q in all_rows if q.get("usable")]
        row = {"commanded_volume": level, "planned_trials": len(all_rows), "usable_trials": len(accepted),
               "excluded_capture_ids": [q["capture_id"] for q in all_rows if not q.get("usable")],
               "metrics": {key: _summary([q["metrics"][key] for q in accepted]) for key in metrics}}
        row["mean_band_snr_db"] = [_summary([q["metrics"]["mean_band_snr_db"][i] for q in accepted])
                                   for i in range(len(accepted[0]["metrics"]["mean_band_snr_db"]))] if accepted else []
        level_rows.append(row)
    paired, anchors = [], []
    for b in blocks:
        entries = [e for e in plan["entries"] if e["block"] == b]
        trials = {e["volume"]: by_id[e["capture_id"]] for e in entries if e["role"] == "volume_trial"}
        base = trials[reference]
        for level in levels:
            q = trials[level]
            row = {"block": b, "volume": level, "reference_volume": reference,
                   "capture_id": q["capture_id"], "reference_capture_id": base["capture_id"],
                   "usable_pair": bool(q.get("usable") and base.get("usable"))}
            if row["usable_pair"]:
                a, z = q["metrics"], base["metrics"]
                row["changes_db"] = {key: _db_ratio(abs(a[key]), abs(z[key])) for key in metrics if key != "normalized_correlation"}
                row["band_snr_change_db"] = (np.asarray(a["mean_band_snr_db"]) - z["mean_band_snr_db"]).tolist()
            paired.append(row)
        before, after = by_id[entries[0]["capture_id"]], by_id[entries[-1]["capture_id"]]
        row = {"block": b, "volume": reference, "before_capture_id": before["capture_id"],
               "after_capture_id": after["capture_id"], "usable_anchors": bool(before.get("usable") and after.get("usable"))}
        if row["usable_anchors"]:
            a, z = after["metrics"], before["metrics"]
            row["after_vs_before_db"] = {key: _db_ratio(abs(a[key]), abs(z[key])) for key in metrics if key != "normalized_correlation"}
            x, y = np.asarray(z["response_trace_mean"]), np.asarray(a["response_trace_mean"])
            den = np.linalg.norm(x) * np.linalg.norm(y)
            row["signed_response_correlation_without_alignment"] = float(x @ y / den) if den else None
            drift = row["after_vs_before_db"]["coherent_command_source_relative_response_norm"]
            row["activity_or_drift_flag"] = bool((drift is not None and abs(drift) > 1.) or den == 0 or x @ y / den < .95)
            row["flag_rule"] = "descriptive: >1 dB response-norm change or signed correlation <0.95; not a confidence interval"
        anchors.append(row)
    return {"levels": level_rows, "within_block_pairs": paired, "reference_anchor_drift": anchors,
            "reference_volume": reference, "winner": None,
            "interpretation": "Noisy-room descriptive volume response; ambient activity and time drift can confound differences."}


def _plot(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    valid = [q for q in report["captures"] if q.get("usable")]
    for block in sorted({q["entry"]["block"] for q in valid}):
        trial = sorted([q for q in valid if q["entry"]["block"] == block and q["entry"]["role"] == "volume_trial"], key=lambda q: q["entry"]["volume"])
        levels = [q["entry"]["volume"] for q in trial]
        for key, style, label in (("mic_probe_window_rms", "o-", "mic"), ("sdk_reference_probe_window_rms", "x--", "SDK reference")):
            means = np.array([20 * np.log10(max(q["metrics"][key], 1e-12)) for q in trial])
            lows = np.array([min(20 * np.log10(max(s[key], 1e-12)) for s in q["metrics"]["shots"]) for q in trial])
            highs = np.array([max(20 * np.log10(max(s[key], 1e-12)) for s in q["metrics"]["shots"]) for q in trial])
            axs[0, 0].errorbar(levels, means, yerr=np.maximum(np.array([means - lows, highs - means]), 0), fmt=style, alpha=.65, capsize=2, label=f"{label}, block {block}")
        pp = [q for q in report["within_block_pairs"] if q["block"] == block and q["usable_pair"]]
        if pp:
            axs[0, 1].plot([q["volume"] for q in pp], [q["changes_db"]["coherent_command_source_relative_response_norm"] for q in pp], "o-", label=f"block {block}")
    for i in range(3):
        rows = [q for q in report["levels"] if len(q["mean_band_snr_db"]) > i]
        if rows:
            median = np.array([q["mean_band_snr_db"][i]["median"] for q in rows])
            low = np.array([q["mean_band_snr_db"][i]["min"] for q in rows])
            high = np.array([q["mean_band_snr_db"][i]["max"] for q in rows])
            band = valid[0]["metrics"]["band_hz"][i]
            axs[1, 0].errorbar([q["commanded_volume"] for q in rows], median,
                               yerr=np.array([median - low, high - median]), fmt="o-", capsize=3,
                               label=f"{band[0] / 1000:g}–{band[1] / 1000:g} kHz")
    axs[1, 1].plot([q["acquisition_order"] for q in valid], [20 * np.log10(max(q["metrics"]["noise_rms"], 1e-12)) for q in valid], "o-", color="tab:orange", label="accepted capture noise")
    rejected = [q for q in report["captures"] if not q["usable"]]
    for q in rejected:
        noise = q.get("qc_diagnostics", {}).get("mean_noise_rms")
        if noise is not None:
            axs[1, 1].plot(q["acquisition_order"], 20 * np.log10(max(noise, 1e-12)), "rx", markersize=8)
        else:
            axs[1, 1].axvline(q["acquisition_order"], color="red", alpha=.25)
    if rejected:
        axs[1, 1].text(.02, .97, f"{len(rejected)} rejected captures marked in red", transform=axs[1, 1].transAxes, va="top", fontsize=8)
    anchor_rows = [q for q in valid if q["entry"]["role"] in ("reference_before", "reference_after")]
    if anchor_rows:
        axs[1, 1].scatter([q["acquisition_order"] for q in anchor_rows],
                          [20 * np.log10(max(q["metrics"]["noise_rms"], 1e-12)) for q in anchor_rows],
                          marker="s", s=45, facecolors="none", edgecolors="tab:blue", label="reference-volume anchors")
    by_id = {q["capture_id"]: q for q in valid}
    for pair in report["reference_anchor_drift"]:
        if pair["usable_anchors"]:
            q = by_id[pair["after_capture_id"]]
            drift = pair["after_vs_before_db"]["coherent_command_source_relative_response_norm"]
            if drift is not None:
                axs[1, 1].annotate(f"ΔH {drift:+.1f} dB", (q["acquisition_order"], 20 * np.log10(max(q["metrics"]["noise_rms"], 1e-12))),
                                  xytext=(0, -15), textcoords="offset points", ha="center", fontsize=7)
    for ax, title, xlabel, ylabel in (
            (axs[0, 0], "Absolute digital levels", "Commanded volume", "RMS dBFS"),
            (axs[0, 1], "Coherent response relative to reference trial", "Commanded volume", "AO-relative response change (dB)"),
            (axs[1, 0], "Observed signal / pre-pilot noise", "Commanded volume", "Band SNR (dB)"),
            (axs[1, 1], "Noise through acquisition", "Capture order", "Noise RMS dBFS")):
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel); ax.grid(alpha=.2)
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=7)
    fig.suptitle("Noisy-room volume experiment — digital levels; bars show repeat/block ranges, not confidence intervals")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def analyze_volume(gather_report, output_dir, *, plot=False, response_window_s=(0., .015)):
    report_path = Path(gather_report).resolve()
    if report_path.is_dir():
        report_path /= "report.json"
    gathered = json.loads(report_path.read_text())
    if gathered.get("schema") != "openpetcam_shot_gathers_v1":
        raise ValueError("verified gather analysis required")
    session = Path(gathered["session"])
    plan_path = session / "plan.json"
    if _sha(plan_path) != gathered["plan_sha256"]:
        raise ValueError("volume plan changed since gather analysis")
    plan = json.loads(plan_path.read_text())
    validate_volume_plan(plan)
    filename = gathered["arrays_file"]
    if Path(filename).name != filename:
        raise ValueError("shot array file must be a local basename")
    array_path = report_path.parent / filename
    if _sha(array_path) != gathered["arrays_sha256"]:
        raise ValueError("shot array hash mismatch")
    entries = {e["capture_id"]: e for e in plan["entries"]}
    captures = []
    with np.load(array_path, allow_pickle=False) as arrays:
        for order, q in enumerate(gathered["captures"]):
            cid = q["capture_id"]
            if entries.get(cid) != q.get("entry"):
                raise ValueError("capture identity differs from frozen volume plan")
            row = {"capture_id": cid, "entry": q["entry"], "acquisition_order": order,
                   "gather_status": q["status"], "usable": False}
            if q.get("qc", {}).get("shots"):
                noise = [s["noise_rms"] for s in q["qc"]["shots"]]
                row["qc_diagnostics"] = {"mean_noise_rms": float(np.mean(noise)),
                                         "clipped_samples": q["qc"].get("clipped_samples"),
                                         "rejection_reasons": q["qc"].get("rejection_reasons", [])}
            try:
                if not gathered.get("session_validation", {}).get("accepted"):
                    raise ValueError("capture session provenance/configuration did not pass")
                row["metrics"] = measure_capture(q, arrays, response_window_s=response_window_s)
                row["usable"] = True
            except (ValueError, KeyError, TypeError) as exc:
                row["exclusion_reason"] = str(exc)
            captures.append(row)
    summary = summarize_volume(plan, captures)
    result = {"schema": "openpetcam_volume_analysis_v1", **summary, "captures": captures,
              "complete_usable": all(q["usable"] for q in captures), "source_report_sha256": _sha(report_path),
              "source_arrays_sha256": _sha(array_path), "plan_sha256": _sha(plan_path),
              "analysis_source_sha256": _sha(Path(__file__)), "response_window_s": list(response_window_s),
              "normalization": "none before measurement; mic/SDK-reference transfer and AO-relative response reported separately",
              "alignment": "source timing from the existing verified map; no room-response or per-shot peak realignment",
              "inversion_allowed": False, "calibrated_spl": False,
              "limitations": ["Commanded volume is a device setting, not a measured voltage gain or calibrated SPL.",
                              "SDK-reference gain is observed digital gain from submitted AO; microphone/reference transfer can still change after that reference tap.",
                              "Raw mic RMS contains probe, room response and ambient noise; total RMS ratios are not isolated acoustic gain.",
                              "Reference-normalized responses divide by actual SDK output. AO-relative response additionally retains observed SDK-reference gain.",
                              "Individual response norms include ambient/deconvolution noise and can be biased upward at low volume. Signed repeat-mean norms reduce only uncorrelated variation; repeat residuals and shared activity remain relevant.",
                              "TV, animal sounds, motion and changing room activity can confound level/SNR differences; anchors diagnose drift without removing it.",
                              "These measurements do not identify THD, AGC operation, amplifier gain staging, quiet-room geometry or a best volume.",
                              "Level summaries count randomized trials only; volume-30 anchors are separate and cannot inflate its sample size."]}
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=False)
    if plot:
        _plot(result, out / "volume.png")
        result["plot_file"] = "volume.png"
    (out / "report.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("gather_report", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--plot", action="store_true")
    a = p.parse_args()
    r = analyze_volume(a.gather_report, a.output, plot=a.plot)
    print(json.dumps({"usable": sum(q["usable"] for q in r["captures"]), "planned": len(r["captures"]),
                      "complete_usable": r["complete_usable"], "inversion_allowed": False}))


if __name__ == "__main__":
    main()
