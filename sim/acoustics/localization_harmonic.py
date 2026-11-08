"""Exploratory observed-power harmonic-line attenuation on DEVELOPMENT data.

The gate imposes a source-comb prior. A narrow room-mode comb can have the same
power spectrum, so this procedure cannot identify physical source origin. Only
derived spectra change; no original signed recording or waveform is modified.
"""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np

from .localization import errors, rank_candidates, ripple_features, spatial_partition
from .localization_resolution import LOW, REPEATS, describe, prefix_mean_powers, sha, write_json


DETECTOR = {"peak_band_hz": [120., 3000.], "f0_prior_hz": [140., 280.], "f0_step_hz": .5,
            "sideband_inner_hz": 12., "sideband_outer_hz": 35.,
            "min_peak_to_sideband_db": 8., "max_linear_power_fwhm_hz": 14.,
            "max_harmonic_order": 12, "match_tolerance_hz": 4.5,
            "minimum_matches": 6, "minimum_expected_fraction": .6,
            "minimum_harmonic_order_span": 6, "minimum_median_prominence_db": 10.,
            "require_first_or_second_harmonic": True, "mask_half_width_hz": 12.}


def _spectrum(power, frequency):
    p, f = np.asarray(power, float), np.asarray(frequency, float)
    if (p.ndim != 1 or f.shape != p.shape or len(f) < 128 or not np.isfinite(p).all()
            or np.any(p < 0) or not np.isfinite(f).all() or f[0] < 0):
        raise ValueError("finite nonnegative one-dimensional powers/frequencies required")
    if f[1] <= f[0] or not np.allclose(np.diff(f), f[1] - f[0]):
        raise ValueError("uniform increasing physical frequency grid required")
    return p, f


def narrow_peaks(power, frequency):
    """Peak prominence uses nearby power; width is linear-power excess FWHM."""
    p, f = _spectrum(power, frequency)
    if not np.max(p):
        return []
    df = f[1] - f[0]
    inner = max(1, math.ceil(DETECTOR["sideband_inner_hz"] / df))
    outer = max(inner + 1, math.floor(DETECTOR["sideband_outer_hz"] / df))
    indexes = np.flatnonzero((p[1:-1] > p[:-2]) & (p[1:-1] >= p[2:])) + 1
    indexes = indexes[(f[indexes] >= DETECTOR["peak_band_hz"][0])
                      & (f[indexes] <= DETECTOR["peak_band_hz"][1])
                      & (indexes >= outer) & (indexes + outer < len(p))]
    floor = max(float(p.max()) * 1e-12, 1e-30)
    result = []
    for index in indexes:
        nearby = np.r_[p[index - outer:index - inner + 1], p[index + inner:index + outer + 1]]
        baseline = max(float(np.median(nearby)), floor)
        prominence = 10 * math.log10(max(float(p[index]), floor) / baseline)
        if prominence < DETECTOR["min_peak_to_sideband_db"]:
            continue
        level = baseline + .5 * (p[index] - baseline)
        left = right = int(index)
        max_width = DETECTOR["max_linear_power_fwhm_hz"]
        while left > 0 and p[left] > level and f[index] - f[left] <= max_width:
            left -= 1
        while right < len(p) - 1 and p[right] > level and f[right] - f[index] <= max_width:
            right += 1
        if p[left] > level or p[right] > level:
            continue
        crossing_left = f[left] + df * (level - p[left]) / (p[left + 1] - p[left])
        crossing_right = f[right - 1] + df * (level - p[right - 1]) / (p[right] - p[right - 1])
        width = float(crossing_right - crossing_left)
        if 0 < width <= max_width:
            result.append({"bin": int(index), "frequency_hz": float(f[index]),
                           "prominence_db": float(prominence), "linear_power_fwhm_hz": width})
    return result


def detect_comb(power, frequency):
    """No scenario labels or true fundamental enter this observed-power gate."""
    peaks = narrow_peaks(power, frequency)
    empty = {"gated": False, "estimated_f0_hz": None, "narrow_peaks": len(peaks),
             "matched_harmonics": [], "matched_count": 0, "expected_count": 0,
             "reason": "Insufficient observed narrow harmonic support under frozen prior."}
    if len(peaks) < DETECTOR["minimum_matches"]:
        return empty
    peak_f = np.array([peak["frequency_hz"] for peak in peaks])
    prominence = np.array([peak["prominence_db"] for peak in peaks])
    low, high = DETECTOR["f0_prior_hz"]
    candidates = np.arange(low, high + DETECTOR["f0_step_hz"] / 2, DETECTOR["f0_step_hz"])
    orders = np.arange(1, DETECTOR["max_harmonic_order"] + 1)
    expected_f = candidates[:, None] * orders
    available = (expected_f >= DETECTOR["peak_band_hz"][0]) & (expected_f <= DETECTOR["peak_band_hz"][1])
    distance = abs(expected_f[:, :, None] - peak_f)
    nearest = distance.argmin(axis=-1)
    difference = np.take_along_axis(distance, nearest[:, :, None], axis=-1)[:, :, 0]
    matched = available & (difference <= DETECTOR["match_tolerance_hz"])
    counts, expected_counts = matched.sum(axis=1), available.sum(axis=1)
    spans = np.max(np.where(matched, orders, 0), axis=1) - np.min(np.where(matched, orders, 99), axis=1)
    eligible = ((counts >= DETECTOR["minimum_matches"])
                & (counts >= DETECTOR["minimum_expected_fraction"] * expected_counts)
                & (spans >= DETECTOR["minimum_harmonic_order_span"]))
    if DETECTOR["require_first_or_second_harmonic"]:
        eligible &= matched[:, 0] | matched[:, 1]
    for index in np.flatnonzero(eligible):
        if np.median(prominence[nearest[index, matched[index]]]) < DETECTOR["minimum_median_prominence_db"]:
            eligible[index] = False
    if not eligible.any():
        return empty
    # Count dominates; clipped prominence then frequency residual break ties.
    quality = (counts * 1000 + np.sum(np.where(matched, np.minimum(prominence[nearest], 40.), 0), axis=1)
               - np.sum(np.where(matched, difference, 0), axis=1))
    quality[~eligible] = -np.inf
    best = int(np.argmax(quality))
    selected = []
    for order_index in np.flatnonzero(matched[best]):
        selected.append(dict(peaks[int(nearest[best, order_index])], harmonic_order=int(orders[order_index])))
    estimate = float(np.median([peak["frequency_hz"] / peak["harmonic_order"] for peak in selected]))
    return {"gated": True, "estimated_f0_hz": estimate, "search_grid_f0_hz": float(candidates[best]),
            "narrow_peaks": len(peaks), "matched_count": len(selected),
            "expected_count": int(expected_counts[best]), "matched_harmonics": selected,
            "reason": "Observed narrow peaks fit a preregistered harmonic source prior; origin remains unidentified."}


def attenuate_lines(power, frequency):
    """Interpolate derived log power only inside matched narrow-peak windows.

    Fixed ±12Hz anchors can contain source wings. This is narrow-line attenuation,
    never recovered room transfer or an identified physical source spectrum.
    """
    p, f = _spectrum(power, frequency)
    diagnostic = detect_comb(p, f)
    mask = np.zeros(len(p), bool)
    if diagnostic["gated"]:
        for peak in diagnostic["matched_harmonics"]:
            mask |= abs(f - peak["frequency_hz"]) <= DETECTOR["mask_half_width_hz"]
    result = p.copy()
    if mask.any():
        floor = max(float(p.max()) * 1e-12, 1e-30)
        logp = np.log(np.maximum(p, floor))
        result[mask] = np.exp(np.interp(f[mask], f[~mask], logp[~mask]))
    band = (f >= LOW["band"][0]) & (f <= LOW["band"][1])
    diagnostic.update(masked_bins=int(mask.sum()), total_bins=len(p),
                      low_band_masked_fraction=float(mask[band].mean()),
                      low_band_power_ratio=float(result[band].sum() / p[band].sum()) if p[band].sum() else None)
    return result, diagnostic, mask


def population(rows):
    source = [row for row in rows if row["source_m"] is not None]
    background = [row for row in rows if row["source_m"] is None]
    return {"source_cases": len(source), "all_source_cases": describe(source) if source else {"cases": 0},
            "background_cases": len(background),
            "background_conditional_candidates": sum(row["best_index"] is not None for row in background),
            "cases_with_gated_burst": sum(row["gated_bursts"] > 0 for row in rows),
            "mean_gated_bursts": float(np.mean([row["gated_bursts"] for row in rows])),
            "mean_low_band_masked_fraction": float(np.mean([row["mean_low_band_masked_fraction"] for row in rows]))}


def run(development_dir, twoheight_dir, benchmark_dir, output_dir, *, deadline_s=590):
    if not 10 <= deadline_s <= 590:
        raise ValueError("cooperative deadline must be10..590seconds")
    development, twoheight, benchmark, output = map(lambda value: Path(value).resolve(),
        (development_dir, twoheight_dir, benchmark_dir, output_dir))
    source_protocol = json.loads((development / "protocol.json").read_text())
    frozen = json.loads((development / "frozen.json").read_text())
    cases_file, powers_file = development / "cases.json", development / "burst_powers.npz"
    if sha(cases_file) != frozen["cases_sha256"] or sha(development / "protocol.json") != frozen["protocol_sha256"]:
        raise ValueError("frozen development cases/protocol changed")
    cases = json.loads(cases_file.read_text())
    sources = [case for case in cases if case["source_m"] is not None]
    if any(case["split"] != "development" for case in cases) or set(spatial_partition([c["source_m"] for c in sources])) != {"development"}:
        raise ValueError("all queries must remain in development spatial tiles")
    reference_protocol = json.loads((twoheight / "protocol.json").read_text())
    baseline_file = twoheight / "rankings.jsonl"
    dictionaries = {.5: benchmark / "dictionary.npz", .125: twoheight / "dictionary_twoheight_0.125m.npz"}
    if sha(dictionaries[.5]) != source_protocol["dictionary_sha256"]:
        raise ValueError("coarse dictionary differs from development source dictionary")
    dictionary_frozen = json.loads((twoheight / "dictionary_frozen.json").read_text())
    if sha(dictionaries[.125]) != dictionary_frozen["sha256"]:
        raise ValueError("fine dictionary differs from its frozen baseline version")
    if (reference_protocol["input_hashes"]["development_cases"] != sha(cases_file)
            or reference_protocol["input_hashes"]["burst_powers"] != sha(powers_file)):
        raise ValueError("reused baseline did not use the same frozen cases and burst powers")
    output.mkdir(parents=True, exist_ok=False)
    source_dir = output / "source"; source_dir.mkdir()
    dependencies = [Path(__file__)] + [Path(__file__).with_name(name) for name in
                                     ("localization.py", "localization_resolution.py")]
    for path in dependencies:
        shutil.copyfile(path, source_dir / path.name)
    protocol = {"schema": "openpetcam.localization-harmonic-development.v1", "created_epoch": time.time(),
        "classification": "EXPLORATORY DEVELOPMENT SOURCE-PRIOR HYPOTHESIS ONLY",
        "source_sha256": {path.name: sha(path) for path in dependencies},
        "input_hashes": {"cases": sha(cases_file), "burst_powers": sha(powers_file),
                         "development_protocol": sha(development / "protocol.json"),
                         "reference_protocol": sha(twoheight / "protocol.json"), "baseline_rankings": sha(baseline_file),
                         "dictionaries": {str(spacing): sha(path) for spacing, path in dictionaries.items()}},
        "cases": len(cases), "emitting_source_cases": len(sources), "background_cases": len(cases) - len(sources),
        "paired_groups": len(set(case["paired_group"] for case in cases)), "chosen_case_ids": [c["case_id"] for c in cases],
        "families": sorted(set(c["scenario"] for c in cases)), "spacings_m": [.5, .125], "repeats": REPEATS,
        "band": LOW, "detector": DETECTOR, "receiver_m": reference_protocol["receiver_m"],
        "gating": "Identical observed-power rule for every burst/case; no scenario labels or true F0 enter detector/mask.",
        "operation": "Interpolate derived log power only in fixed narrow windows around observed matched peaks; mean corrected LINEAR powers across same nested1/4/16bursts before frozen ripple.",
        "baseline": "Reuse exact frozen two-height DEVELOPMENT baseline rows on same cases/power prefixes/dictionaries; no v1held-out results read.",
        "acceptance": "Alwaysfalse; no probability/confidence calibration.", "deadline_s": deadline_s,
        "frozen_before_spectral_analysis": True,
        "limits": ["Source prior only: an identical room/cavity comb is indistinguishable from a source comb in one power spectrum.",
            "Current first-order simulator lacks high-Q room modes; low false gates there do not establish discrimination in real rooms.",
            "Fixed12Hz interpolation anchors can retain source wings; corrected spectra are not recovered room transfer.",
            "F0prior140..280Hz comes from declared synthetic source family, not validated dog-bark coverage.",
            "Missing fundamentals, aliases and room notches can prevent or misdirect detection.",
            "All comparisons reuse development tiles; no fresh confirmation, real accuracy or signed-waveform modification.",
            "No v1held-out labels/predictions, summary or selected-rule files opened; no source/IR oracle used.",
            "No source truth/error assigned to background; unlocalized failures remain in source denominators.",
            "Cases share32pairedgroups; no unclustered confidence intervals."]}
    write_json(output / "protocol.json", protocol)
    started = time.monotonic()
    def progress(phase, done, total):
        status = {"pid": os.getpid(), "phase": phase, "done": done, "total": total,
                  "elapsed_s": time.monotonic() - started, "updated_epoch": time.time()}
        temporary = output / ".progress.tmp"; temporary.write_text(json.dumps(status, indent=2)); temporary.replace(output / "progress.json")
        if status["elapsed_s"] > deadline_s:
            raise TimeoutError("harmonic development deadline reached; partial results preserved")
    with np.load(powers_file, allow_pickle=False) as archive:
        powers, frequency = archive["power"], archive["frequency_hz"]
    if powers.shape != (len(cases), 16, len(frequency)):
        raise ValueError("power archive does not match case inventory and sixteen bursts")
    baseline = {}
    with baseline_file.open() as stream:
        for line in stream:
            row = json.loads(line)
            baseline[(row["case_id"], row["spacing_m"], row["repeats"])] = row
    if len(baseline) != len(cases) * len(dictionaries) * len(REPEATS):
        raise ValueError("reused baseline is incomplete")
    query_features, case_diagnostics = {}, {}
    arguments = {key: value for key, value in LOW.items() if key != "name"}
    with (output / "burst_diagnostics.jsonl").open("x") as diagnostics:
        for index, case in enumerate(cases):
            corrected, details = [], []
            for burst_index, power in enumerate(powers[index]):
                changed, diagnostic, _ = attenuate_lines(power, frequency)
                corrected.append(changed); details.append(diagnostic)
                diagnostics.write(json.dumps({"case_id": case["case_id"], "burst_index": burst_index, **diagnostic}, allow_nan=False) + "\n")
            for repeats, power in prefix_mean_powers(np.asarray(corrected)).items():
                query_features[(case["case_id"], repeats)] = ripple_features(power, frequency, **arguments)[0]
                case_diagnostics[(case["case_id"], repeats)] = {
                    "gated_bursts": sum(detail["gated"] for detail in details[:repeats]),
                    "mean_low_band_masked_fraction": float(np.mean([detail["low_band_masked_fraction"] for detail in details[:repeats]]))}
            if index % 20 == 0:
                diagnostics.flush(); progress("observed_power_detector", index + 1, len(cases))
    del powers
    receiver = np.asarray(protocol["receiver_m"])
    rows = []
    with (output / "rankings.jsonl").open("x") as stream:
        for spacing, dictionary_file in dictionaries.items():
            with np.load(dictionary_file, allow_pickle=False) as archive:
                positions = archive["positions_m"]
                features = archive["low"].astype(np.float32) if spacing == .5 else archive["features"]
            for index, case in enumerate(cases):
                truth = np.asarray(case["source_m"]) if case["source_m"] is not None else None
                for repeats in REPEATS:
                    key = (case["case_id"], spacing, repeats)
                    raw = dict(baseline[key], method="unmodified_reused_baseline", **case_diagnostics[(case["case_id"], repeats)])
                    if raw["source_m"] != case["source_m"] or raw["scenario"] != case["scenario"]:
                        raise ValueError("baseline case identity mismatch")
                    rows.append(raw); stream.write(json.dumps(raw, allow_nan=False) + "\n")
                    rank = rank_candidates(query_features[(case["case_id"], repeats)], features, positions)
                    rank.pop("scores"); rank["near_best_count"] = len(rank.pop("near_best_candidates", []))
                    error = {"distance_error_m": None, "azimuth_error_deg": None, "horizontal_error_m": None,
                             "height_error_m": None, "range_error_m": None}
                    if rank["position_m"] is not None and truth is not None:
                        predicted = np.asarray(rank["position_m"])
                        error.update(errors(predicted, truth, receiver),
                            horizontal_error_m=float(np.linalg.norm((predicted - truth)[[0, 2]])),
                            height_error_m=float(abs(predicted[1] - truth[1])),
                            range_error_m=float(abs(np.linalg.norm(predicted - receiver) - np.linalg.norm(truth - receiver))))
                    changed = {"case_id": case["case_id"], "paired_group": case["paired_group"], "scenario": case["scenario"],
                        "source_m": case["source_m"], "split": "development", "spacing_m": spacing, "repeats": repeats,
                        "candidate_count": len(positions), "method": "observed_narrow_line_attenuation",
                        "grid_oracle_minimum_error_m": raw["grid_oracle_minimum_error_m"],
                        **case_diagnostics[(case["case_id"], repeats)], **rank, **error}
                    rows.append(changed); stream.write(json.dumps(changed, allow_nan=False) + "\n")
                stream.flush()
                if index % 20 == 0:
                    progress(f"ranking_{spacing:g}m", index + 1, len(cases))
    groups, scenario_groups = defaultdict(list), defaultdict(list)
    for row in rows:
        key = f"{row['method']}|{row['spacing_m']:g}m|{row['repeats']}"
        groups[key].append(row); scenario_groups[row["scenario"] + "|" + key].append(row)
    summary = {"complete": True, "classification": protocol["classification"], "cases": len(cases),
        "comparison": {key: population(value) for key, value in groups.items()},
        "by_scenario": {key: population(value) for key, value in scenario_groups.items()},
        "elapsed_s": time.monotonic() - started, "held_out_used": False, "real_accuracy_measured": False,
        "protocol_sha256": sha(output / "protocol.json"), "powers_input_unchanged": sha(powers_file) == protocol["input_hashes"]["burst_powers"],
        "limits": protocol["limits"]}
    write_json(output / "summary.json", summary); progress("complete", len(rows), len(rows))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", required=True)
    parser.add_argument("--twoheight", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--deadline-s", type=float, default=590)
    args = parser.parse_args()
    result = run(args.development, args.twoheight, args.benchmark, args.output, deadline_s=args.deadline_s)
    print(json.dumps({key: result[key] for key in ("complete", "cases", "elapsed_s", "classification")}, indent=2))
