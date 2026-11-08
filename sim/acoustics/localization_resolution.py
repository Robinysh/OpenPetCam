"""Exploratory DEVELOPMENT-only global candidate-grid resolution comparison.

Uses pre-existing burst-power prefixes without regenerating waveforms. Candidate
placement depends only on saved wall endpoints, grid spacing and a fixed nominal
floor+.45m height prior. True source positions only label errors/prior strata.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np

from .localization import errors, rank_candidates, ripple_features, spatial_partition, summarize_errors
from .geometry import Rig
from .localization_model import build_paths, load_room, render_transfer


FAMILIES = ("independent_broadband", "replayed_broadband", "colored_source",
            "harmonic_variable_pitch", "harmonic_fixed_pitch")
SPACINGS = (.5, .25, .125)
REPEATS = (1, 4, 16)
PRESSURE_VARIANTS = (.25, .5, .75)
LOW = {"name": "low", "smooth_hz": 120., "band": (300., 2000.)}


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def low_height_grid(model, spacing, *, height_above_floor=.45):
    """Truth-independent global lattice inside the endpoint bounding rectangle.

    The rectangle is NOT a certified enclosed/walkable/source domain. The .6m
    lens-origin exclusion matches the original benchmark's singularity guard.
    """
    if spacing not in SPACINGS or not math.isfinite(height_above_floor):
        raise ValueError("predeclared spacing and finite conditional height required")
    endpoints = np.asarray([point for wall in model.declared_wall_extents
                            for point in (wall["a_xz_m"], wall["b_xz_m"])])
    if endpoints.ndim != 2 or endpoints.shape[1:] != (2,) or not np.isfinite(endpoints).all():
        raise ValueError("finite declared wall endpoints required")
    floor = next(surface.point_m[1] for surface in model.surfaces if surface.surface_id == "floor")
    lower, upper = endpoints.min(axis=0), endpoints.max(axis=0)
    xz = [[float(x), float(z)]
          for x in np.arange(math.ceil(lower[0] / spacing) * spacing, upper[0], spacing)
          for z in np.arange(math.ceil(lower[1] / spacing) * spacing, upper[1], spacing)
          if np.hypot(x, z) >= .6]
    if not 2 <= len(xz) <= 10000:
        raise ValueError("bounded grid requires2..10000candidates")
    return np.asarray([[x, floor + height_above_floor, z] for x, z in xz], float)


def prefix_mean_powers(powers):
    powers = np.asarray(powers, float)
    if powers.ndim != 2 or len(powers) < max(REPEATS) or not np.isfinite(powers).all() or np.any(powers < 0):
        raise ValueError("at least16finite nonnegative burst spectra required")
    return {n: powers[:n].mean(axis=0) for n in REPEATS}


def dictionary(model, positions, receiver, frequency, progress):
    """Only LOW-band float32 features are retained, one point at a time."""
    keep = (frequency >= LOW["band"][0]) & (frequency <= LOW["band"][1])
    result = np.empty((len(positions), len(PRESSURE_VARIANTS), int(keep.sum())), np.float32)
    visible, path_counts = np.zeros(len(positions), bool), np.zeros(len(positions), np.int16)
    args = {key: value for key, value in LOW.items() if key != "name"}
    for index, point in enumerate(positions):
        paths = build_paths(model, point, receiver, timing="direct_relative")
        direct = render_transfer([path for path in paths if path.order == 0], frequency)
        reflected = render_transfer([path for path in paths if path.order > 0], frequency)
        power = np.asarray([abs(direct + coefficient * reflected) ** 2 for coefficient in PRESSURE_VARIANTS])
        feature, _ = ripple_features(power, frequency, **args)
        result[index] = feature
        visible[index], path_counts[index] = paths.direct_visible, len(paths)
        if index % 100 == 0:
            progress("dictionary", index, len(positions))
    return result, visible, path_counts


def describe(rows):
    result = summarize_errors(rows)
    localized = [row for row in rows if row.get("distance_error_m") is not None]
    result.update(within_0p5m_fraction=sum(r["distance_error_m"] is not None and r["distance_error_m"] <= .5 for r in rows) / len(rows),
                  within_0p25m_fraction=sum(r["distance_error_m"] is not None and r["distance_error_m"] <= .25 for r in rows) / len(rows))
    for key in ("horizontal_error_m", "height_error_m", "range_error_m"):
        values = [row[key] for row in localized if row.get(key) is not None]
        result["median_" + key] = float(np.median(values)) if values else None
    result["median_grid_oracle_minimum_error_m"] = float(np.median([row["grid_oracle_minimum_error_m"] for row in rows])) if all("grid_oracle_minimum_error_m" in row for row in rows) else None
    return result


def grouped_summary(rows):
    return {"all_cases": describe(rows),
            "true_prior_support": describe([r for r in rows if r["true_prior_support"]]) if any(r["true_prior_support"] for r in rows) else {"cases": 0},
            "wrong_prior_support": describe([r for r in rows if not r["true_prior_support"]]) if any(not r["true_prior_support"] for r in rows) else {"cases": 0},
            "paired_groups": len(set(r["paired_group"] for r in rows))}


def run(development_dir, room_path, output_dir, *, deadline_s=590):
    if not 10 <= deadline_s <= 590:
        raise ValueError("deadline must be10..590seconds")
    development, room_path, output = map(lambda value: Path(value).resolve(), (development_dir, room_path, output_dir))
    source_protocol = json.loads((development / "protocol.json").read_text())
    frozen = json.loads((development / "frozen.json").read_text())
    cases_file, powers_file = development / "cases.json", development / "burst_powers.npz"
    if (sha(cases_file) != frozen["cases_sha256"] or sha(development / "protocol.json") != frozen["protocol_sha256"]
            or sha(room_path) != source_protocol["room_sha256"]):
        raise ValueError("development cases/protocol or room no longer matches frozen inputs")
    all_cases = json.loads(cases_file.read_text())
    chosen = [(index, case) for index, case in enumerate(all_cases) if case["scenario"] in FAMILIES]
    if not chosen or any(case["split"] != "development" or case["source_m"] is None for _, case in chosen):
        raise ValueError("predeclared emitting-source DEVELOPMENT families required")
    if set(spatial_partition([case["source_m"] for _, case in chosen])) != {"development"}:
        raise ValueError("source cases cross development tile boundary")
    with np.load(powers_file, allow_pickle=False) as archive:
        power_archive = archive["power"]
        frequency = archive["frequency_hz"]
    if power_archive.shape != (len(all_cases), 16, len(frequency)):
        raise ValueError("burst-power archive does not match frozen cases and sixteen repeats")
    receiver = Rig().pose(0, 0).receiver_m
    model = load_room(room_path)
    floor = next(surface.point_m[1] for surface in model.surfaces if surface.surface_id == "floor")
    positions = {spacing: low_height_grid(model, spacing) for spacing in SPACINGS}
    output.mkdir(parents=True, exist_ok=False)
    source_dir = output / "source"; source_dir.mkdir()
    dependencies = [Path(__file__)] + [Path(__file__).with_name(name) for name in
                                     ("localization.py", "localization_model.py", "geometry.py")]
    for path in dependencies:
        shutil.copyfile(path, source_dir / path.name)
    protocol = {"schema": "openpetcam.localization-grid-development.v1", "created_epoch": time.time(),
                "classification": "EXPLORATORY DEVELOPMENT GRID SENSITIVITY ONLY",
                "source_sha256": {path.name: sha(path) for path in dependencies},
                "input_hashes": {"development_protocol": sha(development / "protocol.json"),
                                 "development_cases": sha(cases_file), "burst_powers": sha(powers_file), "room": sha(room_path)},
                "source_families": FAMILIES, "chosen_case_ids": [case["case_id"] for _, case in chosen],
                "chosen_case_indexes": [index for index, _ in chosen], "cases": len(chosen),
                "paired_groups": len(set(case["paired_group"] for _, case in chosen)),
                "grid_spacings_m": SPACINGS, "grid_positions_m": {str(spacing): value.tolist() for spacing, value in positions.items()},
                "candidate_domain": "global saved-wall-endpoint bounding rectangle; NOT a certified enclosed/walkable/source domain",
                "candidate_placement": "depends only on wall endpoints, global spacing and lens-origin radius guard; never source labels",
                "conditional_prior": {"height_above_saved_floor_m": .45, "true_support_height_tolerance_m": .1},
                "receiver_m": receiver.tolist(), "low_band": LOW, "pressure_variants": PRESSURE_VARIANTS,
                "feature_storage": "float32 LOW-band features only, generated pointwise from exact complex path sum before power",
                "repeats": REPEATS, "aggregation": "same nested1/4/16prefix arithmetic means of existing LINEAR burst powers; no waveforms regenerated",
                "acceptance": "Alwaysfalse; no thresholds tuned and no new confidence calibration",
                "deadline_s": deadline_s, "frozen_before_computation": True,
                "limits": ["No v1held-out labels/predictions, summary or selected-rule file read.",
                           "All case tiles were previously used for development; no fresh confirmatory or real accuracy claim.",
                           "Truth is used only for errors and true/wrong-prior strata, never grid/mask placement.",
                           "Unlocalized source cases remain in all success denominators; no errors/coordinates fabricated.",
                           "Source groups share location and seeds; no independent-sample confidence intervals.",
                           "Saved geometry/scale and first-order acoustics are uncertain priors; candidate-grid refinement does not repair model/source bias."]}
    write_json(output / "protocol.json", protocol)
    started = time.monotonic()
    def progress(phase, done, total):
        status = {"pid": os.getpid(), "phase": phase, "done": done, "total": total,
                  "elapsed_s": time.monotonic() - started, "updated_epoch": time.time()}
        temporary = output / ".progress.tmp"
        temporary.write_text(json.dumps(status, indent=2)); temporary.replace(output / "progress.json")
        if status["elapsed_s"] > deadline_s:
            raise TimeoutError("resolution development deadline reached; partial results preserved")
    query_features = {}
    args = {key: value for key, value in LOW.items() if key != "name"}
    for index, case in chosen:
        for repeats, power in prefix_mean_powers(power_archive[index]).items():
            query_features[(case["case_id"], repeats)] = ripple_features(power, frequency, **args)[0]
    del power_archive
    rows, dictionary_metadata = [], {}
    with (output / "rankings.jsonl").open("x") as stream:
        for spacing in SPACINGS:
            candidate = positions[spacing]
            feature, visible, path_counts = dictionary(model, candidate, receiver, frequency, progress)
            np.savez_compressed(output / f"dictionary_{spacing:g}m.npz", positions_m=candidate, features=feature,
                                direct_visible=visible, path_counts=path_counts)
            dictionary_metadata[str(spacing)] = {"candidates": len(candidate), "feature_bytes": int(feature.nbytes),
                "sha256": sha(output / f"dictionary_{spacing:g}m.npz"),
                "direct_visible": int(visible.sum()), "zero_path": int(np.sum(path_counts == 0)),
                "single_path": int(np.sum(path_counts == 1)),
                "nonzero_features": int(np.sum(np.any(np.linalg.norm(feature, axis=-1) > 1e-8, axis=1)))}
            for case_index, (_, case) in enumerate(chosen):
                true_prior = abs(case["source_m"][1] - (floor + .45)) <= .1
                # This truth-using quantity is evaluation-only, never a search location.
                grid_floor = float(np.min(np.linalg.norm(candidate - case["source_m"], axis=1)))
                for repeats in REPEATS:
                    rank = rank_candidates(query_features[(case["case_id"], repeats)], feature, candidate)
                    rank.pop("scores")
                    rank["near_best_count"] = len(rank.pop("near_best_candidates", []))
                    error = errors(rank["position_m"], case["source_m"], receiver) if rank["position_m"] is not None else {
                        "distance_error_m": None, "azimuth_error_deg": None}
                    if rank["position_m"] is not None:
                        predicted = np.asarray(rank["position_m"])
                        truth = np.asarray(case["source_m"])
                        error.update(horizontal_error_m=float(np.linalg.norm((predicted - truth)[[0, 2]])),
                                     height_error_m=float(abs(predicted[1] - truth[1])),
                                     range_error_m=float(abs(np.linalg.norm(predicted - receiver) - np.linalg.norm(truth - receiver))))
                    else:
                        error.update(horizontal_error_m=None, height_error_m=None, range_error_m=None)
                    row = {"case_id": case["case_id"], "paired_group": case["paired_group"],
                           "source_m": case["source_m"], "scenario": case["scenario"], "split": "development",
                           "spacing_m": spacing, "repeats": repeats, "true_prior_support": bool(true_prior),
                           "grid_oracle_minimum_error_m": grid_floor, **rank, **error}
                    rows.append(row); stream.write(json.dumps(row, allow_nan=False) + "\n")
                stream.flush()
                if case_index % 20 == 0:
                    progress(f"ranking_{spacing:g}m", case_index + 1, len(chosen))
            del feature
    groups, scenario_groups = defaultdict(list), defaultdict(list)
    for row in rows:
        key = f"{row['spacing_m']:g}m|{row['repeats']}"
        groups[key].append(row); scenario_groups[row["scenario"] + "|" + key].append(row)
    summary = {"complete": True, "classification": protocol["classification"], "cases": len(chosen),
               "paired_groups": protocol["paired_groups"], "comparison": {key: grouped_summary(value) for key, value in groups.items()},
               "by_scenario": {key: grouped_summary(value) for key, value in scenario_groups.items()},
               "dictionaries": dictionary_metadata, "elapsed_s": time.monotonic() - started,
               "protocol_sha256": sha(output / "protocol.json"), "held_out_used": False, "real_accuracy_measured": False,
               "limits": protocol["limits"]}
    write_json(output / "summary.json", summary)
    progress("complete", len(rows), len(rows))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--development", required=True)
    parser.add_argument("--room", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--deadline-s", type=float, default=590)
    args = parser.parse_args()
    result = run(args.development, args.room, args.output, deadline_s=args.deadline_s)
    print(json.dumps({key: result[key] for key in ("complete", "cases", "paired_groups", "elapsed_s", "classification")}, indent=2))
