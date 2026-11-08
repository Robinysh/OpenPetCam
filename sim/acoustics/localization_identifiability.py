"""Idealized transfer identifiability; no recording or held-out labels are read.

The complete candidate dictionary is a forward-model hypothesis, not measured
truth. These oracle queries deliberately know a noiseless on-grid transfer. They
diagnose representation ambiguity and available model paths, not real accuracy.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from .localization import spatial_partition
from .localization_model import build_paths, load_room, render_transfer


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


def angular_difference_deg(a, b):
    return np.abs((np.asarray(a) - np.asarray(b) + 180.) % 360. - 180.)


def circular_span_deg(angles):
    """Smallest circular arc containing angles; wrapping never creates 358deg."""
    angle = np.sort(np.asarray(angles, float) % 360.)
    if not len(angle):
        return None
    return float(360. - np.max(np.diff(np.r_[angle, angle[0] + 360.])))


def material_profile_matrix(features, *, query_variant=1):
    """Noiseless true-variant queries against all candidate material variants."""
    features = np.asarray(features, float)
    if features.ndim == 2:
        features = features[:, None, :]
    if features.ndim != 3 or not 0 <= query_variant < features.shape[1] or not np.isfinite(features).all():
        raise ValueError("finite position/variant/feature tensor and valid query variant required")
    queries = features[:, query_variant]
    valid = np.linalg.norm(features, axis=-1) > 1e-8
    query_valid = valid[:, query_variant]
    score = queries @ features.reshape(-1, features.shape[-1]).T
    score = np.clip(score.reshape(len(features), len(features), features.shape[1]), -1., 1.)
    score = np.where(valid[None, :, :], score, -np.inf).max(axis=-1)
    score[~query_valid] = -np.inf
    return score, query_valid


def normalized_profile_matrix(vectors, *, complex_scalar=False):
    """Compare ideal transfer shape after discarding one unknown scalar gain.

    For complex transfers, absolute inner product also discards a global phase.
    A geometric direct-relative time origin is still assumed, not measured.
    """
    vectors = np.asarray(vectors)
    if vectors.ndim != 2 or not np.isfinite(vectors).all():
        raise ValueError("finite candidate/vector matrix required")
    norm = np.linalg.norm(vectors, axis=-1)
    valid = norm > 1e-20
    unit = np.divide(vectors, norm[:, None], out=np.zeros_like(vectors), where=valid[:, None])
    gram = unit @ unit.conj().T
    profile = np.abs(gram) if complex_scalar else gram.real
    profile = np.clip(profile, -1., 1.)
    profile[~valid] = -np.inf
    profile[:, ~valid] = -np.inf
    return profile, valid


def oracle_rows(profile, informative, positions_m, receiver_m, *, near_tolerance=.02, tie_tolerance=1e-9):
    """An exact tie is unidentifiable, even if stable argmax chooses the truth."""
    positions = np.asarray(positions_m, float)
    profile, informative = np.asarray(profile, float), np.asarray(informative, bool)
    receiver = np.asarray(receiver_m, float)
    if profile.shape != (len(positions), len(positions)) or positions.shape[1:] != (3,) or len(positions) < 2:
        raise ValueError("square profiles and at least two xyz positions required")
    if not np.isfinite(positions).all() or receiver.shape != (3,) or not np.isfinite(receiver).all():
        raise ValueError("finite positions and receiver required")
    if informative.shape != (len(positions),) or near_tolerance < tie_tolerance or tie_tolerance < 0:
        raise ValueError("valid information mask and score tolerances required")
    relative = positions - receiver
    ranges = np.linalg.norm(relative, axis=1)
    azimuth = np.degrees(np.arctan2(relative[:, 0], relative[:, 2]))
    partitions = spatial_partition(positions)
    rows = []
    for index, point in enumerate(positions):
        base = {"candidate_index": index, "source_model_m": point.tolist(), "spatial_partition": partitions[index],
                "tile_xz": np.floor(point[[0, 2]]).astype(int).tolist(), "informative": bool(informative[index])}
        if not informative[index]:
            rows.append(dict(base, oracle_uniquely_identifiable=False, score_margin=None,
                             best_index=None, nearest_confounder_index=None, exact_other_candidates=[],
                             near_candidate_count=0, near_box_diagonal_m=None,
                             near_range_span_m=None, near_azimuth_span_deg=None,
                             near_max_azimuth_error_deg=None, near_height_span_m=None,
                             reason="No transfer-dependent feature; unknown gain supplies no range information."))
            continue
        values = profile[index]
        order = np.argsort(-values, kind="stable")
        best, second = int(order[0]), int(order[1])
        top = float(values[best])
        margin = float(top - values[second]) if np.isfinite(values[second]) else None
        other_values = values.copy(); other_values[index] = -np.inf
        confounder = int(np.argmax(other_values)) if np.isfinite(other_values).any() else None
        exact = np.flatnonzero(values >= top - tie_tolerance)
        exact_others = exact[exact != index]
        near = np.flatnonzero(values >= top - near_tolerance)
        near_spread = float(np.linalg.norm(np.ptp(positions[near], axis=0)))
        az_error = angular_difference_deg(azimuth[near], azimuth[index])
        row = dict(base, oracle_uniquely_identifiable=bool(best == index and len(exact) == 1),
                   best_index=best, score_margin=margin,
                   exact_other_candidates=exact_others.tolist(), exact_tie=bool(len(exact) > 1),
                   near_candidate_count=len(near), near_other_candidates=near[near != index].tolist(),
                   near_box_diagonal_m=near_spread, near_range_span_m=float(np.ptp(ranges[near])),
                   near_azimuth_span_deg=circular_span_deg(azimuth[near]),
                   near_max_azimuth_error_deg=float(np.max(az_error)),
                   near_height_span_m=float(np.ptp(positions[near, 1])),
                   nearest_confounder_index=confounder,
                   nearest_confounder_similarity=float(other_values[confounder]) if confounder is not None else None,
                   nearest_confounder_distance_m=float(np.linalg.norm(point - positions[confounder])) if confounder is not None else None,
                   nearest_confounder_range_error_m=float(abs(ranges[index] - ranges[confounder])) if confounder is not None else None,
                   nearest_confounder_azimuth_error_deg=float(angular_difference_deg(azimuth[index], azimuth[confounder])) if confounder is not None else None,
                   nearest_confounder_height_error_m=float(abs(point[1] - positions[confounder, 1])) if confounder is not None else None)
        rows.append(row)
    return rows


def summarize(rows):
    informative = [r for r in rows if r["informative"]]
    unique = [r for r in informative if r["oracle_uniquely_identifiable"]]
    exact = [r for r in informative if r["exact_tie"]]
    near_ambiguous = [r for r in informative if r["near_candidate_count"] > 1]
    def quantile(key, q):
        return float(np.quantile([r[key] for r in informative], q)) if informative else None
    return {"candidates": len(rows), "informative": len(informative), "uninformative": len(rows) - len(informative),
            "oracle_unique_candidates": len(unique), "oracle_unique_all_candidate_fraction": len(unique) / len(rows),
            "exact_tie_candidates": len(exact), "near_ambiguous_candidates": len(near_ambiguous),
            "near_ambiguous_informative_fraction": len(near_ambiguous) / len(informative) if informative else None,
            "median_near_candidate_count": quantile("near_candidate_count", .5),
            "median_near_box_diagonal_m": quantile("near_box_diagonal_m", .5),
            "p90_near_box_diagonal_m": quantile("near_box_diagonal_m", .9),
            "median_near_range_span_m": quantile("near_range_span_m", .5),
            "median_near_azimuth_span_deg": quantile("near_azimuth_span_deg", .5),
            "near_range_ambiguous_azimuth_stable": sum(r["near_range_span_m"] >= 1 and r["near_max_azimuth_error_deg"] <= 15 for r in informative),
            "near_azimuth_ambiguous_range_stable": sum(r["near_max_azimuth_error_deg"] >= 30 and r["near_range_span_m"] <= .5 for r in informative),
            "near_both_azimuth_and_range_ambiguous": sum(r["near_max_azimuth_error_deg"] >= 30 and r["near_range_span_m"] >= 1 for r in informative),
            "near_height_ambiguous": sum(r["near_height_span_m"] >= .5 for r in informative),
            "claim": "ideal noiseless on-grid representation oracle only; no waveform or real-room accuracy"}


def run(benchmark_dir, output_dir, *, deadline_s=600):
    if not 10 <= deadline_s <= 600:
        raise ValueError("deadline must be10..600seconds")
    benchmark, output = Path(benchmark_dir).resolve(), Path(output_dir).resolve()
    protocol_file, dictionary_file = benchmark / "protocol.json", benchmark / "dictionary.npz"
    benchmark_protocol = json.loads(protocol_file.read_text())
    room_file = Path(benchmark_protocol["room_file"])
    if sha(room_file) != benchmark_protocol["room_sha256"]:
        raise ValueError("room differs from frozen benchmark")
    frozen = json.loads((benchmark / "frozen.json").read_text())
    if sha(dictionary_file) != frozen["dictionary_sha256"] or sha(protocol_file) != frozen["protocol_sha256"]:
        raise ValueError("benchmark inputs differ from frozen hashes")
    with np.load(dictionary_file, allow_pickle=False) as archive:
        positions = archive["positions_m"].copy()
        features = {c["name"]: archive[c["name"]].copy() for c in benchmark_protocol["configs"]}
    if len(positions) > 1000:
        raise ValueError("bounded map supports at most1000positions")
    output.mkdir(parents=True, exist_ok=False)
    source_dir = output / "source"; source_dir.mkdir()
    sources = [Path(__file__), Path(__file__).with_name("localization.py"), Path(__file__).with_name("localization_model.py"), Path(__file__).with_name("geometry.py")]
    source_hashes = {}
    for path in sources:
        source_hashes[path.name] = sha(path)
        shutil.copyfile(path, source_dir / path.name)
    protocol = {"schema": "openpetcam.localization-identifiability.v1", "created_epoch": time.time(),
                "benchmark_protocol_sha256": sha(protocol_file), "dictionary_sha256": sha(dictionary_file),
                "room_sha256": sha(room_file), "source_sha256": source_hashes,
                "frozen_before_computation": True, "deadline_s": deadline_s,
                "input_scope": "frozen dictionary/protocol and room forward model only; no held-out results, labels or recordings read",
                "candidate_domain": benchmark_protocol["candidate_domain"],
                "receiver_m": benchmark_protocol["receiver_m"], "configs": benchmark_protocol["configs"],
                "query_pressure_reflection": .5, "candidate_pressure_reflections": benchmark_protocol["pressure_reflection_variants"],
                "feature_oracle": "noiseless exact query at middle material variant; profile candidate materials",
                "feature_near_score_tolerance": .02, "exact_tie_tolerance": 1e-9,
                "raw_shape_near_score_tolerance": .001,
                "raw_shape_policy": "fixed .5material; normalized raw power and complex transfer; complex comparison removes global scalar phase/gain but still assumes geometric direct-relative timing",
                "spatial_partition": "same seed731one-metre tiles; map only, no split used for selection or validation",
                "limits": ["Algorithm/representation oracle, not independent validation or real localization accuracy.",
                           "No unknown source spectrum, transient window, noise, off-grid source or model mismatch is included.",
                           "Known noiseless transfer cannot be extracted from arbitrary unknown source at one mic.",
                           "Endpoint bounding box is not an established enclosed/walkable source domain.",
                           "Near-score sets are sensitivity diagnostics, not confidence regions.",
                           "Default floor/ceiling extents, vertical wall extrusions and pressure materials remain uncertain assumptions."]}
    write_json(output / "protocol.json", protocol)
    started = time.monotonic()
    def progress(phase, done, total):
        value = {"pid": os.getpid(), "phase": phase, "done": done, "total": total,
                 "elapsed_s": time.monotonic() - started, "updated_epoch": time.time()}
        temporary = output / ".progress.tmp"
        temporary.write_text(json.dumps(value, indent=2)); temporary.replace(output / "progress.json")
        if value["elapsed_s"] > deadline_s:
            raise TimeoutError("identifiability map deadline reached; partial results preserved")
    model = load_room(room_file)
    receiver = np.asarray(benchmark_protocol["receiver_m"])
    frequencies = np.fft.rfftfreq(benchmark_protocol["nfft"], 1 / benchmark_protocol["sample_rate"])
    path_rows, transfers = [], []
    for index, point in enumerate(positions):
        paths = build_paths(model, point, receiver, timing="direct_relative")
        direct = render_transfer([p for p in paths if p.order == 0], frequencies)
        reflected = render_transfer([p for p in paths if p.order > 0], frequencies)
        transfers.append(direct + .5 * reflected)
        path_rows.append({"candidate_index": index, "source_model_m": point.tolist(),
                          "spatial_partition": spatial_partition([point])[0],
                          "direct_visible": paths.direct_visible, "path_count": len(paths),
                          "reflection_count": sum(p.order > 0 for p in paths),
                          "reflection_surface_ids": [s for p in paths if p.order > 0 for s in p.surface_ids],
                          "path_delays_s": [p.delay_s for p in paths],
                          "path_lengths_m": [p.length_m for p in paths],
                          "path_gains_at_pressure_reflection_0p5": [p.gain * (.5 if p.order else 1.) for p in paths]})
        if index % 100 == 0:
            progress("paths", index, len(positions))
    write_json(output / "path_map.json", path_rows)
    all_maps, summaries = {}, {}
    for index, config in enumerate(benchmark_protocol["configs"]):
        name = config["name"]
        profile, valid = material_profile_matrix(features[name], query_variant=1)
        rows = oracle_rows(profile, valid, positions, receiver)
        all_maps[name] = rows
        summaries[name] = summarize(rows)
        summaries[name]["by_spatial_partition"] = {split: summarize([r for r in rows if r["spatial_partition"] == split])
                                                       for split in ("development", "held_out")}
        write_json(output / f"ripple_{name}_map.json", rows)
        np.savez_compressed(output / f"ripple_{name}_profiles.npz", profile=profile, informative=valid,
                            positions_m=positions)
        progress("ripple_oracles", index + 1, len(features))
    transfer = np.asarray(transfers)
    raw_keep = (frequencies >= 300) & (frequencies <= 7000)
    raw = {}
    for name, vector, complex_scalar in (("raw_power_unknown_gain", abs(transfer[:, raw_keep]) ** 2, False),
                                         ("complex_transfer_unknown_scalar", transfer[:, raw_keep], True)):
        profile, valid = normalized_profile_matrix(vector, complex_scalar=complex_scalar)
        rows = oracle_rows(profile, valid, positions, receiver, near_tolerance=.001)
        raw[name] = summarize(rows)
        write_json(output / f"{name}_map.json", rows)
        progress(name, 1, 1)
    grouped = defaultdict(list)
    for row in path_rows:
        grouped[row["path_count"]].append(row)
    summary = {"complete": True, "domain": "forward_model_oracle_only", "real_accuracy_measured": False,
               "candidates": len(positions), "zero_path_candidates": len(grouped[0]),
               "direct_visible_candidates": sum(r["direct_visible"] for r in path_rows),
               "path_count_histogram": dict(Counter(str(r["path_count"]) for r in path_rows)),
               "reflection_count_histogram": dict(Counter(str(r["reflection_count"]) for r in path_rows)),
               "reflection_surface_coverage": dict(Counter(s for r in path_rows for s in r["reflection_surface_ids"])),
               "ripple_oracles": summaries, "raw_shape_oracles": raw,
               "claim": "All queries know exact noiseless on-grid model transfers; these are not waveform or real-room accuracy metrics.",
               "elapsed_s": time.monotonic() - started, "protocol_sha256": sha(output / "protocol.json")}
    write_json(output / "summary.json", summary)
    progress("complete", len(positions), len(positions))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--deadline-s", type=float, default=600)
    args = parser.parse_args()
    print(json.dumps(run(args.benchmark, args.output, deadline_s=args.deadline_s), indent=2))
