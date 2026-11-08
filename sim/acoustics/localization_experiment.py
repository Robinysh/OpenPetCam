"""Bounded offline localization benchmark; synthetic scores are not real accuracy.

Freeze protocol and spatial split before any waveform generation. Select feature
configuration and abstention thresholds on development cases only. Save every
held-out result, including negative controls and model-mismatch failures.
"""
import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
from scipy import signal

from .geometry import Rig
from .localization import errors, rank_candidates, ripple_features, spatial_partition, spectrum, summarize_errors
from .localization_model import build_paths, load_room, render_ir, render_transfer


CONFIGS = [dict(name="low", smooth_hz=120., band=(300., 2000.)),
           dict(name="mid", smooth_hz=200., band=(300., 4000.)),
           dict(name="wide", smooth_hz=300., band=(300., 7000.))]
SCENARIOS = ("nominal_grid_algorithm_check", "off_grid", "colored_source", "harmonic_source",
             "geometry_2cm", "geometry_10cm", "diffuse_tail", "directional_mic", "low_snr",
             "occluded_source", "anechoic_negative_control", "silence_negative_control")
RATE, NFFT = 16000, 8192


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    temporary = Path(str(path) + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def candidate_grid(model, spacing):
    """Exploratory endpoint bounding box, not a claim of enclosed/free space."""
    ends = np.array([v for w in model.declared_wall_extents for v in (w["a_xz_m"], w["b_xz_m"])])
    # Public model attributes are also recorded in the protocol for review.
    lo, hi = ends.min(0), ends.max(0)
    floor = next(s.point_m[1] for s in model.surfaces if s.surface_id == "floor")
    positions = []
    for x in np.arange(np.ceil(lo[0] / spacing) * spacing, hi[0], spacing):
        for z in np.arange(np.ceil(lo[1] / spacing) * spacing, hi[1], spacing):
            if np.hypot(x, z) < .6:
                continue
            for height in (.45, 1.2):
                positions.append([float(x), float(floor + height), float(z)])
    if len(positions) < 10 or len(positions) > 5000:
        raise ValueError("candidate grid must have 10..5000 points; adjust spacing")
    return np.array(positions)


def dictionary(model, positions, receiver, progress):
    frequency = np.fft.rfftfreq(NFFT, 1 / RATE)
    powers, visible = [], []
    for i, point in enumerate(positions):
        paths = build_paths(model, point, receiver.receiver_m, timing="direct_relative")
        direct = render_transfer([p for p in paths if p.order == 0], frequency)
        reflected = render_transfer([p for p in paths if p.order > 0], frequency)
        powers.append(np.stack([abs(direct + beta * reflected)**2 for beta in (.25, .5, .75)]))
        visible.append(paths.direct_visible)
        if i % 100 == 0:
            progress("dictionary", i, len(positions))
    power = np.array(powers)
    return [ripple_features(power, frequency, **{k: v for k, v in c.items() if k != "name"})[0]
            for c in CONFIGS], np.array(visible, bool)


def perturbed(model, rng, sigma):
    surfaces = []
    for surface in model.surfaces:
        shift = np.asarray(surface.normal) * rng.normal(0, sigma)
        vertices = None if surface.vertices_m is None else tuple(tuple(np.asarray(v) + shift) for v in surface.vertices_m)
        surfaces.append(replace(surface, point_m=tuple(np.asarray(surface.point_m) + shift),
                                vertices_m=vertices, reflection=float(rng.uniform(.25, .8))))
    return replace(model, surfaces=tuple(surfaces))


def excitation(rng, family, *, held_out=False):
    t = np.arange(int(RATE * .28)) / RATE
    raw = rng.normal(size=len(t))
    envelope = (1 - np.exp(-t / .004)) * np.exp(-t / (.055 if held_out else .09))
    if family == "colored_source":
        raw = signal.sosfilt(signal.butter(3, 1800 if held_out else 3000, fs=RATE, output="sos"), raw)
    elif family == "harmonic_source":
        fundamental = rng.uniform(140, 280)
        raw = .08 * raw + sum(np.sin(2 * np.pi * fundamental * k * t + rng.uniform(-np.pi, np.pi)) / k
                             for k in range(1, 12))
    return raw * envelope


def simulate_case(model, point, receiver, scenario, seed, held_out):
    rng = np.random.default_rng(seed)
    sigma = .02 if scenario == "geometry_2cm" else .10 if scenario == "geometry_10cm" else 0.
    truth = perturbed(model, rng, sigma) if sigma else replace(model, surfaces=tuple(
        replace(s, reflection=.5) for s in model.surfaces))
    speed = float(rng.uniform(339, 347)) if sigma else 343.
    paths = build_paths(truth, point, receiver.receiver_m, receiver_axis=receiver.receiver_axis,
                        mic_directivity=.5 if scenario == "directional_mic" else 0,
                        sound_speed=speed, timing="direct_relative")
    rendered_paths = [p for p in paths if p.order == 0] if scenario == "anechoic_negative_control" else paths
    h = render_ir(rendered_paths, RATE, .25, .01)
    if scenario == "silence_negative_control":
        h[:] = 0
    if scenario == "diffuse_tail":
        t = np.arange(len(h)) / RATE
        tail = rng.normal(size=len(h)) * np.exp(-np.maximum(t - .035, 0) / .05) * (t >= .035)
        tail *= .35 * np.linalg.norm(h) / max(np.linalg.norm(tail), 1e-20)
        h += tail
    source = excitation(rng, scenario, held_out=held_out)
    recording = signal.fftconvolve(source, h)
    snr = 5. if scenario == "low_snr" else 25.
    noise = rng.normal(size=len(recording))
    noise *= np.linalg.norm(recording) / max(np.linalg.norm(noise), 1e-20) * 10**(-snr / 20)
    recording += noise
    return recording, {"direct_visible": paths.direct_visible, "paths": len(paths),
                       "rendered_paths": 0 if scenario == "silence_negative_control" else len(rendered_paths),
                       "requested_snr_db": snr, "snr_db": snr if np.linalg.norm(h) > 0 else None,
                       "recording_energy": float(recording @ recording),
                       "sound_speed_m_s": speed, "surface_shift_sigma_m": sigma,
                       "source_family": "synthetic decaying noise burst; harmonic/colored when named, not a validated dog bark",
                       "claim": "same-model algorithm check only" if scenario == "nominal_grid_algorithm_check"
                       else "synthetic mismatch/negative control; not real-room accuracy"}


def make_cases(positions, visible, *, count, split, seed, spacing):
    rng = np.random.default_rng(seed)
    partition = np.array(spatial_partition(positions))
    indexes = np.flatnonzero(partition == split)
    if len(indexes) < 4:
        raise ValueError("insufficient source tiles in split")
    cases = []
    for i in range(count):
        scenario = SCENARIOS[i % len(SCENARIOS)]
        eligible = indexes[~visible[indexes]] if scenario == "occluded_source" else indexes
        if not len(eligible):
            raise ValueError("occluded source control requires blocked candidate points")
        index = int(rng.choice(eligible))
        point = positions[index].copy()
        on_grid = scenario in ("nominal_grid_algorithm_check", "occluded_source")
        if not on_grid:
            for _ in range(1000):
                proposal = point.copy(); proposal[[0, 2]] += rng.uniform(-.35 * spacing, .35 * spacing, 2)
                if spatial_partition([proposal])[0] == split:
                    point = proposal; break
            else:
                raise ValueError("could not generate off-grid case in its frozen spatial tile")
        cases.append({"case_id": f"{split}-{i:05d}", "source_m": point.tolist(),
                      "scenario": scenario, "seed": int(rng.integers(1, 2**31)),
                      "split": split, "source_grid_index": index,
                      "anchor_direct_visible": bool(visible[index]),
                      "position_is_on_grid": on_grid})
    return cases


def ranked_row(case, rank, diagnostic, receiver_m):
    """Keep uninformative cases; never fabricate positions or their errors."""
    rank = {k: v for k, v in rank.items() if k != "scores"}
    error = (errors(rank["position_m"], case["source_m"], receiver_m)
             if rank["position_m"] is not None else {"distance_error_m": None, "azimuth_error_deg": None})
    return dict(case, **rank, **error, diagnostic=diagnostic)


def selection_loss(rows, missing_penalty_m):
    """A predeclared development loss includes every case and failed location."""
    return float(np.mean([r["distance_error_m"] if r["distance_error_m"] is not None else missing_penalty_m
                          for r in rows]))


def passes_rule(row, similarity, margin):
    # Match rank_candidates exactly: an exact or numerical tie always abstains.
    return (row.get("distance_error_m") is not None and row.get("similarity") is not None
            and row.get("score_margin") is not None and row["score_margin"] > 1e-9
            and row["similarity"] >= similarity and row["score_margin"] >= margin)


def thresholds(rows):
    """Choose a development-only selective rule; report held-out coverage later."""
    best = None
    informative = [r for r in rows if r.get("similarity") is not None and r.get("score_margin") is not None]
    for similarity in np.quantile([r["similarity"] for r in informative], [0, .25, .5, .75, .9]) if informative else []:
        for margin in np.quantile([r["score_margin"] for r in informative], [0, .25, .5, .75, .9]):
            selected = [r for r in rows if passes_rule(r, similarity, margin)]
            if len(selected) < max(8, len(rows) // 20):
                continue
            precision = np.mean([r["distance_error_m"] <= 1 for r in selected])
            if precision >= .8 and (best is None or len(selected) > best[0]):
                best = len(selected), float(similarity), float(margin), float(precision)
    if best is None:
        return {"min_similarity": 1.01, "min_margin": 2.01, "development_precision": None,
                "reason": "No development rule achieved 80% within 1m with minimum coverage; abstain."}
    return {"min_similarity": best[1], "min_margin": best[2], "development_precision": best[3],
            "development_accepted": best[0], "development_acceptance_fraction": best[0] / len(rows),
            "meaning": "calibrated on synthetic development cases only; no real calibration"}


def run(room_path, output, *, cases=512, spacing=.5, seed=20261105, deadline_s=3600):
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def progress(phase, done, total):
        write_json(output / "progress.json", {"pid": os.getpid(), "phase": phase, "done": done,
                   "total": total, "elapsed_s": time.monotonic() - started, "updated": time.time()})
        if time.monotonic() - started > deadline_s:
            raise TimeoutError("experiment deadline reached; partial results preserved")

    model = load_room(room_path)
    receiver = Rig().pose(0, 0)
    positions = candidate_grid(model, spacing)
    missing_penalty = float(np.linalg.norm(np.ptp(positions, axis=0)) + 2 * spacing)
    source_files = list(Path(__file__).parent.glob("localization*.py")) + [Path(__file__).with_name("geometry.py")]
    source_hashes = {p.name: sha(p) for p in source_files}
    (output / "source").mkdir()
    for path in source_files:
        shutil.copyfile(path, output / "source" / path.name)
    protocol = {"schema": "openpetcam.localization.benchmark.v2", "created": time.time(),
                "room_file": str(Path(room_path).resolve()), "room_sha256": sha(room_path),
                "code_sha256": source_hashes,
                "seed": seed, "cases_held_out": cases, "cases_development": max(64, cases // 4),
                "sample_rate": RATE, "nfft": NFFT, "source_grid_spacing_m": spacing,
                "candidate_domain": "wall-endpoint bounding box, NOT a verified enclosed or walkable room",
                "candidate_positions_m": positions.tolist(), "receiver_m": receiver.receiver_m.tolist(),
                "receiver_pose": "assumed canonical orientation in saved model axes, not historical clip pose",
                "configs": CONFIGS, "pressure_reflection_variants": [.25, .5, .75],
                "scenarios": SCENARIOS, "spatial_split": "1m x/z tiles, SHA256 seed731; heights grouped",
                "candidate_spatial_partitions": spatial_partition(positions),
                "selection_loss": "mean distance over all cases; unlocalized cases incur a fixed missing penalty",
                "missing_selection_penalty_m": missing_penalty,
                "abstention_target": "development >=80% within1m, >=max(8,n/20)accepted; no real calibration",
                "source_envelope_decay_s": {"development": .09, "held_out": .055},
                "source_duration_s": .28, "render_ir_duration_s": .25,
                "spectral_window_limit": "Hann-windowed transient spectrum is only an approximation to stationary |H|^2",
                "phase_policy": "signed IR/complex path interference retained; unknown-source scorer uses log-power ripple",
                "source_prior": "smooth source spectrum; intentionally violated by harmonic controls",
                "room_provenance": model.provenance,
                "limits": ["All localization truth here is simulated, not measured.",
                           "Nominal cases use the same first-order geometry family: algorithm check only.",
                           "Mismatch cases perturb geometry/material/speed, add tails/noise or directional response.",
                           "No diffraction or true higher-order furnished-room simulator is validated.",
                           "No real-source accuracy claim; scores are not probabilities."]}
    write_json(output / "protocol.json", protocol)  # before dictionary or waveform fitting
    features, visible = dictionary(model, positions, receiver, progress)
    np.savez_compressed(output / "dictionary.npz", positions_m=positions, direct_visible=visible,
                        **{c["name"]: f for c, f in zip(CONFIGS, features)})
    dev_cases = make_cases(positions, visible, count=protocol["cases_development"], split="development", seed=seed,
                          spacing=spacing)
    test_cases = make_cases(positions, visible, count=cases, split="held_out", seed=seed + 1, spacing=spacing)
    write_json(output / "case_protocol.json", {"development": dev_cases, "held_out": test_cases})
    write_json(output / "frozen.json", {"protocol_sha256": sha(output / "protocol.json"),
               "case_protocol_sha256": sha(output / "case_protocol.json"),
               "dictionary_sha256": sha(output / "dictionary.npz"), "frozen_before_any_waveforms": True,
               "candidate_direct_visible": int(visible.sum()), "candidates": len(positions),
               "nonzero_feature_candidates": {c["name"]: int(np.sum(np.any(np.linalg.norm(f, axis=-1) > 1e-8, axis=1)))
                                              for c, f in zip(CONFIGS, features)}})
    dev = [[] for _ in CONFIGS]
    for i, case in enumerate(dev_cases):
        y, diagnostic = simulate_case(model, case["source_m"], receiver, case["scenario"], case["seed"], False)
        f, power = spectrum(y, RATE)
        for k, config in enumerate(CONFIGS):
            feature, _ = ripple_features(power, f, **{a: b for a, b in config.items() if a != "name"})
            rank = rank_candidates(feature, features[k], positions)
            dev[k].append(ranked_row(case, rank, diagnostic, receiver.receiver_m))
        progress("development", i + 1, len(dev_cases))
    losses = [selection_loss(rows, missing_penalty) for rows in dev]
    winner = int(np.argmin(losses))
    rule = thresholds(dev[winner])
    selected = {"config_index": winner, "config": CONFIGS[winner], "acceptance_rule": rule,
                "selected_using": protocol["selection_loss"], "held_out_seen": False,
                "development_selection_loss_m": dict(zip([c["name"] for c in CONFIGS], losses)),
                "missing_selection_penalty_m": missing_penalty,
                "development": {c["name"]: summarize_errors(rows) for c, rows in zip(CONFIGS, dev)}}
    for row in dev[winner]:
        row["accepted"] = bool(passes_rule(row, rule["min_similarity"], rule["min_margin"]))
    selected["development_after_selected_rule"] = summarize_errors(dev[winner])
    write_json(output / "selected.json", selected)  # frozen before any held-out waveform is generated
    write_json(output / "development.json", dev)
    results = []
    rng = np.random.default_rng(seed + 17)
    center = np.median(positions, axis=0)
    with (output / "held_out.jsonl").open("x") as stream:
        for i, case in enumerate(test_cases):
            y, diagnostic = simulate_case(model, case["source_m"], receiver, case["scenario"], case["seed"], True)
            f, power = spectrum(y, RATE)
            feature, _ = ripple_features(power, f, **{k: v for k, v in CONFIGS[winner].items() if k != "name"})
            rank = rank_candidates(feature, features[winner], positions,
                                   min_margin=rule["min_margin"], min_similarity=rule["min_similarity"])
            random_point = positions[int(rng.integers(len(positions)))]
            nearest_point = positions[int(np.argmin(np.linalg.norm(positions - case["source_m"], axis=1)))]
            row = ranked_row(case, rank, diagnostic, receiver.receiver_m)
            row.update(random_baseline=errors(random_point, case["source_m"], receiver.receiver_m),
                       fixed_prior_baseline=errors(center, case["source_m"], receiver.receiver_m),
                       grid_oracle_floor=errors(nearest_point, case["source_m"], receiver.receiver_m))
            results.append(row); stream.write(json.dumps(row, allow_nan=False) + "\n"); stream.flush()
            progress("held_out", i + 1, len(test_cases))
    groups = defaultdict(list)
    for r in results:
        groups[r["scenario"]].append(r)
    summary = {"complete": True, "domain": "synthetic_only", "selected": selected,
               "overall": summarize_errors(results),
               "by_scenario": {name: summarize_errors(rows) for name, rows in groups.items()},
               "by_direct_visibility": {str(visible): summarize_errors([r for r in results if r["diagnostic"]["direct_visible"] == visible])
                                        for visible in (True, False)},
               "random_baseline": summarize_errors([r["random_baseline"] for r in results]),
               "fixed_prior_baseline": summarize_errors([r["fixed_prior_baseline"] for r in results]),
               "grid_oracle_floor": summarize_errors([r["grid_oracle_floor"] for r in results]),
               "grid_oracle_meaning": "truth-using discretization floor, not a usable estimator",
               "case_dependence": "Repeated random draws can share spatial tiles; no unclustered confidence intervals reported",
               "elapsed_s": time.monotonic() - started, "real_accuracy_measured": False,
               "protocol_sha256": sha(output / "protocol.json"), "case_protocol_sha256": sha(output / "case_protocol.json"),
               "selected_sha256": sha(output / "selected.json")}
    write_json(output / "summary.json", summary); progress("complete", cases, cases)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--room", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cases", type=int, default=512)
    parser.add_argument("--spacing", type=float, default=.5)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--deadline-s", type=float, default=3600)
    args = parser.parse_args()
    if args.cases < 16 or args.cases > 10000 or not .25 <= args.spacing <= 1 or not 10 <= args.deadline_s <= 14400:
        parser.error("cases16..10000, spacing.25..1m and deadline10..14400s required")
    print(json.dumps(run(args.room, args.output, cases=args.cases, spacing=args.spacing,
                         seed=args.seed, deadline_s=args.deadline_s)["overall"], indent=2))
