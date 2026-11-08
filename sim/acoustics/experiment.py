"""Run an offline, pose-held-out, one-surface waveform experiment.

See docs/acoustic-waveforms.md. Reads saved PCM only; no hardware or deployment.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import scipy

from .fitting import fit_surface, permutation_check
from .geometry import Rig
from .waveforms import prepare_recordings, save_dataset

ALIGNMENT_WINDOW_S = (-.0005, .002)
MAX_ALIGNMENT_LAG_S = .0005
ALIGNMENT_GUARD_S = .00025


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")


def split_poses(scans, reference, fraction, seed, volume):
    """Split complete poses before reading any audio; repeats never cross folds."""
    rows = []
    for folder in scans:
        for line in (folder / "log.jsonl").read_text().splitlines():
            q = json.loads(line)
            if q["vol"] == volume:
                rows.append({"id": (folder.name, q["k"]), "pose": tuple(q["pos"])})
    by_id = {tuple(r["id"]): r for r in rows}
    if len(by_id) != len(rows) or reference not in by_id:
        raise ValueError("unique session/k IDs and a selected-volume reference required")
    ref_pose = by_id[reference]["pose"]
    poses = sorted({r["pose"] for r in rows})
    other = [p for p in poses if p != ref_pose]
    if not 0 < fraction < 1 or len(poses) < 5:
        raise ValueError("need at least five distinct poses and holdout fraction between zero and one")
    count = min(len(poses) - 2, max(3, round(len(poses) * fraction)))
    rng = np.random.default_rng(seed)
    held = {other[i] for i in rng.permutation(len(other))[:count]}
    train_ids = [tuple(r["id"]) for r in rows if r["pose"] not in held]
    return train_ids, held, {"seed": seed, "unit": "actual pan/tilt pose across all supplied sessions",
                             "reference_pose_forced_training": list(ref_pose),
                             "training_poses": [list(p) for p in poses if p not in held],
                             "held_out_poses": [list(p) for p in poses if p in held]}


def _triple(text):
    v = tuple(float(x) for x in text.split(","))
    if len(v) != 3:
        raise argparse.ArgumentTypeError("three comma-separated numbers required")
    return v


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--scan", type=Path, action="append", required=True, help="raw session with log.jsonl and PCM; repeatable")
    p.add_argument("--probe", type=Path, required=True, help="16 kHz .npy transmitted sweep")
    p.add_argument("--calibration", type=Path, required=True, help="motor-to-angle calibration JSON")
    p.add_argument("--rig", type=Path, help="JSON overrides for Rig offsets in metres; assumptions are recorded")
    p.add_argument("--reference", required=True, help="session-directory-name:recording-index; chosen before audio fitting")
    p.add_argument("--normal", type=_triple, default=(0., -1., 0.), help="fixed unit plane normal (default downward/floor)")
    p.add_argument("--distance-min", type=float, default=.60)
    p.add_argument("--distance-max", type=float, default=1.05)
    p.add_argument("--distance-step", type=float, default=.005)
    p.add_argument("--timing-bound-us", type=float, default=250., help="assumed common reference uncertainty, not measured timing accuracy")
    p.add_argument("--timing-steps", type=int, default=11)
    p.add_argument("--window-ms", type=float, nargs=2, default=(3.0, 7.5))
    p.add_argument("--volume", type=int, default=40)
    p.add_argument("--split-seed", type=int, default=42)
    p.add_argument("--holdout-fraction", type=float, default=.3)
    p.add_argument("--permutations", type=int, default=199)
    p.add_argument("--output", type=Path, required=True, help="new private output directory, normally under eval/")
    return p


def run(args):
    out = args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("output directory is not empty; use a fresh run directory")
    if args.distance_min <= 0 or args.distance_max <= args.distance_min or args.distance_step <= 0:
        raise ValueError("invalid distance grid")
    if args.timing_bound_us < 0 or args.timing_steps < 1 or args.permutations < 1:
        raise ValueError("invalid timing grid or permutation count")
    # The early alignment is part of the acquisition model; fitting within that
    # window would credit the fitter for differences that alignment already used.
    earliest_fit_s = ALIGNMENT_WINDOW_S[1] + MAX_ALIGNMENT_LAG_S + ALIGNMENT_GUARD_S
    if args.window_ms[0] / 1000 < earliest_fit_s:
        raise ValueError(f"fit window must start at or after {earliest_fit_s * 1000:g} ms: alignment window + lag search + guard")
    name, k = args.reference.rsplit(":", 1)
    reference = (name, int(k))
    scans = [s.resolve() for s in args.scan]
    train_ids, held, split = split_poses(scans, reference, args.holdout_fraction, args.split_seed, args.volume)
    rig = Rig(**(json.loads(args.rig.read_text()) if args.rig else {}))
    calibration = json.loads(args.calibration.read_text())
    ds = np.arange(args.distance_min, args.distance_max + args.distance_step * .1, args.distance_step)
    offsets = (np.linspace(-args.timing_bound_us, args.timing_bound_us, args.timing_steps) * 1e-6
               if args.timing_steps > 1 else np.array([0.]))
    out.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[2]
    sources = sorted(Path(__file__).parent.glob("*.py"))
    protocol = {"version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
                "command": sys.argv, "split": split, "reference": list(reference),
                "surface_normal": args.normal, "distance_grid_m": ds.tolist(), "timing_grid_s": offsets.tolist(),
                "window_s": [v / 1000 for v in args.window_ms], "rig_assumed": asdict(rig),
                "calibration": calibration, "calibration_sha256": digest(args.calibration),
                "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                "source_sha256": {str(p.relative_to(root)): digest(p) for p in sources},
                "versions": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__},
                "selection": "all candidate selection uses training poses only; fixed normal/window/band",
                "distance_claim": "conditional on assumed acoustic centres and prominent-path timing; no measured surface truth"}
    # Written before reading audio; preserve this protocol even if extraction fails.
    write_json(out / "protocol.json", protocol)
    dataset = prepare_recordings(scans, args.probe, reference, designated_training=train_ids,
                                 primary_volume=args.volume, alignment_window_s=ALIGNMENT_WINDOW_S,
                                 max_lag_s=MAX_ALIGNMENT_LAG_S)
    save_dataset(dataset, out / "waveforms")
    # Equal weight per pose. A stationary block with many repeats must not count
    # more than a pose measured once. Every constituent waveform remains saved.
    poses = sorted({tuple(q["pose"]) for q in dataset.recordings})
    waves, groups, ss, rr = [], [], [], []
    for pose in poses:
        ii = [i for i, q in enumerate(dataset.recordings) if tuple(q["pose"]) == pose]
        waves.append(dataset.h_aligned[ii].mean(axis=0))
        pp = rig.pose_motor(calibration, *pose)
        ss.append(pp.source_m); rr.append(pp.receiver_m)
        groups.append({"pose": list(pose), "training": pose not in held,
                       "recording_indices": ii,
                       "recording_ids": [[dataset.recordings[i]["session"], dataset.recordings[i]["k"]] for i in ii]})
    fitted = fit_surface(np.asarray(waves), dataset.time_s, dataset.inverse_kernel,
                         np.asarray(ss), np.asarray(rr), args.normal,
                         np.array([g["training"] for g in groups]), ds, offsets,
                         tuple(v / 1000 for v in args.window_ms))
    fitted["pose_association_control"] = permutation_check(fitted, args.permutations, args.split_seed)
    arrays = fitted.pop("arrays")
    np.savez_compressed(out / "fit_arrays.npz", **arrays, source_m=ss, receiver_m=rr,
                        grouped_waveforms=waves, full_time_s=dataset.time_s,
                        grouped_complex_spectra=np.fft.rfft(waves, axis=-1))
    fitted["groups"] = groups
    fitted["alignment_bound_hits"] = dataset.metadata["alignment_bound_hits"]
    fitted["protocol_sha256"] = digest(out / "protocol.json")
    fitted["fit_arrays_sha256"] = digest(out / "fit_arrays.npz")
    fitted["limitations"] = [
        "This is a one-plane image-source experiment, not a complete room or diffraction inversion.",
        "Rig acoustic centres and motor angle mapping are assumed; optical calibration is not acoustic calibration.",
        "All recordings were aligned to one early training template; this path is not established as physical direct arrival.",
        "The timing bound is a sensitivity assumption, not a measured confidence bound.",
        "The shared background and one complex reflection coefficient are fitted only on training poses.",
        "Signed data and predicted kernels receive the same fixed time window/taper; no per-candidate alignment.",
        "Pose-held-out results from one session do not establish transfer across sessions or a physical distance error.",
        "Only direction-dependent improvement over a shared response is tested; a strong common echo is not sufficient.",
        "The fitted normal is fixed by configuration; hidden-object shape/location is outside this experiment."]
    gain = fitted["held_out"]["gain_vs_shared_background"]
    null = fitted["pose_association_control"]
    fitted["interpretation"] = ("candidate_directional_evidence_requires_independent_calibration"
                                if gain is not None and gain > 0 and null.get("p_lower_sse", 1) < .05
                                and fitted["status"] == "experimental_fit" else "no_validated_surface_measurement")
    write_json(out / "fit.json", fitted)
    print(json.dumps({"output": str(out), "recordings": len(dataset.recordings), "pose_groups": len(groups),
                      "selected": fitted["selected"], "held_out": fitted["held_out"],
                      "permutation_p": null.get("p_lower_sse"), "interpretation": fitted["interpretation"],
                      "near_optimal_distance_band_m": fitted["near_optimal_distance_band_m"]}, indent=2))
    return fitted


def main():
    args = parser().parse_args()
    try:
        run(args)
    except (ValueError, OSError) as e:
        raise SystemExit(f"acoustic experiment: {e}") from e


if __name__ == "__main__":
    main()
