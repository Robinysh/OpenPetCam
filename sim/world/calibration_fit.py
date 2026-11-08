"""Fit a radial lens profile from saved sweep matches, with capture provenance.

No pictures leave this computer and no camera commands are sent. Reserve a
different capture for calibration_validation before choosing a new profile.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import scipy

from .camera import fit_matches, calibration_residuals, Camera
from .calibration_validation import load_matches, sweep_hashes


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run(sweep, matches, out, initial=None, prior_deg=3., max_nfev=300):
    poses_path = Path(sweep) / "poses.json"
    poses = json.loads(poses_path.read_text())["poses"]
    pairs = load_matches(matches, poses)
    guess = json.loads(Path(initial).read_text()) if initial else None
    started = time.monotonic()
    profile, offsets, result = fit_matches(poses, pairs, guess, prior_deg, max_nfev)
    errors = np.linalg.norm(calibration_residuals(Camera(profile), poses, pairs, offsets, scale=1).reshape(-1, 2), axis=1)
    profile.update(frames=[p["file"] for p in poses], offsets_deg=offsets.tolist(),
                   median_px=float(np.median(errors)))
    profile["fit"] = {
        "success": bool(result.success), "message": result.message,
        "nfev": int(result.nfev), "cost": float(result.cost),
        "elapsed_s": time.monotonic() - started,
        "error_scope": "in-sample symmetric point error; not independent validation",
        "objective": "all fixed homography-filtered matches; no residual-based pair exclusion",
        "training_sweep": str(sweep), "training_matches": str(matches),
        "poses_sha256": digest(poses_path), "matches_sha256": digest(matches),
        "frame_sha256": sorted(sweep_hashes(sweep)),
        "source_sha256": {p.name: digest(p) for p in (Path(__file__), Path(__file__).with_name("camera.py"),
                                                     Path(__file__).with_name("calibration_validation.py"))},
        "initial_sha256": digest(initial) if initial else None,
        "pairs": len(pairs), "matches": sum(len(a) for _, _, a, _ in pairs),
        "prior_deg": prior_deg, "numpy_version": np.__version__, "scipy_version": scipy.__version__,
    }
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(profile, indent=2) + "\n")
    print(json.dumps({"profile": {k: profile[k] for k in ("focal_px", "k1", "k2", "median_px")},
                      "solver": {k: profile["fit"][k] for k in ("success", "message", "nfev", "cost", "elapsed_s")},
                      "out": str(out)}, indent=2))
    return profile


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sweep", required=True)
    ap.add_argument("--matches", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--initial")
    ap.add_argument("--prior-deg", type=float, default=3.)
    ap.add_argument("--max-nfev", type=int, default=300)
    a = ap.parse_args()
    profile = run(a.sweep, a.matches, a.out, a.initial, a.prior_deg, a.max_nfev)
    if not profile["fit"]["success"]:
        raise SystemExit("Fit did not converge; output is diagnostic evidence, not an accepted profile")


if __name__ == "__main__":
    main()
