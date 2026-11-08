"""A single planar reflection fitted to signed waveforms, with pose holdouts.

The forward model is the exact image-source path for an infinite rigid plane in
homogeneous air. It does not model housing diffraction or a furnished room.
An arbitrary *shared* background waveform is learned on training poses. A plane
must explain the remaining pose-dependent signal with one shared complex gain.
No per-recording gain, delay, or frequency response is fitted here.

Distances are conditional on the input's early reference being the direct path.
Profile an explicit common timing uncertainty rather than hiding it in a flexible
complex transfer function. A flat profile is not a measured distance.
"""
import numpy as np
from scipy import signal

from .geometry import one_surface_paths


def _kernel_grid(time_s, kernel):
    t, k = np.asarray(time_s, float), np.asarray(kernel, float)
    if t.ndim != 1 or k.shape != t.shape or len(t) < 16:
        raise ValueError("time and real kernel must be matching 1D arrays")
    dt = float(t[1] - t[0])
    if dt <= 0 or not np.allclose(np.diff(t), dt, rtol=1e-6, atol=1e-12):
        raise ValueError("uniform increasing time grid required")
    if not np.isfinite(k).all() or not np.any(k):
        raise ValueError("finite, nonzero measurement kernel required")
    # Padding avoids wrapping a shifted pulse into the other end of the record.
    n = len(k)
    padded = np.pad(k, (n, n))
    analytic = signal.resample(signal.hilbert(padded), 8 * len(padded))
    return t[0] - n * dt + np.arange(len(analytic)) * dt / 8, analytic


def _template(grid, sources, receivers, normal, distance, offset, time, speed):
    paths = one_surface_paths(sources, receivers, normal, float(distance))
    if not np.all(paths.valid):
        raise ValueError("plane must leave all transducers on the same valid side")
    delay = (paths.reflected_m - paths.direct_m) / speed + offset
    kt, kv = grid
    u = time[None, :] - delay[:, None]
    q = (np.interp(u, kt, kv.real, left=0, right=0)
         + 1j * np.interp(u, kt, kv.imag, left=0, right=0))
    return q / paths.reflected_m[:, None], delay


def surface_templates(time_s, kernel, sources_m, receivers_m, normal, distance_m,
                      offset_s=0.0, sound_speed=343.0):
    """Complex quadrature templates after the acquisition deconvolution kernel."""
    t = np.asarray(time_s, float)
    return _template(_kernel_grid(t, kernel), np.asarray(sources_m, float),
                     np.asarray(receivers_m, float), normal, distance_m, offset_s,
                     t, sound_speed)[0]


def _score(y, predicted, baseline, mask):
    if not np.any(mask):
        return None
    sse = float(np.sum((y[mask] - predicted[mask]) ** 2))
    base = float(np.sum((y[mask] - baseline) ** 2))
    return {"records": int(np.sum(mask)), "sse": sse, "baseline_sse": base,
            "gain_vs_shared_background": 1 - sse / base if base > 1e-24 else None}


def fit_surface(observed, time_s, kernel, sources_m, receivers_m, normal,
                train_mask, distances_m, offsets_s=(0.0,), window_s=(0.0005, 0.012),
                sound_speed=343.0):
    """Select distance, common timing offset and complex gain on training only.

    Returns JSON-compatible metrics/profiles plus arrays under ``arrays``.
    Training poses must be separated by the caller; repeated sweeps should first
    be averaged within a recording. Do not count them as independent pose tests.
    """
    t, y = np.asarray(time_s, float), np.asarray(observed, float)
    s, r = np.asarray(sources_m, float), np.asarray(receivers_m, float)
    train = np.asarray(train_mask, bool)
    ds, offsets = np.asarray(distances_m, float), np.asarray(offsets_s, float)
    if y.ndim != 2 or y.shape[1] != len(t) or s.shape != (len(y), 3) or r.shape != s.shape:
        raise ValueError("waveforms and Nx3 transducer positions must have matching rows")
    if train.shape != (len(y),) or train.sum() < 2 or (~train).sum() < 1:
        raise ValueError("at least two training records and one held-out record required")
    if not all(np.isfinite(a).all() for a in (y, s, r, ds, offsets)):
        raise ValueError("nonfinite input")
    if ds.ndim != 1 or offsets.ndim != 1 or not len(ds) or not len(offsets) or np.any(ds <= 0):
        raise ValueError("positive distance grid and nonempty offset grid required")
    if sound_speed <= 0 or window_s[0] >= window_s[1]:
        raise ValueError("invalid speed or window")
    grid = _kernel_grid(t, kernel)
    keep = (t >= window_s[0]) & (t <= window_s[1])
    if keep.sum() < 32:
        raise ValueError("comparison window needs at least 32 samples")
    tt = t[keep]
    taper = signal.windows.tukey(len(tt), 0.2)
    yw = y[:, keep] * taper
    background = yw[train].mean(axis=0)
    residual = yw[train] - background
    base_energy = float(np.sum(residual ** 2))
    rows, best, best_loss = [], None, np.inf
    for distance in ds:
        for offset in offsets:
            try:
                q, delays = _template(grid, s, r, normal, distance, offset, tt, sound_speed)
            except ValueError:
                rows.append({"distance_m": float(distance), "offset_s": float(offset), "valid": False,
                             "reason": "transducers cross or touch candidate plane"})
                continue
            # Entire pulse stays inside a fixed window; never clip around each
            # candidate's best echo. A small guard covers the tapered wavelet.
            guard = 0.00025
            if np.any(delays < tt[0] + guard) or np.any(delays > tt[-1] - guard):
                rows.append({"distance_m": float(distance), "offset_s": float(offset), "valid": False,
                             "reason": "predicted arrival outside comparison window guard"})
                continue
            q *= taper
            qm = q[train].mean(axis=0)
            qc = q - qm
            # Re((a + i*b)*q) = a*Re(q) - b*Im(q).
            X = np.stack((qc.real, -qc.imag), axis=-1)
            design = X[train].reshape(-1, 2)
            singular = np.linalg.svd(design, compute_uv=False)
            energy = float(np.sum(np.abs(q[train]) ** 2))
            contrast = float(np.sum(design ** 2) / max(energy, 1e-30))
            if contrast < 1e-12:
                coeff = np.zeros(2)
                rank = 0
            else:
                coeff, _, rank, _ = np.linalg.lstsq(design, residual.ravel(), rcond=1e-8)
            prediction = background + X @ coeff
            loss = float(np.sum((yw[train] - prediction[train]) ** 2))
            row = {"distance_m": float(distance), "offset_s": float(offset), "valid": True,
                   "training_sse": loss, "training_gain": 1 - loss / base_energy if base_energy > 1e-24 else None,
                   "coefficient_real": float(coeff[0]), "coefficient_imag": float(coeff[1]),
                   "rank": int(rank), "pose_contrast_fraction": contrast,
                   "design_singular_values": singular.tolist()}
            rows.append(row)
            if loss < best_loss:
                best_loss, best = loss, (row, prediction, X @ coeff)
    if best is None:
        raise ValueError("no candidate plane has valid paths inside the fixed comparison window")
    selected, predicted, contribution = best
    selected_paths = one_surface_paths(s, r, normal, selected["distance_m"])
    path_signatures = np.round(np.column_stack((selected_paths.direct_m, selected_paths.reflected_m)), 9)
    valid = [v for v in rows if v["valid"]]
    profile = [min((v for v in valid if v["distance_m"] == float(d)),
                   key=lambda v: v["training_sse"]) for d in ds
               if any(v["distance_m"] == float(d) for v in valid)]
    timing_profile = [min((v for v in valid if v["offset_s"] == float(o)),
                          key=lambda v: v["training_sse"]) for o in offsets
                      if any(v["offset_s"] == float(o) for v in valid)]
    train_score, test_score = _score(yw, predicted, background, train), _score(yw, predicted, background, ~train)
    # Descriptive sensitivity band, not a confidence interval. Its scale is
    # training residual energy, chosen without examining held-out observations.
    near = [v["distance_m"] for v in profile
            if v["training_sse"] <= best_loss + 0.01 * max(base_energy, 1e-30)]
    status = "no_pose_contrast" if selected["rank"] == 0 else "experimental_fit"
    if base_energy <= 1e-24:
        status = "no_pose_dependent_signal"
    return {"model": "single_infinite_plane_image_source_shared_complex_gain",
            "status": status, "selected": selected, "train": train_score, "held_out": test_score,
            "window_s": list(window_s), "sound_speed_m_s": sound_speed,
            "surface_normal": np.asarray(normal, float).tolist(), "candidates": rows,
            "distinct_path_signatures": {"all": int(len(np.unique(path_signatures, axis=0))),
                                         "training": int(len(np.unique(path_signatures[train], axis=0))),
                                         "held_out": int(len(np.unique(path_signatures[~train], axis=0))),
                                         "note": "symmetry can give many motor poses identical direct/reflected lengths; not independent spatial diversity"},
            "distance_profile": profile, "timing_profile": timing_profile,
            "near_optimal_distance_band_m": [min(near), max(near)],
            "near_optimal_band_definition": "within 1% of training background residual energy; NOT a confidence interval",
            "distance_at_grid_boundary": selected["distance_m"] in (float(ds.min()), float(ds.max())),
            "timing_at_grid_boundary": len(offsets) > 1 and selected["offset_s"] in (float(offsets.min()), float(offsets.max())),
            "arrays": {"time_s": tt, "observed_windowed": yw, "predicted_windowed": predicted,
                       "background_windowed": background, "surface_contribution_windowed": contribution,
                       "train_mask": train}}


def permutation_check(result, permutations=199, seed=0):
    """Frozen-fit held-out pose-association control; never selects a new model.

    A negative result rejects directional evidence. Passing remains a diagnostic,
    because empirical poses/scene observations may have nuisance correlations.
    """
    a = result["arrays"]
    test = ~a["train_mask"]
    y, b = a["observed_windowed"][test], a["background_windowed"]
    c = a["surface_contribution_windowed"][test]
    if len(y) < 3:
        return {"available": False, "reason": "at least three held-out pose groups required"}
    true = float(np.sum((y - b - c) ** 2))
    rng = np.random.default_rng(seed)
    losses = [float(np.sum((y - b - c[rng.permutation(len(c))]) ** 2)) for _ in range(permutations)]
    return {"available": True, "permutations": permutations, "seed": seed, "matched_sse": true,
            "shuffled_sse_median": float(np.median(losses)),
            "p_lower_sse": float((1 + np.sum(np.asarray(losses) <= true)) / (1 + permutations)),
            "method": "permute frozen surface predictions among held-out groups; no refitting",
            "shuffled_sse": losses}
