"""Offline, conditional single-channel room-fingerprint matching.

An arbitrary unknown source can explain a recording at any invertible transfer
function. This scorer imposes a *smooth source spectrum* prior, retaining room
comb structure after removing a smooth log-power envelope. Harmonic sources,
unknown microphone coloration and room mismatch can violate that prior. Scores
are not probabilities and never establish a physical location on their own.
Original signed waveforms are not modified or replaced by these features.
"""
import hashlib
import math

import numpy as np
from scipy import ndimage, signal


def spectrum(samples, sample_rate, *, nfft=8192):
    """Welch power on a native sample grid, with no gain/denoising/alignment."""
    x = np.asarray(samples, dtype=float)
    if x.ndim != 1 or len(x) < 128 or not np.isfinite(x).all():
        raise ValueError("finite mono samples, at least 128, required")
    if not math.isfinite(sample_rate) or sample_rate <= 0 or nfft < 256:
        raise ValueError("positive sample rate and nfft >= 256 required")
    segment = min(len(x), nfft)
    return signal.welch(x, fs=sample_rate, window="hann", nperseg=segment,
                        noverlap=segment // 2, nfft=nfft, detrend="constant",
                        scaling="spectrum")


def ripple_features(power, frequencies, *, smooth_hz=300., band=(300., 7000.),
                    denoise_hz=12.):
    """Remove smooth source envelope; return normalized ripple plus its RMS.

    This normalization applies to matching features, not archived audio or room
    impulse responses. Frequency smoothing is fixed by the frozen protocol.
    """
    p, f = np.asarray(power, float), np.asarray(frequencies, float)
    if f.ndim != 1 or len(f) < 32 or p.shape[-1:] != f.shape:
        raise ValueError("power's final axis must match the frequency grid")
    if not np.isfinite(p).all() or np.any(p < 0) or not np.isfinite(f).all():
        raise ValueError("finite nonnegative power and finite frequencies required")
    df = f[1] - f[0]
    if df <= 0 or not np.allclose(np.diff(f), df) or smooth_hz <= denoise_hz or denoise_hz < 0:
        raise ValueError("uniform frequency grid and valid smoothing widths required")
    keep = (f >= band[0]) & (f <= band[1])
    if keep.sum() < 16:
        raise ValueError("insufficient frequency support in comparison band")
    floor = np.maximum(np.max(p, axis=-1, keepdims=True) * 1e-10, 1e-30)
    logp = np.log(np.maximum(p, floor))
    if denoise_hz:
        logp = ndimage.gaussian_filter1d(logp, denoise_hz / df, axis=-1, mode="reflect")
    envelope = ndimage.gaussian_filter1d(logp, smooth_hz / df, axis=-1, mode="reflect")
    ripple = (logp - envelope)[..., keep]
    ripple -= ripple.mean(axis=-1, keepdims=True)
    norms = np.linalg.norm(ripple, axis=-1, keepdims=True)
    features = np.divide(ripple, norms, out=np.zeros_like(ripple), where=norms > 1e-10)
    return features, (norms[..., 0] / np.sqrt(keep.sum()))


def rank_candidates(observed_feature, model_features, positions_m, *, near_score=.02,
                    min_margin=None, min_similarity=None):
    """Profile predeclared material variants; report spatial ambiguity honestly.

    Features have shape (positions, variants, frequencies), or (positions,
    frequencies). All positions are returned in the score profile. Optional
    acceptance thresholds must come from development data, never this recording.
    """
    obs, features, pos = (np.asarray(v, float) for v in
                          (observed_feature, model_features, positions_m))
    if features.ndim == 2:
        features = features[:, None, :]
    if features.ndim != 3 or obs.shape != (features.shape[-1],) or pos.shape != (len(features), 3):
        raise ValueError("feature/position dimensions disagree")
    if len(pos) < 2 or not all(np.isfinite(v).all() for v in (obs, features, pos)):
        raise ValueError("at least two finite candidate positions required")
    if near_score < 0:
        raise ValueError("near_score must be nonnegative")
    valid = np.linalg.norm(features, axis=-1) > 1e-8
    scores = np.einsum("pvd,d->pv", features, obs, optimize=True)
    scores = np.where(valid, np.clip(scores, -1, 1), -np.inf)
    profile = scores.max(axis=1)
    if np.linalg.norm(obs) < 1e-8 or not np.isfinite(profile).any():
        return {"status": "uninformative", "accepted": False, "best_index": None,
                "position_m": None, "scores": [None] * len(pos),
                "reason": "No room-dependent spectral contrast; gain alone cannot range an unknown source."}
    order = np.argsort(-profile, kind="stable")
    best, second = int(order[0]), int(order[1])
    margin = float(profile[best] - profile[second]) if np.isfinite(profile[second]) else None
    near = np.flatnonzero(profile >= profile[best] - near_score)
    # The box diagonal is a conservative upper bound, explicitly not a confidence region.
    spread = float(np.linalg.norm(np.ptp(pos[near], axis=0)))
    accepted = (min_margin is not None and min_similarity is not None and margin is not None and margin > 1e-9
                and margin >= min_margin and profile[best] >= min_similarity)
    return {"status": "conditional_candidate", "accepted": bool(accepted),
            "best_index": best, "position_m": pos[best].tolist(),
            "best_variant": int(np.argmax(scores[best])), "similarity": float(profile[best]),
            "score_margin": margin, "near_best_candidates": near.tolist(),
            "near_best_box_diagonal_m": spread, "near_score_tolerance": near_score,
            "uncertainty_kind": "descriptive score set, NOT a confidence interval",
            "scores": [float(x) if np.isfinite(x) else None for x in profile]}


def spatial_partition(positions_m, *, tile_m=1., seed=731):
    """Whole spatial tiles share a split; adjacent samples never split a tile."""
    pos = np.asarray(positions_m, float)
    if pos.ndim != 2 or pos.shape[1] != 3 or not np.isfinite(pos).all() or tile_m <= 0:
        raise ValueError("finite Nx3 positions and positive tile size required")
    result = []
    for point in pos:
        # Height variants at one floor location must remain in one split.
        tile = np.floor(point[[0, 2]] / tile_m).astype(int)
        digest = hashlib.sha256(f"{seed}:{tile[0]}:{tile[1]}".encode()).digest()
        result.append("development" if int.from_bytes(digest[:4], "big") % 3 == 0 else "held_out")
    return result


def errors(predicted_m, true_m, receiver_m=(0., 0., 0.)):
    p, t, r = (np.asarray(x, float) for x in (predicted_m, true_m, receiver_m))
    spatial = float(np.linalg.norm(p - t))
    pa = math.atan2(p[0] - r[0], p[2] - r[2])
    ta = math.atan2(t[0] - r[0], t[2] - r[2])
    az = abs(math.degrees(math.atan2(math.sin(pa - ta), math.cos(pa - ta))))
    return {"distance_error_m": spatial, "azimuth_error_deg": az}


def summarize_errors(rows):
    """Keep missing predictions in coverage and success denominators.

    Quantiles describe localized cases only. Missing coordinates have no invented
    error: their error stays null and they count as unsuccessful in all-case
    success fractions. This is not a real-room accuracy measurement.
    """
    if not rows:
        return {"cases": 0}
    localized = [r for r in rows if r.get("distance_error_m") is not None]
    err = np.array([r["distance_error_m"] for r in localized])
    az = np.array([r["azimuth_error_deg"] for r in localized])
    accepted = [r for r in localized if r.get("accepted")]
    return {"cases": len(rows), "localized_cases": len(localized),
            "unlocalized_cases": len(rows) - len(localized), "localization_fraction": len(localized) / len(rows),
            "error_quantile_population": "localized cases only; unlocalized errors remain null",
            "median_error_m": float(np.median(err)) if len(err) else None,
            "p90_error_m": float(np.quantile(err, .9)) if len(err) else None,
            "within_1m_fraction": int(np.sum(err <= 1)) / len(rows),
            "median_azimuth_deg": float(np.median(az)) if len(az) else None,
            "within_30deg_fraction": int(np.sum(az <= 30)) / len(rows),
            "accepted_cases": len(accepted), "acceptance_fraction": len(accepted) / len(rows),
            "accepted_median_error_m": float(np.median([r["distance_error_m"] for r in accepted])) if accepted else None,
            "accepted_within_1m_fraction": float(np.mean([r["distance_error_m"] <= 1 for r in accepted])) if accepted else None}
