"""Per-event acquisition QC without stacking or cross-pose alignment.

Search windows and quiet windows are supplied by the acquisition protocol. A
correlation peak is an acoustic anchor, never proof of the DAC emission time.
The original PCM remains authoritative; reported sample bounds recover every
evaluated segment without normalizing its amplitude.
"""
from dataclasses import asdict, dataclass
from fractions import Fraction
from typing import Sequence

import numpy as np
from scipy import signal


@dataclass(frozen=True)
class ShotWindow:
    event_id: str
    source_start_sample: int
    search_start_sample: int
    search_stop_sample: int
    noise_start_sample: int
    noise_stop_sample: int


@dataclass(frozen=True)
class QCConfig:
    min_snr_db: float = 6.0
    min_normalized_correlation: float = 0.15
    max_clipped_samples: int = 0
    max_noise_transient_ratio: float = 8.0
    max_timing_residual_samples: float = 3.0
    noise_frame_s: float = 0.01
    bands_hz: tuple = ((2000., 4000.), (4000., 6000.), (6000., 7400.))


def _samples(pcm):
    if isinstance(pcm, (bytes, bytearray, memoryview)):
        if len(pcm) % 2:
            raise ValueError("odd byte count for mono s16le PCM")
        raw = np.frombuffer(pcm, dtype="<i2")
        x = raw.astype(float) / 32768.0
        clipping = (raw == -32768) | (raw == 32767)
    else:
        raw = np.asarray(pcm)
        if raw.dtype == np.int16:
            x = raw.astype(float) / 32768.0
            clipping = (raw == -32768) | (raw == 32767)
        else:
            x = np.asarray(raw, float)
            clipping = (x <= -1.) | (x >= 32767. / 32768.)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("PCM must be a nonempty finite mono vector")
    if np.max(np.abs(x)) > 1:
        raise ValueError("floating PCM must retain the original [-1,1] scaling")
    return x, clipping


def _config(config):
    q = config or QCConfig()
    values = (q.min_snr_db, q.min_normalized_correlation,
              q.max_noise_transient_ratio, q.max_timing_residual_samples,
              q.noise_frame_s)
    if not np.isfinite(values).all() or not 0 < q.min_normalized_correlation <= 1:
        raise ValueError("invalid QC thresholds")
    if q.max_clipped_samples < 0 or q.max_noise_transient_ratio <= 1:
        raise ValueError("invalid clipping or transient threshold")
    if q.max_timing_residual_samples <= 0 or q.noise_frame_s <= 0:
        raise ValueError("invalid timing or frame threshold")
    return q


def fit_packet_timing(detections, *, source_rate_hz, sample_rate_hz):
    """Fit an acoustic-anchor ADC = intercept + slope * source-sample map.

    Use separate bracketing pilot detections when testing shot timing. A fit to
    two anchors has no residual degrees of freedom; its interpolation must be
    checked against held-out events. This function never certifies physical t0.
    """
    if source_rate_hz <= 0 or sample_rate_hz <= 0 or len(detections) < 2:
        raise ValueError("positive rates and at least two anchors required")
    ids = [d["event_id"] for d in detections]
    if len(set(ids)) != len(ids) or any(not d.get("accepted", False) for d in detections):
        raise ValueError("timing anchors must be unique accepted detections")
    src = np.array([d["source_start_sample"] for d in detections], float)
    adc = np.array([d["detected_start_sample_float"] for d in detections], float)
    if not np.isfinite(src).all() or not np.isfinite(adc).all() or np.ptp(src) <= 0:
        raise ValueError("finite distinct source positions required")
    slope = float(np.dot(src - src.mean(), adc - adc.mean()) / np.sum((src - src.mean()) ** 2))
    intercept = float(adc.mean() - slope * src.mean())
    if slope <= 0:
        raise ValueError("nonpositive clock slope")
    residual = adc - (intercept + slope * src)
    return {"slope": slope, "intercept": intercept,
            "anchor_event_ids": ids, "source_rate_hz": int(source_rate_hz),
            "sample_rate_hz": int(sample_rate_hz),
            "relative_rate_error_ppm": float((slope / (sample_rate_hz / source_rate_hz) - 1) * 1e6),
            "anchor_residual_samples": residual.tolist(),
            "residual_degrees_of_freedom": len(src) - 2,
            "timing_status": "acoustic_anchor_relative",
            "absolute_timing_calibrated": False,
            "limitation": "Includes acoustic propagation and hardware delay; correlation alone cannot establish emission time."}


def repeat_coherence(traces, *, sample_rate_hz, bands_hz):
    """Phase-and-amplitude consistency at a supplied common packet time origin.

    No shifting or per-trace normalization occurs here. A single trace cannot
    establish repeat coherence. This is not source/receiver transfer coherence.
    """
    y = np.asarray(traces, float)
    if y.ndim != 2 or y.shape[0] < 2 or y.shape[1] < 16 or not np.isfinite(y).all():
        raise ValueError("at least two matching finite repeated traces required")
    z = np.fft.rfft(y * signal.windows.tukey(y.shape[1], .1), axis=1)
    f = np.fft.rfftfreq(y.shape[1], 1 / sample_rate_hz)
    coherent = np.abs(z.mean(axis=0)) ** 2
    total = np.mean(np.abs(z) ** 2, axis=0)
    result = []
    for low, high in bands_hz:
        if not 0 <= low < high <= sample_rate_hz / 2:
            raise ValueError("coherence bands must be inside Nyquist")
        use = (f >= low) & (f < high)
        energy = float(total[use].sum())
        result.append({"band_hz": [float(low), float(high)],
                       "coherent_energy_fraction": float(np.clip(coherent[use].sum() / energy, 0, 1)) if energy > 0 else None,
                       "mean_spectral_energy": energy, "frequency_bins": int(use.sum())})
    rms = np.sqrt(np.mean(y ** 2, axis=1))
    return {"trace_count": len(y), "bands": result, "rms_per_trace": rms.tolist(),
            "rms_coefficient_of_variation": float(rms.std() / rms.mean()) if rms.mean() else None,
            "alignment": "provided common packet map only; no individual realignment",
            "normalization": "none"}


def mapped_window(samples, start_sample, length):
    """Evaluate one supplied fractional map origin using padded Fourier interpolation.

    The origin must come from a common source/pilot map, not the measured shot
    arrival. No best-lag search, gain adjustment or raw-data mutation occurs.
    """
    x = np.asarray(samples, float)
    if (x.ndim != 1 or not np.isfinite(start_sample) or not isinstance(length, (int, np.integer))
            or length < 1 or start_sample < 0 or start_sample + length > len(x)):
        raise ValueError("fractional common-map window is outside captured PCM")
    lo = int(np.floor(start_sample))
    fraction = float(start_sample - lo)
    if fraction == 0:
        return x[lo:lo + length].copy()
    guard = 256
    begin, end = max(0, lo - guard), min(len(x), lo + length + guard + 1)
    segment = x[begin:end]
    n = len(segment)
    nfft = 1 << (3 * n - 1).bit_length()
    padded = np.zeros(nfft)
    padded[n:2 * n] = segment
    phase = np.exp(2j * np.pi * np.fft.rfftfreq(nfft) * fraction)
    shifted = np.fft.irfft(np.fft.rfft(padded) * phase, nfft)
    return shifted[n + lo - begin:n + lo - begin + length]


def assess_capture(pcm, *, sample_rate_hz, source_rate_hz, probe,
                   shots: Sequence[ShotWindow], response_tail_s=.15,
                   config=None, timing_map=None):
    """Return JSON-compatible QC for explicit event onset-search/quiet windows.

    ``search_start_sample:search_stop_sample`` enumerates possible ADC onset
    indices, not complete signal segments. The complete probe plus response tail
    must fit even at the last allowed onset. Quiet windows must not overlap any
    event's possible signal interval. Pilot-derived ``timing_map`` predictions
    are evaluated without shifting each shot to its measured correlation peak.
    """
    cfg = _config(config)
    x, clipped = _samples(pcm)
    if (not isinstance(sample_rate_hz, int) or not isinstance(source_rate_hz, int)
            or sample_rate_hz <= 0 or source_rate_hz <= 0):
        raise ValueError("positive integer sample rates required")
    p = np.asarray(probe, float)
    if p.ndim != 1 or len(p) < 16 or not np.isfinite(p).all() or not np.any(p):
        raise ValueError("finite nonzero source probe required")
    if not np.isfinite(response_tail_s) or response_tail_s < 0:
        raise ValueError("invalid response tail")
    rate = Fraction(sample_rate_hz, source_rate_hz)
    p = signal.resample_poly(p, rate.numerator, rate.denominator)
    n = len(p)
    tail = round(response_tail_s * sample_rate_hz)
    if not shots or len({s.event_id for s in shots}) != len(shots):
        raise ValueError("nonempty unique event windows required")
    if len({s.source_start_sample for s in shots}) != len(shots):
        raise ValueError("source event positions must be distinct")
    intervals = []
    for s in shots:
        fields = (s.source_start_sample, s.search_start_sample, s.search_stop_sample,
                  s.noise_start_sample, s.noise_stop_sample)
        if not all(isinstance(v, (int, np.integer)) for v in fields) or not s.event_id:
            raise ValueError("event IDs and integer sample bounds required")
        if s.source_start_sample < 0 or not 0 <= s.search_start_sample < s.search_stop_sample:
            raise ValueError("invalid source/search position")
        stop = s.search_stop_sample - 1 + n + tail
        if stop > len(x):
            raise ValueError(f"truncated probe/response window for {s.event_id}")
        if not 0 <= s.noise_start_sample < s.noise_stop_sample <= len(x) or s.noise_stop_sample - s.noise_start_sample < 32:
            raise ValueError("quiet window requires at least 32 in-bounds samples")
        intervals.append((s.search_start_sample, stop))
    ordered = sorted(intervals)
    if any(b[0] < a[1] for a, b in zip(ordered, ordered[1:])):
        raise ValueError("event search/response windows overlap")
    for s in shots:
        if any(s.noise_start_sample < hi and s.noise_stop_sample > lo for lo, hi in intervals):
            raise ValueError("quiet window overlaps possible event response")
    if timing_map is not None:
        if (timing_map.get("source_rate_hz") != source_rate_hz
                or timing_map.get("sample_rate_hz") != sample_rate_hz
                or not np.isfinite([timing_map.get("slope", np.nan), timing_map.get("intercept", np.nan)]).all()
                or timing_map["slope"] <= 0
                or not timing_map.get("anchor_event_ids")):
            raise ValueError("invalid packet timing map")
    detections, repeat_traces = [], []
    probe_energy = float(p @ p)
    floor_rms = 1 / (32768 * np.sqrt(12))
    for s in shots:
        segment = x[s.search_start_sample:s.search_stop_sample + n - 1]
        corr = signal.correlate(segment, p, mode="valid", method="fft")
        energies = np.maximum(signal.correlate(segment ** 2, np.ones(n), mode="valid", method="fft"), 0)
        normalized = np.divide(corr, np.sqrt(probe_energy * energies), out=np.zeros_like(corr), where=energies > 0)
        j = int(np.argmax(np.abs(normalized)))
        frac = 0.
        at_bound = j == 0 or j == len(corr) - 1
        if not at_bound:
            left, mid, right = np.abs(corr[j - 1:j + 2])
            den = left - 2 * mid + right
            if den < 0:
                frac = float(np.clip(.5 * (left - right) / den, -.5, .5))
        start = s.search_start_sample + j
        trace = x[start:start + n + tail]
        quiet = x[s.noise_start_sample:s.noise_stop_sample]
        noise_rms = float(np.sqrt(np.mean(quiet ** 2)))
        signal_rms = float(np.sqrt(np.mean(x[start:start + n] ** 2)))
        snr_db = float(20 * np.log10(max(signal_rms, floor_rms) / max(noise_rms, floor_rms)))
        frame = max(1, round(cfg.noise_frame_s * sample_rate_hz))
        chunks = [quiet[i:i + frame] for i in range(0, len(quiet), frame) if len(quiet[i:i + frame]) == frame]
        nr = np.array([np.sqrt(np.mean(q ** 2)) for q in chunks])
        transient_ratio = float(nr.max() / max(float(np.median(nr)), floor_rms)) if len(nr) else None
        count = int(clipped[start:start + n + tail].sum())
        reasons = []
        score = float(abs(normalized[j]))
        if score < cfg.min_normalized_correlation:
            reasons.append("missing_or_inconsistent_probe")
        if at_bound:
            reasons.append("detection_at_search_boundary")
        if snr_db < cfg.min_snr_db:
            reasons.append("low_snr")
        if count > cfg.max_clipped_samples:
            reasons.append("clipping")
        if transient_ratio is None:
            reasons.append("quiet_window_too_short_for_transient_qc")
        elif transient_ratio > cfg.max_noise_transient_ratio:
            reasons.append("noise_transient")
        row = {"event_id": s.event_id, "source_start_sample": int(s.source_start_sample),
               "detected_start_sample": int(start), "detected_start_sample_float": float(start + frac),
               "raw_segment_start_sample": int(start), "raw_segment_stop_sample": int(start + n + tail),
               "search_window_samples": [s.search_start_sample, s.search_stop_sample],
               "noise_window_samples": [s.noise_start_sample, s.noise_stop_sample],
               "normalized_correlation": score, "correlation_polarity": 1 if corr[j] >= 0 else -1,
               "matched_filter_amplitude": float(corr[j] / probe_energy),
               "signal_rms": signal_rms, "noise_rms": noise_rms, "snr_db": snr_db,
               "snr_definition": "probe-window total RMS / quiet RMS; quantization floor applied",
               "peak_absolute_amplitude": float(np.max(np.abs(trace))),
               "clipped_samples": count, "noise_transient_ratio": transient_ratio,
               "detected_at_search_boundary": at_bound}
        if timing_map is not None:
            predicted = float(timing_map["intercept"] + timing_map["slope"] * s.source_start_sample)
            residual = float(start + frac - predicted)
            independent = s.event_id not in timing_map["anchor_event_ids"]
            row.update(packet_predicted_start_sample=predicted, timing_residual_samples=residual,
                       timing_evaluation_independent_of_anchor_fit=independent)
            if not s.search_start_sample <= predicted < s.search_stop_sample:
                reasons.append("timing_prediction_outside_search_window")
            if independent and abs(residual) > cfg.max_timing_residual_samples:
                reasons.append("timing_inconsistent_with_independent_anchors")
            cut = int(np.floor(predicted))
            if 0 <= predicted and predicted + n + tail <= len(x):
                repeat_traces.append(mapped_window(x, predicted, n + tail))
                row["common_map_segment_samples"] = [cut, cut + n + tail]
                row["common_map_fractional_advance_samples"] = predicted - cut
                row["common_map_interpolation"] = "padded Fourier interpolation at supplied map origin; no measured-shot alignment"
            else:
                reasons.append("truncated_common_map_segment")
        row.update(accepted=not reasons, rejection_reasons=reasons)
        detections.append(row)
    global_reasons = []
    if int(clipped.sum()) > cfg.max_clipped_samples:
        global_reasons.append("capture_clipping")
    if any(not d["accepted"] for d in detections):
        global_reasons.append("rejected_events")
    repeat = None
    if len(repeat_traces) == len(shots) and len(shots) >= 2:
        repeat = repeat_coherence(repeat_traces, sample_rate_hz=sample_rate_hz, bands_hz=cfg.bands_hz)
        repeat["includes_rejected_events"] = any(not d["accepted"] for d in detections)
    return {"version": 1, "accepted": not global_reasons, "rejection_reasons": global_reasons,
            "sample_rate_hz": sample_rate_hz, "source_rate_hz": source_rate_hz,
            "capture_samples": len(x), "probe_samples_at_capture_rate": n,
            "response_tail_samples": tail, "clipped_samples": int(clipped.sum()),
            "config": asdict(cfg), "shots": detections, "repeat_coherence": repeat,
            "timing_map": timing_map, "absolute_timing_calibrated": False,
            "timing_status": "acoustic_anchor_relative" if timing_map else "unreferenced",
            "normalization": "none; normalized correlation is detection-only",
            "limitations": ["Clipping detection finds PCM rails, not hidden analog saturation or nonlinear distortion.",
                            "An acoustic matched-filter anchor is not calibrated emission time.",
                            "Pilot-to-shot residuals include waveform-dependent acoustic/instrument anchor bias unless that transfer is calibrated.",
                            "No per-event or cross-pose phase alignment is performed."]}
