"""Signed raw-microphone responses, referenced only to a designated training capture.

No camera access. The reference is a prominent acoustic path, not a known direct
arrival. Keep original matched-filter locations: extracting each sweep already
removes its integer emission offset before any additional alignment is applied.
"""
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import signal


@dataclass
class WaveformDataset:
    h_raw: np.ndarray
    h_aligned: np.ndarray
    sweep_h_raw: np.ndarray
    sweep_h_aligned: np.ndarray
    time_s: np.ndarray
    inverse_kernel: np.ndarray
    filter_frequencies_hz: np.ndarray
    inverse_filter_spectrum: np.ndarray
    inverse_transfer_spectrum: np.ndarray
    metadata: dict

    @property
    def fs(self):
        return self.metadata["config"]["fs"]

    @property
    def recordings(self):
        return self.metadata["recordings"]

    @property
    def sweep_metadata(self):
        return self.metadata["sweeps"]

    def complex_spectra(self, aligned=True, window=None):
        """FFT of signed means; optional explicit identical window supplied by caller.

        Frequencies are rfftfreq(len(time_s), 1/fs). No magnitude operation or
        per-trace normalization is performed. Caller owns train-only fitting.
        """
        h = self.h_aligned if aligned else self.h_raw
        if window is not None:
            w = np.asarray(window)
            if w.shape != self.time_s.shape:
                raise ValueError("window must match the full response time grid")
            h = h * w
        return np.fft.rfft(h, axis=-1)


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _correlation(a, b):
    a, b = a - np.mean(a), b - np.mean(b)
    den = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / den) if den else -1.0


def advance(x, lag):
    """Advance a signed response: output[n] = input[n + lag], padded FFT interpolation."""
    n = len(x)
    nfft = 1 << (3 * n - 1).bit_length()
    z = np.zeros(nfft)
    z[n:2 * n] = x
    f = np.fft.rfftfreq(nfft)
    y = np.fft.irfft(np.fft.rfft(z) * np.exp(2j * np.pi * f * lag), nfft)
    return y[n:2 * n]


def estimate_lag(reference, x, lo, hi, bound):
    """One signed early-window lag; positive means x is delayed and should be advanced."""
    if lo - bound < 0 or hi + bound > len(x) or hi <= lo:
        raise ValueError("alignment gate and lag bound must fit inside the response")
    lags = np.arange(-bound, bound + 1)
    cc = np.array([_correlation(reference[lo:hi], x[lo + k:hi + k]) for k in lags])
    j = int(np.argmax(cc))
    frac = 0.0
    at_bound = j in (0, len(lags) - 1)
    if not at_bound:
        left, middle, right = cc[j - 1:j + 2]
        den = left - 2 * middle + right
        frac = float(.5 * (left - right) / den) if den else 0.0
    return float(lags[j] + frac), float(cc[j]), at_bound


def _make_filter(probe, fs, pre, tail, band, regularization):
    segment_length = pre + len(probe) + tail
    nfft = 1 << (segment_length + len(probe) - 1).bit_length()
    S = np.fft.rfft(probe, nfft)
    f = np.fft.rfftfreq(nfft, 1 / fs)
    mask = (f > band[0]) & (f < band[1])
    w = np.zeros_like(f)
    w[mask] = signal.windows.tukey(int(mask.sum()), .5)
    den = abs(S) ** 2 + regularization * abs(S).max() ** 2
    inverse = w * np.conj(S) / den
    transfer = inverse * S
    return f, inverse, transfer, nfft


def _extract(x, probe, pre, tail, inverse, nfft, fs, expected, spacing, threshold):
    match = np.abs(signal.hilbert(signal.correlate(x, probe, "valid", method="fft")))
    if not np.any(match > 0):
        return [], [], [], "zero matched-filter response"
    peaks, _ = signal.find_peaks(match, distance=round(spacing * fs), height=match.max() * threshold)
    offsets, hs = [], []
    for p in peaks:
        lo, hi = int(p) - pre, int(p) + len(probe) + tail
        if lo < 0 or hi > len(x):
            return [], peaks.tolist(), match[peaks].tolist(), "incomplete sweep extraction at recording edge"
        h = np.fft.irfft(np.fft.rfft(x[lo:hi], nfft) * inverse, nfft)[:pre + tail]
        offsets.append(int(p))
        hs.append(h)
    if len(peaks) != expected:
        return [], peaks.tolist(), match[peaks].tolist(), f"expected {expected} sweeps, detected {len(peaks)}"
    return hs, offsets, match[peaks].tolist(), None


def prepare_recordings(scan_dirs, probe_path, reference, *, designated_training,
                       primary_volume=40, include_volumes=None, expected_sweeps=4,
                       fs=48000, probe_fs=16000, band_hz=(2100, 7400),
                       regularization=1e-3, pre_s=.005, tail_s=.15,
                       anchor_gate_s=(.003, .008), alignment_window_s=(-.0005, .002),
                       max_lag_s=.0005, sweep_spacing_s=1.2, match_threshold=.3,
                       allow_incomplete=False):
    """Read logged PCM sessions; extract signed sweeps, then align to one train-only template.

    reference and designated_training contain (session_directory_name, integer_k)
    identifiers. A selected failure makes the run fail closed unless explicitly
    allowed; accepted arrays never contain replacement zeros. Silence and other
    volumes are counted in inventory. No room-state labels are inferred.
    """
    reference = (str(reference[0]), int(reference[1]))
    training = {(str(s), int(k)) for s, k in designated_training}
    if reference not in training:
        raise ValueError("reference must belong to designated_training")
    if fs != 48000 or probe_fs != 16000:
        raise ValueError("this raw format requires declared 48000 Hz PCM and 16000 Hz probe")
    if not 0 < band_hz[0] < band_hz[1] < fs / 2 or regularization <= 0:
        raise ValueError("invalid inverse-filter band or regularization")
    if expected_sweeps < 1 or pre_s <= 0 or tail_s <= 0 or max_lag_s <= 0:
        raise ValueError("invalid extraction size or sweep count")
    if round(max_lag_s * fs) < 1 or sweep_spacing_s <= 0 or not 0 < match_threshold < 1:
        raise ValueError("invalid lag bound or matched-filter peak configuration")
    selected_volumes = set(include_volumes if include_volumes is not None else [primary_volume])
    probe_path = Path(probe_path).resolve()
    source_probe = np.load(probe_path, allow_pickle=False)
    if source_probe.ndim != 1 or len(source_probe) < 2 or not np.isfinite(source_probe).all() or not np.any(source_probe):
        raise ValueError("probe must be a finite nonzero one-dimensional waveform")
    probe = signal.resample_poly(source_probe, 3, 1)
    pre, tail = round(pre_s * fs), round(tail_s * fs)
    f, inverse, transfer, nfft = _make_filter(probe, fs, pre, tail, band_hz, regularization)
    inventory, captures, input_paths = [], [], {probe_path}
    seen_ids, sessions = set(), set()
    for scan in sorted(Path(s).resolve() for s in scan_dirs):
        if scan.name in sessions:
            raise ValueError("session directory names must be unique")
        sessions.add(scan.name)
        lp = scan / "log.jsonl"
        input_paths.add(lp)
        if (scan / "plan.json").exists():
            input_paths.add(scan / "plan.json")
        for lineno, line in enumerate(lp.read_text().splitlines(), 1):
            if not line.strip():
                continue
            q = {"session": scan.name, "line": lineno, "status": "failed"}
            try:
                row = json.loads(line)
                key = (scan.name, int(row["k"]))
                q.update(k=key[1], kind=str(row["kind"]), volume=int(row["vol"]),
                         requested_pose=[int(row["pan"]), int(row["tilt"])],
                         pose=[int(v) for v in row["pos"]],
                         t_rec=float(row["t_rec"]), t_play=float(row["t_play"]))
                if key in seen_ids:
                    raise ValueError("duplicate session/k identifier")
                seen_ids.add(key)
                filename = row.get("file") or f"{key[1]:02d}_{row['kind']}_{row['pan']}_{row['tilt']}_v{row['vol']}.pcm"
                path = scan / filename
                q["file"] = str(path)
                if path.exists():
                    input_paths.add(path)
                if q["volume"] == 0 or q["volume"] not in selected_volumes:
                    q.update(status="excluded", reason="silence" if q["volume"] == 0 else "volume_not_selected")
                    inventory.append(q)
                    continue
                raw = path.read_bytes()
                if len(raw) % 2:
                    raise ValueError("odd byte count for s16le PCM")
                x = np.frombuffer(raw, "<i2").astype(float) / 32768
                if len(x) < len(probe) + pre + tail:
                    raise ValueError("recording too short for one sweep")
                q.update(bytes=len(raw), seconds=len(x) / fs)
                hs, offsets, amplitudes, error = _extract(x, probe, pre, tail, inverse, nfft, fs,
                                                         expected_sweeps, sweep_spacing_s, match_threshold)
                q.update(detected_sweeps=len(offsets), matched_filter_peak_samples=offsets,
                         matched_filter_peak_values=amplitudes)
                if error:
                    raise ValueError(error)
                q["status"] = "accepted"
                q["training"] = key in training
                captures.append((key, q, np.array(hs)))
            except (OSError, ValueError, TypeError, KeyError) as e:
                q["reason"] = str(e)
            inventory.append(q)
    counts = {s: sum(q["status"] == s for q in inventory) for s in ("accepted", "excluded", "failed")}
    refs = [c for c in captures if c[0] == reference]
    if len(refs) != 1:
        raise ValueError("designated training reference was not accepted; " + json.dumps(inventory))
    if counts["failed"] and not allow_incomplete:
        raise ValueError("incomplete extraction: " + json.dumps([q for q in inventory if q["status"] == "failed"]))
    template = refs[0][2][0].copy()
    anchor_lo, anchor_hi = (round(s * fs) for s in anchor_gate_s)
    if not 0 <= anchor_lo < anchor_hi <= len(template):
        raise ValueError("anchor gate must fit inside the response")
    anchor = anchor_lo + int(np.argmax(np.abs(signal.hilbert(template))[anchor_lo:anchor_hi]))
    lo, hi = (anchor + round(s * fs) for s in alignment_window_s)
    bound = round(max_lag_s * fs)
    raw_means, aligned_means, sweep_raw, sweep_aligned, sweeps, records = [], [], [], [], [], []
    for key, q, hs in captures:
        start = len(sweeps)
        aa = []
        for j, h in enumerate(hs):
            lag, score, at_bound = (0.0, 1.0, False) if key == reference and j == 0 else estimate_lag(template, h, lo, hi, bound)
            a = advance(h, lag)
            sweep_raw.append(h)
            sweep_aligned.append(a)
            aa.append(a)
            sweeps.append({"session": key[0], "k": key[1], "sweep": j,
                           "recording_index": len(records), "training": key in training,
                           "matched_filter_peak_sample": q["matched_filter_peak_samples"][j],
                           "matched_filter_peak_seconds": q["matched_filter_peak_samples"][j] / fs,
                           "additional_advance_samples": lag, "additional_advance_seconds": lag / fs,
                           "alignment_correlation": score, "alignment_at_bound": at_bound})
        raw_means.append(hs.mean(axis=0))
        aligned_means.append(np.mean(aa, axis=0))
        records.append({**q, "sweep_start": start, "sweep_count": len(hs)})
    bound_hits = sum(q["alignment_at_bound"] for q in sweeps)
    if bound_hits and not allow_incomplete:
        raise ValueError(f"incomplete alignment: {bound_hits} sweeps reached the lag search bound")
    time_s = (np.arange(pre + tail) - anchor) / fs
    # The actual deconvolution transfer of a unit impulse, placed at t=0 on this grid.
    kernel = np.fft.irfft(transfer * np.exp(-2j * np.pi * f * anchor / fs), nfft)[:pre + tail]
    config = {"fs": fs, "pcm_format": "s16le mono", "probe_fs": probe_fs, "reference": list(reference),
              "reference_sweep": 0, "designated_training": [list(k) for k in sorted(training)],
              "selected_volumes": sorted(selected_volumes), "expected_sweeps": expected_sweeps,
              "band_hz": list(band_hz), "regularization": regularization, "pre_s": pre_s, "tail_s": tail_s,
              "anchor_gate_s": list(anchor_gate_s), "anchor_sample": anchor,
              "alignment_window_s": list(alignment_window_s), "max_lag_s": max_lag_s,
              "sweep_spacing_s": sweep_spacing_s, "match_threshold": match_threshold,
              "filter_nfft": nfft, "allow_incomplete": allow_incomplete}
    config_bytes = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    provenance = {"inputs": {str(p): {"sha256": _sha(p), "bytes": p.stat().st_size} for p in sorted(input_paths)},
                  "source_sha256": _sha(Path(__file__)), "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
                  "probe_sha256": _sha(probe_path)}
    metadata = {"version": 1, "created_utc": datetime.now(timezone.utc).isoformat(), "config": config,
                "complete": counts["failed"] == 0 and bound_hits == 0, "counts": counts,
                "alignment_bound_hits": bound_hits, "inventory": inventory, "recordings": records,
                "sweeps": sweeps, "provenance": provenance,
                "timing_assumption": "shared prominent-path reference across poses; not a validated physical direct arrival",
                "normalization": "none; signed absolute PCM scaling retained",
                "raw_coordinates": "each sweep segmented at its own integer matched-filter peak; original offsets retained",
                "alignment": "one bounded early-window common lag per sweep, applied to entire response; no tail alignment or stretch"}
    return WaveformDataset(np.array(raw_means), np.array(aligned_means), np.array(sweep_raw),
                           np.array(sweep_aligned), time_s, kernel, f, inverse, transfer, metadata)


_ARRAYS = ("h_raw", "h_aligned", "sweep_h_raw", "sweep_h_aligned", "time_s", "inverse_kernel",
           "filter_frequencies_hz", "inverse_filter_spectrum", "inverse_transfer_spectrum")


def save_dataset(dataset, prefix):
    """Save numeric arrays without pickle and a JSON metadata/provenance sidecar."""
    prefix = Path(prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    npz_path, json_path = Path(str(prefix) + ".npz"), Path(str(prefix) + ".json")
    np.savez_compressed(npz_path, **{k: getattr(dataset, k) for k in _ARRAYS})
    payload = {**dataset.metadata, "array_sha256": _sha(npz_path)}
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    return npz_path, json_path


def load_dataset(prefix):
    prefix = Path(prefix)
    npz_path, json_path = Path(str(prefix) + ".npz"), Path(str(prefix) + ".json")
    metadata = json.loads(json_path.read_text())
    if metadata.get("array_sha256") != _sha(npz_path):
        raise ValueError("dataset array hash mismatch")
    with np.load(npz_path, allow_pickle=False) as arrays:
        return WaveformDataset(**{k: arrays[k] for k in _ARRAYS}, metadata=metadata)
