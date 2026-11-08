"""Validate framed device observations and export exact, contiguous PCM.

Matching SDK reference/mic block timestamps establish an observed digital block
grid only. They do not establish analog DAC emission time or codec group delay.
No waveform peak or room echo is used to construct the grid.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct
import zlib

MAGIC = b"OPCAUD01"
PREFIX = struct.Struct("<8sIII")
MAX_META, MAX_PAYLOAD = 8192, 256 * 1024


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def records(data):
    """Decode an entire stream; require one hello and one terminal end record."""
    offset, rows, stream, dropped = 0, [], None, None
    while offset < len(data):
        if len(data) - offset < PREFIX.size:
            raise ValueError("truncated frame prefix")
        magic, nj, npcm, checksum = PREFIX.unpack_from(data, offset)
        offset += PREFIX.size
        if magic != MAGIC or not 0 < nj <= MAX_META or npcm > MAX_PAYLOAD:
            raise ValueError("invalid frame magic or size")
        end = offset + nj + npcm
        if end > len(data):
            raise ValueError("truncated frame body")
        meta_bytes, payload = data[offset:offset + nj], data[offset + nj:end]
        if zlib.crc32(meta_bytes + payload) != checksum:
            raise ValueError("frame CRC mismatch")
        try:
            row = json.loads(meta_bytes)
        except (ValueError, UnicodeError) as e:
            raise ValueError("invalid frame metadata") from e
        if not isinstance(row, dict) or row.get("schema") != "openpetcam.audio.v1" or row.get("record_index") != len(rows):
            raise ValueError("schema or record sequence discontinuity")
        if not rows:
            if row.get("kind") != "hello" or payload:
                raise ValueError("stream must start with empty hello record")
            stream, dropped = row.get("stream_id"), row.get("dropped_records")
            if not isinstance(stream, str) or not isinstance(dropped, int) or dropped < 0:
                raise ValueError("invalid stream identity/drop counter")
        if row.get("stream_id") != stream or row.get("dropped_records") != dropped:
            raise ValueError("stream identity changed or producer dropped a record")
        if rows and rows[-1][0]["kind"] == "end":
            raise ValueError("records after terminal end")
        if row.get("kind") not in ("hello", "mic", "ao", "end"):
            raise ValueError("producer reported an invalid frame or SDK fault")
        if row.get("kind") in ("mic", "ao"):
            before, after = row.get("monotonic_before_ns", 0), row.get("monotonic_after_ns", 0)
            if not isinstance(before, int) or not isinstance(after, int) or not 0 < before <= after or row.get("rc") != 0:
                raise ValueError("invalid SDK call timing or status")
        elif payload:
            raise ValueError("control record cannot contain PCM")
        rows.append((row, payload))
        offset = end
    if len(rows) < 3 or rows[-1][0]["kind"] != "end":
        raise ValueError("missing terminal end: recording incomplete, disconnected, or backpressured")
    if any(row["kind"] == "hello" for row, _ in rows[1:]):
        raise ValueError("multiple stream beginnings")
    return rows


def _format(meta, size, name):
    if meta.get("bytes") != size or meta.get("rate_hz") != 48000 or meta.get("bitwidth") != 16 or meta.get("channels") != 1 or size <= 0 or size % 2:
        raise ValueError(f"{name} must be declared 48000 Hz mono s16 PCM with exact byte count")
    if not isinstance(meta.get("seq"), int) or not isinstance(meta.get("timestamp_us"), int) or meta["timestamp_us"] <= 0:
        raise ValueError(f"{name} requires positive SDK timestamp and sequence")
    return size // 2


def _continuous(previous, current, count, name, *, sequence_available=True):
    if previous is None:
        return
    pmeta, pcount = previous
    if sequence_available and (current["seq"] - pmeta["seq"]) % (2 ** 32) != 1:
        raise ValueError(f"{name} SDK sequence discontinuity")
    delta = current["timestamp_us"] - pmeta["timestamp_us"]
    # These observed SDK timestamps have scheduling jitter. They are NOT used
    # as a phase clock. Sequence + exact payload count establish continuity;
    # this broad bound rejects gross timestamp discontinuities independently.
    if count != pcount or abs(delta - pcount * 1e6 / 48000) > .5 * pcount * 1e6 / 48000:
        raise ValueError(f"{name} timestamp/sample-count discontinuity")


def timestamp_diagnostics(intervals):
    """Describe SDK time versus sample count; never certify a hardware clock."""
    x = [r["sample_start"] for r in intervals]
    y = [r["mic"]["timestamp_us"] for r in intervals]
    mx, my = math.fsum(x) / len(x), math.fsum(y) / len(y)
    xx = math.fsum((v - mx) ** 2 for v in x)
    slope = math.fsum((a - mx) * (b - my) for a, b in zip(x, y)) / xx
    residual = [b - (my + slope * (a - mx)) for a, b in zip(x, y)]
    return {"timestamp_clock_verified": False, "affine_us_per_sample": slope,
            "effective_samples_per_sdk_second": 1e6 / slope if slope > 0 else None,
            "residual_min_us": min(residual), "residual_max_us": max(residual),
            "residual_rms_us": math.sqrt(math.fsum(v * v for v in residual) / len(residual)),
            "residual_us": residual, "meaning": "observed block timestamps; affine fit is diagnostic, not a sample clock"}


def verify_ao_reference(source_pcm, reference_pcm, *, minimum_correlation=.999,
                        maximum_relative_error=.01, quiet_pad_samples=240,
                        source_rate_hz=48000, reference_rate_hz=48000):
    """One global AO-to-reference offset/gain; no microphone or room peaks.

    Full AO duration must fit inside reference with quiet margins. Three
    energetic subwindows must agree with that one mapping. Local lag searches
    are diagnostics only and never warp the exported arrays.
    """
    import numpy as np
    if not source_pcm or not reference_pcm:
        return {"verified": False, "reason": "no active AO PCM or output-reference PCM"}
    s = np.frombuffer(source_pcm, "<i2").astype(float)
    r = np.frombuffer(reference_pcm, "<i2").astype(float)
    if source_rate_hz <= 0 or reference_rate_hz != 48000 or reference_rate_hz % source_rate_hz:
        return {"verified": False, "reason": "AO-to-reference rate ratio must be a declared integer"}
    factor = reference_rate_hz // source_rate_hz
    # Established by the active camera pilot: the SDK output reference holds
    # each AO input sample on the AI grid, rather than polyphase resampling.
    # This is a fixed digital reference model, not a fitted room filter.
    s = np.repeat(s, factor)
    if len(s) < 2048 or len(r) < len(s) + 2 * quiet_pad_samples:
        return {"verified": False, "reason": "insufficient complete AO duration and reference margins"}
    energy = float(s @ s)
    if energy < len(s) or float(np.var(s)) < .1 * float(np.mean(s * s)):
        return {"verified": False, "reason": "AO excitation lacks sufficient nonconstant energy"}
    nfft = 1 << (len(r) + len(s) - 1).bit_length()
    cross = np.fft.irfft(np.fft.rfft(r, nfft) * np.conj(np.fft.rfft(s, nfft)), nfft)
    valid = cross[:len(r) - len(s) + 1]
    cumulative = np.r_[0., np.cumsum(r * r)]
    window_energy = cumulative[len(s):] - cumulative[:-len(s)]
    # Normalized correlation is undefined for a silent reference window.
    # FFT roundoff is nonzero there; dividing it by a tiny denominator can
    # otherwise make trailing silence outrank an exact energetic match.
    score = np.full_like(valid, -1.)
    np.divide(valid * valid, window_energy * energy, out=score, where=window_energy > 0)
    score[:quiet_pad_samples] = -1; score[-quiet_pad_samples:] = -1
    if not np.any(score >= 0):
        return {"verified": False, "reason": "no energetic complete reference window inside quiet margins"}
    lag = int(np.argmax(score))
    target = r[lag:lag + len(s)]
    gain = float(s @ target / energy)
    relative = float(np.linalg.norm(target - gain * s) / max(np.linalg.norm(target), 1e-30))
    correlation = float(abs(s @ target) / max(math.sqrt(energy * float(target @ target)), 1e-30))
    # A fractional offset is allowed globally, but never a local time warp.
    fractional = 0.
    if relative > maximum_relative_error and 0 < lag < len(valid) - 1:
        den = valid[lag - 1] - 2 * valid[lag] + valid[lag + 1]
        delta = float(.5 * (valid[lag - 1] - valid[lag + 1]) / den) if den else 0.
        if abs(delta) <= .5:
            padded = np.pad(r, (len(s), len(s)))
            shifted = np.fft.irfft(np.fft.rfft(padded) * np.exp(2j * np.pi * np.fft.rfftfreq(len(padded)) * delta), len(padded))
            trial = shifted[len(s) + lag:len(s) + lag + len(s)]
            g = float(s @ trial / energy)
            err = float(np.linalg.norm(trial - g * s) / max(np.linalg.norm(trial), 1e-30))
            if err < relative:
                target, gain, relative, fractional = trial, g, err, delta
                correlation = float(abs(s @ target) / max(math.sqrt(energy * float(target @ target)), 1e-30))
    active = np.flatnonzero(abs(s) > max(float(np.max(abs(s))) * .01, 1.))
    checks = []
    if len(active):
        edges = np.linspace(active[0], active[-1] + 1, 4).astype(int)
        for lo, hi in zip(edges[:-1], edges[1:]):
            sub = s[lo:hi]
            en = float(sub @ sub)
            if en < .02 * energy or len(sub) < 256:
                continue
            err = float(np.linalg.norm(target[lo:hi] - gain * sub) / max(np.linalg.norm(target[lo:hi]), 1e-30))
            diagnostics = [(d, float(abs(sub @ r[lag + lo + d:lag + hi + d]))) for d in range(-4, 5)]
            best_delta = max(diagnostics, key=lambda q: q[1])[0]
            checks.append({"source_start": int(lo), "source_end": int(hi), "relative_error": err,
                           "local_best_integer_delta": best_delta})
    ok = (correlation >= minimum_correlation and relative <= maximum_relative_error
          and math.isfinite(gain) and abs(gain) > 1e-9 and len(checks) == 3
          and all(q["relative_error"] <= maximum_relative_error * 2
                  and abs(q["local_best_integer_delta"]) <= 1 for q in checks))
    return {"verified": bool(ok), "global_lag_samples": lag + fractional, "global_gain": gain,
            "correlation": correlation, "relative_error": relative, "source_samples": len(s),
            "reference_samples": len(r), "quiet_pad_samples": quiet_pad_samples,
            "minimum_correlation": minimum_correlation, "maximum_relative_error": maximum_relative_error,
            "subwindows": checks, "method": "one global source-AO/reference lag and gain; no mic alignment or local warps"}


def export_capture(framed_path, output_dir, *, prefix="device", allow_empty_reference_edges=True):
    """Export only after validating every frame; output folder must not overwrite files.

    Leading/trailing empty SDK references may be trimmed as an explicit common
    interval. An empty reference inside that interval is rejected. The complete
    mic payload is retained separately even when a paired interval is exported.
    """
    if not prefix or Path(prefix).name != prefix:
        raise ValueError("prefix must be a simple filename")
    data = Path(framed_path).read_bytes()
    rr = records(data)
    mic, refs, ao, intervals, ao_rates = [], [], [], [], set()
    prev_mic = prev_ref = None
    prev_ao = None
    samples = 0
    ref_sequences = [row.get("reference", {}).get("seq") for row, _ in rr
                     if row["kind"] == "mic" and row.get("reference", {}).get("bytes", 0)]
    ref_seq_available = bool(ref_sequences) and not all(v == 0 for v in ref_sequences)
    for row, payload in rr:
        if row["kind"] == "mic":
            m, r = row.get("mic"), row.get("reference")
            if not isinstance(m, dict) or not isinstance(r, dict):
                raise ValueError("mic/reference descriptors required")
            nm, nr = m.get("bytes"), r.get("bytes")
            if not isinstance(nm, int) or not isinstance(nr, int) or nm + nr != len(payload) or nr < 0:
                raise ValueError("mic/reference payload lengths do not match")
            count = _format(m, nm, "microphone")
            _continuous(prev_mic, m, count, "microphone")
            prev_mic = (m, count)
            if nr:
                if row.get("reference_state") != 1:
                    raise ValueError("reference payload without successful paired SDK state")
                # IMPAudioFrame has no sample-rate field. Older observer
                # headers mistakenly copied AO rate here; preserve that raw
                # field and test the AI-grid candidate with active PCM below.
                rcount = _format({**r, "rate_hz": 48000}, nr, "output reference")
                if rcount != count or abs(r["timestamp_us"] - m["timestamp_us"]) > 2:
                    raise ValueError("paired reference is not on the observed mic block grid")
                if prev_ref is not None:
                    _continuous(prev_ref, r, rcount, "output reference", sequence_available=ref_seq_available)
                prev_ref = (r, rcount)
            mic.append(payload[:nm]); refs.append(payload[nm:] if nr else None)
            intervals.append({"record_index": row["record_index"], "sample_start": samples,
                              "sample_end": samples + count, "mic": m, "reference": r,
                              "monotonic_before_ns": row["monotonic_before_ns"],
                              "monotonic_after_ns": row["monotonic_after_ns"]})
            samples += count
        elif row["kind"] == "ao":
            a = row["ao"]
            if a.get("bytes") != len(payload) or a.get("rate_hz") not in (8000, 16000, 24000, 48000) or a.get("bitwidth") != 16 or a.get("channels") != 1 or len(payload) % 2:
                raise ValueError("AO submission has invalid PCM format or byte count")
            index = a.get("submission_index")
            if not isinstance(index, int) or prev_ao is not None and index != prev_ao + 1:
                raise ValueError("AO submission sequence discontinuity")
            prev_ao = index
            ao_rates.add(a["rate_hz"])
            ao.append(payload)
    if len(mic) < 2:
        raise ValueError("at least two microphone frames required to verify continuity")
    valid = [i for i, ref in enumerate(refs) if ref is not None]
    lo, hi = (valid[0], valid[-1] + 1) if valid else (0, len(mic))
    if valid and any(ref is None for ref in refs[lo:hi]):
        raise ValueError("empty SDK reference inside the common interval")
    if valid and not allow_empty_reference_edges and len(valid) != len(mic):
        raise ValueError("empty SDK reference at an edge requires explicit trimming policy")
    all_mic = b"".join(mic)
    selected_mic = b"".join(mic[lo:hi])
    reference = b"".join(refs[lo:hi]) if valid else None
    if len(ao_rates) > 1:
        raise ValueError("AO sample rate changed inside one capture")
    ao_rate = next(iter(ao_rates)) if ao_rates else 0
    active_check = verify_ao_reference(b"".join(ao), reference, source_rate_hz=ao_rate)
    active_check.update(source_rate_hz=ao_rate, reference_grid_rate_hz=48000,
                        resampling="fixed zero-order hold integer ratio; established by active SDK pilot")
    contract_ok = len(valid) >= 2
    grid_ok = contract_ok and active_check["verified"]
    out = Path(output_dir)
    blobs = {f"{prefix}.mic.pcm": selected_mic, f"{prefix}.mic.all.pcm": all_mic,
             f"{prefix}.source.pcm": b"".join(ao)}
    if reference is not None:
        blobs[f"{prefix}.reference.pcm"] = reference
    paths = [out / name for name in blobs] + [out / f"{prefix}.json"]
    if any(path.exists() for path in paths):
        raise ValueError("refusing to overwrite an exported capture")
    metadata = {"schema": "openpetcam.device.capture.v1", "complete": True,
                "stream_id": rr[0][0]["stream_id"], "hello": rr[0][0], "end": rr[-1][0],
                "framed_file": str(Path(framed_path).resolve()), "framed_sha256": _sha(data),
                "mic_file": str((out / f"{prefix}.mic.pcm").resolve()), "mic_sha256": _sha(selected_mic),
                "all_mic_file": str((out / f"{prefix}.mic.all.pcm").resolve()), "all_mic_sha256": _sha(all_mic),
                "reference_file": str((out / f"{prefix}.reference.pcm").resolve()) if reference else None,
                "reference_sha256": _sha(reference) if reference else None,
                "source_file": str((out / f"{prefix}.source.pcm").resolve()), "source_sha256": _sha(b"".join(ao)),
                "source_rate_hz": ao_rate,
                "sample_rate_hz": 48000, "format": "s16le mono", "mic_continuity_verified": True,
                "reference_pair_contract_verified": contract_ok,
                "reference_sample_grid_verified": grid_ok, "reference_continuity_verified": grid_ok,
                "reference_sequence_provenance": "SDK reference sequence" if ref_seq_available else "unavailable_all_zero; paired call + mic sequence + matching block timestamp and count",
                "timestamp_clock_verified": False, "sdk_timestamp_diagnostics": timestamp_diagnostics(intervals),
                "reference_rate_provenance": "48k paired-AI grid candidate verified only by active AO waveform mapping; original per-frame header fields preserved (legacy observer copied AO rate)",
                "active_ao_reference_check": active_check,
                "physical_DAC_timing_verified": False,
                "reference_kind": "paired_sdk_output_reference_with_matching_block_metadata; analog_DAC_mapping_unverified",
                "common_mic_sample_interval": [intervals[lo]["sample_start"], intervals[hi - 1]["sample_end"]],
                "empty_reference_leading_frames": lo if valid else len(mic),
                "empty_reference_trailing_frames": len(mic) - hi if valid else 0,
                "allow_empty_reference_edges": allow_empty_reference_edges,
                "frames": intervals, "ao_submissions": [row for row, _ in rr if row["kind"] == "ao"],
                "limitations": ["SDK timestamps establish an observed digital block grid, not analog emission time.",
                                "SDK timestamp start/end boundary semantics and codec group delays remain unverified.",
                                "AO call brackets and queue block counts are diagnostics, not a DAC sample clock."]}
    out.mkdir(parents=True, exist_ok=True)
    for name, blob in blobs.items():
        (out / name).write_bytes(blob)
    (out / f"{prefix}.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return metadata


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("framed", type=Path); p.add_argument("output", type=Path)
    p.add_argument("--prefix", default="device")
    p.add_argument("--require-reference-every-frame", action="store_true")
    a = p.parse_args()
    result = export_capture(a.framed, a.output, prefix=a.prefix,
                            allow_empty_reference_edges=not a.require_reference_every_frame)
    print(json.dumps({k: v for k, v in result.items() if k not in ("frames", "ao_submissions")}, indent=2))


if __name__ == "__main__":
    main()
