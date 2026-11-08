"""Deterministic, archived excitation packets for offline acoustic comparisons.

Generation never plays audio. Templates are decoded from the exact quantized
WAV bytes intended for playback, not reconstructed from floating-point recipes.
Acoustic pilots help diagnose latency/rate stability; they are not DAC time zero.
"""
from dataclasses import asdict, dataclass
import argparse
import hashlib
import json
from pathlib import Path
import re
import wave

import numpy as np


@dataclass(frozen=True)
class ProbeConfig:
    sample_rate_hz: int = 48000
    band_hz: tuple = (2000., 7600.)
    durations_s: tuple = (.5, 2.)
    repetitions: int = 4
    peak_ceiling: float = .5
    target_rms: float = .34
    fade_s: float = .01
    pre_s: float = .5
    gap_s: float = 1.
    post_s: float = 1.
    pilot_duration_s: float = .08
    pilot_gap_s: float = 1.


def _json_bytes(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _sha(data):
    return hashlib.sha256(data).hexdigest()


def _sample_count(seconds, fs, label):
    value = float(seconds) * fs
    if not np.isfinite(value) or value < 0 or abs(value - round(value)) > 1e-7:
        raise ValueError(f"{label} must be nonnegative and an integer number of samples")
    return round(value)


def _validate(config):
    if type(config.sample_rate_hz) is not int or config.sample_rate_hz not in (16000, 48000):
        raise ValueError("playback archives require 48000 Hz, or explicit diagnostic 16000 Hz")
    if not isinstance(config.repetitions, int) or config.repetitions < 2:
        raise ValueError("at least two integer repetitions are required")
    if len(config.band_hz) != 2 or not 0 < config.band_hz[0] < config.band_hz[1] < config.sample_rate_hz / 2:
        raise ValueError("band must lie strictly inside Nyquist")
    if not 0 < config.target_rms < config.peak_ceiling <= .5:
        raise ValueError("require 0 < target_rms < peak_ceiling <= 0.5")
    if not config.durations_s or len(set(config.durations_s)) != len(config.durations_s):
        raise ValueError("durations must be nonempty and unique")
    for label in ("fade_s", "pre_s", "gap_s", "post_s", "pilot_duration_s", "pilot_gap_s"):
        if _sample_count(getattr(config, label), config.sample_rate_hz, label) < 1:
            raise ValueError(f"{label} must be positive")
    if config.post_s < 1 or config.gap_s < 1 or config.pilot_gap_s < 1:
        raise ValueError("post, inter-shot and pilot gaps must be at least one second")
    for duration in config.durations_s:
        if _sample_count(duration, config.sample_rate_hz, "duration") < 16:
            raise ValueError("probe duration must contain at least 16 samples")


def _sweep_pcm(config, duration, family, rms):
    fs = config.sample_rate_hz
    n = _sample_count(duration, fs, "duration")
    t = np.arange(n, dtype=np.float64) / fs
    low, high = config.band_hz
    if family == "linear":
        phase = 2 * np.pi * (low * t + .5 * (high - low) / duration * t * t)
    elif family == "exponential":
        rate = np.log(high / low) / duration
        phase = 2 * np.pi * low * np.expm1(rate * t) / rate
    elif family == "linear_down":
        phase = 2 * np.pi * (high * t + .5 * (low - high) / duration * t * t)
    else:
        raise ValueError("unknown sweep family")
    x = np.sin(phase)
    fade = min(round(config.fade_s * fs), max(2, n // 20))
    taper = .5 - .5 * np.cos(np.linspace(0, np.pi, fade))
    x[:fade] *= taper
    x[-fade:] *= taper[::-1]
    x *= rms / np.sqrt(np.mean(x * x))
    if np.max(np.abs(x)) > config.peak_ceiling:
        raise ValueError("requested RMS is infeasible under peak ceiling; lower target_rms")
    pcm = np.rint(x * 32768).astype("<i2")
    # Quantization can increase a sample by half an LSB. Never silently clip it.
    if np.max(np.abs(pcm.astype(np.int32))) / 32768 > config.peak_ceiling:
        raise ValueError("quantized signal exceeds peak ceiling")
    return pcm


def _write_wav(path, pcm, fs):
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(fs)
        out.writeframes(pcm.astype("<i2", copy=False).tobytes())
    x = pcm.astype(np.float64) / 32768
    return {"file": path.name, "sha256": _sha(path.read_bytes()),
            "pcm_sha256": _sha(pcm.astype("<i2", copy=False).tobytes()),
            "samples": len(pcm), "duration_s": len(pcm) / fs,
            "actual_rms": float(np.sqrt(np.mean(x * x))),
            "actual_peak": float(np.max(np.abs(x))),
            "energy_sample_sum": float(np.dot(x, x))}


def build_probe_suite(output, config=ProbeConfig()):
    """Archive equal-duration/target-energy linear and exponential packets.

    Returns the suite manifest. The output directory must not exist. RMS and
    energy are equal before quantization and differ only by recorded quantization
    error afterward. Longer-duration pairs deliberately have greater total energy;
    compare waveform families within each duration before comparing durations.
    """
    if isinstance(config, dict):
        config = ProbeConfig(**config)
    _validate(config)
    # Validate/generate all audio before creating any output directory.
    pilot = _sweep_pcm(config, config.pilot_duration_s, "linear_down", min(.30, config.target_rms))
    probes = [(duration, family, _sweep_pcm(config, duration, family, config.target_rms))
              for duration in config.durations_s for family in ("linear", "exponential")]
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    fs = config.sample_rate_hz
    config_dict = asdict(config)
    suite_id = "probe-suite-" + _sha(_json_bytes(config_dict))[:16]
    packets = []
    for duration, family, probe in probes:
        identity = {"config": config_dict, "family": family, "duration_s": duration}
        packet_id = f"{family}-{len(probe)}-" + _sha(_json_bytes(identity))[:12]
        directory = root / packet_id
        directory.mkdir()
        blocks, events, position = [], [], 0

        def silence(seconds):
            nonlocal position
            count = _sample_count(seconds, fs, "silence")
            blocks.append(np.zeros(count, dtype="<i2"))
            position += count

        def event(pcm, kind, event_id, repeat_id=None):
            nonlocal position
            events.append({"type": kind, "event_id": event_id, "repeat_id": repeat_id,
                           "sample_start": position, "sample_end": position + len(pcm)})
            blocks.append(pcm)
            position += len(pcm)

        silence(config.pre_s)
        event(pilot, "pilot", "pilot_start")
        silence(config.pilot_gap_s)
        for i in range(config.repetitions):
            if i:
                silence(config.gap_s)
            event(probe, "shot", f"shot_{i:03d}", i)
        silence(config.pilot_gap_s)
        event(pilot, "pilot", "pilot_end")
        silence(config.post_s)
        packet = np.concatenate(blocks)
        manifest = {
            "schema": "openpetcam.acoustic.probe_packet.v1", "packet_id": packet_id,
            "suite_id": suite_id, "sample_rate_hz": fs, "pcm_format": "s16le mono",
            "config": config_dict, "events": events,
            "packet": _write_wav(directory / "packet.wav", packet, fs),
            "probe": dict(_write_wav(directory / "probe.wav", probe, fs), family=family,
                          band_hz=list(config.band_hz), target_rms=config.target_rms),
            "pilot": dict(_write_wav(directory / "pilot.wav", pilot, fs), family="linear_down",
                          band_hz=list(config.band_hz)),
            "comparison_group": f"{suite_id}-samples-{len(probe)}",
            "timing_status": "source_sample_schedule_only; acoustic_pilots_are_not_physical_emission_t0",
            "template_status": "decode_archived_quantized_wav; not_prequantized_recipe",
            "processing_requirements": [
                "Retain unmodified continuous ADC capture, including preceding and following silence.",
                "Do not use acoustic peak alignment to claim physical emission time or cross-pose phase calibration.",
                "Retain negative-lag deconvolution support for exponential-sweep harmonic diagnostics.",
                "Do not normalize captured traces; record playback level and microphone processing settings.",
                "Pilots estimate relative packet timing/rate only when the acoustic reference path stays stable."
            ],
        }
        raw = _json_bytes(manifest)
        (directory / "packet.json").write_bytes(raw)
        packets.append({"packet_id": packet_id, "manifest": f"{packet_id}/packet.json",
                        "manifest_sha256": _sha(raw), "family": family,
                        "duration_s": duration, "comparison_group": manifest["comparison_group"],
                        "packet_duration_s": len(packet) / fs})
    suite = {"schema": "openpetcam.acoustic.probe_suite.v1", "suite_id": suite_id,
             "config": config_dict, "packets": packets,
             "fairness": "Same duration, band, target RMS, repetitions, pilots and peak ceiling within each duration pair; measured quantized energy is archived.",
             "hardware_status": "generated_only; no_playback_or_device_configuration_verified"}
    (root / "suite.json").write_bytes(_json_bytes(suite))
    return suite


def load_probe_packet(manifest_path):
    """Load verified exact source PCM as floats: packet, probe, pilot, manifest."""
    path = Path(manifest_path)
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "openpetcam.acoustic.probe_packet.v1":
        raise ValueError("unsupported probe packet schema")
    fs = manifest.get("sample_rate_hz")
    if type(fs) is not int or fs not in (16000, 48000) or manifest.get("config", {}).get("sample_rate_hz") != fs:
        raise ValueError("unsupported or inconsistent packet sample rate")
    arrays = []
    for key in ("packet", "probe", "pilot"):
        info = manifest[key]
        if type(info.get("samples")) is not int or info["samples"] <= 0:
            raise ValueError("invalid declared asset sample count")
        filename = info["file"]
        if Path(filename).name != filename:
            raise ValueError("packet asset must be a local basename")
        asset = path.parent / filename
        if _sha(asset.read_bytes()) != info["sha256"]:
            raise ValueError(f"{key} WAV hash mismatch")
        with wave.open(str(asset), "rb") as src:
            if (src.getnchannels(), src.getsampwidth(), src.getframerate(), src.getcomptype()) != (1, 2, fs, "NONE"):
                raise ValueError("packet WAV format mismatch")
            if src.getnframes() != info["samples"]:
                raise ValueError("WAV header sample count differs from manifest")
            raw = src.readframes(src.getnframes())
        if _sha(raw) != info["pcm_sha256"] or len(raw) != 2 * info["samples"]:
            raise ValueError(f"{key} PCM hash or sample count mismatch")
        arrays.append(np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768)
    packet, probe, pilot = arrays
    previous_end = 0
    repeats = []
    events = manifest["events"]
    count = manifest["config"]["repetitions"]
    if [e["type"] for e in events] != ["pilot"] + ["shot"] * count + ["pilot"]:
        raise ValueError("packet must have exactly two bracketing pilots")
    if events[0]["event_id"] != "pilot_start" or events[-1]["event_id"] != "pilot_end":
        raise ValueError("packet pilot identity mismatch")
    for event in events:
        start, end = event["sample_start"], event["sample_end"]
        if not isinstance(start, int) or not isinstance(end, int) or not previous_end <= start < end <= len(packet):
            raise ValueError("invalid packet event interval")
        if event["type"] not in ("shot", "pilot"):
            raise ValueError("invalid packet event type")
        template = probe if event["type"] == "shot" else pilot
        if not np.array_equal(packet[start:end], template):
            raise ValueError("event does not match archived template")
        if event["type"] == "shot":
            repeats.append(event["repeat_id"])
        previous_end = end
    if repeats != list(range(count)):
        raise ValueError("packet repetition inventory mismatch")
    return packet, probe, pilot, manifest


def derive_probe_suite_rate(suite_path, output, *, target_rate_hz=48000):
    """Archive exact 16-to-48 kHz sample repetition without changing originals.

    This opt-in digital representation repeats each signed sample three times;
    it is not a newly synthesized sweep or a calibrated analog equivalence.
    All inputs validate before the new output directory is created.
    """
    if type(target_rate_hz) is not int or target_rate_hz != 48000:
        raise ValueError("exact archived derivation supports only 16000 to 48000 Hz")
    source = Path(suite_path)
    raw_suite = source.read_bytes()
    suite = json.loads(raw_suite)
    if (suite.get("schema") != "openpetcam.acoustic.probe_suite.v1"
            or suite.get("config", {}).get("sample_rate_hz") != 16000
            or not suite.get("packets")):
        raise ValueError("requires a nonempty original 16000 Hz probe suite")
    validated, ids = [], set()
    for entry in suite["packets"]:
        relative = Path(entry["manifest"])
        path = (source.parent / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(source.parent.resolve()):
            raise ValueError("manifest escapes source suite")
        raw_manifest = path.read_bytes()
        if _sha(raw_manifest) != entry["manifest_sha256"]:
            raise ValueError("source manifest hash mismatch")
        packet, probe, pilot, manifest = load_probe_packet(path)
        if (manifest["sample_rate_hz"] != 16000 or manifest["packet_id"] != entry["packet_id"]
                or manifest["suite_id"] != suite["suite_id"] or manifest["packet_id"] in ids):
            raise ValueError("source suite identity or rate mismatch")
        if not isinstance(manifest["packet_id"], str) or not re.fullmatch(r"[A-Za-z0-9_-]+", manifest["packet_id"]):
            raise ValueError("source packet ID must be a safe identifier")
        ids.add(manifest["packet_id"])
        validated.append((entry, manifest, (packet, probe, pilot), raw_manifest))
    recipe = {"kind": "exact_signed_s16_sample_repeat", "factor": 3,
              "source_rate_hz": 16000, "target_rate_hz": 48000,
              "source_suite_sha256": _sha(raw_suite), "source_suite_id": suite["suite_id"],
              "amplitude_normalization": False, "filtering": False,
              "energy_note": "Sample energy sum triples; sum(x^2)/sample_rate and physical event durations stay equal.",
              "scope": "Digital representation only; no physical DAC or cross-capture phase calibration."}
    suite_id = "probe-suite-zoh3-" + _sha(_json_bytes(recipe))[:16]
    root = Path(output)
    root.mkdir(parents=True, exist_ok=False)
    (root / "source-suite.json").write_bytes(raw_suite)
    packets = []
    for entry, original, arrays, raw_manifest in validated:
        manifest_sha = _sha(raw_manifest)
        manifest = json.loads(json.dumps(original))
        identity = {"source_manifest_sha256": manifest_sha, "recipe": recipe}
        packet_id = original["packet_id"] + "-zoh3-" + _sha(_json_bytes(identity))[:12]
        directory = root / packet_id
        directory.mkdir()
        (directory / "source-packet.json").write_bytes(raw_manifest)
        manifest.update(packet_id=packet_id, suite_id=suite_id, sample_rate_hz=48000,
                        source_derivation=dict(recipe, source_manifest_sha256=manifest_sha,
                                               source_packet_id=original["packet_id"]))
        manifest["config"]["sample_rate_hz"] = 48000
        manifest["comparison_group"] = suite_id + "-" + _sha(str(original["comparison_group"]).encode())[:12]
        for event in manifest["events"]:
            event["sample_start"] *= 3
            event["sample_end"] *= 3
        for name, array in zip(("packet", "probe", "pilot"), arrays):
            # Binary /32768 values are exact; this recovers original signed PCM.
            pcm = np.repeat(np.rint(array * 32768).astype("<i2"), 3)
            manifest[name].update(_write_wav(directory / original[name]["file"], pcm, 48000))
        raw = _json_bytes(manifest)
        (directory / "packet.json").write_bytes(raw)
        packets.append(dict(entry, packet_id=packet_id, manifest=f"{packet_id}/packet.json",
                            manifest_sha256=_sha(raw), comparison_group=manifest["comparison_group"]))
    result = json.loads(json.dumps(suite))
    result.update(suite_id=suite_id, packets=packets, source_derivation=recipe,
                  hardware_status="derived_only; constant_output_rate_and_phase_not_verified")
    result["config"]["sample_rate_hz"] = 48000
    (root / "suite.json").write_bytes(_json_bytes(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="new output directory; never plays sound")
    parser.add_argument("--durations", type=float, nargs="+", default=[.5, 2.])
    parser.add_argument("--repetitions", type=int, default=4)
    parser.add_argument("--target-rms", type=float, default=.34)
    parser.add_argument("--sample-rate", type=int, choices=(48000, 16000), default=48000,
                        help="native playback rate (default48k);16k is a diagnostic rate-switching control")
    parser.add_argument("--derive-from", type=Path,
                        help="derive an exact ZOH3 48 kHz copy of this archived 16 kHz suite; never plays sound")
    args = parser.parse_args()
    if args.derive_from and args.sample_rate != 48000:
        parser.error("--derive-from always creates48k archives; --sample-rate applies to synthesis")
    suite = (derive_probe_suite_rate(args.derive_from, args.output) if args.derive_from else
             build_probe_suite(args.output, ProbeConfig(sample_rate_hz=args.sample_rate,
                                                        durations_s=tuple(args.durations),
                                                        repetitions=args.repetitions,
                                                        target_rms=args.target_rms)))
    print(json.dumps({"suite_id": suite["suite_id"], "packets": len(suite["packets"]),
                      "manifest": str(args.output / "suite.json")}, indent=2))


if __name__ == "__main__":
    main()
