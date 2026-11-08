"""Known gain/noise fixtures for offline, scale-preserving volume summaries."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.acoustics.survey import make_plan
from sim.acoustics.volume import analyze_volume, measure_capture, summarize_volume


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def metric_fixture(cid="capture", reference_gain=.5, acoustic_gain=.4, noise=.002, fraction=.0):
    """Numerical metrics fixture; hardware proof validation is tested by gathers.

    Independently synthesize analytic chirp samples, a delayed scaled microphone
    signal with additive noise, and its known signed impulse response. All time
    offsets are prescribed; this fixture performs no peak-based alignment.
    """
    fs, n, pre, tail = 48000, 24000, 1200, 7200
    m = n + pre + tail
    k = np.arange(m) - pre - fraction
    t = k / fs
    envelope = np.where((k >= 0) & (k < n), np.sin(np.pi * np.clip(k / n, 0, 1)) ** 2, 0.)
    source = .4 * envelope * np.cos(2 * np.pi * (2200 * t + 4000 * t ** 2))
    # SDK reference captures digital gain. The acoustic path gain remains a
    # separate multiplier after the reference tap.
    reference = reference_gain * source
    shifted = np.interp(np.arange(m) - 24, np.arange(m), reference, left=0, right=0)
    rng = np.random.default_rng(813)
    raw = acoustic_gain * shifted + rng.normal(0, noise, (4, m))
    refs = np.tile(reference, (4, 1))
    times = np.arange(-n, tail) / fs
    impulse = np.exp(-.5 * ((times - 24 / fs) * fs / 2) ** 2)
    impulse -= .15 * np.exp(-.5 * ((times - 216 / fs) * fs / 2) ** 2)
    responses = acoustic_gain * np.tile(impulse, (4, 1))
    ranges, shots = [], []
    for i in range(4):
        src = i * 24000
        origin = 100000 + fraction + 3 * src
        ranges.append([int(np.floor(origin)) - pre, int(np.floor(origin)) - pre + m])
        bands = []
        for lo, hi in ((2000, 4000), (4000, 6000), (6000, 7400)):
            total = (acoustic_gain * reference_gain) ** 2 * .02 + noise ** 2
            bands.append({"band_hz": [lo, hi], "total_signal_power": total, "noise_power": noise ** 2,
                          "snr_db": float(10 * np.log10(total / noise ** 2))})
        shots.append({"accepted": True, "event_id": f"shot_{i}", "source_start_sample": src,
                      "noise_rms": noise, "normalized_correlation": .98,
                      "matched_filter_amplitude": acoustic_gain * reference_gain,
                      "clipped_samples": 0, "band_snr_db": bands})
    values = {"raw": raw, "reference": refs, "responses": responses, "time_s": times,
              "adc_sample_ranges": np.array(ranges)}
    keys = {name: cid + "_" + name for name in values}
    arrays = {keys[name]: value for name, value in values.items()}
    q = {"capture_id": cid, "status": "accepted", "timing_status": "sdk_paired_reference_continuity_checked",
         "reference_proof": {"independently_redecoded": True, "source_packet_bit_identical": True,
                             "active_ao_reference_check": {"verified": True, "global_gain": reference_gain}},
         "array_keys": keys, "timing_map": {"intercept": 100000 + fraction, "slope": 3.},
         "probe": {"samples": 8000, "family": "linear"},
         "qc": {"sample_rate_hz": fs, "shots": shots, "clipped_samples": 0,
                "repeat_coherence": {"trace_count": 4, "bands": []}, "rejection_reasons": []}}
    return q, arrays


class VolumeTest(unittest.TestCase):
    def plan(self, blocks=1):
        cal = {"deg_per_step_pan": .08333, "deg_per_step_tilt": .08333, "level_tilt": 699}
        return make_plan(["same-probe"], cal, mode="volume", blocks=blocks, volume=30, volumes=(20, 30, 40))

    def test_reference_gain_is_separate_from_reference_normalized_transfer(self):
        low, la = metric_fixture(reference_gain=.25)
        high, ha = metric_fixture(reference_gain=.5)
        a, b = measure_capture(low, la), measure_capture(high, ha)
        self.assertAlmostEqual(b["reference_normalized_response_norm"] / a["reference_normalized_response_norm"], 1.)
        self.assertAlmostEqual(b["coherent_command_source_relative_response_norm"] / a["coherent_command_source_relative_response_norm"], 2.)
        self.assertAlmostEqual(b["sdk_reference_probe_window_rms"] / a["sdk_reference_probe_window_rms"], 2.)
        self.assertAlmostEqual(b["mic_probe_window_rms"] / a["mic_probe_window_rms"], 2., delta=.02)
        self.assertEqual(low["timing_map"], high["timing_map"])

    def test_gain_after_reference_tap_changes_transfer_while_reference_stays_fixed(self):
        low, la = metric_fixture(reference_gain=.5, acoustic_gain=.2)
        high, ha = metric_fixture(reference_gain=.5, acoustic_gain=.4)
        a, b = measure_capture(low, la), measure_capture(high, ha)
        self.assertEqual(a["sdk_reference_gain_from_submitted_ao"], b["sdk_reference_gain_from_submitted_ao"])
        self.assertAlmostEqual(b["coherent_reference_normalized_response_norm"] / a["coherent_reference_normalized_response_norm"], 2.)
        self.assertAlmostEqual(b["nominal_source_matched_amplitude"] / a["nominal_source_matched_amplitude"], 2.)

    def test_fractional_raw_sample_bookkeeping_is_corrected_once_only(self):
        integer, ia = metric_fixture(noise=.000001, fraction=0)
        fractional, fa = metric_fixture(noise=.000001, fraction=.63)
        original = fa[fractional["array_keys"]["raw"]].copy()
        a, b = measure_capture(integer, ia), measure_capture(fractional, fa)
        self.assertAlmostEqual(a["sdk_reference_probe_window_rms"], b["sdk_reference_probe_window_rms"], places=8)
        self.assertAlmostEqual(a["mic_probe_window_rms"], b["mic_probe_window_rms"], places=6)
        np.testing.assert_array_equal(fa[fractional["array_keys"]["raw"]], original)

    def test_signed_mean_and_repeat_residual_expose_norm_noise_bias(self):
        q, arrays = metric_fixture()
        clean = measure_capture(q, arrays)
        h = arrays[q["array_keys"]["responses"]]
        rng = np.random.default_rng(14)
        noise = rng.normal(0, .08, h.shape[1])
        h[0] += noise; h[1] -= noise; h[2] += noise; h[3] -= noise
        noisy = measure_capture(q, arrays)
        self.assertGreater(noisy["reference_normalized_response_norm"], clean["reference_normalized_response_norm"])
        self.assertAlmostEqual(noisy["coherent_reference_normalized_response_norm"], clean["coherent_reference_normalized_response_norm"], places=10)
        self.assertGreater(noisy["repeat_response_residual_norm"], 0)

    def test_block_pairs_do_not_count_anchors_as_extra_reference_trials(self):
        plan = self.plan(3)
        captures = []
        for e in plan["entries"]:
            q, a = metric_fixture(reference_gain={20: .1, 30: .2, 40: .4}[e["volume"]])
            captures.append({"capture_id": e["capture_id"], "entry": e, "usable": True, "metrics": measure_capture(q, a)})
        summary = summarize_volume(plan, captures)
        self.assertEqual([q["usable_trials"] for q in summary["levels"]], [3, 3, 3])
        self.assertEqual(len(summary["reference_anchor_drift"]), 3)
        pair = next(q for q in summary["within_block_pairs"] if q["volume"] == 40)
        self.assertAlmostEqual(pair["changes_db"]["coherent_command_source_relative_response_norm"], 20 * np.log10(2))
        self.assertIsNone(summary["winner"])

    def test_rejected_capture_remains_in_inventory_and_hash_tamper_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); session = root / "session"; session.mkdir()
            plan = self.plan(); (session / "plan.json").write_text(json.dumps(plan))
            all_arrays, captures = {}, []
            for e in plan["entries"]:
                q, a = metric_fixture(e["capture_id"])
                q["entry"] = e
                if e["role"] == "volume_trial" and e["volume"] == 20:
                    q["status"] = "rejected"
                    q["qc"].update(clipped_samples=3, rejection_reasons=["capture_clipping"])
                captures.append(q); all_arrays.update(a)
            np.savez_compressed(root / "shots.npz", **all_arrays)
            report = {"schema": "openpetcam_shot_gathers_v1", "session": str(session),
                      "session_validation": {"accepted": True}, "plan_sha256": digest(session / "plan.json"),
                      "arrays_file": "shots.npz", "arrays_sha256": digest(root / "shots.npz"), "captures": captures}
            (root / "report.json").write_text(json.dumps(report))
            has_plotting = importlib.util.find_spec("matplotlib") is not None
            result = analyze_volume(root / "report.json", root / "volume", plot=has_plotting)
            self.assertFalse(result["complete_usable"])
            self.assertEqual(len(result["captures"]), 5)
            rejected = next(q for q in result["captures"] if not q["usable"])
            self.assertEqual(rejected["qc_diagnostics"]["clipped_samples"], 3)
            self.assertFalse(next(q for q in result["within_block_pairs"] if q["volume"] == 20)["usable_pair"])
            self.assertFalse(result["calibrated_spl"])
            if has_plotting:
                self.assertGreater((root / "volume/volume.png").stat().st_size, 10000)
            raw = (root / "shots.npz").read_bytes()
            (root / "shots.npz").write_bytes(raw + b"changed")
            with self.assertRaisesRegex(ValueError, "hash"):
                analyze_volume(root / "report.json", root / "volume_bad")

    def test_unverified_reference_and_changed_source_pose_are_rejected(self):
        q, arrays = metric_fixture()
        q["reference_proof"]["active_ao_reference_check"]["verified"] = False
        with self.assertRaisesRegex(ValueError, "verified"):
            measure_capture(q, arrays)
        plan = copy.deepcopy(self.plan())
        plan["entries"][1]["pan"] += 1
        with self.assertRaisesRegex(ValueError, "pose"):
            summarize_volume(plan, [])


if __name__ == "__main__":
    unittest.main()
