"""Independent delayed-convolution fixtures for complete acquisition analysis."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zlib

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.acoustics.gathers import analyze_session, _band_power
from sim.acoustics.device.decode import MAGIC, PREFIX, export_capture
from sim.acoustics.probes import ProbeConfig, build_probe_suite, load_probe_packet
from sim.acoustics.survey import make_plan


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def framed_fixture(path, mic, reference, source):
    records = []

    def add(kind, payload=b"", **values):
        index = len(records)
        meta = {"schema": "openpetcam.audio.v1", "record_index": index, "stream_id": "independent-synthetic",
                "dropped_records": 0, "kind": kind, "monotonic_before_ns": 1000000000 + index * 1000,
                "monotonic_after_ns": 1000000100 + index * 1000, "rc": 0, **values}
        raw = json.dumps(meta).encode()
        records.append(PREFIX.pack(MAGIC, len(raw), len(payload), zlib.crc32(raw + payload)) + raw + payload)

    add("hello")
    source_bytes = np.rint(source * 32768).astype("<i2").tobytes()
    for i, lo in enumerate(range(0, len(source_bytes), 2560)):
        chunk = source_bytes[lo:lo + 2560]
        add("ao", chunk, ao={"bytes": len(chunk), "rate_hz": 16000, "bitwidth": 16, "channels": 1, "submission_index": i})
    for i, lo in enumerate(range(0, len(mic), 1920)):
        m = np.rint(mic[lo:lo + 1920] * 32768).astype("<i2").tobytes()
        r = np.rint(reference[lo:lo + 1920] * 32768).astype("<i2").tobytes()
        desc = {"bytes": len(m), "rate_hz": 48000, "bitwidth": 16, "channels": 1,
                "seq": i + 1, "timestamp_us": 10000000 + i * 40000}
        add("mic", m + r, mic=desc, reference={**desc, "seq": 0}, reference_state=1)
    add("end")
    path.write_bytes(b"".join(records))


class GatherTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.session = self.root / "session"
        self.session.mkdir()
        self.suite = build_probe_suite(self.session / "suite", ProbeConfig(sample_rate_hz=16000, durations_s=(.05,), repetitions=2))
        self.cal = {"deg_per_step_pan": .08333, "deg_per_step_tilt": .08333, "level_tilt": 699}

    def tearDown(self):
        self.temp.cleanup()

    def fixture(self, mode="comparison", reference=True):
        ids = [p["packet_id"] for p in self.suite["packets"]]
        plan = make_plan(ids if mode == "comparison" else ids[:1], self.cal,
                         mode=mode, tilts=(610, 790), blocks=1)
        plan["suite_sha256"] = sha(self.session / "suite/suite.json")
        (self.session / "plan.json").write_text(json.dumps(plan))
        by_id = {p["packet_id"]: p for p in self.suite["packets"]}
        rows = []
        for k, e in enumerate(plan["entries"]):
            packet, _, _, manifest = load_probe_packet(self.session / "suite" / by_id[e["packet_id"]]["manifest"])
            up = np.repeat(packet, 3)
            ref = np.pad(up, (12000, 12000))
            ref = np.pad(ref, (0, (-len(ref)) % 1920))
            rng = np.random.default_rng(240 + k)
            mic = rng.normal(0, .00003, len(ref))
            delay = 40 if e["tilt"] <= 699 else 80
            gain = .4 if e["tilt"] <= 699 else .2
            # Independently construct a signed two-path convolution. No forward
            # model, alignment, deconvolution or detector helper is reused.
            mic[delay:] += gain * ref[:-delay]
            mic[delay + 192:] -= .15 * gain * ref[:-(delay + 192)]
            filename = e["capture_id"] + ".pcm"
            path = self.session / filename
            path.write_bytes(np.rint(mic * 32768).astype("<i2").tobytes())
            acq = {"sample_rate_hz": 48000, "samples": len(mic), "timing_status": "unreferenced"}
            if reference:
                framed = self.session / (e["capture_id"] + ".opcaud")
                framed_fixture(framed, mic, ref, packet)
                prefix = e["capture_id"] + ".device"
                device = export_capture(framed, self.session, prefix=prefix)
                dp = self.session / (prefix + ".json")
                acq.update(reference_file=Path(device["reference_file"]).name, reference_sha256=device["reference_sha256"],
                           reference_sample_grid_verified=True, reference_continuity_verified=True,
                           active_ao_reference_verified=True, framed_file=framed.name, framed_sha256=sha(framed),
                           device_metadata_file=dp.name, device_metadata_sha256=sha(dp))
            rows.append(dict(e, file=filename, status="recorded", actual_pose_before=[e["pan"], e["tilt"]],
                             actual_pose_after=[e["pan"], e["tilt"]], pose_stable=True,
                             pcm_sha256=sha(path), packet_sha256=manifest["packet"]["sha256"], acquisition=acq))
        self.write_rows(rows)
        (self.session / "session.json").write_text(json.dumps({"schema": "openpetcam_capture_v1",
            "status": "captured_requires_qc_and_timing_calibration", "completed": len(rows),
            "plan_sha256": sha(self.session / "plan.json"), "suite_sha256": sha(self.session / "suite/suite.json")}))
        for name in ("device_before.json", "device_after.json"):
            (self.session / name).write_text(json.dumps({"audio": {"agc": False}, "build": "synthetic fixture"}))
        return plan, rows

    def write_rows(self, rows):
        (self.session / "captures.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))

    def test_band_power_is_full_window_energy_for_both_sweep_families(self):
        bands = ((2000., 4000.), (4000., 6000.), (6000., 7400.))
        for entry in self.suite["packets"]:
            with self.subTest(family=entry["family"]):
                _, probe, _, _ = load_probe_packet(self.session / "suite" / entry["manifest"])
                # Independent Parseval calculation. A time-domain Hann window
                # weights chirp frequencies differently depending on sweep law.
                frequencies = np.fft.rfftfreq(len(probe), 1 / 16000)
                energy = abs(np.fft.rfft(probe)) ** 2
                energy[1:-1] *= 2
                expected = [energy[(frequencies >= low) & (frequencies < high)].sum() / len(probe) ** 2
                            for low, high in bands]
                np.testing.assert_allclose(_band_power(probe, 16000, bands), expected,
                                           rtol=1e-12, atol=1e-14)

    def test_paired_reference_preserves_signed_delays_amplitude_and_gather_order(self):
        plan, _ = self.fixture("gather")
        out = self.root / "analysis"
        report = analyze_session(self.session, out)
        self.assertTrue(report["complete"], [(q["capture_id"], q.get("error"), q.get("qc", {}).get("rejection_reasons")) for q in report["captures"]])
        self.assertFalse(report["inversion_allowed"])
        self.assertEqual(len(report["gathers"]), 1)
        group = report["gathers"][0]
        self.assertFalse(group["stacked"])
        self.assertEqual(len(group["matrix_rows"]), 8)
        self.assertEqual([r["capture_id"] for r in group["matrix_rows"]],
                         [e["capture_id"] for e in plan["entries"] for _ in range(2)])
        with np.load(out / "shots.npz", allow_pickle=False) as arrays:
            peaks = {}
            for q in report["captures"]:
                h = arrays[q["array_keys"]["responses"]]
                time = arrays[q["array_keys"]["time_s"]]
                peak = int(np.argmax(abs(h[0])))
                delay = 40 if q["entry"]["tilt"] <= 699 else 80
                self.assertAlmostEqual(time[peak] * 48000, delay, delta=1)
                peaks[q["entry"]["tilt"]] = h[0, peak]
                self.assertLess(h[0, peak + 192], 0)
                self.assertEqual(q["timing_status"], "sdk_paired_reference_continuity_checked")
                self.assertFalse(q["deconvolution"]["per_shot_peak_alignment"])
                self.assertLess(time[0], -.049)
            self.assertAlmostEqual(peaks[790] / peaks[610], .5, delta=.01)
        json.dumps(report, allow_nan=False)

    def test_comparison_is_paired_by_block_duration_pose_and_volume_without_winner(self):
        self.fixture()
        report = analyze_session(self.session, self.root / "analysis")
        self.assertTrue(report["complete"])
        self.assertEqual(len(report["comparisons"]), 1)
        pair = report["comparisons"][0]
        self.assertTrue(pair["complete_accepted_pair"])
        self.assertIsNone(pair["winner"])
        self.assertEqual({r["family"] for r in pair["captures"]}, {"linear", "exponential"})
        self.assertTrue(all(len(r["mean_band_snr_db"]) == 3 for r in pair["captures"]))

    def test_missing_corrupt_wrong_pose_and_clipping_are_all_inventory_failures(self):
        _, rows = self.fixture("gather")
        # Keep all four planned entries: one missing, one corrupt, one wrong
        # actual pose, and one clipped but correctly hashed captured trace.
        rows[0]["actual_pose_after"][1] += 1
        cp = self.session / rows[1]["file"]
        data = bytearray(cp.read_bytes()); data[20] ^= 1; cp.write_bytes(data)
        clipped = rows[2]
        p = self.session / clipped["file"]
        x = np.frombuffer(p.read_bytes(), "<i2").copy()
        x[12000 + round(1.58 * 48000) + 100] = 32767
        p.write_bytes(x.tobytes()); clipped["pcm_sha256"] = sha(p)
        acq = clipped["acquisition"]
        old = json.loads((self.session / acq["device_metadata_file"]).read_text())
        ref = np.frombuffer((self.session / acq["reference_file"]).read_bytes(), "<i2").astype(float) / 32768
        src = np.frombuffer(Path(old["source_file"]).read_bytes(), "<i2").astype(float) / 32768
        framed = self.session / acq["framed_file"]
        framed_fixture(framed, x.astype(float) / 32768, ref, src)
        prefix = clipped["capture_id"] + ".device_clipped"
        device = export_capture(framed, self.session, prefix=prefix)
        dp = self.session / (prefix + ".json")
        acq.update(framed_sha256=sha(framed), device_metadata_file=dp.name, device_metadata_sha256=sha(dp),
                   reference_file=Path(device["reference_file"]).name, reference_sha256=device["reference_sha256"])
        self.write_rows(rows[:3])
        report = analyze_session(self.session, self.root / "analysis")
        self.assertFalse(report["complete"])
        self.assertEqual(report["planned_captures"], 4)
        self.assertEqual(report["accepted_captures"], 0)
        self.assertIn("pose", report["captures"][0]["error"])
        self.assertIn("hash", report["captures"][1]["error"])
        self.assertEqual(report["captures"][2]["status"], "rejected")
        self.assertIn("capture_clipping", report["captures"][2]["qc"]["rejection_reasons"])
        self.assertIn("missing", report["captures"][3]["error"])

    def test_fifo_fallback_is_conditional_and_unverified_reference_is_rejected(self):
        _, rows = self.fixture(reference=False)
        report = analyze_session(self.session, self.root / "analysis_fifo")
        self.assertTrue(report["complete"])
        self.assertTrue(all(q["timing_status"] == "local_acoustic_pilot_origin_conditional" for q in report["captures"]))
        self.assertFalse(report["inversion_allowed"])
        rows[0]["acquisition"].update(reference_file="absent.pcm", reference_sha256="notverified")
        self.write_rows(rows)
        bad = analyze_session(self.session, self.root / "analysis_bad")
        self.assertEqual(bad["captures"][0]["status"], "failed")
        self.assertIn("sample grid", bad["captures"][0]["error"])

    def test_source_change_within_gather_rejected_before_analysis(self):
        plan, _ = self.fixture("gather")
        plan["entries"][1]["source_m_assumed"][0] += .01
        (self.session / "plan.json").write_text(json.dumps(plan))
        with self.assertRaisesRegex(ValueError, "source|geometry"):
            analyze_session(self.session, self.root / "analysis")

    def test_failed_session_or_changed_settings_prevents_accepted_comparison(self):
        self.fixture()
        (self.session / "device_after.json").write_text(json.dumps({"audio": {"agc": True}}))
        report = analyze_session(self.session, self.root / "analysis")
        self.assertFalse(report["complete"])
        self.assertFalse(report["session_validation"]["accepted"])
        self.assertEqual(report["accepted_captures"], 0)
        self.assertTrue(all(q["signal_qc_accepted"] for q in report["captures"]))
        self.assertFalse(report["comparisons"][0]["complete_accepted_pair"])

    def test_two_failed_families_remain_in_comparison_inventory(self):
        self.fixture()
        self.write_rows([])
        report = analyze_session(self.session, self.root / "analysis")
        self.assertEqual(len(report["comparisons"]), 1)
        pair = report["comparisons"][0]
        self.assertFalse(pair["complete_accepted_pair"])
        self.assertEqual({q["family"] for q in pair["captures"]}, {"linear", "exponential"})
        self.assertTrue(all(not q["metrics_available"] for q in pair["captures"]))

    def test_framed_proof_and_metadata_are_reverified_not_trusted_flags(self):
        _, rows = self.fixture()
        framed = self.session / rows[0]["acquisition"]["framed_file"]
        data = bytearray(framed.read_bytes()); data[100] ^= 1; framed.write_bytes(data)
        rows[0]["acquisition"]["framed_sha256"] = sha(framed)
        self.write_rows(rows)
        report = analyze_session(self.session, self.root / "analysis")
        self.assertEqual(report["captures"][0]["status"], "failed")
        self.assertIn("device metadata", report["captures"][0]["error"])


if __name__ == "__main__":
    unittest.main()
