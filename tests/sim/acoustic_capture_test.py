"""Acquisition invariants exercised with an in-memory camera, never hardware."""
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sim.acoustics.capture import assets, run_plan
from sim.acoustics.probes import ProbeConfig, build_probe_suite
from sim.acoustics.survey import make_plan


CALIBRATION = {"deg_per_step_pan": .1, "deg_per_step_tilt": .1, "level_tilt": 699}


class FakeCamera:
    def __init__(self, fail=None):
        self.calls = []
        self.original = [1234, 650]
        self.current = self.original.copy()
        self.fail = fail
        self.used_failure = False

    def snapshot(self):
        self.calls.append(("snapshot",))
        if self.fail == "config_change" and sum(c[0] == "snapshot" for c in self.calls) > 1:
            return {"audio": {"gain": 11}, "timing_capability": "unframed_pcm", "runtime_audio_format": {"record_rate_hz": 48000, "output_rate_hz": 48000, "record_format_bits": 16, "output_format_bits": 16}}
        return {"audio": {"gain": 10}, "timing_capability": "unframed_pcm", "runtime_audio_format": {"record_rate_hz": 48000, "output_rate_hz": 48000, "record_format_bits": 16, "output_format_bits": 16}}

    def pose(self):
        self.calls.append(("pose",))
        return self.current.copy()

    @contextlib.contextmanager
    def hold(self):
        self.calls.append(("hold_enter",))
        try:
            yield
        finally:
            self.calls.append(("hold_exit",))

    def goto(self, pan, tilt):
        self.calls.append(("goto", pan, tilt))
        restoring = [pan, tilt] == self.original and any(c[0] == "record" for c in self.calls)
        if self.fail == "restore_error" and restoring:
            raise RuntimeError("fake restore failure")
        if self.fail == "goto" and not self.used_failure:
            self.used_failure = True
            raise RuntimeError("fake motor failure")
        self.current = [pan, tilt]
        if self.fail == "restore_mismatch" and restoring:
            self.current[0] += 1
        if self.fail == "pose_mismatch" and not self.used_failure:
            self.used_failure = True
            self.current[0] += 1

    def record(self, wav_path, output, volume, duration_s):
        self.calls.append(("record", str(wav_path), str(output), volume, duration_s))
        count = round((duration_s + 3) * 48000)
        Path(output).write_bytes(np.zeros(count, dtype="<i2").tobytes())
        if self.fail == "record" and not self.used_failure:
            self.used_failure = True
            raise RuntimeError("fake interrupted reader")
        if self.fail == "movement" and not self.used_failure:
            self.used_failure = True
            self.current[1] += 1
        return {"sample_rate_hz": 48000, "samples": count, "hardware_timing_verified": False}


class CaptureTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.suite = build_probe_suite(self.root / "suite", ProbeConfig(durations_s=(.5,), repetitions=2))
        self.suite_path = self.root / "suite" / "suite.json"
        self.plan = make_plan([e["packet_id"] for e in self.suite["packets"]], CALIBRATION, blocks=1)
        self.plan["suite_sha256"] = hashlib.sha256(self.suite_path.read_bytes()).hexdigest()
        self.plan_path = self.root / "plan.json"
        self.write_plan()

    def write_plan(self):
        self.plan_path.write_text(json.dumps(self.plan))

    def rows(self, output):
        return [json.loads(line) for line in (output / "captures.jsonl").read_text().splitlines()]

    def test_all_asset_hashes_checked_before_any_hardware_or_output(self):
        # Even a packet absent from the selected plan must validate before capture.
        self.plan["entries"] = self.plan["entries"][:1]
        self.write_plan()
        unused = next(e for e in self.suite["packets"] if e["packet_id"] != self.plan["entries"][0]["packet_id"])
        directory = self.root / "suite" / Path(unused["manifest"]).parent
        probe = directory / "probe.wav"
        probe.write_bytes(probe.read_bytes()[:-2] + b"\x01\x02")
        camera = FakeCamera()
        output = self.root / "bad"
        with self.assertRaises(ValueError):
            run_plan(self.plan_path, self.suite_path, output, camera)
        self.assertEqual(camera.calls, [])
        self.assertFalse(output.exists())

    def test_success_inventory_archives_exact_playback_and_restores_pose(self):
        camera = FakeCamera()
        output = self.root / "captured"
        summary = run_plan(self.plan_path, self.suite_path, output, camera)
        self.assertEqual(summary["completed"], 2)
        self.assertEqual(summary["status"], "captured_requires_qc_and_timing_calibration")
        self.assertFalse(summary["inversion_allowed"])
        self.assertFalse(summary["hardware_timing_verified"])
        self.assertEqual(camera.current, camera.original)
        rows = self.rows(output)
        self.assertEqual(len(rows), 2)
        self.assertEqual({r["capture_id"] for r in rows}, {r["capture_id"] for r in self.plan["entries"]})
        for row in rows:
            raw = (output / row["file"]).read_bytes()
            self.assertEqual(row["pcm_sha256"], hashlib.sha256(raw).hexdigest())
            self.assertEqual(len(raw), 2 * row["acquisition"]["samples"])
            self.assertTrue(row["pose_stable"])
        for call in camera.calls:
            if call[0] == "record":
                self.assertTrue(Path(call[1]).resolve().is_relative_to((output / "suite").resolve()), "play the archived verified copy")

    def test_pose_mismatch_goto_and_record_failures_are_logged_and_restore(self):
        for failure in ("pose_mismatch", "goto", "record"):
            with self.subTest(failure=failure):
                camera = FakeCamera(fail=failure)
                output = self.root / failure
                with self.assertRaises(RuntimeError):
                    run_plan(self.plan_path, self.suite_path, output, camera)
                summary = json.loads((output / "session.json").read_text())
                self.assertEqual(summary["status"], "failed")
                self.assertEqual(summary["completed"], 0)
                self.assertEqual(camera.current, camera.original)
                rows = self.rows(output)
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]["capture_id"], self.plan["entries"][0]["capture_id"])
                self.assertEqual(rows[0]["status"], "failed")
                self.assertTrue(rows[0]["error"])

    def test_movement_during_record_is_logged_as_rejected(self):
        camera = FakeCamera(fail="movement")
        output = self.root / "moved"
        with self.assertRaises(RuntimeError):
            run_plan(self.plan_path, self.suite_path, output, camera)
        row = self.rows(output)[0]
        self.assertFalse(row["pose_stable"])
        self.assertEqual(json.loads((output / "session.json").read_text())["completed"], 0)
        self.assertEqual(camera.current, camera.original)

    def test_deadline_prevents_next_motion_and_invalid_bounds_touch_no_hardware(self):
        camera = FakeCamera()
        output = self.root / "deadline"
        with self.assertRaisesRegex(RuntimeError, "deadline"):
            run_plan(self.plan_path, self.suite_path, output, camera, deadline_s=.1)
        self.assertFalse(any(call[0] == "record" for call in camera.calls))
        self.assertEqual([call for call in camera.calls if call[0] == "goto"], [("goto", *camera.original)])
        self.assertEqual(camera.current, camera.original)
        for index, deadline in enumerate((0, -1, float("nan"), float("inf"))):
            camera = FakeCamera()
            with self.subTest(deadline=deadline), self.assertRaises(ValueError):
                run_plan(self.plan_path, self.suite_path, self.root / f"invalid_deadline_{index}", camera, deadline_s=deadline)
            self.assertEqual(camera.calls, [])

    def test_deadline_reserves_upload_startup_capture_and_cleanup_before_motion(self):
        # One minute cannot cover bounded upload/startup plus the complete packet.
        # The old packet+15 lookahead wrongly began this capture.
        camera = FakeCamera()
        with self.assertRaisesRegex(RuntimeError, "deadline"):
            run_plan(self.plan_path, self.suite_path, self.root / "budget", camera, deadline_s=60)
        self.assertFalse(any(call[0] == "record" for call in camera.calls))
        self.assertEqual([call for call in camera.calls if call[0] == "goto"], [("goto", *camera.original)])

    def test_partial_capture_cannot_be_resumed_or_overwritten(self):
        output = self.root / "partial"
        with self.assertRaises(RuntimeError):
            run_plan(self.plan_path, self.suite_path, output, FakeCamera(fail="record"))
        before = {str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()}
        camera = FakeCamera()
        with self.assertRaises(FileExistsError):
            run_plan(self.plan_path, self.suite_path, output, camera)
        self.assertEqual(camera.calls, [])
        after = {str(p.relative_to(output)): p.read_bytes() for p in output.rglob("*") if p.is_file()}
        self.assertEqual(before, after)

    def test_restore_errors_or_wrong_restored_pose_cannot_report_success(self):
        for failure in ("restore_error", "restore_mismatch"):
            with self.subTest(failure=failure):
                output = self.root / failure
                with self.assertRaisesRegex(RuntimeError, "restor"):
                    run_plan(self.plan_path, self.suite_path, output, FakeCamera(fail=failure))
                summary = json.loads((output / "session.json").read_text())
                self.assertIn("failed", summary["status"])
                self.assertTrue(summary["restore_error"])

    def test_configuration_change_is_failed_and_pose_restored(self):
        camera = FakeCamera(fail="config_change")
        output = self.root / "configuration"
        with self.assertRaisesRegex(RuntimeError, "configuration changed"):
            run_plan(self.plan_path, self.suite_path, output, camera)
        self.assertEqual(camera.current, camera.original)
        self.assertEqual(json.loads((output / "session.json").read_text())["status"], "failed")

    def test_mutated_plan_coordinates_identity_and_selection_rejected_before_hardware(self):
        original = json.loads(json.dumps(self.plan))
        cases = [("pan", 1850.5), ("tilt", 699.5), ("volume", 30.5),
                 ("volume", True), ("source_id", "pan_999"), ("receiver_id", "wrong"),
                 ("source_m_assumed", [999, 999, 999]), ("packet_id", "missing")]
        for index, (field, value) in enumerate(cases):
            self.plan = json.loads(json.dumps(original))
            self.plan["entries"][0][field] = value
            self.write_plan()
            camera = FakeCamera()
            with self.subTest(field=field), self.assertRaises(ValueError):
                run_plan(self.plan_path, self.suite_path, self.root / f"invalid_{index}", camera)
            self.assertEqual(camera.calls, [])
        # One fixed-source gather cannot silently change its source pan.
        self.plan = make_plan([self.suite["packets"][0]["packet_id"]], CALIBRATION,
                              mode="gather", blocks=1, tilts=(520, 880))
        self.plan["suite_sha256"] = original["suite_sha256"]
        self.plan["entries"][1]["pan"] = 1900
        self.write_plan()
        camera = FakeCamera()
        with self.assertRaises(ValueError):
            run_plan(self.plan_path, self.suite_path, self.root / "bad_gather", camera)
        self.assertEqual(camera.calls, [])

    def test_duplicate_suite_packet_ids_are_not_silently_overwritten(self):
        self.suite["packets"].append(self.suite["packets"][0])
        self.suite_path.write_text(json.dumps(self.suite))
        self.plan["suite_sha256"] = hashlib.sha256(self.suite_path.read_bytes()).hexdigest()
        self.write_plan()
        with self.assertRaises(ValueError):
            assets(self.plan_path, self.suite_path)

    def test_missing_or_misordered_gather_reference_visits_fail_before_hardware(self):
        base = make_plan([self.suite["packets"][0]["packet_id"]], CALIBRATION,
                         mode="gather", blocks=1, tilts=(520, 880))
        base["suite_sha256"] = self.plan["suite_sha256"]
        for variant in ("missing_after", "wrong_order", "changed_probe"):
            with self.subTest(variant=variant):
                self.plan = json.loads(json.dumps(base))
                if variant == "missing_after":
                    self.plan["entries"].pop()
                elif variant == "wrong_order":
                    self.plan["entries"][0], self.plan["entries"][1] = self.plan["entries"][1], self.plan["entries"][0]
                else:
                    self.plan["entries"][1]["packet_id"] = self.suite["packets"][1]["packet_id"]
                self.write_plan()
                camera = FakeCamera()
                with self.assertRaises(ValueError):
                    run_plan(self.plan_path, self.suite_path, self.root / variant, camera)
                self.assertEqual(camera.calls, [])

    def test_volume_study_validates_and_records_each_level_without_changing_pose(self):
        self.plan = make_plan([self.suite["packets"][0]["packet_id"]], CALIBRATION,
                              mode="volume", volumes=(20, 30), volume=30, blocks=1)
        self.plan["suite_sha256"] = hashlib.sha256(self.suite_path.read_bytes()).hexdigest()
        self.write_plan()
        plan, _ = assets(self.plan_path, self.suite_path)
        self.assertEqual(plan["mode"], "volume")
        camera = FakeCamera()
        output = self.root / "volume-session"
        result = run_plan(self.plan_path, self.suite_path, output, camera)
        self.assertEqual(result["completed"], 4)
        rows = self.rows(output)
        self.assertEqual([rows[0]["volume"], rows[-1]["volume"]], [30, 30])
        self.assertEqual({row["volume"] for row in rows[1:-1]}, {20, 30})
        self.assertTrue(all(row["actual_pose_before"] == [1850, 699] for row in rows))
        self.assertEqual(camera.current, camera.original)

    def test_volume_study_rejects_changed_pose_probe_levels_anchors_or_block_order(self):
        base = make_plan([self.suite["packets"][0]["packet_id"]], CALIBRATION,
                         mode="volume", blocks=2)
        base["suite_sha256"] = hashlib.sha256(self.suite_path.read_bytes()).hexdigest()
        variants = ("pose", "probe", "missing_trial", "duplicate_level", "unexpected_level",
                    "anchor_volume", "anchor_role", "noncontiguous", "missing_block",
                    "metadata_levels", "metadata_reference", "absent_reference_trial", "metadata_purpose", "metadata_blocks")
        for variant in variants:
            with self.subTest(variant=variant):
                self.plan = json.loads(json.dumps(base))
                entries = self.plan["entries"]
                if variant == "pose":
                    entries[1]["tilt"] = 700
                    entries[1]["receiver_id"] = "pan_1850_tilt_700"
                elif variant == "probe":
                    entries[1]["packet_id"] = self.suite["packets"][1]["packet_id"]
                elif variant == "missing_trial":
                    entries.pop(1)
                elif variant == "duplicate_level":
                    entries[2]["volume"] = entries[1]["volume"]
                elif variant == "unexpected_level":
                    entries[1]["volume"] = 10
                elif variant == "anchor_volume":
                    entries[0]["volume"] = 20
                elif variant == "anchor_role":
                    entries[0]["role"] = "volume_trial"
                elif variant == "noncontiguous":
                    entries[1], entries[6] = entries[6], entries[1]
                elif variant == "missing_block":
                    del entries[5:]
                elif variant == "metadata_levels":
                    self.plan["volumes"] = [20, 20, 40]
                elif variant == "metadata_reference":
                    self.plan["reference_volume"] = True
                elif variant == "absent_reference_trial":
                    self.plan["volumes"] = [10, 20, 40]
                elif variant == "metadata_purpose":
                    self.plan["purpose"] = "unknown"
                elif variant == "metadata_blocks":
                    self.plan["blocks"] = 2.5
                self.write_plan()
                camera = FakeCamera()
                with self.assertRaises(ValueError):
                    run_plan(self.plan_path, self.suite_path, self.root / ("bad_volume_" + variant), camera)
                self.assertEqual(camera.calls, [])


if __name__ == "__main__":
    unittest.main()
