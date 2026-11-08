"""IR acquisition contracts with an injected camera; never communicates with hardware."""
import contextlib
from copy import deepcopy
from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sim.world.ir_exp import CaptureError, ThinginoBackend, run_capture


class FakeBackend:
    def __init__(self, controlled=True, fail=None):
        self.controlled, self.fail, self.calls = controlled, fail, []
        self.state = {"auto_enabled": True, "force_mode": "", "running_mode": 0, "ircut": 1, "led": 1}
        self.original = deepcopy(self.state)
        self.frames = 0
        self.reads = 0

    def capabilities(self):
        self.calls.append("capabilities")
        return {"exposure_lock": self.controlled, "gain_lock": self.controlled, "processing_lock": self.controlled,
                "radiometry": {"response_model": "linear" if self.controlled else "unknown", "verified": self.controlled,
                               "black_level": 0, "white_level": 255}}

    @contextlib.contextmanager
    def session(self):
        self.calls.append("lock")
        if self.fail == "privacy":
            raise RuntimeError("privacy is on")
        try:
            yield
        finally:
            self.calls.append("unlock")

    def snapshot_state(self):
        return deepcopy(self.state)

    def prepare_night(self, qualitative):
        self.calls.append("prepare")
        self.state.update(auto_enabled=False, force_mode="night", running_mode=1, ircut=0, led=1)
        if self.fail == "prepare":
            raise RuntimeError("partial preparation failed")

    def set_led(self, on):
        self.calls.append(("led", on))
        self.state["led"] = int(on)

    def read_controls(self):
        self.reads += 1
        out = {"night_mode": "night", "ircut": "out", "led": "on" if self.state["led"] else "off", "pan": 1234,
               "tilt": 650, "moving": False, "verified": self.controlled,
               "exposure_us": 12000., "analog_gain": 2., "digital_gain": 1., "processing_signature": "fixed-isp"}
        if self.reads > 3:
            if self.fail == "exposure":
                out["exposure_us"] *= 2
            if self.fail == "gain":
                out["analog_gain"] *= 2
            if self.fail == "processing":
                out["processing_signature"] = "changed-isp"
            if self.fail == "motion":
                out["pan"] += 1
            if self.fail == "filter":
                out["ircut"] = "in"
            if self.fail == "led":
                out["led"] = "off" if out["led"] == "on" else "on"
        return out

    def snapshot_jpeg(self):
        self.frames += 1
        if self.frames == 2 and self.fail == "snapshot":
            raise RuntimeError("JPEG failed")
        if self.frames == 2 and self.fail == "interrupt":
            raise KeyboardInterrupt
        if self.fail == "bad_jpeg":
            return b"not a JPEG"
        out = BytesIO()
        Image.new("L", (16, 10), 20 + self.state["led"] * 40).save(out, format="JPEG")
        return out.getvalue()

    def restore(self, original):
        self.calls.append("restore")
        self.state = deepcopy(original)
        if self.fail == "restore":
            self.state["led"] = 0
            raise RuntimeError("LED restore failed")
        if self.fail == "restore_mismatch":
            self.state["ircut"] = 0


class CaptureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name) / "capture"

    def run_fake(self, backend, **kwargs):
        return run_capture(backend, self.out, burst=3, settle_s=0, frame_interval_s=0,
                           sleep=lambda _: None, **kwargs)

    def test_preflight_rejects_unsupported_locks_before_session_or_mutation(self):
        backend = FakeBackend(controlled=False)
        with self.assertRaisesRegex(CaptureError, "unsupported"):
            self.run_fake(backend)
        self.assertEqual(backend.calls, ["capabilities"])
        self.assertEqual(backend.state, backend.original)
        self.assertEqual(json.loads((self.out / "log.json").read_text())["status"], "failed")

    def test_current_real_backend_does_not_claim_locks_or_contact_camera_on_preflight(self):
        with patch("sim.world.ir_exp.subprocess.run", side_effect=AssertionError("No camera calls permitted")):
            with self.assertRaisesRegex(CaptureError, "unsupported"):
                self.run_fake(ThinginoBackend(["fake-ssh"]))

    def test_controlled_bursts_change_only_led_and_restore_exact_original(self):
        backend = FakeBackend()
        log = self.run_fake(backend)
        self.assertEqual([b["state"] for b in log["bursts"]], ["off", "on", "off", "on", "off"])
        self.assertEqual([f["phase"] for f in log["frames"]], [p for p in range(5) for _ in range(3)])
        self.assertEqual(len(log["controls_readbacks"]), 30)
        self.assertEqual(len(list(self.out.glob("*.jpg"))), 15)
        self.assertTrue(all(log[k] for k in ("controls_verified", "exposure_locked", "gain_locked", "processing_locked",
                                            "same_night_mode", "ircut_unchanged", "pose_fixed")))
        self.assertEqual(backend.calls.count("prepare"), 1)
        self.assertEqual(backend.state, backend.original)
        self.assertTrue(log["restoration"]["success"])
        self.assertEqual(log["status"], "complete")
        self.assertEqual(backend.calls[-2:], ["restore", "unlock"])

    def test_qualitative_never_labels_unlocked_capture_as_verified(self):
        log = self.run_fake(FakeBackend(controlled=False), qualitative=True)
        self.assertFalse(log["controls_verified"])
        self.assertFalse(log["exposure_locked"])
        self.assertFalse(log["gain_locked"])
        self.assertFalse(log["processing_locked"])
        self.assertFalse(log["radiometry"]["verified"])
        self.assertEqual(log["radiometry"]["response_model"], "unknown")
        self.assertTrue(log["limitations"])

    def test_capture_failures_and_interruptions_restore_and_leave_failure_log(self):
        for failure in ("prepare", "snapshot", "interrupt", "bad_jpeg", "exposure", "gain", "processing", "motion", "filter", "led"):
            with self.subTest(failure=failure):
                self.out = Path(self.temp.name) / failure
                backend = FakeBackend(fail=failure)
                with self.assertRaises(CaptureError):
                    self.run_fake(backend)
                log = json.loads((self.out / "log.json").read_text())
                self.assertEqual(log["status"], "failed")
                self.assertTrue(log["restoration"]["success"])
                self.assertFalse(log["controls_verified"])
                self.assertEqual(backend.state, backend.original)
                self.assertEqual(backend.calls[-1], "unlock")

    def test_restoration_failure_or_mismatch_is_not_reported_successful(self):
        for failure in ("restore", "restore_mismatch"):
            with self.subTest(failure=failure):
                self.out = Path(self.temp.name) / failure
                with self.assertRaisesRegex(CaptureError, "restoration"):
                    self.run_fake(FakeBackend(fail=failure))
                log = json.loads((self.out / "log.json").read_text())
                self.assertFalse(log["restoration"]["success"])
                self.assertEqual(log["status"], "failed")
                self.assertFalse(log["controls_verified"])

    def test_privacy_or_lock_prevents_preparation(self):
        backend = FakeBackend(fail="privacy")
        with self.assertRaisesRegex(CaptureError, "privacy"):
            self.run_fake(backend)
        self.assertNotIn("prepare", backend.calls)
        self.assertEqual(backend.state, backend.original)

    def test_nonempty_output_is_never_overwritten(self):
        self.out.mkdir()
        (self.out / "old.jpg").write_bytes(b"old")
        backend = FakeBackend()
        with self.assertRaisesRegex(ValueError, "empty"):
            self.run_fake(backend)
        self.assertEqual(backend.calls, [])
        self.assertEqual((self.out / "old.jpg").read_bytes(), b"old")

    def test_real_session_uses_exclusive_lock_and_never_removes_an_existing_lock(self):
        class App:
            @staticmethod
            def require_camera_active():
                pass
        directory = Path(self.temp.name)
        backend = ThinginoBackend([], directory, App())
        lock = directory / "capture.lock"
        lock.write_text("another capture")
        with self.assertRaises(FileExistsError):
            with backend.session():
                self.fail("Existing capture lock must prevent entry")
        self.assertEqual(lock.read_text(), "another capture")
        lock.unlink()
        with backend.session():
            self.assertTrue(lock.exists())
            self.assertTrue((directory / "moved_at").exists())
        self.assertFalse(lock.exists())

    def test_real_restore_attempts_all_steps_after_an_individual_failure(self):
        class Backend(ThinginoBackend):
            def __init__(self):
                super().__init__([])
                self.commands = []

            def _run(self, args):
                self.commands.append(args)
                if args[0] == "ircut":
                    raise RuntimeError("filter failure")
                return b""

            def _wait_policy(self, expected=None, timeout=15.):
                self.commands.append(["ack", expected])
        backend = Backend()
        original = FakeBackend().original
        with self.assertRaisesRegex(CaptureError, "filter failure"):
            backend.restore(original)
        self.assertEqual([c[0] for c in backend.commands], ["jct", "jct", "sh", "ack", "prudyntctl", "ircut", "light"])
        self.assertEqual(backend.commands[6], ["light", "ir940", "on"])
        self.assertEqual(backend.commands[0][-1], "true")
        self.assertEqual(backend.commands[1][-1], "")

    def test_malformed_radiometry_is_rejected_before_session_or_camera_mutation(self):
        valid = {"response_model": "lut", "verified": True, "black_level": 0, "white_level": 255, "lut": [0., .4, 1.]}
        malformed = [None, [], {k: v for k, v in valid.items() if k != "black_level"},
                     {k: v for k, v in valid.items() if k != "white_level"},
                     dict(valid, black_level=True), dict(valid, white_level=False), dict(valid, black_level="0"),
                     dict(valid, black_level=float("nan")), dict(valid, white_level=float("inf")),
                     dict(valid, white_level=0), dict(valid, noise_floor=True), dict(valid, noise_floor=-1),
                     dict(valid, max_off_drift=float("nan")), dict(valid, max_off_drift=-.1)]
        malformed += [dict(valid, lut=v) for v in ([], [1], [0, False, 1], [0, "0.4", 1], [0, float("nan"), 1],
                                                   [0, float("inf"), 1], [0, 1, .5], [1, 1], "012")]
        for i, radiometry in enumerate(malformed):
            with self.subTest(radiometry=radiometry):
                self.out = Path(self.temp.name) / f"malformed_{i}"
                backend = FakeBackend()
                original = backend.capabilities
                backend.capabilities = lambda: dict(original(), radiometry=radiometry)
                with self.assertRaisesRegex(CaptureError, "unsupported"):
                    self.run_fake(backend)
                self.assertEqual(backend.calls, ["capabilities"])
                self.assertEqual(backend.state, backend.original)

    def test_controlled_even_bursts_fail_before_camera_access_but_qualitative_is_allowed(self):
        backend = FakeBackend()
        with self.assertRaisesRegex(ValueError, "odd burst"):
            run_capture(backend, self.out, burst=2)
        self.assertEqual(backend.calls, [])
        log = run_capture(backend, self.out, burst=2, qualitative=True, settle_s=0, frame_interval_s=0, sleep=lambda _: None)
        self.assertEqual(log["status"], "complete")
        self.assertFalse(log["controls_verified"])

    def test_boolean_and_non_numeric_readbacks_are_rejected_and_restored(self):
        for i, (key, value) in enumerate((("pan", True), ("tilt", False), ("exposure_us", True),
                                          ("analog_gain", "2"), ("digital_gain", False), ("processing_signature", 42))):
            with self.subTest(key=key, value=value):
                self.out = Path(self.temp.name) / f"readback_{i}"
                backend = FakeBackend()
                original = backend.read_controls
                backend.read_controls = lambda: dict(original(), **{key: value})
                with self.assertRaises(CaptureError):
                    self.run_fake(backend)
                self.assertEqual(backend.frames, 0)
                self.assertEqual(backend.state, backend.original)
                self.assertTrue(json.loads((self.out / "log.json").read_text())["restoration"]["success"])

    def test_policy_ack_requires_two_fresh_matching_daemon_samples(self):
        class Backend(ThinginoBackend):
            def __init__(self):
                super().__init__([])
                self.samples = iter([(101, "night"), (101, "night"), (102, "day"), (103, "night"), (104, "night")])
                self.reads = 0

            def _guard(self):
                pass

            def _text(self, args):
                self.reads += 1
                stamp, mode = next(self.samples)
                return json.dumps({"time_now": stamp, "mode": mode})
        backend = Backend()
        with patch("sim.world.ir_exp.time.time", return_value=100.1), \
                patch("sim.world.ir_exp.time.monotonic", return_value=1), \
                patch("sim.world.ir_exp.time.sleep"):
            backend._wait_policy("night")
        self.assertEqual(backend.reads, 5)

    def test_real_restore_waits_for_policy_before_hardware_and_checks_stability(self):
        class Backend(ThinginoBackend):
            def __init__(self):
                super().__init__([])
                self.events = []
                self.original = FakeBackend().original

            def _run(self, args):
                self.events.append(args[0])
                return b""

            def _wait_policy(self, expected=None, timeout=15.):
                self.events.append("policy_ack")

            def snapshot_state(self):
                self.events.append("stable_readback")
                return deepcopy(self.original)
        backend = Backend()
        with patch("sim.world.ir_exp.time.sleep"):
            backend.restore(backend.original)
        self.assertEqual(backend.events, ["jct", "jct", "sh", "policy_ack", "prudyntctl", "ircut", "light",
                                          "stable_readback", "stable_readback"])

    def test_policy_freshness_uses_camera_clock_even_when_far_ahead_or_behind_server(self):
        class Backend(ThinginoBackend):
            def __init__(self, baseline):
                super().__init__([])
                self.samples = iter([baseline, baseline, baseline + 1, baseline + 1, baseline + 2])
                self.reads = 0

            def _guard(self):
                pass

            def _text(self, args):
                self.reads += 1
                return json.dumps({"time_now": next(self.samples), "mode": "night"})
        for baseline in (100, 4_000_000_000):
            with self.subTest(camera_baseline=baseline):
                backend = Backend(baseline)
                with patch("sim.world.ir_exp.time.time", side_effect=AssertionError("Must not compare cross-device wall clocks")), \
                        patch("sim.world.ir_exp.time.monotonic", return_value=1), \
                        patch("sim.world.ir_exp.time.sleep"):
                    backend._wait_policy("night")
                # Neither the initial matching sample nor duplicate timestamps
                # acknowledge a new daemon loop.
                self.assertEqual(backend.reads, 5)

    def test_stale_matching_policy_sample_times_out_instead_of_acknowledging(self):
        class Backend(ThinginoBackend):
            def __init__(self):
                super().__init__([])
                self.reads = 0

            def _guard(self):
                pass

            def _text(self, args):
                self.reads += 1
                return json.dumps({"time_now": 4_000_000_000, "mode": "night"})
        backend = Backend()
        with patch("sim.world.ir_exp.time.time", side_effect=AssertionError("No wall clock comparison")), \
                patch("sim.world.ir_exp.time.monotonic", side_effect=[0, 1, 2, 3, 16]), \
                patch("sim.world.ir_exp.time.sleep"):
            with self.assertRaisesRegex(CaptureError, "acknowledge"):
                backend._wait_policy("night")
        self.assertEqual(backend.reads, 3)


if __name__ == "__main__":
    unittest.main()
