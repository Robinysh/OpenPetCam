"""Hardware adapter failure behavior without a camera or emitted sound."""
import io
import hashlib
import json
import shlex
import wave
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.acoustics.capture import Camera  # noqa: E402


class CompletedReader:
    def __init__(self, args, stdout, stderr):
        stdout.write(bytes(8 * 96000))
        stdout.flush()
        self.returncode = 0
        self.stderr = io.BytesIO()

    def poll(self):
        return self.returncode

    def communicate(self, timeout):
        return None, b""


class AdapterTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.wav = self.root / "probe.wav"
        with wave.open(str(self.wav), "wb") as wav:
            wav.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            wav.writeframes(bytes(2 * 16000 * 2))
        self.camera = Camera.__new__(Camera)
        self.camera.allow_rate_switching = True
        self.camera.state = self.root
        self.camera.ssh = ["fake-camera"]
        self.camera.reader = "/tmp/opc-audio-stream"
        self.camera.socket = "/tmp/opc-audio.sock"
        self.camera.legacy_fifo = True
        self.commands = []

        def command(cmd, data=None, timeout=30):
            self.commands.append((cmd, data))
            return b""

        self.camera.command = command

    def test_privacy_enabled_after_upload_prevents_playback_and_stops_queue(self):
        calls = []

        def privacy():
            calls.append(True)
            if len(calls) == 2:
                raise RuntimeError("privacy became active")

        self.camera.check_privacy = privacy
        with patch("sim.acoustics.capture.subprocess.Popen", CompletedReader), patch("sim.acoustics.capture.time.sleep"):
            with self.assertRaisesRegex(RuntimeError, "privacy"):
                self.camera.record(self.wav, self.root / "out.pcm", 30, 2)
        self.assertFalse(any(cmd.startswith("play -") for cmd, _ in self.commands))
        self.assertTrue(any(cmd == "play stop" for cmd, _ in self.commands))
        # Only the owned unique uploaded path is removed; no wildcard cleanup.
        deletes = [cmd for cmd, _ in self.commands if cmd.startswith("rm")]
        self.assertEqual(len(deletes), 1)
        self.assertNotIn("*", deletes[0])

    def test_legacy_capture_can_never_claim_continuity_or_emission_time(self):
        self.camera.check_privacy = lambda: None
        with patch("sim.acoustics.capture.subprocess.Popen", CompletedReader), patch("sim.acoustics.capture.time.sleep"):
            result = self.camera.record(self.wav, self.root / "out.pcm", 30, 2)
        self.assertFalse(result["sample_continuity_verified"])
        self.assertFalse(result["hardware_timing_verified"])
        self.assertEqual(result["samples"], 8 * 48000)

    def test_framed_reader_rejects_missing_paired_reference_without_fifo_fallback(self):
        self.camera.check_privacy = lambda: None
        self.camera.legacy_fifo = False
        observed_commands = []

        def reader(args, **kwargs):
            observed_commands.extend(args)
            return CompletedReader(args, **kwargs)

        bad = {"reference_sample_grid_verified": False, "reference_continuity_verified": False}
        with patch("sim.acoustics.capture.subprocess.Popen", reader), patch("sim.acoustics.capture.time.sleep"), \
                patch("sim.acoustics.device.decode.export_capture", return_value=bad):
            with self.assertRaisesRegex(RuntimeError, "reference"):
                self.camera.record(self.wav, self.root / "out.pcm", 30, 2)
        self.assertFalse((self.root / "out.pcm").exists())
        self.assertTrue((self.root / "out.opcaud").exists())
        self.assertFalse(any("audio_mic.pcm" in cmd for cmd in observed_commands))
        self.assertTrue(any(cmd == "play stop" for cmd, _ in self.commands))


class FakeClock:
    def __init__(self):
        self.now = 0.
    def monotonic(self):
        return self.now
    def time(self):
        return 1794000000. + self.now
    def sleep(self, seconds):
        self.now += seconds


class CaptureLivenessTest(unittest.TestCase):
    """Exercise the real Camera.record control flow with synthetic transport.

    Framed decoding is stubbed explicitly: these checks establish scheduling and
    cleanup only. Real framing/reference integrity is covered by decoder tests.
    """
    def fixture(self, root, readiness_s, source_rate=48000, queue_s=12.):
        clock = FakeClock()
        source = root / "source.wav"
        duration = 14.66
        with wave.open(str(source), "wb") as wav:
            wav.setparams((1, 2, source_rate, 0, "NONE", "not compressed"))
            wav.writeframes(bytes(round(duration * source_rate) * 2))
        camera = Camera.__new__(Camera)
        camera.state = root
        camera.ssh = ["fake-camera"]
        camera.reader, camera.socket = "/tmp/owned-reader", "/tmp/owned-socket"
        camera.legacy_fifo = False
        camera.allow_rate_switching = source_rate == 16000
        camera.check_privacy = lambda: None
        state = {"commands": [], "readers": [], "playbacks": []}

        def command(cmd, data=None, timeout=30):
            state["commands"].append((cmd, timeout))
            if cmd == "cat /proc/jz/audio/audio_aic_info":
                return b"The living rate of record : 48000\nThe living format of record : 16\nThe living rate of replay : 48000\nThe living format of replay : 16\n"
            if cmd.startswith("play -"):
                began = clock.now
                clock.sleep(queue_s)
                state["playbacks"].append({"queued_before": began, "emission_start": clock.now,
                    "source_end": clock.now + duration + .040})
            return b""
        camera.command = command

        class Reader:
            def __init__(self, args, stdout, stderr):
                self.stdout = stdout
                self.started = clock.now
                self.end = self.started + float(shlex.split(args[-1])[-1])
                self.returncode = None
                self.terminated = False
                self.initial_written = False
                state["readers"].append(self)
            def poll(self):
                if self.returncode is None and clock.now - self.started >= readiness_s and not self.initial_written:
                    self.stdout.write(bytes(9600)); self.stdout.flush(); self.initial_written = True
                if self.returncode is None and clock.now >= self.end:
                    self.returncode = 0
                return self.returncode
            def communicate(self, timeout):
                return None, b""
            def terminate(self):
                self.terminated = True
                self.returncode = -15
            def wait(self, timeout):
                return self.returncode
            def kill(self):
                self.returncode = -9

        def decoded(path, output_dir, prefix):
            # A truncated source must not be hidden by an invented long PCM file.
            self.assertTrue(state["playbacks"])
            self.assertLess(state["playbacks"][-1]["source_end"], state["readers"][-1].end,
                            "Reader ended before complete source plus exact40ms SDK tail")
            mic, ref = root / "decoded-mic.pcm", root / "decoded-reference.pcm"
            mic.write_bytes(bytes(round((duration + 1) * 96000)))
            ref.write_bytes(mic.read_bytes())
            device = {"mic_file": str(mic), "reference_file": str(ref),
                "reference_sha256": hashlib.sha256(ref.read_bytes()).hexdigest(),
                "reference_sample_grid_verified": True, "reference_continuity_verified": True,
                "mic_continuity_verified": True, "active_ao_reference_check": {"verified": True},
                "common_mic_sample_interval": [0, mic.stat().st_size // 2], "complete": True,
                "source_rate_hz": source_rate, "frames": [{"reference": {"ao_input_rate_hz": 48000}}],
                "ao_submissions": [{"ao": {"rate_hz": source_rate}}]}
            (root / (prefix + ".json")).write_text(json.dumps(device))
            return device
        return camera, source, duration, clock, state, Reader, decoded

    def test_delayed_data_and_slow_play_keep_complete_source_inside_reader(self):
        for readiness in (12., 19.5):
            for rate in (16000, 48000):
                with self.subTest(readiness=readiness, rate=rate), tempfile.TemporaryDirectory() as tmp:
                    root = Path(tmp)
                    camera, source, duration, clock, state, reader, decoded = self.fixture(root, readiness, rate)
                    with patch("sim.acoustics.capture.time", clock), patch("sim.acoustics.capture.subprocess.Popen", reader), \
                            patch("sim.acoustics.device.decode.export_capture", decoded):
                        try:
                            result = camera.record(source, root / "capture.pcm", 40, duration)
                        except RuntimeError as exc:
                            self.fail("Valid delayed initial data rejected: " + str(exc))
                    self.assertEqual(len(state["readers"]), 1)
                    self.assertEqual(len(state["playbacks"]), 1)
                    self.assertGreaterEqual(state["playbacks"][0]["queued_before"], readiness + .4 - 1e-8)
                    self.assertFalse(result["hardware_timing_verified"])
                    self.assertEqual([timeout for cmd, timeout in state["commands"] if cmd.startswith("play -")], [12])
                    self.assertEqual(len([cmd for cmd, _ in state["commands"] if cmd.startswith("rm -f ")]), 1)

    def test_readiness_later_than_twenty_seconds_queues_no_audio_and_cleans_up(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            camera, source, duration, clock, state, reader, decoded = self.fixture(root, 20.5)
            with patch("sim.acoustics.capture.time", clock), patch("sim.acoustics.capture.subprocess.Popen", reader), \
                    patch("sim.acoustics.device.decode.export_capture", decoded):
                with self.assertRaisesRegex(RuntimeError, "initial PCM"):
                    camera.record(source, root / "capture.pcm", 40, duration)
            self.assertFalse(state["playbacks"])
            self.assertTrue(state["readers"][0].terminated)
            self.assertIn("play stop", [cmd for cmd, _ in state["commands"]])
            self.assertEqual(len([cmd for cmd, _ in state["commands"] if cmd.startswith("rm -f ")]), 1)
            self.assertTrue((root / "capture.opcaud").exists())
            self.assertFalse((root / "capture.pcm").exists())


if __name__ == "__main__":
    unittest.main()
