"""Native mock SDK integration; this never certifies the camera ABI/reference.

Runs the actual interceptor with a fake dynamically linked SDK and concurrent
AI/AO callers. Requires Linux's actual LD_PRELOAD semantics. A source-hashed
prebuilt fixture bundle may be supplied on hosts without a compiler.
No network access, hardware, service restart or deployment is performed.
"""
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sim.acoustics.device.build import build
from sim.acoustics.device.decode import export_capture, records


@unittest.skipUnless(platform.system() == "Linux" and (shutil.which("cc") or os.environ.get("OPC_NATIVE_FIXTURES")), "Linux LD_PRELOAD plus compiler or source-hashed fixtures required")
class DeviceNativeTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if os.environ.get("OPC_NATIVE_FIXTURES"):
            cls.temp = None
            cls.root = Path(os.environ["OPC_NATIVE_FIXTURES"]).resolve()
            provenance = json.loads((cls.root / "fixture-manifest.json").read_text())
            for relative, digest in provenance["source_sha256"].items():
                if hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() != digest:
                    raise AssertionError("prebuilt mock source differs from current source: " + relative)
            for name, digest in provenance["artifacts_sha256"].items():
                if hashlib.sha256((cls.root / name).read_bytes()).hexdigest() != digest:
                    raise AssertionError("prebuilt mock artifact hash mismatch: " + name)
            return
        # macOS's default per-user temporary path can exceed sockaddr_un limits.
        cls.temp = tempfile.TemporaryDirectory(prefix="opc-native-", dir="/tmp")
        cls.root = Path(cls.temp.name).resolve()
        build(cls.root, "native")
        fixture = ROOT / "tests/sim/fixtures/acoustic_device"
        include = ROOT / "sim/acoustics/device"
        cc = shutil.which("cc")
        flags = [cc, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-I", str(include)]
        shared = ["-shared", "-Wl,-soname,libopc_mock.so"]
        subprocess.run(flags + ["-fPIC", str(fixture / "mock_imp.c"), *shared, "-o", str(cls.root / "libopc_mock.so")], check=True, capture_output=True)
        subprocess.run(flags + [str(fixture / "driver.c"), "-L", str(cls.root), "-lopc_mock", "-pthread",
                               "-Wl,-rpath," + str(cls.root), "-o", str(cls.root / "driver")], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        if cls.temp is not None:
            cls.temp.cleanup()

    def environment(self, mode="valid", enabled=True, fast=False):
        env = os.environ.copy()
        for name in ("OPC_AUDIO_SOCKET", "OPC_AUDIO_REFERENCE", "OPC_MOCK_FAST"):
            env.pop(name, None)
        if platform.system() == "Darwin":
            env["DYLD_INSERT_LIBRARIES"] = str(self.root / "opc_audio_tap.so")
            env["DYLD_FORCE_FLAT_NAMESPACE"] = "1"
        else:
            env["LD_PRELOAD"] = str(self.root / "opc_audio_tap.so")
        if enabled:
            env["OPC_AUDIO_SOCKET"] = str(self.root / (self._testMethodName[-25:] + ".sock"))
            env["OPC_AUDIO_REFERENCE"] = "1"
        env["OPC_MOCK_MODE"] = mode
        if fast:
            env["OPC_MOCK_FAST"] = "1"
        return env

    def start(self, env):
        process = subprocess.Popen([str(self.root / "driver"), "3"], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        path = Path(env["OPC_AUDIO_SOCKET"])
        limit = time.monotonic() + 1
        while not path.exists():
            if process.poll() is not None or time.monotonic() > limit:
                process.terminate()
                stdout, stderr = process.communicate(timeout=2)
                self.fail(f"mock interceptor did not create endpoint: {stdout!r} {stderr!r}")
            time.sleep(.005)
        return process

    def stop(self, process):
        if process.poll() is None:
            process.terminate()
        stdout, stderr = process.communicate(timeout=3)
        self.assertEqual(process.returncode, 0, stderr.decode())
        return json.loads(stdout)

    def capture(self, mode="valid"):
        env = self.environment(mode)
        driver = self.start(env)
        try:
            result = subprocess.run([str(self.root / "opc-audio-stream"), env["OPC_AUDIO_SOCKET"], ".25"], capture_output=True, timeout=4)
        finally:
            stats = self.stop(driver)
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        stream = self.root / (self._testMethodName + ".frames")
        stream.write_bytes(result.stdout)
        return stream, stats

    def test_disabled_interceptor_preserves_normal_calls(self):
        result = subprocess.run([str(self.root / "driver"), ".1"], env=self.environment(enabled=False), capture_output=True, timeout=3)
        self.assertEqual(result.returncode, 0, result.stderr.decode())
        stats = json.loads(result.stdout)
        self.assertGreater(stats["normal_gets"], 0)
        self.assertGreater(stats["sends"], 0)
        self.assertEqual(stats["paired_gets"], 0)
        self.assertEqual(stats["enables"], 0)
        self.assertEqual(stats["get_failures"], 0)

    def test_paired_frames_preserve_payloads_sequence_and_timestamp_grid(self):
        stream, stats = self.capture()
        metadata = export_capture(stream, self.root / self._testMethodName)
        self.assertGreater(stats["paired_gets"], 4)
        self.assertTrue(metadata["mic_continuity_verified"])
        self.assertTrue(metadata["reference_pair_contract_verified"])
        # This native fixture deliberately supplies unrelated constant AO and
        # reference PCM. Matching metadata alone must not certify the grid.
        self.assertFalse(metadata["reference_sample_grid_verified"])
        self.assertFalse(metadata["reference_continuity_verified"])
        self.assertFalse(metadata["physical_DAC_timing_verified"])
        raw_mic = Path(metadata["mic_file"]).read_bytes()
        self.assertEqual(len(raw_mic), len(Path(metadata["reference_file"]).read_bytes()))
        self.assertEqual(hashlib.sha256(raw_mic).hexdigest(), metadata["mic_sha256"])
        self.assertGreater(len(metadata["ao_submissions"]), 0)
        rows = records(stream.read_bytes())
        self.assertEqual([r[0]["record_index"] for r in rows], list(range(len(rows))))

    def test_inactive_reference_does_not_gain_paired_certification(self):
        stream, stats = self.capture("reference_unavailable")
        metadata = export_capture(stream, self.root / self._testMethodName)
        self.assertEqual(stats["paired_gets"], 0)
        self.assertFalse(metadata["reference_pair_contract_verified"])
        self.assertFalse(metadata["reference_sample_grid_verified"])
        self.assertFalse(metadata["reference_continuity_verified"])
        self.assertIsNone(metadata["reference_file"])

    def test_malformed_reference_pair_is_rejected(self):
        stream, _ = self.capture("malformed_reference")
        with self.assertRaisesRegex(ValueError, "grid"):
            export_capture(stream, self.root / self._testMethodName)

    def test_sdk_sequence_gap_is_rejected(self):
        stream, _ = self.capture("sequence_gap")
        with self.assertRaisesRegex(ValueError, "sequence discontinuity"):
            export_capture(stream, self.root / self._testMethodName)

    def test_backpressure_disconnects_instead_of_splicing_records(self):
        env = self.environment(fast=True)
        driver = self.start(env)
        connection = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        connection.settimeout(2)
        try:
            connection.connect(env["OPC_AUDIO_SOCKET"])
            time.sleep(.2)  # Deliberately stop draining the producer socket.
            pieces = []
            while True:
                part = connection.recv(65536)
                if not part:
                    break
                pieces.append(part)
            with self.assertRaisesRegex(ValueError, "terminal end|truncated"):
                records(b"".join(pieces))
        finally:
            connection.close()
            self.stop(driver)

    def test_reader_rejects_nonfinite_duration(self):
        result = subprocess.run([str(self.root / "opc-audio-stream"), str(self.root / "absent.sock"), "nan"], capture_output=True, timeout=2)
        self.assertEqual(result.returncode, 2)

    def test_reader_stdout_backpressure_still_has_bounded_exit(self):
        env = self.environment(fast=True)
        driver = self.start(env)
        reader = None
        try:
            reader = subprocess.Popen([str(self.root / "opc-audio-stream"), env["OPC_AUDIO_SOCKET"], ".1"],
                                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            # Do not drain stdout: a blocking write must not defeat the deadline.
            code = reader.wait(timeout=4.5)
            self.assertNotEqual(code, 0)
            self.assertIn(code, (7, 11, 12))
        finally:
            if reader is not None:
                if reader.poll() is None:
                    reader.kill()
                reader.communicate(timeout=1)
            self.stop(driver)


if __name__ == "__main__":
    unittest.main()
