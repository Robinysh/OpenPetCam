"""Evidence for reproducible, fair and independently readable probe archives."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.acoustics.probes import ProbeConfig, build_probe_suite, load_probe_packet


class ProbeArchiveTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def test_exact_archived_pcm_schedule_and_hashes(self):
        suite = build_probe_suite(self.root / "suite")
        self.assertEqual(len(suite["packets"]), 4)
        for entry in suite["packets"]:
            path = self.root / "suite" / entry["manifest"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), entry["manifest_sha256"])
            packet, probe, pilot, manifest = load_probe_packet(path)
            # Independently decode the WAV and verify the digital excitation.
            with wave.open(str(path.parent / "packet.wav"), "rb") as src:
                self.assertEqual((src.getnchannels(), src.getsampwidth(), src.getframerate()), (1, 2, 48000))
                pcm = np.frombuffer(src.readframes(src.getnframes()), dtype="<i2")
            np.testing.assert_array_equal(packet, pcm / 32768.)
            events = manifest["events"]
            self.assertEqual([e["type"] for e in events], ["pilot", "shot", "shot", "shot", "shot", "pilot"])
            for event in events:
                expected = probe if event["type"] == "shot" else pilot
                np.testing.assert_array_equal(packet[event["sample_start"]:event["sample_end"]], expected)
            self.assertTrue(np.all(packet[:events[0]["sample_start"]] == 0))
            self.assertGreaterEqual(len(packet) - events[-1]["sample_end"], 48000)
            self.assertIn("not_physical_emission_t0", manifest["timing_status"])

    def test_comparison_pairs_match_energy_rms_duration_and_peak_budget(self):
        config = ProbeConfig()
        suite = build_probe_suite(self.root / "suite", config)
        groups = {}
        for entry in suite["packets"]:
            _, probe, _, manifest = load_probe_packet(self.root / "suite" / entry["manifest"])
            self.assertLessEqual(abs(probe).max(), config.peak_ceiling)
            self.assertLessEqual(abs(np.sqrt(np.mean(probe ** 2)) - config.target_rms), 1 / 32768)
            groups.setdefault(entry["comparison_group"], []).append(probe)
            self.assertEqual(manifest["probe"]["band_hz"], [2000., 7600.])
        for linear, exponential in groups.values():
            self.assertEqual(len(linear), len(exponential))
            self.assertFalse(np.array_equal(linear, exponential))
            relative_energy_difference = abs(np.dot(linear, linear) / np.dot(exponential, exponential) - 1)
            self.assertLess(relative_energy_difference, .00002)

    def test_deterministic_bytes_and_ids(self):
        a = build_probe_suite(self.root / "a")
        b = build_probe_suite(self.root / "b")
        self.assertEqual(a, b)
        for path in (self.root / "a").rglob("*"):
            if path.is_file():
                self.assertEqual(path.read_bytes(), (self.root / "b" / path.relative_to(self.root / "a")).read_bytes())
        with self.assertRaises(FileExistsError):
            build_probe_suite(self.root / "a")

    def test_archived_waveforms_have_distinct_intended_frequency_progressions(self):
        suite = build_probe_suite(self.root / "suite", ProbeConfig(durations_s=(.5,)))
        measured = {}
        for entry in suite["packets"]:
            _, probe, pilot, _ = load_probe_packet(self.root / "suite" / entry["manifest"])
            # Count complete positive-going cycles in independent interior windows.
            frequencies = []
            for centre in (.1, .4):
                segment = probe[round((centre - .02) * 48000):round((centre + .02) * 48000)]
                crossings = np.flatnonzero((segment[:-1] <= 0) & (segment[1:] > 0))
                frequencies.append((len(crossings) - 1) * 48000 / (crossings[-1] - crossings[0]))
            measured[entry["family"]] = frequencies
            self.assertGreater(len(pilot), 0)
        np.testing.assert_allclose(measured["linear"], [3120, 6480], atol=35, rtol=0)
        np.testing.assert_allclose(measured["exponential"], [2612, 5819], atol=35, rtol=0)

    def test_hash_and_schedule_tampering_fail_closed(self):
        suite = build_probe_suite(self.root / "suite", ProbeConfig(durations_s=(.5,)))
        path = self.root / "suite" / suite["packets"][0]["manifest"]
        info = json.loads(path.read_text())
        info["events"][1]["sample_start"] += 1
        path.write_text(json.dumps(info))
        with self.assertRaisesRegex(ValueError, "template"):
            load_probe_packet(path)
        info["events"][1]["sample_start"] -= 1
        path.write_text(json.dumps(info))
        wav = path.parent / "probe.wav"
        wav.write_bytes(wav.read_bytes()[:-2] + b"\x01\x02")
        with self.assertRaisesRegex(ValueError, "hash"):
            load_probe_packet(path)

    def test_invalid_or_infeasible_configuration_creates_nothing(self):
        for config in (ProbeConfig(target_rms=.499), ProbeConfig(post_s=.2),
                       ProbeConfig(band_hz=(2000., 24000.)), ProbeConfig(repetitions=1)):
            with self.subTest(config=config):
                output = self.root / "bad"
                with self.assertRaises(ValueError):
                    build_probe_suite(output, config)
                self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
