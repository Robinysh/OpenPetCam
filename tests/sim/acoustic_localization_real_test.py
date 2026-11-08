"""Behavioral controls for read-only unvalidated original-audio diagnostics."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.acoustics.localization import ripple_features, spectrum
from sim.acoustics.localization_real import eligible_windows, grouping, native_window_samples, run, sha


class RealDiagnosticsTests(unittest.TestCase):
    def test_native_duration_matches_frequency_spacing_without_resampling(self):
        self.assertEqual(native_window_samples(48000, 16000, 8192), 24576)
        self.assertEqual(native_window_samples(16000, 16000, 8192), 8192)
        with self.assertRaisesRegex(ValueError, "without resampling"):
            native_window_samples(44100, 16000, 8192)
        with self.assertRaisesRegex(ValueError, "at least .512"):
            native_window_samples(48000, 16000, 4096)

    def test_no_window_crosses_gap_or_pads_short_segment(self):
        row = {"decoded_samples": 104000, "sample_rate_hz": 48000,
               "segments_timing_eligible": True,
               "segments": [{"start_sample": 0, "stop_sample": 30000, "container_start_s": 1.},
                            {"start_sample": 30000, "stop_sample": 54000, "container_start_s": 2.},
                            {"start_sample": 54000, "stop_sample": 104000, "container_start_s": 5.}]}
        original = copy.deepcopy(row)
        windows = eligible_windows(row, 24576)
        self.assertEqual([(w["start_sample"], w["stop_sample"]) for w in windows],
                         [(0, 24576), (54000, 78576), (78576, 103152)])
        self.assertEqual(row, original)
        self.assertEqual(windows[1]["container_start_s"], 5.)
        self.assertAlmostEqual(windows[2]["container_start_s"], 5.512)
        row["segments_timing_eligible"] = False
        self.assertEqual(eligible_windows(row, 24576), [])

    def test_corrupt_overlapping_segment_declarations_are_rejected(self):
        row = {"decoded_samples": 20000, "sample_rate_hz": 16000,
               "segments_timing_eligible": True,
               "segments": [{"start_sample": 0, "stop_sample": 10000, "container_start_s": 0.},
                            {"start_sample": 9000, "stop_sample": 20000, "container_start_s": 1.}]}
        with self.assertRaisesRegex(ValueError, "overlapping"):
            eligible_windows(row, 8192)

    def test_grouping_keeps_conservative_day_and_overlap_keys(self):
        row = {"source_sha256": "source", "overlap_group": "overlap",
               "recommended_split_group": "whole-day", "split_eligible": False,
               "record_id": "ignored"}
        self.assertEqual(grouping(row), {k: v for k, v in row.items() if k != "record_id"})
        del row["overlap_group"]
        with self.assertRaisesRegex(ValueError, "leakage-group"):
            grouping(row)

    def test_diagnostics_preserve_signed_native_bytes_and_never_certify_location(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            corpus = root / "corpus"; corpus.mkdir()
            benchmark = root / "benchmark"; benchmark.mkdir()
            count = 24576
            # Deliberate signed samples and peaks above unity remain untouched.
            samples = (np.random.default_rng(71).normal(size=count) * .6).astype("<f4")
            native = corpus / "original.f32le"; native.write_bytes(samples.tobytes())
            before = native.read_bytes()
            config = {"name": "control", "smooth_hz": 300., "denoise_hz": 12., "band": [300., 7000.]}
            f, p = spectrum(samples, 48000, nfft=count)
            keep = f <= 8000
            feature, _ = ripple_features(p[keep], f[keep], **{k: v for k, v in config.items() if k != "name"})
            np.savez(benchmark / "dictionary.npz", positions_m=[[1., 0., 2.], [-1., 0., 2.]],
                     control=np.stack([feature, -feature]))
            (benchmark / "protocol.json").write_text(json.dumps({"sample_rate": 16000, "nfft": 8192,
                                                                  "receiver_m": [0., 0., 0.]}))
            (benchmark / "selected.json").write_text(json.dumps({"config": config,
                "acceptance_rule": {"min_margin": 0., "min_similarity": .5}}))
            row = {"record_id": "one", "source_sha256": "source", "source_kind": "observed_room_recording",
                   "format": "f32le", "channels": 1, "sample_rate_hz": 48000,
                   "decoded_samples": count, "file": native.name, "sha256": sha(native),
                   "segments_timing_eligible": True, "segments": [{"start_sample": 0,
                       "stop_sample": count, "container_start_s": .125}],
                   "overlap_group": "overlap", "recommended_split_group": "whole-day", "split_eligible": True}
            manifest = corpus / "manifest.json"
            manifest.write_text(json.dumps({"complete": True, "records": [row]}))
            result = run(manifest, benchmark, root / "diagnostic")
            self.assertEqual(result["windows"], 1)
            self.assertEqual(result["accepted_real_locations"], 0)
            self.assertFalse(result["real_accuracy_measured"])
            ranking = json.loads((root / "diagnostic" / "rankings.jsonl").read_text())
            self.assertTrue(ranking["passes_synthetic_rule"])
            self.assertFalse(ranking["accepted"])
            self.assertIsNone(ranking["source_position_m"])
            self.assertIsNone(ranking["microphone_pose"])
            self.assertEqual(ranking["native_sample_rate_hz"], 48000)
            self.assertEqual(ranking["native_window_samples"], count)
            self.assertEqual(ranking["window_duration_s"], .512)
            self.assertEqual(ranking["resampling"], {"applied": False, "native_rate_hz": 48000})
            self.assertGreater(ranking["native_peak_absolute"], 1.)
            self.assertEqual(ranking["recommended_split_group"], "whole-day")
            self.assertNotIn("distance_error_m", ranking)
            self.assertEqual(native.read_bytes(), before)
            with self.assertRaises(FileExistsError):
                run(manifest, benchmark, root / "diagnostic")


if __name__ == "__main__":
    unittest.main()
