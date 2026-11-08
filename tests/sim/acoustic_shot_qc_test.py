"""Acquisition QC with independently constructed PCM and shot schedules."""
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.acoustics.shot_qc import (QCConfig, ShotWindow, assess_capture,
                                  fit_packet_timing, mapped_window, repeat_coherence)


class ShotQCTest(unittest.TestCase):
    def setUp(self):
        self.fs = 16000
        t = np.arange(640) / self.fs
        # Explicit linear-phase law, independent of the detection implementation.
        self.probe = .6 * np.sin(2 * np.pi * (2100 * t + 4000 / (.08) * t ** 2)) * np.hanning(len(t))
        self.starts = [1600, 8000, 14400]
        self.source = [0, 6400, 12800]
        self.windows = [ShotWindow(f"shot_{i}", src, start - 80, start + 81, 0, 1000)
                        for i, (src, start) in enumerate(zip(self.source, self.starts))]
        self.x = np.random.default_rng(17).normal(0, .0001, 18000)
        for start in self.starts:
            self.x[start:start + len(self.probe)] += self.probe

    def assess(self, x=None, **kwargs):
        return assess_capture(self.x if x is None else x, sample_rate_hz=self.fs,
                              source_rate_hz=self.fs, probe=self.probe,
                              shots=self.windows, response_tail_s=.02, **kwargs)

    def test_sample_windows_preserve_amplitude_and_are_json_safe(self):
        a = self.assess()
        b = self.assess(self.x * .5)
        self.assertTrue(a["accepted"])
        self.assertFalse(self.assess(config=QCConfig(min_snr_db=100))["accepted"])
        self.assertEqual([q["detected_start_sample"] for q in a["shots"]], self.starts)
        self.assertEqual(a["timing_status"], "unreferenced")
        self.assertFalse(a["absolute_timing_calibrated"])
        self.assertIsNone(a["repeat_coherence"])
        for r, s in zip(a["shots"], b["shots"]):
            self.assertAlmostEqual(s["signal_rms"], r["signal_rms"] / 2, places=10)
            self.assertAlmostEqual(s["matched_filter_amplitude"], r["matched_filter_amplitude"] / 2, places=10)
            self.assertAlmostEqual(s["normalized_correlation"], r["normalized_correlation"], places=10)
            self.assertEqual(r["raw_segment_stop_sample"] - r["raw_segment_start_sample"], 960)
        json.dumps(a, allow_nan=False)

    def test_independent_pilot_map_checks_shots_without_individual_alignment(self):
        pilots = [{"event_id": "pilot_a", "source_start_sample": -1000,
                   "detected_start_sample_float": 600., "accepted": True},
                  {"event_id": "pilot_b", "source_start_sample": 15000,
                   "detected_start_sample_float": 16600., "accepted": True}]
        timing = fit_packet_timing(pilots, source_rate_hz=self.fs, sample_rate_hz=self.fs)
        a = self.assess(timing_map=timing)
        self.assertTrue(a["accepted"])
        self.assertEqual(timing["residual_degrees_of_freedom"], 0)
        self.assertFalse(timing["absolute_timing_calibrated"])
        self.assertEqual(timing["timing_status"], "acoustic_anchor_relative")
        self.assertTrue(all(q["timing_evaluation_independent_of_anchor_fit"] for q in a["shots"]))
        self.assertGreater(a["repeat_coherence"]["bands"][0]["coherent_energy_fraction"], .999)
        x = self.x.copy()
        start = self.starts[1]
        x[start:start + len(self.probe)] -= self.probe
        x[start + 8:start + 8 + len(self.probe)] += self.probe
        b = self.assess(x, timing_map=timing)
        self.assertFalse(b["accepted"])
        self.assertIn("timing_inconsistent_with_independent_anchors", b["shots"][1]["rejection_reasons"])
        self.assertAlmostEqual(b["shots"][1]["timing_residual_samples"], 8, delta=.01)
        self.assertEqual(b["shots"][1]["common_map_segment_samples"][0], start)

    def test_missing_clipped_noisy_and_transient_shots_fail(self):
        missing = self.x.copy()
        missing[self.starts[1]:self.starts[1] + len(self.probe)] -= self.probe
        self.assertFalse(self.assess(missing)["shots"][1]["accepted"])
        pcm = np.round(self.x * 32768).astype("<i2")
        pcm[self.starts[0] + 50] = 32767
        clip = self.assess(pcm.tobytes())
        self.assertFalse(clip["accepted"])
        self.assertIn("clipping", clip["shots"][0]["rejection_reasons"])
        self.assertEqual(clip["clipped_samples"], 1)
        noise = self.x.copy()
        noise[:1000] = np.random.default_rng(18).normal(0, .2, 1000)
        self.assertIn("low_snr", self.assess(noise)["shots"][0]["rejection_reasons"])
        transient = self.x.copy()
        transient[100:120] = .1
        self.assertIn("noise_transient", self.assess(transient)["shots"][0]["rejection_reasons"])

    def test_malformed_truncated_or_overlapping_windows_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "odd byte"):
            self.assess(b"\x01")
        with self.assertRaisesRegex(ValueError, "truncated"):
            self.assess(self.x[:14500])
        bad = [ShotWindow("overlap", 0, 1550, 1650, 1600, 1700)]
        with self.assertRaisesRegex(ValueError, "quiet window overlaps"):
            assess_capture(self.x, sample_rate_hz=self.fs, source_rate_hz=self.fs,
                           probe=self.probe, shots=bad)
        with self.assertRaisesRegex(ValueError, "unique"):
            assess_capture(self.x, sample_rate_hz=self.fs, source_rate_hz=self.fs,
                           probe=self.probe, shots=[self.windows[0], self.windows[0]])
        with self.assertRaisesRegex(ValueError, "floating PCM"):
            self.assess(self.x * 10)

    def test_clock_fit_reports_relative_rate_and_rejects_bad_anchors(self):
        ds = [{"event_id": f"pilot_{i}", "source_start_sample": p,
               "detected_start_sample_float": 123.4 + 3.0006 * p, "accepted": True}
              for i, p in enumerate([0, 8000, 24000])]
        fit = fit_packet_timing(ds, source_rate_hz=16000, sample_rate_hz=48000)
        self.assertAlmostEqual(fit["relative_rate_error_ppm"], 200, places=6)
        self.assertAlmostEqual(fit["intercept"], 123.4, places=8)
        self.assertFalse(fit["absolute_timing_calibrated"])
        ds[1]["accepted"] = False
        with self.assertRaisesRegex(ValueError, "accepted"):
            fit_packet_timing(ds, source_rate_hz=16000, sample_rate_hz=48000)

    def test_repeat_coherence_does_not_hide_polarity_or_amplitude(self):
        traces = np.stack((self.probe, self.probe))
        a = repeat_coherence(traces, sample_rate_hz=self.fs, bands_hz=((2000, 7400),))
        b = repeat_coherence(np.stack((self.probe, -self.probe)), sample_rate_hz=self.fs,
                             bands_hz=((2000, 7400),))
        c = repeat_coherence(np.stack((self.probe, .5 * self.probe)), sample_rate_hz=self.fs,
                             bands_hz=((2000, 7400),))
        self.assertAlmostEqual(a["bands"][0]["coherent_energy_fraction"], 1)
        self.assertAlmostEqual(b["bands"][0]["coherent_energy_fraction"], 0)
        self.assertAlmostEqual(c["rms_per_trace"][1] / c["rms_per_trace"][0], .5)
        self.assertAlmostEqual(c["bands"][0]["coherent_energy_fraction"], .9)

    def test_fractional_common_map_preserves_repeat_phase_without_measured_lags(self):
        starts = [1000.2, 5000.8]
        t = np.arange(8000)

        def wave(v):
            return .4 * np.exp(-.5 * ((v - 250) / 50) ** 2) * np.cos(2 * np.pi * 4500 / self.fs * v)

        x = sum(wave(t - start) for start in starts)
        traces = np.array([mapped_window(x, start, 600) for start in starts])
        good = repeat_coherence(traces, sample_rate_hz=self.fs, bands_hz=((4000, 5000),))
        self.assertGreater(good["bands"][0]["coherent_energy_fraction"], .999999)
        # An actual propagation change must remain visible: supplied origins
        # stay frozen while only the second waveform is delayed.
        changed = wave(t - starts[0]) + wave(t - starts[1] - 2.4)
        bad = repeat_coherence(np.array([mapped_window(changed, start, 600) for start in starts]),
                               sample_rate_hz=self.fs, bands_hz=((4000, 5000),))
        self.assertLess(bad["bands"][0]["coherent_energy_fraction"], .4)


if __name__ == "__main__":
    unittest.main()
