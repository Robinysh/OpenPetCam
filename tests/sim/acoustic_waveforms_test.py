"""Synthetic evidence for signed, training-referenced acoustic extraction (no hardware)."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
HAS_SCIPY = importlib.util.find_spec("scipy") is not None
if HAS_SCIPY:
    from scipy import signal
    from sim.acoustics.waveforms import advance, estimate_lag, prepare_recordings, save_dataset, load_dataset


@unittest.skipUnless(HAS_SCIPY, "scipy is needed for acoustic waveform extraction")
class WaveformsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.scan = self.root / "session_a"
        self.scan.mkdir()
        t = np.arange(1600) / 16000
        self.probe = 0.18 * signal.chirp(t, 2200, t[-1], 7000) * signal.windows.tukey(len(t), 0.15)
        self.probe_path = self.root / "probe.npy"
        np.save(self.probe_path, self.probe)
        self.rows = []

    def recording(self, k, gain=1, emission_shift=0, missing=False, volume=40):
        p = signal.resample_poly(self.probe, 3, 1)
        h = np.zeros(800)
        h[0], h[144], h[384] = 0.75, -0.22, 0.12
        pulse = signal.fftconvolve(p, h)
        x = np.zeros(6 * 48000)
        for j in range(3 if missing else 4):
            i = 24000 + j * 72000 + emission_shift
            x[i:i + len(pulse)] += gain * pulse
        fn = f"{k:02d}_test_100_200_v{volume}.pcm"
        (self.scan / fn).write_bytes(np.round(x * 32767).astype("<i2").tobytes())
        self.rows.append({"k": k, "kind": "test", "pan": 100, "tilt": 200, "pos": [100, 200],
                          "vol": volume, "t_rec": 1000 + k * 20, "t_play": 1001 + k * 20,
                          "bytes": len(x) * 2, "seconds": len(x) / 48000})
        (self.scan / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in self.rows))

    def prepare(self, **kwargs):
        return prepare_recordings([self.scan], self.probe_path, ("session_a", 0),
                                  designated_training=[("session_a", 0)], **kwargs)

    def test_signed_echoes_amplitudes_and_emission_offsets_survive(self):
        self.recording(0)
        self.recording(1, gain=1.5, emission_shift=137)
        ds = self.prepare()
        self.assertEqual(ds.h_aligned.shape[0], 2)
        self.assertEqual(ds.sweep_h_raw.shape[0], 8)
        # Gain must remain observable, rather than being normalized away.
        ratio = np.linalg.norm(ds.h_aligned[1]) / np.linalg.norm(ds.h_aligned[0])
        self.assertAlmostEqual(ratio, 1.5, delta=0.01)
        i0 = int(np.argmin(abs(ds.time_s)))
        echo = i0 + 144
        self.assertLess(ds.h_aligned[0, echo], 0)
        self.assertGreater(ds.h_aligned[0, i0], 0)
        offsets = [m["matched_filter_peak_sample"] for m in ds.sweep_metadata]
        self.assertEqual(offsets[4] - offsets[0], 137)
        self.assertEqual(ds.inverse_kernel.shape, ds.time_s.shape)
        self.assertTrue(np.iscomplexobj(ds.complex_spectra()))
        save_dataset(ds, self.root / "bundle")
        restored = load_dataset(self.root / "bundle")
        np.testing.assert_array_equal(restored.sweep_h_raw, ds.sweep_h_raw)
        self.assertEqual(restored.metadata["provenance"], ds.metadata["provenance"])

    def test_delay_sign_and_fractional_alignment(self):
        n = np.arange(1024)
        def f(t):
            return np.exp(-((t - 150) / 30) ** 2) * (np.cos(.48 * (t - 150)) + .6 * np.cos(.79 * (t - 150)))
        a = f(n)
        b = f(n - 2.37)
        lag, _, boundary = estimate_lag(a, b, 70, 240, 24)
        self.assertFalse(boundary)
        self.assertAlmostEqual(lag, 2.37, delta=.04)
        c = advance(b, lag)
        self.assertGreater(np.corrcoef(a[70:240], c[70:240])[0, 1], .9999)

    def test_kernel_matches_signed_echo_model_on_identical_grid(self):
        self.recording(0)
        ds = self.prepare()
        p = ds.sweep_metadata[0]["matched_filter_peak_sample"]
        anchor = ds.metadata["config"]["anchor_sample"]
        direct_t = (24000 - p + round(.005 * ds.fs) - anchor) / ds.fs
        model = sum(gain * np.interp(ds.time_s - direct_t - delay, ds.time_s, ds.inverse_kernel, left=0, right=0)
                    for delay, gain in [(0, .75), (.003, -.22), (.008, .12)])
        error = np.linalg.norm(ds.sweep_h_raw[0] - model) / np.linalg.norm(model)
        self.assertLess(error, .003)

    def test_reference_requires_designated_training_and_holdout_cannot_change_it(self):
        self.recording(0)
        original = self.prepare()
        self.recording(1, gain=1.5, emission_shift=90)
        extended = self.prepare()
        np.testing.assert_array_equal(original.h_raw[0], extended.h_raw[0])
        np.testing.assert_array_equal(original.h_aligned[0], extended.h_aligned[0])
        np.testing.assert_array_equal(original.inverse_kernel, extended.inverse_kernel)
        with self.assertRaisesRegex(ValueError, "training"):
            prepare_recordings([self.scan], self.probe_path, ("session_a", 1),
                               designated_training=[("session_a", 0)])

    def test_failed_capture_fails_closed_and_exclusions_are_counted(self):
        self.recording(0)
        self.recording(1, missing=True)
        self.recording(2, volume=0)
        self.recording(3, volume=30)
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.prepare()
        ds = self.prepare(allow_incomplete=True)
        self.assertFalse(ds.metadata["complete"])
        self.assertEqual(ds.metadata["counts"]["failed"], 1)
        self.assertEqual(ds.metadata["counts"]["excluded"], 2)
        self.assertEqual(ds.metadata["counts"]["accepted"], 1)
        failed = [q for q in ds.metadata["inventory"] if q["status"] == "failed"]
        self.assertEqual(failed[0]["detected_sweeps"], 3)


if __name__ == "__main__":
    unittest.main()
