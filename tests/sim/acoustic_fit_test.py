"""Independent synthetic evidence for the offline single-surface waveform fit.

The fixture evaluates a continuous Gaussian wave packet at analytic image-source
delays. It does not call the fitter's path, interpolation, or template helpers.
No recordings, camera, private configuration, or network are required.
"""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
HAS_SCIPY = importlib.util.find_spec("scipy") is not None
if HAS_SCIPY:
    from sim.acoustics.fitting import fit_surface, permutation_check
    from sim.acoustics.experiment import parser, run, split_poses


def packet(t):
    """Continuous analytic pulse; negative-frequency mass is negligible."""
    return np.exp(-0.5 * (t / .00020) ** 2) * np.exp(2j * np.pi * 4300 * t)


def scene(distance=.46, offset=.000125, coefficient=.43 - .18j, plane=True):
    time_s = np.arange(-480, 1200) / 48000.
    theta = np.deg2rad(np.linspace(-82, 82, 15))
    sources = np.column_stack((.025 * np.sin(theta), np.full(len(theta), .012), .025 * np.cos(theta)))
    receivers = np.column_stack((.045 * np.sin(theta), np.full(len(theta), -.008), .045 * np.cos(theta)))
    # For z=d, the source image is (sx, sy, 2*d-sz).
    direct = np.sqrt(np.sum((sources - receivers) ** 2, axis=1))
    reflected = np.sqrt((sources[:, 0] - receivers[:, 0]) ** 2
                        + (sources[:, 1] - receivers[:, 1]) ** 2
                        + (2 * distance - sources[:, 2] - receivers[:, 2]) ** 2)
    delays = (reflected - direct) / 343. + offset
    common = .26 * np.exp(-.5 * ((time_s - .006) / .0005) ** 2) * np.cos(2 * np.pi * 3100 * time_s)
    observed = np.tile(common, (len(theta), 1))
    if plane:
        for i, delay in enumerate(delays):
            observed[i] += np.real(coefficient * packet(time_s - delay)) / reflected[i]
    train = np.arange(len(theta)) % 3 != 1
    return dict(observed=observed, time_s=time_s, kernel=np.real(packet(time_s)),
                sources_m=sources, receivers_m=receivers, normal=np.array([0., 0., 1.]),
                train_mask=train, distances_m=np.arange(.30, .621, .005),
                offsets_s=(0., .000125, .00025), window_s=(.0005, .012))


@unittest.skipUnless(HAS_SCIPY, "scipy is needed for acoustic waveform fitting")
class AcousticFitTest(unittest.TestCase):
    def test_recovers_independently_synthesized_fractional_echo(self):
        for truth in (.405, .46, .565):
            with self.subTest(distance_m=truth):
                data = scene(distance=truth)
                result = fit_surface(**data)
                selected = result["selected"]
                self.assertAlmostEqual(selected["distance_m"], truth, delta=.0051)
                self.assertAlmostEqual(selected["offset_s"], .000125, delta=1e-12)
                # Re(c*q) = Re(c)*Re(q) - Im(c)*Im(q).
                self.assertAlmostEqual(selected["coefficient_real"], .43, delta=.01)
                self.assertAlmostEqual(selected["coefficient_imag"], -.18, delta=.01)
                self.assertGreater(result["held_out"]["gain_vs_shared_background"], .999)

    def test_heldout_values_cannot_change_training_selection(self):
        data = scene()
        before = fit_surface(**data)
        mutated = data["observed"].copy()
        rng = np.random.default_rng(81)
        mutated[~data["train_mask"]] = 1e4 * rng.normal(size=mutated[~data["train_mask"]].shape)
        after = fit_surface(**dict(data, observed=mutated))
        self.assertEqual(before["selected"], after["selected"])
        self.assertEqual(before["distance_profile"], after["distance_profile"])
        self.assertEqual(before["timing_profile"], after["timing_profile"])
        self.assertNotEqual(before["held_out"]["sse"], after["held_out"]["sse"])

    def test_shared_trace_does_not_create_a_surface_coefficient(self):
        result = fit_surface(**scene(plane=False))
        self.assertEqual(result["status"], "no_pose_dependent_signal")
        selected = result["selected"]
        if selected is not None:
            self.assertLess(abs(selected["coefficient_real"]), 1e-10)
            self.assertLess(abs(selected["coefficient_imag"]), 1e-10)
        control = permutation_check(result, permutations=39, seed=4)
        self.assertEqual(control["p_lower_sse"], 1.)

    def test_zero_aperture_cannot_claim_pose_evidence(self):
        data = scene()
        for key in ("sources_m", "receivers_m"):
            data[key] = np.tile(data[key][0], (len(data[key]), 1))
        result = fit_surface(**data)
        self.assertEqual(result["status"], "no_pose_contrast")
        self.assertEqual(result["selected"]["rank"], 0)
        self.assertAlmostEqual(result["train"]["gain_vs_shared_background"], 0., delta=1e-12)
        self.assertAlmostEqual(result["held_out"]["gain_vs_shared_background"], 0., delta=1e-12)
        np.testing.assert_array_equal(result["arrays"]["surface_contribution_windowed"], 0.)

    def test_fixed_window_ignores_outside_samples_and_predicts_tapered_echo(self):
        data = scene()
        data["window_s"] = (.0021, .0031)
        before = fit_surface(**data)
        self.assertAlmostEqual(before["selected"]["distance_m"], .46, delta=.0051)
        self.assertGreater(before["held_out"]["gain_vs_shared_background"], .999)
        outside = (data["time_s"] < .0021) | (data["time_s"] > .0031)
        mutated = data["observed"].copy()
        mutated[:, outside] = 1e6
        after = fit_surface(**dict(data, observed=mutated))
        self.assertEqual(before["selected"], after["selected"])
        self.assertEqual(before["held_out"], after["held_out"])

    def test_pose_permutation_rejects_broken_association(self):
        result = fit_surface(**scene())
        control = permutation_check(result, permutations=199, seed=7)
        self.assertTrue(control["available"])
        self.assertLess(control["matched_sse"], control["shuffled_sse_median"])
        self.assertLess(control["p_lower_sse"], .05)

    def test_invalid_geometry_and_windows_are_rejected(self):
        data = scene()
        bad_normal = dict(data, normal=[0., 0., 2.])
        with self.assertRaises(ValueError):
            fit_surface(**bad_normal)
        with self.assertRaises(ValueError):
            fit_surface(**dict(data, window_s=(.012, .0005)))
        with self.assertRaises(ValueError):
            fit_surface(**dict(data, window_s=(.08, .09)))
        with self.assertRaises(ValueError):
            fit_surface(**dict(data, distances_m=(0.,)))


@unittest.skipUnless(HAS_SCIPY, "scipy is needed for the acoustic experiment module")
class AcousticPoseSplitTest(unittest.TestCase):
    def test_fit_window_cannot_overlap_early_alignment_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            args = parser().parse_args([
                "--scan", str(Path(tmp) / "scan"), "--probe", str(Path(tmp) / "probe.npy"),
                "--calibration", str(Path(tmp) / "calib.json"), "--reference", "scan:0",
                "--output", str(Path(tmp) / "result"), "--window-ms", "2.5", "7.5"])
            # The alignment window ends at 2 ms but its lag search reads a
            # further .5 ms. A declared .25 ms guard makes 2.75 ms the limit.
            with self.assertRaisesRegex(ValueError, "2.75 ms"):
                run(args)
            self.assertFalse(args.output.exists())
            args.window_ms = [2.75, 7.5]
            with patch("sim.acoustics.experiment.split_poses", side_effect=RuntimeError("split reached")):
                with self.assertRaisesRegex(RuntimeError, "split reached"):
                    run(args)

    def test_repeated_poses_across_sessions_stay_together_without_reading_pcm(self):
        with tempfile.TemporaryDirectory() as tmp:
            folders = [Path(tmp) / name for name in ("scan_a", "scan_b")]
            for folder in folders:
                folder.mkdir()
                rows = [{"k": k, "vol": 40, "pos": [k * 100, 300]} for k in range(9)]
                rows.append({"k": 20, "vol": 20, "pos": [9999, 9999]})
                # There are deliberately no PCM files: allocation depends on
                # recorded poses, not on response quality or audio contents.
                (folder / "log.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
            train, held, protocol = split_poses(folders, ("scan_a", 0), .3, 42, 40)
            self.assertIn(("scan_a", 0), train)
            self.assertIn(("scan_b", 0), train)
            self.assertEqual(len(held), 3)
            self.assertNotIn((9999, 9999), held)
            self.assertEqual(len(protocol["training_poses"]), 6)
            for k in range(9):
                self.assertEqual(("scan_a", k) in train, ("scan_b", k) in train)
                self.assertEqual(("scan_a", k) in train, (k * 100, 300) not in held)
            again = split_poses(list(reversed(folders)), ("scan_a", 0), .3, 42, 40)
            self.assertEqual(held, again[1])
            self.assertEqual(protocol, again[2])
            with self.assertRaises(ValueError):
                split_poses(folders, ("scan_a", 20), .3, 42, 40)


if __name__ == "__main__":
    unittest.main()
