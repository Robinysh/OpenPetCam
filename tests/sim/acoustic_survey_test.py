"""Independent checks of fixed-source gathers and balanced probe trial plans."""
from collections import Counter
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sim.acoustics.probes import ProbeConfig, build_probe_suite
from sim.acoustics.survey import make_plan


CALIBRATION = {"deg_per_step_pan": .1, "deg_per_step_tilt": .1,
               "level_tilt": 699, "pan_zero": 1850}


class SurveyPlanTest(unittest.TestCase):
    def test_each_gather_keeps_source_position_and_orientation_fixed(self):
        plan = make_plan(["selected"], CALIBRATION, mode="gather", pans=(950, 1850),
                         tilts=(520, 610, 699, 790, 880), blocks=2)
        groups = {}
        for entry in plan["entries"]:
            groups.setdefault(entry["gather_id"], []).append(entry)
        self.assertEqual(len(groups), 4)
        for entries in groups.values():
            source = entries[0]["source_m_assumed"]
            axis = entries[0]["source_axis_assumed"]
            for entry in entries:
                np.testing.assert_array_equal(entry["source_m_assumed"], source)
                np.testing.assert_array_equal(entry["source_axis_assumed"], axis)
            pan = entries[0]["pan"]
            # Independent reference poses: zero and a quarter-turn about y.
            expected_source = [-.048, .048, -.029] if pan == 950 else [0, .048, -.077]
            expected_axis = [-1, 0, 0] if pan == 950 else [0, 0, -1]
            np.testing.assert_allclose(source, expected_source, atol=1e-14, rtol=0)
            np.testing.assert_allclose(axis, expected_axis, atol=1e-14, rtol=0)
            receiver_positions = np.array([e["receiver_m_assumed"] for e in entries])
            self.assertGreater(np.linalg.norm(receiver_positions.max(axis=0) - receiver_positions.min(axis=0)), .02)
            receiver_axes = np.array([e["receiver_axis_assumed"] for e in entries])
            self.assertGreater(np.linalg.norm(receiver_axes.max(axis=0) - receiver_axes.min(axis=0)), .5)

    def test_reference_visits_bracket_contiguous_receiver_blocks(self):
        plan = make_plan(["selected"], CALIBRATION, mode="gather", pans=(950, 1850),
                         tilts=(520, 610, 699, 790, 880), blocks=3, seed=518)
        entries = plan["entries"]
        self.assertEqual(len(entries), 3 * 2 * 7)
        for start in range(0, len(entries), 7):
            group = entries[start:start + 7]
            self.assertEqual(len({e["gather_id"] for e in group}), 1)
            self.assertEqual(group[0]["role"], "reference_before")
            self.assertEqual(group[-1]["role"], "reference_after")
            self.assertEqual(group[0]["receiver_id"], group[-1]["receiver_id"])
            self.assertEqual(group[0]["tilt"], 699)
            self.assertEqual({e["tilt"] for e in group[1:-1]}, {520, 610, 699, 790, 880})
            self.assertTrue(all(e["role"] == "receiver" for e in group[1:-1]))
        self.assertEqual(len({e["capture_id"] for e in entries}), len(entries))

    def test_comparison_balances_every_packet_at_identical_pose_and_level(self):
        ids = ["linear-short", "exponential-short", "linear-long", "exponential-long"]
        plan = make_plan(ids, CALIBRATION, blocks=5, seed=518, volume=30)
        entries = plan["entries"]
        self.assertEqual(Counter(e["packet_id"] for e in entries), {k: 5 for k in ids})
        for block in range(5):
            self.assertEqual({e["packet_id"] for e in entries if e["block"] == block}, set(ids))
        self.assertEqual({(e["pan"], e["tilt"], e["volume"]) for e in entries}, {(1850, 699, 30)})
        for field in ("source_m_assumed", "source_axis_assumed", "receiver_m_assumed", "receiver_axis_assumed"):
            self.assertTrue(all(e[field] == entries[0][field] for e in entries))

    def test_randomization_is_reproducible_and_never_grants_timing_or_inversion(self):
        kwargs = dict(mode="gather", pans=(950, 1850, 2750), blocks=3)
        first = make_plan(["selected"], CALIBRATION, seed=101, **kwargs)
        self.assertEqual(first, make_plan(["selected"], CALIBRATION, seed=101, **kwargs))
        changed = make_plan(["selected"], CALIBRATION, seed=102, **kwargs)
        self.assertNotEqual(first["entries"], changed["entries"])
        self.assertFalse(first["timing_verified"])
        self.assertTrue(all(value is False for value in first["processing"].values()))
        self.assertIn("independently_calibrated_direct_path", first["timing_requirement"])

    def test_volume_study_balances_levels_between_fixed_level_anchors(self):
        plan = make_plan(["exact-probe"], CALIBRATION, mode="volume")
        self.assertEqual(plan["purpose"], "noisy_room_volume_robustness")
        self.assertEqual(plan["volumes"], [20, 30, 40])
        self.assertEqual(plan["reference_volume"], 30)
        self.assertEqual(plan["blocks"], 3)
        self.assertEqual(len(plan["entries"]), 15)
        self.assertEqual({(e["pan"], e["tilt"], e["packet_id"]) for e in plan["entries"]}, {(1850, 699, "exact-probe")})
        for block in range(3):
            entries = plan["entries"][5 * block:5 * block + 5]
            self.assertTrue(all(e["block"] == block for e in entries))
            self.assertEqual([entries[0]["role"], entries[-1]["role"]], ["reference_before", "reference_after"])
            self.assertEqual([entries[0]["volume"], entries[-1]["volume"]], [30, 30])
            self.assertTrue(all(e["role"] == "volume_trial" for e in entries[1:-1]))
            self.assertEqual(Counter(e["volume"] for e in entries[1:-1]), {20: 1, 30: 1, 40: 1})
        self.assertFalse(plan["processing"]["inversion_allowed"])

    def test_custom_volume_levels_randomize_reproducibly_and_reject_ambiguous_designs(self):
        kwargs = dict(mode="volume", volumes=(10, 30, 40), volume=30, blocks=4)
        first = make_plan(["exact-probe"], CALIBRATION, seed=101, **kwargs)
        self.assertEqual(first, make_plan(["exact-probe"], CALIBRATION, seed=101, **kwargs))
        self.assertNotEqual(first["entries"], make_plan(["exact-probe"], CALIBRATION, seed=102, **kwargs)["entries"])
        for levels in ((), (20,), (20, 20), (20, 41), (0, 30), (20, 30.5), (True, 30), (10, 20, 40)):
            with self.subTest(levels=levels), self.assertRaises(ValueError):
                make_plan(["exact-probe"], CALIBRATION, mode="volume", volumes=levels)
        with self.assertRaises(ValueError):
            make_plan(["first", "second"], CALIBRATION, mode="volume")
        with self.assertRaises(ValueError):
            make_plan(["exact-probe"], CALIBRATION, mode="volume", pans=(950, 1850))
        for mode in ("comparison", "gather"):
            with self.subTest(mode=mode), self.assertRaises(ValueError):
                make_plan(["exact-probe"], CALIBRATION, mode=mode, volumes=(20, 30))

    def test_cli_volume_study_records_explicit_levels_and_reference_volume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = build_probe_suite(root / "suite", ProbeConfig(durations_s=(.05,)))
            calibration = root / "calibration.json"
            calibration.write_text(json.dumps(CALIBRATION))
            output = root / "volume-plan.json"
            command = [sys.executable, "-m", "sim.acoustics.survey", "--suite", str(root / "suite/suite.json"),
                       "--calibration", str(calibration), "--output", str(output), "--mode", "volume",
                       "--packet", suite["packets"][0]["packet_id"], "--volumes", "20", "30", "40",
                       "--volume", "30", "--blocks", "2"]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(output.read_text())
            self.assertEqual(plan["volumes"], [20, 30, 40])
            self.assertEqual(plan["reference_volume"], 30)
            self.assertEqual(len(plan["entries"]), 10)

    def test_invalid_geometry_level_and_packet_selection_are_rejected(self):
        cases = [dict(pans=()), dict(pans=(1850, 1850)), dict(tilts=()),
                 dict(tilts=(699, 699)), dict(pans=(-1,)), dict(pans=(3701,)),
                 dict(tilts=(39,)), dict(tilts=(961,)), dict(reference_tilt=0),
                 dict(pans=(float("nan"),)), dict(volume=0), dict(volume=41),
                 dict(blocks=0), dict(blocks=21), dict(mode="unknown"),
                 dict(pans=(1850.5,)), dict(tilts=(699.5,)), dict(reference_tilt=699.5),
                 dict(pans=(True,)), dict(volume=30.5), dict(volume=True),
                 dict(blocks=2.5), dict(blocks=True), dict(seed=1.5), dict(seed=True),
                 dict(mode="comparison", pans=(950, 1850))]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                make_plan(["selected"], CALIBRATION, **kwargs)
        for ids in ([], ["same", "same"], [""], [None], [42]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                make_plan(ids, CALIBRATION)
        with self.assertRaises(ValueError):
            make_plan(["linear", "exponential"], CALIBRATION, mode="gather")
        with self.assertRaises(ValueError):
            make_plan(["selected"], dict(CALIBRATION, deg_per_step_pan=float("nan")))

    def test_cli_accepts_real_suite_schema_and_rejects_unknown_packet_or_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            suite = build_probe_suite(root / "suite", ProbeConfig(durations_s=(.5,)))
            calibration = root / "calibration.json"
            calibration.write_text(json.dumps(CALIBRATION))
            output = root / "plan.json"
            command = [sys.executable, "-m", "sim.acoustics.survey", "--suite", str(root / "suite" / "suite.json"),
                       "--calibration", str(calibration), "--output", str(output), "--blocks", "2"]
            result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(output.read_text())
            self.assertEqual({e["packet_id"] for e in plan["entries"]}, {e["packet_id"] for e in suite["packets"]})
            self.assertEqual(len(plan["entries"]), 4)
            original = output.read_bytes()
            self.assertNotEqual(subprocess.run(command, cwd=ROOT, capture_output=True).returncode, 0)
            self.assertEqual(output.read_bytes(), original)
            output.unlink()
            result = subprocess.run(command + ["--packet", "not-in-suite"], cwd=ROOT, capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("absent from suite", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
