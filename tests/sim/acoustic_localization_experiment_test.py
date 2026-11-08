"""Coverage, selection and frozen-protocol controls for the offline benchmark."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.acoustics import localization_experiment as experiment
from sim.acoustics.localization import spatial_partition, summarize_errors
from sim.acoustics.localization_model import RoomModel


class ExperimentTests(unittest.TestCase):
    def test_missing_predictions_keep_coverage_and_do_not_get_errors(self):
        rows = [{"distance_error_m": .5, "azimuth_error_deg": 10, "accepted": True},
                {"distance_error_m": None, "azimuth_error_deg": None, "accepted": False}]
        summary = summarize_errors(rows)
        self.assertEqual(summary["cases"], 2)
        self.assertEqual(summary["unlocalized_cases"], 1)
        self.assertEqual(summary["within_1m_fraction"], .5)
        self.assertEqual(summary["acceptance_fraction"], .5)
        self.assertEqual(summary["median_error_m"], .5)
        self.assertEqual(experiment.selection_loss(rows, 12), 6.25)
        self.assertEqual(summarize_errors(rows[1:])["median_error_m"], None)

    def test_missing_predictions_cannot_win_by_omission(self):
        missing = [{"distance_error_m": None}] * 9 + [{"distance_error_m": 0}]
        complete = [{"distance_error_m": 2}] * 10
        self.assertGreater(experiment.selection_loss(missing, 10), experiment.selection_loss(complete, 10))

    def test_exact_ties_are_excluded_from_development_calibration(self):
        ties = [{"similarity": 1., "score_margin": 0., "distance_error_m": 0.}] * 20
        rule = experiment.thresholds(ties)
        self.assertIsNone(rule["development_precision"])
        self.assertFalse(experiment.passes_rule(ties[0], 0, 0))

    def test_frozen_spatial_partition_and_occluded_control(self):
        positions = np.array([[x, -.35, z] for x in np.arange(-3, 4, .5) for z in np.arange(-3, 4, .5)])
        visible = np.arange(len(positions)) % 3 != 0
        for split in ("development", "held_out"):
            cases = experiment.make_cases(positions, visible, count=48, split=split, seed=42, spacing=.5)
            self.assertEqual(set(spatial_partition([r["source_m"] for r in cases])), {split})
            for row in cases:
                if row["scenario"] == "nominal_grid_algorithm_check":
                    np.testing.assert_array_equal(row["source_m"], positions[row["source_grid_index"]])
                elif row["scenario"] == "occluded_source":
                    self.assertFalse(row["anchor_direct_visible"])
                else:
                    self.assertGreater(np.linalg.norm(np.asarray(row["source_m"]) - positions[row["source_grid_index"]]), 0)

    def test_all_zero_transfer_run_completes_and_preserves_every_case(self):
        with tempfile.TemporaryDirectory() as tmp:
            room_path, output = Path(tmp) / "room.json", Path(tmp) / "run"
            room_path.write_text("{}")
            positions = np.array([[i, 0, i] for i in range(6)], float)
            def make_cases(_positions, _visible, *, count, split, **_):
                return [dict(case_id=f"{split}-{i}", source_m=[1, 0, 1], scenario="off_grid", seed=i, split=split)
                        for i in range(count)]
            def waveform(_model, _source, _receiver, _scenario, _seed, held_out):
                self.assertTrue((output / "frozen.json").exists())
                if held_out:
                    self.assertTrue((output / "selected.json").exists())
                return np.zeros(8192), {"direct_visible": False}
            with patch.object(experiment, "load_room", return_value=RoomModel(())), \
                 patch.object(experiment, "candidate_grid", return_value=positions), \
                 patch.object(experiment, "dictionary", return_value=([np.zeros((6, 3, 10))] * 3, np.zeros(6, bool))), \
                 patch.object(experiment, "make_cases", side_effect=make_cases), \
                 patch.object(experiment, "simulate_case", side_effect=waveform), \
                 patch.object(experiment, "ripple_features", return_value=(np.zeros(10), 0)):
                result = experiment.run(room_path, output, cases=16, deadline_s=60)
            self.assertTrue(result["complete"])
            self.assertEqual(result["overall"]["unlocalized_cases"], 16)
            self.assertEqual(result["overall"]["within_1m_fraction"], 0)
            rows = [json.loads(line) for line in (output / "held_out.jsonl").read_text().splitlines()]
            self.assertEqual(len(rows), 16)
            self.assertTrue(all(r["position_m"] is None and r["distance_error_m"] is None for r in rows))
            self.assertGreater(result["selected"]["development_selection_loss_m"]["low"], 0)


if __name__ == "__main__":
    unittest.main()
