"""Circular bearing sets and spatially grouped selective-development controls."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.acoustics import localization_bearing as bearing


class BearingTests(unittest.TestCase):
    def test_small_arc_crosses_the_azimuth_wrap(self):
        center, width = bearing.circular_arc([-179, 179, -179])
        self.assertAlmostEqual(width, 2)
        self.assertAlmostEqual(bearing.angle_error(center, 180), 0)
        self.assertEqual(bearing.circular_arc([35]), (35, 0))
        self.assertEqual(bearing.circular_arc([35, 35]), (35, 0))
        self.assertAlmostEqual(bearing.circular_arc([0, 90, 180])[1], 180)

    def test_height_and_same_floor_tile_stay_in_one_fold(self):
        self.assertEqual(bearing.tile_fold([1.1, -.3, -2.2]), bearing.tile_fold([1.9, 1.2, -2.9]))
        self.assertEqual(bearing.tile_fold([1.1, -.3, -2.2]), bearing.tile_fold([1.1, -.3, -2.2]))

    def test_exact_tie_always_abstains_at_zero_threshold(self):
        row = dict(estimated_bearing_deg=10, score_margin=0, similarity=1, arc_width_deg=0)
        rule = dict(min_similarity=0, min_margin=0, max_arc_deg=60)
        self.assertFalse(bearing.accepts(row, rule))
        row['score_margin'] = .1
        self.assertTrue(bearing.accepts(row, rule))

    def test_background_errors_stay_null_and_lower_precision(self):
        positive = dict(estimated_bearing_deg=10, score_margin=.1, similarity=.9, arc_width_deg=10,
                        bearing_error_deg=0, source_m=[1, 0, 1])
        background = dict(positive, source_m=None, bearing_error_deg=None)
        rule = bearing.select_rule([positive] * 8 + [background] * 8)
        self.assertIsNone(rule['training_precision'])
        self.assertFalse(bearing.accepts(positive, rule))
        self.assertFalse(bearing.correct(background))

    def test_one_repeated_source_tile_cannot_calibrate_a_rule(self):
        common = dict(estimated_bearing_deg=0, score_margin=.1, similarity=.95, arc_width_deg=5,
                      bearing_error_deg=0, source_m=[1, 0, 1], spatial_tile='1:1')
        rule = bearing.select_rule([common] * 12)
        self.assertIsNone(rule['training_precision'])
        diverse = [dict(common, spatial_tile=f'{i % 3}:1') for i in range(12)]
        rule = bearing.select_rule(diverse)
        self.assertEqual(rule['training_precision'], 1)
        self.assertEqual(rule['training_accepted_tiles'], 3)

    def test_cross_validation_never_gives_fold_labels_to_training(self):
        rows = [dict(fold=i, case_id=str(i), estimated_bearing_deg=None) for i in range(5)]
        seen = []
        def fit(training):
            seen.append({row['fold'] for row in training})
            return dict(min_similarity=2, min_margin=2, max_arc_deg=0)
        with patch.object(bearing, 'select_rule', side_effect=fit):
            decisions, _ = bearing.cross_validate(rows)
        self.assertEqual(len(decisions), 5)
        for fold, training_folds in enumerate(seen):
            self.assertNotIn(fold, training_folds)
            self.assertEqual(len(training_folds), 4)

    def test_missing_sources_and_no_emitter_remain_in_denominators(self):
        common = dict(estimated_bearing_deg=0, top1_bearing_error_deg=0, acted_on=True)
        rows = [dict(common, source_m=[1, 0, 1], bearing_error_deg=0),
                dict(common, source_m=[2, 0, 1], bearing_error_deg=None,
                     estimated_bearing_deg=None, top1_bearing_error_deg=None, acted_on=False),
                dict(common, source_m=None, bearing_error_deg=None, top1_bearing_error_deg=None)]
        summary = bearing.summarize(rows)
        self.assertEqual(summary['unconditional_arc_center_within30deg_fraction'], .5)
        self.assertEqual(summary['source_success_coverage'], .5)
        self.assertEqual(summary['background_false_actions'], 1)
        self.assertEqual(summary['action_precision_within30deg_including_background_failures'], .5)


if __name__ == '__main__':
    unittest.main()
