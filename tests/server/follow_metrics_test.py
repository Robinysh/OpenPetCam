"""Visibility metrics exclude disabled Follow, missing telemetry and presence debounce."""
from pathlib import Path
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "server"))
from followmetrics import retention, status_age


class FollowMetricsTest(unittest.TestCase):
    def rows(self, yes):
        return [{"ts": i * 3., "score": 1 if v else 0, "box": [0, 0, .5, .5] if v else None,
                 "present": 1} for i, v in enumerate(yes)]

    def test_debounced_presence_does_not_inflate_visible_time(self):
        rows = self.rows([True, True, False, False, False])
        r = retention(rows, rows)
        self.assertEqual(r["visible_seconds"], 6)
        self.assertEqual(r["eligible_seconds"], 12)
        self.assertEqual(r["visible_percent"], 50)
        self.assertFalse(r["independent_ground_truth"])

    def test_gaps_and_disabled_follow_are_excluded(self):
        rows = self.rows([True, False, True, False, False])
        self.assertEqual(retention(rows, rows[:2])["eligible_seconds"], 3)
        rows[2]["ts"] = 100
        rows[3]["ts"] = 103
        rows[4]["ts"] = 106
        r = retention(rows, rows)
        self.assertEqual(r["logged_follow_seconds"], 9)
        self.assertEqual(r["visible_seconds"], 6)

    def test_no_evidence_is_unavailable_and_old_sighting_expires(self):
        self.assertIsNone(retention(self.rows([False] * 5), self.rows([False] * 5))["visible_percent"])
        rows = self.rows([True, False, False, False, False])
        r = retention(rows, rows, recent_s=5)
        self.assertEqual(r["eligible_seconds"], 5)
        self.assertEqual(r["visible_seconds"], 3)

    def test_status_age_uses_post_read_clock_and_never_claims_negative_freshness(self):
        self.assertEqual(status_age({"t": 100.125}, 100.25), .125)
        self.assertEqual(status_age({"t": 100.251}, 100.25), 0)
        self.assertIsNone(status_age({"t": 110}, 100))
        self.assertIsNone(status_age({"t": float("nan")}, 100))


if __name__ == "__main__":
    unittest.main()
