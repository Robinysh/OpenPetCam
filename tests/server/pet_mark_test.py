"""The owner can see the pet while the app says she isn't in view. The detector's complete misses never became
training data (watch.py only saves a candidate when it proposed an animal), so the owner's "Mark <pet>" tap has to
save the picture with where they tapped, flagged as a miss when the watcher said she was absent. Marks are labels
only, rate-limited and capped per day.
  python3 tests/server/pet_mark_test.py
"""
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("PETCAM_SITE_ENV", str(ROOT / "config/site.env.example"))
sys.path.insert(0, str(ROOT / "server"))
import app


class PetMarkTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.dir = Path(temp.name)
        self.clock = 1_800_000_000.0
        self.live = {"present": False, "status": "ok", "score": 0.0, "box": None}
        self.aimed = []
        for p in [
            patch.object(app, "MARKS_DIR", temp.name),
            patch.object(app, "live_frame", side_effect=lambda: f"frame-{self.clock:.0f}".encode()),
            patch.object(app, "position", return_value=(1154, 616)),
            patch.object(app, "read_live", side_effect=lambda: dict(self.live)),
            patch.object(app, "aim", side_effect=lambda *a, **k: self.aimed.append(a)),
            patch.object(app.time, "time", side_effect=lambda: self.clock),
            patch.dict(app._mark_last, clear=True),
        ]:
            p.start()
            self.addCleanup(p.stop)

    def records(self):
        return [json.loads(p.read_text()) for p in sorted(self.dir.glob("*/*.json"))]

    def test_a_tap_while_the_app_says_absent_saves_a_missed_pet_label(self):
        out = app.mark_pet("phone", "Owner", {"fx": -0.35, "fy": 0.1})
        self.assertTrue(out["missed"], "a mark over present=False is a detector miss")
        [rec] = self.records()
        self.assertEqual(rec["label"], "pet")
        self.assertEqual(rec["point"], [0.15, 0.6])
        self.assertEqual(rec["pose"], [1154, 616])
        self.assertIs(rec["watch"]["present"], False)
        jpg = self.dir / (rec["id"] + ".jpg")
        self.assertEqual(jpg.read_bytes(), f"frame-{self.clock:.0f}".encode())
        self.assertEqual(self.aimed, [], "marking must not move the camera")

    def test_absent_mark_and_a_mark_she_was_seen_in(self):
        self.live["present"] = True
        self.assertFalse(app.mark_pet("phone", "Owner", {"fx": 0, "fy": 0})["missed"])
        self.clock += app.MARK_MIN_GAP_S
        out = app.mark_pet("phone", "Owner", {"absent": True})
        self.assertEqual(out["label"], "absent")
        self.assertEqual([r["label"] for r in self.records()], ["pet", "absent"])
        self.assertIsNone(self.records()[1]["point"])

    def test_bad_taps_are_refused_and_save_nothing(self):
        for body in ({}, {"fx": "x", "fy": 0}, {"fx": 0.7, "fy": 0}, {"fx": float("nan"), "fy": 0}):
            with self.assertRaises(ValueError):
                app.mark_pet("phone", "Owner", body)
        self.assertEqual(self.records(), [])

    def test_rate_limit_per_device(self):
        app.mark_pet("phone", "Owner", {"fx": 0, "fy": 0})
        self.clock += 1
        with self.assertRaises(app.Busy):
            app.mark_pet("phone", "Owner", {"fx": 0, "fy": 0})
        app.mark_pet("laptop", "Owner", {"fx": 0, "fy": 0})
        self.clock += app.MARK_MIN_GAP_S
        app.mark_pet("phone", "Owner", {"fx": 0, "fy": 0})
        self.assertEqual(len(self.records()), 3)

    def test_daily_cap(self):
        with patch.object(app, "MARK_MAX_PER_DAY", 3):
            for _ in range(3):
                app.mark_pet("phone", "Owner", {"fx": 0, "fy": 0})
                self.clock += app.MARK_MIN_GAP_S
            with self.assertRaises(app.Busy):
                app.mark_pet("phone", "Owner", {"fx": 0, "fy": 0})
        self.assertEqual(len(self.records()), 3)

    def test_no_picture_no_mark(self):
        with patch.object(app, "live_frame", side_effect=app.Busy("paused")):
            with self.assertRaises(app.Busy):
                app.mark_pet("phone", "Owner", {"fx": 0, "fy": 0})
        self.assertEqual(self.records(), [])
        app.mark_pet("phone", "Owner", {"fx": 0, "fy": 0})   # a failed grab doesn't use up the rate limit

    def test_the_live_view_has_a_mark_mode(self):
        html = (ROOT / "server/index.html").read_text()
        self.assertIn('id="mark-open"', html)
        self.assertIn('"/api/mark"', html)
        self.assertIn('if (marking) return sendMark(', html, "in mark mode a tap must label, not aim")


if __name__ == "__main__":
    unittest.main()
