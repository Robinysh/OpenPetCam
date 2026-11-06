"""Old requests remain reachable and a reply cannot stand in for resolution."""
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


class ReportHistoryTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for n in range(45):
            p = self.root / f"20260101-{n:06d}"
            p.mkdir()
            (p / "report.json").write_text(json.dumps({"id": p.name, "ts": n, "note": f"Request {n}"}))
            (p / "reply.json").write_text(json.dumps({"status": "answered", "text": "A draft exists", "fix": {"status": "failed_checks"}}))
        self.mock = patch.object(app, "REPORTS_DIR", str(self.root))
        self.mock.start()
        self.addCleanup(self.mock.stop)

    def test_answered_failed_fix_is_not_done(self):
        row = app.list_reports(1)[0]
        self.assertEqual(row.get("status", {}).get("state"), "not_started")

    def test_all_older_requests_are_reachable_once(self):
        seen, before = [], None
        while True:
            page = app.report_page(20, before)
            self.assertEqual(page["total"], 45)
            seen.extend(q["id"] for q in page["reports"])
            before = page["next_before"]
            if before is None:
                break
        self.assertEqual(len(seen), 45)
        self.assertEqual(len(set(seen)), 45)
        self.assertEqual(seen[-1], "20260101-000000")

    def test_resolution_preserves_original_reply_and_requires_all_components(self):
        p = self.root / "20260101-000044"
        resolution = {"components": [{"id": "software", "title": "Software", "state": "done", "evidence": ["checked"]},
                                      {"id": "hardware", "title": "Physical switch", "state": "needs_hardware"}]}
        (p / "resolution.json").write_text(json.dumps(resolution))
        row = app.report_page()["reports"][0]
        self.assertEqual(row["status"]["state"], "needs_hardware")
        self.assertEqual(row["reply"]["text"], "A draft exists")
        self.assertEqual(row["resolution"], resolution)

    def test_malformed_record_cannot_hide_next_page(self):
        (self.root / "20260101-000024/report.json").write_text("broken")
        seen, cursor = [], None
        while True:
            page = app.report_page(10, cursor)
            seen.extend(q["id"] for q in page["reports"])
            cursor = page["next_before"]
            if cursor is None:
                break
        self.assertEqual(len(seen), 44)
        with self.assertRaises(ValueError):
            app.report_page(0)
        with self.assertRaises(ValueError):
            app.report_page(20, "../../outside")


if __name__ == "__main__":
    unittest.main()
