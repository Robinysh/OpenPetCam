"""Retry/overlap/report completeness behavior with network, clocks and codecs at their boundaries."""
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("PETCAM_SITE_ENV", str(ROOT / "config/site.env.example"))
sys.path.insert(0, str(ROOT / "server"))
import app
import clipmedia
import watch


def info(duration=30):
    return {"streams": [{"codec_type": "video", "codec_name": "h264", "duration": str(duration), "start_time": "0"}],
            "format": {"duration": str(duration), "start_time": "0"},
            "video_timeline": {"usable_packets": 30, "unique_pts": 30, "invalid_packets": 0,
                               "first_pts_s": 0., "last_end_s": duration,
                               "covered_duration_s": duration, "largest_gap_s": 0.}}


class ClipAcquisitionTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for module, field, value in [(watch,"CLIPS",str(self.root / "clips")), (watch,"DB",str(self.root / "test.db"))]:
            context=patch.object(module,field,value);context.start();self.addCleanup(context.stop)
        watch.init_db()
        con=watch.db()
        for n in [1,2,3]: con.execute("INSERT INTO events(id,ts,kind) VALUES(?,?,'barking')",(n,1000+n))
        con.close()

    def rows(self):
        con=watch.db(); rows=con.execute("SELECT id,clip FROM events ORDER BY id").fetchall();con.close()
        return dict(rows)

    def media(self):
        return patch.object(clipmedia,"browser_copy",side_effect=lambda source,output: Path(output).write_bytes(Path(source).read_bytes()))

    def test_invalid_video_timeline_fails_closed(self):
        import copy
        for duration in ("-28.919889", "0", "nan", "inf", None):
            media=info();media["streams"][0]["duration"]=duration
            with self.subTest(duration=duration), self.assertRaisesRegex(ValueError,"video"):
                clipmedia.validate_video_timeline(media)
        for changes in ({"unique_pts":1}, {"covered_duration_s":.04}, {"largest_gap_s":29.},
                        {"first_pts_s":float("nan")}, {"invalid_packets":1}):
            media=copy.deepcopy(info());media["video_timeline"].update(changes)
            with self.subTest(packet_timeline=changes), self.assertRaisesRegex(ValueError,"video"):
                clipmedia.validate_video_timeline(media)

    def test_pending_overlap_and_later_overlap_both_link_to_the_existing_clip(self):
        worker=watch.Clips();worker.want(1,1000);worker.want(2,1005)
        with patch.object(watch.urllib.request,"urlopen",return_value=io.BytesIO(b"x"*20000)), \
                patch.object(clipmedia,"probe",return_value=info()), self.media():
            worker.cut(1,992)
        self.assertIsNotNone(self.rows()[2],"an overlapping moment was left without any playable clip")
        worker.want(3,1007)
        self.assertEqual(self.rows()[1],self.rows()[3],"an overlap requested after completion was not linked")

    def test_event_404_is_retried_before_the_clip_is_abandoned(self):
        responses=[urllib.error.HTTPError("local",404,"not ready",{},None),io.BytesIO(b"x"*20000)]
        with patch.object(watch.urllib.request,"urlopen",side_effect=responses) as fetch, \
                patch.object(clipmedia,"probe",return_value=info()), patch("time.sleep"), self.media():
            try: watch.Clips().cut(1,992)
            except Exception as exc: self.fail(f"temporary recording unavailability was not retried: {exc}")
        self.assertEqual(fetch.call_count,2)
        self.assertIsNotNone(self.rows()[1])

    def test_report_short_recording_is_explicitly_partial_and_retried(self):
        with patch.object(app.urllib.request,"urlopen",side_effect=lambda *a,**kw: io.BytesIO(b"x"*20000)) as fetch, \
                patch.object(clipmedia,"probe",return_value=info(40)), patch.object(app.time,"sleep"), \
                patch("time.sleep"), patch.object(app,"with_deadline",side_effect=lambda fn,_:fn()):
            app._report_clip(str(self.root),1000)
        error=self.root / "clip-error.txt"
        self.assertTrue(error.exists(),"a 40 s file was presented as the promised 100 s report recording")
        self.assertIn("partial",error.read_text().lower())
        self.assertEqual(fetch.call_count,3)
        receipt=json.loads((self.root / "clip-capture.json").read_text())
        self.assertFalse(receipt["duration_complete"])
        self.assertFalse(receipt["absolute_window_verified"])

    def test_report_retries_short_file_and_preserves_both_source_attempts(self):
        with patch.object(app.urllib.request,"urlopen",side_effect=[io.BytesIO(b"a"*20000),io.BytesIO(b"b"*20000)]) as fetch, \
                patch.object(clipmedia,"probe",side_effect=[info(40),info(100)]), patch.object(app.time,"sleep"), \
                patch("time.sleep"), patch.object(app,"with_deadline",side_effect=lambda fn,_:fn()):
            app._report_clip(str(self.root),1000)
        self.assertEqual(fetch.call_count,2,"the first partial file was silently accepted")
        self.assertEqual((self.root / "clip.mp4").read_bytes(),b"b"*20000)
        self.assertEqual(len(list(self.root.glob("clip.source.attempt-*.mp4"))),2)

    def test_report_permanent_unavailability_stops_after_three_attempts(self):
        with patch.object(app.urllib.request,"urlopen",side_effect=urllib.error.HTTPError("local",404,"not ready",{},None)) as fetch, \
                patch("time.sleep"), patch.object(app,"with_deadline",side_effect=lambda fn,_:fn()):
            app._report_clip(str(self.root),1000)
        self.assertEqual(fetch.call_count,3)
        self.assertFalse((self.root / "clip.mp4").exists())
        self.assertTrue((self.root / "clip-error.txt").exists())

    def test_event_near_the_old_clip_end_gets_its_own_tail(self):
        worker=watch.Clips(); worker.want(1,1000)
        self.assertTrue(worker.want(2,1021),"a moment without five seconds of following footage was incorrectly shared")

    def test_failed_job_is_not_reused_by_future_events(self):
        worker=watch.Clips();worker.want(1,1000)
        with patch.object(worker,"cut",side_effect=RuntimeError("unavailable")), patch.object(worker.jobs,"get",side_effect=[(1,992),StopIteration]), patch("time.sleep"):
            with self.assertRaises(StopIteration): worker.run()
        self.assertTrue(worker.want(2,1005),"a failed clip job swallowed a later moment")

    def test_failed_capture_explains_missing_clips_for_all_pending_events(self):
        worker=watch.Clips();worker.want(1,1000);worker.want(2,1005)
        with patch.object(worker,"cut",side_effect=RuntimeError("incomplete audio")), \
                patch.object(worker.jobs,"get",side_effect=[(1,992),StopIteration]), patch("time.sleep"):
            with self.assertRaises(StopIteration):worker.run()
        con=watch.db()
        try:
            for event_id in (1,2):
                raw=con.execute("SELECT detail FROM events WHERE id=?",(event_id,)).fetchone()[0]
                detail=json.loads(raw or '{}')
                self.assertIn('recording',detail.get('clip_warning','').lower())
                self.assertTrue(detail['clip_unusable'])
        finally:con.close()

    def test_failed_database_link_does_not_leave_a_dead_overlap_job(self):
        worker=watch.Clips();worker.want(1,1000)
        with patch.object(watch.urllib.request,"urlopen",return_value=io.BytesIO(b"x"*20000)), \
                patch.object(clipmedia,"probe",return_value=info()), self.media(), \
                patch.object(worker,"link_events",side_effect=RuntimeError("database locked")), \
                patch.object(worker.jobs,"get",side_effect=[(1,992),StopIteration]), patch("time.sleep"):
            with self.assertRaises(StopIteration):worker.run()
        self.assertTrue(worker.want(2,1005),"a DB failure after successful conversion left an unqueued overlap job")


if __name__ == "__main__": unittest.main()
