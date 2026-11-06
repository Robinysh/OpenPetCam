"""Real codec conversion preserves source/video and never replaces existing media."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("PETCAM_SITE_ENV", str(ROOT / "config/site.env.example"))
sys.path.insert(0, str(ROOT / "server"))
import clipmedia
import watch


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg codec tools unavailable")
class ClipMediaTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / "source.mp4"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=320x180:rate=15",
                        "-f", "lavfi", "-i", "sine=frequency=600:sample_rate=48000", "-t", "2",
                        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "libopus", str(self.source)], check=True)

    def test_existing_opus_clip_gets_aac_without_touching_source_or_video(self):
        before = hashlib.sha256(self.source.read_bytes()).hexdigest()
        output = self.root / "browser.mp4"
        result = clipmedia.browser_copy(self.source, output)
        self.assertEqual(hashlib.sha256(self.source.read_bytes()).hexdigest(), before)
        self.assertEqual([s["codec_name"] for s in result["output"]["streams"]], ["h264", "aac"])
        def video_hash(path):
            return subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-map", "0:v:0",
                                            "-c", "copy", "-f", "hash", "-"])
        self.assertEqual(video_hash(self.source), video_hash(output))
        with self.assertRaises(FileExistsError):
            clipmedia.browser_copy(self.source, output)
        with self.assertRaises(FileExistsError):
            clipmedia.browser_copy(self.source, self.source)

    def short_video_source(self):
        path = self.root / "short-video-long-audio.mp4"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                        "testsrc2=size=320x180:rate=15:duration=0.5", "-f", "lavfi", "-i",
                        "sine=frequency=600:sample_rate=48000:duration=2", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-c:a", "libopus", str(path)], check=True)
        return path

    def short_audio_source(self):
        path = self.root / "long-video-short-audio.mp4"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                        "testsrc2=size=320x180:rate=15:duration=2", "-f", "lavfi", "-i",
                        "sine=frequency=600:sample_rate=48000:duration=0.5", "-c:v", "libx264",
                        "-pix_fmt", "yuv420p", "-c:a", "libopus", str(path)], check=True)
        return path

    def test_complete_video_does_not_claim_complete_short_audio_capture(self):
        data = self.short_audio_source().read_bytes()
        with patch.object(clipmedia.time, "sleep"):
            result = clipmedia.collect_recording(lambda: data, self.root / "capture.mp4", 2, self.root / "capture.json")
        self.assertFalse(result["duration_complete"], "video duration alone must not claim a complete audio/video window")
        self.assertTrue(result["video_window_complete"])
        self.assertFalse(result["audio_window_complete"])
        output = self.root / "partial-audio-browser.mp4"
        copied = clipmedia.browser_copy(self.root / "capture.mp4", output)
        self.assertTrue(output.exists(), "usable video may be retained with explicit partial-audio provenance")
        self.assertFalse(copied["full_av_window_complete"])
        self.assertFalse(copied["output_audio"]["audio_window_complete"])

    def test_deliberate_video_only_capture_remains_usable(self):
        source = self.root / "video-only.mp4"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", str(self.source),
                        "-map", "0:v:0", "-c", "copy", str(source)], check=True)
        result = clipmedia.collect_recording(lambda: source.read_bytes(), self.root / "capture.mp4", 2, self.root / "capture.json")
        self.assertTrue(result["duration_complete"])
        self.assertFalse(result["audio_present"])
        self.assertIsNone(result["audio_window_complete"])
        copied = clipmedia.browser_copy(source, self.root / "video-only-browser.mp4")
        self.assertTrue(copied["full_av_window_complete"])
        self.assertFalse(copied["output_audio"]["audio_present"])

    def test_short_video_cannot_borrow_duration_from_long_audio(self):
        source = self.short_video_source()
        before = hashlib.sha256(source.read_bytes()).hexdigest()
        output = self.root / "false-complete.mp4"
        with self.assertRaisesRegex(ValueError, "video"):
            clipmedia.browser_copy(source, output)
        self.assertFalse(output.exists(), "short video must not be published as a full audio-length clip")
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(), before)

    def test_capture_receipt_marks_short_video_incomplete_even_with_full_audio(self):
        data = self.short_video_source().read_bytes()
        source, receipt = self.root / "capture.mp4", self.root / "capture.json"
        with patch.object(clipmedia.time, "sleep"):
            result = clipmedia.collect_recording(lambda: data, source, 2, receipt)
        self.assertFalse(result["duration_complete"], "container duration cannot establish complete video")
        self.assertFalse(result["video_window_complete"])
        self.assertEqual(len(result["attempts"]), 3)
        self.assertEqual(len(list(self.root.glob("capture.attempt-*.mp4"))), 3)

    def test_backfill_rejects_short_video_without_changing_source_or_pointer(self):
        import sqlite3
        clips = self.root / "clips"; clips.mkdir()
        old = clips / "short.mp4"; shutil.copyfile(self.short_video_source(), old)
        before = hashlib.sha256(old.read_bytes()).hexdigest()
        con = sqlite3.connect(self.root / "petcam.db"); con.execute("CREATE TABLE events(clip TEXT)")
        con.execute("INSERT INTO events VALUES('short.mp4')"); con.commit(); con.close()
        result = self.backfill_tool().backfill(self.root, limit=1, execute=True)
        self.assertEqual(result["items"][0]["state"], "error")
        self.assertIn("video", result["items"][0]["error"])
        self.assertFalse((clips / "short.browser.mp4").exists())
        con = sqlite3.connect(self.root / "petcam.db")
        self.assertEqual(con.execute("SELECT clip FROM events").fetchone()[0], "short.mp4"); con.close()
        self.assertEqual(hashlib.sha256(old.read_bytes()).hexdigest(), before)

    def test_browser_output_is_checked_independently_before_publication(self):
        import copy
        good = clipmedia.probe(self.source)
        damaged = copy.deepcopy(good)
        damaged["streams"][0]["duration"] = "0.04"
        output = self.root / "damaged-output.mp4"
        with patch.object(clipmedia, "probe", side_effect=[good, damaged]):
            with self.assertRaisesRegex(ValueError, "video"):
                clipmedia.browser_copy(self.source, output)
        self.assertFalse(output.exists())

    def test_backfill_rejects_previously_receipted_short_video_output(self):
        import json
        import sqlite3
        clips = self.root / "clips"; clips.mkdir()
        old, output = clips / "old.mp4", clips / "old.browser.mp4"
        shutil.copyfile(self.source, old); shutil.copyfile(self.short_video_source(), output)
        output.with_suffix(".json").write_text(json.dumps({
            "source_sha256": hashlib.sha256(old.read_bytes()).hexdigest(),
            "browser_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}))
        con = sqlite3.connect(self.root / "petcam.db"); con.execute("CREATE TABLE events(clip TEXT)")
        con.execute("INSERT INTO events VALUES('old.mp4')"); con.commit(); con.close()
        result = self.backfill_tool().backfill(self.root, limit=1, execute=True)
        self.assertEqual(result["items"][0]["state"], "error")
        self.assertIn("video", result["items"][0]["error"])
        con = sqlite3.connect(self.root / "petcam.db")
        self.assertEqual(con.execute("SELECT clip FROM events").fetchone()[0], "old.mp4"); con.close()

    def test_backfill_audits_already_published_browser_video(self):
        import sqlite3
        clips = self.root / "clips"; clips.mkdir()
        shutil.copyfile(self.short_video_source(), clips / "broken.browser.mp4")
        con = sqlite3.connect(self.root / "petcam.db"); con.execute("CREATE TABLE events(clip TEXT)")
        con.execute("INSERT INTO events VALUES('broken.browser.mp4')"); con.commit(); con.close()
        result = self.backfill_tool().backfill(self.root, limit=1)
        self.assertTrue(result["items"], "a browser filename must not bypass video validation")
        self.assertEqual(result["items"][0]["state"], "error")

    def test_backfill_partial_audio_warning_is_published_with_matching_event_links(self):
        import sqlite3
        clips = self.root / "clips"; clips.mkdir()
        source = clips / "partial.mp4"; shutil.copyfile(self.short_audio_source(), source)
        con = sqlite3.connect(self.root / "petcam.db")
        con.execute("CREATE TABLE events(id INTEGER PRIMARY KEY, clip TEXT, detail TEXT)")
        con.executemany("INSERT INTO events VALUES(?,?,?)", [
            (1, "partial.mp4", json.dumps({"label": "keep", "sound": {"level": 3}})),
            (2, "partial.mp4", json.dumps({"clip_warning": "Separate video concern.", "clip_unusable": True})),
            (3, "partial.mp4", json.dumps({"label": "concurrent"}))])
        con.commit(); con.close()
        tool = self.backfill_tool()
        preview = tool.backfill(self.root, limit=1)
        self.assertTrue(preview["items"][0]["audio_partial"])
        con = sqlite3.connect(self.root / "petcam.db")
        self.assertFalse(any("clip_warning" in json.loads(row[0]) for row in con.execute("SELECT detail FROM events WHERE id != 2")))
        con.close()
        real_copy = tool.browser_copy

        def concurrent_copy(old, output):
            copied = real_copy(old, output)
            con = sqlite3.connect(self.root / "petcam.db")
            con.execute("UPDATE events SET clip='replacement.mp4', detail=? WHERE id=3", (json.dumps({"label": "updated"}),))
            con.commit(); con.close()
            return copied

        with patch.object(tool, "browser_copy", side_effect=concurrent_copy):
            result = tool.backfill(self.root, limit=1, execute=True)
        self.assertEqual(result["items"][0]["events_updated"], 2)
        con = sqlite3.connect(self.root / "petcam.db")
        rows = {ident: (clip, json.loads(detail)) for ident, clip, detail in con.execute("SELECT * FROM events")}
        con.close()
        self.assertEqual(rows[1][0], "partial.browser.mp4")
        self.assertIn("Audio", rows[1][1].get("clip_warning", ""), "partial audio was published without a visible event warning")
        self.assertIs(rows[1][1]["clip_unusable"], False, "valid video must remain playable")
        self.assertEqual(rows[1][1]["sound"], {"level": 3})
        self.assertEqual(rows[1][1]["label"], "keep")
        self.assertIn("Separate video concern.", rows[2][1]["clip_warning"])
        self.assertTrue(rows[2][1]["clip_unusable"], "backfill must not clear an independent video warning")
        self.assertEqual(rows[3], ("replacement.mp4", {"label": "updated"}))

    def test_backfill_adds_missing_warning_to_already_compatible_partial_audio(self):
        import sqlite3
        clips = self.root / "clips"; clips.mkdir()
        clipmedia.browser_copy(self.short_audio_source(), clips / "old.browser.mp4")
        shutil.copyfile(clips / "old.browser.mp4", clips / "already-aac.mp4")
        con = sqlite3.connect(self.root / "petcam.db")
        con.execute("CREATE TABLE events(clip TEXT, detail TEXT)")
        con.executemany("INSERT INTO events VALUES(?,?)", [(name, '{"keep":1}') for name in ("old.browser.mp4", "already-aac.mp4")])
        con.commit(); con.close()
        tool = self.backfill_tool()
        preview = tool.backfill(self.root, limit=2)
        self.assertEqual(len(preview["items"]), 2, "compatible codecs must not hide partial audio warnings")
        for _ in range(2):
            result = tool.backfill(self.root, limit=2, execute=True)
            self.assertTrue(all(item["state"] == "warning_updated" for item in result["items"]))
        con = sqlite3.connect(self.root / "petcam.db")
        for clip, raw in con.execute("SELECT * FROM events"):
            detail = json.loads(raw)
            self.assertEqual(detail["clip_warning"].count("Audio"), 1, "repeated audits duplicated the warning")
            self.assertFalse(detail["clip_unusable"])
            self.assertEqual(detail["keep"], 1)
            self.assertIn(clip, ("old.browser.mp4", "already-aac.mp4"))
        con.close()

    def test_backfill_malformed_event_detail_preserves_pointer_and_other_events(self):
        import sqlite3
        clips = self.root / "clips"; clips.mkdir()
        shutil.copyfile(self.short_audio_source(), clips / "partial.mp4")
        con = sqlite3.connect(self.root / "petcam.db")
        con.execute("CREATE TABLE events(clip TEXT, detail TEXT)")
        con.executemany("INSERT INTO events VALUES('partial.mp4',?)", [('{}',), ('[]',)])
        con.commit(); con.close()
        result = self.backfill_tool().backfill(self.root, limit=1, execute=True)
        self.assertEqual(result["items"][0]["state"], "error")
        con = sqlite3.connect(self.root / "petcam.db")
        self.assertEqual(con.execute("SELECT * FROM events").fetchall(), [("partial.mp4", "{}"), ("partial.mp4", "[]")])
        con.close()

    def test_new_event_clip_is_browser_compatible(self):
        data = self.source.read_bytes()
        with patch.object(watch, "CLIPS", str(self.root / "clips")), patch.object(watch, "DB", str(self.root / "test.db")), \
                patch.object(watch.urllib.request, "urlopen", return_value=io.BytesIO(data)), patch.object(watch, "CLIP_LEN", 2):
            watch.init_db()
            c = watch.db()
            c.execute("INSERT INTO events(id,ts,kind) VALUES(1,1000,'barking')")
            c.close()
            watch.Clips().cut(1, 1000)
            c = watch.db()
            name = c.execute("SELECT clip FROM events WHERE id=1").fetchone()[0]
            c.close()
            codecs = [s["codec_name"] for s in clipmedia.probe(self.root / "clips" / name)["streams"]]
            self.assertEqual(codecs, ["h264", "aac"])

    def backfill_tool(self):
        spec = importlib.util.spec_from_file_location("clip_backfill_tool", ROOT / "tools/backfill_clips.py")
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        return module

    def test_backfill_dry_run_changes_no_media_or_database_then_execution_preserves_source(self):
        import sqlite3
        clips = self.root / "clips"; clips.mkdir()
        old = clips / "old.mp4"; shutil.copyfile(self.source, old)
        con = sqlite3.connect(self.root / "petcam.db")
        con.execute("CREATE TABLE events(ts REAL,clip TEXT)")
        con.executemany("INSERT INTO events VALUES(?,?)", [(1,"old.mp4"),(2,"old.mp4")]);con.commit();con.close()
        tool = self.backfill_tool(); before = hashlib.sha256(old.read_bytes()).hexdigest()
        preview = tool.backfill(self.root, limit=1)
        self.assertEqual(preview["items"][0]["state"], "planned")
        self.assertFalse((clips / "old.browser.mp4").exists())
        result = tool.backfill(self.root, limit=1, execute=True)
        self.assertEqual(result["items"][0]["events_updated"],2)
        self.assertEqual(hashlib.sha256(old.read_bytes()).hexdigest(),before)
        con = sqlite3.connect(self.root / "petcam.db")
        self.assertEqual(con.execute("SELECT DISTINCT clip FROM events").fetchall(),[("old.browser.mp4",)])
        con.close()
        self.assertEqual([s["codec_name"] for s in clipmedia.probe(clips / "old.browser.mp4")["streams"]],["h264","aac"])

    def test_backfill_rejects_an_existing_unproven_output_without_changing_links(self):
        import sqlite3
        clips = self.root / "clips"; clips.mkdir()
        shutil.copyfile(self.source,clips / "old.mp4");(clips / "old.browser.mp4").write_bytes(b"untrusted")
        con = sqlite3.connect(self.root / "petcam.db");con.execute("CREATE TABLE events(ts REAL,clip TEXT)")
        con.execute("INSERT INTO events VALUES(1,'old.mp4')");con.commit();con.close()
        result = self.backfill_tool().backfill(self.root,limit=1,execute=True)
        self.assertEqual(result["items"][0]["state"],"error")
        self.assertEqual((clips / "old.browser.mp4").read_bytes(),b"untrusted")
        con = sqlite3.connect(self.root / "petcam.db")
        self.assertEqual(con.execute("SELECT clip FROM events").fetchone()[0],"old.mp4");con.close()

    def test_missing_sources_do_not_consume_conversion_quota_and_cursor_reaches_later_files(self):
        import sqlite3
        clips=self.root / "clips";clips.mkdir();shutil.copyfile(self.source,clips / "z-old.mp4")
        con=sqlite3.connect(self.root / "petcam.db");con.execute("CREATE TABLE events(ts REAL,clip TEXT)")
        con.executemany("INSERT INTO events VALUES(?,?)",[(i,f"missing-{i:04d}.mp4") for i in range(201)]+[(999,"z-old.mp4")]);con.commit();con.close()
        tool=self.backfill_tool();first=tool.backfill(self.root,limit=1)
        self.assertIsNotNone(first.get("next_after"),"bounded scan has no continuation cursor")
        second=tool.backfill(self.root,limit=1,after=first["next_after"])
        self.assertTrue(any(x["state"]=="planned" for x in second["items"]),"missing originals prevented a later valid clip from being reached")


if __name__ == "__main__":
    unittest.main()
