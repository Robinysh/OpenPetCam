"""Real codec conversion preserves source/video and never replaces existing media."""
import hashlib
import importlib.util
import io
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
