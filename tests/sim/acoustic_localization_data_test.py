"""Independent native PCM fixtures exercise scale, timing gaps and leakage guards."""
import copy
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from sim.acoustics import localization_data as data


def timeline_fixture():
    return {"streams": [{"codec_type": "audio", "codec_name": "pcm_s16le", "sample_rate": "16000", "channels": 1}],
            "packets_and_frames": [
                {"type": "packet", "pts_time": str(t), "duration_time": ".1"}
                for t in (0, .1, .7, .8)] + [
                {"type": "frame", "pts_time": str(t), "nb_samples": 1600, "channels": 1}
                for t in (0, .1, .7, .8)]}


class TimelineTest(unittest.TestCase):
    def test_gaps_split_sample_intervals_without_inventing_samples(self):
        result = data.analyze_timeline(timeline_fixture())
        self.assertEqual(result["decoded_samples"], 6400)
        self.assertFalse(result["container_timeline_continuous"])
        self.assertEqual([(s["start_sample"], s["stop_sample"]) for s in result["segments"]], [(0, 3200), (3200, 6400)])
        self.assertAlmostEqual(result["frame_discontinuities"][0]["gap_s"], .5)
        self.assertFalse(result["absolute_time_calibrated"])

    def test_nonfinite_missing_and_changing_format_fail_closed(self):
        changes = [("frame", "pts_time", None), ("frame", "pts_time", "nan"),
                   ("frame", "nb_samples", 0), ("frame", "channels", 2),
                   ("packet", "duration_time", -1)]
        for kind, key, value in changes:
            fixture = timeline_fixture()
            next(row for row in fixture["packets_and_frames"] if row["type"] == kind)[key] = value
            with self.subTest(kind=kind, key=key), self.assertRaises(ValueError):
                data.analyze_timeline(fixture)

    def test_overlap_duplicate_and_unknown_time_groups_are_conservative(self):
        rows = [{"source_sha256": "a", "events": [{"event_ts": 1000}]},
                {"source_sha256": "b", "events": [{"event_ts": 1010}]},
                {"source_sha256": "a", "events": [{"event_ts": 200000}]},
                {"source_sha256": "c", "events": [{"event_ts": 400000}]},
                {"source_sha256": "d", "events": []}, {"source_sha256": "e", "events": []}]
        data.leakage_groups(rows)
        self.assertEqual(rows[0]["overlap_group"], rows[1]["overlap_group"])
        self.assertEqual(rows[0]["recommended_split_group"], rows[2]["recommended_split_group"])
        self.assertNotEqual(rows[0]["recommended_split_group"], rows[3]["recommended_split_group"])
        self.assertEqual(rows[4]["recommended_split_group"], rows[5]["recommended_split_group"])
        self.assertFalse(rows[4].get("split_eligible", True), "unknown timing cannot establish a disjoint holdout group")
        self.assertTrue(rows[0]["split_eligible"])

    def test_packet_gap_hidden_by_decoder_is_not_certified_as_a_continuous_segment(self):
        fixture = timeline_fixture()
        for frame, stamp in zip((x for x in fixture['packets_and_frames'] if x['type'] == 'frame'), (0, .1, .2, .3)):
            frame['pts_time'] = str(stamp)
        result = data.analyze_timeline(fixture)
        self.assertFalse(result.get('segments_timing_eligible', True), "decoder-smoothed frame PTS must not erase packet loss evidence")


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "local FFmpeg/FFprobe required")
class CorpusTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.rate = 16000
        t = np.arange(self.rate) / self.rate
        self.samples = np.rint(9000 * np.sin(2 * np.pi * 733 * t)).astype('<i2')
        self.samples[10], self.samples[11] = -24576, 12288
        self.source = self.root / "original.wav"
        self.write_wav(self.source, self.samples, channels=1)

    def write_wav(self, path, samples, channels):
        with wave.open(str(path), 'wb') as out:
            out.setnchannels(channels); out.setsampwidth(2); out.setframerate(self.rate); out.writeframes(samples.tobytes())

    def entry(self, path, ts=1000):
        return {"file": path.name, "sha256": data.sha256(path), "source_provenance": "current_original",
                "events": [{"event_id": 1, "event_ts": ts}]}

    def inventory(self, rows, references=()):
        path = self.root / "inventory.json"
        path.write_text(json.dumps({"repository_root": str(self.root), "passive_event_originals": rows,
                                    "raw_probe_references": list(references)}))
        return path

    def test_native_signed_samples_and_probe_bytes_preserved_and_duplicates_share_asset(self):
        alias = self.root / "alias.wav"; shutil.copyfile(self.source, alias)
        inventory = self.inventory([self.entry(self.source), self.entry(alias, 1010)],
                                   [{"file": self.source.name, "sha256": data.sha256(self.source), "role": "probe"}])
        before = data.sha256(self.source)
        corpus = self.root / "corpus"
        report = data.build_corpus(inventory, corpus)
        self.assertEqual(report["failures"], [])
        self.assertEqual(len(report["records"]), 1)
        row = report["records"][0]
        self.assertEqual(row["sample_rate_hz"], self.rate)
        self.assertEqual(row["decoded_samples"], len(self.samples))
        actual = np.fromfile(corpus / row["file"], dtype='<f4')
        np.testing.assert_array_equal(actual, self.samples.astype(np.float32) / 32768)
        self.assertLess(actual[10], 0); self.assertGreater(actual[11], 0)
        self.assertEqual(len(row["source_aliases"]), 2)
        self.assertIsNone(row["source_position_m"]); self.assertIsNone(row["microphone_pose"])
        self.assertEqual(row["source_kind"], "observed_room_recording")
        self.assertEqual(data.sha256(corpus / report["probe_references"][0]["file"]), before)
        self.assertEqual(data.sha256(self.source), before)
        with self.assertRaises(FileExistsError): data.build_corpus(inventory, corpus)

    def test_actual_container_gap_keeps_samples_and_separate_segments_without_padding(self):
        source = self.root / "gap.mka"
        subprocess.run(["ffmpeg", "-v", "error", "-nostdin", "-n", "-i", str(self.source),
                        "-af", "asetnsamples=n=1600:p=0,asetpts=PTS+gte(N\\,3200)*0.5/TB",
                        "-c:a", "pcm_s16le", str(source)], check=True)
        corpus = self.root / "corpus"
        report = data.build_corpus(self.inventory([self.entry(source)]), corpus)
        self.assertEqual(report["failures"], [])
        row = report["records"][0]
        self.assertEqual(row["status"], "decoded_with_timestamp_discontinuities")
        self.assertEqual(len(row["segments"]), 2)
        self.assertEqual(row["decoded_samples"], len(self.samples))
        np.testing.assert_array_equal(np.fromfile(corpus / row["file"], dtype='<f4'), self.samples.astype(np.float32) / 32768)

    def test_hash_corrupt_stereo_and_browser_sources_are_explicit_failures(self):
        stereo = self.root / "stereo.wav"; self.write_wav(stereo, np.repeat(self.samples[:, None], 2, axis=1), 2)
        broken = self.root / "broken.mp4"; broken.write_bytes(b"invalid media")
        derivative = self.root / "original.browser.mp4"; shutil.copyfile(self.source, derivative)
        bad_hash = self.entry(self.source); bad_hash["sha256"] = '0' * 64
        rows = [bad_hash, self.entry(stereo), self.entry(broken), self.entry(derivative)]
        report = data.build_corpus(self.inventory(rows), self.root / "corpus")
        self.assertEqual(len(report["failures"]), 4)
        self.assertEqual(report["records"], [])
        self.assertEqual(report["summary"]["input_originals"], 4)

    def test_decode_count_disagreement_cannot_be_published_as_valid(self):
        real_inspect = data.inspect_audio
        def wrong_count(path):
            result = copy.deepcopy(real_inspect(path))
            next(row for row in result["packets_and_frames"] if row['type'] == 'frame')['nb_samples'] += 1
            return result
        with patch.object(data, "inspect_audio", side_effect=wrong_count):
            report = data.build_corpus(self.inventory([self.entry(self.source)]), self.root / "corpus")
        self.assertEqual(report["records"], [])
        self.assertIn("sample count", report["failures"][0]["error"])


if __name__ == "__main__": unittest.main()
