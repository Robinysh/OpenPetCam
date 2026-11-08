"""Independent producer-format fixtures; no camera or SDK needed."""
import copy
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import zlib

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.acoustics.device.decode import export_capture, records, verify_ao_reference


def encoded(rows):
    out = bytearray()
    for meta, payload in rows:
        m = json.dumps(meta, separators=(",", ":")).encode()
        out.extend(struct.pack("<8sIII", b"OPCAUD01", len(m), len(payload), zlib.crc32(m + payload)))
        out.extend(m); out.extend(payload)
    return bytes(out)


def fixture(empty_edges=False):
    rows = []
    def add(kind, payload=b"", **fields):
        rows.append(({"schema": "openpetcam.audio.v1", "kind": kind, "stream_id": "demo",
                      "record_index": len(rows), "dropped_records": 7, **fields}, payload))
    add("hello")
    for i in range(5):
        samples = bytes([i, 0]) * 960
        ref = None if empty_edges and i in (0, 4) else bytes([50 + i, 0]) * 960
        desc = {"seq": 100 + i, "timestamp_us": 1000000 + i * 20000, "bytes": len(samples),
                "rate_hz": 48000, "bitwidth": 16, "channels": 1}
        reference = {**desc, "seq": 200 + i, "bytes": len(ref) if ref else 0}
        add("mic", samples + (ref or b""), mic=desc, reference=reference, rc=0,
            reference_state=1, monotonic_before_ns=1000000 + i * 1000,
            monotonic_after_ns=1000100 + i * 1000)
        if i == 2:
            add("ao", samples, ao={"submission_index": 25, "bytes": len(samples), "rate_hz": 48000,
                                   "bitwidth": 16, "channels": 1}, rc=0,
                monotonic_before_ns=1001000, monotonic_after_ns=1001100)
    add("end")
    return rows


class DecodeTest(unittest.TestCase):
    def export(self, data, **kwargs):
        tmp = tempfile.TemporaryDirectory(); self.addCleanup(tmp.cleanup)
        root = Path(tmp.name); path = root / "capture.frames"; path.write_bytes(data)
        return export_capture(path, root / "out", **kwargs)

    def test_exact_pcm_hashes_and_original_metadata_are_retained(self):
        rows = fixture()
        meta = self.export(encoded(rows))
        expected_mic = b"".join(p[:m["mic"]["bytes"]] for m, p in rows if m["kind"] == "mic")
        expected_ref = b"".join(p[m["mic"]["bytes"]:] for m, p in rows if m["kind"] == "mic")
        self.assertEqual(Path(meta["mic_file"]).read_bytes(), expected_mic)
        self.assertEqual(Path(meta["reference_file"]).read_bytes(), expected_ref)
        self.assertEqual(meta["frames"][0]["mic"]["seq"], 100)
        self.assertEqual(meta["common_mic_sample_interval"], [0, 4800])
        self.assertTrue(meta["reference_pair_contract_verified"])
        # Constant mock PCM is insufficient to validate an active timing map.
        self.assertFalse(meta["reference_sample_grid_verified"])
        self.assertFalse(meta["reference_continuity_verified"])
        self.assertFalse(meta["physical_DAC_timing_verified"])

    def test_empty_reference_edges_trim_both_arrays_on_common_interval(self):
        meta = self.export(encoded(fixture(empty_edges=True)))
        self.assertEqual(meta["common_mic_sample_interval"], [960, 3840])
        self.assertEqual(Path(meta["reference_file"]).stat().st_size, 3 * 1920)
        self.assertEqual(Path(meta["mic_file"]).stat().st_size, 3 * 1920)
        self.assertEqual(Path(meta["all_mic_file"]).stat().st_size, 5 * 1920)
        with self.assertRaisesRegex(ValueError, "edge"):
            self.export(encoded(fixture(empty_edges=True)), allow_empty_reference_edges=False)

    def test_missing_end_truncation_corruption_and_drop_change_reject(self):
        good = encoded(fixture())
        for data in (good[:-1], encoded(fixture()[:-1]), good + b"x"):
            with self.subTest(size=len(data)), self.assertRaises(ValueError):
                records(data)
        bad = bytearray(good); bad[-30] ^= 1
        with self.assertRaisesRegex(ValueError, "CRC"):
            records(bytes(bad))
        rows = fixture(); rows[2][0]["dropped_records"] += 1
        with self.assertRaisesRegex(ValueError, "dropped"):
            records(encoded(rows))

    def test_sdk_sequence_timestamp_and_reference_grid_gaps_reject(self):
        for field, value in (("seq", 999), ("timestamp_us", 9999999)):
            rows = copy.deepcopy(fixture()); rows[2][0]["mic"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.export(encoded(rows))
        rows = fixture(); rows[2][0]["reference"]["timestamp_us"] += 100
        with self.assertRaisesRegex(ValueError, "block grid"):
            self.export(encoded(rows))

    def test_reference_absence_never_sets_verification_flags(self):
        rows = fixture()
        for i, (meta, payload) in enumerate(rows):
            if meta["kind"] == "mic":
                meta["reference"]["bytes"] = 0
                rows[i] = meta, payload[:meta["mic"]["bytes"]]
        result = self.export(encoded(rows))
        self.assertIsNone(result["reference_file"])
        self.assertFalse(result["reference_sample_grid_verified"])
        self.assertFalse(result["reference_continuity_verified"])

    def test_sdk_jitter_and_unpopulated_reference_sequence_are_observations(self):
        rows = fixture()
        for i, (meta, _) in enumerate(q for q in rows if q[0]["kind"] == "mic"):
            stamp = 1000000 + i * 20002 + [0, 8, -4, 30, 3][i]
            meta["mic"]["timestamp_us"] = stamp
            meta["reference"]["timestamp_us"] = stamp
            meta["reference"]["seq"] = 0
        result = self.export(encoded(rows))
        self.assertTrue(result["mic_continuity_verified"])
        self.assertTrue(result["reference_pair_contract_verified"])
        self.assertFalse(result["timestamp_clock_verified"])
        self.assertIn("unavailable_all_zero", result["reference_sequence_provenance"])
        self.assertGreater(result["sdk_timestamp_diagnostics"]["residual_max_us"], 2)

    def test_active_reference_requires_whole_waveform_without_local_warps(self):
        rng = np.random.default_rng(47)
        source = rng.integers(-6000, 6000, 9600, dtype=np.int16)
        reference = np.r_[np.zeros(480, np.int16), source, np.zeros(960, np.int16)]
        result = verify_ao_reference(source.tobytes(), reference.tobytes())
        self.assertTrue(result["verified"], result)
        self.assertEqual(result["global_lag_samples"], 480)
        self.assertEqual(len(result["subwindows"]), 3)
        # A hidden sample deletion cannot be repaired by aligning each third.
        warped = np.r_[reference[:5000], reference[5005:], np.zeros(5, np.int16)]
        self.assertFalse(verify_ao_reference(source.tobytes(), warped.tobytes())["verified"])
        self.assertFalse(verify_ao_reference(source.tobytes(), bytes(len(reference) * 2))["verified"])

    def test_declared_16k_ao_is_checked_on_48k_paired_grid_with_fixed_hold(self):
        source = np.random.default_rng(19).integers(-8000, 8000, 9600, dtype=np.int16)
        held = np.round(.18 * np.repeat(source.astype(float), 3)).astype(np.int16)
        reference = np.r_[np.zeros(480, np.int16), held, np.zeros(960, np.int16)]
        verified = verify_ao_reference(source.tobytes(), reference.tobytes(), source_rate_hz=16000)
        self.assertTrue(verified["verified"], verified)
        self.assertAlmostEqual(verified["global_gain"], .18, delta=1e-4)
        self.assertEqual(verified["global_lag_samples"], 480)
        self.assertFalse(verify_ao_reference(source.tobytes(), reference.tobytes(), source_rate_hz=48000)["verified"])

    def test_long_silent_tail_cannot_win_normalized_reference_match(self):
        # FFT correlation has tiny nonzero roundoff even in silent windows.
        # Normalized correlation is undefined there, rather than enormous.
        rng = np.random.default_rng(90210)
        source = (rng.integers(-3000, 3001, 8192) * 4).astype('<i2')
        held = (np.repeat(source, 3) / 4).astype('<i2')
        reference = np.r_[np.zeros(1000, '<i2'), held, np.zeros(30000, '<i2')]
        result = verify_ao_reference(source.tobytes(), reference.tobytes(), source_rate_hz=16000)
        self.assertTrue(result['verified'], result)
        self.assertEqual(result['global_lag_samples'], 1000)
        self.assertAlmostEqual(result['global_gain'], .25, places=12)


if __name__ == "__main__":
    unittest.main()
