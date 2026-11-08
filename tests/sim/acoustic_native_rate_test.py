"""Native output-rate archives and raw reported-rate contracts; no hardware."""
import copy
import hashlib
import json
from pathlib import Path
import struct
import sys
import tempfile
import unittest
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.acoustics import capture, probes
from sim.acoustics.survey import make_plan


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value))


class NativeRateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / 'source'
        self.suite = probes.build_probe_suite(self.source, probes.ProbeConfig(sample_rate_hz=16000, durations_s=(.5,), repetitions=2))

    def native_fixture(self):
        """Independent PCM/header construction, not the production converter."""
        suite = copy.deepcopy(self.suite)
        suite['config']['sample_rate_hz'] = 48000
        for entry in suite['packets']:
            path = self.source / entry['manifest']
            m = json.loads(path.read_text())
            m['sample_rate_hz'] = m['config']['sample_rate_hz'] = 48000
            for event in m['events']:
                event['sample_start'] *= 3
                event['sample_end'] *= 3
            for key in ('packet', 'probe', 'pilot'):
                info = m[key]
                wav = path.parent / info['file']
                with wave.open(str(wav), 'rb') as src:
                    raw = np.frombuffer(src.readframes(src.getnframes()), '<i2')
                held = np.repeat(raw, 3).astype('<i2').tobytes()
                with wave.open(str(wav), 'wb') as dst:
                    dst.setparams((1, 2, 48000, 0, 'NONE', 'not compressed'))
                    dst.writeframes(held)
                info.update(samples=len(held)//2, pcm_sha256=hashlib.sha256(held).hexdigest(), sha256=sha(wav))
            write_json(path, m)
            entry['manifest_sha256'] = sha(path)
        write_json(self.source / 'suite.json', suite)
        return suite

    def plan(self, suite_path, suite):
        p = make_plan([x['packet_id'] for x in suite['packets']],
                      {'deg_per_step_pan': .1, 'deg_per_step_tilt': .1, 'level_tilt': 699}, blocks=1)
        p['suite_sha256'] = sha(suite_path)
        path = self.root / 'plan.json'
        write_json(path, p)
        return path

    def test_native_48k_archive_is_accepted_without_resampling(self):
        suite = self.native_fixture()
        try:
            _, assets = capture.assets(self.plan(self.source/'suite.json', suite), self.source/'suite.json')
        except ValueError as exc:
            self.fail('A valid native48k archive was rejected: '+str(exc))
        self.assertEqual({m['sample_rate_hz'] for m, _ in assets.values()}, {48000})
        for m, wav in assets.values():
            with wave.open(str(wav), 'rb') as src:
                self.assertEqual(src.getnframes(), m['packet']['samples'])
                self.assertEqual(src.getframerate(), 48000)

    def test_truncated_wav_header_cannot_pass_matching_payload_hash(self):
        entry = self.suite['packets'][0]
        path = self.source/entry['manifest']
        m = json.loads(path.read_text())
        wav = path.parent / m['probe']['file']
        raw = bytearray(wav.read_bytes())
        self.assertEqual(raw[36:40], b'data')
        struct.pack_into('<I', raw, 40, struct.unpack_from('<I', raw, 40)[0] + 2)
        struct.pack_into('<I', raw, 4, struct.unpack_from('<I', raw, 4)[0] + 2)
        wav.write_bytes(raw)
        m['probe']['sha256'] = sha(wav)  # Payload/sample manifest stays unchanged.
        write_json(path, m)
        with self.assertRaises(ValueError):
            probes.load_probe_packet(path)

    def test_derivation_preserves_exact_signed_samples_energy_time_and_events(self):
        before = {str(p.relative_to(self.source)): p.read_bytes() for p in self.source.rglob('*') if p.is_file()}
        derived = probes.derive_probe_suite_rate(self.source/'suite.json', self.root/'native')
        self.assertEqual(derived['config']['sample_rate_hz'], 48000)
        self.assertNotEqual(derived['suite_id'], self.suite['suite_id'])
        for original_entry, entry in zip(self.suite['packets'], derived['packets']):
            old = probes.load_probe_packet(self.source/original_entry['manifest'])
            new = probes.load_probe_packet(self.root/'native'/entry['manifest'])
            for a, b in zip(old[:3], new[:3]):
                np.testing.assert_array_equal(b, np.repeat(a, 3))
                self.assertEqual(len(a)/16000, len(b)/48000)
                self.assertEqual(float(abs(a).max()), float(abs(b).max()))
                self.assertAlmostEqual(float(a @ a)/16000, float(b @ b)/48000, places=12)
            for a, b in zip(old[3]['events'], new[3]['events']):
                self.assertEqual(a['event_id'], b['event_id'])
                self.assertEqual(b['sample_start'], a['sample_start']*3)
                self.assertEqual(b['sample_end'], a['sample_end']*3)
            self.assertEqual(new[3]['source_derivation']['source_manifest_sha256'], sha(self.source/original_entry['manifest']))
            self.assertEqual((self.root/'native'/Path(entry['manifest']).parent/'source-packet.json').read_bytes(),
                             (self.source/original_entry['manifest']).read_bytes())
        self.assertEqual(before, {str(p.relative_to(self.source)):p.read_bytes() for p in self.source.rglob('*') if p.is_file()})
        _, assets = capture.assets(self.plan(self.root/'native/suite.json', derived), self.root/'native/suite.json')
        self.assertEqual(len(assets), len(self.suite['packets']))
        with self.assertRaises(FileExistsError):
            probes.derive_probe_suite_rate(self.source/'suite.json', self.root/'native')

    def test_derivation_rejects_corruption_unsupported_rate_and_manifest_escape_before_output(self):
        for rate in (8000, 16000, 32000, True):
            with self.subTest(rate=rate), self.assertRaises(ValueError):
                probes.derive_probe_suite_rate(self.source/'suite.json', self.root/'bad', target_rate_hz=rate)
            self.assertFalse((self.root/'bad').exists())
        self.suite['packets'][0]['manifest'] = '../outside.json'
        write_json(self.source/'suite.json', self.suite)
        with self.assertRaises(ValueError):
            probes.derive_probe_suite_rate(self.source/'suite.json', self.root/'bad')
        self.assertFalse((self.root/'bad').exists())

    def test_corrupt_source_and_inconsistent_rate_fail_before_derivation(self):
        entry = self.suite['packets'][0]
        path = self.source / entry['manifest']
        original = path.read_bytes()
        m = json.loads(original)
        m['sample_rate_hz'] = 48000
        write_json(path, m)
        entry['manifest_sha256'] = sha(path)
        write_json(self.source/'suite.json', self.suite)
        with self.assertRaisesRegex(ValueError, 'rate'):
            probes.derive_probe_suite_rate(self.source/'suite.json', self.root/'bad')
        self.assertFalse((self.root/'bad').exists())
        path.write_bytes(original)
        entry['manifest_sha256'] = sha(path)
        write_json(self.source/'suite.json', self.suite)
        wav = path.parent / 'probe.wav'
        wav.write_bytes(wav.read_bytes()[:-2])
        with self.assertRaisesRegex(ValueError, 'hash'):
            probes.derive_probe_suite_rate(self.source/'suite.json', self.root/'bad')
        self.assertFalse((self.root/'bad').exists())

    def test_raw_rate_contract_checks_entire_capture_not_only_active_ao(self):
        device = {'complete':True,'source_rate_hz':48000,
                  'frames':[{'reference':{'ao_input_rate_hz':48000}} for _ in range(4)],
                  'ao_submissions':[{'ao':{'rate_hz':48000}} for _ in range(3)]}
        proof = capture.verify_constant_output_rate(device)
        self.assertTrue(proof['reported_rate_constant'])
        self.assertFalse(proof['physical_clock_verified'])
        for kind in ('idle_switch','missing_idle_rate','missing_frames','missing_ao','wrong_source'):
            bad = copy.deepcopy(device)
            if kind == 'idle_switch':bad['frames'][-1]['reference']['ao_input_rate_hz'] = 16000
            elif kind == 'missing_idle_rate':bad['frames'][0]['reference'].clear()
            elif kind == 'missing_frames':bad['frames'] = []
            elif kind == 'missing_ao':bad['ao_submissions'] = []
            else:bad['source_rate_hz'] = 16000
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                capture.verify_constant_output_rate(bad)

    def test_native_rate_refuses_unframed_mode_before_upload(self):
        suite = self.native_fixture()
        wav = self.source/Path(suite['packets'][0]['manifest']).parent/'packet.wav'
        camera = capture.Camera.__new__(capture.Camera)
        camera.legacy_fifo = True
        camera.check_privacy = lambda: None
        camera.command = lambda *a, **k: self.fail('native unframed source reached hardware')
        with self.assertRaisesRegex(ValueError, 'framed'):
            camera.record(wav, self.root/'bad.pcm', 40, 8)


if __name__ == '__main__':
    unittest.main()
