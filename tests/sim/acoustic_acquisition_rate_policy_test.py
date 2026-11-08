"""Acquisition defaults and explicit legacy controls, without hardware."""
import contextlib
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import tempfile
import unittest
import wave

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.acoustics import capture, probes
from sim.acoustics.survey import make_plan


class MemoryCamera:
    def __init__(self, rate=48000):
        self.rate, self.calls, self.current = rate, [], [1234, 650]
    def snapshot(self):
        self.calls.append('snapshot')
        return {'runtime_audio_format': {'record_rate_hz': 48000, 'output_rate_hz': self.rate, 'record_format_bits': 16, 'output_format_bits': 16}}
    def pose(self):
        self.calls.append('pose')
        return self.current.copy()
    @contextlib.contextmanager
    def hold(self):
        yield
    def goto(self, pan, tilt):
        self.calls.append('goto')
        self.current = [pan, tilt]
    def record(self, wav, output, volume, duration):
        self.calls.append('record')
        Path(output).write_bytes(bytes(round((duration + 3) * 96000)))
        return {'sample_rate_hz': 48000, 'hardware_timing_verified': False}


class RatePolicyTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def suite_plan(self, rate):
        suite = probes.build_probe_suite(self.root / 'suite', probes.ProbeConfig(
            sample_rate_hz=rate, durations_s=(.05,), repetitions=2))
        sp = self.root / 'suite/suite.json'
        plan = make_plan([p['packet_id'] for p in suite['packets']],
                         {'deg_per_step_pan': .1, 'deg_per_step_tilt': .1, 'level_tilt': 699}, blocks=1)
        plan['suite_sha256'] = hashlib.sha256(sp.read_bytes()).hexdigest()
        pp = self.root / 'plan.json'
        pp.write_text(json.dumps(plan))
        return suite, pp, sp

    def test_default_generator_is_direct_native48(self):
        suite = probes.build_probe_suite(self.root / 'suite', probes.ProbeConfig(durations_s=(.05,)))
        self.assertEqual(suite['config']['sample_rate_hz'], 48000)
        m = self.root / 'suite' / suite['packets'][0]['manifest']
        _, p, _, info = probes.load_probe_packet(m)
        self.assertEqual(len(p), 2400)
        with wave.open(str(m.parent / 'packet.wav'), 'rb') as w:
            self.assertEqual(w.getframerate(), 48000)
        # A directly sampled sweep is not a threefold hold of a16k archive.
        self.assertTrue(np.any(p[::3] != p[1::3]))
        self.assertEqual(info['sample_rate_hz'], 48000)

    def test_standard_capture_rejects_legacy16_before_hardware(self):
        _, pp, sp = self.suite_plan(16000)
        test = self
        class NoHardware:
            def snapshot(self): test.fail('legacy16 reached hardware without explicit opt-in')
        out = self.root / 'out'
        with self.assertRaisesRegex(ValueError, '48|rate'):
            capture.run_plan(pp, sp, out, NoHardware())
        self.assertFalse(out.exists())

    def test_direct_record_rejects_legacy16_before_upload(self):
        suite, _, _ = self.suite_plan(16000)
        source = self.root / 'suite' / Path(suite['packets'][0]['manifest']).parent / 'packet.wav'
        camera = capture.Camera.__new__(capture.Camera)
        camera.check_privacy = lambda: None
        camera.legacy_fifo = False
        camera.command = lambda *a, **k: self.fail('legacy16 reached upload without opt-in')
        with self.assertRaisesRegex(ValueError, '48|rate'):
            camera.record(source, self.root / 'out.pcm', 30, 4)

    def test_explicit16_diagnostic_keeps_original_bytes_and_records_intent(self):
        _, pp, sp = self.suite_plan(16000)
        self.assertEqual(hashlib.sha256(sp.read_bytes()).hexdigest(),
                         'da9b4b491cd6f1359c34b57099566fdde1ed4fb8da451cfa97e10d7fc1c923fb')
        before = {str(p.relative_to(sp.parent)):p.read_bytes() for p in sp.parent.rglob('*') if p.is_file()}
        _, archives = capture.assets(pp, sp)  # Historical analysis still reads16k.
        self.assertEqual({m['sample_rate_hz'] for m, _ in archives.values()}, {16000})
        result = capture.run_plan(pp, sp, self.root/'diagnostic', MemoryCamera(), allow_rate_switching=True)
        self.assertEqual(result['completed'], 2)
        self.assertTrue(result['acquisition_rate_policy']['allow_rate_switching'])
        self.assertEqual(result['acquisition_rate_policy']['purpose'], 'diagnostic_rate_switching')
        self.assertEqual(before, {str(p.relative_to(sp.parent)):p.read_bytes() for p in sp.parent.rglob('*') if p.is_file()})

    def test_default_native_plan_requires_configured48_before_movement(self):
        _, pp, sp = self.suite_plan(48000)
        wrong = MemoryCamera(16000)
        with self.assertRaisesRegex(RuntimeError, 'live48k'):
            capture.run_plan(pp, sp, self.root/'wrong', wrong)
        self.assertEqual(wrong.calls, ['snapshot'])
        ok = MemoryCamera()
        result = capture.run_plan(pp, sp, self.root/'ok', ok)
        self.assertEqual(result['completed'], 2)
        self.assertFalse(result['acquisition_rate_policy']['allow_rate_switching'])
        self.assertEqual(result['acquisition_rate_policy']['selected_source_rates_hz'], [48000])

    def test_direct_native_rejects_wrong_output_before_upload(self):
        suite, _, _ = self.suite_plan(48000)
        source = self.root/'suite'/Path(suite['packets'][0]['manifest']).parent/'packet.wav'
        camera = capture.Camera.__new__(capture.Camera)
        camera.check_privacy = lambda: None
        camera.legacy_fifo = False
        def command(cmd, *a, **k):
            if cmd == 'cat /proc/jz/audio/audio_aic_info':
                return b'The living rate of record : 48000\nThe living format of record : 16\nThe living rate of replay : 16000\nThe living format of replay : 16\n'
            self.fail('wrong live driver rate reached upload')
        camera.command = command
        with self.assertRaisesRegex(RuntimeError, 'live48k'):
            camera.record(source, self.root/'out.pcm', 30, 4)

    def test_driver_format_missing_ambiguous_or_wrong_fails_closed(self):
        text = 'The living rate of record : 48000\nThe living format of record : 16\nThe living rate of replay : 48000\nThe living format of replay : 16\n'
        fmt = capture.parse_audio_aic_format(text)
        self.assertEqual(capture.verify_configured_output_rate({'runtime_audio_format': fmt})['basis'],
                         'kernel_audio_aic_reported_format')
        for bad in (text.replace('rate of replay : 48000', 'rate of replay : 16000'),
                    text.replace('format of replay : 16', 'format of replay : 24')):
            with self.assertRaises(RuntimeError):
                capture.verify_configured_output_rate({'runtime_audio_format': capture.parse_audio_aic_format(bad)})
        for bad in ('', text.replace('The living rate of replay : 48000\n', ''),
                    text + 'The living rate of replay : 48000\n'):
            with self.assertRaises(RuntimeError):
                capture.parse_audio_aic_format(bad)
        # Unsupported JSON/config keys never establish the runtime rate.
        with self.assertRaises(RuntimeError):
            capture.verify_configured_output_rate({'runtime_audio': {'output_sample_rate': 48000}})

    def test_snapshot_uses_supported_queries_and_only_stable_driver_format(self):
        outer = self
        class App:
            def prudynt(self, query):
                outer.assertEqual(query, {'audio': {'mic_hq': None}})
                return {'audio': {'mic_hq': True}}
        camera = capture.Camera.__new__(capture.Camera)
        camera.legacy_fifo = False
        camera.reader, camera.socket = '/tmp/reader', '/tmp/socket'
        camera.check_privacy = lambda: None
        camera.app = App()
        gain = [25]
        def command(cmd):
            if cmd.startswith('test -x '): return b''
            if cmd == 'cat /etc/prudynt.json':
                return json.dumps({'audio': {'mic_hq': True, 'output_sample_rate': 16000,
                                             'unregistered_future_field': 42}}).encode()
            if cmd == 'cat /proc/jz/audio/audio_aic_info':
                return ('The living rate of record : 48000\nThe living format of record : 16\n'
                        'The living rate of replay : 48000\nThe living format of replay : 16\n'
                        f'The living left again of record : {gain[0]}\nDMA count : {gain[0]}\n').encode()
            if cmd == 'cat /etc/os-release': return b'test-build'
            if cmd == 'sha256sum /usr/bin/prudynt': return b'0' * 64 + b' /usr/bin/prudynt\n'
            self.fail('unexpected snapshot command: ' + cmd)
        camera.command = command
        before = camera.snapshot()
        gain[0] = 99
        after = camera.snapshot()
        self.assertEqual(before, after)
        self.assertEqual(before['runtime_output_rate_hz'], 48000)
        self.assertEqual(capture.verify_configured_output_rate(before)['basis'],
                         'kernel_audio_aic_reported_format')

    def test_cli_synthesizes48_by_default_and16_only_when_requested(self):
        for rate in (48000, 16000):
            out = self.root / f'cli{rate}'
            cmd = [sys.executable, '-m', 'sim.acoustics.probes', str(out), '--durations', '.05', '--repetitions', '2']
            if rate == 16000:
                cmd += ['--sample-rate', '16000']
            p = subprocess.run(cmd, capture_output=True, text=True)
            self.assertEqual(p.returncode, 0, p.stderr)
            self.assertEqual(json.loads((out/'suite.json').read_text())['config']['sample_rate_hz'], rate)


if __name__ == '__main__':
    unittest.main()
