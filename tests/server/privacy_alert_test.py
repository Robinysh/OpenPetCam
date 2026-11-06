"""Expected privacy gaps do not become capture incidents or false recovery pushes."""
import copy
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'server'))
import alert  # noqa: E402


class PrivacyAlertTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='opc-privacy-alert-')
        self.addCleanup(self.temp.cleanup)
        self.clock = 10000.0
        self.flag = Path(self.temp.name) / 'privacy'
        self.flag.write_text('1')
        os.utime(self.flag, (self.clock, self.clock))
        self.sent = []
        self.state = {'seen_events': []}
        self.heartbeat_age = 0
        self.recording_age = 9999
        self.events = []
        replacements = {
            'STATE': self.temp.name, 'REPORTS': str(Path(self.temp.name) / 'reports'),
            'PRIVACY': str(self.flag), 'load_state': lambda: self.state,
            'save_state': lambda st: setattr(self, 'state', copy.deepcopy(st)),
            'newest_recording_age': lambda: self.recording_age,
            'daily_summary': lambda st: None,
            'send': lambda *a, **kw: self.sent.append((a, kw)),
            'db': lambda: types.SimpleNamespace(execute=lambda *a: types.SimpleNamespace(fetchall=lambda: self.events),
                                                close=lambda: None),
            'age': lambda path: self.heartbeat_age if path == alert.HEARTBEAT else None,
        }
        for name, value in replacements.items():
            p = patch.object(alert, name, value, create=True)
            p.start(); self.addCleanup(p.stop)
        p = patch.object(alert.time, 'time', lambda: self.clock)
        p.start(); self.addCleanup(p.stop)
        p = patch.object(alert.subprocess, 'run', lambda *a, **kw: types.SimpleNamespace(returncode=0))
        p.start(); self.addCleanup(p.stop)
        p = patch.dict(sys.modules, {'scheduled_sweep': types.SimpleNamespace(check=lambda: [])})
        p.start(); self.addCleanup(p.stop)

    def test_paused_stale_recording_is_expected_and_sends_nothing(self):
        alert.check()
        self.assertEqual(self.sent, [])
        self.assertNotIn('video', self.state.get('incidents', {}))

    def test_pause_quietly_clears_old_capture_incidents_without_false_back_push(self):
        self.state['incidents'] = {'video': 9000, 'watch': 9000}
        alert.check()
        self.assertEqual(self.sent, [])
        self.assertNotIn('video', self.state['incidents'])
        self.assertNotIn('watch', self.state['incidents'])

    def test_paused_watcher_heartbeat_is_still_monitored(self):
        self.heartbeat_age = 300
        alert.check()
        self.assertEqual(len(self.sent), 1)
        title = self.sent[0][0][0].lower()
        self.assertIn('privacy', title)
        self.assertNotIn('video stopped', title)

    def test_resume_has_bounded_grace_then_detects_real_missing_recording(self):
        alert.check()
        self.flag.write_text('0')
        os.utime(self.flag, (self.clock, self.clock))
        alert.check()
        self.assertEqual(self.sent, [])
        self.clock += 91
        alert.check()
        self.assertEqual(len(self.sent), 1)
        self.assertIn('video stopped', self.sent[0][0][0].lower())

    def test_paused_camera_events_are_consumed_without_replaying_on_resume(self):
        self.events = [{'id': 1, 'ts': 9990, 'kind': 'camera_frozen',
                        'detail': '{"action":"restart streamer (no sound, mic on)"}'}]
        alert.check()
        self.assertEqual(self.sent, [])
        self.flag.write_text('0')
        self.recording_age = 0
        alert.check()
        self.assertEqual(self.sent, [])
        self.assertEqual(self.state['seen_events'], [(1, 9990)])


if __name__ == '__main__':
    unittest.main()
