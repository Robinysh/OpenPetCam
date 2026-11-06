"""Privacy must latch before I/O, reject missing proof and remain closed on failed resume.

Camera/process/config doubles exercise the real API state machine. No camera,
network, service, media or deployment command is executed.
"""
import json
import io
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('PETCAM_SITE_ENV', str(ROOT / 'config/site.env.example'))
sys.path.insert(0, str(ROOT / 'server'))
import app


class PrivacyTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.flag = self.root / 'privacy'
        self.flag.write_text('0\n')
        self.installed = True
        self.paused = False
        self.running = True
        self.camera_picture = False
        self.camera_mic = True
        self.commands = []
        self.resume_hook = None
        self.patches = [patch.object(app, 'STATE_DIR', str(self.root)),
                        patch.object(app, 'PRIVACY_RESUME_WAIT', 0),
                        patch.object(app._camsh, 'run', side_effect=self.shell),
                        patch.object(app, 'prudynt', side_effect=self.config),
                        patch.object(app, 'cam_get', return_value={})]
        self.mocks = [p.start() for p in self.patches]
        for p in self.patches:
            self.addCleanup(p.stop)
        self.prudynt = self.mocks[3]
        self.cgi = self.mocks[4]

    def shell(self, command, timeout=0):
        action = re.search(r'/etc/opc-privacy/control (status|pause|resume)', command).group(1)
        self.assertTrue(app.privacy_requested(), 'acquisition must be latched before every transition command')
        self.commands.append(action)
        if not self.installed:
            return json.dumps({'installed': False})
        if action == 'pause':
            self.paused, self.running = True, False
        elif action == 'resume':
            self.paused, self.running = False, True
            if self.resume_hook:
                self.resume_hook()
        return json.dumps(self.guard_status())

    def guard_status(self):
        mode = 'capture_stopped' if self.paused and not self.running else \
               'capture_running' if not self.paused and self.running else 'capture_unavailable'
        return {'mode': mode, 'privacy': self.paused, 'paused': self.paused, 'running': self.running,
                'verified': self.paused != self.running, 'boot_persistent': True}

    def config(self, body):
        self.assertTrue(app.privacy_requested(), 'camera readiness must be checked before acquisition is reopened')
        picture = body.get('privacy', {}).get('enabled')
        mic = body.get('audio', {}).get('mic_enabled')
        if picture is not None:
            self.camera_picture = picture
        if mic is not None:
            self.camera_mic = mic
        return {'privacy': {'enabled': self.camera_picture}, 'audio': {'mic_enabled': self.camera_mic}}

    def state(self):
        return app.privacy_status()

    def test_strict_bool_rejects_strings_null_and_numbers_without_io(self):
        for value in (None, 'false', '', 0, 1, [], {}):
            with self.subTest(value=value), self.assertRaises(ValueError):
                app.set_privacy(value)
        self.assertFalse(app.privacy_requested())
        self.assertEqual(self.commands, [])
        self.prudynt.assert_not_called()

    def test_pause_latches_before_any_camera_io_and_records_process_stop_proof(self):
        result = app.set_privacy(True)
        self.assertEqual(self.commands, ['status', 'pause'])
        self.assertTrue(app.privacy_requested())
        self.assertFalse(self.running)
        self.assertEqual(result['privacy_status']['mode'], 'capture_stopped')
        self.assertEqual(result['privacy_status']['phase'], 'paused')
        self.assertTrue(result['privacy_status']['confirmed'])
        self.assertTrue(result['privacy_status']['boot_persistent'])
        self.prudynt.assert_not_called()
        self.assertEqual(self.flag.read_text(), '1\n')
        self.assertEqual((self.root / 'privacy-guard-required').read_text(), '1\n')

    def test_failed_guard_never_silently_uses_weaker_suppression(self):
        original = self.shell
        def failing(command, timeout=0):
            if 'control pause;' in command:
                raise OSError('untrusted transport detail')
            return original(command, timeout)
        self.mocks[2].side_effect = failing
        with self.assertRaises(app.PrivacyError):
            app.set_privacy(True)
        self.assertTrue(app.privacy_requested())
        self.assertFalse(self.state()['confirmed'])
        self.assertEqual(self.state()['phase'], 'error')
        self.assertNotIn('untrusted', self.state()['message'])
        self.prudynt.assert_not_called()

    def test_guard_verification_and_both_process_marker_fields_are_required(self):
        original = self.shell
        for field in ('verified', 'paused', 'privacy', 'running', 'mode', 'boot_persistent'):
            with self.subTest(field=field):
                def incomplete(command, timeout=0):
                    value = json.loads(original(command, timeout))
                    if 'control pause;' in command:
                        value.pop(field)
                    return json.dumps(value)
                self.mocks[2].side_effect = incomplete
                with self.assertRaises(app.PrivacyError):
                    app.set_privacy(True)
                self.assertTrue(app.privacy_requested())
                self.assertFalse(self.state()['confirmed'])
                self.prudynt.assert_not_called()

    def test_missing_previously_installed_guard_is_an_error(self):
        app.set_privacy(True)
        self.installed = False
        with self.assertRaises(app.PrivacyError):
            app.set_privacy(True)
        self.assertTrue(app.privacy_requested())
        self.prudynt.assert_not_called()

    def test_transport_failure_is_not_proof_helper_is_absent(self):
        self.mocks[2].side_effect = OSError('camera is unreachable')
        with self.assertRaises(app.PrivacyError):
            app.set_privacy(True)
        self.assertTrue(app.privacy_requested())
        self.assertFalse(self.state()['confirmed'])
        self.prudynt.assert_not_called()

    def test_explicit_retry_recovers_only_a_cached_channel_after_camera_is_online(self):
        app.set_privacy(True)
        key = self.root / 'test-key'
        key.touch()
        original = self.shell
        def cached(command, timeout=0):
            if time.time() < app._camsh.down_until:
                raise OSError('camera ssh unavailable')
            return original(command, timeout)
        self.mocks[2].side_effect = cached
        with patch.object(app._camsh, 'down_until', time.time() + 30), \
                patch.object(app, 'CAM_SSH_KEY', str(key)), patch.object(app.socket, 'create_connection') as online:
            result = app.set_privacy(False)
            self.assertFalse(app.privacy_requested())
            self.assertTrue(result['privacy_status']['confirmed'])
            online.assert_called_once_with((app.CAM_HOST, 80), timeout=2)
            self.assertEqual(app._camsh.down_until, 0.0)
        self.assertEqual(self.commands, ['status', 'pause', 'status', 'resume'])

    def test_cached_channel_does_not_retry_while_camera_is_still_offline(self):
        key = self.root / 'test-key'
        key.touch()
        self.mocks[2].side_effect = OSError('camera ssh unavailable')
        until = time.time() + 30
        with patch.object(app._camsh, 'down_until', until), patch.object(app, 'CAM_SSH_KEY', str(key)), \
                patch.object(app.socket, 'create_connection', side_effect=OSError('offline')):
            with self.assertRaises(app.PrivacyError):
                app.set_privacy(False)
            self.assertEqual(app._camsh.down_until, until)
        self.mocks[2].assert_called_once()
        self.assertTrue(app.privacy_requested())
        self.prudynt.assert_not_called()

    def test_fresh_auth_failure_and_watcher_refresh_are_not_retried(self):
        for refresh in (False, True):
            with self.subTest(refresh=refresh):
                self.flag.write_text('1\n')
                self.mocks[2].reset_mock()
                self.mocks[2].side_effect = OSError('Permission denied' if not refresh else 'camera ssh unavailable')
                with patch.object(app._camsh, 'down_until', time.time() + 30 if refresh else 0), \
                        patch.object(app.socket, 'create_connection') as online:
                    with self.assertRaises(app.PrivacyError):
                        app.ensure_privacy_paused() if refresh else app.set_privacy(False)
                    online.assert_not_called()
                    self.mocks[2].assert_called_once()
                self.assertTrue(app.privacy_requested())

    def test_mutating_guard_commands_are_never_retried(self):
        self.flag.write_text('1\n')
        self.mocks[2].side_effect = OSError('camera ssh unavailable')
        with patch.object(app._camsh, 'down_until', time.time() + 30), \
                patch.object(app.socket, 'create_connection') as online:
            with self.assertRaises(OSError):
                app._privacy_guard('pause', retry_cached=True)
            self.mocks[2].assert_called_once()
            online.assert_not_called()

    def test_definitely_absent_helper_uses_explicit_legacy_blackout_mode(self):
        self.installed = False
        result = app.set_privacy(True)
        self.assertEqual(self.prudynt.call_args.args[0], {'privacy': {'enabled': True}, 'audio': {'mic_enabled': False}})
        self.assertTrue(result['privacy_status']['confirmed'])
        self.assertFalse(result['privacy_status']['boot_persistent'])
        self.assertEqual(result['privacy_status']['mode'], 'blackout')
        self.assertIn('Capture remains running', result['privacy_status']['message'])

    def test_legacy_empty_or_partial_reply_cannot_confirm_privacy(self):
        self.installed = False
        for value in ({}, {'privacy': {'enabled': True}},
                      {'privacy': {'enabled': True}, 'audio': {'mic_enabled': True}},
                      {'privacy': {'enabled': 'true'}, 'audio': {'mic_enabled': 'false'}}):
            with self.subTest(value=value):
                self.prudynt.side_effect = None
                self.prudynt.return_value = value
                with self.assertRaises(app.PrivacyError):
                    app.set_privacy(True)
                self.assertTrue(app.privacy_requested())
                self.assertFalse(self.state()['confirmed'])

    def test_resume_clears_latch_only_after_guard_and_picture_mic_readback(self):
        app.set_privacy(True)
        result = app.set_privacy(False)
        self.assertEqual(self.commands, ['status', 'pause', 'status', 'resume'])
        self.assertEqual(self.prudynt.call_args.args[0], {'privacy': {'enabled': None}, 'audio': {'mic_enabled': None}})
        self.assertFalse(app.privacy_requested())
        self.assertFalse(result['privacy'])
        self.assertTrue(result['mic'])
        self.assertTrue(result['privacy_status']['confirmed'])
        self.assertEqual(result['privacy_status']['phase'], 'active')
        self.cgi.assert_not_called()

    def test_resume_already_running_verifies_readback_without_a_restart(self):
        result = app.set_privacy(False)
        self.assertEqual(self.commands, ['status'])
        self.assertFalse(app.privacy_requested())
        self.assertTrue(result['privacy_status']['confirmed'])
        self.prudynt.assert_called_once()
        self.cgi.assert_not_called()

    def test_reconciliation_uses_current_stop_proof_without_repeating_stop(self):
        app.set_privacy(True)
        before = list(self.commands)
        state = app.ensure_privacy_paused()
        self.assertEqual(self.commands, before + ['status'])
        self.assertTrue(state['confirmed'])
        self.assertEqual(state['mode'], 'capture_stopped')

    def test_reconciliation_stops_unexpected_capture_after_pause(self):
        app.set_privacy(True)
        self.running = True
        state = app.ensure_privacy_paused()
        self.assertEqual(self.commands[-2:], ['status', 'pause'])
        self.assertFalse(self.running)
        self.assertTrue(state['confirmed'])

    def test_intent_timestamp_is_current_in_post_receipt_and_stable_during_reconciliation(self):
        result = app.set_privacy(True)
        stamp = self.flag.stat().st_mtime_ns / 1_000_000_000
        self.assertEqual(result['privacy_status']['intent_changed_at'], stamp)
        self.assertEqual(self.state()['intent_changed_at'], stamp)
        self.assertEqual(app.ensure_privacy_paused()['intent_changed_at'], stamp)
        self.assertEqual(self.state()['intent_changed_at'], stamp)
        self.mocks[2].side_effect = OSError('temporary lost connection')
        with self.assertRaises(app.PrivacyError):
            app.ensure_privacy_paused()
        self.assertEqual(self.state()['intent_changed_at'], stamp)
        self.assertEqual(self.flag.stat().st_mtime_ns / 1_000_000_000, stamp)

    def test_explicit_intent_changes_timestamp_and_ignores_stored_timestamp(self):
        app.set_privacy(True)
        os.utime(self.flag, ns=(1_000_000_000, 1_000_000_000))
        metadata = self.root / 'privacy-state.json'
        state = json.loads(metadata.read_text())
        state['intent_changed_at'] = 99999999999
        metadata.write_text(json.dumps(state))
        self.assertEqual(self.state()['intent_changed_at'], 1.0)
        result = app.set_privacy(False)
        self.assertGreater(result['privacy_status']['intent_changed_at'], 1.0)
        self.assertEqual(result['privacy_status']['intent_changed_at'], self.state()['intent_changed_at'])

    def test_missing_intent_file_has_no_timestamp(self):
        self.flag.unlink()
        self.assertIsNone(self.state()['intent_changed_at'])

    def test_atomic_pause_cannot_attach_new_intent_version_to_an_old_active_value(self):
        os.utime(self.flag, ns=(1_000_000_000, 1_000_000_000))
        original_open = open
        class OldIntent:
            def __init__(inner, stream):
                inner.stream = stream
            def __enter__(inner):
                return inner
            def __exit__(inner, *args):
                inner.stream.close()
            def read(inner):
                text = inner.stream.read()
                app._privacy_atomic('privacy', '1\n')
                return text
            def fileno(inner):
                return inner.stream.fileno()
        def racing_open(path, *args, **kwargs):
            stream = original_open(path, *args, **kwargs)
            return OldIntent(stream) if str(path) == str(self.flag) else stream
        with patch.object(app, 'open', side_effect=racing_open, create=True):
            old = self.state()
        self.assertFalse(old['requested'])
        self.assertEqual(old['intent_changed_at'], 1.0)
        self.assertTrue(app.privacy_requested())
        self.assertGreater(self.state()['intent_changed_at'], old['intent_changed_at'])

    def test_resume_readback_failure_repauses_running_guard_and_keeps_latch(self):
        app.set_privacy(True)
        self.prudynt.side_effect = None
        self.prudynt.return_value = {}
        with self.assertRaises(app.PrivacyError):
            app.set_privacy(False)
        self.assertEqual(self.commands[-2:], ['resume', 'pause'])
        self.assertTrue(app.privacy_requested())
        self.assertFalse(self.running)
        self.assertTrue(self.paused)
        self.assertEqual(self.state()['phase'], 'error')
        self.assertTrue(self.state()['confirmed'])
        self.assertEqual(self.state()['mode'], 'capture_stopped')

    def test_resume_failure_with_unverified_rollback_never_claims_confirmation(self):
        app.set_privacy(True)
        original = self.shell
        def fail_pause(command, timeout=0):
            if 'control pause;' in command:
                raise OSError('stop failed')
            return original(command, timeout)
        self.mocks[2].side_effect = fail_pause
        self.prudynt.side_effect = OSError('not ready')
        with self.assertRaises(app.PrivacyError):
            app.set_privacy(False)
        self.assertTrue(app.privacy_requested())
        self.assertFalse(self.state()['confirmed'])
        self.assertEqual(self.state()['phase'], 'error')

    def test_legacy_restart_failure_is_an_error_and_suppression_is_reapplied(self):
        self.installed = False
        app.set_privacy(True)
        self.cgi.side_effect = OSError('restart failed')
        with self.assertRaises(app.PrivacyError):
            app.set_privacy(False)
        self.assertTrue(app.privacy_requested())
        self.assertTrue(self.camera_picture)
        self.assertFalse(self.camera_mic)
        self.assertTrue(self.state()['confirmed'])
        self.assertEqual(self.state()['phase'], 'error')

    def test_non_json_legacy_restart_response_requires_subsequent_readback(self):
        self.installed = False
        app.set_privacy(True)
        self.cgi.side_effect = ValueError('non-JSON CGI body')
        result = app.set_privacy(False)
        self.assertFalse(app.privacy_requested())
        self.assertEqual(result['privacy_status']['phase'], 'active')
        self.assertIsNone(self.prudynt.call_args.args[0]['privacy']['enabled'])

    def test_reconciler_does_not_repause_after_successful_resume(self):
        app.set_privacy(True)
        queued, finished = threading.Event(), threading.Event()
        errors = []
        def reconcile():
            queued.set()
            try:
                self.assertFalse(app.ensure_privacy_paused()['requested'])
            except Exception as exc:
                errors.append(exc)
            finally:
                finished.set()
        worker = None
        def during_resume():
            nonlocal worker
            worker = threading.Thread(target=reconcile)
            worker.start()
            self.assertTrue(queued.wait(2))
            self.assertFalse(finished.is_set(), 'reconciliation must wait for the active resume transaction')
        self.resume_hook = during_resume
        app.set_privacy(False)
        self.assertTrue(finished.wait(2))
        worker.join(2)
        self.assertEqual(errors, [])
        self.assertEqual(self.commands, ['status', 'pause', 'status', 'resume'])
        self.assertFalse(app.privacy_requested())

    def test_transitions_serialize_with_another_process(self):
        code = '''import fcntl,pathlib,sys,time
p=pathlib.Path(sys.argv[1]);f=(p/'privacy-operation.lock').open('a+')
fcntl.flock(f,fcntl.LOCK_EX);(p/'held').touch()
while not (p/'release').exists():time.sleep(.01)
'''
        other = subprocess.Popen([sys.executable, '-c', code, str(self.root)], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.addCleanup(lambda: other.kill() if other.poll() is None else None)
        deadline = time.monotonic() + 3
        while not (self.root / 'held').exists() and time.monotonic() < deadline:
            time.sleep(.01)
        self.assertTrue((self.root / 'held').exists())
        finished, started = threading.Event(), threading.Event()
        errors = []
        def pause():
            started.set()
            try:
                app.set_privacy(True)
            except Exception as exc:
                errors.append(exc)
            finally:
                finished.set()
        worker = threading.Thread(target=pause)
        worker.start()
        self.assertTrue(started.wait(2))
        self.assertFalse(finished.wait(.05))
        self.assertEqual(self.commands, [])
        (self.root / 'release').touch()
        self.assertTrue(finished.wait(3))
        worker.join(2)
        other.communicate(timeout=3)
        self.assertEqual(errors, [])
        self.assertEqual(other.returncode, 0)
        self.assertTrue(app.privacy_requested())

    def test_live_masks_cached_detection_and_reports_unconfirmed_request_immediately(self):
        (self.root / 'live.json').write_text(json.dumps({'status': 'ok', 'box': [.1, .2, .3, .4], 'present': True,
                                                       'score': .9, 'det_t': time.time(), 'audio_ok': True,
                                                       'last_bark': {'t': time.time()}, 'imaging': {'mic_enabled': 1}}))
        self.flag.write_text('1\n')
        live = app.read_live()
        self.assertEqual(live['status'], 'privacy')
        self.assertFalse(live['present'])
        self.assertFalse(live['audio_ok'])
        self.assertIsNone(live['box'])
        self.assertIsNone(live['last_bark'])
        self.assertEqual(live['imaging'], {})
        self.assertTrue(live['privacy']['requested'])
        self.assertFalse(live['privacy']['confirmed'])
        self.assertEqual(self.commands, [])

    def test_confirmed_resume_does_not_leave_ui_stuck_on_old_privacy_status(self):
        (self.root / 'live.json').write_text(json.dumps({'status': 'privacy', 'present': False}))
        app.set_privacy(True)
        app.set_privacy(False)
        live = app.read_live()
        self.assertEqual(live['status'], 'starting')
        self.assertFalse(live['privacy']['requested'])
        self.assertTrue(live['privacy']['confirmed'])

    def test_old_or_invalid_confirmation_is_not_claimed_as_current(self):
        app.set_privacy(True)
        stored = self.root / 'privacy-state.json'
        state = json.loads(stored.read_text())
        state['last_verified_at'] = time.time() - app.PRIVACY_CONFIRM_FRESH - 1
        stored.write_text(json.dumps(state))
        self.assertFalse(self.state()['confirmed'])
        self.assertEqual(self.state()['phase'], 'checking')
        stored.write_text('[]')
        self.assertFalse(self.state()['confirmed'])

    def test_malformed_or_unreadable_local_latch_fails_closed(self):
        for value in ('', 'not-a-state', 'true', '2'):
            self.flag.write_text(value)
            self.assertTrue(app.privacy_requested())
        self.flag.unlink()
        self.flag.mkdir()
        self.assertTrue(app.privacy_requested())

    def test_frames_actions_and_queued_talk_are_blocked_while_requested(self):
        self.flag.write_text('1\n')
        for function, args in [(app.live_frame, ()), (app.request_find, ()), (app.nudge, (1, 1)),
                               (app.goto, (1, 1)), (app.aim, (0, 0)), (app.recalibrate, ()),
                               (app.set_imaging, ('day',)), (app.motor_cmd, ('g', 1, 1)),
                               (app.talk, ('test', 'Test', b'x' * 3000, 'audio/webm'))]:
            with self.subTest(function=function.__name__), self.assertRaises(app.Busy):
                function(*args)
        self.assertEqual(self.commands, [])
        self.prudynt.assert_not_called()
        self.cgi.assert_not_called()

    def test_inflight_frame_cannot_be_returned_after_privacy_was_requested(self):
        class Response:
            def __enter__(inner):
                return inner
            def __exit__(inner, *args):
                return None
            def read(inner, size):
                self.flag.write_text('1\n')
                return b'private jpeg'
        with patch.dict(app._cam, cookie='test'), patch.object(app.urllib.request, 'urlopen', return_value=Response()):
            with self.assertRaises(app.Busy):
                app.live_frame()

    def test_converting_voice_is_rejected_after_a_quick_pause_and_resume(self):
        calls = []
        def convert(*args, **kwargs):
            calls.append(args)
            if len(calls) == 1:
                app._privacy_atomic('privacy', '1\n')
                app._privacy_atomic('privacy', '0\n')
            return SimpleNamespace(stdout='1.0')
        with patch.object(app, 'TALK_DIR', str(self.root / 'talk')), patch.dict(app._talk_last, clear=True), \
                patch.object(subprocess, 'run', side_effect=convert), patch.object(app, 'ensure_chime', return_value='test.wav'), \
                patch.object(app, 'camera_run') as play, patch.object(app.threading, 'Thread'):
            with self.assertRaisesRegex(app.Busy, 'privacy changed'):
                app.talk('test', 'Test', b'x' * 3000, 'audio/webm')
            play.assert_not_called()
        self.assertFalse(app.privacy_requested(), 'the final off flag alone must not authorize old voice')

    def test_delayed_voice_fetch_rejects_the_old_intent_after_pause_and_resume(self):
        token = 'testprivacymessage000001'
        directory = self.root / 'talk'
        directory.mkdir()
        (directory / (token + '.wav')).write_bytes(b'test voice')
        class Handler:
            path = '/' + token + '.wav'
            def __init__(inner):
                inner.wfile = io.BytesIO()
                inner.status = None
            def send_error(inner, status, *args):
                inner.status = status
            def send_response(inner, status):
                inner.status = status
            def send_header(inner, *args):
                pass
            def end_headers(inner):
                pass
        with patch.object(app, 'TALK_DIR', str(directory)), patch.dict(app._talk_intents, {token: app._privacy_intent()}, clear=True):
            current = Handler()
            app.TalkFiles.do_GET(current)
            self.assertEqual(current.status, 200)
            self.assertEqual(current.wfile.getvalue(), b'test voice')
            app._privacy_atomic('privacy', '1\n')
            app._privacy_atomic('privacy', '0\n')
            old = Handler()
            app.TalkFiles.do_GET(old)
            self.assertEqual(old.status, 409)
            self.assertEqual(old.wfile.getvalue(), b'')

    def test_camera_route_returns_failed_confirmation_with_structured_state(self):
        class Handler:
            def send_json(inner, status, body):
                return status, body
        self.mocks[2].side_effect = OSError('unreachable')
        status, body = app.Handler.camera_call(Handler(), app.set_privacy, True)
        self.assertEqual(status, 503)
        self.assertTrue(body['privacy']['requested'])
        self.assertFalse(body['privacy']['confirmed'])
        self.assertIn('error', body)


if __name__ == '__main__':
    unittest.main()
