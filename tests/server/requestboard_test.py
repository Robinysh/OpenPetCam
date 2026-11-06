"""Workflow moves persist without replacing investigation or verification evidence."""
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
import threading
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('PETCAM_SITE_ENV', str(ROOT / 'config/site.env.example'))
sys.path.insert(0, str(ROOT / 'server'))
import app
import requestboard


class RequestBoardTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.ident = '20261105-000001'
        self.directory = self.root / self.ident
        self.directory.mkdir()
        self.report = {'id': self.ident, 'ts': 1, 'by': 'Test device', 'note': 'Follow a moving pet',
                       'client': {'shown_status': 'Following'}, 'obs_15min': [{'large': 'snapshot'}]}
        self.reply = {'status': 'answered', 'text': 'A change was proposed', 'fix': {'status': 'queued'}}
        self.resolution = {'components': [
            {'id': 'movement', 'title': 'Follow movement', 'state': 'done', 'evidence': ['A replay passed']},
            {'id': 'retention', 'title': 'Retain the target', 'state': 'needs_review', 'evidence': []}]}
        for name in ('report', 'reply', 'resolution'):
            self.write(name, getattr(self, name))

    def write(self, name, data):
        (self.directory / (name + '.json')).write_text(json.dumps(data))

    def row(self):
        return requestboard.load(self.root, self.ident)

    def move(self, state, reason='', revision=None):
        return requestboard.move(self.root, self.ident, state, revision or self.row()['revision'],
                                 'Test device', reason)

    def test_api_rows_supply_revisions_for_safe_moves(self):
        with patch.object(app, 'REPORTS_DIR', str(self.root)):
            row = app.report_page()['reports'][0]
        self.assertRegex(row.get('revision', ''), r'^[0-9a-f]{64}$')
        self.assertEqual(row['verified_status']['state'], 'not_started')

    def test_move_survives_reload_and_keeps_verification_separate(self):
        row = self.move('in_progress')
        self.assertEqual(row['status']['state'], 'in_progress')
        self.assertEqual(self.row()['status']['state'], 'in_progress')
        self.assertEqual(row['verified_status']['state'], 'not_started')
        self.assertEqual(row['workflow']['actor'], 'Test device')
        self.assertTrue(row['workflow']['updated_at'] > 0)

    def test_evidence_and_original_request_are_byte_identical_after_a_move(self):
        paths = [self.directory / (n + '.json') for n in ('report', 'reply', 'resolution')]
        before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
        self.move('blocked', 'A physical measurement is needed')
        self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths})
        self.assertEqual(self.row()['status']['reason'], 'A physical measurement is needed')

    def test_done_requires_every_component_and_evidence(self):
        with self.assertRaisesRegex(ValueError, 'verification evidence'):
            self.move('done')
        self.resolution['components'][1]['state'] = 'done'
        self.write('resolution', self.resolution)
        with self.assertRaisesRegex(ValueError, 'verification evidence'):
            self.move('done')
        self.resolution['components'][1]['evidence'] = ['Confirmed in a separate live sequence']
        self.write('resolution', self.resolution)
        self.assertEqual(self.move('done')['status']['state'], 'done')

    def test_legacy_live_reply_is_not_completion_evidence(self):
        (self.directory / 'resolution.json').unlink()
        self.reply['fix']['status'] = 'live'
        self.write('reply', self.reply)
        with self.assertRaisesRegex(ValueError, 'verification evidence'):
            self.move('done')

    def test_verified_request_can_be_reopened_without_erasing_its_evidence(self):
        self.resolution['components'] = self.resolution['components'][:1]
        self.write('resolution', self.resolution)
        row = self.move('monitoring', 'Watch for the same symptom in normal use')
        self.assertEqual(row['status']['state'], 'monitoring')
        self.assertEqual(row['verified_status']['state'], 'done')
        self.assertEqual(row['resolution'], self.resolution)

    def test_reopened_done_stays_open_when_only_the_reply_changes(self):
        self.resolution['components'] = self.resolution['components'][:1]
        self.write('resolution', self.resolution)
        for index, state in enumerate(('monitoring', 'in_progress')):
            with self.subTest(state=state):
                reopened = self.move(state, 'The symptom returned')
                self.reply = {'status': 'looking', 'ts': 9999999999 + index}
                self.write('reply', self.reply)
                row = self.row()
                self.assertEqual(row['status']['state'], state,
                                 'A new investigation reply must not reuse old evidence to close a reopened request.')
                self.assertEqual(row['verified_status']['state'], 'done')
                self.assertIsNotNone(row['workflow'])
                self.assertEqual(row['resolution'], self.resolution)
                self.assertNotEqual(row['revision'], reopened['revision'])
                with self.assertRaises(requestboard.Conflict):
                    self.move('not_started', revision=reopened['revision'])

    def test_new_resolution_supersedes_reopened_done(self):
        self.resolution['components'] = self.resolution['components'][:1]
        self.write('resolution', self.resolution)
        self.move('monitoring', 'Watch the deployed behavior in normal use')
        self.reply = {'status': 'looking', 'ts': 9999999999}
        self.write('reply', self.reply)
        self.assertEqual(self.row()['status']['state'], 'monitoring')
        self.resolution['components'][0]['evidence'].append('The repeated symptom was independently checked again')
        self.write('resolution', self.resolution)
        row = self.row()
        self.assertEqual(row['status']['state'], 'done')
        self.assertIsNone(row['workflow'])

    def test_reply_update_still_supersedes_normal_in_progress_move(self):
        self.move('in_progress')
        self.reply = {'status': 'looking', 'ts': 9999999999}
        self.write('reply', self.reply)
        row = self.row()
        self.assertEqual(row['status']['state'], 'not_started')
        self.assertIsNone(row['workflow'])

    def test_resolution_update_still_supersedes_normal_in_progress_move(self):
        self.move('in_progress')
        self.resolution['components'] = self.resolution['components'][:1]
        self.write('resolution', self.resolution)
        row = self.row()
        self.assertEqual(row['status']['state'], 'done')
        self.assertIsNone(row['workflow'])

    def test_monitoring_is_persistent_without_claiming_completion(self):
        row = self.move('monitoring', 'Keep an eye on follow during ordinary use')
        self.assertEqual(row['status']['state'], 'monitoring')
        self.assertEqual(self.row()['status']['state'], 'monitoring')
        self.assertEqual(row['verified_status']['state'], 'not_started')
        self.assertEqual(row['resolution'], self.resolution)
        with self.assertRaisesRegex(ValueError, 'verification evidence'):
            self.move('done')

    def test_manual_review_needs_a_concrete_decision_prompt(self):
        with self.assertRaisesRegex(ValueError, 'review or decide'):
            self.move('needs_review')
        row = self.move('needs_review', 'Does the retained before/after clip show acceptable follow?')
        self.assertEqual(row['status']['state'], 'needs_review')
        self.assertEqual(row['status']['reason'], 'Does the retained before/after clip show acceptable follow?')
        self.assertEqual(row['verified_status']['state'], 'not_started')

    def test_legacy_review_move_without_a_decision_does_not_fill_review_queue(self):
        self.move('in_progress')
        workflow = json.loads((self.directory / 'workflow.json').read_text())
        workflow.update(state='needs_review', reason='')
        self.write('workflow', workflow)
        row = self.row()
        self.assertEqual(row['status']['state'], 'not_started')
        self.assertIsNone(row['workflow'])

    def test_two_owners_cannot_overwrite_one_another(self):
        revision = self.row()['revision']
        self.move('in_progress', revision=revision)
        with self.assertRaises(requestboard.Conflict):
            self.move('not_started', revision=revision)
        self.assertEqual(self.row()['status']['state'], 'in_progress')

    def test_simultaneous_moves_have_one_winner(self):
        revision = self.row()['revision']
        ready = threading.Barrier(2)

        def attempt(state):
            ready.wait(timeout=2)
            try:
                return requestboard.move(self.root, self.ident, state, revision, 'Test device')['status']['state']
            except requestboard.Conflict:
                return 'conflict'

        with ThreadPoolExecutor(max_workers=2) as pool:
            outcomes = list(pool.map(attempt, ['in_progress', 'not_started']))
        self.assertEqual(outcomes.count('conflict'), 1)
        self.assertIn(self.row()['status']['state'], outcomes)

    def test_failed_atomic_replace_preserves_last_move(self):
        self.move('in_progress')
        path = self.directory / 'workflow.json'
        before = path.read_bytes()
        with patch.object(requestboard.os, 'replace', side_effect=OSError('disk write failed')):
            with self.assertRaises(OSError):
                self.move('not_started')
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(self.row()['status']['state'], 'in_progress')
        self.assertFalse(list(self.directory.glob('.workflow-*')))

    def test_new_reply_invalidates_a_previous_manual_lane_and_snapshot(self):
        old = self.move('blocked', 'Waiting for a measurement')
        self.reply['fix']['status'] = 'failed_checks'
        self.write('reply', self.reply)
        row = self.row()
        self.assertIsNone(row['workflow'])
        self.assertEqual(row['status']['state'], 'not_started')
        self.assertNotEqual(row['revision'], old['revision'])
        with self.assertRaises(requestboard.Conflict):
            self.move('in_progress', revision=old['revision'])

    def test_new_resolution_can_complete_a_previously_blocked_request(self):
        self.move('blocked', 'Waiting for a measurement')
        self.resolution['components'] = self.resolution['components'][:1]
        self.write('resolution', self.resolution)
        self.assertEqual(self.row()['status']['state'], 'done')
        self.assertIsNone(self.row()['workflow'])

    def test_large_observation_changes_are_not_workflow_changes(self):
        revision = self.move('in_progress')['revision']
        self.report['obs_15min'].append({'another': 'frame'})
        self.write('report', self.report)
        self.assertEqual(self.row()['revision'], revision)
        self.assertEqual(self.row()['status']['state'], 'in_progress')

    def test_malformed_workflow_cannot_bypass_done_gate(self):
        self.move('in_progress')
        workflow = json.loads((self.directory / 'workflow.json').read_text())
        workflow['state'] = 'done'
        self.write('workflow', workflow)
        self.assertEqual(self.row()['status']['state'], 'not_started')
        self.assertIsNone(self.row()['workflow'])
        self.write('workflow', ['invalid shape'])
        self.assertEqual(self.row()['status']['state'], 'not_started')

    def test_unhashable_workflow_state_does_not_hide_request_history(self):
        self.move('in_progress')
        workflow = json.loads((self.directory / 'workflow.json').read_text())
        for invalid in ([], {'state': 'done'}):
            with self.subTest(state=invalid):
                workflow['state'] = invalid
                self.write('workflow', workflow)
                try:
                    row = self.row()
                except TypeError:
                    self.fail('A malformed workflow state must not crash the request history.')
                self.assertEqual(row['status']['state'], 'not_started')
                self.assertIsNone(row['workflow'])

    def test_unhashable_component_state_keeps_request_visible_and_unverified(self):
        for invalid in ([], {'state': 'done'}):
            with self.subTest(state=invalid):
                self.resolution['components'][1]['state'] = invalid
                self.write('resolution', self.resolution)
                row = self.row()
                self.assertEqual(row['status']['state'], 'not_started')
                self.assertIn('malformed', row['status']['reason'])
                with self.assertRaisesRegex(ValueError, 'verification evidence'):
                    self.move('done')

    def test_blocked_reason_and_move_validation_fail_without_writing(self):
        for state, reason, revision in [('blocked', '', self.row()['revision']),
                                        ('invalid', '', self.row()['revision']),
                                        ('in_progress', 'x' * 1001, self.row()['revision']),
                                        ('in_progress', '', ''),
                                        ('in_progress', None, self.row()['revision'])]:
            with self.subTest(state=state, reason=reason, revision=revision):
                with self.assertRaises(ValueError):
                    requestboard.move(self.root, self.ident, state, revision, 'Test device', reason)
        self.assertFalse((self.directory / 'workflow.json').exists())

    def test_path_traversal_and_symlink_escape_are_rejected(self):
        for ident in ('../20261105-000001', self.ident + '/..', '/tmp/outside', '２０２６１１０５-000001'):
            with self.subTest(ident=ident):
                with self.assertRaises(ValueError):
                    requestboard.load(self.root, ident)
        with tempfile.TemporaryDirectory() as outside:
            (self.root / '20261105-000002').symlink_to(outside, target_is_directory=True)
            with self.assertRaises(ValueError):
                requestboard.load(self.root, '20261105-000002')
        with self.assertRaises(FileNotFoundError):
            requestboard.load(self.root, '20261105-000003')

    def test_move_history_is_bounded_and_temporary_files_are_not_left_behind(self):
        for n in range(105):
            self.move('in_progress' if n % 2 else 'not_started')
        row = self.row()
        self.assertEqual(len(row['workflow']['history']), 100)
        self.assertEqual(row['workflow']['history'][-1]['state'], row['status']['state'])
        self.assertFalse(list(self.directory.glob('.workflow-*')))


if __name__ == '__main__':
    unittest.main()
