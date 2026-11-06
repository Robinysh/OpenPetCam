"""Lifecycle and recovery regressions; no server, worker, deploy, or notifications.

Run: python3 tests/server/agent_backlog_test.py
"""
import contextlib
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('PETCAM_CONFIG', str(ROOT / 'config/site.env.example'))
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT)]
import superagent as sa  # noqa: E402
from server import agentdb  # noqa: E402
from server.reportstate import classify  # noqa: E402


def record(i=0, status='failed_checks', reply_status='answered'):
    return {'id': f'20261103-{i:06d}', 'report': {'ts': 100},
            'reply': {'status': reply_status, 'ts': 200, 'fix': {'status': status}}}


class ReportStateTests(unittest.TestCase):
    def test_every_legacy_outcome_needs_evidence_including_live(self):
        for status in ('live', 'no_change', 'duplicate', 'failed_checks', 'replay_worse',
                       'rolled_back', 'error', 'planned', 'redo', 'unrecognized'):
            with self.subTest(status=status):
                row = record(status=status)
                self.assertEqual(classify(row['report'], row['reply'])['state'], 'not_started')

    def test_one_completed_component_does_not_close_remaining_requests(self):
        resolution = {'components': [
            {'id': 'software', 'state': 'done', 'evidence': ['regression passed and installed source verified']},
            {'id': 'hardware', 'state': 'needs_hardware'}]}
        self.assertEqual(classify({}, None, resolution)['state'], 'needs_hardware')
        resolution['components'][1]['state'] = 'done'
        self.assertEqual(classify({}, None, resolution)['state'], 'not_started')
        resolution['components'][1]['evidence'] = {'inspection': 'physical capability verified'}
        self.assertEqual(classify({}, None, resolution)['state'], 'done')

    def test_invalid_or_empty_resolution_is_not_complete(self):
        for resolution in ({}, {'components': []}, {'components': [None]},
                           {'components': [{'id': 'a', 'state': 'done', 'evidence': [False, ' ']}]},
                           {'components': [{'id': 'a', 'state': 'done', 'evidence': ['test']}] * 2}):
            with self.subTest(resolution=resolution):
                self.assertEqual(classify({}, None, resolution)['state'], 'not_started')

    def test_acknowledgement_timeout_and_pending_draft(self):
        self.assertEqual(classify({}, None)['state'], 'not_started')
        self.assertEqual(classify({}, {'status': 'looking', 'ts': 100}, now=200)['state'], 'in_progress')
        self.assertEqual(classify({}, {'status': 'looking', 'ts': 100}, now=5000)['state'], 'not_started')
        self.assertEqual(classify({}, {'status': 'looking', 'ts': float('nan')}, now=200)['state'], 'not_started')
        self.assertEqual(classify({}, record(status='queued')['reply'])['state'], 'in_progress')
        for reply in ([], {}, 'bad'):
            self.assertEqual(classify({}, reply)['state'], 'not_started')

    def test_monitoring_keeps_verified_changes_open_for_observation(self):
        resolution = {'components': [
            {'id': 'fix', 'state': 'done', 'evidence': ['Deployed and checked']},
            {'id': 'follow', 'state': 'monitoring', 'evidence': ['Watch ordinary follow behavior over time']} ]}
        state = classify({}, None, resolution)
        self.assertEqual(state['state'], 'monitoring')
        self.assertEqual(state['label'], 'Monitoring')
        self.assertIn('no immediate review', state['reason'])

    def test_needs_review_requires_a_question_and_material_to_inspect(self):
        component = {'id': 'choice', 'state': 'needs_review', 'evidence': ['An implementation exists']}
        for review in (None, {}, {'question': 'Approve this?'},
                       {'question': ' ', 'evidence': ['clip.mp4']},
                       {'question': 'Approve this?', 'evidence': []},
                       {'question': 'Approve this?', 'evidence': {'path': 'clip.mp4'}}):
            with self.subTest(review=review):
                component['review'] = review
                self.assertEqual(classify({}, None, {'components': [component]})['state'], 'not_started')
        component['review'] = {'question': 'Use this proposed framing?', 'evidence': ['before-after.mp4']}
        state = classify({}, None, {'components': [component]})
        self.assertEqual(state['state'], 'needs_review')
        self.assertEqual(state['reason'], 'Use this proposed framing?')

    def test_pending_work_does_not_disappear_into_monitoring(self):
        resolution = {'components': [
            {'id': 'behavior', 'state': 'monitoring'},
            {'id': 'implementation', 'state': 'not_started'}]}
        self.assertEqual(classify({}, None, resolution)['state'], 'not_started')
        resolution['components'][1]['state'] = 'in_progress'
        self.assertEqual(classify({}, None, resolution)['state'], 'in_progress')
        resolution['components'][1]['state'] = 'needs_hardware'
        self.assertEqual(classify({}, None, resolution)['state'], 'needs_hardware')

    def test_malformed_states_and_fix_outcomes_stay_backlog(self):
        for bad in ([], {}, None, 3):
            with self.subTest(value=bad):
                self.assertEqual(classify({}, None, {'components': [{'id': 'a', 'state': bad}]})['state'], 'not_started')
                self.assertEqual(classify({}, {'fix': {'status': bad}})['state'], 'not_started')

    def test_typed_reply_review_is_actionable_without_a_full_resolution(self):
        review = {'question': 'Use the proposed wider follow framing?', 'evidence': ['before-after.mp4']}
        for status in ('live', 'no_change', 'planned'):
            with self.subTest(status=status):
                reply = {'status': 'answered', 'fix': {'status': status, 'review': review}}
                result = classify({}, reply)
                self.assertEqual(result['state'], 'needs_review')
                self.assertEqual(result['reason'], review['question'])

    def test_freeform_owner_prompt_or_unready_reply_does_not_create_a_review(self):
        review = {'question': 'Use this?', 'evidence': ['before-after.mp4']}
        for status in ('failed_checks', 'replay_worse', 'rolled_back', 'error', 'duplicate', 'unknown'):
            with self.subTest(status=status):
                reply = {'status': 'answered', 'fix': {'status': status, 'review': review}}
                self.assertEqual(classify({}, reply)['state'], 'not_started')
        for status in ('live', 'no_change', 'planned'):
            with self.subTest(prompt_status=status):
                reply = {'status': 'answered', 'fix': {'status': status, 'needs_owner': 'Should we try this?'}}
                self.assertEqual(classify({}, reply)['state'], 'not_started')
        for status in (None, 'error', 'unknown'):
            with self.subTest(reply_status=status):
                reply = {'status': status, 'fix': {'status': 'planned', 'review': review}}
                self.assertEqual(classify({}, reply)['state'], 'not_started')
        for reply in ({'status': 'looking', 'ts': 100, 'fix': {'status': 'planned', 'review': review}},
                      {'status': 'answered', 'fix': {'status': 'queued', 'review': review}}):
            self.assertEqual(classify({}, reply, now=200)['state'], 'in_progress')
        self.assertEqual(classify({}, {'status': 'looking', 'ts': 100,
                                      'fix': {'status': 'planned', 'review': review}}, now=5000)['state'], 'not_started')


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(sa, 'RUN', self.tmp.name))
        self.stack.enter_context(patch.object(sa, 'ARCHIVE', str(Path(self.tmp.name) / 'archive.jsonl')))
        self.conn = sqlite3.connect(':memory:')
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(agentdb.SCHEMA)
        self.addCleanup(self.conn.close)
        self.stack.enter_context(patch.object(sa, 'adb', side_effect=lambda cmd, value=None:
                                  agentdb.COMMANDS[cmd](self.conn, value or {})))

    def test_all_historical_reports_are_reviewable_without_dispatch(self):
        rows = [record(i, 'live' if i % 2 else 'failed_checks') for i in range(60)]
        with patch.object(sa, 'start_worker') as worker, patch.object(sa.ra, 'write_reply') as reply, \
             patch.object(sa.ra, 'notify') as notify, patch.object(sa.ra, 'ship') as ship:
            first = sa.reconcile_reports(rows)
            second = sa.reconcile_reports(rows)
        self.assertEqual((first['reports'], first['unresolved'], first['new'], second['new']), (60, 60, 60, 0))
        self.assertEqual(agentdb.cmd_pending(self.conn, {}), [])
        self.assertEqual(agentdb.cmd_dashboard(self.conn, {})['review_queue'], 60)
        for forbidden in (worker, reply, notify, ship):
            forbidden.assert_not_called()

    def test_duplicate_trigger_does_not_reopen_closed_thread(self):
        sa.reconcile_reports([record()])
        ident = 'report:' + record()['id']
        agentdb.cmd_thread(self.conn, {'id': ident, 'status': 'closed'})
        sa.reconcile_reports([record()])
        self.assertEqual(agentdb.cmd_recall(self.conn, {'ref': ident})['status'], 'closed')

    def test_verified_resolution_and_working_capture_not_requeued(self):
        row = record(status='live')
        row['resolution'] = {'components': [{'id': 'a', 'state': 'done', 'evidence': ['verified source + test']}]}
        self.assertEqual(sa.reconcile_reports([row])['new'], 0)
        self.assertEqual(sa.ra.report_backlog([row])[0]['state'], 'resolved')

    def test_verified_resolution_retires_review_without_consuming_other_work(self):
        row = record()
        sa.reconcile_reports([row])
        thread = 'report:' + row['id']
        agentdb.cmd_add_trigger(self.conn, {'source': 'test', 'kind': 'report', 'thread': thread,
                                          'dedupe': 'new-independent-follow-up'})
        row['resolution'] = {'components': [{'id': 'a', 'state': 'done', 'evidence': ['checked deployment + test']}]}
        sa.reconcile_reports([row])
        self.assertEqual(agentdb.cmd_dashboard(self.conn, {})['review_queue'], 0)
        self.assertEqual(len(agentdb.cmd_pending(self.conn, {})), 1)
        self.assertEqual(agentdb.cmd_recall(self.conn, {'ref': thread})['status'], 'open')

    def test_verified_resolution_does_not_close_active_worker(self):
        row = record()
        sa.reconcile_reports([row])
        thread = 'report:' + row['id']
        agentdb.cmd_thread(self.conn, {'id': thread, 'worker': 'pid:123'})
        row['resolution'] = {'components': [{'id': 'a', 'state': 'done', 'evidence': ['checked deployment + test']}]}
        sa.reconcile_reports([row])
        result = agentdb.cmd_recall(self.conn, {'ref': thread})
        self.assertEqual((result['status'], result['worker']), ('open', 'pid:123'))

    def test_stale_looking_and_malformed_reply_survive_inventory(self):
        old = record(reply_status='looking')
        broken = record(1)
        broken['reply'] = []
        self.assertEqual(sa.reconcile_reports([old, broken])['new'], 2)
        self.assertEqual(sa.ra.report_backlog([old], now=5000)[0]['state'], 'stalled')

    def test_queue_is_not_duplicated_but_missing_worktree_is_recoverable(self):
        row = record(status='queued')
        path = Path(self.tmp.name) / f"queued_report_{row['id']}.json"
        path.write_text(json.dumps({'wt': self.tmp.name}))
        self.assertEqual(sa.reconcile_reports([row])['new'], 0)
        path.write_text(json.dumps({'wt': self.tmp.name + '/gone'}))
        self.assertEqual(sa.reconcile_reports([row])['new'], 1)

    def test_read_only_inventory_never_touches_heartbeat_or_replies(self):
        with patch.object(sa.ra, 'run', return_value=subprocess.CompletedProcess([], 0, '[]', '')) as run:
            self.assertEqual(sa.ra.report_inventory(), [])
        command = run.call_args.args[0]
        script = run.call_args.kwargs['input']
        self.assertEqual(command[-1], 'sudo python3 -')
        self.assertNotIn('.touch(', script)
        self.assertNotIn('write_text', script)

    def test_new_collection_cannot_dispatch_historical_review_queue(self):
        fresh = record(10)
        fresh['reply'] = None
        with patch.object(sa.ra, 'report_inventory', return_value=[fresh, record()]), \
             patch.object(sa, 'read', return_value='{}'):
            sa.collect()
        pending = agentdb.cmd_pending(self.conn, {})
        reports = [row for row in pending if row['kind'] == 'report']
        self.assertEqual(len(reports), 1)
        self.assertEqual(reports[0]['thread'], 'report:' + fresh['id'])
        self.assertFalse(any(row['kind'] == 'report_review' for row in pending))

    def test_previous_agent_result_cannot_stand_in_for_new_attempt(self):
        path = Path(self.tmp.name) / 'agent_result.json'
        path.write_text(json.dumps({'reply': 'old success'}))
        with patch.object(sa.ra, 'run', return_value=subprocess.CompletedProcess([], 0, '{}', '')):
            with self.assertRaisesRegex(RuntimeError, 'agent failed'):
                sa.ra.run_agent(record()['id'], self.tmp.name, self.tmp.name, {})
        self.assertFalse(path.exists())
        self.assertEqual(len(list(Path(self.tmp.name).glob('agent_result.previous-*.json'))), 1)

    def test_report_worker_records_actual_outcome_and_keeps_unfinished_thread_open(self):
        for status in ('live', 'queued', 'failed_checks', 'no_change'):
            with self.subTest(status=status):
                thread = 'report:' + record()['id']
                brief = Path(self.tmp.name) / f"brief_{thread.replace(':', '_')}.json"
                brief.write_text(json.dumps({'kind': 'report'}))
                with patch.object(sa, 'beat'), patch.object(sa.ra, 'save_state'), \
                     patch.object(sa.ra, 'load_state', return_value={'pending': {}, 'deploys': {}}), \
                     patch.object(sa.ra, 'handle', return_value={'status': status, 'summary': 'tested diagnosis'}):
                    sa.work(thread)
                thread_row = agentdb.cmd_recall(self.conn, {'ref': thread})
                self.assertEqual(thread_row['status'], 'closed' if status == 'live' else 'open')
                self.assertTrue(thread_row['summary'].startswith(status + ':'))
                archive = [json.loads(line) for line in Path(sa.ARCHIVE).read_text().splitlines()]
                self.assertEqual((archive[-1]['kind'], archive[-1]['verdict']), ('report', status))

    def test_report_worker_error_preserves_pending_draft_and_creates_review(self):
        thread = 'report:' + record()['id']
        brief = Path(self.tmp.name) / f"brief_{thread.replace(':', '_')}.json"
        brief.write_text(json.dumps({'kind': 'report'}))
        state = {'pending': {record()['id']: 'saved-draft'}, 'deploys': {}}
        with patch.object(sa, 'beat'), patch.object(sa, 'log'), patch.object(sa, 'note'), \
             patch.object(sa.ra, 'save_state') as save, patch.object(sa.ra, 'load_state', return_value=state), \
             patch.object(sa.ra, 'handle', side_effect=RuntimeError('reply interrupted')), \
             patch.object(sa.ra, 'write_reply') as reply, patch.object(sa.ra, 'notify') as notify:
            sa.work(thread)
        save.assert_called_once_with(state)
        reply.assert_not_called()
        notify.assert_not_called()
        self.assertEqual(agentdb.cmd_recall(self.conn, {'ref': thread})['status'], 'open')
        self.assertEqual(agentdb.cmd_dashboard(self.conn, {})['review_queue'], 1)
        self.assertEqual(json.loads(Path(sa.ARCHIVE).read_text())['verdict'], 'error')


class QueueTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(sa, 'RUN', self.tmp.name))
        self.stack.enter_context(patch.object(sa, 'ARCHIVE', str(Path(self.tmp.name) / 'archive.jsonl')))
        self.stack.enter_context(patch.object(sa, 'running_workers', return_value=[]))
        self.stack.enter_context(patch.object(sa.ra, 'git', return_value=''))
        self.stack.enter_context(patch.object(sa.ra, 'prepare_candidate',
            return_value=('ready', '', {'main': 'base', 'candidate': '123abcd'})))
        self.stack.enter_context(patch.object(sa.ra, 'load_state', return_value={'deploys': {}}))
        self.stack.enter_context(patch.object(sa.ra, 'save_state'))
        self.stack.enter_context(patch.object(sa, 'note'))
        self.stack.enter_context(patch.object(sa, 'offer_upstream', return_value='skipped'))
        self.stack.enter_context(patch.object(sa, 'adb', return_value={'id': 1, 'new': True}))
        self.stack.enter_context(patch.object(sa.subprocess, 'run', return_value=subprocess.CompletedProcess([], 0, '', '')))
        self.q = {'thread': 'goal:test', 'wt': self.tmp.name, 'branch': 'agent/test', 'commit': '123abcd', 'queued_at': 10}
        self.path = Path(self.tmp.name) / 'queued_goal_test.json'
        self.path.write_text(json.dumps(self.q))

    def test_replay_exception_keeps_queue_and_draft(self):
        with patch.object(sa, 'replay_compare', side_effect=RuntimeError('test interruption')):
            with self.assertRaisesRegex(RuntimeError, 'interruption'):
                sa.retry_queued()
        self.assertEqual(json.loads(self.path.read_text()), self.q)

    def test_queue_result_checkpoint_prevents_redeploy_after_interruption(self):
        with patch.object(sa, 'replay_compare', return_value={'ok': True}), \
             patch.object(sa.ra, 'ship', return_value=('live', 'verified')) as ship, \
             patch.object(sa, 'note', side_effect=RuntimeError('test interruption')):
            with self.assertRaises(RuntimeError):
                sa.retry_queued()
            ship.assert_called_once()
        self.assertEqual(json.loads(self.path.read_text())['result']['status'], 'live')
        with patch.object(sa, 'replay_compare') as replay, patch.object(sa.ra, 'ship') as ship:
            sa.retry_queued()
        ship.assert_not_called()
        replay.assert_not_called()
        self.assertFalse(self.path.exists())

    def test_missing_worktree_keeps_queue_evidence(self):
        self.q['wt'] = self.tmp.name + '/missing'
        self.path.write_text(json.dumps(self.q))
        with patch.object(sa, 'replay_compare') as replay:
            sa.retry_queued()
        replay.assert_not_called()
        self.assertTrue(self.path.exists())
        self.assertEqual(sa.adb.call_args.args[1]['status'], 'review')

    def test_dirty_checkout_is_not_changed_or_replayed(self):
        with patch.object(sa.ra, 'git', return_value=' M user_work.py'), patch.object(sa, 'replay_compare') as replay:
            sa.retry_queued()
        replay.assert_not_called()
        self.assertTrue(self.path.exists())

    def test_preparation_precedes_replay_and_shipping_uses_validated_hashes(self):
        order = []
        binding = {'main': 'newbase', 'candidate': 'rebasedcommit'}
        def prepare(*args):
            order.append('prepare')
            return 'ready', '', binding
        def replay(*args):
            order.append('replay')
            return {'ok': True}
        def ship(*args, **kwargs):
            order.append('ship')
            self.assertEqual(kwargs['expected'], binding)
            return 'live', 'verified'
        with patch.object(sa.ra, 'prepare_candidate', side_effect=prepare), \
             patch.object(sa, 'replay_compare', side_effect=replay), patch.object(sa.ra, 'ship', side_effect=ship):
            sa.retry_queued()
        self.assertEqual(order, ['prepare', 'replay', 'ship'])
        saved = json.loads(Path(sa.ARCHIVE).read_text())
        self.assertEqual(saved['commit'], 'rebasedcommit')
        self.assertEqual(saved['replay']['validated_heads'], binding)


class ShipBindingTests(unittest.TestCase):
    def test_changed_head_or_dirty_worktree_prevents_shipping(self):
        expected = {'main': 'base', 'candidate': 'candidate'}
        for changed in ('main', 'candidate', 'dirty'):
            def git(*args, **kwargs):
                if args[:2] == ('status', '--porcelain'):
                    return ' M draft.py' if changed == 'dirty' and kwargs.get('cwd') else ''
                if args == ('rev-parse', 'HEAD'):
                    return 'candidate' if kwargs.get('cwd') else ('new-base' if changed == 'main' else 'base')
                if args == ('rev-parse', 'branch'):
                    return 'changed' if changed == 'candidate' else 'candidate'
                if args[0] == 'merge-base':
                    return 'base'
                raise AssertionError(f'Unexpected mutation: {args}')
            with self.subTest(changed=changed), patch.object(sa.ra, 'git', side_effect=git), \
                 patch.object(sa.ra, 'run') as run, patch.object(sa.ra, 'prepare_candidate') as prepare:
                result = sa.ra.ship('report', 'worktree', 'branch', {'deploys': {}}, expected=expected)
            self.assertEqual(result[0], 'queued')
            run.assert_not_called()
            prepare.assert_not_called()  # no rebase after validating another tree

    def test_ship_merges_exact_validated_object(self):
        calls = []
        def git(*args, **kwargs):
            calls.append(args)
            if args[0] == 'status':
                return ''
            if args[0] == 'merge-base':
                return 'base'
            if args[0] == 'rev-parse':
                return 'candidate' if args[1] == 'branch' or kwargs.get('cwd') else 'base'
            if args == ('merge', '--ff-only', 'candidate'):
                return ''
            raise AssertionError(args)
        with patch.object(sa.ra, 'git', side_effect=git), patch.object(sa.ra, 'DRY', False), \
             patch.object(sa.ra, 'run', return_value=subprocess.CompletedProcess([], 0, '', '')), \
             patch.object(sa.ra, 'healthy', return_value=(True, 'healthy')):
            result = sa.ra.ship('report', 'worktree', 'branch', {'deploys': {}},
                                expected={'main': 'base', 'candidate': 'candidate'})
        self.assertEqual(result[0], 'live')
        self.assertIn(('merge', '--ff-only', 'candidate'), calls)

    def test_prepare_rebases_before_returning_new_candidate_binding(self):
        rebased = False
        def git(*args, **kwargs):
            if args[0] == 'status':
                return ''
            if args == ('rev-parse', 'HEAD'):
                return ('rebased' if rebased else 'old') if kwargs.get('cwd') else 'base'
            if args[0] == 'merge-base':
                return 'old-base'
            if args == ('rev-parse', 'branch'):
                return 'rebased'
            raise AssertionError(args)
        def run(cmd, **kwargs):
            nonlocal rebased
            self.assertEqual(cmd, ['git', 'rebase', 'base'])
            rebased = True
            return subprocess.CompletedProcess(cmd, 0, '', '')
        with patch.object(sa.ra, 'git', side_effect=git), patch.object(sa.ra, 'run', side_effect=run), \
             patch.object(sa.ra, 'gates', return_value=(True, 'passed')) as gates:
            state, _, binding = sa.ra.prepare_candidate('worktree', 'branch')
        self.assertEqual((state, binding), ('ready', {'main': 'base', 'candidate': 'rebased'}))
        gates.assert_called_once_with('worktree')


if __name__ == '__main__':
    unittest.main()
