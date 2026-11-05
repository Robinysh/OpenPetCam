"""The directly shipped and shadow-tested code must be the exact compared code.

All subprocesses, deployment, remote calls and model calls are replaced with local
fixtures; tests exercise lifecycle transitions and evidence bindings, not models.
"""
import contextlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('PETCAM_CONFIG', str(ROOT / 'config/site.env.example'))
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT)]
import superagent as sa  # noqa: E402


class ValidationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.run = Path(self.tmp.name)
        self.wt = self.run / 'worktree'
        self.wt.mkdir()
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.binding = {'main': 'tested-base', 'candidate': 'tested-candidate'}
        patches = [
            patch.object(sa, 'RUN', str(self.run)), patch.object(sa, 'ARCHIVE', str(self.run / 'archive.jsonl')),
            patch.object(sa, 'note'), patch.object(sa, 'adb', return_value={'id': 1, 'new': True}),
            patch.object(sa, 'offer_upstream', return_value='skipped'),
            patch.object(sa, 'replay_compare', return_value={'ok': True}),
            patch.object(sa, 'touches_detection', return_value=False),
            patch.object(sa.ra, 'git', return_value=''),
            patch.object(sa.ra, 'prepare_candidate', return_value=('ready', '', self.binding)),
            patch.object(sa.ra, 'validation_matches', return_value=True),
            patch.object(sa.ra, 'load_state', return_value={'deploys': {}}), patch.object(sa.ra, 'save_state'),
            patch.object(sa.ra, 'ship', return_value=('live', 'healthy')),
            patch.object(sa.subprocess, 'run', side_effect=self.process),
        ]
        for entry in patches:
            self.stack.enter_context(entry)

    @staticmethod
    def process(cmd, **kwargs):
        if 'pipeline' in cmd:
            return subprocess.CompletedProcess(cmd, 0, json.dumps({'001.jpg': None}), '')
        return subprocess.CompletedProcess(cmd, 0, b'', b'')

    def shadow(self, binding=True):
        frames = self.run / 'old-frames'
        frames.mkdir()
        (frames / '001.jpg').write_bytes(b'fixture: no actual image decoding in lifecycle test')
        (frames / 'labels.json').write_text('{}')
        job = {'thread': 'goal:test', 'wt': str(self.wt), 'branch': 'agent/test',
               'commit': self.binding['candidate'], 'started': time.time() - 7 * 3600,
               'since': time.time() - 3600, 'frames_dir': str(frames), 'frames': 999, 'disagree': ['old.jpg']}
        if binding:
            job['validated_heads'] = self.binding
        path = self.run / 'shadow_goal_test.json'
        path.write_text(json.dumps(job))
        return path, job

    def test_direct_rebase_then_replay_then_ship_exact_binding(self):
        order = []
        def prepare(*args):
            order.append('prepare')
            return 'ready', '', self.binding
        def replay(*args):
            order.append('replay')
            return {'ok': True}
        def ship(*args, **kwargs):
            order.append('ship')
            self.assertEqual(kwargs['expected'], self.binding)
            return 'live', 'healthy'
        result = {}
        with patch.object(sa.ra, 'prepare_candidate', side_effect=prepare), \
             patch.object(sa, 'replay_compare', side_effect=replay), patch.object(sa.ra, 'ship', side_effect=ship):
            status, _, commit = sa.judge_improvement('goal:test', str(self.wt), 'agent/test', result)
        self.assertEqual(order, ['prepare', 'replay', 'ship'])
        self.assertEqual((status, commit), ('live', self.binding['candidate']))
        self.assertEqual(result['replay']['validated_heads'], self.binding)

    def test_direct_mutation_during_replay_prevents_shipping(self):
        sa.ra.validation_matches.return_value = False
        result = {}
        status, _, _ = sa.judge_improvement('goal:test', str(self.wt), 'agent/test', result)
        self.assertEqual(status, 'queued')
        self.assertFalse(result['replay']['ok'])
        sa.ra.ship.assert_not_called()

    def test_dirty_main_does_not_start_replay_or_drop_shadow_requirement(self):
        sa.ra.prepare_candidate.return_value = ('queued', 'main has WIP', None)
        sa.touches_detection.return_value = True
        result = {}
        status, _, _ = sa.judge_improvement('goal:test', str(self.wt), 'agent/test', result)
        self.assertEqual(status, 'queued')
        self.assertTrue(result['requires_shadow'])
        sa.replay_compare.assert_not_called()
        sa.ra.ship.assert_not_called()

    def test_detection_improvement_records_bound_shadow_instead_of_shipping(self):
        sa.touches_detection.return_value = True
        result = {}
        status, _, _ = sa.judge_improvement('goal:test', str(self.wt), 'agent/test', result)
        job = json.loads((self.run / 'shadow_goal_test.json').read_text())
        self.assertEqual(status, 'shadow')
        self.assertEqual(job['validated_heads'], self.binding)
        self.assertEqual(job['frames'], 0)
        sa.ra.ship.assert_not_called()

    def test_legacy_shadow_cannot_inherit_old_elapsed_time_or_frames(self):
        path, old = self.shadow(binding=False)
        sa.ra.validation_matches.side_effect = lambda wt, branch, binding: bool(binding)
        sa.shadow_step()
        fresh = json.loads(path.read_text())
        self.assertGreater(fresh['started'], old['started'] + 6 * 3600)
        self.assertEqual((fresh['frames'], fresh['disagree']), (0, []))
        self.assertNotEqual(fresh['frames_dir'], old['frames_dir'])
        self.assertTrue(Path(old['frames_dir'], '001.jpg').exists())
        self.assertEqual(fresh['previous_rounds'][0]['frames_dir'], old['frames_dir'])
        sa.subprocess.run.assert_not_called()
        sa.ra.ship.assert_not_called()

    def test_changed_bound_shadow_also_restarts_after_new_replay(self):
        path, old = self.shadow()
        new_binding = {'main': 'new-base', 'candidate': 'rebased-candidate'}
        sa.ra.validation_matches.side_effect = lambda wt, branch, binding: binding == new_binding
        sa.ra.prepare_candidate.return_value = ('ready', '', new_binding)
        sa.shadow_step()
        fresh = json.loads(path.read_text())
        self.assertEqual(fresh['validated_heads'], new_binding)
        self.assertEqual(fresh['commit'], 'rebased-candidate')
        self.assertGreater(fresh['started'], old['started'])
        sa.ra.ship.assert_not_called()

    def test_shadow_with_dirty_code_holds_without_remote_reads(self):
        path, old = self.shadow()
        sa.ra.validation_matches.return_value = False
        sa.ra.prepare_candidate.return_value = ('queued', 'main has WIP', None)
        sa.shadow_step()
        held = json.loads(path.read_text())
        self.assertEqual(held['started'], old['started'])
        self.assertEqual(held['hold_reason'], 'main has WIP')
        sa.replay_compare.assert_not_called()
        sa.subprocess.run.assert_not_called()
        sa.ra.ship.assert_not_called()

    def test_changed_during_shadow_votes_cannot_reach_adjudication_or_ship(self):
        path, _ = self.shadow()
        sa.ra.validation_matches.side_effect = [True, False]
        with patch.object(sa, 'adjudicate') as judge:
            sa.shadow_step()
        judge.assert_not_called()
        sa.ra.ship.assert_not_called()
        self.assertTrue(json.loads(path.read_text())['validation_invalidated'])

    def test_shadow_ship_uses_exact_binding_and_only_image_frames(self):
        path, _ = self.shadow()
        with patch.object(sa, 'SHADOW_MIN_FRAMES', 1):
            sa.shadow_step()
        self.assertFalse(path.exists())
        self.assertEqual(sa.ra.ship.call_args.kwargs['expected'], self.binding)
        archived = json.loads(Path(sa.ARCHIVE).read_text())
        self.assertEqual(archived['validated_heads'], self.binding)
        self.assertEqual(archived['frames'], 1)  # labels.json is not another camera frame

    def test_queued_shadow_keeps_job_worktree_and_branch(self):
        for changed in (False, True):
            path = self.run / 'shadow_goal_test.json'
            if not path.exists():
                self.shadow()
            sa.ra.ship.return_value = ('queued', 'main changed' if changed else 'daily cap; wait tomorrow')
            sa.ra.validation_matches.side_effect = [True, True, not changed]
            sa.subprocess.run.reset_mock()
            with patch.object(sa, 'SHADOW_MIN_FRAMES', 1):
                sa.shadow_step()
            self.assertTrue(path.exists())
            self.assertEqual(json.loads(path.read_text())['validation_invalidated'], changed)
            self.assertFalse(any(call.args[0][0] == 'git' for call in sa.subprocess.run.call_args_list))

    def test_shadow_terminal_checkpoint_prevents_redeploy_after_log_failure(self):
        path, _ = self.shadow()
        with patch.object(sa, 'SHADOW_MIN_FRAMES', 1), patch.object(sa, 'note', side_effect=RuntimeError('log down')):
            with self.assertRaisesRegex(RuntimeError, 'log down'):
                sa.shadow_step()
        self.assertEqual(json.loads(path.read_text())['result']['status'], 'live')
        sa.ra.ship.reset_mock()
        sa.ra.validation_matches.reset_mock()
        sa.shadow_step()
        sa.ra.ship.assert_not_called()
        sa.ra.validation_matches.assert_not_called()
        self.assertFalse(path.exists())

    def test_queued_detection_fix_cannot_bypass_shadow_after_rebase(self):
        queued = self.run / 'queued_goal_test.json'
        queued.write_text(json.dumps({'thread': 'goal:test', 'wt': str(self.wt), 'branch': 'agent/test',
                                      'commit': 'old-candidate', 'requires_shadow': True}))
        with patch.object(sa, 'running_workers', return_value=[]):
            sa.retry_queued()
        self.assertFalse(queued.exists())
        self.assertTrue((self.run / 'shadow_goal_test.json').exists())
        sa.ra.ship.assert_not_called()
        self.assertFalse(any(call.args[0][0] == 'git' for call in sa.subprocess.run.call_args_list))


class RealGitBindingTests(unittest.TestCase):
    """Use real isolated commits/rebases; the only stub is the unrelated project gate."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.repo = Path(self.tmp.name) / 'repo'
        self.wt = Path(self.tmp.name) / 'candidate'
        subprocess.run(['git', 'init', '--quiet', '-b', 'main', str(self.repo)], check=True, capture_output=True)
        self.git('config', 'user.name', 'Fixture')
        self.git('config', 'user.email', 'fixture@example.invalid')
        (self.repo / 'base.txt').write_text('base\n')
        self.git('add', 'base.txt')
        self.git('commit', '--quiet', '-m', 'base')
        self.git('worktree', 'add', '--quiet', '-b', 'candidate', str(self.wt), 'HEAD')

    def git(self, *args, cwd=None, check=True, timeout=60):
        result = subprocess.run(['git', *args], cwd=cwd or self.repo, capture_output=True, text=True, timeout=timeout)
        if check and result.returncode:
            raise AssertionError(result.stderr)
        return result.stdout.strip()

    @staticmethod
    def command(cmd, **kwargs):
        return subprocess.run(cmd, capture_output=True, text=True, **kwargs)

    def test_replay_observes_rebased_commit_and_later_dirty_main_blocks_ship(self):
        (self.wt / 'candidate.txt').write_text('candidate\n')
        self.git('add', 'candidate.txt', cwd=self.wt)
        self.git('commit', '--quiet', '-m', 'candidate', cwd=self.wt)
        old_candidate = self.git('rev-parse', 'HEAD', cwd=self.wt)
        (self.repo / 'new-main.txt').write_text('new main\n')
        self.git('add', 'new-main.txt')
        self.git('commit', '--quiet', '-m', 'main advances')
        base = self.git('rev-parse', 'HEAD')
        observed = []
        def replay(wt):
            observed.append(self.git('rev-parse', 'HEAD', cwd=wt))
            self.assertTrue(Path(wt, 'new-main.txt').exists())
            return {'ok': True}
        with patch.object(sa.ra, 'git', side_effect=self.git), patch.object(sa.ra, 'run', side_effect=self.command), \
             patch.object(sa.ra, 'gates', return_value=(True, 'fixture gate')), \
             patch.object(sa, 'replay_compare', side_effect=replay), patch.object(sa.ra, 'DRY', True):
            status, _, binding, _ = sa.validated_replay(str(self.wt), 'candidate')
            self.assertEqual(status, 'ready')
            self.assertEqual(binding['main'], base)
            self.assertNotEqual(binding['candidate'], old_candidate)
            self.assertEqual(observed, [binding['candidate']])
            self.assertEqual(sa.ra.ship('fixture', str(self.wt), 'candidate', {'deploys': {}}, expected=binding)[0], 'dry')
            (self.repo / 'base.txt').write_text('owner work after validation\n')
            self.assertEqual(sa.ra.ship('fixture', str(self.wt), 'candidate', {'deploys': {}}, expected=binding)[0], 'queued')
        self.assertEqual(self.git('rev-parse', 'HEAD'), base)

    def test_conflicting_rebase_aborts_and_preserves_named_draft(self):
        (self.wt / 'base.txt').write_text('candidate change\n')
        self.git('commit', '--quiet', '-am', 'candidate', cwd=self.wt)
        old_candidate = self.git('rev-parse', 'HEAD', cwd=self.wt)
        (self.repo / 'base.txt').write_text('main change\n')
        self.git('commit', '--quiet', '-am', 'main advances')
        with patch.object(sa.ra, 'git', side_effect=self.git), patch.object(sa.ra, 'run', side_effect=self.command), \
             patch.object(sa.ra, 'gates') as gates, patch.object(sa, 'replay_compare') as replay:
            status, detail, binding, _ = sa.validated_replay(str(self.wt), 'candidate')
        self.assertEqual(status, 'queued')
        self.assertIn('no longer applies', detail)
        self.assertIsNone(binding)
        self.assertEqual(self.git('rev-parse', 'candidate'), old_candidate)
        self.assertEqual(self.git('status', '--porcelain', cwd=self.wt), '')
        self.assertEqual((self.wt / 'base.txt').read_text(), 'candidate change\n')
        gates.assert_not_called()
        replay.assert_not_called()


if __name__ == '__main__':
    unittest.main()
