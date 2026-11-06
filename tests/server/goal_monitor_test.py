"""Source-backed goal cards stay continuous and never mutate agent work on read."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('PETCAM_CONFIG', str(ROOT / 'config/site.env.example'))
sys.path[:0] = [str(ROOT / 'tools'), str(ROOT)]
from server import agentdb, goalmonitor
import superagent


def definition(ident='cpu', target='<= 8'):
    return {'id': ident, 'what': 'Keep ' + ident + ' in range', 'metric': 'Measured value',
            'target': target, 'ground_truth': 'Observed source', 'note': 'A proxy, not independent truth.'}


class GoalMonitorTest(unittest.TestCase):
    def setUp(self):
        self.c = sqlite3.connect(':memory:')
        self.c.row_factory = sqlite3.Row
        self.c.executescript(agentdb.SCHEMA)
        self.addCleanup(self.c.close)
        self.now = 2000000000.0
        agentdb.cmd_goal_catalog(self.c, {'goals': [definition()]})

    def sample(self, goal='cpu', value=5, stamp=None, target='<= 8', ok=1, detail='"Measured"'):
        self.c.execute('INSERT INTO metrics VALUES(?,?,?,?,?,?)',
                       (self.now if stamp is None else stamp, goal, value, target, ok, detail))
        self.c.commit()

    def card(self, ident='cpu'):
        return next(row for row in goalmonitor.project(self.c, now=self.now)['goals'] if row['goal_id'] == ident)

    def test_all_catalog_goals_exist_without_a_measurement(self):
        agentdb.cmd_goal_catalog(self.c, {'goals': [definition(), definition('follow_retention', '>= 80')]})
        rows = goalmonitor.project(self.c, now=self.now)['goals']
        self.assertEqual({r['goal_id'] for r in rows}, {'cpu', 'follow_retention'})
        self.assertTrue(all(r['sample'] is None and r['health']['state'] == 'unmeasured' for r in rows))
        self.assertTrue(all(r['status']['state'] == 'monitoring' and not r['movable'] for r in rows))

    def test_fresh_on_and_off_target_are_monitoring_not_completion(self):
        for value, ok, state in [(5, 1, 'on_target'), (10, 0, 'off_target')]:
            self.sample(value=value, ok=ok)
            row = self.card()
            self.assertEqual(row['health']['state'], state)
            self.assertEqual(row['status']['state'], 'monitoring')
            self.assertEqual(row['health']['expires_at'], self.now + 90 * 60)
            self.assertEqual(row['source'], 'Observed source')
            self.assertIn('proxy', row['note'])

    def test_latest_timestamp_ties_produce_one_card_and_one_latest_sample(self):
        self.sample(value=10, ok=0)
        self.sample(value=6, ok=1)
        self.assertEqual(len(agentdb.latest_goals(self.c)), 1)
        self.assertEqual(agentdb.latest_goals(self.c)[0]['value'], 6)
        self.assertEqual(self.card()['sample']['value'], 6)
        self.assertEqual(self.card()['series'], [[self.now, 6, True]])

    def test_newer_unmeasured_sample_does_not_reuse_an_old_success(self):
        self.sample(stamp=self.now - 60)
        self.sample(value=None)
        self.assertEqual(self.card()['health']['state'], 'unmeasured')
        self.assertIsNone(self.card()['sample']['value'])

    def test_stale_sample_cannot_be_green(self):
        self.sample(stamp=self.now - 90 * 60 - 1)
        self.assertEqual(self.card()['health']['state'], 'stale')

    def test_invalid_future_or_non_numeric_samples_cannot_be_green(self):
        cases = [dict(value=float('inf')), dict(value='bad'), dict(stamp=self.now + 61), dict(ok=2)]
        for changes in cases:
            with self.subTest(changes=changes):
                self.c.execute('DELETE FROM metrics')
                self.sample(**changes)
                self.assertNotEqual(self.card()['health']['state'], 'on_target')

    def test_target_change_does_not_relabel_a_historical_verdict(self):
        self.sample(target='<= 10')
        row = self.card()
        self.assertEqual(row['target'], '<= 8')
        self.assertEqual(row['sample']['target'], '<= 10')
        self.assertEqual(row['health']['state'], 'invalid')
        self.assertEqual(row['health']['label'], 'Target changed')

    def test_legacy_qa_source_time_is_unknown_not_fresh(self):
        agentdb.cmd_goal_catalog(self.c, {'goals': [definition('qa', '== 0')]})
        self.sample(goal='qa', value=0, target='== 0')
        row = self.card('qa')
        self.assertEqual(row['health']['state'], 'invalid')
        self.assertEqual(row['health']['label'], 'Source time unknown')

    def test_legacy_report_zeros_are_unverified_until_a_new_source_scan(self):
        for ident, target in [('reports_answered', '<= 15'), ('owner_reports', '<= 1')]:
            with self.subTest(goal=ident):
                agentdb.cmd_goal_catalog(self.c, {'goals': [definition(ident, target)]})
                self.sample(goal=ident, value=0, target=target)
                self.assertEqual(self.card(ident)['health']['label'], 'Source not verified')
                detail = json.dumps({'measurement': 'Observed source', 'observed_at': self.now})
                self.sample(goal=ident, value=0, target=target, detail=detail)
                self.assertEqual(self.card(ident)['health']['state'], 'on_target')

    def test_cached_qa_age_uses_source_not_reinsertion_time(self):
        agentdb.cmd_goal_catalog(self.c, {'goals': [definition('qa', '== 0')]})
        detail = json.dumps({'measurement': 'all pass', 'observed_at': self.now - 37 * 3600})
        self.sample(goal='qa', value=0, target='== 0', detail=detail)
        self.assertEqual(self.card('qa')['health']['state'], 'stale')
        self.c.execute('DELETE FROM metrics')
        detail = json.dumps({'measurement': 'all pass', 'observed_at': self.now - 20 * 3600})
        self.sample(goal='qa', value=0, target='== 0', detail=detail)
        self.assertEqual(self.card('qa')['health']['state'], 'on_target')
        self.assertEqual(self.card('qa')['health']['expires_at'], self.now + 16 * 3600)

    def test_work_links_do_not_close_goal_or_claim_pid_is_live(self):
        self.sample()
        self.c.execute("INSERT INTO threads(id,title,kind,status,worker,created,updated,summary) VALUES(?,?,?,?,?,?,?,?)",
                       ('goal:cpu', 'CPU improvement', 'improve', 'closed', 'pid:123', self.now-10, self.now, 'Worker finished'))
        self.c.execute("INSERT INTO threads(id,title,status) VALUES('report:123','Unrelated','open')")
        self.c.commit()
        row = self.card()
        self.assertEqual(row['status']['state'], 'monitoring')
        self.assertEqual(len(row['work']), 1)
        self.assertEqual(row['work'][0]['status'], 'closed')
        self.assertTrue(row['work'][0]['worker_assigned'])
        self.assertNotIn('running', row['work'][0])

    def test_read_projection_is_select_only_with_default_or_row_factory(self):
        self.sample()
        before = self.c.total_changes
        snapshot = [tuple(row) for row in self.c.execute('SELECT * FROM metrics')]
        statements = []
        self.c.set_trace_callback(statements.append)
        self.c.row_factory = None
        self.assertEqual(self.card()['health']['state'], 'on_target')
        self.c.set_trace_callback(None)
        self.assertEqual(self.c.total_changes, before)
        self.assertEqual([tuple(row) for row in self.c.execute('SELECT * FROM metrics')], snapshot)
        self.assertTrue(all(s.lstrip().upper().startswith('SELECT') for s in statements))

    def test_bounded_recent_trend_preserves_missing_sample_gaps(self):
        self.c.executemany('INSERT INTO metrics VALUES(?,?,?,?,?,?)',
                           [(self.now - i * 60, 'cpu', None if i == 3 else i, '<= 8', int(i <= 8), 'null') for i in range(400)])
        self.c.commit()
        points = self.card()['series']
        self.assertEqual(len(points), goalmonitor.TREND_POINTS)
        self.assertTrue(any(value is None for _, value, _ in points))
        self.assertEqual([p[0] for p in points], sorted(p[0] for p in points))

    def test_catalog_retirement_preserves_history_without_recreating_goal(self):
        self.sample()
        agentdb.cmd_goal_catalog(self.c, {'goals': [definition('other')]})
        self.assertEqual([r['goal_id'] for r in goalmonitor.project(self.c, now=self.now)['goals']], ['other'])
        self.assertEqual(self.c.execute('SELECT COUNT(*) FROM metrics').fetchone()[0], 1)
        agentdb.cmd_goal_catalog(self.c, {'goals': []})
        self.assertEqual(goalmonitor.project(self.c, now=self.now)['goals'], [])

    def test_unknown_legacy_definition_is_visible_but_not_healthy(self):
        self.sample()
        self.c.execute('DROP TABLE goal_catalog')
        self.c.execute('DROP TABLE goal_catalog_state')
        result = goalmonitor.project(self.c, now=self.now)
        self.assertFalse(result['catalog_available'])
        self.assertEqual(result['goals'][0]['health']['state'], 'invalid')

    def test_bad_catalog_cannot_partially_replace_snapshot(self):
        before = self.c.execute('SELECT definition FROM goal_catalog').fetchone()[0]
        for definitions in ([definition(), definition()], [definition(), {'id': 'bad'}], None):
            with self.subTest(definitions=definitions):
                with self.assertRaises(ValueError):
                    agentdb.cmd_goal_catalog(self.c, {'goals': definitions})
                self.assertEqual(self.c.execute('SELECT definition FROM goal_catalog').fetchone()[0], before)

    def test_catalog_and_sample_batch_roll_back_together(self):
        self.c.execute("CREATE TRIGGER reject_bad BEFORE INSERT ON metrics WHEN NEW.goal='bad' BEGIN SELECT RAISE(ABORT,'bad'); END")
        self.c.commit()
        with self.assertRaises(sqlite3.IntegrityError):
            agentdb.cmd_metric(self.c, {'goals': [definition('new')], 'metrics': [
                {'goal':'cpu','value':1,'target':'<= 8','ok':True}, {'goal':'bad','value':1}]})
        self.assertEqual(self.c.execute('SELECT id FROM goal_catalog').fetchone()[0], 'cpu')
        self.assertEqual(self.c.execute('SELECT COUNT(*) FROM metrics').fetchone()[0], 0)

    def test_source_time_survives_long_detail_serialization(self):
        detail = agentdb._metric_detail({'detail': '"😺' * 10000, 'observed_at': self.now-100})
        self.assertLessEqual(len(detail), 2000)
        self.assertEqual(json.loads(detail)['observed_at'], self.now-100)

    def test_legacy_metric_writer_does_not_erase_catalog(self):
        agentdb.cmd_metric(self.c, {'metrics': [{'goal':'cpu','value':1,'target':'<= 8','ok':True}]})
        self.assertEqual(self.c.execute('SELECT id FROM goal_catalog').fetchone()[0], 'cpu')

    def test_read_only_file_access_and_missing_db_create_nothing(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'agent.db'
            self.assertFalse(goalmonitor.read(path)['available'])
            self.assertFalse(path.exists())
            with sqlite3.connect(path) as c:
                c.executescript(agentdb.SCHEMA)
                agentdb.cmd_goal_catalog(c, {'goals':[definition()]})
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(len(goalmonitor.read(path)['goals']), 1)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)


class MetricProducerTest(unittest.TestCase):
    def test_unreadable_existing_report_cannot_produce_verified_zero(self):
        text = superagent.METRICS_REMOTE
        block = text[text.index('report_meta = {}'):text.index('# video start:')]
        with tempfile.TemporaryDirectory() as temp:
            report = Path(temp)/'reports'/'20260101-010101'
            report.mkdir(parents=True)
            (report/'report.json').write_text('{"note":"problem"}')
            out = {}
            with patch('builtins.open', side_effect=PermissionError('unreadable report')):
                with self.assertRaises(PermissionError):
                    exec(compile(block, 'metric-report-scan', 'exec'),
                         {'S':temp, 'now':time.time(), 'out':out, 'os':os})
            self.assertNotIn('reports_observed_at', out)
            self.assertNotIn('oldest_unanswered_min', out)
            self.assertNotIn('owner_reports_per_day', out)

    def test_incomplete_report_directory_does_not_abort_remote_measurements(self):
        text = superagent.METRICS_REMOTE
        block = text[text.index('report_meta = {}'):text.index('# video start:')]
        with tempfile.TemporaryDirectory() as temp:
            base = Path(temp)/'reports';base.mkdir()
            (base/'20260101-010101').mkdir()  # interruption before report.json
            valid = base/'20260101-010102';valid.mkdir();(valid/'report.json').write_text('{"note":"problem"}')
            empty = base/'20260101-010103';empty.mkdir();(empty/'report.json').touch()
            out={};exec(compile(block,'metric-report-scan','exec'), {'S':temp,'now':time.time(),'out':out,'os':os})
            self.assertEqual(out['owner_reports_per_day'], round(1/7,2))
            self.assertGreaterEqual(out['oldest_unanswered_min'], 0)
            self.assertGreater(out['reports_observed_at'], 0)

    def test_failed_remote_query_is_unmeasured_and_qa_keeps_source_age(self):
        with tempfile.TemporaryDirectory() as temp, contextlib.ExitStack() as stack:
            qa=Path(temp)/'qa_last.json';qa.write_text('[{"name":"snapshot","status":"PASS"}]')
            observed=time.time()-37*3600;os.utime(qa,(observed,observed))
            stack.enter_context(patch.object(superagent,'RUN',temp))
            stack.enter_context(patch.object(superagent.subprocess,'run',return_value=SimpleNamespace(returncode=1,stdout='',stderr='offline')))
            adb=stack.enter_context(patch.object(superagent,'adb',return_value={'ok':True}))
            stack.enter_context(patch.object(superagent,'shadow_step'))
            stack.enter_context(patch.object(superagent,'lessons'))
            stack.enter_context(patch.object(superagent,'log'))
            samples=superagent.metrics();by={r['goal']:r for r in samples}
            for ident in ['reports_answered','owner_reports']:
                self.assertIsNone(by[ident]['value'])
                self.assertFalse(by[ident]['ok'])
                self.assertGreater(by[ident]['observed_at'], observed)
            self.assertEqual(by['qa']['observed_at'], observed)
            batch=next(call.args[1] for call in adb.call_args_list if call.args[0]=='metric')
            canonical=json.loads(Path(ROOT/'agent/goals.json').read_text())['goals']
            self.assertEqual(batch['goals'], canonical)
            self.assertTrue(all(call.args[0] in ['metric','beat'] for call in adb.call_args_list))


if __name__ == '__main__':
    unittest.main()
