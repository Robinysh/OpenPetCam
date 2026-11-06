"""Startup goals use page-open observations, never faster connection-only substitutes."""
import contextlib
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('PETCAM_CONFIG', str(ROOT/'config/site.env.example'))
sys.path[:0] = [str(ROOT/'server'), str(ROOT/'tools')]
import agentdb
import startupmetrics as startup
import superagent

NOW = 2000000000.0


def frame(open_ms=None, ms=10, ts=NOW, detail=None):
    return {'ts': ts, 'name': 'live_first_frame', 'ms': ms,
            'detail': json.dumps({'open_ms': open_ms}) if detail is None else detail}


def good_stats(open_ms=500):
    return startup.summarize([frame(open_ms, ts=NOW-i) for i in range(5)], now=NOW)


class StartupMetricsTest(unittest.TestCase):
    def test_fast_connect_only_events_never_satisfy_the_page_open_goal(self):
        stats = startup.summarize([frame(ms=100) for _ in range(50)], now=NOW)
        self.assertFalse(stats['ready'])
        self.assertIsNone(stats['p50'])
        self.assertEqual((stats['n'], stats['connect_n'], stats['connect_p50']), (0, 50, .1))
        metric = startup.goal_metric(stats)
        self.assertIsNone(metric['value'])
        self.assertFalse(metric['ok'])
        self.assertNotIn('observed_at', metric)

    def test_fewer_than_five_opens_are_visible_but_unassessed(self):
        rows = [frame(400, ts=NOW-10-i) for i in range(4)] + [frame(ms=1) for _ in range(100)]
        stats = startup.summarize(rows, now=NOW)
        self.assertEqual((stats['n'], stats['p50'], stats['connect_n']), (4, .4, 104))
        self.assertFalse(stats['ready'])
        metric = startup.goal_metric(stats)
        self.assertIsNone(metric['value'])
        self.assertFalse(metric['ok'])
        self.assertEqual(metric['observed_at'], NOW-10)

    def test_page_open_percentiles_are_not_diluted_by_reconnects(self):
        stats = startup.summarize([frame(x*1000, ts=NOW-x) for x in range(1, 6)]
                                  + [frame(ms=1) for _ in range(100)], now=NOW)
        self.assertTrue(stats['ready'])
        self.assertEqual((stats['p50'], stats['p90'], stats['n']), (3, 5, 5))
        self.assertEqual(stats['observed_at'], NOW-1)
        self.assertEqual(stats['first_observed_at'], NOW-5)
        self.assertEqual(stats['from'], 'open')
        self.assertFalse(startup.goal_metric(stats)['ok'])

    def test_target_is_unchanged_and_both_percentiles_must_pass(self):
        good = startup.goal_metric(good_stats(1000))
        self.assertEqual((good['value'], good['target'], good['ok']), (1, 'p50 <= 1, p90 <= 2', True))
        rows = [frame(500) for _ in range(9)] + [frame(2001)]
        metric = startup.goal_metric(startup.summarize(rows, now=NOW))
        self.assertEqual(metric['value'], .5)
        self.assertEqual(metric['detail']['stats']['p90'], 2.001)
        self.assertFalse(metric['ok'])

    def test_display_rounding_cannot_turn_a_threshold_miss_green(self):
        metric = startup.goal_metric(good_stats(1000.0001))
        self.assertGreater(metric['value'], 1)
        self.assertFalse(metric['ok'])

    def test_source_age_is_original_event_time_not_recollection_time(self):
        stamp = NOW - 6*86400
        stats = startup.summarize([frame(400, ts=stamp-i) for i in range(5)], now=NOW)
        self.assertEqual(stats['observed_at'], stamp)
        self.assertEqual(startup.goal_metric(stats)['observed_at'], stamp)
        later = startup.summarize([frame(400, ts=stamp-i) for i in range(5)], now=NOW+3600)
        self.assertEqual(later['observed_at'], stamp)

    def test_expired_future_and_invalid_timestamps_are_not_observations(self):
        invalid = [NOW-7*86400, NOW+1, -1, True, None, float('nan'), float('inf'), 10**400, '2000000000']
        stats = startup.summarize([frame(1, ts=stamp) for stamp in invalid], now=NOW)
        self.assertEqual((stats['n'], stats['connect_n']), (0, 0))
        self.assertIsNone(stats['observed_at'])

    def test_malformed_and_non_object_details_do_not_crash_or_create_open_samples(self):
        details = ['{broken', '[]', 'null', '"text"', '123', 'true', '{}', '', '['*1500 + '0' + ']'*1500]
        stats = startup.summarize([frame(ms=100, detail=value) for value in details], now=NOW)
        self.assertEqual(stats['n'], 0)
        self.assertEqual(stats['connect_n'], len(details))
        self.assertFalse(startup.goal_metric(stats)['ok'])

    def test_non_finite_negative_bool_and_string_timings_are_rejected(self):
        invalid = [-1, True, False, '10', None, float('nan'), float('inf'), 10**400]
        stats = startup.summarize([frame(value, ms=value) for value in invalid], now=NOW)
        self.assertEqual((stats['n'], stats['connect_n']), (0, 0))
        zero = startup.summarize([frame(0, ms=0)], now=NOW)
        self.assertEqual((zero['n'], zero['p50'], zero['connect_n']), (1, 0, 1))

    def test_timeout_and_reconnect_counts_are_context_not_invented_attempt_rates(self):
        rows = [frame(400) for _ in range(5)]
        rows += [{'ts': NOW, 'name': 'live_start_timeout'} for _ in range(20)]
        rows += [{'ts': NOW, 'name': 'live_reconnect'} for _ in range(3)]
        rows += [{'ts': NOW-8*86400, 'name': 'live_start_timeout'}]
        stats = startup.summarize(rows, now=NOW)
        self.assertEqual((stats['n'], stats['timeouts'], stats['reconnects']), (5, 20, 3))
        summary = startup.goal_metric(stats)['detail']['summary']
        self.assertIn('20 startup-timeout events', summary)
        self.assertIn('3 reconnect events', summary)
        self.assertIn('not an attempt success rate', summary)
        self.assertIn('foreground-resume latency is not separately measured', summary)

    def test_dict_details_and_sqlite_rows_are_supported(self):
        c = sqlite3.connect(':memory:');c.row_factory=sqlite3.Row
        self.addCleanup(c.close)
        c.execute('CREATE TABLE usage(ts REAL,name TEXT,ms REAL,detail TEXT)')
        c.execute('INSERT INTO usage VALUES(?,?,?,?)', (NOW,'live_first_frame',1,'{"open_ms": 1200}'))
        stats = startup.summarize(c.execute('SELECT * FROM usage'), now=NOW)
        self.assertEqual(stats['p50'], 1.2)
        self.assertEqual(startup.summarize([frame(detail={'open_ms': 500})], now=NOW)['p50'], .5)

    def test_missing_and_legacy_remote_results_become_explicit_unknown(self):
        for supplied in [None, [], {}, {'n': 10, 'p50': .1, 'p90': .2, 'from': 'connect'},
                         {'n': 10, 'p50': .1, 'p90': .2, 'from': 'open'}]:
            with self.subTest(stats=supplied):
                metric = startup.goal_metric(supplied)
                self.assertIsNone(metric['value'])
                self.assertFalse(metric['ok'])
                self.assertFalse(metric['detail']['stats']['ready'])

    def test_untrusted_ready_flag_cannot_bypass_minimum_or_provenance(self):
        for change in [{'n': 4}, {'n': True}, {'p90': None}, {'p90': -.1}, {'p50': float('nan')},
                       {'observed_at': None}, {'min_samples': 1}, {'window_days': 1}, {'from':'connect'}]:
            stats = good_stats();stats.update(change)
            with self.subTest(change=change):
                self.assertFalse(startup.goal_metric(stats)['ok'])
                self.assertIsNone(startup.goal_metric(stats)['value'])

    def test_structured_stats_survive_the_existing_agent_database_detail_bound(self):
        c = sqlite3.connect(':memory:');c.row_factory=sqlite3.Row;c.executescript(agentdb.SCHEMA)
        self.addCleanup(c.close)
        sample = startup.goal_metric(good_stats())
        agentdb.cmd_metric(c, {'metrics': [sample]})
        raw = c.execute('SELECT detail FROM metrics').fetchone()[0]
        self.assertLessEqual(len(raw), 2000)
        wrapper = json.loads(raw);detail = json.loads(wrapper['measurement'])
        self.assertEqual(detail['schema'], 'startup-v1')
        self.assertEqual(detail['stats'], sample['detail']['stats'])
        self.assertEqual(wrapper['observed_at'], NOW)

    def test_remote_collector_uses_real_event_rows_including_timeout_context(self):
        c = sqlite3.connect(':memory:');c.row_factory=sqlite3.Row
        self.addCleanup(c.close)
        c.execute('CREATE TABLE usage(ts REAL,name TEXT,ms REAL,detail TEXT)')
        c.executemany('INSERT INTO usage VALUES(?,?,?,?)', [(NOW-i, 'live_first_frame', 10, '{"open_ms": 1500}') for i in range(5)])
        c.execute('INSERT INTO usage VALUES(?,?,?,?)', (NOW,'live_start_timeout',None,None))
        text = superagent.METRICS_REMOTE
        block = text[text.index('# video start:'):text.index('print(json.dumps(out))', text.index('# video start:'))]
        out = {}
        exec(compile(block, 'metric-startup-scan', 'exec'), {'c':c, 'out':out, 'now':NOW, 'time':SimpleNamespace(time=lambda:NOW), 'startupmetrics':startup})
        self.assertEqual((out['video_start']['n'], out['video_start']['p90'], out['video_start']['timeouts']), (5, 1.5, 1))

    def producer(self, payload=None, failed=False):
        with tempfile.TemporaryDirectory() as temp, contextlib.ExitStack() as stack:
            (Path(temp)/'qa_last.json').write_text('[]')
            stack.enter_context(patch.object(superagent,'RUN',temp))
            stack.enter_context(patch.object(superagent,'SITE',{}))
            stack.enter_context(patch.object(superagent.time,'localtime',return_value=SimpleNamespace(tm_hour=12)))
            run = stack.enter_context(patch.object(superagent.subprocess,'run',return_value=SimpleNamespace(returncode=1 if failed else 0,stdout=json.dumps(payload or {}),stderr='')))
            adb = stack.enter_context(patch.object(superagent,'adb',return_value={'ok':True}))
            stack.enter_context(patch.object(superagent,'shadow_step'))
            stack.enter_context(patch.object(superagent,'lessons'))
            stack.enter_context(patch.object(superagent,'log'))
            samples=superagent.metrics()
            self.assertEqual(run.call_count, 1)
            self.assertTrue(all(call.args[0] in ('metric','beat') for call in adb.call_args_list))
            return next(row for row in samples if row['goal']=='video_start')

    def test_producer_records_missing_or_failed_collection_instead_of_retaining_success(self):
        for failed in (False, True):
            metric = self.producer(failed=failed)
            self.assertIsNone(metric['value'])
            self.assertFalse(metric['ok'])
            self.assertIsNone(metric['detail']['stats']['n'])

    def test_producer_rejects_legacy_connection_source_as_page_open_success(self):
        metric = self.producer({'video_start': {'from':'connect', 'n':25, 'p50':.1, 'p90':.2}})
        self.assertIsNone(metric['value'])
        self.assertFalse(metric['ok'])

    def test_successful_empty_scan_preserves_real_zero_counts(self):
        metric = self.producer({'video_start': startup.summarize([], now=NOW)})
        self.assertIsNone(metric['value'])
        for name in ['n', 'connect_n', 'timeouts', 'reconnects']:
            self.assertEqual(metric['detail']['stats'][name], 0)
        self.assertEqual(metric['detail']['stats']['from'], 'open')

    def test_failed_or_invalid_source_does_not_manufacture_zero_event_counts(self):
        for supplied in [None, {}, [], {'from':'connect','n':50}]:
            metric = startup.goal_metric(supplied)
            for name in ['n', 'connect_n', 'timeouts', 'reconnects']:
                self.assertIsNone(metric['detail']['stats'][name])
            self.assertIn('counts are unknown', metric['detail']['summary'])

    def test_producer_keeps_page_open_source_time_and_both_target_tests(self):
        metric = self.producer({'video_start': good_stats(1200)})
        self.assertEqual(metric['value'], 1.2)
        self.assertFalse(metric['ok'])
        self.assertEqual(metric['observed_at'], NOW)
        self.assertEqual(metric['detail']['schema'], 'startup-v1')


if __name__ == '__main__':
    unittest.main()
