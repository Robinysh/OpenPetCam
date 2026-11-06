"""A fast connection is not evidence that the page-open goal is met."""
import sqlite3
import sys
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'server'))
import agentdb
import goalmonitor


class StartupGoalProjectionTest(unittest.TestCase):
    def setUp(self):
        self.now = 2_000_000_000
        self.c = sqlite3.connect(':memory:')
        self.c.row_factory = sqlite3.Row
        self.c.executescript(agentdb.SCHEMA)
        self.addCleanup(self.c.close)
        goal = {'id': 'video_start', 'what': 'Fast page opening', 'metric': 'Page open to playing',
                'target': 'p50 <= 1, p90 <= 2', 'ground_truth': 'Recorded page openings'}
        agentdb.cmd_goal_catalog(self.c, {'goals': [goal]})

    def project(self, stats=None, value=.4, ok=True, observed=None, legacy=False):
        stamp = self.now if observed is None else observed
        stats = dict({'from': 'open', 'n': 5, 'ready': True, 'p50': .4, 'p90': .8,
                      'observed_at': stamp}, **(stats or {}))
        detail = 'Five fast connection timings' if legacy else {
            'schema': 'startup-v1', 'summary': 'Median .4 s, p90 .8 s, five page openings.', 'stats': stats}
        sample = {'goal': 'video_start', 'target': 'p50 <= 1, p90 <= 2', 'value': value,
                  'ok': ok, 'detail': detail, 'observed_at': stamp}
        agentdb.cmd_metric(self.c, {'metrics': [sample]})
        return goalmonitor.project(self.c, now=self.now)['goals'][0]

    def test_legacy_connection_or_unknown_basis_never_looks_healthy(self):
        row = self.project(legacy=True)
        self.assertEqual(row['health']['state'], 'invalid')
        self.assertIn('unverified', row['health']['label'])

    def test_both_percentiles_and_observed_page_open_counts_are_visible(self):
        row = self.project()
        self.assertEqual(row['health']['state'], 'on_target')
        self.assertEqual(row['sample']['startup']['p90'], .8)
        self.assertEqual(row['sample']['startup']['n'], 5)
        self.assertEqual(row['sample']['detail'], 'Median .4 s, p90 .8 s, five page openings.')
        self.assertEqual(row['status']['state'], 'monitoring')

    def test_fast_median_with_slow_tail_is_off_target(self):
        row = self.project({'p90': 8}, ok=False)
        self.assertEqual(row['health']['state'], 'off_target')

    def test_inconsistent_p90_verdict_cannot_be_green(self):
        row = self.project({'p90': 8}, ok=True)
        self.assertEqual(row['health']['state'], 'invalid')

    def test_too_few_openings_cannot_use_fast_connection_diagnostics(self):
        row = self.project({'n': 4, 'ready': False, 'connect_p50': .1}, value=None, ok=False)
        self.assertEqual(row['health']['state'], 'unmeasured')
        self.assertEqual(row['health']['label'], 'Need more app openings')

    def test_old_source_openings_stay_stale_after_new_collection(self):
        row = self.project(observed=self.now - 2 * 3600)
        self.assertEqual(row['health']['state'], 'stale')

    def test_invalid_or_mismatched_source_stats_are_unverified(self):
        for changes in ({'p90': None}, {'p50': -.1}, {'p90': .1}, {'observed_at': self.now + 1}):
            with self.subTest(changes=changes):
                self.c.execute('DELETE FROM metrics')
                self.assertEqual(self.project(changes)['health']['state'], 'invalid')


if __name__ == '__main__':
    unittest.main()
