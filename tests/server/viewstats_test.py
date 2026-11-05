"""Evidence and duration accounting for passive view suggestions. No hardware."""
import json
from pathlib import Path
import random
import sqlite3
import sys
import tracemalloc
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'server'))
from viewstats import summarize, summarize_sorted  # noqa: E402


BOX = [0.3, 0.3, 0.6, 0.7]
PRESET = {'name': 'Saved view', 'x': 1000, 'y': 500}


def observations(start=0, stop=20, step=2, visible=True):
    return [{'ts': t, 'score': 1 if visible else 0, 'box': json.dumps(BOX) if visible else None}
            for t in range(start, stop + 1, step)]


def tracks(start=-40, stop=120, step=2, pose=(1000, 500)):
    return [{'ts': t, 'pan': pose[0], 'tilt': pose[1], 'act': None, 'moves': 0}
            for t in range(start, stop + 1, step)]


class ViewStatsTests(unittest.TestCase):
    def test_duration_presence_and_visits_are_logged_intervals(self):
        obs = observations()
        for row in obs:
            if row['ts'] >= 10:
                row.update(score=0, box=None)
        result = summarize(obs, tracks(), [PRESET])
        self.assertEqual((result['observed_seconds'], result['visible_seconds']), (20, 10))
        view = result['views'][0]
        self.assertEqual((view['visits'], view['observed_visits']), (1, 1))
        self.assertIn('not full-day', result['assumptions']['coverage'])

    def test_duplicate_and_overlapping_presets_cannot_double_count(self):
        obs = observations()
        result = summarize(obs + obs, tracks() + tracks(), [PRESET, dict(PRESET, name='Same place')])
        self.assertEqual(result['observed_seconds'], 20)
        self.assertEqual(sum(v['observed_seconds'] for v in result['views']), 20)
        self.assertEqual([v['observed_seconds'] for v in result['views']], [20, 0])

    def test_majority_overlap_uses_area_in_both_axes(self):
        diagonal = summarize(observations(), tracks(pose=(1200, 700)), [PRESET])
        self.assertEqual(diagonal['views'][0]['observed_seconds'], 0)
        self.assertEqual(diagonal['unsaved_observed_seconds'], 20)
        horizontal = summarize(observations(), tracks(pose=(1250, 500)), [PRESET])
        self.assertEqual(horizontal['views'][0]['observed_seconds'], 20)
        boundary = summarize(observations(), tracks(pose=(1366.5, 500)), [PRESET])
        self.assertEqual(boundary['views'][0]['observed_seconds'], 0)  # exactly half is not a majority

    def test_observation_gap_caps_extrapolation_and_starts_new_visit(self):
        obs = [dict(observations()[0], ts=t) for t in (0, 2, 40, 42)]
        result = summarize(obs, tracks(), [PRESET])
        self.assertEqual(result['observed_seconds'], 10)  # 2 + 6 + 2; no tail after final observation
        self.assertEqual(result['views'][0]['visits'], 2)

    def test_pose_gap_and_missing_pose_exclude_unsupported_time(self):
        trace = tracks(-40, 0) + tracks(40, 120)
        result = summarize(observations(0, 50), trace, [PRESET])
        self.assertEqual(result['observed_seconds'], 10)
        trace = tracks()
        for row in trace:
            if row['ts'] == 8:
                row['pan'] = None
        result = summarize(observations(), trace, [PRESET])
        self.assertEqual(result['observed_seconds'], 6)  # no bracket for 6..8, then cache guard

    def test_motor_action_excludes_stale_cache_and_settling(self):
        trace = tracks()
        next(row for row in trace if row['ts'] == 10)['moves'] = 1
        result = summarize(observations(0, 50), trace, [PRESET])
        # Adjacent action row also invalidates 8..10; action/cache/settling ends at43.
        self.assertEqual(result['observed_seconds'], 15)  # 0..8 and43..50
        self.assertEqual(result['views'][0]['visits'], 2)
        trace = tracks()
        next(row for row in trace if row['ts'] == 10)['act'] = 'search found after turn'
        self.assertEqual(summarize(observations(0, 50), trace, [PRESET])['observed_seconds'], 15)

    def test_late_cached_pose_change_excludes_previous_thirty_seconds(self):
        trace = tracks()
        for row in trace:
            if row['ts'] >= 40:
                row['pan'] = 2200
        result = summarize(observations(0, 60), trace, [PRESET])
        self.assertEqual(result['views'][0]['observed_seconds'], 10)  # 0..10, cache may have moved from10 onward
        self.assertEqual(result['unsaved_observed_seconds'], 17)  # new pose43..60 aftersettle

    def test_no_context_does_not_claim_unknown_cached_edges(self):
        result = summarize(observations(0, 100), tracks(0, 100), [PRESET])
        self.assertEqual(result['observed_seconds'], 37)  #33..70 only

    def test_unsaved_frequent_view_is_suggested_without_creating_presets(self):
        presets = [dict(PRESET)]
        result = summarize(observations(0, 80), tracks(stop=140, pose=(2400, 700)), presets)
        self.assertEqual(presets, [PRESET])
        self.assertEqual(len(result['suggestions']), 1)
        suggestion = result['suggestions'][0]
        self.assertEqual((suggestion['x'], suggestion['y'], suggestion['visible_seconds']), (2400, 700, 80))
        brief = summarize(observations(), tracks(pose=(2400, 700)), presets)
        self.assertEqual(brief['suggestions'], [])
        absent = summarize(observations(0, 80, visible=False), tracks(stop=140, pose=(2400, 700)), presets)
        self.assertEqual(absent['suggestions'], [])

    def test_low_score_or_invalid_boxes_do_not_become_visible_time(self):
        obs = observations()
        obs[0].update(score=0.1)
        obs[1].update(score=1, box='invalid')
        obs[2].update(score=1, box=[0.7, 0.3, 0.2, 0.8])
        result = summarize(obs, tracks(), [PRESET])
        self.assertEqual((result['observed_seconds'], result['visible_seconds']), (16, 14))
        self.assertEqual(result['coverage']['invalid_observations'], 2)

    def test_conflicting_duplicate_track_row_does_not_add_false_coverage(self):
        trace = tracks()
        trace.append(dict(next(row for row in trace if row['ts'] == 8), pan=3000))
        self.assertEqual(summarize(observations(), trace, [PRESET])['observed_seconds'], 6)

    def test_inputs_can_be_sqlite_rows_and_order_does_not_matter(self):
        with sqlite3.connect(':memory:') as con:
            con.row_factory = sqlite3.Row
            con.execute('create table obs(ts real, score real, box text)')
            con.executemany('insert into obs values(:ts,:score,:box)', observations())
            sqlite_obs = con.execute('select * from obs order by ts desc').fetchall()
        self.assertEqual(summarize(sqlite_obs, list(reversed(tracks())), [PRESET]),
                         summarize(observations(), tracks(), [PRESET]))

    def test_empty_and_bad_geometry(self):
        self.assertEqual(summarize([], [], [PRESET])['observed_seconds'], 0)
        for frame in (0, -1, float('nan'), True):
            with self.assertRaises(ValueError):
                summarize([], [], [], pan_frame=frame)

    def test_streaming_matches_batch_with_cache_changes_gaps_and_duplicates(self):
        rng = random.Random(417)
        for case in range(30):
            obs = observations(0, 1000)
            trace = tracks(-40, 1040)
            for row in obs:
                if rng.random() < 0.25:
                    row.update(score=0, box=None)
            for row in trace:
                if row['ts'] >= 300:
                    row['pan'] = 2400
                if 600 <= row['ts'] < 900:
                    row['tilt'] = 850
                if row['ts'] == 150 + case * 2:
                    row['moves'] = 1
            obs = [row for row in obs if rng.random() > 0.08]
            trace = [row for row in trace if not 450 < row['ts'] < 500 and rng.random() > 0.02]
            obs += [dict(obs[20]), dict(obs[21], score=0, box=None)]
            trace += [dict(trace[20]), dict(trace[21], pan=None)]
            obs.sort(key=lambda row: row['ts'])
            trace.sort(key=lambda row: row['ts'])
            with self.subTest(case=case):
                self.assertEqual(summarize_sorted(iter(obs), iter(trace), [PRESET]), summarize(obs, trace, [PRESET]))

    def test_streaming_accepts_two_sqlite_cursors_without_fetchall(self):
        with sqlite3.connect(':memory:') as con:
            con.row_factory = sqlite3.Row
            con.execute('create table obs(ts real, score real, box text)')
            con.execute('create table track(ts real, pan real, tilt real, act text, moves integer)')
            con.executemany('insert into obs values(:ts,:score,:box)', observations())
            con.executemany('insert into track values(:ts,:pan,:tilt,:act,:moves)', tracks())
            result = summarize_sorted(con.execute('select * from obs order by ts'),
                                      con.execute('select * from track order by ts'), [PRESET])
        self.assertEqual(result, summarize(observations(), tracks(), [PRESET]))

    def test_streaming_rejects_unsorted_input_and_handles_empty_edges(self):
        with self.assertRaisesRegex(ValueError, 'ordered'):
            summarize_sorted(reversed(observations()), iter(tracks()), [])
        with self.assertRaisesRegex(ValueError, 'ordered'):
            summarize_sorted(iter(observations()), reversed(tracks()), [])
        for obs, trace in (([], []), (observations(), []), ([], tracks()), (observations(), tracks(0, 0))):
            self.assertEqual(summarize_sorted(iter(obs), iter(trace), []), summarize(obs, trace, []))

    def test_streaming_does_not_retain_full_history(self):
        def obs():
            for stamp in range(0, 20001, 2):
                yield {'ts': stamp, 'score': 1, 'box': BOX}
        def trace():
            for stamp in range(-40, 20041, 2):
                yield {'ts': stamp, 'pan': 1000, 'tilt': 500, 'act': None, 'moves': 0}
        tracemalloc.start()
        try:
            result = summarize_sorted(obs(), trace(), [PRESET])
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertEqual(result['observed_seconds'], 20000)
        self.assertLess(peak, 2_000_000, f'streaming retained too much history: {peak} bytes')


if __name__ == '__main__':
    unittest.main()
