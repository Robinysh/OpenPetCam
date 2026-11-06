"""Independent metric, split and production-controller checks without private inputs."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim import search_bench as bench
from sim.gate1 import load_tracker
from sim.search_prior import load_prior
from sim.search_scene import SearchDeadline, SearchScenario, SearchScene


def result_row(ident, family='prior', right=False, elapsed=120, loss=120, motor=0,
               score=None, wrong=False, deadline=False):
    return {'id': ident, 'family': family, 'found_right': right, 'wrong_find': wrong,
            'elapsed_s': elapsed, 'loss_s': loss, 'motor_steps': motor, 'moves': int(motor > 0),
            'score_s': loss + bench.MOTOR_COST * motor if score is None else score,
            'deadline': deadline}


class SearchBenchTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.params = {
            'geometry': {'pan_max': 3700, 'tilt_max': 1000, 'pan_frame': 600,
                         'tilt_frame': 400, 'pan_backlash': 0},
            'motor': {'speed_steps_s': {'250': 214}},
            'pet': {'pan_hotspots': {'900': 1}, 'room_map': [[1200, 500, .7], [2700, 500, .3]]},
            'detect_persist': {'moving': {'p': 1}, 'still': {'p': 1}},
        }
        self.map_doc = {
            'version': 1, 'pan_step': 500, 'tilt_step': 250, 'min_floor': .02,
            'poses': [{'pan': x, 'tilt': y, 'floor_share': 0 if (x, y) == (3500, 500) else .5,
                       'tiles': list(range(9))}
                      for x in range(0, 3501, 500) for y in range(0, 1001, 250)],
            'stops': {str(h): [[2500, 500, .4], [500, 500, .3]] for h in range(24)},
            'places': {}, 'exclude_day': None,
        }
        self.policy = {'search_stops': [[1000, 500], [2600, 500], [3300, 500]],
                       'constants': {'LEAD': 1.7}}
        params_path, map_path = self.root / 'params.json', self.root / 'room_map.json'
        params_path.write_text(json.dumps(self.params))
        map_path.write_text(json.dumps(self.map_doc))
        self.prior = load_prior(params_path, map_path, target_step=(1000, 500))
        self.roommap = bench.load_roommap(map_path)

    def experiment(self, budget=12, policy=None):
        return bench.Experiment(self.params, self.prior, self.roommap,
                                self.policy if policy is None else policy, budget=budget)

    def case(self, **scenario):
        values = dict(seed=73, start_pose=[1019, 534], target_start=[1019, 534],
                      false_positive_p=0, detection_p=1, lag_s=.9)
        values.update(scenario)
        return {'id': 'synthetic-case', 'family': 'absent' if values.get('absent') else 'prior',
                'scenario': values, 'hour': 0}

    def test_summary_includes_failed_absent_and_occluded_trials(self):
        rows = [result_row('a', right=True, elapsed=10, loss=10, motor=20),
                result_row('b', elapsed=35, loss=120, motor=40),
                result_row('c', family='absent', elapsed=7, loss=120, wrong=True),
                result_row('d', family='occluded', elapsed=120, loss=120, deadline=True)]
        summary = bench.summarize(rows)
        self.assertEqual(summary['n'], 4)
        self.assertEqual(summary['found_right'], .25)
        self.assertEqual(summary['within_30s'], .25)
        self.assertEqual(summary['within_60s'], .25)
        self.assertEqual(summary['findable_within_60s'], .5)
        self.assertEqual(summary['mean_capped_loss_s'], 92.5)
        self.assertEqual(summary['mean_elapsed_s'], 43)
        self.assertEqual(summary['mean_motor_steps'], 15)
        self.assertEqual(summary['wrong_finds'], 1)
        self.assertEqual(summary['deadline_count'], 1)
        self.assertEqual(bench.summarize([]), {'n': 0})

    def test_paired_comparison_requires_same_order_and_complete_trial_ids(self):
        base = [result_row('a', loss=20, motor=100), result_row('b', loss=120, motor=200)]
        candidate = [result_row('a', loss=10, motor=50), result_row('b', loss=100, motor=150)]
        difference = bench.paired(base, candidate, seed=33, repeats=200)
        self.assertEqual(difference, bench.paired(base, candidate, seed=33, repeats=200))
        self.assertEqual(difference['loss_s']['candidate_minus_base'], -15)
        self.assertEqual(difference['motor_steps']['candidate_minus_base'], -50)
        for mismatched in [candidate[::-1], candidate[:1], [dict(candidate[0], id='other'), candidate[1]]]:
            with self.subTest(ids=[r['id'] for r in mismatched]), self.assertRaisesRegex(ValueError, 'IDs differ'):
                bench.paired(base, mismatched, repeats=10)
        identical = bench.paired(base, base, repeats=10)
        self.assertEqual(identical['loss_s']['paired_bootstrap_95'], [0, 0])

    def test_development_selection_respects_quality_before_lower_score(self):
        baseline = {'wrong_finds': 0, 'found_right': .80, 'within_60s': .60, 'mean_score_s': 50}
        summaries = {name: dict(baseline, mean_score_s=60) for name in bench.POLICIES}
        summaries['incumbent'] = baseline
        summaries['nearest'] = dict(baseline, mean_score_s=40)
        summaries['exact_existing'] = dict(baseline, wrong_finds=1, mean_score_s=1)
        summaries['room_greedy'] = dict(baseline, found_right=.779, mean_score_s=1)
        summaries['room_exact'] = dict(baseline, within_60s=.579, mean_score_s=1)
        self.assertEqual(bench.select(summaries), 'nearest')
        summaries['room_exact_rows85'] = dict(baseline, found_right=.78, within_60s=.58, mean_score_s=30)
        self.assertEqual(bench.select(summaries), 'room_exact_rows85')

    def test_acceptance_requires_every_guardrail_on_both_fresh_banks(self):
        baseline = {'wrong_finds': 1, 'found_right': .8, 'within_60s': .6, 'mean_score_s': 50}
        good = dict(baseline, found_right=.78, within_60s=.58, mean_score_s=49)
        comparisons = {label: {'base': dict(baseline), 'candidate': dict(good)}
                       for label in ('test', 'stress')}
        accepted = bench.acceptance(comparisons)
        self.assertTrue(accepted['passes_simulation_gate'])
        self.assertFalse(accepted['real_world_validated'])
        self.assertEqual(set(accepted['checks']), {'test', 'stress'})
        violations = [('wrong_finds', 2, 'no_extra_wrong_finds'),
                      ('found_right', .779, 'correct_find_nonregression'),
                      ('within_60s', .579, 'within_60s_nonregression'),
                      ('mean_score_s', 50, 'lower_mean_score')]
        for label in ('test', 'stress'):
            for field, value, check in violations:
                changed = {stage: {'base': dict(pair['base']), 'candidate': dict(pair['candidate'])}
                           for stage, pair in comparisons.items()}
                changed[label]['candidate'][field] = value
                with self.subTest(stage=label, field=field):
                    rejected = bench.acceptance(changed)
                    self.assertFalse(rejected['passes_simulation_gate'])
                    self.assertFalse(rejected['checks'][label][check])

    def test_seed_banks_are_reproducible_paired_inputs_not_household_holdouts(self):
        development = bench.bank(self.prior, 12, 10101)
        test = bench.bank(self.prior, 12, 80101)
        self.assertEqual(development, bench.bank(self.prior, 12, 10101))
        self.assertTrue(set(x['id'] for x in development).isdisjoint(x['id'] for x in test))
        self.assertTrue(set(x['scenario']['seed'] for x in development).isdisjoint(x['scenario']['seed'] for x in test))
        self.assertEqual({f: sum(c['family'] == f for c in test) for f in bench.FAMILIES},
                         {f: 2 for f in bench.FAMILIES})
        self.assertFalse(self.prior.provenance['held_out_claim'])
        self.assertFalse(self.prior.provenance['real_world_validated'])
        stress = bench.bank(self.prior, 6, 180101, stress=True)
        self.assertTrue(all(.8 <= c['speed_scale'] <= 1.1 and 707.2 <= c['pan_frame'] <= 740
                            and 27 <= c['backlash'] <= 40.9 for c in stress))

    def test_incumbent_route_is_actual_production_merge_and_floor_filter(self):
        experiment = self.experiment()
        # Compute the baseline independently through a freshly loaded production tracker.
        tracker = load_tracker()
        tracker.ROOM_MAP = self.roommap
        tracker.SEARCH_STOPS = tracker.policy_stops(self.policy)
        tracker.hour_of = lambda _: 0
        room = SearchScene(SearchScenario(start_pose=(1019, 534)), params=self.params)
        live = tracker.Tracker(room, room.find_pet, lambda: room.t)
        expected = [(p['x'], p['y']) for p in live.search_stops()]
        self.assertEqual(expected, [(2500, 500), (500, 500), (1000, 500)])
        route = experiment.route('incumbent', (1019, 534), 0)
        self.assertEqual(route['poses'], expected)
        self.assertEqual(route['method'], 'production_search_stops')
        self.assertFalse(route['optimality']['real_world_validated'])

    def test_only_non_incumbent_policies_execute_injected_stop_order(self):
        created = []

        class RecordingScene(SearchScene):
            def __init__(self, *args, **kwargs):
                self.goto_commands = []
                super().__init__(*args, **kwargs)
                created.append(self)

            def goto(self, x, y):
                self.goto_commands.append((x, y))
                return super().goto(x, y)

        experiment = self.experiment(budget=10)
        injected = {'poses': [(3400, 800)], 'row_step': .85, 'method': 'test-injection'}
        for name in bench.POLICIES:
            with self.subTest(policy=name), mock.patch.object(experiment, 'route', return_value=injected), \
                    mock.patch.object(bench, 'SearchScene', RecordingScene):
                result = experiment.run(name, self.case(absent=True), keep_trace=True)
                self.assertTrue(created[-1].goto_commands)
                expected = (2500, 500) if name == 'incumbent' else (3400, 800)
                self.assertEqual(created[-1].goto_commands[0], expected)
                self.assertFalse(result['found_right'])
                self.assertEqual(result['loss_s'], 10)
                self.assertEqual(experiment.T.ROW_STEP, .55)

    def test_policy_modules_and_globals_are_fresh_for_each_experiment(self):
        first = self.experiment()
        first.T.ROW_STEP = .99
        second = self.experiment(policy={})
        defaults = load_tracker()
        self.assertIsNot(first.T, second.T)
        self.assertIsNot(first.T.Tracker, second.T.Tracker)
        self.assertEqual(first.T.LEAD, 1.7)
        self.assertEqual(second.T.LEAD, defaults.LEAD)
        self.assertEqual(second.T.SEARCH_STOPS, [])
        self.assertEqual(second.T.ROW_STEP, defaults.ROW_STEP)
        self.assertIs(second.T.ROOM_MAP, self.roommap)
        self.assertEqual(first.T.ROW_STEP, .99)

    def test_resting_target_in_current_view_is_confirmed_before_any_move(self):
        experiment = self.experiment()
        fake_route = {'poses': [(3400, 800)], 'row_step': .55}
        for name in ('incumbent', 'room_exact'):
            with self.subTest(policy=name), mock.patch.object(experiment, 'route', return_value=fake_route):
                result = experiment.run(name, self.case(), keep_trace=True)
                self.assertTrue(result['found_right'])
                self.assertEqual(result['motor_steps'], 0)
                self.assertEqual(result['moves'], 0)
                self.assertIn('where the camera already was', result['message'])
                self.assertTrue(result['confirmation']['true_detection'])
                self.assertTrue(all(tuple(row[1:]) == (1019, 534, 1019) for row in result['trace']))

    def test_confirmation_requires_successful_controller_found_return(self):
        experiment = self.experiment()
        fake_route = {'poses': [(3400, 800)], 'row_step': .55}
        with mock.patch.object(experiment, 'route', return_value=fake_route), \
                mock.patch.object(experiment.T.Tracker, '_found', side_effect=SearchDeadline(12)):
            result = experiment.run('incumbent', self.case())
        self.assertTrue(result['deadline'])
        self.assertFalse(result['found_right'])
        self.assertFalse(result['wrong_find'])
        self.assertIsNone(result['confirmation'])
        self.assertEqual(result['loss_s'], 12)

    def test_cli_rejects_overlapping_and_negative_bank_seeds_before_loading_inputs(self):
        for arguments in [('--test-seed', '10101'), ('--stress-seed', '-1')]:
            with self.subTest(arguments=arguments), contextlib.redirect_stderr(io.StringIO()), \
                    mock.patch.object(bench, 'load_prior') as load, self.assertRaises(SystemExit) as error:
                bench.main(['--out', str(self.root / 'invalid'), '--dev', '6', '--test', '6',
                            '--stress', '6', *arguments])
            self.assertEqual(error.exception.code, 2)
            load.assert_not_called()

    def test_main_freezes_selection_before_evaluation_and_states_scope(self):
        out = self.root / 'run'
        (self.root / 'models').mkdir()
        (self.root / 'models/follow_policy.json').write_text('{}')
        observed_frozen_selection = []

        class FakeExperiment:
            def __init__(self, *args, **kwargs):
                self.cache = {}

            def run(self, name, case, keep_trace=False):
                if not case['id'].startswith('10101-'):
                    frozen = json.loads((out / 'selection.json').read_text())
                    observed_frozen_selection.append(frozen['selected'])
                    if not frozen['frozen_before_test']:
                        raise AssertionError('selection was not frozen before evaluation')
                duration = 10 if name == 'nearest' else 20 if name == 'incumbent' else 30
                right = case['family'] not in ('absent', 'occluded')
                row = result_row(case['id'], family=case['family'], right=right,
                                 elapsed=duration, loss=duration if right else 120)
                row.update(policy=name, cancelled=case.get('cancel_at_s') is not None)
                return row

        with mock.patch.object(bench, 'ROOT', self.root), \
                mock.patch.object(bench, 'load_prior', return_value=self.prior), \
                mock.patch.object(bench, 'load_params', return_value=self.params), \
                mock.patch.object(bench, 'load_roommap', return_value=self.roommap), \
                mock.patch.object(bench, 'Experiment', FakeExperiment), \
                mock.patch.object(bench, '_sha', return_value='fixture-hash'), \
                mock.patch.object(bench, 'select', wraps=bench.select) as selection, \
                contextlib.redirect_stdout(io.StringIO()):
            status = bench.main(['--out', str(out), '--dev', '6', '--test', '6', '--stress', '6',
                                 '--test-seed', '380101', '--stress-seed', '480101'])
        self.assertEqual(status, 0)
        self.assertEqual(selection.call_count, 1)
        self.assertEqual(set(observed_frozen_selection), {'nearest'})
        manifest = json.loads((out / 'manifest.json').read_text())
        summary = json.loads((out / 'summary.json').read_text())
        self.assertIn('not independent household days or labels', manifest['scope'])
        self.assertEqual(manifest['split'], {'development_seed': 10101, 'test_seed': 380101,
                                            'stress_seed': 480101})
        self.assertTrue(any('no validated room occlusion geometry' in x for x in manifest['limits']))
        self.assertTrue(any('all recorded days' in x for x in manifest['limits']))
        self.assertEqual(summary['selected'], 'nearest')
        self.assertTrue(summary['inputs_unchanged'])
        self.assertTrue(summary['acceptance']['passes_simulation_gate'])
        self.assertFalse(summary['acceptance']['real_world_validated'])


if __name__ == '__main__':
    unittest.main()
