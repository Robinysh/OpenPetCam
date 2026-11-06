"""Offline split, oracle-boundary and lifecycle accounting with synthetic fixtures."""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim import obstruction_bench as bench
from sim.view_obstructions import Config


class ColorModel:
    """A fixture whose only input is RGB; hidden truth is not available here."""
    def predict(self, rgb):
        if not isinstance(rgb, np.ndarray) or rgb.shape != (8, 12, 3):
            raise AssertionError('classifier must receive RGB, not evaluator metadata')
        return .99 if float(np.median(rgb)) > 150 else .01


class PixelScene:
    def __init__(self, pixels, truth=None):
        self.pixels = pixels
        self.hidden = pixels if truth is None else truth

    def render(self, pose, now, version):
        return np.full((8, 12, 3), 180 if self.pixels(pose, now, version) else 80, dtype=np.uint8)

    def truth(self, pose, now, version):
        blocked = bool(self.hidden(pose, now, version))
        return types.SimpleNamespace(blocking=blocked, coverage=float(blocked),
                                     persistent_coverage=float(blocked), transient_coverage=0.)


def counters(opportunities=10, inspections=10, skips=0, useful_skips=0, false_skips=0,
             deferred_probes=0, missed_pet_opportunities=0, persistent_skips=0, transient_skips=0):
    return dict(opportunities=opportunities, inspections=inspections, skips=skips,
                useful_skips=useful_skips, false_skips=false_skips,
                deferred_probes=deferred_probes, missed_pet_opportunities=missed_pet_opportunities,
                persistent_skips=persistent_skips, transient_skips=transient_skips)


class ObstructionBenchTest(unittest.TestCase):
    def config(self, **kw):
        values = dict(high_threshold=.9, low_threshold=.2, min_samples=2, min_gap_s=15,
                      min_span_s=15, probe_interval_s=60, exclusion_ttl_s=180)
        values.update(kw)
        return Config(**values)

    def episode(self, family, scene, end=185, change=65, config=None, model=None, budget=2, stress=False):
        with mock.patch.object(bench, 'make_episode', return_value=(family, scene)), \
                mock.patch.multiple(bench, END_S=end, CHANGE_S=change, PROBE_BUDGET=budget):
            return bench.episode(model or ColorModel(), config or self.config(),
                                 'synthetic-test', 0, keep_rows=True, stress=stress)

    def test_opposite_hidden_truth_cannot_change_image_scores_or_memory_decisions(self):
        anchor = bench.POSES[2]
        pixels = lambda pose, now, version: pose == anchor
        a = self.episode('persistent', PixelScene(pixels))
        b = self.episode('persistent', PixelScene(pixels, truth=lambda *args: False))
        decision_keys = ('at_s', 'pose', 'score', 'state', 'skip', 'probe_due', 'deferred', 'context', 'accepted')
        self.assertEqual([{k: r[k] for k in decision_keys} for r in a['samples']],
                         [{k: r[k] for k in decision_keys} for r in b['samples']])
        self.assertEqual(a['transitions'], b['transitions'])
        visual_a, visual_b = a['counts']['visual_memory'], b['counts']['visual_memory']
        self.assertGreater(visual_a['skips'], 0)
        self.assertEqual(visual_a['skips'], visual_b['skips'])
        self.assertEqual(visual_a['useful_skips'], visual_b['false_skips'])
        self.assertEqual(visual_a['false_skips'], 0)

    def test_every_opportunity_is_inspected_skipped_or_explicitly_deferred(self):
        result = self.episode('persistent', PixelScene(lambda *args: True), end=185, budget=2)
        expected_opportunities = len(range(5, 186, bench.TICK_S)) * len(bench.POSES)
        for policy, count in result['counts'].items():
            with self.subTest(policy=policy):
                self.assertEqual(count['opportunities'], expected_opportunities)
                self.assertEqual(count['opportunities'], count['inspections'] + count['skips'] + count['deferred_probes'])
                self.assertEqual(count['skips'], count['useful_skips'] + count['false_skips'])
        baseline = result['counts']['always_inspect']
        self.assertEqual(baseline['inspections'], expected_opportunities)
        self.assertEqual(baseline['skips'], 0)
        self.assertGreater(result['counts']['visual_memory']['deferred_probes'], 0)

    def test_mandatory_reinspection_observations_obey_per_tick_probe_budget(self):
        result = self.episode('persistent', PixelScene(lambda *args: True), end=185, budget=2)
        actual_probes = {}
        for row in result['samples']:
            if row['state'] == 'reinspection_due' and row['score'] is not None:
                tick_index = int((row['at_s'] - 5) // bench.TICK_S)
                actual_probes[tick_index] = actual_probes.get(tick_index, 0) + 1
        self.assertTrue(actual_probes)
        self.assertLessEqual(max(actual_probes.values()), 2)
        self.assertTrue(any(row['deferred'] and row['score'] is None and not row['skip']
                            for row in result['samples']))
        # Deferral is not represented as learned obstruction savings.
        visual = result['counts']['visual_memory']
        self.assertEqual(visual['skips'], sum(row['skip'] for row in result['samples']))

    def test_empty_pet_detections_do_not_exclude_clear_views(self):
        result = self.episode('empty', PixelScene(lambda *args: False), end=305)
        visual, naive = result['counts']['visual_memory'], result['counts']['empty_means_blocked']
        self.assertEqual(visual['skips'], 0)
        self.assertEqual(visual['false_skips'], 0)
        self.assertEqual(visual['missed_pet_opportunities'], 0)
        self.assertGreater(naive['false_skips'], 0)
        self.assertGreater(naive['missed_pet_opportunities'], 0)
        self.assertTrue(all(row['score'] == .01 for row in result['samples']))

    def test_low_quality_images_cannot_teach_even_confident_obstructions(self):
        class DarkScene(PixelScene):
            def render(self, *args):
                return np.zeros((8, 12, 3), dtype=np.uint8)

        result = self.episode('dark', DarkScene(lambda *args: True),
                              model=types.SimpleNamespace(predict=lambda rgb: .999))
        self.assertEqual(result['counts']['visual_memory']['skips'], 0)
        self.assertTrue(all(row['accepted'] is False for row in result['samples']))
        self.assertTrue(all(row['state'] == 'unknown' for row in result['samples']))

    def test_announced_setup_change_reopens_immediately_and_unannounced_change_is_probed(self):
        anchor = bench.POSES[2]
        scene = PixelScene(lambda pose, now, version: pose == anchor and version == 0)
        announced = self.episode('changed_setup', scene, end=185)
        unannounced = self.episode('unannounced_move', scene, end=185)
        self.assertEqual(announced['first_inspection_after_change_s'], .5)
        self.assertEqual(announced['clear_accepted_after_change_s'], .5)
        self.assertGreater(unannounced['first_inspection_after_change_s'], .5)
        self.assertLessEqual(unannounced['first_inspection_after_change_s'],
                             self.config().probe_interval_s + 2 * bench.TICK_S + 1)
        changed = next(row for row in announced['samples'] if row['pose'] == anchor and row['at_s'] >= 65)
        self.assertTrue(changed['context'].endswith('setup:1'))
        self.assertFalse(changed['skip'])
        self.assertEqual(changed['state'], 'unknown')
        self.assertTrue(changed['accepted'])

    def test_short_transient_cannot_satisfy_long_persistence_requirement(self):
        anchor = bench.POSES[2]
        transient = PixelScene(lambda pose, now, version: pose == anchor and 80 <= now < 110,
                               truth=lambda *args: False)
        config = self.config(min_samples=3, min_span_s=45)
        result = self.episode('transient', transient, end=185, config=config)
        self.assertEqual(result['counts']['visual_memory']['skips'], 0)
        self.assertTrue(any(row['score'] == .99 for row in result['samples']))

    def test_dataset_holds_out_scene_domains_and_angular_sectors(self):
        seen_domains = []
        intervals = {'train': (250, 1400), 'development': (1450, 2300), 'test': (2350, 3450)}
        for split, (lo, hi) in intervals.items():
            features, labels, rows, examples = bench.dataset(split, 1)
            self.assertEqual(features.shape[0], 8)
            self.assertEqual(labels.shape, (8,))
            self.assertEqual(set(labels), {0, 1})
            self.assertFalse(examples)
            self.assertTrue(all(lo <= row['pose'][0] <= hi for row in rows))
            self.assertTrue(all(row['split'] == split and row['group'] == 0 for row in rows))
            seen_domains.append(set(rows[0]['domain'].values()))
        for i, first in enumerate(seen_domains):
            for second in seen_domains[i + 1:]:
                self.assertTrue(first.isdisjoint(second))

    def test_dataset_oracle_label_is_not_appended_to_rgb_features(self):
        class IdenticalPixelsScene:
            def __init__(self, **kwargs):
                self.objects = kwargs['occluders']

            def render(self, pose, now):
                return np.full((8, 12, 3), [80, 100, 120], dtype=np.uint8)

            def truth(self, pose, now):
                return types.SimpleNamespace(coverage=.95 if self.objects else 0)

        with mock.patch.object(bench, 'ObstructionScene', IdenticalPixelsScene):
            features, labels, rows, _ = bench.dataset('train', 1)
        self.assertEqual(set(labels), {0, 1})
        np.testing.assert_array_equal(features, np.repeat(features[:1], 8, axis=0))
        self.assertEqual(len({row['image_sha256'] for row in rows}), 1)

    def test_paired_dataset_keeps_identical_current_images_labels_and_domains(self):
        plainX, plainY, plain_rows, plain_examples = bench.dataset('test', 2, keep_examples=True,
                                                                 namespace='paired-identity-fixture')
        pairedX, pairedY, paired_rows, paired_examples = bench.dataset('test', 2, keep_examples=True,
                                                                      namespace='paired-identity-fixture', reference=True)
        self.assertEqual(plain_rows, paired_rows)
        np.testing.assert_array_equal(plainY, pairedY)
        np.testing.assert_array_equal(plainX, pairedX[:, :plainX.shape[1]])
        self.assertGreater(pairedX.shape[1], plainX.shape[1])
        for plain, paired in zip(plain_examples, paired_examples):
            self.assertEqual(len(plain), 3)
            self.assertEqual(len(paired), 4)
            self.assertEqual(plain[0], paired[0])
            self.assertEqual(plain[2], paired[2])
            np.testing.assert_array_equal(plain[1], paired[1])
            self.assertEqual(paired[3].shape, paired[1].shape)

    def test_reference_is_acquired_once_before_injection_and_invalidated_on_remount(self):
        anchor = bench.POSES[2]

        class RecordingScene(PixelScene):
            def __init__(self):
                super().__init__(lambda pose, now, version: pose == anchor and version == 0)
                self.calls = []
                self.last_now = None

            def render(self, pose, now, version):
                self.calls.append((pose, now, version))
                self.last_now = now
                return np.full((8, 12, 3), 80, dtype=np.uint8) if now < 0 else super().render(pose, now, version)

        scene = RecordingScene()

        class PairModel:
            requires_reference = True

            def __init__(self):
                self.calls = []

            def predict(self, rgb):
                raise AssertionError('paired model must not use a single-image fallback')

            def predict_pair(self, current, reference, registered):
                self.calls.append((scene.last_now, registered, reference is None))
                if not registered:
                    if reference is not None:
                        raise AssertionError('invalidated reference must not reach prediction')
                    return None
                np.testing.assert_array_equal(reference, np.full((8, 12, 3), 80, dtype=np.uint8))
                return .99 if np.median(current) > 150 else .01

        model = PairModel()
        result = self.episode('changed_setup', scene, end=95, model=model)
        acquired = [row for row in scene.calls if row[1] < 0]
        self.assertEqual(len(acquired), len(bench.POSES))
        self.assertTrue(all(now == -60 and version == 0 for pose, now, version in acquired))
        self.assertTrue(all(now < 0 for pose, now, version in scene.calls[:len(bench.POSES)]))
        for nominal, (actual, now, version) in zip(bench.POSES, acquired):
            self.assertLessEqual(abs(actual[0] - nominal[0]), 8)
            self.assertLessEqual(abs(actual[1] - nominal[1]), 6)
        self.assertTrue(any(registered for now, registered, missing in model.calls if now < 65))
        self.assertTrue(all(not registered and missing for now, registered, missing in model.calls if now >= 65))
        self.assertEqual(result['first_inspection_after_change_s'], .5)
        self.assertIsNone(result['clear_accepted_after_change_s'])
        after = [row for row in result['samples'] if row['at_s'] >= 65]
        self.assertTrue(all(row['score'] is None and row['accepted'] is False and not row['skip'] for row in after))

    def test_reference_wrapper_advertises_required_pair_contract(self):
        from sim.obstruction_reference import ReferenceModel
        self.assertIs(getattr(ReferenceModel, 'requires_reference', False), True)

    def test_reference_stress_has_declared_larger_acquisition_pose_error(self):
        class RecordingScene(PixelScene):
            def __init__(self):
                super().__init__(lambda *args: False)
                self.acquisitions = []

            def render(self, pose, now, version):
                if now < 0:
                    self.acquisitions.append((pose, now, version))
                return super().render(pose, now, version)

        model = types.SimpleNamespace(requires_reference=True,
                                      predict_pair=lambda current, reference, registered: None)
        scene = RecordingScene()
        self.episode('empty', scene, end=5, model=model, stress=True)
        self.assertEqual(len(scene.acquisitions), len(bench.POSES))
        offsets = [np.asarray(actual) - nominal for nominal, (actual, _, _) in zip(bench.POSES, scene.acquisitions)]
        self.assertTrue(all(np.max(np.abs(offset)) <= 40 for offset in offsets))
        self.assertTrue(any(np.max(np.abs(offset)) > 8 for offset in offsets))

    def test_summary_separates_inspection_reduction_from_actual_exclusion(self):
        counts = {policy: counters() for policy in bench.POLICIES}
        counts['visual_memory'] = counters(inspections=6, skips=2, useful_skips=2, deferred_probes=2)
        episode = {'family': 'persistent', 'counts': counts, 'first_inspection_after_change_s': None,
                   'learned_before_change': False}
        summary = bench.lifecycle_summary([episode])
        visual = summary['policies']['visual_memory']
        self.assertAlmostEqual(visual['inspection_reduction'], .4)
        self.assertAlmostEqual(visual['actual_exclusion_reduction'], .2)
        self.assertEqual(summary['by_family']['persistent']['visual_skips'], 2)

    def test_recovery_denominator_keeps_learned_cases_that_never_recover(self):
        counts = {policy: counters() for policy in bench.POLICIES}
        cases = [{'family': 'removed', 'counts': counts, 'learned_before_change': learned,
                  'first_inspection_after_change_s': delay}
                 for learned, delay in [(True, 30), (True, None), (False, None)]]
        summary = bench.lifecycle_summary(cases)['by_family']['removed']
        self.assertEqual(summary['episodes'], 3)
        self.assertEqual(summary['learned_before_change'], 2)
        self.assertEqual(summary['learned_cases_requiring_recovery'], 2)
        self.assertEqual(summary['learned_cases_recovered_within_bound'], 1)
        self.assertEqual(summary['max_first_inspection_after_change_s'], 30)

    def test_serial_bank_delegates_canonical_indices_and_untouched_domain_inputs(self):
        model, background, calib = object(), object(), object()
        config = self.config()
        count = len(bench.FAMILIES) + 2

        def run(fitted, supplied_config, split, index, photo, camera, stress, keep_rows):
            self.assertIs(fitted, model)
            self.assertIs(supplied_config, config)
            self.assertIs(photo, background)
            self.assertIs(camera, calib)
            self.assertTrue(stress)
            self.assertEqual(keep_rows, index < len(bench.FAMILIES))
            return {'id': f'{split}-{index}', 'domain': bench.domain_seeds(split, index)}

        with mock.patch.object(bench, 'episode', side_effect=run) as episodes, \
                mock.patch.object(bench, 'ProcessPoolExecutor') as pool:
            rows = bench.run_bank(model, config, 'frozen-bank-fixture', count, background, calib,
                                  stress=True, jobs=1)
        pool.assert_not_called()
        self.assertEqual(episodes.call_count, count)
        self.assertEqual([row['id'] for row in rows], [f'frozen-bank-fixture-{i}' for i in range(count)])
        self.assertEqual([row['domain'] for row in rows],
                         [bench.domain_seeds('frozen-bank-fixture', i) for i in range(count)])
        self.assertNotEqual(rows[0]['domain'], bench.domain_seeds('another-bank-fixture', 0))

    def test_cli_rejects_invalid_worker_count_before_input_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            for workers in ('0', '9'):
                with self.subTest(workers=workers), mock.patch.object(bench, 'dataset') as data, \
                        contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as failure:
                    bench.main(['--out', str(Path(directory) / 'invalid'), '--jobs', workers])
                self.assertEqual(failure.exception.code, 2)
                data.assert_not_called()

    def test_model_comparison_fits_training_only_and_selects_using_development_only(self):
        train, labels = np.array([[10, 0], [10, 1]]), np.array([0, 1])
        dev, dev_labels = np.array([[20, 0], [20, 1]]), np.array([0, 1])

        class LightModel:
            def __init__(self, scores):
                self.development_scores = np.array(scores)

            def scores(self, features):
                np.testing.assert_array_equal(features, dev)
                return self.development_scores

        linear, depth4, depth8 = LightModel([.96, .9]), LightModel([.1, .99]), LightModel([.1, .99])
        forest_fit = mock.Mock(side_effect=[depth4, depth8])
        module = types.ModuleType('sim.obstruction_forest')
        module.fit = forest_fit
        with mock.patch.dict(sys.modules, {'sim.obstruction_forest': module}), \
                mock.patch.object(bench, 'fit', return_value=linear) as linear_fit, \
                mock.patch.object(bench, 'select_threshold', wraps=bench.select_threshold) as selection:
            model, selected = bench.choose_model(train, labels, dev, dev_labels)
        self.assertIs(model, depth4)
        self.assertEqual(selected['model'], 'forest_depth4')
        self.assertEqual(selected['development']['fp'], 0)
        self.assertEqual(selected['development']['tp'], 1)
        self.assertEqual(linear_fit.call_count, 1)
        for call in [linear_fit.call_args, *forest_fit.call_args_list]:
            np.testing.assert_array_equal(call.args[0], train)
            np.testing.assert_array_equal(call.args[1], labels)
        self.assertEqual(selection.call_count, 3)
        for call in selection.call_args_list:
            np.testing.assert_array_equal(call.args[0], dev_labels)

    def test_model_selection_disables_unsafe_candidate_instead_of_selecting_its_threshold_one(self):
        features, labels = np.array([[0, 0], [0, 1]]), np.array([0, 1])
        unsafe = types.SimpleNamespace(scores=lambda X: np.ones(len(X)))
        valid = types.SimpleNamespace(scores=lambda X: np.array([.1, .99]))
        module = types.ModuleType('sim.obstruction_forest')
        module.fit = mock.Mock(side_effect=[valid, valid])
        with mock.patch.dict(sys.modules, {'sim.obstruction_forest': module}), \
                mock.patch.object(bench, 'fit', return_value=unsafe):
            model, selected = bench.choose_model(features, labels, features, labels)
        self.assertIs(model, valid)
        self.assertTrue(selected['enabled'])
        self.assertEqual(selected['model'], 'forest_depth4')
        candidates = {row['model']: row for row in selected['model_candidates']}
        self.assertFalse(candidates['linear']['enabled'])
        self.assertEqual(candidates['linear']['development']['fp'], 1)
        self.assertTrue(candidates['forest_depth4']['enabled'])

    def test_all_unsafe_models_leave_learning_disabled(self):
        features, labels = np.array([[0, 0], [0, 1]]), np.array([0, 1])
        unsafe = types.SimpleNamespace(scores=lambda X: np.ones(len(X)))
        module = types.ModuleType('sim.obstruction_forest')
        module.fit = mock.Mock(return_value=unsafe)
        with mock.patch.dict(sys.modules, {'sim.obstruction_forest': module}), \
                mock.patch.object(bench, 'fit', return_value=unsafe), \
                self.assertRaisesRegex(ValueError, 'leave obstruction learning disabled'):
            bench.choose_model(features, labels, features, labels)

    def test_reference_candidate_fits_only_paired_train_and_uses_paired_development(self):
        train, labels = np.array([[10, 0], [10, 1]]), np.array([0, 1])
        dev = np.array([[20, 0], [20, 1]])
        ref_train, ref_dev = np.array([[1010, 0, 0], [1010, 1, 1]]), np.array([[1020, 0, 0], [1020, 1, 1]])

        class SingleModel:
            def scores(self, features):
                np.testing.assert_array_equal(features, dev)
                return np.array([.96, .9])

        class PairModel:
            requires_reference = True

            def scores(self, features):
                np.testing.assert_array_equal(features, ref_dev)
                return np.array([.1, .99])

        pair_model = PairModel()
        forest_module, reference_module = types.ModuleType('sim.obstruction_forest'), types.ModuleType('sim.obstruction_reference')
        forest_module.fit = mock.Mock(return_value=SingleModel())
        reference_module.fit = mock.Mock(return_value=pair_model)
        with mock.patch.dict(sys.modules, {'sim.obstruction_forest': forest_module,
                                          'sim.obstruction_reference': reference_module}), \
                mock.patch.object(bench, 'fit', return_value=SingleModel()), \
                mock.patch.object(bench, 'select_threshold', wraps=bench.select_threshold) as selection:
            chosen, selected = bench.choose_model(train, labels, dev, labels, ref_train, ref_dev)
        self.assertIs(chosen, pair_model)
        self.assertEqual(selected['model'], 'registered_reference_forest')
        self.assertEqual(selected['development']['fp'], 0)
        self.assertEqual(selected['development']['tp'], 1)
        np.testing.assert_array_equal(reference_module.fit.call_args.args[0], ref_train)
        np.testing.assert_array_equal(reference_module.fit.call_args.args[1], labels)
        self.assertEqual(selection.call_count, 4)
        for call in selection.call_args_list:
            np.testing.assert_array_equal(call.args[0], labels)

    def test_main_fits_only_train_freezes_dev_selection_and_reports_test_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'run'
            call_order = []

            class LightModel:
                def scores(self, features):
                    return np.asarray(features)[:, 1] * .94 + .05

                def predict(self, rgb):
                    raise AssertionError('no example RGB or real-room assets belong in this fixture')

                def to_dict(self):
                    return {'fixture': 'light model'}

            model = LightModel()

            def data(split, groups, keep_examples=False, namespace=None, reference=False):
                call_order.append((split, reference))
                if split == 'test':
                    frozen = json.loads((out / 'selection.json').read_text())
                    self.assertTrue(frozen['frozen_before_test'])
                    self.assertEqual(frozen['threshold'], .99)
                    self.assertEqual(namespace, 'bench-fixture-independent')
                code = {'train': 10, 'development': 20, 'test': 30}[split]
                code += 1000 if reference else 0
                # Test truth is deliberately reversed: it must fail the frozen classifier.
                labels = np.array([1, 0] if split == 'test' else [0, 1])
                return np.array([[code, 0], [code, 1]]), labels, [{'split': split}] * 2, []

            def choose(trainX, trainY, devX, devY, ref_train, ref_dev):
                np.testing.assert_array_equal(trainX, [[10, 0], [10, 1]])
                np.testing.assert_array_equal(trainY, [0, 1])
                np.testing.assert_array_equal(devX, [[20, 0], [20, 1]])
                np.testing.assert_array_equal(ref_train, [[1010, 0], [1010, 1]])
                np.testing.assert_array_equal(ref_dev, [[1020, 0], [1020, 1]])
                selected = bench.select_threshold(devY, model.scores(devX))
                selected['model'] = 'light-fixture'
                return model, selected

            def lifecycle(fitted, config, split, index, *args, **kwargs):
                self.assertIs(fitted, model)
                self.assertEqual(config.high_threshold, .99)
                self.assertTrue((out / 'selection.json').exists())
                self.assertIn(split, ('lifecycle-bench-fixture-independent', 'stress-bench-fixture-independent'))
                counts = {policy: counters() for policy in bench.POLICIES}
                counts['visual_memory'] = counters(inspections=8, skips=2, useful_skips=2)
                return {'id': f'{split}-{index}', 'family': 'removed', 'counts': counts,
                        'first_inspection_after_change_s': None, 'learned_before_change': True}

            with mock.patch.object(bench, 'dataset', side_effect=data), \
                    mock.patch.object(bench, 'choose_model', side_effect=choose) as choosing, \
                    mock.patch.object(bench, 'select_threshold', wraps=bench.select_threshold) as selection, \
                    mock.patch.object(bench, 'episode', side_effect=lifecycle), \
                    mock.patch.object(bench, 'sha', return_value='fixture-hash'), \
                    contextlib.redirect_stdout(io.StringIO()):
                status = bench.main(['--out', str(out), '--train-groups', '1',
                                     '--development-groups', '1', '--test-groups', '1', '--episodes', '1',
                                     '--test-namespace', 'bench-fixture-independent'])
            self.assertEqual(status, 0)
            self.assertEqual(call_order, [('train', False), ('development', False),
                                          ('train', True), ('development', True), ('test', False)])
            self.assertEqual(choosing.call_count, 1)
            self.assertEqual(selection.call_count, 1)
            np.testing.assert_array_equal(selection.call_args.args[0], [0, 1])
            report = json.loads((out / 'summary.json').read_text())
            manifest = json.loads((out / 'manifest.json').read_text())
            self.assertEqual(report['selected_threshold'], .99)
            self.assertEqual(report['pixel_test']['fp'], 1)
            self.assertFalse(report['simulation_gate'])
            self.assertFalse(report['simulation_checks']['bounded_reinspection'])
            self.assertFalse(report['production_changed'])
            self.assertTrue(report['inputs_unchanged'])
            self.assertEqual(manifest['splits']['test_namespace'], 'bench-fixture-independent')
            self.assertTrue(manifest['reference_protocol']['candidate_supplied'])
            self.assertFalse(manifest['reference_protocol']['selected'])
            self.assertTrue(any('synthetic holdouts' in limit for limit in manifest['limits']))
            self.assertTrue(any('original background is unlabelled' in limit for limit in manifest['limits']))

    def test_unsafe_dev_threshold_stops_before_test_or_lifecycle_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'run'

            def data(split, *args, **kwargs):
                if split == 'test':
                    raise AssertionError('unsafe dev calibration must stop before test')
                return np.array([[0, 0], [0, 1]]), np.array([0, 1]), [], []

            def unsafe_choose(X, y, devX, devY, *reference):
                bench.select_threshold(devY, np.ones(len(devY)))

            with mock.patch.object(bench, 'dataset', side_effect=data), \
                    mock.patch.object(bench, 'choose_model', side_effect=unsafe_choose), \
                    mock.patch.object(bench, 'episode') as lifecycle, \
                    mock.patch.object(bench, 'sha', return_value='fixture-hash'), \
                    contextlib.redirect_stdout(io.StringIO()), \
                    self.assertRaisesRegex(ValueError, 'disable the classifier'):
                bench.main(['--out', str(out), '--train-groups', '1', '--development-groups', '1',
                            '--test-groups', '1', '--episodes', '1'])
            lifecycle.assert_not_called()
            self.assertFalse((out / 'model.json').exists())
            self.assertFalse((out / 'selection.json').exists())


if __name__ == '__main__':
    unittest.main()
