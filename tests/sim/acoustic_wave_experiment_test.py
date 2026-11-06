"""Independent protocol, phase, input and denominator contracts without propagation."""
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
from sim.acoustics import wave_experiment as experiment
from sim.acoustics.wave3d import Grid
from sim.acoustics.wave_room import air_point


class AcousticWaveExperimentTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.inventory_path = self.root / 'inventory.json'
        self.archive = self.root / 'original-windows.npz'
        signals = np.zeros((4, 512), dtype='<f4')
        signals[:, 128] = [.2, .4, -.2, -.4]
        np.savez(self.archive, **{f'source{i}': row for i, row in enumerate(signals)})
        self.original_signals = signals
        self.inventory = {
            'complete': True,
            'window_archive': {'path': str(self.archive), 'sha256': experiment.sha(self.archive)},
            'entries': [{'id': f'source{i}', 'npz_array': f'source{i}',
                         'window_sha256': __import__('hashlib').sha256(row.tobytes()).hexdigest(),
                         'sample_rate_hz': 1000, 'source_kind': 'observed_room_recording_used_as_virtual_emitter_proxy',
                         'category': 'dog_sound_machine_hint' if i < 2 else 'other_unverified',
                         'overlap_group': f'overlap-{i}', 'recommended_split_group': f'day-{i % 2}'}
                        for i, row in enumerate(signals)]}
        self.inventory_path.write_text(json.dumps(self.inventory))

    def rooms(self, blocked=False):
        solid = np.zeros((13, 7, 13), dtype=bool)
        if blocked:
            solid[6:, :, :] = True
        grid = Grid(origin_m=(-3, -1, -3), shape_xyz=(13, 7, 13), spacing_m=.5, solid=solid)
        return types.SimpleNamespace(grid=grid,
                                     metadata={'declared_segment_bounds_xz_m': [[-2, 2], [-2, 2]],
                                               'requested_floor_y_m': -.8})

    def test_source_spectra_preserves_signed_native_scale_and_analytic_phase(self):
        before = experiment.sha(self.archive), experiment.sha(self.inventory_path)
        spectra, metadata, _ = experiment.source_spectra(self.inventory_path)
        amplitude, sample, rate, count = float(self.original_signals[0, 128]), 128, 1000, 512
        gate = .5 * (1 - np.cos(2 * np.pi * sample / (count - 1)))
        expected = amplitude * gate / rate * np.exp(-2j * np.pi * experiment.FREQUENCIES * sample / rate)
        np.testing.assert_allclose(spectra[0], expected, rtol=1e-13, atol=1e-17)
        np.testing.assert_allclose(spectra[1], 2 * spectra[0], rtol=1e-14)
        np.testing.assert_allclose(spectra[2], -spectra[0], rtol=1e-14)
        np.testing.assert_allclose(spectra[3], -2 * spectra[0], rtol=1e-14)
        self.assertEqual(before, (experiment.sha(self.archive), experiment.sha(self.inventory_path)))
        self.assertTrue(all(row['physical_source_position_m'] is None and row['actual_sound_category'] is None
                            for row in metadata))
        self.assertTrue(all('no per-recording normalization' in row['source_filter'] for row in metadata))
        again, repeated, _ = experiment.source_spectra(self.inventory_path)
        np.testing.assert_array_equal(again, spectra)
        self.assertEqual(metadata, repeated)

    def test_source_archive_and_each_window_have_independent_hash_checks(self):
        self.archive.write_bytes(self.archive.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'archive hash'):
            experiment.source_spectra(self.inventory_path)
        self.inventory['window_archive']['sha256'] = experiment.sha(self.archive)
        self.inventory['entries'][0]['window_sha256'] = '0' * 64
        self.inventory_path.write_text(json.dumps(self.inventory))
        with self.assertRaisesRegex(ValueError, 'native source samples'):
            experiment.source_spectra(self.inventory_path)

    def test_green_transform_uses_native_physical_fft_bins_and_signed_phase(self):
        rate = 512.
        t = np.arange(int(rate)) / rate
        signals = np.array([np.sin(2 * np.pi * 120 * t), .5 * np.cos(2 * np.pi * 88 * t)])
        result = types.SimpleNamespace(signals=signals, dt_s=1 / rate, sample_rate_hz=rate)
        room = types.SimpleNamespace(grid=types.SimpleNamespace(sample_rate_hz=rate))
        target = self.root / 'green.npz'
        with mock.patch.object(experiment, 'simulate', return_value=result) as propagation:
            _, transfer = experiment.propagate(room, [0, 0, 0], [[1, 0, 0], [0, 1, 0]], target, self.root)
        i120 = list(experiment.FREQUENCIES).index(120)
        i88 = list(experiment.FREQUENCIES).index(88)
        self.assertAlmostEqual(transfer[0, i120].imag, -rate / 2, places=9)
        self.assertAlmostEqual(transfer[1, i88].real, rate / 4, places=9)
        self.assertAlmostEqual(transfer[0, i120].real, 0, places=9)
        impulse = propagation.call_args.args[2]
        self.assertEqual(impulse[0], 1)
        self.assertEqual(np.count_nonzero(impulse), 1)
        with np.load(target) as archive:
            np.testing.assert_array_equal(archive['signals'], signals)
        bad_rate = 512.25
        bad_result = types.SimpleNamespace(signals=signals, dt_s=1 / bad_rate, sample_rate_hz=bad_rate)
        bad_room = types.SimpleNamespace(grid=types.SimpleNamespace(sample_rate_hz=bad_rate))
        with mock.patch.object(experiment, 'simulate', return_value=bad_result), \
                self.assertRaisesRegex(ValueError, 'physical frequencies'):
            experiment.propagate(bad_room, [0, 0, 0], [[1, 0, 0]], self.root / 'bad.npz', self.root)

    def test_cases_are_frozen_paired_and_all_offgrid_truth_is_common_air(self):
        rooms = {name: self.rooms(blocked=name == 'fine_2cm')
                 for name in experiment.scene_definitions()}
        positions = np.array([[-1.5, -.4, 1], [-1, -.4, -1], [1, -.4, 1]])
        before = positions.copy()
        anchors, cases = experiment.freeze_cases(rooms, positions, 3, seed=2026112801)
        repeat = experiment.freeze_cases(rooms, positions, 3, seed=2026112801)
        self.assertEqual((anchors, cases), repeat)
        np.testing.assert_array_equal(positions, before)
        self.assertEqual(len(cases), 4 * 3 * 4 + len(positions) * 4 + 24)
        self.assertEqual(len(cases), len({case['id'] for case in cases}))
        self.assertTrue(all(air_point(room.grid, point) for room in rooms.values() for point in anchors))
        self.assertTrue(all(point[0] <= -.5 for point in anchors))
        for scene in ('coarse_offgrid_control', 'fine_same_prior', 'fine_2cm', 'fine_10cm_completion'):
            for anchor in range(3):
                group = [case for case in cases if case['scene'] == scene and case['anchor'] == anchor
                         and case['kind'] == 'single']
                self.assertEqual({case['source_index'] for case in group}, {0, 1, 2, 3})
                self.assertTrue(all(case['true_position_m'] == anchors[anchor] for case in group))
                self.assertTrue(all(-.012 <= case['onset_s'] <= .012 for case in group))
        controls = [case for case in cases if case['kind'] != 'single']
        self.assertEqual(len(controls), 24)
        self.assertTrue(all(case['true_position_m'] is None for case in controls))
        self.assertTrue(all(case['second_probe_index'] != case['probe_index'] for case in controls))

    def test_absolute_background_and_silence_do_not_scale_with_source_or_transfer(self):
        case = dict(kind='background', source_index=0, noise_seed=43)
        a = experiment.observed(case, {}, np.ones((4, len(experiment.FREQUENCIES))), 2)
        b = experiment.observed(case, {'unrelated': 'not read'}, np.ones((4, len(experiment.FREQUENCIES))) * 1e12, 50)
        np.testing.assert_array_equal(a, b)
        self.assertGreater(np.linalg.norm(a), 0)
        silence = experiment.observed(dict(case, kind='silence'), {}, np.ones((4, len(experiment.FREQUENCIES))), 2)
        np.testing.assert_array_equal(silence, np.zeros(len(experiment.FREQUENCIES)))

    def test_observation_routes_coarse_anchor_and_multisource_spectra_without_fake_single_truth(self):
        n = len(experiment.FREQUENCIES)
        sources = np.array([np.ones(n) * value for value in (2, 3, 5, 7)], dtype=complex)
        transfers = {'nominal': np.array([np.ones(n) * value for value in (10, 20, 30, 40)]),
                     'fine_same_prior': np.array([np.ones(n) * value for value in (11, 17)])}
        case = dict(kind='single', source_index=0, noise_seed=1, scene='coarse_offgrid_control',
                    probe_index=1, gain=2., onset_s=.003, snr_db=None)
        expected = 2 * 40 * 2 * np.exp(-2j * np.pi * experiment.FREQUENCIES * .003)
        np.testing.assert_allclose(experiment.observed(case, transfers, sources, 2), expected)
        multiple = dict(case, kind='multiple', scene='fine_same_prior', probe_index=0,
                        second_probe_index=1, gain=1., onset_s=0, true_position_m=None)
        np.testing.assert_allclose(experiment.observed(multiple, transfers, sources, 2), 11 * 2 + .8 * 17 * 3)
        self.assertIsNone(multiple['true_position_m'])

    def test_source_scaled_noise_has_declared_snr_without_changing_clean_phase(self):
        n = len(experiment.FREQUENCIES)
        sources = np.full((4, n), 2 + 3j)
        transfers = {'fine_same_prior': np.array([np.linspace(.5, 1.5, n) * (1 - .3j)])}
        case = dict(kind='single', scene='fine_same_prior', probe_index=0, source_index=0,
                    gain=.7, onset_s=.004, snr_db=None, noise_seed=21)
        clean = experiment.observed(case, transfers, sources, 1)
        noisy = experiment.observed(dict(case, snr_db=10), transfers, sources, 1)
        self.assertAlmostEqual(np.linalg.norm(noisy - clean) / np.linalg.norm(clean), 10**(-10 / 20), places=13)
        np.testing.assert_array_equal(noisy, experiment.observed(dict(case, snr_db=10), transfers, sources, 1))

    def test_information_ladder_only_known_branch_receives_exact_source(self):
        observation, source, other, green, ray, positions = (object() for _ in range(6))
        with mock.patch.object(experiment, 'known_source_spectra', return_value={}) as matcher:
            experiment.infer('wave_known_source', observation, source, other, green, ray, positions)
            self.assertIs(matcher.call_args.args[1], source)
            experiment.infer('wave_other_recording_template', observation, source, other, green, ray, positions)
            self.assertIs(matcher.call_args.args[1], other)
        with mock.patch.object(experiment, 'blind_fir_spectra', return_value={}) as blind:
            experiment.infer('wave_blind32', observation, source, other, green, ray, positions)
            self.assertEqual(blind.call_args.args, (observation, green, experiment.FREQUENCIES, 1000.))
            self.assertEqual(blind.call_args.kwargs['source_length'], 32)
        with mock.patch.object(experiment, 'arbitrary_source_control_spectra', return_value={}) as arbitrary:
            experiment.infer('arbitrary_source_control', observation, source, other, green, ray, positions)
            self.assertIs(arbitrary.call_args.args[0], observation)
            self.assertIs(arbitrary.call_args.args[1], green)

    def test_summary_retains_failures_and_counts_multisource_and_null_controls(self):
        def row(kind, distance, bearing, position):
            return dict(kind=kind, distance_error_m=distance, bearing_error_deg=bearing,
                        position_m=position, accepted=False)
        rows = [row('single', .2, 10, [1, 0, 0]), row('single', None, None, None),
                row('single', 3, 90, [2, 0, 0]), row('silence', None, None, None),
                row('background', None, None, [3, 0, 0]), row('multiple', None, None, [4, 0, 0])]
        summary = experiment.summarize(rows)
        self.assertEqual(summary['total_cases'], 6)
        self.assertEqual(summary['single_cases'], 3)
        self.assertEqual(summary['localized_cases'], 2)
        self.assertEqual(summary['unlocalized'], 1)
        self.assertAlmostEqual(summary['within_1m_fraction'], 1 / 3)
        self.assertAlmostEqual(summary['within_30deg_fraction'], 1 / 3)
        self.assertEqual(summary['null_cases'], 2)
        self.assertEqual(summary['null_conditional_candidates'], 1)
        self.assertEqual(summary['multiple_cases'], 1)
        self.assertEqual(summary['multiple_conditional_candidates'], 1)
        self.assertEqual(summary['real_accepted'], 0)

    def run_main(self, mutate=None):
        room_path = self.root / 'room.json'
        room_path.write_text(json.dumps({'room': {'floor_height_m': .8, 'ceiling_m': 2.6, 'walls': []}}))
        original_room = room_path.read_bytes()
        original_inventory = self.inventory_path.read_bytes()
        output = self.root / ('changed' if mutate else 'run')
        positions = np.array([[-1.5, -.4, 1], [1, -.4, -1]])
        propagation_calls, inference_calls, ray_reads = [], [], []

        def room(path, **kwargs):
            self.assertEqual(Path(path).read_bytes(), original_room)
            return self.rooms()

        def propagate(voxels, microphone, probes, target, cache):
            protocol = json.loads((output / 'protocol.json').read_text())
            cases = json.loads((output / 'cases.json').read_text())
            self.assertTrue(protocol['cases_frozen_before_propagation'])
            self.assertEqual(len(cases), 4 * 2 * 4 + 2 * 4 + 24)
            self.assertEqual(protocol['candidates'], positions.tolist())
            for name in ('rooms.json', 'source_inventory.json', 'source_spectra.npz', 'frozen_room.json', 'frozen_sources.json'):
                self.assertTrue((output / name).exists())
            self.assertEqual((output / 'frozen_room.json').read_bytes(), original_room)
            self.assertEqual((output / 'frozen_sources.json').read_bytes(), original_inventory)
            propagation_calls.append(np.asarray(probes).copy())
            if mutate and len(propagation_calls) == 1:
                if mutate == 'room':
                    room_path.write_text(json.dumps({'room': {'floor_height_m': 1.4}}))
                elif mutate == 'archive':
                    self.archive.write_bytes(self.archive.read_bytes() + b'changed after loading')
                else:
                    self.inventory_path.write_text(original_inventory.decode() + '\n')
            rate = voxels.grid.sample_rate_hz
            signals = np.zeros((len(probes), int(round(rate))))
            result = types.SimpleNamespace(signals=signals, sample_rate_hz=rate, dt_s=1 / rate,
                                           metadata={'run_wall_s': 0})
            return result, np.ones((len(probes), len(experiment.FREQUENCIES)), dtype=complex)

        def load_ray(path):
            ray_reads.append(Path(path).read_bytes())
            return object()

        def infer(method, y, source, template, green, ray, poses):
            np.testing.assert_array_equal(poses, positions)
            inference_calls.append((method, source.copy(), template.copy()))
            return {'best_index': 0, 'status': 'fixture_conditional', 'profiles': [.5, .1]}

        with mock.patch.object(experiment, 'build_room', side_effect=room), \
                mock.patch.object(experiment, 'candidates', return_value=positions) as grid_builder, \
                mock.patch.object(experiment, 'propagate', side_effect=propagate), \
                mock.patch.object(experiment, 'load_room', side_effect=load_ray), \
                mock.patch.object(experiment, 'build_paths', return_value=object()), \
                mock.patch.object(experiment, 'render_transfer', return_value=np.ones(len(experiment.FREQUENCIES))), \
                mock.patch.object(experiment, 'infer', side_effect=infer), contextlib.redirect_stdout(io.StringIO()):
            status = experiment.main(['--room', str(room_path), '--sources', str(self.inventory_path),
                                      '--output', str(output), '--anchors', '2', '--spacing', '1.5', '--seed', '2026112802'])
        return status, output, propagation_calls, inference_calls, ray_reads, original_room, grid_builder

    def test_main_freezes_before_any_green_solve_and_pairs_source_templates_by_group(self):
        status, output, propagations, inferences, ray_reads, original_room, builder = self.run_main()
        self.assertEqual(status, 0)
        self.assertEqual(len(propagations), 4)
        self.assertEqual(ray_reads, [original_room])
        self.assertEqual(builder.call_count, 1)
        self.assertEqual(set(builder.call_args.kwargs), {'horizontal_spacing'})
        protocol = json.loads((output / 'protocol.json').read_text())
        cases = json.loads((output / 'cases.json').read_text())
        rows = json.loads((output / 'results.json').read_text())
        self.assertEqual(len(rows), len(cases) * len(experiment.METHODS))
        spectra = np.load(output / 'source_spectra.npz')['spectra']
        for index, case in enumerate(cases):
            source = case['source_index']
            template = source ^ 1
            self.assertEqual(protocol['source_waveforms'][source]['category_hint'],
                             protocol['source_waveforms'][template]['category_hint'])
            self.assertNotEqual(protocol['source_waveforms'][source]['recommended_split_group'],
                                protocol['source_waveforms'][template]['recommended_split_group'])
            for method, supplied, other in inferences[index * len(experiment.METHODS):(index + 1) * len(experiment.METHODS)]:
                np.testing.assert_array_equal(supplied, spectra[source])
                np.testing.assert_array_equal(other, spectra[template])
        self.assertTrue(all(row['true_position_m'] is None and row['distance_error_m'] is None
                            for row in rows if row['kind'] != 'single'))
        self.assertTrue(all(row['accepted'] is False for row in rows))
        self.assertEqual(protocol['room_sha256'], experiment.sha(output / 'frozen_room.json'))
        self.assertEqual(protocol['source_inventory_sha256'], experiment.sha(output / 'frozen_sources.json'))
        self.assertTrue(json.loads((output / 'summary.json').read_text())['input_hashes_match'])

    def test_midrun_room_change_does_not_change_frozen_ray_and_fails_integrity(self):
        status, output, _, _, rays, original, _ = self.run_main(mutate='room')
        self.assertEqual(rays, [original])
        self.assertEqual(status, 1)
        self.assertFalse(json.loads((output / 'summary.json').read_text())['input_hashes_match'])

    def test_midrun_source_inventory_change_fails_integrity(self):
        status, output, _, _, _, _, _ = self.run_main(mutate='sources')
        self.assertEqual(status, 1)
        self.assertFalse(json.loads((output / 'summary.json').read_text())['input_hashes_match'])

    def test_midrun_source_archive_change_fails_integrity(self):
        status, output, _, _, _, _, _ = self.run_main(mutate='archive')
        self.assertEqual(status, 1)
        self.assertFalse(json.loads((output / 'summary.json').read_text())['input_hashes_match'])


if __name__ == '__main__':
    unittest.main()
