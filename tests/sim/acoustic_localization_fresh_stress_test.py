"""Fresh design invariants; these tests never generate the locked fresh waveforms."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from sim.acoustics.localization_fresh_stress import CONDITIONS, FAMILIES, design, match_methods, result_row, room_model
from sim.acoustics.localization_model import build_paths, render_transfer


class FreshStressTests(unittest.TestCase):
    def test_seeded_design_is_balanced_and_sources_are_continuous_inside_truth_rooms(self):
        rooms, cases = design(77)  # Separate unit-test seed; no fresh cohort waveforms.
        self.assertEqual((rooms, cases), design(77))
        source = [case for case in cases if case['source_m'] is not None]
        self.assertEqual(len(source), 128)
        self.assertEqual(len(cases) - len(source), 16)
        self.assertEqual(len(set(case['source_group'] for case in source)), 128)
        for room in rooms:
            chosen = [case for case in source if case['room_id'] == room['room_id']]
            self.assertEqual({(case['source_family'], case['condition']) for case in chosen}, {(a, b) for a in FAMILIES for b in CONDITIONS})
            for case in chosen:
                truth = room_model(room, geometry=case['geometry'], coefficients=case['pressure_coefficients'])
                x, y, z = case['source_m'];bounds = truth.declared_bounds_m
                self.assertTrue(bounds['x_min'] < x < bounds['x_max'] and bounds['z_min'] < z < bounds['z_max'])
                self.assertTrue(.25 <= y - bounds['floor_y'] <= 1.6)
                self.assertGreaterEqual(np.hypot(x, z), .75)

    def test_geometry_perturbations_keep_wall_corners_closed(self):
        rooms, cases = design(78)
        case = next(case for case in cases if case['condition'] == 'geometry_10cm')
        model = room_model(rooms[0], geometry=case['geometry'])
        extents = model.declared_wall_extents
        for index, wall in enumerate(extents):
            self.assertEqual(wall['b_xz_m'], extents[(index + 1) % 4]['a_xz_m'])

    def test_unit_base_dictionary_profile_matches_nominal_half_pressure_truth(self):
        rooms, _ = design(80)
        base = room_model(rooms[0])
        truth = room_model(rooms[0], coefficients={surface.surface_id: .5 for surface in base.surfaces})
        self.assertTrue(all(surface.reflection == 1 for surface in base.surfaces))
        receiver = [0, .012, .0114];point = [1, rooms[0]['floor_y'] + .45, 1]
        frequency = np.fft.rfftfreq(8192, 1 / 16000)
        paths = build_paths(base, point, receiver, timing='direct_relative')
        profiled = render_transfer([p for p in paths if p.order == 0], frequency) + .5 * render_transfer([p for p in paths if p.order > 0], frequency)
        exact = render_transfer(build_paths(truth, point, receiver, timing='direct_relative'), frequency)
        np.testing.assert_allclose(profiled, exact, rtol=1e-13, atol=1e-14)

    def test_fixed_harmonic_stress_spans_beyond_frozen_detector_prior(self):
        _, cases = design(79)
        fixed = [case for case in cases if case['source_family'] == 'fixed_harmonic']
        self.assertEqual(min(case['f0_hz'][0] for case in fixed), 120)
        self.assertEqual(max(case['f0_hz'][0] for case in fixed), 360)
        self.assertTrue(all(len(set(case['f0_hz'])) == 1 for case in fixed))

    def test_matching_api_receives_only_identical_power_inputs_and_fixed_flags(self):
        class FakeMatcher:
            def __init__(self):self.calls = []
            def match(self, power, **kwargs):self.calls.append((power.copy(), kwargs));return {}
        coarse, fine = FakeMatcher(), FakeMatcher()
        power = np.arange(40.).reshape(4, 10);original = power.copy()
        result = match_methods(coarse, fine, power, lambda *args: None)
        self.assertEqual(len(result), 4)
        self.assertEqual(len(coarse.calls), 1);self.assertEqual(len(fine.calls), 3)
        for saved, options in coarse.calls + fine.calls:
            np.testing.assert_array_equal(saved, original)
            self.assertEqual(set(options), {'harmonic', 'refine', 'progress'})
        np.testing.assert_array_equal(power, original)

    def test_background_and_missing_predictions_have_no_fabricated_errors(self):
        match = {'final': {'position_m': None}, 'gated_bursts': 0, 'new_candidates_m': [], 'global_center_indexes': [], 'mean_masked_fraction': 0.}
        case = {'case_id': 'toy', 'room_id': 'toy-room', 'source_group': 'toy', 'source_family': 'toy', 'condition': 'toy', 'source_m': [1, 0, 2]}
        self.assertIsNone(result_row(case, 'toy', match, np.zeros(3))['distance_error_m'])
        case['source_m'] = None;match['final']['position_m'] = [1, 0, 2]
        self.assertIsNone(result_row(case, 'toy', match, np.zeros(3))['distance_error_m'])


if __name__ == '__main__':
    unittest.main()
