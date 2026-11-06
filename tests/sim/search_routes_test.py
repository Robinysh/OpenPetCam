"""Finite-model route guarantees, checked against an independent enumeration."""
import itertools
import json
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'sim'))
from search_routes import RouteProblem, evaluate, exact_order, greedy_order, improve_order, nearest_order, plan


def direct_objective(problem, route):
    """Independent state-by-state first-hit cost, including every failed state."""
    now = travel = expected = expected_travel = 0.0
    here = problem.start
    arrivals, steps = [], []
    for index in route:
        dx, dy = abs(here[0]-problem.poses[index, 0]), abs(here[1]-problem.poses[index, 1])
        distance = (max(dx, dy) if problem.travel_norm == 'max' else
                    np.hypot(dx, dy) if problem.travel_norm == 'euclidean' else dx+dy)
        now += problem.look_s + distance/problem.speed + (problem.start_s+problem.settle_s if dx+dy else 0)
        travel += dx+dy
        arrivals.append(now); steps.append(travel)
        here = problem.poses[index]
    for state, weight in enumerate(problem.weights):
        left = 1.0
        for i, index in enumerate(route):
            hit = left * problem.visibility[index, state]
            expected += weight * hit * arrivals[i]
            expected_travel += weight * hit * steps[i]
            left *= 1-problem.visibility[index, state]
        expected += weight * left * (now+problem.failure_penalty_s)
        expected_travel += weight * left * travel
    return expected + problem.travel_penalty*expected_travel


def brute(problem):
    best = None
    for count in range(problem.n+1):
        for route in itertools.permutations(range(problem.n), count):
            value = direct_objective(problem, route)
            if best is None or value < best[0]:
                best = (value, route)
    return best


class SearchRoutesTest(unittest.TestCase):
    def make(self, **kw):
        values = dict(poses=[[0, 0], [10, 0], [20, 5]], visibility=[[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                      weights=[.5, .3, .2], start=[5, 5], speed=5, start_s=.1, settle_s=.4, look_s=1,
                      failure_penalty_s=40)
        values.update(kw)
        return RouteProblem(**values)

    def test_expectation_includes_absent_and_missed_mass(self):
        problem = self.make(visibility=[[1, 0, 0], [0, 1, 0], [0, 0, 0]], weights=[.4, .3, .3])
        result = evaluate(problem, [0, 1])
        self.assertAlmostEqual(result['objective_s'], direct_objective(problem, [0, 1]))
        self.assertAlmostEqual(result['missed_probability'], .3)
        self.assertAlmostEqual(result['terminal_failure_cost_s'], 12)
        self.assertGreater(result['expected_completion_s'], result['expected_elapsed_s'])

    def test_fractional_visibility_expectation_matches_state_enumeration(self):
        problem = self.make(visibility=[[.8, .2, 0], [.2, .7, 0], [.6, .5, 0]], travel_penalty=.03)
        for route in [[], [0], [2, 0], [1, 2, 0]]:
            self.assertAlmostEqual(evaluate(problem, route)['objective_s'], direct_objective(problem, route))

    def test_exact_matches_brute_force_for_overlap_starts_and_travel_costs(self):
        for seed in range(12):
            rng = np.random.default_rng(seed)
            n = 3 + seed % 3
            problem = RouteProblem(rng.uniform(0, 30, (n, 2)), rng.integers(0, 2, (n, 7)),
                                   rng.uniform(.01, 1, 7), rng.uniform(0, 30, 2), speed=6,
                                   look_s=.8, failure_penalty_s=15, travel_penalty=(seed % 4)*.02,
                                   travel_norm=['max', 'euclidean', 'manhattan'][seed % 3])
            route = exact_order(problem)
            self.assertAlmostEqual(evaluate(problem, route)['objective_s'], brute(problem)[0], places=10)

    def test_exact_can_skip_useless_or_expensive_stop(self):
        problem = self.make(poses=[[0, 0], [10000, 0]], visibility=[[1, 0], [0, 1]], weights=[.99, .01], start=[0, 0])
        self.assertEqual(exact_order(problem), [0])

    def test_exact_full_order_matches_independent_permutation_enumeration(self):
        for seed in range(6):
            rng = np.random.default_rng(seed + 20)
            problem = RouteProblem(rng.uniform(0, 30, (5, 2)), rng.integers(0, 2, (5, 7)),
                                   rng.uniform(.01, 1, 7), rng.uniform(0, 30, 2), speed=6,
                                   failure_penalty_s=seed, travel_norm='euclidean')
            route = exact_order(problem, require_all=True)
            self.assertEqual(set(route), set(range(5)))
            self.assertAlmostEqual(evaluate(problem, route)['objective_s'],
                                   min(direct_objective(problem, p) for p in itertools.permutations(range(5))))
        result = plan(problem, 'exact', require_all=True)
        self.assertTrue(result['optimality']['requires_all_stops'])
        self.assertIn('full permutations', result['optimality']['scope'])

    def test_residual_prior_preserves_unconditional_cost_and_first_look_resolution(self):
        full = self.make(weights=[.5, .3, .2])
        residual = self.make(weights=[.15, .09, .06], normalize_weights=False)
        self.assertEqual(exact_order(full), exact_order(residual))
        for route in [[], [2, 0], [0, 1, 2]]:
            result = evaluate(residual, route)
            self.assertAlmostEqual(result['objective_s'], .3 * evaluate(full, route)['objective_s'])
            self.assertAlmostEqual(result['objective_s'], direct_objective(residual, route))
            self.assertAlmostEqual(result['initial_probability'], .3)
            self.assertAlmostEqual(result['already_resolved_probability'], .7)
            self.assertAlmostEqual(result['found_probability'] + result['missed_probability'], .3)
        result = evaluate(residual, [0, 1, 2])
        self.assertAlmostEqual(result['found_probability'], .3)

    def test_zero_residual_prior_costs_no_further_search(self):
        problem = self.make(weights=[0, 0, 0], normalize_weights=False)
        self.assertEqual(greedy_order(problem), [])
        self.assertEqual(exact_order(problem), [])
        self.assertEqual(evaluate(problem, [2, 1, 0])['objective_s'], 0)
        self.assertEqual(evaluate(problem, [])['already_resolved_probability'], 1)
        self.assertEqual(set(exact_order(problem, require_all=True)), {0, 1, 2})

    def test_zero_failure_penalty_prefers_no_search(self):
        problem = self.make(failure_penalty_s=0)
        self.assertEqual(exact_order(problem), [])
        self.assertEqual(greedy_order(problem), [])

    def test_impossible_targets_do_not_trigger_pointless_travel(self):
        problem = self.make(visibility=np.zeros((3, 3)))
        self.assertEqual(exact_order(problem), [])
        self.assertEqual(greedy_order(problem), [])
        self.assertEqual(evaluate(problem, [])['objective_s'], 40)

    def test_exact_tie_prefers_coverage_without_extra_objective(self):
        problem = self.make(poses=[[0, 0]], visibility=[[1]], weights=[1], start=[0, 0], failure_penalty_s=1)
        self.assertEqual(exact_order(problem), [0])

    def test_motor_norm_and_effort_are_separate(self):
        a = self.make(poses=[[3, 4]], visibility=[[1]], weights=[1], start=[0, 0], speed=1,
                      start_s=0, settle_s=0, look_s=1, travel_norm='max')
        b = self.make(poses=[[3, 4]], visibility=[[1]], weights=[1], start=[0, 0], speed=1,
                      start_s=0, settle_s=0, look_s=1, travel_norm='euclidean')
        self.assertEqual(evaluate(a, [0])['expected_movement_s'], 4)
        self.assertEqual(evaluate(b, [0])['expected_movement_s'], 5)
        self.assertEqual(evaluate(a, [0])['expected_motor_steps'], 7)
        self.assertEqual(evaluate(b, [0])['expected_motor_steps'], 7)

    def test_current_pose_has_no_artificial_move_or_settle_delay(self):
        problem = self.make(poses=[[5, 5]], visibility=[[1]], weights=[1], start=[5, 5], look_s=.7)
        result = evaluate(problem, [0])
        self.assertEqual(result['expected_completion_s'], .7)
        self.assertEqual(result['expected_start_settle_s'], 0)

    def test_travel_penalty_can_change_selected_route(self):
        a = self.make(poses=[[0, 0], [40, 0]], visibility=[[1, 0], [0, 1]], weights=[.3, .7], start=[0, 0],
                      speed=100, failure_penalty_s=40, travel_penalty=0)
        b = self.make(poses=[[0, 0], [40, 0]], visibility=[[1, 0], [0, 1]], weights=[.3, .7], start=[0, 0],
                      speed=100, failure_penalty_s=40, travel_penalty=2)
        self.assertEqual(set(exact_order(a)), {0, 1})
        self.assertEqual(exact_order(b), [0])

    def test_greedy_and_local_search_are_reproducible_and_nonmutating(self):
        poses = np.array([[0., 0], [10, 0], [20, 5]])
        vis = np.array([[.9, .1, 0], [0, .8, .1], [.2, 0, .7]])
        weights = np.array([5., 3, 2])
        originals = [v.copy() for v in (poses, vis, weights)]
        problem = self.make(poses=poses, visibility=vis, weights=weights)
        first = plan(problem, 'greedy_local')
        self.assertEqual(first, plan(problem, 'greedy_local'))
        self.assertLessEqual(first['objective_s'], plan(problem, 'greedy')['objective_s'] + 1e-12)
        for value, original in zip((poses, vis, weights), originals):
            np.testing.assert_array_equal(value, original)
        with self.assertRaises(ValueError):
            problem.visibility[0, 0] = 1

    def test_plans_are_json_safe_and_optimality_is_limited(self):
        problem = self.make()
        for method in ('fixed', 'nearest', 'greedy', 'greedy_local', 'exact'):
            result = plan(problem, method)
            json.dumps(result, allow_nan=False)
            self.assertFalse(result['optimality']['real_world_validated'])
            self.assertEqual(result['optimality']['kind'], 'exact_finite_model' if method == 'exact' else 'heuristic_or_baseline')

    def test_exact_rejects_fractional_and_oversized_problem(self):
        with self.assertRaisesRegex(ValueError, 'binary'):
            exact_order(self.make(visibility=np.full((3, 3), .5)))
        with self.assertRaisesRegex(ValueError, 'bound'):
            exact_order(self.make(), max_exact=2)
        with self.assertRaises(ValueError):
            exact_order(self.make(), max_exact=13)

    def test_nearest_and_fixed_order_are_explicit_baselines(self):
        problem = self.make(start=[19, 5])
        self.assertEqual(nearest_order(problem), [2, 1, 0])
        self.assertEqual(plan(problem, 'fixed')['route'], [0, 1, 2])
        self.assertEqual(plan(problem, 'fixed', order=[2, 0])['route'], [2, 0])

    def test_empty_action_set_has_declared_failure_cost(self):
        problem = self.make(poses=[], visibility=[], weights=[.2, .8])
        self.assertEqual(exact_order(problem), [])
        self.assertEqual(plan(problem, 'exact')['objective_s'], 40)

    def test_invalid_models_are_rejected(self):
        cases = [dict(weights=[0, 0, 0]), dict(weights=[1, -1, 1]), dict(weights=[1, np.nan, 1]),
                 dict(poses=[[1, 1], [1, 1], [3, 3]]), dict(poses=[[1, 2, 3]]),
                 dict(poses=np.zeros((3, 0))), dict(poses=[], visibility=np.zeros((2, 0))),
                 dict(visibility=np.full((3, 3), 1.01)), dict(visibility=np.ones((2, 3))),
                 dict(start=[1]), dict(speed=0), dict(speed=True), dict(look_s=0),
                 dict(start_s=-1), dict(failure_penalty_s=float('inf')), dict(travel_penalty=-1),
                 dict(travel_norm='unknown'), dict(normalize_weights=1),
                 dict(normalize_weights=False, weights=[.7, .3, .2])]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                self.make(**case)

    def test_invalid_routes_are_rejected(self):
        problem = self.make()
        for route in [[0, 0], [-1], [3], [True], [1.1], None]:
            with self.subTest(route=route), self.assertRaises(ValueError):
                evaluate(problem, route)

    def test_local_search_does_not_worsen_arbitrary_starting_route(self):
        problem = self.make(visibility=[[.8, .2, 0], [.2, .7, 0], [.6, .5, .8]])
        before = evaluate(problem, [2, 1, 0])
        after = evaluate(problem, improve_order(problem, [2, 1, 0]))
        self.assertLessEqual(after['objective_s'], before['objective_s'] + 1e-12)


if __name__ == '__main__':
    unittest.main()
