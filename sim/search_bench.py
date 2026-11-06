"""Offline Find-route experiment using the production search controller.

Run ``python3 -m sim.search_bench --out eval/route_search/run``. The output
contains the frozen development selection, independent simulation seeds, every
trial (including misses/absence), paired intervals and input hashes. It never
writes a policy, calls a camera, or deploys anything.

The room-map floor mask constrains actions. Target locations are an explicitly
biased angular prior mixed with broad floor-bearing placements. This is a
controlled simulator experiment, not a validation of the 3D room or detector.
"""
import argparse
from dataclasses import asdict
import hashlib
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

from .gate1 import load_tracker
from .room import load_params
from .search_prior import load_prior
from .search_routes import RouteProblem, plan
from .search_scene import SearchDeadline, SearchScenario, SearchScene


ROOT = Path(__file__).resolve().parent.parent
POLICIES = ('incumbent', 'nearest', 'exact_existing', 'room_greedy',
            'room_exact', 'room_exact_rows85', 'room_exact_rows100')
FAMILIES = ('prior', 'shifted_prior', 'moving', 'hard', 'occluded', 'absent')
STARTS = ((1019, 534), (1850, 500), (300, 400), (3200, 600),
          (2400, 800), (1500, 80), (0, 500), (3700, 500))
HOURS = (0, 6, 12, 18)
BUDGET = 120.0
MOTOR_COST = .002  # explicit preference: 2 seconds per 1000 actual motor steps


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_roommap(path):
    spec = importlib.util.spec_from_file_location('search_bench_roommap', ROOT / 'server/roommap.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RoomMap.load(path)


class Experiment:
    def __init__(self, params, prior, roommap, policy_doc, budget=BUDGET):
        self.params, self.prior, self.roommap = params, prior, roommap
        self.policy_doc, self.budget = policy_doc, float(budget)
        self.T = load_tracker()
        self.T.ROOM_MAP = roommap
        self.T.SEARCH_STOPS = self.T.policy_stops(policy_doc)
        for key, value in self.T.policy_constants(policy_doc).items():
            setattr(self.T, key, value)
        self.cache = {}

    def route(self, name, start, hour):
        key = name, tuple(start), hour
        if key in self.cache:
            return self.cache[key]
        self.T.hour_of = lambda _: hour
        scene = SearchScene(SearchScenario(seed=0, start_pose=start), params=self.params)
        tr = self.T.Tracker(scene, scene.find_pet, lambda: scene.t)
        existing = [(p['x'], p['y']) for p in tr.search_stops()]
        if name == 'incumbent':
            result = {'poses': existing, 'method': 'production_search_stops',
                      'optimality': {'kind': 'baseline', 'real_world_validated': False}}
        else:
            poses = np.asarray(existing if name in ('nearest', 'exact_existing') else self.prior.candidates)
            # The controller looks here first. Planning conditions on its ideal
            # geometric miss only; the executed experiment still models misses.
            weights = self.prior.weights.copy()
            weights[self.prior.visibility_for(np.asarray([start]))[0]] = 0
            problem = self.problem(poses, weights, start)
            method = 'nearest' if name == 'nearest' else 'exact' if name == 'exact_existing' else 'greedy'
            result = plan(problem, method, require_all=method == 'exact')
            if name.startswith('room_'):
                # Bound planning complexity before looking at rollout results.
                chosen = np.asarray(result['poses'][:8], dtype=float).reshape(-1, 2)
                result = plan(self.problem(chosen, weights, start),
                              'greedy' if name == 'room_greedy' else 'exact',
                              require_all=name != 'room_greedy')
            result['search_controller'] = 'production Tracker.search; route injection only'
        result['row_step'] = .85 if name.endswith('rows85') else 1.0 if name.endswith('rows100') else .55
        self.cache[key] = result
        return result

    def problem(self, poses, weights, start):
        return RouteProblem(poses, self.prior.visibility_for(poses), weights, start,
                            speed=float(self.params['motor']['speed_steps_s']['250']),
                            start_s=.1, settle_s=.9, look_s=1.6,
                            failure_penalty_s=self.budget, travel_penalty=MOTOR_COST,
                            travel_norm='euclidean', normalize_weights=False)

    def run(self, name, case, keep_trace=False):
        scenario = SearchScenario(**case['scenario'])
        room = SearchScene(scenario, params=self.params, budget_s=self.budget)
        if case.get('speed_scale'):
            room.speed *= case['speed_scale']
        if case.get('pan_frame'):
            room.FW = case['pan_frame']
        if case.get('backlash'):
            room.backlash = case['backlash']
        T = self.T
        T.hour_of = lambda _: case['hour']
        route = self.route(name, scenario.start_pose, case['hour'])
        T.ROW_STEP = route['row_step']
        confirmation = []

        class TrialTracker(T.Tracker):
            def search_stops(self):
                if name == 'incumbent':
                    return super().search_stops()
                return [{'x': x, 'y': y, 'map': True} for x, y in route['poses']]

            def _found(self, box, *args, **kwargs):
                message = super()._found(box, *args, **kwargs)
                confirmation.append(getattr(box, 'evidence', None))
                return message

        tr = TrialTracker(room, room.find_pet, clock=lambda: room.t, log=lambda _: None,
                          sleep=room.sleep, find_now=room.find_now,
                          cancelled=lambda _: case.get('cancel_at_s') is not None and room.t >= case['cancel_at_s'])
        try:
            message = tr.search('offline route experiment')
            deadline = False
        except SearchDeadline:
            room.stop()
            message, deadline = 'fixed experiment deadline', True
        finally:
            T.ROW_STEP = .55
        found = bool(confirmation)
        right = bool(found and confirmation[-1] and confirmation[-1].true_detection)
        elapsed = min(self.budget, room.t)
        loss = elapsed if right else self.budget
        result = {'id': case['id'], 'family': case['family'], 'policy': name,
                  'found_right': right, 'wrong_find': found and not right,
                  'elapsed_s': elapsed, 'loss_s': loss,
                  'motor_steps': room.actual_travel_steps,
                  'optical_steps': room.actual_optical_travel_steps,
                  'moves': room.actual_move_count, 'deadline': deadline,
                  'cancelled': bool(case.get('cancel_at_s') is not None and 'cancelled' in (message or '')),
                  'score_s': loss + MOTOR_COST * room.actual_travel_steps,
                  'confirmation': asdict(confirmation[-1]) if found and confirmation[-1] else None}
        if keep_trace:
            result.update({'trace': room.pose_trace, 'scenario': case['scenario'],
                           'planned_route': route, 'message': message})
        return result


def bank(prior, n, seed, stress=False):
    """Declared balanced stress strata, not an estimate of household prevalence."""
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        family = FAMILIES[i % len(FAMILIES)]
        index = int(rng.choice(len(prior.targets), p=prior.weights))
        target = prior.targets[index].copy()
        if family == 'shifted_prior':
            target = prior.targets[int(rng.integers(len(prior.targets)))].copy()
        # Distinct continuous placements avoid testing only exact planning cells.
        target += rng.uniform(-20, 20, 2)
        target = np.clip(target, [0, 40], [prior.limits[0], prior.limits[1] - 40])
        end = prior.targets[int(rng.integers(len(prior.targets)))].copy()
        scenario = {'seed': int(seed + i * 7919), 'target_start': target.tolist(),
                    'target_end': end.tolist() if family == 'moving' else target.tolist(),
                    'move_start_s': float(rng.uniform(2, 20)),
                    'move_duration_s': float(rng.uniform(12, 35)),
                    'start_pose': list(STARTS[int(rng.integers(len(STARTS)))]),
                    'absent': family == 'absent', 'occluded': family == 'occluded',
                    'detection_p': .20 if family == 'hard' else .80,
                    'false_positive_p': .02 if stress else .002,
                    'lag_s': float(rng.uniform(1.2, 2.0)) if stress else .9}
        case = {'id': f'{seed}-{i:04d}', 'family': family, 'scenario': scenario,
                'hour': HOURS[int(rng.integers(len(HOURS)))]}
        if stress:
            case.update({'speed_scale': float(rng.uniform(.8, 1.1)),
                         'pan_frame': float(rng.uniform(707.2, 740)),
                         'backlash': float(rng.uniform(27, 40.9))})
        out.append(case)
    return out


def summarize(rows):
    n = len(rows)
    if not n:
        return {'n': 0}
    present = [r for r in rows if r['family'] not in ('absent', 'occluded')]
    return {'n': n, 'found_right': sum(r['found_right'] for r in rows) / n,
            'within_30s': sum(r['found_right'] and r['elapsed_s'] <= 30 for r in rows) / n,
            'within_60s': sum(r['found_right'] and r['elapsed_s'] <= 60 for r in rows) / n,
            'findable_within_60s': (sum(r['found_right'] and r['elapsed_s'] <= 60 for r in present) / len(present)
                                    if present else None),
            'wrong_finds': sum(r['wrong_find'] for r in rows),
            'mean_capped_loss_s': float(np.mean([r['loss_s'] for r in rows])),
            'mean_elapsed_s': float(np.mean([r['elapsed_s'] for r in rows])),
            'mean_motor_steps': float(np.mean([r['motor_steps'] for r in rows])),
            'mean_moves': float(np.mean([r['moves'] for r in rows])),
            'mean_score_s': float(np.mean([r['score_s'] for r in rows])),
            'deadline_count': sum(r['deadline'] for r in rows)}


def paired(base, candidate, seed=991, repeats=2000):
    if [r['id'] for r in base] != [r['id'] for r in candidate]:
        raise ValueError('paired trial IDs differ')
    rng = np.random.default_rng(seed)
    out = {}
    for metric in ('loss_s', 'motor_steps', 'score_s'):
        delta = np.array([b[metric] - a[metric] for a, b in zip(base, candidate)])
        boot = delta[rng.integers(len(delta), size=(repeats, len(delta)))].mean(1)
        out[metric] = {'candidate_minus_base': float(delta.mean()),
                       'paired_bootstrap_95': np.percentile(boot, [2.5, 97.5]).tolist()}
    return out


def select(summaries):
    baseline = summaries['incumbent']
    eligible = [name for name, s in summaries.items()
                if s['wrong_finds'] <= baseline['wrong_finds']
                and s['found_right'] >= baseline['found_right'] - .02
                and s['within_60s'] >= baseline['within_60s'] - .02]
    return min(eligible, key=lambda name: (summaries[name]['mean_score_s'], POLICIES.index(name)))


def acceptance(comparisons):
    checks = {}
    for label, comparison in comparisons.items():
        base, candidate = comparison['base'], comparison['candidate']
        checks[label] = {'no_extra_wrong_finds': candidate['wrong_finds'] <= base['wrong_finds'],
                         'correct_find_nonregression': candidate['found_right'] >= base['found_right'] - .02,
                         'within_60s_nonregression': candidate['within_60s'] >= base['within_60s'] - .02,
                         'lower_mean_score': candidate['mean_score_s'] < base['mean_score_s']}
    return {'checks': checks, 'passes_simulation_gate': all(all(v.values()) for v in checks.values()),
            'real_world_validated': False,
            'meaning': 'Descriptive finite-bank guardrails; passing would only make an offline candidate, never authorize installation.'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--dev', type=int, default=96)
    parser.add_argument('--test', type=int, default=360)
    parser.add_argument('--stress', type=int, default=180)
    parser.add_argument('--development-seed', type=int, default=10101)
    parser.add_argument('--test-seed', type=int, default=80101)
    parser.add_argument('--stress-seed', type=int, default=180101)
    args = parser.parse_args(argv)
    if min(args.dev, args.test, args.stress) < 6:
        parser.error('each bank needs at least six trials')
    seeds = (args.development_seed, args.test_seed, args.stress_seed)
    if min(seeds) < 0 or len(set(seeds)) != 3:
        parser.error('development, test and stress seeds must be distinct and nonnegative')
    args.out.mkdir(parents=True, exist_ok=False)
    prior = load_prior(ROOT / 'sim/params.json', ROOT / 'models/room_map.json')
    experiment = Experiment(load_params(), prior, load_roommap(ROOT / 'models/room_map.json'),
                            json.loads((ROOT / 'models/follow_policy.json').read_text()))
    inputs = ('server/tracker.py', 'server/roommap.py', 'sim/room.py', 'sim/search_routes.py',
              'sim/search_prior.py', 'sim/search_scene.py', 'sim/search_bench.py',
              'sim/params.json', 'models/room_map.json', 'models/follow_policy.json')
    manifest = {'input_sha256': {p: _sha(ROOT / p) for p in inputs},
                'budget_s': BUDGET, 'motor_score_s_per_step': MOTOR_COST,
                'policies': list(POLICIES), 'families': list(FAMILIES),
                'prior': prior.provenance,
                'split': {'development_seed': args.development_seed, 'test_seed': args.test_seed,
                          'stress_seed': args.stress_seed},
                'scope': 'Independent simulation seeds, not independent household days or labels.',
                'selection': 'On development only: no more wrong finds, at most 2pp loss in correct/60s finds; minimize capped loss + motor cost.',
                'limits': ['Angular target model; no validated room occlusion geometry.',
                           'Prior and floor mask include all recorded days; seeds alone do not validate their fidelity.',
                           'Balanced strata are stress coverage, not estimated household prevalence.',
                           'Motion is a prescribed angular path, not a validated walk around furniture.',
                           'Detector probabilities and persistent occlusion are explicit assumptions.',
                           'Exact optimality covers only supplied finite stops with binary static visibility.',
                           'The planning surrogate removes ideal initial-view mass; execution can miss that target and skips near-start stops.',
                           'Planning ignores detections during travel; executed controller checks fresh moving frames.',
                           'Planning CPU time is excluded; motor travel stops when search returns, before any pending centering move completes.',
                           'Motor steps proxy movement; noise and pet disturbance are not measured.']}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    rows, summaries = {}, {}
    development = bank(prior, args.dev, args.development_seed)
    for name in POLICIES:
        start = time.monotonic()
        result = [experiment.run(name, case) for case in development]
        rows[name] = result
        summaries[name] = summarize(result)
        print(json.dumps({'stage': 'development', 'policy': name, **summaries[name],
                          'wall_s': round(time.monotonic() - start, 1)}), flush=True)
    winner = select(summaries)
    frozen = {'selected': winner, 'development': summaries,
              'development_bank_sha256': hashlib.sha256(json.dumps(development, sort_keys=True).encode()).hexdigest(),
              'frozen_before_test': True}
    (args.out / 'selection.json').write_text(json.dumps(frozen, indent=2) + '\n')
    (args.out / 'development.json').write_text(json.dumps({'cases': development, 'results': rows}, indent=1) + '\n')
    comparisons = {}
    for label, size, seed, stress in [('test', args.test, args.test_seed, False),
                                      ('stress', args.stress, args.stress_seed, True)]:
        cases = bank(prior, size, seed, stress)
        results = {}
        for name in dict.fromkeys(('incumbent', winner)):
            results[name] = [experiment.run(name, case, keep_trace=i < 12) for i, case in enumerate(cases)]
            print(json.dumps({'stage': label, 'policy': name, **summarize(results[name])}), flush=True)
        base, candidate = results['incumbent'], results[winner]
        comparison = {'base': summarize(base), 'candidate': summarize(candidate),
                      'paired': paired(base, candidate),
                      'by_family': {f: {name: summarize([r for r in rr if r['family'] == f])
                                        for name, rr in results.items()} for f in FAMILIES}}
        comparisons[label] = comparison
        (args.out / f'{label}.json').write_text(json.dumps({'cases': cases, 'results': results,
                                                          'comparison': comparison}, indent=1) + '\n')
    cancellation = bank(prior, 12, 280101)
    for case in cancellation:
        case['cancel_at_s'] = 7.0
        case['scenario']['absent'] = True
        case['scenario']['false_positive_p'] = 0
    cancel_results = {name: [experiment.run(name, case) for case in cancellation]
                      for name in dict.fromkeys(('incumbent', winner))}
    (args.out / 'cancellation.json').write_text(json.dumps(cancel_results, indent=1) + '\n')
    unchanged = all(_sha(ROOT / p) == h for p, h in manifest['input_sha256'].items())
    report = {'selected': winner, 'development': summaries, 'comparisons': comparisons,
              'acceptance': acceptance(comparisons),
              'cancellation': {name: {'n': len(rr), 'stopped': sum(r['cancelled'] for r in rr),
                                      'max_elapsed_s': max(r['elapsed_s'] for r in rr)}
                               for name, rr in cancel_results.items()},
              'inputs_unchanged': unchanged,
              'promotion': 'offline only; real-world validation and calibrated geometry remain open'}
    (args.out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    (args.out / 'routes.json').write_text(json.dumps({str(key): value for key, value in experiment.cache.items()}, indent=1) + '\n')
    print(json.dumps(report), flush=True)
    return 0 if unchanged else 1


if __name__ == '__main__':
    raise SystemExit(main())
