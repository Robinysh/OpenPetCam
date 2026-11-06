"""Offline camera-stop routing under an explicit, finite observation model.

This module has no camera, file, training or deployment access. A state is a
possible target location (an all-zero visibility column can represent absence).
Every state's prior mass remains in the cost, including states never found.

For a route, each step costs travel time, a start/settle delay when it moves,
and an observation. Time and motor steps are weighted by the probability that
the search has not already found its target. Remaining mass pays an additional
``failure_penalty_s`` after the route. This is a declared planning objective,
not a fixed wall-clock search deadline or a real-world optimality claim.

The exact solver considers every subset and order (or every full permutation)
of at most twelve supplied stops, for *binary static visibility*. Stochastic routing assumes independent
per-view detections; correlated misses and moving targets need an external
simulation and cannot inherit the exact solver's guarantee.
"""
from dataclasses import dataclass
import math

import numpy as np


EPS = 1e-12
EXACT_LIMIT = 12


def _array(value, name):
    source = np.asarray(value)
    if source.dtype.kind not in 'biuf':
        raise ValueError(name + ' must be numeric')
    result = np.array(source, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(result)):
        raise ValueError(name + ' must be finite')
    return result


def _scalar(value, name, positive=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(name + ' must be a finite number')
    try:
        value = float(value)
    except (ValueError, TypeError, OverflowError):
        raise ValueError(name + ' must be a finite number') from None
    if not math.isfinite(value) or value < 0 or positive and value <= 0:
        raise ValueError(name + (' must be positive' if positive else ' must be nonnegative'))
    return value


@dataclass(frozen=True)
class RouteProblem:
    """Finite actions and target-state prior.

    By default, weights are normalized. Set normalize_weights=False for a
    residual prior after an earlier observation: total mass must be in [0, 1],
    and all costs/probabilities retain that unconditional probability scale.
    Omitted mass is already resolved and incurs no future search cost.
    """
    poses: object
    visibility: object
    weights: object
    start: object
    speed: float = 250.0
    start_s: float = 0.1
    settle_s: float = 1.6
    look_s: float = 0.8
    failure_penalty_s: float = 120.0
    travel_penalty: float = 0.0
    travel_norm: str = 'max'
    normalize_weights: bool = True

    def __post_init__(self):
        poses = _array(self.poses, 'poses')
        if poses.shape == (0,):
            poses = poses.reshape(0, 2)
        if poses.ndim != 2 or poses.shape[1] != 2:
            raise ValueError('poses must be an N by 2 matrix')
        if len(np.unique(poses, axis=0)) != len(poses):
            raise ValueError('poses must be unique')
        weights = _array(self.weights, 'weights')
        if weights.ndim != 1 or not len(weights) or np.any(weights < 0):
            raise ValueError('weights must be a nonempty nonnegative vector')
        total = float(weights.sum())
        if type(self.normalize_weights) is not bool:
            raise ValueError('normalize_weights must be a boolean')
        if not math.isfinite(total):
            raise ValueError('weights must have finite total mass')
        if self.normalize_weights:
            if total <= 0:
                raise ValueError('normalized weights must have positive total mass')
            weights /= total
        elif total > 1 + EPS:
            raise ValueError('residual weights must have total mass at most one')
        elif total > 1:
            weights /= total  # Accept floating-point summation error only.
        visibility = _array(self.visibility, 'visibility')
        if not len(poses) and visibility.shape == (0,):
            visibility = visibility.reshape(0, len(weights))
        if visibility.shape != (len(poses), len(weights)):
            raise ValueError('visibility must have one row per pose and one column per state')
        if np.any(visibility < 0) or np.any(visibility > 1):
            raise ValueError('visibility probabilities must be in [0, 1]')
        start = _array(self.start, 'start')
        if start.shape != (2,):
            raise ValueError('start must have two coordinates')
        if self.travel_norm not in ('max', 'euclidean', 'manhattan'):
            raise ValueError('travel_norm must be max, euclidean or manhattan')
        for name in ('speed', 'start_s', 'settle_s', 'look_s', 'failure_penalty_s', 'travel_penalty'):
            value = _scalar(getattr(self, name), name, positive=name in ('speed', 'look_s'))
            object.__setattr__(self, name, value)
        for value in (poses, visibility, weights, start):
            value.flags.writeable = False
        object.__setattr__(self, 'poses', poses)
        object.__setattr__(self, 'visibility', visibility)
        object.__setattr__(self, 'weights', weights)
        object.__setattr__(self, 'start', start)

    @property
    def n(self):
        return len(self.poses)

    @property
    def deterministic(self):
        return bool(np.all((self.visibility == 0) | (self.visibility == 1)))


def _edge(problem, a, b):
    dx, dy = abs(float(b[0]) - float(a[0])), abs(float(b[1]) - float(a[1]))
    motor_steps = dx + dy
    distance = (max(dx, dy) if problem.travel_norm == 'max' else
                math.hypot(dx, dy) if problem.travel_norm == 'euclidean' else motor_steps)
    move_s = distance / problem.speed
    delay_s = problem.start_s + problem.settle_s if motor_steps else 0.0
    step_s = move_s + delay_s + problem.look_s
    if not all(math.isfinite(x) for x in (motor_steps, move_s, delay_s, step_s)):
        raise ValueError('motor coordinates and speed produce a non-finite cost')
    return step_s, motor_steps, move_s, delay_s


def _route(problem, route):
    try:
        route = list(route)
    except TypeError:
        raise ValueError('route must be a sequence of distinct stop indices') from None
    if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
           or not 0 <= i < problem.n for i in route):
        raise ValueError('route indices must refer to provided stops')
    route = [int(i) for i in route]
    if len(set(route)) != len(route):
        raise ValueError('a stop may appear only once in a route')
    return route


def evaluate(problem, route):
    """Score every prior state, rather than averaging only successful searches."""
    route = _route(problem, route)
    residual = problem.weights.copy()
    here = problem.start
    elapsed = expected_elapsed = motor = expected_motor = 0.0
    movement = delays = observation = 0.0
    arrivals = []
    for index in route:
        step_s, steps, move_s, delay_s = _edge(problem, here, problem.poses[index])
        alive = float(residual.sum())
        expected_elapsed += alive * step_s
        expected_motor += alive * steps
        movement += alive * move_s
        delays += alive * delay_s
        observation += alive * problem.look_s
        elapsed += step_s
        motor += steps
        arrivals.append(elapsed)
        residual *= 1 - problem.visibility[index]
        here = problem.poses[index]
    missed = float(residual.sum())
    initial = float(problem.weights.sum())
    penalty = missed * problem.failure_penalty_s
    completion = expected_elapsed + penalty
    objective = completion + problem.travel_penalty * expected_motor
    if not all(math.isfinite(v) for v in (completion, objective, expected_motor, elapsed)):
        raise ValueError('route produces a non-finite objective')
    return {'route': route, 'poses': problem.poses[route].tolist(),
            'objective_s': float(objective), 'expected_completion_s': float(completion),
            'expected_elapsed_s': float(expected_elapsed), 'terminal_failure_cost_s': float(penalty),
            'failure_penalty_s': problem.failure_penalty_s,
            'initial_probability': initial, 'already_resolved_probability': max(0.0, 1 - initial),
            'found_probability': max(0.0, initial - missed), 'missed_probability': missed,
            'normalized_prior': problem.normalize_weights,
            'expected_motor_steps': float(expected_motor), 'full_route_motor_steps': float(motor),
            'full_route_elapsed_s': float(elapsed), 'arrival_times_s': arrivals,
            'expected_movement_s': float(movement), 'expected_start_settle_s': float(delays),
            'expected_observation_s': float(observation),
            'travel_penalty_s_per_motor_step': problem.travel_penalty,
            'travel_norm': problem.travel_norm,
            'visibility_model': 'deterministic_static' if problem.deterministic else 'independent_static_detections'}


def fixed_order(problem, order=None):
    return list(range(problem.n)) if order is None else _route(problem, order)


def nearest_order(problem):
    left = set(range(problem.n))
    route, here = [], problem.start
    while left:
        index = min(left, key=lambda i: (_edge(problem, here, problem.poses[i])[:2], i))
        route.append(index)
        left.remove(index)
        here = problem.poses[index]
    return route


def greedy_order(problem):
    """Add useful probability per expected second; stop when its penalty saving ends."""
    left = set(range(problem.n))
    residual = problem.weights.copy()
    route, here = [], problem.start
    while left:
        choices = []
        alive = float(residual.sum())
        if alive <= EPS:
            break
        for index in sorted(left):
            gain = float(residual @ problem.visibility[index])
            step_s, steps, *_ = _edge(problem, here, problem.poses[index])
            cost = alive * (step_s + problem.travel_penalty * steps)
            benefit = gain * problem.failure_penalty_s - cost
            if gain > EPS and benefit > EPS:
                choices.append((gain / cost, benefit, gain, -steps, -index, index))
        if not choices:
            break
        index = max(choices)[-1]
        route.append(index)
        left.remove(index)
        residual *= 1 - problem.visibility[index]
        here = problem.poses[index]
    return route


def _better(candidate, incumbent):
    if candidate['objective_s'] < incumbent['objective_s'] - EPS:
        return True
    if abs(candidate['objective_s'] - incumbent['objective_s']) > EPS:
        return False
    return (candidate['missed_probability'], candidate['expected_motor_steps'], len(candidate['route']), candidate['route']) < \
           (incumbent['missed_probability'], incumbent['expected_motor_steps'], len(incumbent['route']), incumbent['route'])


def improve_order(problem, route=None, passes=4):
    """Bounded deterministic insert/delete/relocate/reverse search; no global guarantee."""
    if type(passes) is not int or not 1 <= passes <= 20:
        raise ValueError('passes must be an integer in 1..20')
    if problem.n > 24:
        raise ValueError('local route search is bounded to 24 stops')
    current = evaluate(problem, greedy_order(problem) if route is None else route)
    for _ in range(passes):
        order = current['route']
        proposals = {tuple(order[:i] + order[i+1:]) for i in range(len(order))}
        for index in set(range(problem.n)) - set(order):
            proposals.update(tuple(order[:i] + [index] + order[i:]) for i in range(len(order)+1))
        for i in range(len(order)):
            shortened = order[:i] + order[i+1:]
            proposals.update(tuple(shortened[:j] + [order[i]] + shortened[j:]) for j in range(len(order)))
            proposals.update(tuple(order[:i] + list(reversed(order[i:j+1])) + order[j+1:]) for j in range(i+1, len(order)))
        best = current
        for proposed in sorted(proposals):
            candidate = evaluate(problem, proposed)
            if _better(candidate, best):
                best = candidate
        if best['route'] == current['route']:
            break
        current = best
    return current['route']


def exact_order(problem, max_exact=EXACT_LIMIT, require_all=False):
    """Exact static DP; require_all restricts selection to full permutations."""
    if type(max_exact) is not int or not 0 <= max_exact <= EXACT_LIMIT:
        raise ValueError('max_exact must be an integer in 0..12')
    if problem.n > max_exact:
        raise ValueError('exact routing exceeds the declared finite action bound')
    if not problem.deterministic:
        raise ValueError('exact routing requires binary deterministic visibility')
    if type(require_all) is not bool:
        raise ValueError('require_all must be a boolean')
    n = problem.n
    if not n:
        return []
    count = 1 << n
    state_masks = np.zeros(len(problem.weights), dtype=np.uint16)
    for i in range(n):
        state_masks |= problem.visibility[i].astype(np.uint16) << i
    survival = np.array([problem.weights[(state_masks & mask) == 0].sum() for mask in range(count)])
    edges = [[_edge(problem, problem.start if i == n else problem.poses[i], problem.poses[j])
              for j in range(n)] for i in range(n+1)]
    costs = np.full((count, n), np.inf)
    travel = np.full((count, n), np.inf)
    paths = {}
    for j in range(n):
        step_s, motor, *_ = edges[n][j]
        costs[1 << j, j] = survival[0] * (step_s + problem.travel_penalty * motor)
        travel[1 << j, j] = survival[0] * motor
        paths[1 << j, j] = (j,)
    for mask in range(1, count):
        alive = float(survival[mask])
        for last in range(n):
            if not math.isfinite(costs[mask, last]):
                continue
            path = paths[mask, last]
            for nxt in range(n):
                if mask & (1 << nxt):
                    continue
                next_mask = mask | (1 << nxt)
                step_s, motor, *_ = edges[last][nxt]
                value = costs[mask, last] + alive * (step_s + problem.travel_penalty * motor)
                motor_value = travel[mask, last] + alive * motor
                next_path = path + (nxt,)
                old = costs[next_mask, nxt]
                if value < old - EPS or abs(value-old) <= EPS and \
                        (motor_value, next_path) < (travel[next_mask, nxt], paths.get((next_mask, nxt), ())):
                    costs[next_mask, nxt] = value
                    travel[next_mask, nxt] = motor_value
                    paths[next_mask, nxt] = next_path
    best = None if require_all else evaluate(problem, [])
    for key, route in paths.items():
        mask, last = key
        if require_all and mask != count - 1:
            continue
        value = float(costs[mask, last] + survival[mask] * problem.failure_penalty_s)
        if best is None or value <= best['objective_s'] + EPS:
            candidate = evaluate(problem, route)
            if best is None or _better(candidate, best):
                best = candidate
    return best['route']


def plan(problem, method='greedy', order=None, max_exact=EXACT_LIMIT, local_passes=4, require_all=False):
    """Return JSON-safe planning results and the precise scope of any guarantee."""
    if not isinstance(problem, RouteProblem):
        raise ValueError('plan expects a RouteProblem')
    if order is not None and method != 'fixed':
        raise ValueError('an explicit order is only valid for the fixed method')
    if type(require_all) is not bool or require_all and method != 'exact':
        raise ValueError('require_all is a boolean option for the exact method only')
    if method == 'fixed':
        route = fixed_order(problem, order)
    elif method == 'nearest':
        route = nearest_order(problem)
    elif method == 'greedy':
        route = greedy_order(problem)
    elif method == 'greedy_local':
        route = improve_order(problem, passes=local_passes)
    elif method == 'exact':
        route = exact_order(problem, max_exact, require_all)
    else:
        raise ValueError('unknown route method')
    result = evaluate(problem, route)
    result.update({'method': method, 'provided_stops': problem.n, 'selected_stops': len(route),
                   'prior_states': len(problem.weights), 'optimality': {
                       'kind': 'exact_finite_model' if method == 'exact' else 'heuristic_or_baseline',
                       'scope': ('All full permutations' if require_all else 'All subsets and orders') + ' of the provided stops under binary static visibility and the declared failure-penalized cost.'
                       if method == 'exact' else 'No global optimality guarantee.',
                       'requires_all_stops': require_all if method == 'exact' else None,
                       'evaluated_action_set_bound': max_exact if method == 'exact' else None,
                       'real_world_validated': False},
                   'assumptions': ['Static target state and fixed observation probabilities.',
                                   'Observation outcomes are independent between different views when visibility is fractional.',
                                   'Each stop is visited at most once; no continuous pan observations are modeled.',
                                   'Motor steps and movement time are planning proxies; sound and stress are not measured.']})
    return result
