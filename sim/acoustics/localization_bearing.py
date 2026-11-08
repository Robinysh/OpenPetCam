"""Exploratory, spatially grouped DEVELOPMENT-only selective bearing study.

Near-score angular arcs are descriptive sensitivity sets, not confidence regions.
Cross-validation uses already-inspected development tiles and supplies no fresh
confirmatory result or real-room confidence calibration.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import time

import numpy as np

from .localization import rank_candidates, ripple_features, spatial_partition


SIMILARITIES = (.3, .5, .65, .8, .9)
MARGINS = (0., .001, .005, .02)
ARC_WIDTHS = (10., 20., 30., 45., 60.)
REPEATS = (1, 4, 16)
NEAR_SCORE = .02
FOLDS = 5


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def angle_error(predicted, truth):
    return abs((predicted - truth + 180) % 360 - 180)


def circular_arc(angles):
    """Shortest closed circular arc covering every supplied bearing, in degrees."""
    angles = np.asarray(angles, float)
    if angles.ndim != 1 or not len(angles) or not np.isfinite(angles).all():
        raise ValueError('a nonempty finite bearing vector is required')
    ordered = np.sort(np.mod(angles, 360))
    gaps = np.diff(np.r_[ordered, ordered[0] + 360])
    gap = int(np.argmax(gaps))
    width = float(360 - gaps[gap])
    start = float(ordered[(gap + 1) % len(ordered)])
    center = (start + width / 2 + 180) % 360 - 180
    return center, width


def tile_fold(point, folds=FOLDS):
    if len(point) != 3 or not np.isfinite(point).all() or folds < 2:
        raise ValueError('finite source anchor and at least two folds required')
    tile = math.floor(point[0]), math.floor(point[2])
    digest = hashlib.sha256(f'bearing-development-v1:{tile[0]}:{tile[1]}'.encode()).digest()
    return int.from_bytes(digest[:4], 'big') % folds


def accepts(row, rule):
    # Exact ties must abstain even if a development threshold is zero.
    return (row.get('estimated_bearing_deg') is not None and row.get('score_margin') is not None
            and row['score_margin'] > 1e-9 and row['score_margin'] >= rule['min_margin']
            and row['similarity'] >= rule['min_similarity'] and row['arc_width_deg'] <= rule['max_arc_deg'])


def correct(row):
    # No-emitter controls have no invented truth/error and are failures if acted on.
    return row.get('bearing_error_deg') is not None and row['bearing_error_deg'] <= 30


def select_rule(training):
    minimum = max(8, len(training) // 20)
    best = None
    for similarity in SIMILARITIES:
        for margin in MARGINS:
            for width in ARC_WIDTHS:
                rule = {'min_similarity': similarity, 'min_margin': margin, 'max_arc_deg': width}
                chosen = [row for row in training if accepts(row, rule)]
                precision = sum(correct(row) for row in chosen) / len(chosen) if chosen else 0
                tiles = {row.get('spatial_tile') for row in chosen} - {None}
                if len(chosen) >= minimum and precision >= .8 and len(tiles) >= 3:
                    objective = len(chosen), precision, -width, similarity, margin
                    if best is None or objective > best[0]:
                        best = objective, rule
    if best is None:
        return {'min_similarity': 1.01, 'min_margin': 2.01, 'max_arc_deg': 0.,
                'training_cases': len(training), 'training_accepted': 0, 'training_precision': None,
                'reason': 'No training rule met80%within30deg, minimumcoverage and3acceptedsource-anchor tiles; abstain.'}
    rule = dict(best[1]); chosen = [row for row in training if accepts(row, rule)]
    return dict(rule, training_cases=len(training), training_accepted=len(chosen),
                training_accepted_tiles=len({row['spatial_tile'] for row in chosen}),
                training_precision=sum(correct(row) for row in chosen) / len(chosen))


def cross_validate(rows):
    """Fit only other tile folds; attach one out-of-fold decision per row."""
    output, rules = [], {}
    for fold in range(FOLDS):
        training = [row for row in rows if row['fold'] != fold]
        validation = [row for row in rows if row['fold'] == fold]
        rule = select_rule(training); rules[str(fold)] = rule
        for row in validation:
            output.append(dict(row, acted_on=bool(accepts(row, rule)), rule_fold=fold))
    return output, rules


def summarize(rows):
    source = [row for row in rows if row['source_m'] is not None]
    background = [row for row in rows if row['source_m'] is None]
    acted = [row for row in rows if row['acted_on']]
    source_acted = [row for row in source if row['acted_on']]
    return {'cases': len(rows), 'source_cases': len(source), 'background_cases': len(background),
            'source_candidates': sum(row['estimated_bearing_deg'] is not None for row in source),
            'unconditional_arc_center_within30deg_fraction': sum(correct(row) for row in source) / len(source) if source else None,
            'unconditional_top1_within30deg_fraction': sum(row['top1_bearing_error_deg'] is not None and row['top1_bearing_error_deg'] <= 30 for row in source) / len(source) if source else None,
            'acted_cases': len(acted), 'action_fraction_all_cases': len(acted) / len(rows) if rows else 0,
            'source_acted_cases': len(source_acted), 'source_action_fraction': len(source_acted) / len(source) if source else None,
            'action_precision_within30deg_including_background_failures': sum(correct(row) for row in acted) / len(acted) if acted else None,
            'source_action_precision_within30deg': sum(correct(row) for row in source_acted) / len(source_acted) if source_acted else None,
            'background_false_actions': sum(row['acted_on'] for row in background),
            'source_success_coverage': sum(row['acted_on'] and correct(row) for row in source) / len(source) if source else None}


def run(development, benchmark, output, *, fine=None, deadline_s=290):
    development, benchmark, output = map(lambda p: Path(p).resolve(), (development, benchmark, output))
    if not 10 <= deadline_s <= 590:
        raise ValueError('deadline10..590required')
    dev_protocol = json.loads((development / 'protocol.json').read_text())
    frozen = json.loads((development / 'frozen.json').read_text())
    if sha(development / 'cases.json') != frozen['cases_sha256'] or sha(development / 'protocol.json') != frozen['protocol_sha256']:
        raise ValueError('development inputs changed')
    cases = json.loads((development / 'cases.json').read_text())
    if set(spatial_partition([case['simulation_anchor_m'] for case in cases])) != {'development'}:
        raise ValueError('only already-used development tiles allowed')
    # Never open baseline held_out.jsonl, summary.json or selected.json.
    benchmark_protocol = json.loads((benchmark / 'protocol.json').read_text())
    receiver = np.asarray(benchmark_protocol['receiver_m'])
    with np.load(benchmark / 'dictionary.npz', allow_pickle=False) as archive:
        positions = archive['positions_m'].copy()
        dictionary = {config['name']: archive[config['name']].copy() for config in dev_protocol['configs']}
    settings = [(config['name'], config, dictionary[config['name']], positions) for config in dev_protocol['configs']]
    fine_hashes = {}
    if fine is not None:
        fine = Path(fine).resolve()
        fine_file = fine / 'dictionary_twoheight_0.125m.npz'
        with np.load(fine_file, allow_pickle=False) as archive:
            fine_positions, fine_features = archive['positions_m'].copy(), archive['features'].copy()
        low = next(config for config in dev_protocol['configs'] if config['name'] == 'low')
        settings.append(('fine_low', low, fine_features, fine_positions))
        fine_hashes = {'fine_protocol': sha(fine / 'protocol.json'), 'fine_dictionary': sha(fine_file)}
    with np.load(development / 'burst_powers.npz', allow_pickle=False) as archive:
        powers, frequency = archive['power'], archive['frequency_hz']
    if powers.shape != (len(cases), 16, len(frequency)):
        raise ValueError('complete sixteen-burst archive required')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'source').mkdir()
    dependencies = [Path(__file__), Path(__file__).with_name('localization.py')]
    for path in dependencies:
        shutil.copyfile(path, output / 'source' / path.name)
    protocol = {'schema': 'openpetcam.development-selective-bearing.v1',
        'classification': 'EXPLORATORY DEVELOPMENT CROSS-VALIDATION ONLY',
        'source_sha256': {path.name: sha(path) for path in dependencies},
        'input_sha256': {'cases': sha(development / 'cases.json'), 'powers': sha(development / 'burst_powers.npz'),
                         'dictionary': sha(benchmark / 'dictionary.npz'), 'development_protocol': sha(development / 'protocol.json'), **fine_hashes},
        'cases': len(cases), 'folds': FOLDS, 'repeats': REPEATS, 'configs': dev_protocol['configs'],
        'dictionary_settings': {name: {'config': config, 'candidates': len(points)} for name, config, _, points in settings},
        'source_tile_folds': {case['case_id']: tile_fold(case['simulation_anchor_m']) for case in cases},
        'arc_policy': f'minimum circular arc containing every material-profiled candidate within{NEAR_SCORE}ofbestcosine; midpoint is bearing candidate',
        'target': 'within30deg of source bearing; background/noemitter is always wrong if acted on',
        'rule_grid': {'similarity': SIMILARITIES, 'margin': MARGINS, 'maximum_arc_width_deg': ARC_WIDTHS},
        'training_target': '>=80%within30deg inclbackgroundfailures, >=max(8,n/20)actions from>=3source-anchor tiles; largestcoverage wins',
        'fold_policy': 'SHA256 seedbearing-development-v1 of whole1mxzsourceanchor tiles, allheights/scenarios/repeats grouped',
        'acceptance_limit': 'Actions are simulated out-of-fold development decisions, NEVER calibrated real camera commands.',
        'limits': ['Already-inspected development domain; no fresh confirmation or realaccuracy.',
                   'Arc is a descriptive score set, not a confidence interval.',
                   'Floor/ceiling aliases, sourcecomb and geometry errors can still produce narrow wrong arcs.',
                   'All missing predictions stay in coverage; background has null source/error.',
                   'Dictionaries are coarse/fine two-height endpointbbox hypotheses, not certified household space.',
                   'Trainingprecision is row-weighted across correlated pairedsourcegroups; not independent-case confidence.',
                   'Occludedcases use their ownactualanchor tilefold, while sourcewaveform seeds can be shared acrossdifferentlocations.',
                   'No baseline held-out predictions/labels/summary/selected file is read.']}
    write_json(output / 'protocol.json', protocol)
    started = time.monotonic()
    bearings = {name: np.degrees(np.arctan2(points[:, 0] - receiver[0], points[:, 2] - receiver[2]))
                for name, _, _, points in settings}
    scored = defaultdict(list)
    for index, case in enumerate(cases):
        if time.monotonic() - started > deadline_s:
            raise TimeoutError('bearing development deadline')
        source = case['source_m']
        true_bearing = math.degrees(math.atan2(source[0] - receiver[0], source[2] - receiver[2])) if source is not None else None
        for repeats in REPEATS:
            power = powers[index, :repeats].astype(float).mean(0)
            for band, config, model_features, candidate_positions in settings:
                feature, rms = ripple_features(power, frequency, **{k: v for k, v in config.items() if k != 'name'})
                rank = rank_candidates(feature, model_features, candidate_positions, near_score=NEAR_SCORE)
                if rank['best_index'] is None:
                    center, width, top1_error = None, None, None
                else:
                    center, width = circular_arc(bearings[band][rank['near_best_candidates']])
                    top1_error = angle_error(float(bearings[band][rank['best_index']]), true_bearing) if true_bearing is not None else None
                scored[f'{band}|{repeats}'].append({'case_id': case['case_id'], 'scenario': case['scenario'],
                    'source_m': source, 'fold': tile_fold(case['simulation_anchor_m']), 'band': band, 'repeats': repeats,
                    'spatial_tile': f"{math.floor(case['simulation_anchor_m'][0])}:{math.floor(case['simulation_anchor_m'][2])}",
                    'estimated_bearing_deg': center, 'arc_width_deg': width,
                    'bearing_error_deg': angle_error(center, true_bearing) if center is not None and true_bearing is not None else None,
                    'top1_bearing_error_deg': top1_error, 'similarity': rank.get('similarity'), 'score_margin': rank.get('score_margin'),
                    'feature_ripple_rms': float(rms)})
        if index % 20 == 0:
            tmp = output / '.progress.tmp'; tmp.write_text(json.dumps({'pid': os.getpid(), 'done': index + 1,
                'total': len(cases), 'elapsed_s': time.monotonic() - started})); tmp.replace(output / 'progress.json')
    decisions, summaries, rules, scenario_groups = [], {}, {}, defaultdict(list)
    for key, rows in scored.items():
        result, rules[key] = cross_validate(rows)
        decisions.extend(result); summaries[key] = summarize(result)
        for row in result:
            scenario_groups[row['scenario'] + '|' + key].append(row)
    with (output / 'decisions.jsonl').open('x') as stream:
        for row in decisions:
            stream.write(json.dumps(row, allow_nan=False) + '\n')
    write_json(output / 'rules.json', rules)
    summary = {'complete': True, 'classification': protocol['classification'], 'comparison': summaries,
        'by_scenario': {key: summarize(rows) for key, rows in scenario_groups.items()},
        'elapsed_s': time.monotonic() - started, 'real_accuracy_measured': False, 'held_out_used': False,
        'protocol_sha256': sha(output / 'protocol.json'), 'limits': protocol['limits']}
    write_json(output / 'summary.json', summary)
    write_json(output / 'complete.json', {'pid': os.getpid(), 'elapsed_s': time.monotonic() - started})
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development', required=True)
    parser.add_argument('--benchmark', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--fine')
    parser.add_argument('--deadline-s', type=float, default=290)
    args = parser.parse_args()
    result = run(args.development, args.benchmark, args.output, fine=args.fine, deadline_s=args.deadline_s)
    print(json.dumps({'complete': result['complete'], 'elapsed_s': result['elapsed_s'],
                      'classification': result['classification']}, indent=2))
