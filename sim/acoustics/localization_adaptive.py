"""Exploratory development-only data-driven local refinement of a global grid.

No true source coordinate chooses candidates. Eight spatially distinct high-score
global candidates each seed a fixed local x/z lattice at the same modeled height.
Acceptance remains false; denser searches can fit background/source/model error.
"""
import argparse
from collections import defaultdict, OrderedDict
import hashlib
import json
from pathlib import Path
import shutil
import os
import time

import numpy as np

from .localization import errors, rank_candidates, ripple_features, spatial_partition, summarize_errors
from .localization_model import load_room
from .localization_resolution import dictionary as build_dictionary


FAMILIES = ('independent_broadband', 'harmonic_fixed_pitch', 'geometry_10cm', 'no_source_background')
REPEATS = 4
CENTERS = 8
SEPARATION_M = .5
LOCAL_STEP_M = .03125
LOCAL_RADIUS_M = .0625


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def distinct_centers(profile, positions, *, count=CENTERS, separation=SEPARATION_M):
    scores = np.array([-np.inf if value is None else value for value in profile])
    selected = []
    for index in np.argsort(-scores, kind='stable'):
        if not np.isfinite(scores[index]):
            break
        if not selected or np.min(np.linalg.norm(positions[selected] - positions[index], axis=1)) >= separation:
            selected.append(int(index))
            if len(selected) == count:
                break
    return selected


def local_positions(centers, global_positions, bounds):
    """Fixed lattice around scored centers, clipped only to declared x/z bounds."""
    existing = {tuple(point) for point in global_positions}
    result = set()
    offsets = np.arange(-2, 3) * LOCAL_STEP_M
    for point in centers:
        for dx in offsets:
            for dz in offsets:
                proposal = (float(point[0] + dx), float(point[1]), float(point[2] + dz))
                if (bounds[0] <= proposal[0] <= bounds[1] and bounds[2] <= proposal[2] <= bounds[3]
                        and np.hypot(proposal[0], proposal[2]) >= .6 and proposal not in existing):
                    result.add(proposal)
    return np.asarray(sorted(result), float).reshape((-1, 3))


def describe(rows, key):
    source = [row[key] for row in rows if row['source_m'] is not None]
    background = [row[key] for row in rows if row['source_m'] is None]
    similarities = [r['similarity'] for r in background if r.get('similarity') is not None]
    return {'source': summarize_errors(source), 'background_cases': len(background),
            'background_conditional_candidates': sum(r['position_m'] is not None for r in background),
            'background_median_similarity': float(np.median(similarities)) if similarities else None}


def run(development, fine, room, output, *, deadline_s=590):
    development, fine, room, output = map(lambda p: Path(p).resolve(), (development, fine, room, output))
    if not 10 <= deadline_s <= 590:
        raise ValueError('deadline10..590required')
    dev_protocol = json.loads((development / 'protocol.json').read_text())
    frozen = json.loads((development / 'frozen.json').read_text())
    if sha(development / 'cases.json') != frozen['cases_sha256'] or sha(development / 'protocol.json') != frozen['protocol_sha256']:
        raise ValueError('development input changed')
    if sha(room) != dev_protocol['room_sha256']:
        raise ValueError('room changed')
    cases = json.loads((development / 'cases.json').read_text())
    chosen = [(index, case) for index, case in enumerate(cases) if case['scenario'] in FAMILIES]
    if len(chosen) != 128 or set(spatial_partition([c['simulation_anchor_m'] for _, c in chosen])) != {'development'}:
        raise ValueError('predeclared128development cases required')
    dictionary_file = fine / 'dictionary_twoheight_0.125m.npz'
    with np.load(dictionary_file, allow_pickle=False) as archive:
        positions, features = archive['positions_m'].copy(), archive['features'].copy()
    with np.load(development / 'burst_powers.npz', allow_pickle=False) as archive:
        powers, frequency = archive['power'], archive['frequency_hz']
    fine_protocol = json.loads((fine / 'protocol.json').read_text())
    receiver = np.asarray(fine_protocol['receiver_m'])
    model = load_room(room)
    endpoints = np.asarray([p for wall in model.declared_wall_extents for p in (wall['a_xz_m'], wall['b_xz_m'])])
    bounds = (float(endpoints[:, 0].min()), float(endpoints[:, 0].max()),
              float(endpoints[:, 1].min()), float(endpoints[:, 1].max()))
    output.mkdir(parents=True, exist_ok=False); (output / 'source').mkdir()
    dependencies = [Path(__file__)] + [Path(__file__).with_name(name) for name in (
        'localization.py', 'localization_model.py', 'localization_resolution.py', 'geometry.py')]
    for path in dependencies:
        shutil.copyfile(path, output / 'source' / path.name)
    protocol = {'schema': 'openpetcam.development-adaptive-grid.v1',
        'classification': 'EXPLORATORY DEVELOPMENT ADAPTIVE GRID ONLY',
        'source_sha256': {path.name: sha(path) for path in dependencies},
        'input_sha256': {'cases': sha(development / 'cases.json'), 'powers': sha(development / 'burst_powers.npz'),
                         'dictionary': sha(dictionary_file), 'room': sha(room)},
        'case_ids': [case['case_id'] for _, case in chosen], 'families': FAMILIES,
        'repeats': REPEATS, 'centers': CENTERS, 'center_separation_m': SEPARATION_M,
        'local_xz_step_m': LOCAL_STEP_M, 'local_xz_radius_m': LOCAL_RADIUS_M,
        'same_two_modeled_heights_only': True, 'global_candidates': len(positions), 'xz_bounds': bounds,
        'config': {'smooth_hz': 120., 'band': (300., 2000.)}, 'receiver_m': receiver.tolist(),
        'placement': 'Only observed spectral scores select8spatially distinct centers; fixed5x5local x/z lattices percenter, never true positions.',
        'acceptance': 'Alwaysfalse; fine scores and dense local refinement are not confidence.',
        'limits': ['Already-used development domain/families; no fresh confirmation or realaccuracy.',
                   'Fixedharmonic,10cmgeometryandbackground controls processed by the identical algorithm.',
                   'Candidate domain/geometry/source assumptions remain uncertain; refinements can fit wrong sources or background.',
                   'No waveforms generated and no baseline heldout/summary/selected file read.',
                   'All missing source predictions retained; background has no emitting-source coordinates/errors.']}
    write_json(output / 'protocol.json', protocol)
    started = time.monotonic(); rows, cache = [], OrderedDict()
    def progress(_phase=None, _done=None, _total=None):
        if time.monotonic() - started > deadline_s:
            raise TimeoutError('adaptive development deadline')
    with (output / 'rankings.jsonl').open('x') as stream:
        for index, case in chosen:
            progress()
            power = powers[index, :REPEATS].astype(float).mean(0)
            observed, _ = ripple_features(power, frequency, **protocol['config'])
            baseline = rank_candidates(observed, features, positions)
            centers = distinct_centers(baseline['scores'], positions)
            local = local_positions(positions[centers], positions, bounds)
            missing = np.array([point for point in local if tuple(point) not in cache], float).reshape((-1, 3))
            if len(missing):
                new_features, _, _ = build_dictionary(model, missing, receiver, frequency, progress)
                for point, feature in zip(missing, new_features):
                    cache[tuple(point)] = feature
            if len(local):
                local_features = np.array([cache[tuple(point)] for point in local])
                adaptive = rank_candidates(observed, np.concatenate((features, local_features)), np.concatenate((positions, local)))
            else:
                adaptive = dict(baseline)
            for rank in (baseline, adaptive):
                rank.pop('scores', None); rank.pop('near_best_candidates', None)
                rank.update(errors(rank['position_m'], case['source_m'], receiver) if (
                    rank['position_m'] is not None and case['source_m'] is not None) else {
                    'distance_error_m': None, 'azimuth_error_deg': None})
            row = {'case_id': case['case_id'], 'paired_group': case['paired_group'], 'scenario': case['scenario'],
                'source_m': case['source_m'], 'global_baseline': baseline, 'adaptive': adaptive,
                'global_center_indexes': centers, 'new_candidates_m': local.tolist(), 'new_candidate_count': len(local)}
            rows.append(row); stream.write(json.dumps(row, allow_nan=False) + '\n'); stream.flush()
            while len(cache) > 8192:
                cache.popitem(last=False)
            tmp = output / '.progress.tmp'; tmp.write_text(json.dumps({'pid': os.getpid(), 'done': len(rows),
                'total': len(chosen), 'elapsed_s': time.monotonic() - started})); tmp.replace(output / 'progress.json')
    groups = defaultdict(list)
    for row in rows:
        groups[row['scenario']].append(row)
    summary = {'complete': True, 'classification': protocol['classification'], 'cases': len(rows),
        'global_baseline': describe(rows, 'global_baseline'), 'adaptive': describe(rows, 'adaptive'),
        'by_scenario': {scenario: {'global_baseline': describe(value, 'global_baseline'), 'adaptive': describe(value, 'adaptive')}
                        for scenario, value in groups.items()}, 'elapsed_s': time.monotonic() - started,
        'new_candidate_counts': {'median': float(np.median([r['new_candidate_count'] for r in rows])),
                                 'maximum': max(r['new_candidate_count'] for r in rows)},
        'protocol_sha256': sha(output / 'protocol.json'), 'held_out_used': False, 'real_accuracy_measured': False,
        'limits': protocol['limits']}
    write_json(output / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('development', 'fine', 'room', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--deadline-s', type=float, default=590)
    args = parser.parse_args()
    result = run(args.development, args.fine, args.room, args.output, deadline_s=args.deadline_s)
    print(json.dumps({'complete': result['complete'], 'elapsed_s': result['elapsed_s'],
                      'classification': result['classification']}, indent=2))
