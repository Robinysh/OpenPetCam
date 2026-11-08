"""Paired DEVELOPMENT-only isolation of uncertain forward-model ingredients.

Positions and source/noise draws stay fixed across geometry/material/speed
factors. Independent patch perturbations are uncertainty hypotheses, not a
validated physically closed room. No held-out data or real source truth is used.
"""
import argparse
from collections import defaultdict
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from .geometry import Rig
from .localization import errors, rank_candidates, ripple_features, spatial_partition, summarize_errors
from .localization_development import burst_spectra
from .localization_experiment import NFFT, RATE, perturbed
from .localization_model import build_paths, load_room, render_ir


FACTORS = ('nominal', 'geometry_2cm', 'geometry_10cm', 'material_only', 'speed_only',
           'combined_2cm', 'combined_10cm', 'floor_2cm', 'floor_10cm', 'walls_2cm', 'walls_10cm')
LOW = dict(smooth_hz=120., band=(300., 2000.))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, allow_nan=False); stream.write('\n')


def isolated_models(model, seed):
    """Separate matched geometry/material/speed draws from the same composite."""
    nominal = replace(model, surfaces=tuple(replace(s, reflection=.5) for s in model.surfaces))
    rng = np.random.default_rng(seed)
    composite_small = perturbed(model, rng, .02)
    speed = float(rng.uniform(339, 347))
    composite_large = perturbed(model, np.random.default_rng(seed), .10)
    geometry_small = replace(composite_small, surfaces=tuple(replace(s, reflection=.5) for s in composite_small.surfaces))
    geometry_large = replace(composite_large, surfaces=tuple(replace(s, reflection=.5) for s in composite_large.surfaces))
    material = replace(nominal, surfaces=tuple(replace(base, reflection=other.reflection)
        for base, other in zip(nominal.surfaces, composite_small.surfaces)))
    result = {'nominal': (nominal, 343.), 'geometry_2cm': (geometry_small, 343.),
        'geometry_10cm': (geometry_large, 343.), 'material_only': (material, 343.),
        'speed_only': (nominal, speed), 'combined_2cm': (composite_small, speed),
        'combined_10cm': (composite_large, speed)}
    for label, geometry in (('2cm', geometry_small), ('10cm', geometry_large)):
        result['floor_' + label] = (replace(nominal, surfaces=tuple(
            shifted if base.surface_id == 'floor' else base for base, shifted in zip(nominal.surfaces, geometry.surfaces))), 343.)
        result['walls_' + label] = (replace(nominal, surfaces=tuple(
            shifted if base.surface_id not in ('floor', 'ceiling') else base
            for base, shifted in zip(nominal.surfaces, geometry.surfaces))), 343.)
    return result


def run(development, benchmark, fine, output, *, deadline_s=590):
    development, benchmark, fine, output = map(lambda p: Path(p).resolve(), (development, benchmark, fine, output))
    if not 10 <= deadline_s <= 590:
        raise ValueError('deadline10..590required')
    frozen = json.loads((development / 'frozen.json').read_text())
    if sha(development / 'cases.json') != frozen['cases_sha256'] or sha(development / 'protocol.json') != frozen['protocol_sha256']:
        raise ValueError('development inputs changed')
    cases = [case for case in json.loads((development / 'cases.json').read_text()) if case['scenario'] == 'independent_broadband']
    if len(cases) != 32 or set(spatial_partition([case['source_m'] for case in cases])) != {'development'}:
        raise ValueError('the32frozen development source anchors are required')
    # No baseline held_out.jsonl, summary.json or selected.json reads.
    base_protocol = json.loads((benchmark / 'protocol.json').read_text())
    room_path = Path(base_protocol['room_file'])
    if sha(room_path) != base_protocol['room_sha256']:
        raise ValueError('room model changed')
    model, receiver = load_room(room_path), Rig().pose(0, 0)
    with np.load(fine / 'dictionary_twoheight_0.125m.npz', allow_pickle=False) as archive:
        positions, features = archive['positions_m'].copy(), archive['features'].copy()
    models = {case['case_id']: isolated_models(model, case['room_seed']) for case in cases}
    case_parameters = []
    for case in cases:
        for factor in FACTORS:
            truth, speed = models[case['case_id']][factor]
            case_parameters.append({'case_id': case['case_id'], 'factor': factor, 'source_m': case['source_m'],
                'paired_group': case['paired_group'], 'room_seed': case['room_seed'], 'burst_seeds': case['burst_seeds'],
                'sound_speed_m_s': speed, 'surfaces': [{'id': s.surface_id, 'point_m': s.point_m,
                    'normal': s.normal, 'vertices_m': s.vertices_m, 'pressure_reflection': s.reflection} for s in truth.surfaces]})
    output.mkdir(parents=True, exist_ok=False)
    (output / 'source').mkdir()
    dependencies = [Path(__file__)] + [Path(__file__).with_name(name) for name in (
        'localization.py', 'localization_development.py', 'localization_experiment.py', 'localization_model.py', 'geometry.py')]
    for path in dependencies:
        shutil.copyfile(path, output / 'source' / path.name)
    protocol = {'schema': 'openpetcam.localization-ingredient-sensitivity.v1',
        'classification': 'EXPLORATORY DEVELOPMENT FACTOR ISOLATION ONLY',
        'source_sha256': {path.name: sha(path) for path in dependencies},
        'input_sha256': {'development_protocol': sha(development / 'protocol.json'), 'development_cases': sha(development / 'cases.json'),
            'fine_dictionary': sha(fine / 'dictionary_twoheight_0.125m.npz'), 'room': sha(room_path)},
        'source_groups': 32, 'factor_cases': len(case_parameters), 'factors': FACTORS,
        'source': 'same16independent broadband bursts/noise seeds per source anchor under every factor,25dBrelativeSNR',
        'repeats': 16, 'aggregation': 'arithmetic mean linear power, original signed convolution for each burst',
        'dictionary': 'frozen LOW .125m global TWOheight dictionary; no truth-directedcandidateplacement',
        'config': LOW, 'receiver_m': receiver.receiver_m.tolist(),
        'isolation': 'small/large geometry use identical normal deviates scaled5x; material/speeddraws match composite; all others held nominal',
        'truth_use': 'simulated labels for errors and physical forward generation only, never candidate dictionary selection',
        'limits': ['Already-used development source anchors, correlated sourcegroups; no fresh confirmation/realaccuracy.',
                   'Independent planarpatch shifts can change gaps/visibility; not a validated closed-room remodeling.',
                   'Floor shifts keep physical source/microphone worldcoordinates fixed.',
                   'Material coefficients are uncertain signed-pressure hypotheses, not calibrated real materials.',
                   'Known-IR oracle is a hypothetical representation diagnostic unavailable for unknown real sources.',
                   'All unlocalized cases remain in denominators and errors stay null; acceptance alwaysfalse.']}
    write_json(output / 'protocol.json', protocol); write_json(output / 'case_parameters.json', case_parameters)
    write_json(output / 'frozen.json', {'protocol_sha256': sha(output / 'protocol.json'),
        'case_parameters_sha256': sha(output / 'case_parameters.json'), 'before_any_new_waveforms': True})
    started = time.monotonic(); frequency = np.fft.rfftfreq(NFFT, 1 / RATE)
    rows, power_archive, h_archive = [], [], []
    with (output / 'rankings.jsonl').open('x') as stream:
        for case in cases:
            for factor in FACTORS:
                if time.monotonic() - started > deadline_s:
                    raise TimeoutError('factor study deadline')
                truth, speed = models[case['case_id']][factor]
                paths = build_paths(truth, case['source_m'], receiver.receiver_m, sound_speed=speed, timing='direct_relative')
                h = render_ir(paths, RATE, .25, .01)
                powers = burst_spectra(case, h)
                power, known_power = powers.mean(0), abs(np.fft.rfft(h, n=NFFT))**2
                feature, _ = ripple_features(power, frequency, **LOW)
                known, _ = ripple_features(known_power, frequency, **LOW)
                rank = rank_candidates(feature, features, positions); rank.pop('scores'); rank.pop('near_best_candidates', None)
                oracle = rank_candidates(known, features, positions); oracle.pop('scores'); oracle.pop('near_best_candidates', None)
                empty = {'distance_error_m': None, 'azimuth_error_deg': None}
                error = errors(rank['position_m'], case['source_m'], receiver.receiver_m) if rank['position_m'] is not None else empty
                oracle_error = errors(oracle['position_m'], case['source_m'], receiver.receiver_m) if oracle['position_m'] is not None else empty
                row = {'case_id': case['case_id'], 'paired_group': case['paired_group'], 'factor': factor,
                    'source_m': case['source_m'], **rank, **error,
                    'direct_visible': paths.direct_visible, 'paths': len(paths),
                    'known_ir_oracle': dict(oracle, **oracle_error),
                    'feature_cosine_to_known_ir': float(feature @ known) if np.linalg.norm(known) > 1e-8 else None}
                rows.append(row); stream.write(json.dumps(row, allow_nan=False) + '\n'); stream.flush()
                power_archive.append(power.astype(np.float32)); h_archive.append(h.astype(np.float32))
                tmp = output / '.progress.tmp'; tmp.write_text(json.dumps({'pid': os.getpid(), 'done': len(rows),
                    'total': len(case_parameters), 'elapsed_s': time.monotonic() - started})); tmp.replace(output / 'progress.json')
    groups = defaultdict(list)
    for row in rows:
        groups[row['factor']].append(row)
    np.savez_compressed(output / 'derived_signals.npz', mean_power=np.array(power_archive), signed_ir=np.array(h_archive), frequency_hz=frequency)
    summary = {'complete': True, 'classification': protocol['classification'], 'source_groups': 32,
        'factor_cases': len(rows), 'by_factor': {key: {'waveform': summarize_errors(value),
            'known_ir_oracle': summarize_errors([row['known_ir_oracle'] for row in value]),
            'direct_visible_cases': sum(row['direct_visible'] for row in value),
            'median_feature_cosine_to_known_ir': float(np.median([row['feature_cosine_to_known_ir'] for row in value
                if row['feature_cosine_to_known_ir'] is not None])) if any(row['feature_cosine_to_known_ir'] is not None for row in value) else None}
            for key, value in groups.items()}, 'elapsed_s': time.monotonic() - started,
        'protocol_sha256': sha(output / 'protocol.json'), 'real_accuracy_measured': False, 'held_out_used': False,
        'limits': protocol['limits']}
    write_json(output / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--development', required=True); parser.add_argument('--benchmark', required=True)
    parser.add_argument('--fine', required=True); parser.add_argument('--output', required=True)
    parser.add_argument('--deadline-s', type=float, default=590)
    args = parser.parse_args()
    result = run(args.development, args.benchmark, args.fine, args.output, deadline_s=args.deadline_s)
    print(json.dumps({'complete': result['complete'], 'elapsed_s': result['elapsed_s'],
                      'classification': result['classification']}, indent=2))
