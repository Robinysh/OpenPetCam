"""DEVELOPMENT-only exact-transfer audit of finite-rendered-IR oracles.

Exact noiseless model transfers are unavailable for arbitrary unknown real
sources. This diagnostic corrects representation-oracle bookkeeping only; it is
not a localization method or evidence of real-room accuracy.
"""
import argparse
from collections import defaultdict
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np

from .localization import errors, rank_candidates, ripple_features, summarize_errors
from .localization_model import RoomModel, Surface, build_paths, render_ir, render_transfer
from .localization_resolution import sha, write_json


LOW = {'smooth_hz': 120., 'band': (300., 2000.)}
RATE, NFFT = 16000, 8192


def exact_feature(paths, frequency):
    """One active path has flat power: never normalize numerical phase roundoff.

    Zero-path, single-active-path and other flat-power queries stay explicitly
    uninformative. Multiple paths sharing a delay can also yield flat power.
    """
    active = [path for path in paths if path.gain != 0]
    power = abs(render_transfer(active, frequency)) ** 2
    feature, rms = ripple_features(power, frequency, **LOW)
    informative = len(active) >= 2 and np.linalg.norm(feature) > 1e-8
    if not informative:
        feature = np.zeros_like(feature)
    return feature, {'active_paths': len(active), 'exact_log_power_ripple_rms': float(rms),
                     'informative': bool(informative)}


def uninformative_rank():
    return {'status': 'uninformative', 'accepted': False, 'best_index': None,
            'position_m': None, 'similarity': None, 'score_margin': None,
            'reason': 'No varying relative-path power contrast; one unknown-gain path cannot identify location.'}


def parameter_model(parameter):
    surfaces = []
    for surface in parameter['surfaces']:
        coefficient = surface.get('pressure_reflection', surface.get('reflection'))
        surfaces.append(Surface(surface['id'], tuple(surface['point_m']), tuple(surface['normal']),
            None if surface['vertices_m'] is None else tuple(map(tuple, surface['vertices_m'])), coefficient))
    return RoomModel(tuple(surfaces))


def run(study_dirs, fine_dir, output_dir, *, deadline_s=290):
    if not 10 <= deadline_s <= 290:
        raise ValueError('cooperative deadline must be10..290seconds')
    studies = [Path(path).resolve() for path in study_dirs]
    fine, output = Path(fine_dir).resolve(), Path(output_dir).resolve()
    fine_file = fine / 'dictionary_twoheight_0.125m.npz'
    fine_protocol = json.loads((fine / 'protocol.json').read_text())
    fine_frozen = json.loads((fine / 'dictionary_frozen.json').read_text())
    if sha(fine_file) != fine_frozen['sha256']:
        raise ValueError('fine dictionary changed from its frozen version')
    inputs, packets = {}, []
    for study in studies:
        protocol = json.loads((study / 'protocol.json').read_text())
        frozen = json.loads((study / 'frozen.json').read_text())
        if sha(study / 'protocol.json') != frozen['protocol_sha256'] or sha(study / 'case_parameters.json') != frozen['case_parameters_sha256']:
            raise ValueError('frozen factor-study inputs changed')
        dictionary_hash = protocol['input_sha256'].get('fine_dictionary', protocol['input_sha256'].get('dictionary'))
        if dictionary_hash != sha(fine_file):
            raise ValueError('factor study used a different candidate dictionary')
        if protocol['config'] != json.loads(json.dumps(LOW)):
            raise ValueError('factor study used different feature parameters')
        receiver = protocol.get('receiver_m', fine_protocol['receiver_m'])
        if receiver != fine_protocol['receiver_m']:
            raise ValueError('factor study and dictionary receiver assumptions differ')
        parameters = json.loads((study / 'case_parameters.json').read_text())
        rows = [json.loads(line) for line in (study / 'rankings.jsonl').read_text().splitlines()]
        if len(rows) != len(parameters):
            raise ValueError('factor parameters/rows differ in length')
        for parameter, row in zip(parameters, rows):
            if (parameter['case_id'], parameter['factor'], parameter['source_m']) != (row['case_id'], row['factor'], row['source_m']):
                raise ValueError('factor case identity differs between parameters and saved row')
            if 'tail' in parameter['factor']:
                raise ValueError('diffuse tails cannot be replaced by a first-order exact transfer')
            packets.append((study.name, protocol, parameter, row, receiver))
        inputs[study.name] = {name: sha(study / name) for name in ('protocol.json', 'case_parameters.json', 'rankings.jsonl')}
    output.mkdir(parents=True, exist_ok=False)
    source_dir = output / 'source'; source_dir.mkdir()
    dependencies = [Path(__file__)] + [Path(__file__).with_name(name) for name in
        ('localization.py', 'localization_model.py', 'localization_resolution.py', 'geometry.py')]
    for source in dependencies:
        shutil.copyfile(source, source_dir / source.name)
    protocol = {'schema': 'openpetcam.exact-transfer-oracle-audit.v1', 'created_epoch': time.time(),
        'classification': 'EXPLORATORY DEVELOPMENT REPRESENTATION-ORACLE AUDIT ONLY',
        'source_sha256': {path.name: sha(path) for path in dependencies}, 'study_inputs': inputs,
        'fine_dictionary_sha256': sha(fine_file), 'fine_protocol_sha256': sha(fine / 'protocol.json'),
        'factor_cases': len(packets), 'studies': [path.name for path in studies],
        'band': LOW, 'sample_rate_hz': RATE, 'nfft': NFFT, 'deadline_s': deadline_s,
        'query': 'Exact signed complex path sum H, then |H|^2 and frozen LOW ripple; never finite fractional-IR interpolation.',
        'zero_single_path_policy': 'Fewer than2nonzero-gain paths, or flat exact-power ripple, explicitly abstain with NULL coordinates/errors.',
        'candidate_dictionary': 'Same frozen full TWOheight .125m LOW dictionary as factor studies.',
        'finite_comparison': 'Retain original finite-rendered-IR oracle decisions as an archived baseline, without modifying them.',
        'finite_ripple_diagnostic': 'Re-render signed fractional IR only to quantify numerical interpolation ripple; never substitute it for exact H.',
        'frozen_before_computation': True, 'tail_policy': 'No diffuse-tail cases included; any future tail cases must retain a separately labeled rendered-tail oracle.',
        'limits': ['Exact transfers are ideal simulator oracles, unavailable for arbitrary unknown real sources.',
            'All source anchors and factor draws are reused development data; no new confirmation or real accuracy claim.',
            'Geometry/material/speed and independent-patch uncertainty assumptions remain unchanged.',
            'All cases remain in success denominators, uninformative errors stay NULL, and acceptance alwaysfalse.',
            'No v1held-out labels/predictions, summary or selected-rule file is opened.',
            'Development-repeats and diffuse-tail extensions are outside this bounded480factor-case pass.']}
    write_json(output / 'protocol.json', protocol)
    with np.load(fine_file, allow_pickle=False) as archive:
        positions, features = archive['positions_m'], archive['features']
    frequency = np.fft.rfftfreq(NFFT, 1 / RATE)
    started = time.monotonic()
    rows = []
    with (output / 'comparisons.jsonl').open('x') as stream:
        for index, (study, source_protocol, parameter, original, receiver) in enumerate(packets):
            if time.monotonic() - started > deadline_s:
                raise TimeoutError('exact-transfer audit deadline reached; partial rows preserved')
            speed = parameter.get('sound_speed_m_s', source_protocol.get('sound_speed_m_s'))
            paths = build_paths(parameter_model(parameter), parameter['source_m'], receiver,
                                sound_speed=speed, timing='direct_relative')
            feature, contrast = exact_feature(paths, frequency)
            rank = rank_candidates(feature, features, positions) if contrast['informative'] else uninformative_rank()
            rank.pop('scores', None); rank.pop('near_best_candidates', None)
            rank.update(errors(rank['position_m'], parameter['source_m'], receiver) if rank['position_m'] is not None else {
                'distance_error_m': None, 'azimuth_error_deg': None})
            h = render_ir(paths, RATE, .25, .01)
            finite_feature, finite_rms = ripple_features(abs(np.fft.rfft(h, n=NFFT)) ** 2, frequency, **LOW)
            row = {'study': study, 'case_id': parameter['case_id'], 'paired_group': parameter['paired_group'],
                'factor': parameter['factor'], 'source_m': parameter['source_m'], 'sound_speed_m_s': speed,
                'physical_paths': len(paths), 'direct_visible': paths.direct_visible, **contrast,
                'finite_ir_log_power_ripple_rms': float(finite_rms),
                'finite_ir_normalized_feature_norm': float(np.linalg.norm(finite_feature)),
                'archived_finite_ir_oracle': original['known_ir_oracle'], 'exact_transfer_oracle': rank}
            rows.append(row); stream.write(json.dumps(row, allow_nan=False) + '\n'); stream.flush()
            temporary = output / '.progress.tmp'
            temporary.write_text(json.dumps({'pid': os.getpid(), 'done': index + 1, 'total': len(packets),
                'elapsed_s': time.monotonic() - started})); temporary.replace(output / 'progress.json')
    groups = defaultdict(list)
    for row in rows:
        groups[row['study'] + '|' + row['factor']].append(row)
    def compare(group):
        return {'cases': len(group), 'finite_rendered_ir_oracle': summarize_errors([row['archived_finite_ir_oracle'] for row in group]),
            'exact_transfer_oracle': summarize_errors([row['exact_transfer_oracle'] for row in group]),
            'zero_path_cases': sum(row['active_paths'] == 0 for row in group),
            'single_path_cases': sum(row['active_paths'] == 1 for row in group),
            'finite_candidates_from_single_path_interpolation_contrast': sum(row['active_paths'] == 1 and row['archived_finite_ir_oracle']['best_index'] is not None for row in group)}
    summary = {'complete': True, 'classification': protocol['classification'], 'overall': compare(rows),
        'by_factor': {key: compare(value) for key, value in groups.items()},
        'elapsed_s': time.monotonic() - started, 'protocol_sha256': sha(output / 'protocol.json'),
        'held_out_used': False, 'real_accuracy_measured': False, 'limits': protocol['limits']}
    write_json(output / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--studies', nargs='+', required=True)
    parser.add_argument('--fine', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--deadline-s', type=float, default=290)
    args = parser.parse_args()
    result = run(args.studies, args.fine, args.output, deadline_s=args.deadline_s)
    print(json.dumps({'complete': result['complete'], 'cases': result['overall']['cases'],
        'elapsed_s': result['elapsed_s'], 'classification': result['classification']}, indent=2))
