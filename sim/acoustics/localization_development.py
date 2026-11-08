"""Exploratory DEVELOPMENT-only repeated-burst and conditional-prior study.

No held-out labels, predictions or summary are opened. Repeats share one fixed
room transfer, with independent waveform/noise draws except explicit replay and
fixed-harmonic controls. These are synthetic bursts, not validated dog barks.
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
from scipy import signal

from .geometry import Rig
from .localization import errors, rank_candidates, ripple_features, spatial_partition, spectrum, summarize_errors
from .localization_experiment import CONFIGS, NFFT, RATE, excitation, perturbed
from .localization_model import build_paths, load_room, render_ir


SCENARIOS = ("independent_broadband", "replayed_broadband", "colored_source",
             "harmonic_variable_pitch", "harmonic_fixed_pitch", "geometry_2cm",
             "geometry_10cm", "diffuse_tail", "directional_mic", "low_snr",
             "occluded_source", "no_source_background")
REPEATS = (1, 4, 16)
PRIOR_NAMES = ("all_candidates", "low_height", "range_4m", "low_height_range_4m")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, document):
    temporary = Path(str(path) + '.tmp')
    temporary.write_text(json.dumps(document, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def priors(positions, receiver, floor):
    """Conditional source hypotheses, never surveyed household/free-space truth."""
    positions, receiver = np.asarray(positions), np.asarray(receiver)
    low = np.abs(positions[:, 1] - (floor + .45)) <= .1
    nearby = np.linalg.norm(positions - receiver, axis=1) <= 4
    return dict(zip(PRIOR_NAMES, (np.ones(len(positions), bool), low, nearby, low & nearby)))


def prior_support(source, receiver, floor):
    if source is None:
        return {name: None for name in PRIOR_NAMES}
    low = abs(source[1] - (floor + .45)) <= .1
    nearby = np.linalg.norm(np.asarray(source) - receiver) <= 4
    return dict(zip(PRIOR_NAMES, map(bool, (True, low, nearby, low and nearby))))


def freeze_cases(positions, visible, *, count, spacing, seed):
    rng = np.random.default_rng(seed)
    indexes = np.flatnonzero(np.asarray(spatial_partition(positions)) == 'development')
    if len(indexes) < 4:
        raise ValueError('development tile support is insufficient')
    cases = []
    for i in range(count):
        scenario = SCENARIOS[i % len(SCENARIOS)]
        if i % len(SCENARIOS) == 0:
            paired_index = int(rng.choice(indexes))
            paired_source = positions[paired_index].copy()
            for _ in range(1000):
                candidate = paired_source.copy()
                candidate[[0, 2]] += rng.uniform(-.35 * spacing, .35 * spacing, 2)
                if spatial_partition([candidate])[0] == 'development':
                    paired_source = candidate
                    break
            else:
                raise ValueError('could not jitter source within development partition')
            seed_sequence = np.random.SeedSequence(int(rng.integers(1, 2**31)))
            draws = [int(s.generate_state(1)[0]) for s in seed_sequence.spawn(17)]
        index, source = paired_index, paired_source.copy()
        if scenario == 'occluded_source':
            eligible = indexes[~visible[indexes]]
            if not len(eligible):
                raise ValueError('occlusion control requires blocked candidates')
            index = int(rng.choice(eligible)); source = positions[index].copy()
        cases.append({'case_id': f'development-repeat-{i:05d}', 'scenario': scenario,
                      'source_m': None if scenario == 'no_source_background' else source.tolist(),
                      'simulation_anchor_m': source.tolist(), 'source_grid_index': index,
                      'paired_group': i // len(SCENARIOS),
                      'split': 'development', 'room_seed': draws[0], 'replay_seed': draws[1],
                      'burst_seeds': draws[1:], 'anchor_direct_visible': bool(visible[index])})
    return cases


def fixed_transfer(model, case, receiver):
    """All repeats retain geometry, material, speed, direction and diffuse tail."""
    rng = np.random.default_rng(case['room_seed'])
    scenario = case['scenario']
    sigma = .02 if scenario == 'geometry_2cm' else .10 if scenario == 'geometry_10cm' else 0
    truth = perturbed(model, rng, sigma) if sigma else replace(model, surfaces=tuple(
        replace(surface, reflection=.5) for surface in model.surfaces))
    speed = float(rng.uniform(339, 347)) if sigma else 343.
    paths = build_paths(truth, case['simulation_anchor_m'], receiver.receiver_m,
                        receiver_axis=receiver.receiver_axis,
                        mic_directivity=.5 if scenario == 'directional_mic' else 0,
                        sound_speed=speed, timing='direct_relative')
    h = render_ir(paths, RATE, .25, .01)
    if scenario == 'diffuse_tail':
        t = np.arange(len(h)) / RATE
        tail = rng.normal(size=len(h)) * np.exp(-np.maximum(t - .035, 0) / .05) * (t >= .035)
        tail *= .35 * np.linalg.norm(h) / max(np.linalg.norm(tail), 1e-20)
        h += tail
    return h, {'direct_visible': paths.direct_visible, 'paths': len(paths),
               'ir_energy': float(h @ h), 'sound_speed_m_s': speed,
               'geometry_sigma_m': sigma, 'fixed_across_bursts': True}


def source_burst(case, seed):
    scenario = case['scenario']
    rng = np.random.default_rng(case['replay_seed'] if scenario == 'replayed_broadband' else seed)
    if scenario == 'harmonic_fixed_pitch':
        t = np.arange(int(RATE * .28)) / RATE
        waveform = .08 * rng.normal(size=len(t)) + sum(
            np.sin(2 * np.pi * 220 * k * t + rng.uniform(-np.pi, np.pi)) / k for k in range(1, 12))
        return waveform * (1 - np.exp(-t / .004)) * np.exp(-t / .09)
    family = 'harmonic_source' if scenario == 'harmonic_variable_pitch' else scenario
    return excitation(rng, family, held_out=False)


def burst_spectra(case, h):
    """Separate signed convolution per burst; no alignment or waveform stacking."""
    result = []
    snr = 5. if case['scenario'] == 'low_snr' else 25.
    for seed in case['burst_seeds']:
        source = source_burst(case, seed)
        recording = signal.fftconvolve(source, h)
        rng = np.random.default_rng(seed ^ 0x63E2471)
        noise = rng.normal(size=len(recording))
        if case['scenario'] == 'no_source_background':
            recording[:] = 0
            noise *= .001 / np.sqrt(np.mean(noise**2))
        else:
            noise *= np.linalg.norm(recording) / max(np.linalg.norm(noise), 1e-20) * 10**(-snr / 20)
        _, power = spectrum(recording + noise, RATE, nfft=NFFT)
        result.append(power)
    return np.array(result)


def nested_mean_powers(powers, repeats=REPEATS):
    powers = np.asarray(powers, float)
    if powers.ndim != 2 or not np.isfinite(powers).all() or np.any(powers < 0):
        raise ValueError('finite nonnegative burst-by-frequency power required')
    if not repeats or any(not isinstance(n, int) or not 1 <= n <= len(powers) for n in repeats):
        raise ValueError('repeat counts must fit existing independent burst spectra')
    return {n: powers[:n].mean(axis=0) for n in repeats}


def rank_subset(feature, dictionary, positions, mask):
    indexes = np.flatnonzero(mask)
    if len(indexes) < 2:
        raise ValueError('every conditional prior requires at least two candidate positions')
    rank = rank_candidates(feature, dictionary[indexes], positions[indexes])
    rank.pop('scores')
    rank['near_best_count'] = len(rank.pop('near_best_candidates', []))
    if rank['best_index'] is not None:
        rank['best_index'] = int(indexes[rank['best_index']])
    return rank


def describe(rows):
    """No truth/error is assigned to the no-emitting-source background control."""
    source_rows = [row for row in rows if row['source_m'] is not None]
    background = [row for row in rows if row['source_m'] is None]
    return {'source_cases': len(source_rows), 'all_source_cases': summarize_errors(source_rows),
            'true_prior_support': summarize_errors([r for r in source_rows if r['true_prior_support']]),
            'wrong_prior_support': summarize_errors([r for r in source_rows if not r['true_prior_support']]),
            'background_only_cases': len(background),
            'background_conditional_candidates': sum(r['best_index'] is not None for r in background),
            'background_positive_margin': sum((r.get('score_margin') or 0) > 1e-9 for r in background),
            'median_feature_cosine_to_known_ir': float(np.median([r['feature_cosine_to_known_ir'] for r in rows
               if r['feature_cosine_to_known_ir'] is not None])) if any(r['feature_cosine_to_known_ir'] is not None for r in rows) else None}


def run(benchmark, output, *, cases=384, seed=20261108, deadline_s=590):
    if not 24 <= cases <= 1536 or not 10 <= deadline_s <= 3600:
        raise ValueError('bounded cases24..1536, deadline10..3600 required')
    benchmark, output = Path(benchmark).resolve(), Path(output).resolve()
    # Deliberately never open held_out.jsonl, summary.json or selected.json.
    frozen_protocol = json.loads((benchmark / 'protocol.json').read_text())
    if frozen_protocol['configs'] != json.loads(json.dumps(CONFIGS)):
        raise ValueError('frozen feature definitions differ from imported implementation')
    room_path = Path(frozen_protocol['room_file'])
    if sha(room_path) != frozen_protocol['room_sha256']:
        raise ValueError('frozen room model changed')
    model, receiver = load_room(room_path), Rig().pose(0, 0)
    floor = next(s.point_m[1] for s in model.surfaces if s.surface_id == 'floor')
    with np.load(benchmark / 'dictionary.npz', allow_pickle=False) as archive:
        positions, visible = archive['positions_m'].copy(), archive['direct_visible'].copy()
        dictionaries = {config['name']: archive[config['name']].copy() for config in CONFIGS}
    masks = priors(positions, receiver.receiver_m, floor)
    case_protocol = freeze_cases(positions, visible, count=cases,
        spacing=frozen_protocol['source_grid_spacing_m'], seed=seed)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    source_dir = output / 'source'; source_dir.mkdir()
    dependencies = [Path(__file__)] + [Path(__file__).with_name(name) for name in (
        'localization.py', 'localization_experiment.py', 'localization_model.py', 'geometry.py')]
    for path in dependencies:
        shutil.copyfile(path, source_dir / path.name)
    protocol = {'schema': 'openpetcam.localization-development.v1', 'seed': seed,
                'classification': 'EXPLORATORY DEVELOPMENT EVIDENCE ONLY',
                'benchmark_protocol_sha256': sha(benchmark / 'protocol.json'),
                'dictionary_sha256': sha(benchmark / 'dictionary.npz'), 'room_sha256': sha(room_path),
                'source_sha256': {path.name: sha(path) for path in dependencies},
                'cases': cases, 'repeats': REPEATS, 'configs': CONFIGS, 'scenarios': SCENARIOS,
                'sample_rate_hz': RATE, 'nfft': NFFT, 'deadline_s': deadline_s,
                'partition': 'development tiles ONLY, SHA256 seed731; already-used domain, no confirmatory claim',
                'source': 'synthetic decaying noise/colored/harmonic burst; not a validated dog bark',
                'repetition': 'Nested1/4/16 prefixes of same16bursts; each burst has independent noise. Geometry, material, speed, direction and diffuse tail fixed within case.',
                'scenario_pairing': 'Each12scenario group shares source anchor and random seeds, except the explicitly occluded anchor. Independent/replay broadband share the exact first waveform/noise draw.',
                'aggregation': 'Arithmetic mean of LINEAR power spectra; no waveform stacking, gain adjustment, alignment or original sample edits.',
                'controls': ['independent broadband spectra', 'identical broadband waveform replay with independent noise',
                             'fixed220Hz harmonic comb vs varying140..280Hz fundamental',
                             'absoluteRMS.001 background-only/noemitter; no source truth/error fabricated'],
                'priors': {name: {'candidate_count': int(mask.sum())} for name, mask in masks.items()},
                'prior_definitions': {'all_candidates': 'all frozen endpointbbox candidates; not certified enclosed/free/walkable space',
                    'low_height': 'within.1m of nominal saved floor+.45m; conditional pet-height hypothesis only',
                    'range_4m': 'Euclidean distance<=4m from assumed microphone; conditional range hypothesis only',
                    'low_height_range_4m': 'intersection of conditional low-height/range hypotheses'},
                'prior_reporting': 'Each prior evaluated on all source cases, declared true-support subset and wrong-prior complement; no true-label-based mask selection.',
                'acceptance': 'Alwaysfalse; no new real or held-out confidence calibration.',
                'limits': ['No v1held-out labels/predictions/summary opened.',
                           'All metrics simulated, exploratory and development-only.',
                           'Repeated bursts assume a stationary source position and room response; real barks may move.',
                           'Welch/Hann transient spectral bias and forward-model errors do not vanish by averaging.',
                           'No surveyed household footprint, source/mic calibration, diffraction/higher-order validation.',
                           'Cases share spatial tiles; no unclustered confidence intervals.']}
    write_json(output / 'protocol.json', protocol)
    write_json(output / 'cases.json', case_protocol)
    write_json(output / 'frozen.json', {'protocol_sha256': sha(output / 'protocol.json'),
               'cases_sha256': sha(output / 'cases.json'), 'before_any_waveforms': True})
    frequency = np.fft.rfftfreq(NFFT, 1 / RATE)
    all_rows, oracle_rows, powers_archive = [], [], []
    with (output / 'rankings.jsonl').open('x') as stream:
        for case_i, case in enumerate(case_protocol):
            if time.monotonic() - started > deadline_s:
                raise TimeoutError('development study deadline; partial rows preserved')
            h, diagnostic = fixed_transfer(model, case, receiver)
            diagnostic['anchor_transfer_unused_no_emitter'] = case['source_m'] is None
            powers = burst_spectra(case, h)
            powers_archive.append(powers.astype(np.float32))
            mean_powers = nested_mean_powers(powers)
            known_power = abs(np.fft.rfft(h, n=NFFT))**2 if case['source_m'] is not None else np.zeros_like(frequency)
            supports = prior_support(case['source_m'], receiver.receiver_m, floor)
            for config in CONFIGS:
                config_name = config['name']
                arguments = {key: value for key, value in config.items() if key != 'name'}
                known_feature, _ = ripple_features(known_power, frequency, **arguments)
                for prior_name, mask in masks.items():
                    ideal = rank_subset(known_feature, dictionaries[config_name], positions, mask)
                    ideal_errors = errors(ideal['position_m'], case['source_m'], receiver.receiver_m) if (
                        ideal['position_m'] is not None and case['source_m'] is not None) else {
                        'distance_error_m': None, 'azimuth_error_deg': None}
                    oracle_rows.append({'case_id': case['case_id'], 'scenario': case['scenario'], 'band': config_name,
                        'prior': prior_name, **ideal, **ideal_errors, 'source_m': case['source_m'],
                        'true_prior_support': supports[prior_name]})
                for repeats, power in mean_powers.items():
                    feature, ripple_rms = ripple_features(power, frequency, **arguments)
                    known_norm = np.linalg.norm(known_feature)
                    similarity = float(feature @ known_feature) if known_norm > 1e-8 and case['source_m'] is not None else None
                    for prior_name, mask in masks.items():
                        rank = rank_subset(feature, dictionaries[config_name], positions, mask)
                        error = errors(rank['position_m'], case['source_m'], receiver.receiver_m) if (
                            rank['position_m'] is not None and case['source_m'] is not None) else {
                            'distance_error_m': None, 'azimuth_error_deg': None}
                        row = {'case_id': case['case_id'], 'scenario': case['scenario'], 'source_m': case['source_m'],
                            'band': config_name, 'prior': prior_name, 'repeats': repeats,
                            'true_prior_support': supports[prior_name], 'candidate_count': int(mask.sum()),
                            **rank, **error, 'diagnostic': diagnostic, 'feature_ripple_rms': float(ripple_rms),
                            'feature_cosine_to_known_ir': similarity}
                        all_rows.append(row); stream.write(json.dumps(row, allow_nan=False) + '\n')
            stream.flush()
            write_json(output / 'progress.json', {'pid': os.getpid(), 'cases_done': case_i + 1,
                       'cases_total': cases, 'elapsed_s': time.monotonic() - started, 'updated': time.time()})
    groups, scenario_groups, oracle_groups = defaultdict(list), defaultdict(list), defaultdict(list)
    for row in all_rows:
        key = f"{row['band']}|{row['prior']}|{row['repeats']}"
        groups[key].append(row); scenario_groups[row['scenario'] + '|' + key].append(row)
    for row in oracle_rows:
        oracle_groups[f"{row['band']}|{row['prior']}"].append(row)
    np.savez_compressed(output / 'burst_powers.npz', power=np.array(powers_archive), frequency_hz=frequency)
    write_json(output / 'known_transfer_oracle.json', oracle_rows)
    summary = {'complete': True, 'classification': protocol['classification'], 'cases': cases,
               'primary_population': 'all simulated emitting-source development cases; background control has no source/error',
               'comparison': {key: describe(rows) for key, rows in groups.items()},
               'by_scenario': {key: describe(rows) for key, rows in scenario_groups.items()},
               'known_transfer_oracle': {key: {'all_source_cases': summarize_errors([r for r in rows if r['source_m'] is not None]),
                   'true_prior_support': summarize_errors([r for r in rows if r['true_prior_support']]),
                   'wrong_prior_support': summarize_errors([r for r in rows if r['true_prior_support'] is False])}
                   for key, rows in oracle_groups.items()},
               'elapsed_s': time.monotonic() - started, 'real_accuracy_measured': False,
               'held_out_used': False, 'protocol_sha256': sha(output / 'protocol.json'),
               'cases_sha256': sha(output / 'cases.json'), 'limits': protocol['limits']}
    write_json(output / 'summary.json', summary)
    write_json(output / 'progress.json', {'pid': os.getpid(), 'phase': 'complete', 'cases_done': cases,
               'cases_total': cases, 'elapsed_s': time.monotonic() - started, 'updated': time.time()})
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--benchmark', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--cases', type=int, default=384)
    parser.add_argument('--seed', type=int, default=20261108)
    parser.add_argument('--deadline-s', type=float, default=590)
    args = parser.parse_args()
    result = run(args.benchmark, args.output, cases=args.cases, seed=args.seed, deadline_s=args.deadline_s)
    print(json.dumps({'complete': result['complete'], 'cases': result['cases'], 'elapsed_s': result['elapsed_s'],
                      'classification': result['classification']}, indent=2))
