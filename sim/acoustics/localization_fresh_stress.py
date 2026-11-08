"""Locked fresh synthetic SOURCE-LOCALIZATION stress in four new path-model rooms.

This is first-order ray/path SOURCE LOCALIZATION using unknown-source power
features. It is NOT geometry FWI, NOT a replacement for signed one-plane analytic
controls, and NOT evidence against future wave-equation inversion. All truth is
simulated; the finite-patch forward family is shared with the earlier research.
"""
import argparse
from collections import defaultdict
import json
import math
import os
from pathlib import Path
import platform
import shutil
import time

import numpy as np
from scipy import signal
import scipy

from .geometry import Rig
from .localization import errors, spectrum
from .localization_harmonic import DETECTOR
from .localization_model import RoomModel, Surface, build_paths, render_ir
from .localization_pipeline import ResearchMatcher
from .localization_resolution import LOW, describe, dictionary, low_height_grid, sha, write_json


SCOPE = ("First-order ray/path SOURCE LOCALIZATION using unknown-source power features; NOT geometry FWI, "
         "NOT a replacement for signed one-plane analytic controls, and NOT evidence against future wave-equation inversion.")
FAMILIES = ('broadband', 'colored', 'fixed_harmonic', 'variable_harmonic')
CONDITIONS = ('nominal', 'geometry_2cm', 'geometry_10cm', 'material_only', 'speed_only',
              'diffuse_tail', 'directional_mic', 'low_snr')
METHODS = ('coarse_raw4', 'fine_raw4', 'fine_harmonic4', 'fine_harmonic_adaptive4')
RATE, NFFT, BURSTS = 16000, 8192, 4
DEFAULT_SEED = 20261117


def rectangle_surface(ident, a, b, bottom, top, reflection=1.):
    a, b = np.asarray(a), np.asarray(b)
    direction = b - a
    normal = np.array([-direction[1], 0., direction[0]]) / np.linalg.norm(direction)
    vertices = ((a[0], bottom, a[1]), (b[0], bottom, b[1]),
                (b[0], top, b[1]), (a[0], top, a[1]))
    return Surface(ident, vertices[0], tuple(normal), vertices, reflection)


def room_model(spec, *, geometry=None, coefficients=None):
    """Construct a closed rectangle plus explicitly synthetic finite patches.

    Boundary coordinate perturbations keep adjoining wall corners closed.
    Furnished patches are opaque plates, not a calibrated furnished-room solver.
    """
    geometry, coefficients = geometry or {}, coefficients or {}
    xmin = spec['xmin'] + geometry.get('xmin', 0.)
    xmax = spec['xmax'] + geometry.get('xmax', 0.)
    zmin = spec['zmin'] + geometry.get('zmin', 0.)
    zmax = spec['zmax'] + geometry.get('zmax', 0.)
    floor = spec['floor_y'] + geometry.get('floor', 0.)
    ceiling = spec['ceiling_y'] + geometry.get('ceiling', 0.)
    if not (xmax - xmin > 3 and zmax - zmin > 3 and ceiling - floor > 2):
        raise ValueError('invalid synthetic room dimensions')
    def coefficient(ident):
        return float(coefficients.get(ident, 1.))
    surfaces = [Surface('floor', (0., floor, 0.), (0., 1., 0.), reflection=coefficient('floor')),
                Surface('ceiling', (0., ceiling, 0.), (0., -1., 0.), reflection=coefficient('ceiling'))]
    corners = [(xmin, zmin), (xmax, zmin), (xmax, zmax), (xmin, zmax)]
    extents = []
    for index, (a, b) in enumerate(zip(corners, corners[1:] + corners[:1])):
        ident = f'wall_{index}'
        surfaces.append(rectangle_surface(ident, a, b, floor, ceiling, coefficient(ident)))
        extents.append({'surface_id': ident, 'a_xz_m': list(a), 'b_xz_m': list(b)})
    width, depth = xmax - xmin, zmax - zmin
    center = np.array([(xmin + xmax) / 2, (zmin + zmax) / 2])
    for index, panel in enumerate(spec['panels']):
        ident = f'panel_{index}'
        tangent = np.array([math.cos(panel['angle']), math.sin(panel['angle'])])
        normal = np.array([-tangent[1], tangent[0]])
        midpoint = center + np.array([panel['x_fraction'] * width, panel['z_fraction'] * depth])
        midpoint += normal * geometry.get(ident, 0.)
        a, b = midpoint - tangent * panel['width'] / 2, midpoint + tangent * panel['width'] / 2
        surfaces.append(rectangle_surface(ident, a, b, floor, floor + panel['height'], coefficient(ident)))
    table = spec['table']
    tx, tz = center + np.array([table['x_fraction'] * width, table['z_fraction'] * depth])
    y = floor + table['height'] + geometry.get('table', 0.)
    half_x, half_z = table['width'] / 2, table['depth'] / 2
    vertices = ((tx - half_x, y, tz - half_z), (tx + half_x, y, tz - half_z),
                (tx + half_x, y, tz + half_z), (tx - half_x, y, tz + half_z))
    surfaces.append(Surface('table', vertices[0], (0., 1., 0.), vertices, coefficient('table')))
    return RoomModel(tuple(surfaces),
        {'x_min': xmin, 'x_max': xmax, 'z_min': zmin, 'z_max': zmax, 'floor_y': floor, 'ceiling_y': ceiling,
         'synthetic_closed_rectangle': True, 'real_world_footprint_measured': False},
        tuple(extents), {'source': 'seeded synthetic rectangle plus opaque finite patches', 'scope': SCOPE,
                         'real_room_or_acoustic_calibration': False})


def design(seed=DEFAULT_SEED):
    """Recipes, coordinates and all seeds are chosen before any waveform exists."""
    rooms, cases = [], []
    pitch_rng = np.random.default_rng(seed ^ 0x7A91)
    fixed_pitches = pitch_rng.permutation(np.linspace(120., 360., 32)).tolist()
    for room_index in range(4):
        room_seed = int(seed + 1009 * (room_index + 1))
        rng = np.random.default_rng(room_seed)
        width, depth = float(rng.uniform(4.2, 6.3)), float(rng.uniform(4.6, 6.4))
        floor = -float(rng.uniform(.85, 1.15))
        spec = {'room_id': f'fresh-room-{room_index}', 'seed': room_seed,
                'xmin': -width / 2, 'xmax': width / 2, 'zmin': -depth / 2, 'zmax': depth / 2,
                'floor_y': floor, 'ceiling_y': floor + float(rng.uniform(2.45, 2.95)),
                'panels': [{'x_fraction': .15, 'z_fraction': .15, 'width': float(rng.uniform(.8, 1.4)),
                            'height': float(rng.uniform(.8, 1.4)), 'angle': float(rng.uniform(-.4, .4))},
                           {'x_fraction': -.26, 'z_fraction': -.12, 'width': float(rng.uniform(.8, 1.4)),
                            'height': float(rng.uniform(1.0, 1.6)), 'angle': float(rng.uniform(1.2, 1.9))}],
                'table': {'x_fraction': -.18, 'z_fraction': .10, 'width': 1.1, 'depth': .8, 'height': .75}}
        rooms.append(spec)
        nominal = room_model(spec)
        names = [surface.surface_id for surface in nominal.surfaces]
        height_strata = rng.permutation(32)
        for family_index, family in enumerate(FAMILIES):
            for condition_index, condition in enumerate(CONDITIONS):
                index = family_index * len(CONDITIONS) + condition_index
                case_seed = int(seed + (room_index + 1) * 100003 + index * 1009)
                draws = np.random.SeedSequence(case_seed).spawn(12)
                random = np.random.default_rng(int(draws[0].generate_state(1)[0]))
                sigma = .02 if condition == 'geometry_2cm' else .10 if condition == 'geometry_10cm' else 0.
                geometry = {name: float(random.normal(0, sigma)) for name in
                            ('xmin', 'xmax', 'zmin', 'zmax', 'floor', 'ceiling', 'panel_0', 'panel_1', 'table')} if sigma else {}
                coefficients = {name: float(random.uniform(.2, .85)) if condition == 'material_only' else .5 for name in names}
                truth = room_model(spec, geometry=geometry, coefficients=coefficients)
                bounds = truth.declared_bounds_m
                # Each case is a unique source group; no visibility/path filtering.
                for _ in range(1000):
                    x, z = random.uniform(bounds['x_min'] + .3, bounds['x_max'] - .3), random.uniform(bounds['z_min'] + .3, bounds['z_max'] - .3)
                    if math.hypot(x, z) >= .75:
                        break
                else:
                    raise ValueError('could not place a nonsingular synthetic source')
                height = .25 + (height_strata[index] + random.uniform(0, 1)) * (1.35 / 32)
                source = [float(x), float(bounds['floor_y'] + height), float(z)]
                pitch = fixed_pitches[room_index * 8 + condition_index]
                f0 = ([float(pitch)] * BURSTS if family == 'fixed_harmonic' else
                      random.uniform(120., 360., BURSTS).tolist() if family == 'variable_harmonic' else None)
                cases.append({'case_id': f"{spec['room_id']}-source-{index:02d}", 'room_id': spec['room_id'],
                    'source_group': f"{spec['room_id']}-source-{index:02d}", 'source_family': family, 'condition': condition,
                    'source_m': source, 'height_above_truth_floor_m': float(height), 'geometry': geometry,
                    'pressure_coefficients': coefficients, 'sound_speed_m_s': float(random.uniform(339., 347.)) if condition == 'speed_only' else 343.,
                    'mic_directivity': .5 if condition == 'directional_mic' else 0.,
                    'requested_snr_db': 5. if condition == 'low_snr' else 25.,
                    'f0_hz': f0, 'colored_cutoff_hz': float(random.uniform(1800., 3200.)),
                    'tail_seed': int(draws[1].generate_state(1)[0]),
                    'source_seeds': [int(draws[i].generate_state(1)[0]) for i in range(2, 6)],
                    'noise_seeds': [int(draws[i].generate_state(1)[0]) for i in range(6, 10)]})
        for background_index in range(4):
            random = np.random.default_rng(seed + room_index * 101 + background_index + 17)
            cases.append({'case_id': f"{spec['room_id']}-background-{background_index}", 'room_id': spec['room_id'],
                'source_group': f"{spec['room_id']}-background-{background_index}", 'source_family': 'no_source_background',
                'condition': 'background_white' if background_index < 2 else 'background_colored',
                'source_m': None, 'noise_seeds': random.integers(0, 2**32, BURSTS).astype(int).tolist(),
                'background_rms': .001, 'background_cutoff_hz': 2500.})
    return rooms, cases


def freeze(output, *, seed=DEFAULT_SEED, deadline_s=1190):
    output = Path(output).resolve()
    if not 10 <= deadline_s <= 1190:
        raise ValueError('cooperative deadline must be10..1190seconds')
    rooms, cases = design(seed)
    output.mkdir(parents=True, exist_ok=False); (output / 'source').mkdir()
    dependencies = [Path(__file__)] + [Path(__file__).with_name(name) for name in
        ('localization.py', 'localization_model.py', 'localization_resolution.py', 'localization_harmonic.py',
         'localization_adaptive.py', 'localization_pipeline.py', 'geometry.py')]
    for source in dependencies:
        shutil.copyfile(source, output / 'source' / source.name)
    write_json(output / 'rooms.json', rooms); write_json(output / 'cases.json', cases)
    protocol = {'schema': 'openpetcam.fresh-synthetic-source-stress.v1', 'scope': SCOPE,
        'classification': 'FRESH SYNTHETIC PATH-MODEL SOURCE-LOCALIZATION STRESS ONLY; NEVER REAL ACCURACY',
        'created_epoch': time.time(), 'seed': seed, 'rooms': 4, 'emitting_source_cases': 128, 'background_cases': 16,
        'source_groups': 128, 'room_clusters': 4, 'source_bursts_per_group': BURSTS,
        'source_groups_policy': 'Each emitting case is a unique fixed continuous source point with four repeated bursts; room clusters must remain explicit.',
        'planned_reporting_strata': 'Room, sourcefamily, condition and trueharmonic-F0 prior support; labels used only after matching, never as algorithm inputs.',
        'comparison_units': 'Pipeline comparisons are paired on identical fourburst observations. Condition contrasts are descriptive because each family/condition combination uses a different unique source point.',
        'source_families': FAMILIES, 'conditions': CONDITIONS, 'methods': METHODS,
        'sample_rate_hz': RATE, 'nfft': NFFT, 'band': LOW, 'harmonic_detector': DETECTOR,
        'candidate_grids_m': [.5, .125], 'candidate_heights_above_prior_floor_m': [.45, 1.2],
        'dictionary_base_pressure_reflection': 1., 'profile_pressure_variants': [.25, .5, .75],
        'nominal_truth_pressure_reflection': .5,
        'source_height_above_truth_floor_m': [.25, 1.6], 'true_harmonic_f0_range_hz': [120., 360.],
        'source_recipe': 'Independent Gaussian/colored or11harmonic synthetic decaying bursts;0.28s,4ms attack,90ms decay,noisecomponent0.08forharmonics; not validated dogbarks.',
        'geometry_recipe': 'New seeded closed rectangle plus finite opaque patches. Geometry controls perturb coherent room boundary coordinates and panel normal positions, preserving closed walls.',
        'candidate_placement': 'Global prior endpoint grids, two fixed prior heights; adaptive points from observed scores only. True source coordinates/heights/F0 never enter matching.',
        'source_policies': 'Truth/recipe coordinates are chosen before waveforms. Source positions are inside the simulated truth rectangle, not selected for visibility/path count. All cases remain in metrics.',
        'noise_policy': 'Source cases have relative25dB or5dB noise; zero modeled signal has zero relative noise and undefined realizedSNR. Separate white/colored no-emitter controls haveRMS.001.',
        'tail_policy': 'Fixed signed random exponentially decaying tail,35%IRnorm, begins35ms with50ms decay, same within case; not a validated higher-order solver.',
        'forward_model': 'Signed fractional IR and finite direct+first-order complex path physics; same family as development, with synthetic tail/direction/noise controls.',
        'selection': 'No fitting, threshold tuning or method selection; all four fixed methods evaluated once on all frozen cases and then locked.',
        'acceptance': 'Alwaysfalse; no real or synthetic confidence calibration is carried over.',
        'source_sha256': {source.name: sha(source) for source in dependencies},
        'rooms_sha256': sha(output / 'rooms.json'), 'cases_sha256': sha(output / 'cases.json'),
        'deadline_s': deadline_s, 'frozen_before_any_fresh_waveform': True,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'scipy': scipy.__version__},
        'limits': ['First-order ray/path source localization only, not geometryFWI or a wave-equation inversion feasibility test.',
            'New synthetic room/source draws are fresh, but generatorphysics is shared; no independently measured real-room accuracy or calibration.',
            'Synthetic rectangles and plates do not establish any real household footprint, material or sound-source prior.',
            'Only four room clusters; source groups/fourbursts are not independent room replications. No unclustered confidence intervals.',
            'Continuous source heights and120..360Hz trueharmonics intentionally exceed the twoheight and140..280Hz detector priors.',
            'Background has no physical source/error truth; missing source predictions remain NULL and count unsuccessful.',
            'No v1held-out labels/predictions, summary or selected-rule files are opened. Existing outputs stay frozen.',
            'No camera operation, upload, paid service, remote heavy job, deployment, commit orpush.']}
    write_json(output / 'protocol.json', protocol)
    write_json(output / 'frozen.json', {'protocol_sha256': sha(output / 'protocol.json'),
        'rooms_sha256': sha(output / 'rooms.json'), 'cases_sha256': sha(output / 'cases.json'),
        'before_any_fresh_waveform': True})
    return protocol


def emission(case, burst_index):
    rng = np.random.default_rng(case['source_seeds'][burst_index])
    t = np.arange(int(RATE * .28)) / RATE
    raw = rng.normal(size=len(t))
    if case['source_family'] == 'colored':
        raw = signal.sosfilt(signal.butter(3, case['colored_cutoff_hz'], fs=RATE, output='sos'), raw)
    elif case['source_family'] in ('fixed_harmonic', 'variable_harmonic'):
        f0 = case['f0_hz'][burst_index]
        raw = .08 * raw + sum(np.sin(2 * np.pi * f0 * order * t + rng.uniform(-np.pi, np.pi)) / order for order in range(1, 12))
    return raw * (1 - np.exp(-t / .004)) * np.exp(-t / .09)


def observations(prior_spec, case, receiver):
    """Generate one case once; all methods later receive these same powers."""
    count = int(RATE * .28) + math.ceil(RATE * .25) - 1
    recordings, sources, h = [], [], None
    if case['source_m'] is not None:
        model = room_model(prior_spec, geometry=case['geometry'], coefficients=case['pressure_coefficients'])
        paths = build_paths(model, case['source_m'], receiver.receiver_m, receiver_axis=receiver.receiver_axis,
                            mic_directivity=case['mic_directivity'], sound_speed=case['sound_speed_m_s'], timing='direct_relative')
        h = render_ir(paths, RATE, .25, .01)
        if case['condition'] == 'diffuse_tail':
            rng = np.random.default_rng(case['tail_seed']);t = np.arange(len(h)) / RATE
            tail = rng.normal(size=len(h)) * np.exp(-np.maximum(t - .035, 0) / .05) * (t >= .035)
            tail *= .35 * np.linalg.norm(h) / max(np.linalg.norm(tail), 1e-20);h += tail
        diagnostic = {'direct_visible': paths.direct_visible, 'paths': len(paths), 'ir_energy': float(h @ h),
                      'realized_snr_db': case['requested_snr_db'] if h @ h > 0 else None}
    else:
        diagnostic = {'direct_visible': None, 'paths': None, 'ir_energy': None, 'realized_snr_db': None}
    for burst_index, seed in enumerate(case['noise_seeds']):
        rng = np.random.default_rng(seed);noise = rng.normal(size=count)
        if case['source_m'] is None:
            if case['condition'] == 'background_colored':
                noise = signal.sosfilt(signal.butter(3, case['background_cutoff_hz'], fs=RATE, output='sos'), noise)
            noise *= case['background_rms'] / np.sqrt(np.mean(noise**2));recording = noise
        else:
            source = emission(case, burst_index);sources.append(source)
            recording = signal.fftconvolve(source, h)
            noise *= np.linalg.norm(recording) / max(np.linalg.norm(noise), 1e-20) * 10**(-case['requested_snr_db'] / 20)
            recording += noise
        recordings.append(recording)
    powers = np.asarray([spectrum(recording, RATE, nfft=NFFT)[1] for recording in recordings])
    archive = {'recording': np.asarray(recordings), 'burst_power': powers}
    if h is not None:
        archive.update(impulse_response=h, emission=np.asarray(sources))
    return powers, archive, diagnostic


def match_methods(coarse, fine, powers, progress):
    return {'coarse_raw4': coarse.match(powers, harmonic=False, refine=False, progress=progress),
            'fine_raw4': fine.match(powers, harmonic=False, refine=False, progress=progress),
            'fine_harmonic4': fine.match(powers, harmonic=True, refine=False, progress=progress),
            'fine_harmonic_adaptive4': fine.match(powers, harmonic=True, refine=True, progress=progress)}


def result_row(case, method, match, receiver):
    rank = match['final']
    error = {'distance_error_m': None, 'azimuth_error_deg': None, 'horizontal_error_m': None,
             'height_error_m': None, 'range_error_m': None}
    if rank['position_m'] is not None and case['source_m'] is not None:
        predicted, truth = np.asarray(rank['position_m']), np.asarray(case['source_m'])
        error.update(errors(predicted, truth, receiver), horizontal_error_m=float(np.linalg.norm((predicted - truth)[[0, 2]])),
                     height_error_m=float(abs(predicted[1] - truth[1])),
                     range_error_m=float(abs(np.linalg.norm(predicted - receiver) - np.linalg.norm(truth - receiver))))
    pitch = case.get('f0_hz')
    pitch_support = (float(np.mean([(DETECTOR['f0_prior_hz'][0] <= value <= DETECTOR['f0_prior_hz'][1])
                                   for value in pitch])) if pitch is not None else None)
    return {'case_id': case['case_id'], 'room_id': case['room_id'], 'source_group': case['source_group'],
        'source_family': case['source_family'], 'condition': case['condition'], 'source_m': case['source_m'],
        'true_f0_in_declared_prior_fraction': pitch_support,
        'method': method, **rank, **error, 'gated_bursts': match['gated_bursts'],
        'new_candidate_count': len(match['new_candidates_m']), 'new_candidates_m': match['new_candidates_m'],
        'global_center_indexes': match['global_center_indexes'], 'mean_masked_fraction': match['mean_masked_fraction']}


def population(rows):
    source = [row for row in rows if row['source_m'] is not None]
    background = [row for row in rows if row['source_m'] is None]
    scores = [row['similarity'] for row in background if row.get('similarity') is not None]
    return {'source_cases': len(source), 'all_source_cases': describe(source) if source else {'cases': 0},
            'background_cases': len(background), 'background_conditional_candidates': sum(row['best_index'] is not None for row in background),
            'background_median_similarity': float(np.median(scores)) if scores else None,
            'cases_with_gated_burst': sum(row['gated_bursts'] > 0 for row in rows)}


def run_frozen(output):
    output = Path(output).resolve()
    protocol, frozen = [json.loads((output / name).read_text()) for name in ('protocol.json', 'frozen.json')]
    for name in ('protocol', 'rooms', 'cases'):
        if sha(output / f'{name}.json') != frozen[f'{name}_sha256']:
            raise ValueError('fresh plan no longer matches frozen hashes')
    for name, expected in protocol['source_sha256'].items():
        if sha(Path(__file__).with_name(name)) != expected or sha(output / 'source' / name) != expected:
            raise ValueError('pipeline/source changed after freeze; do not run this preregistration')
    # Exclusive start marker forbids re-evaluating or resuming the fresh cases.
    write_json(output / 'STARTED.json', {'pid': os.getpid(), 'epoch': time.time(), 'evaluation_policy': 'once; never resume/re-evaluate this frozen stress cohort'})
    rooms, cases = [json.loads((output / name).read_text()) for name in ('rooms.json', 'cases.json')]
    (output / 'observations').mkdir();(output / 'dictionaries').mkdir()
    started = time.monotonic()
    def progress(phase=None, done=None, total=None):
        value = {'pid': os.getpid(), 'phase': phase, 'done': done, 'total': total,
                 'elapsed_s': time.monotonic() - started, 'updated_epoch': time.time()}
        temporary = output / '.progress.tmp';temporary.write_text(json.dumps(value, indent=2));temporary.replace(output / 'progress.json')
        if value['elapsed_s'] > protocol['deadline_s']:
            raise TimeoutError('fresh stress deadline; all partial artifacts preserved, no automatic rerun')
    receiver = Rig().pose(0, 0);frequency = np.fft.rfftfreq(NFFT, 1 / RATE)
    rows, inventory = [], []
    with (output / 'rankings.jsonl').open('x') as stream:
        for spec in rooms:
            model = room_model(spec)
            if any(surface.reflection != 1. for surface in model.surfaces):
                raise ValueError('base dictionary model must use unit pressure coefficients before profiling')
            matchers = {}
            for spacing in (.5, .125):
                lower = low_height_grid(model, spacing, height_above_floor=.45)
                upper = low_height_grid(model, spacing, height_above_floor=1.2)
                positions = np.empty((len(lower) * 2, 3));positions[::2], positions[1::2] = lower, upper
                features, visible, path_counts = dictionary(model, positions, receiver.receiver_m, frequency,
                    lambda phase, done, total: progress(f"{spec['room_id']}_{spacing:g}_{phase}", done, total))
                dictionary_file = output / 'dictionaries' / f"{spec['room_id']}_{spacing:g}m.npz"
                np.savez_compressed(dictionary_file, positions_m=positions, features=features,
                                    direct_visible=visible, path_counts=path_counts)
                inventory.append({'room_id': spec['room_id'], 'spacing_m': spacing, 'candidates': len(positions),
                                  'file': str(dictionary_file.relative_to(output)), 'sha256': sha(dictionary_file)})
                matchers[spacing] = ResearchMatcher(model, positions, features, receiver.receiver_m, frequency)
            room_cases = [case for case in cases if case['room_id'] == spec['room_id']]
            for index, case in enumerate(room_cases):
                powers, observation, diagnostic = observations(spec, case, receiver)
                file = output / 'observations' / f"{case['case_id']}.npz"
                np.savez_compressed(file, **observation)
                inventory.append({'case_id': case['case_id'], 'file': str(file.relative_to(output)),
                                  'sha256': sha(file), 'diagnostic': diagnostic})
                matcher_progress = lambda *args: progress(f"{spec['room_id']}_{case['case_id']}_matching", index + 1, len(room_cases))
                for method, match in match_methods(matchers[.5], matchers[.125], powers, matcher_progress).items():
                    row = result_row(case, method, match, receiver.receiver_m)
                    row['observation_sha256'] = inventory[-1]['sha256'];row['diagnostic'] = diagnostic
                    rows.append(row);stream.write(json.dumps(row, allow_nan=False) + '\n');stream.flush()
                progress(f"{spec['room_id']}_cases", index + 1, len(room_cases))
            del matchers
    groups, room_groups, family_groups, condition_groups, pitch_groups = [defaultdict(list) for _ in range(5)]
    for row in rows:
        groups[row['method']].append(row);room_groups[row['room_id'] + '|' + row['method']].append(row)
        family_groups[row['source_family'] + '|' + row['method']].append(row)
        condition_groups[row['condition'] + '|' + row['method']].append(row)
        if row['true_f0_in_declared_prior_fraction'] is not None:
            support = 'all_true_pitches_inside_prior' if row['true_f0_in_declared_prior_fraction'] == 1 else 'some_true_pitch_outside_prior'
            pitch_groups[row['source_family'] + '|' + support + '|' + row['method']].append(row)
    write_json(output / 'inventory.json', inventory)
    summary = {'complete': True, 'scope': SCOPE, 'classification': protocol['classification'],
        'emitting_source_cases': 128, 'background_cases': 16, 'room_clusters': 4, 'source_groups': 128,
        'comparison': {key: population(value) for key, value in groups.items()},
        'by_room': {key: population(value) for key, value in room_groups.items()},
        'by_source_family': {key: population(value) for key, value in family_groups.items()},
        'by_condition': {key: population(value) for key, value in condition_groups.items()},
        'by_true_pitch_prior_support': {key: population(value) for key, value in pitch_groups.items()},
        'elapsed_s': time.monotonic() - started, 'protocol_sha256': sha(output / 'protocol.json'),
        'real_accuracy_measured': False, 'evaluated_once': True, 'v1_held_out_used': False, 'limits': protocol['limits']}
    write_json(output / 'summary.json', summary)
    write_json(output / 'LOCKED.json', {'protocol_sha256': sha(output / 'protocol.json'), 'cases_sha256': sha(output / 'cases.json'),
        'rooms_sha256': sha(output / 'rooms.json'), 'rankings_sha256': sha(output / 'rankings.jsonl'),
        'summary_sha256': sha(output / 'summary.json'), 'inventory_sha256': sha(output / 'inventory.json'),
        'policy': 'Fresh stress evaluated once and locked. Never tune/re-evaluate this cohort.'})
    progress('complete', len(rows), len(rows));return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--deadline-s', type=float, default=1190)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--freeze-only', action='store_true');mode.add_argument('--run-frozen', action='store_true')
    args = parser.parse_args()
    if args.freeze_only:
        result = freeze(args.output, seed=args.seed, deadline_s=args.deadline_s)
        print(json.dumps({'frozen': True, 'scope': result['scope'], 'emitting_source_cases': result['emitting_source_cases'], 'background_cases': result['background_cases']}, indent=2))
    else:
        result = run_frozen(args.output)
        print(json.dumps({key: result[key] for key in ('complete', 'classification', 'elapsed_s')}, indent=2))
