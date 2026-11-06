"""Fresh low-frequency full-wave source-localization information ladder.

No camera, playback, network, installed-model writes or geometry inversion.
All cases and inputs freeze before propagation. Original room recordings are
prescribed digital emitters here, not dry sources or real location labels.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import scipy
from scipy import signal

from .geometry import Rig
from .localization import errors, rank_candidates, ripple_features
from .localization_model import build_paths, load_room, render_transfer
from .phase_localization import (arbitrary_source_control_spectra, blind_fir_spectra,
                                 known_source_spectra)
from .wave3d import simulate
from .wave_room import air_point, build_room, candidates


ROOT = Path(__file__).resolve().parents[2]
FREQUENCIES = np.arange(80., 222., 2.)
ONSETS = np.linspace(-.02, .02, 81)
METHODS = ('ray_power', 'wave_power', 'wave_known_source', 'wave_other_recording_template',
           'wave_blind32', 'arbitrary_source_control')
SEED = 2026112701
IR_SECONDS = 1.


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def source_spectra(inventory_path):
    inventory_path = Path(inventory_path)
    inventory = json.loads(inventory_path.read_text())
    archive = Path(inventory['window_archive']['path'])
    if sha(archive) != inventory['window_archive']['sha256']:
        raise ValueError('frozen source archive hash differs')
    spectra, metadata = [], []
    with np.load(archive, allow_pickle=False) as arrays:
        for row in inventory['entries']:
            original = arrays[row['npz_array']]
            if hashlib.sha256(original.tobytes()).hexdigest() != row['window_sha256']:
                raise ValueError('native source samples differ')
            rate = row['sample_rate_hz']
            t = np.arange(len(original)) / rate
            drive = original.astype(float) * np.hanning(len(original))
            # Direct physical-frequency integration preserves signed phase;
            # original samples and scale remain immutable in their archive.
            spectrum = np.exp(-2j * np.pi * FREQUENCIES[:, None] * t) @ drive / rate
            spectra.append(spectrum)
            metadata.append({'id': row['id'], 'sample_rate_hz': rate, 'samples': len(original),
                             'original_window_sha256': row['window_sha256'],
                             'derived_drive_sha256': hashlib.sha256(drive.tobytes()).hexdigest(),
                             'source_kind': row['source_kind'], 'category_hint': row['category'],
                             'overlap_group': row['overlap_group'],
                             'recommended_split_group': row['recommended_split_group'],
                             'actual_sound_category': None, 'physical_source_position_m': None,
                             'source_filter': 'fixed Hann gate; complex DTFT on 80..220Hz; no per-recording normalization'})
    return np.asarray(spectra), metadata, inventory


def scene_definitions(seed=SEED):
    return {'nominal': {'spacing_m': .125, 'padding_m': .8, 'sound_speed': 343., 'damping': 3.},
            'fine_same_prior': {'spacing_m': .1, 'padding_m': .8, 'sound_speed': 343., 'damping': 3.},
            'fine_2cm': {'spacing_m': .1, 'padding_m': .8, 'sound_speed': 343., 'damping': 3.,
                         'wall_shift_m': .02, 'floor_shift_m': .02, 'ceiling_shift_m': -.02, 'seed': seed + 2},
            'fine_10cm_completion': {'spacing_m': .1, 'padding_m': 1., 'sound_speed': 338., 'damping': 4.,
                                    'wall_shift_m': .1, 'floor_shift_m': .1, 'ceiling_shift_m': -.1, 'seed': seed + 3}}


def freeze_cases(rooms, positions, n_anchors, seed=SEED):
    rng = np.random.default_rng(seed)
    bounds = rooms['nominal'].metadata['declared_segment_bounds_xz_m']
    floor = rooms['nominal'].metadata['requested_floor_y_m']
    anchors = []
    for _ in range(100000):
        p = [rng.uniform(*bounds[0]), floor + rng.uniform(.3, 1.35), rng.uniform(*bounds[1])]
        if np.linalg.norm(np.asarray(p)[[0, 2]]) < .8:
            continue
        if all(air_point(room.grid, p) for room in rooms.values()):
            anchors.append(p)
        if len(anchors) == n_anchors:
            break
    if len(anchors) != n_anchors:
        raise ValueError('could not freeze enough common-air source points')
    rows = []
    nuisance = {(index, source): {'onset_s': float(rng.uniform(-.012, .012)),
                                  'snr_db': 10. if index % 4 == 0 else 30.,
                                  'gain': float(rng.uniform(.3, 2.)),
                                  'noise_seed': int(rng.integers(2**31))}
                for index in range(len(anchors)) for source in range(4)}
    for scene in ('coarse_offgrid_control', 'fine_same_prior', 'fine_2cm', 'fine_10cm_completion'):
        for index, p in enumerate(anchors):
            for source in range(4):
                rows.append({'id': f'{scene}-{index:03d}-{source}', 'scene': scene,
                             'kind': 'single', 'anchor': index, 'source_index': source,
                             'true_position_m': p, 'probe_index': index,
                             **nuisance[index, source]})
    selected = rng.choice(len(positions), min(8, len(positions)), replace=False)
    for index in selected:
        for source in range(4):
            rows.append({'id': f'same_grid-{index}-{source}', 'scene': 'same_grid_control',
                         'kind': 'single', 'anchor': int(index), 'source_index': source,
                         'true_position_m': positions[index].tolist(), 'probe_index': int(index),
                         'onset_s': float(ONSETS[int(rng.integers(len(ONSETS)))]),
                         'snr_db': None, 'gain': float(rng.uniform(.3, 2.)), 'noise_seed': int(rng.integers(2**31))})
    for kind in ('silence', 'background', 'multiple'):
        for i in range(8):
            rows.append({'id': f'{kind}-{i}', 'scene': 'fine_same_prior', 'kind': kind,
                         'anchor': i % len(anchors), 'source_index': i % 4,
                         'true_position_m': None, 'probe_index': i % len(anchors),
                         'second_probe_index': (i + 1) % len(anchors),
                         'onset_s': 0., 'snr_db': None, 'gain': 1.,
                         'noise_seed': int(rng.integers(2**31))})
    return anchors, rows


def propagate(room, microphone, probes, output, cache):
    n = int(round(IR_SECONDS * room.grid.sample_rate_hz))
    drive = np.zeros(n); drive[0] = 1.
    result = simulate(room.grid, microphone, drive, probes, cache_dir=cache, timeout_s=180)
    np.savez_compressed(output, signals=result.signals, dt_s=result.dt_s, positions_m=probes,
                        microphone_m=microphone, impulse_drive=drive)
    bins = np.rint(FREQUENCIES * n / result.sample_rate_hz).astype(int)
    actual = bins * result.sample_rate_hz / n
    if not np.allclose(actual, FREQUENCIES, rtol=0, atol=1e-10):
        raise ValueError('native Green FFT does not share declared physical frequencies')
    transfer = np.fft.rfft(result.signals, axis=1)[:, bins]
    return result, transfer


def observed(case, transfers, sources, candidate_count):
    rng = np.random.default_rng(case['noise_seed'])
    reference = sources[case['source_index']]
    if case['kind'] == 'silence':
        return np.zeros(len(FREQUENCIES), complex)
    if case['kind'] == 'background':
        # Absolute no-emitter control, not source-scaled SNR noise.
        return (rng.normal(size=len(FREQUENCIES)) + 1j * rng.normal(size=len(FREQUENCIES))) * 1e-9
    if case['scene'] == 'same_grid_control':
        g = transfers['nominal'][case['probe_index']]
    elif case['scene'] == 'coarse_offgrid_control':
        g = transfers['nominal'][candidate_count + case['probe_index']]
    else:
        g = transfers[case['scene']][case['probe_index']]
    y = case['gain'] * g * reference * np.exp(-2j * np.pi * FREQUENCIES * case['onset_s'])
    if case['kind'] == 'multiple':
        y += .8 * transfers[case['scene']][case['second_probe_index']] * sources[(case['source_index'] + 1) % 4]
    if case['snr_db'] is not None and np.any(y):
        noise = rng.normal(size=len(y)) + 1j * rng.normal(size=len(y))
        noise *= np.linalg.norm(y) / np.linalg.norm(noise) * 10**(-case['snr_db'] / 20)
        y += noise
    return y


def power_match(y, dictionary, positions):
    features, _ = ripple_features(abs(dictionary)**2, FREQUENCIES, band=(80, 220), smooth_hz=50, denoise_hz=3)
    feature, _ = ripple_features(abs(y)**2, FREQUENCIES, band=(80, 220), smooth_hz=50, denoise_hz=3)
    result = rank_candidates(feature, features, positions)
    if result.get('score_margin') is not None and result['score_margin'] <= 1e-9:
        result.update(best_index=None, position_m=None, status='ambiguous')
    # This rule has no sound detector or real calibration; all branches stay off.
    result['accepted'] = False
    return result


def infer(method, y, source, template, green, ray, positions):
    if method in ('ray_power', 'wave_power'):
        return power_match(y, ray if method == 'ray_power' else green, positions)
    if method in ('wave_known_source', 'wave_other_recording_template'):
        return known_source_spectra(y, source if method == 'wave_known_source' else template,
                                    green, FREQUENCIES, onsets_s=ONSETS)
    if method == 'wave_blind32':
        return blind_fir_spectra(y, green, FREQUENCIES, 1000., source_length=32,
                                 ridge=.001, onsets_s=(-.02, 0., .02))
    return arbitrary_source_control_spectra(y, green, FREQUENCIES)


def summarize(rows):
    single = [r for r in rows if r['kind'] == 'single']
    located = [r for r in single if r['distance_error_m'] is not None]
    nulls = [r for r in rows if r['kind'] in ('silence', 'background')]
    multiple = [r for r in rows if r['kind'] == 'multiple']
    return {'total_cases': len(rows), 'single_cases': len(single), 'localized_cases': len(located),
            'within_1m': sum(r['distance_error_m'] <= 1 for r in located),
            'within_30deg': sum(r['bearing_error_deg'] <= 30 for r in located),
            'within_1m_fraction': sum(r['distance_error_m'] <= 1 for r in located) / len(single) if single else None,
            'within_30deg_fraction': sum(r['bearing_error_deg'] <= 30 for r in located) / len(single) if single else None,
            'unlocalized': len(single) - len(located),
            'median_error_m_when_localized': float(np.median([r['distance_error_m'] for r in located])) if located else None,
            'null_cases': len(nulls), 'null_conditional_candidates': sum(r['position_m'] is not None for r in nulls),
            'multiple_cases': len(multiple), 'multiple_conditional_candidates': sum(r['position_m'] is not None for r in multiple),
            'real_accepted': sum(r['accepted'] for r in rows),
            'quantile_population': 'localized single-source simulations only; all misses stay in success denominator'}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--room', type=Path, required=True)
    parser.add_argument('--sources', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--anchors', type=int, default=16)
    parser.add_argument('--spacing', type=float, default=.75)
    parser.add_argument('--seed', type=int, default=SEED)
    args = parser.parse_args(argv)
    if not 2 <= args.anchors <= 64 or not .5 <= args.spacing <= 1.5 or args.seed < 0:
        parser.error('anchors 2..64 and candidate spacing .5..1.5 required')
    args.output.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    room_digest, inventory_digest = sha(args.room), sha(args.sources)
    frozen_room, frozen_inventory = args.output / 'frozen_room.json', args.output / 'frozen_sources.json'
    frozen_room.write_bytes(args.room.read_bytes()); frozen_inventory.write_bytes(args.sources.read_bytes())
    if sha(frozen_room) != room_digest or sha(frozen_inventory) != inventory_digest:
        raise ValueError('input changed during snapshot')
    definitions = scene_definitions(args.seed)
    rooms = {name: build_room(frozen_room, **parameters) for name, parameters in definitions.items()}
    positions = candidates(rooms['nominal'], horizontal_spacing=args.spacing)
    if len(positions) < 2:
        raise ValueError('insufficient computational candidate positions')
    mic = np.asarray(Rig().pose(0, 0).receiver_m)
    microphones = {'nominal': mic, 'fine_same_prior': mic, 'fine_2cm': mic + [.01, 0, -.01],
                   'fine_10cm_completion': mic + [.02, 0, -.02]}
    if not all(air_point(room.grid, microphones[name]) for name, room in rooms.items()):
        raise ValueError('assumed microphone touches a rigid voxel')
    sources, source_meta, inventory = source_spectra(frozen_inventory)
    if len(sources) != 4:
        raise ValueError('four frozen virtual emitters required')
    if any(source_meta[i]['recommended_split_group'] == source_meta[i ^ 1]['recommended_split_group'] for i in range(4)):
        raise ValueError('other-recording templates must use distinct source/day groups')
    anchors, cases = freeze_cases(rooms, positions, args.anchors, args.seed)
    files = ('wave3d.py', 'wave3d_kernel.cpp', 'wave_room.py', 'phase_localization.py', 'wave_experiment.py',
             'localization.py', 'localization_model.py', 'geometry.py')
    protocol = {'seed': args.seed, 'input_sha256': {'sim/acoustics/' + p: sha(ROOT / 'sim/acoustics' / p) for p in files},
                'room_sha256': room_digest, 'source_inventory_sha256': inventory_digest,
                'source_archive_sha256': inventory['window_archive']['sha256'],
                'frequencies_hz': FREQUENCIES.tolist(), 'ir_seconds': IR_SECONDS,
                'onsets_s': ONSETS.tolist(), 'methods': METHODS, 'scene_parameters': definitions,
                'source_waveforms': source_meta, 'candidates': positions.tolist(),
                'versions': {'python': sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__},
                'cases_frozen_before_propagation': True,
                'same_grid_kind': 'algorithm check with identical discrete Green dictionary; not independent accuracy',
                'assumptions': ['80..220Hz low-frequency scalar3D propagation; not the active2.1..7.4kHz sweep band.',
                                'Artificial rigid rectangular closure and full-height extrusions of disconnected prior segments.',
                                'Unknown floor/ceiling, voxelization and volumetric loss; no calibrated wall impedance.',
                                'Known-source fitting receives privileged exact digital emitter waveform.',
                                'Four archived room recordings are prescribed replay proxies, already reverberant and unlabelled.',
                                'Other-recording templates and32ms causal FIR impose source assumptions; arbitrary sources remain unidentifiable.',
                                'All receiver positions are assumed simulator coordinates, not historical camera poses.',
                                'Repeated source/anchor/scene rows are correlated; no independent-case confidence intervals.',
                                'Off-grid scene comparisons share gain/onset/noise draws; independent on-grid control has different anchors and exactly gridded onsets.',
                                'Off-grid emission onsets lie between fitting steps; timing quantization adds at most0.25ms nuisance mismatch.',
                                'Ray baseline omits artificial closing walls as well as higher-order/diffraction physics; comparison is not isolated solver error.',
                                'Observations are complex physical-frequency samples of finite1s Green convolutions, not independently generated time recordings.',
                                'No geometry gradient/adjoint or geometry FWI, microphone housing, directionality or real-room validation.',
                                'All comparisons remain conditional; no real sound coordinates or camera actions are accepted.']}
    write(args.output / 'protocol.json', protocol)
    for source_path, expected_hash in protocol['input_sha256'].items():
        destination = args.output / 'source' / source_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = (ROOT / source_path).read_bytes()
        if hashlib.sha256(data).hexdigest() != expected_hash:
            raise ValueError('source changed during freeze')
        destination.write_bytes(data)
    write(args.output / 'cases.json', cases)
    write(args.output / 'rooms.json', {name: room.metadata for name, room in rooms.items()})
    write(args.output / 'source_inventory.json', inventory)
    np.savez_compressed(args.output / 'source_spectra.npz', frequency_hz=FREQUENCIES, spectra=sources)
    # A horizontal slice of a genuinely3D field, not a2D wave approximation.
    grid = rooms['nominal'].grid
    xs = grid.origin_m[0] + np.arange(1, grid.shape_xyz[0] - 1, 4) * grid.spacing_m
    zs = grid.origin_m[2] + np.arange(1, grid.shape_xyz[2] - 1, 4) * grid.spacing_m
    slice_points, slice_indices = [], []
    for j, z in enumerate(zs):
        for i, x in enumerate(xs):
            point = [x, float(mic[1]), z]
            if air_point(grid, point):
                slice_points.append(point); slice_indices.append((j, i))
    transfers, engine_meta = {}, {}
    nominal_result = None
    for name, room in rooms.items():
        probes = np.concatenate((positions, np.asarray(anchors), np.asarray(slice_points))) if name == 'nominal' else np.asarray(anchors)
        result, transfer = propagate(room, microphones[name], probes, args.output / (name + '-greens.npz'), args.output / 'native-cache')
        transfers[name], engine_meta[name] = transfer, result.metadata
        if name == 'nominal':
            nominal_result = result
        print(json.dumps({'stage': 'propagation', 'scene': name, 'shape': room.grid.shape_xyz,
                          'probes': len(probes), 'wall_s': result.metadata['run_wall_s']}), flush=True)
    write(args.output / 'engine.json', engine_meta)
    green = transfers['nominal'][:len(positions)]
    ray_room = load_room(frozen_room)
    ray = np.asarray([render_transfer(build_paths(ray_room, p, mic), FREQUENCIES) for p in positions])
    np.savez_compressed(args.output / 'dictionary.npz', positions_m=positions, full_wave=green, first_order=ray,
                        frequency_hz=FREQUENCIES, assumed_microphone_m=mic)
    # Gaussian-windowed120Hz packet, convolved causally with the signed impulse
    # field under a common colour scale. Arbitrary wave-equation units, not Pa.
    fs = nominal_result.sample_rate_hz
    pulse_t = np.arange(int(.15 * fs)) / fs
    pulse = np.sin(2 * np.pi * 120 * (pulse_t - .06)) * np.exp(-.5 * ((pulse_t - .06) / .012)**2)
    field = signal.fftconvolve(nominal_result.signals[len(positions) + len(anchors):], pulse[None, :], axes=1)
    requested_times = np.arange(0, .301, .001)
    frame_indices = np.rint(requested_times * fs).astype(int)
    times = frame_indices / fs
    frames = np.zeros((len(times), len(zs), len(xs)))
    solid = np.ones((len(zs), len(xs)), dtype=bool)
    for k, (j, i) in enumerate(slice_indices):
        frames[:, j, i] = field[k, frame_indices]
        solid[j, i] = False
    np.savez_compressed(args.output / 'wavefield.npz', times_s=times, x_m=xs, z_m=zs,
                        pressure=frames, solid=solid, source_xz_m=mic[[0, 2]], slice_y_m=mic[1])
    rows, observations, profiles = [], [], []
    for ci, case in enumerate(cases):
        y = observed(case, transfers, sources, len(positions)); observations.append(y)
        si = case['source_index']
        template_index = si ^ 1  # same category hint, distinct frozen source/day group
        for method in METHODS:
            result = infer(method, y, sources[si], sources[template_index], green, ray, positions)
            index = result['best_index']
            coordinate = positions[index].tolist() if index is not None else None
            err = errors(coordinate, case['true_position_m'], mic) if coordinate is not None and case['true_position_m'] is not None else None
            truth_mic = microphones.get(case['scene'], mic)
            if err:
                p, t = np.asarray(coordinate) - mic, np.asarray(case['true_position_m']) - truth_mic
                delta = np.arctan2(p[0], p[2]) - np.arctan2(t[0], t[2])
                err['azimuth_error_deg'] = float(abs(np.degrees(np.arctan2(np.sin(delta), np.cos(delta)))))
            row = {'case_id': case['id'], 'scene': case['scene'], 'kind': case['kind'], 'method': method,
                   'source_id': source_meta[si]['id'], 'source_kind': source_meta[si]['source_kind'],
                   'true_position_m': case['true_position_m'], 'position_m': coordinate,
                   'assumed_receiver_m': mic.tolist(), 'simulated_receiver_m': truth_mic.tolist(),
                   'distance_error_m': err['distance_error_m'] if err else None,
                   'bearing_error_deg': err['azimuth_error_deg'] if err else None,
                   'status': result['status'], 'normalized_residual': result.get('normalized_residual'),
                   'source_onset_s': result.get('best_onset_s'), 'accepted': False}
            rows.append(row)
            scores = result.get('profiles', result.get('scores'))
            profiles.append({'case_id': case['id'], 'method': method, 'candidate_profiles': scores})
        if ci % 16 == 0:
            print(json.dumps({'stage': 'matching', 'cases_done': ci + 1, 'total': len(cases)}), flush=True)
    write(args.output / 'results.json', rows)
    write(args.output / 'profiles.json', profiles)
    np.savez_compressed(args.output / 'observations.npz', frequency_hz=FREQUENCIES, spectra=np.asarray(observations))
    by_scene = defaultdict(dict)
    for scene in sorted({c['scene'] for c in cases}):
        for method in METHODS:
            by_scene[scene][method] = summarize([r for r in rows if r['scene'] == scene and r['method'] == method])
    unchanged = (all(sha(ROOT / path) == digest for path, digest in protocol['input_sha256'].items())
                 and sha(args.room) == room_digest and sha(args.sources) == inventory_digest
                 and sha(inventory['window_archive']['path']) == protocol['source_archive_sha256'])
    summary = {'by_scene': dict(by_scene), 'methods': METHODS, 'cases': len(cases), 'candidate_count': len(positions),
               'source_count': len(sources), 'real_acceptance': False, 'input_hashes_match': unchanged,
               'wall_s': time.monotonic() - start, 'numerical_model_only': True}
    write(args.output / 'summary.json', summary)
    print(json.dumps({'stage': 'complete', **summary}), flush=True)
    return 0 if unchanged else 1


if __name__ == '__main__':
    raise SystemExit(main())
