"""Learn obstruction appearance in simulation, then test bounded pose memory.

    python3 -m sim.obstruction_bench --out eval/obstruction_learning/run

All model files, frames and results go into the explicit output directory.
The production watcher, camera, route and room models are never changed.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .obstruction_model import classifier_metrics, fit, pixel_quality, select_threshold
from .obstruction_scene import Occluder, ObstructionScene, domain_seeds, image_features
from .view_obstructions import Config, Observation, ObstructionMemory


ROOT = Path(__file__).resolve().parents[1]
POSES = ((400, 600), (1100, 600), (1850, 600), (2600, 600), (3300, 600))
FAMILIES = ('persistent', 'empty', 'partial', 'transient', 'removed',
            'changed_setup', 'unannounced_move', 'dark', 'blur', 'long_transient')
POLICIES = ('always_inspect', 'empty_means_blocked', 'visual_memory')
CHANGE_S = 185
END_S = 485
TICK_S = 15
PROBE_BUDGET = 2


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dataset(split, groups, keep_examples=False, namespace=None, reference=False):
    """Hold out scene/background/object identities and entire angular sectors."""
    intervals = {'train': (250, 1400), 'development': (1450, 2300), 'test': (2350, 3450)}
    lo, hi = intervals[split]
    X, y, rows, examples = [], [], [], []
    for group in range(groups):
        seeds = domain_seeds(namespace or split, group)
        rng = np.random.default_rng(seeds['seed'])
        pose = (float(rng.uniform(lo, hi)), float(rng.uniform(300, 900)))
        for variant in range(8):
            objects = []
            if variant in (1, 2, 3, 4, 6):
                size = (.45, .7) if variant == 1 else (1, 1) if variant == 2 else (1.3, 1.3) if variant == 3 else (2.4, 2.4)
                objects = [Occluder(anchor_pose=pose, size_uv=size,
                                    center_uv=(-.15, .5) if variant == 2 else (.5, .5),
                                    shape='sphere' if variant == 4 else 'box',
                                    texture='flat' if variant == 6 else ('woven', 'wood', 'speckle')[group % 3],
                                    distance_m=float(rng.uniform(.15, .65)), seed=variant)]
            scene = ObstructionScene(**seeds, occluders=objects, width=96, height=56,
                                     background_kind='blank' if variant == 5 else ('room', 'texture')[group % 2],
                                     low_light=float(rng.uniform(.35, 1.15)),
                                     blur=3 if variant == 7 else 0)
            image = scene.render(pose, 5)
            hidden = scene.truth(pose, 5)
            label = int(hidden.coverage >= .85)
            reference_image = None
            if reference:
                from .obstruction_reference import pair_features
                # This is a supplied, pre-injection observation, not a mask or
                # a clean current-time oracle. Object start times are >=0.
                ref_rng = np.random.default_rng(domain_seeds((namespace or split) + '-reference', group * 8 + variant)['seed'])
                ref_pose = tuple(np.asarray(pose) + ref_rng.uniform([-8, -6], [8, 6]))
                original_light = scene.low_light
                scene.low_light = float(ref_rng.uniform(.5, 1.1))
                reference_image = scene.render(ref_pose, -60)
                scene.low_light = original_light
                X.append(pair_features(image, reference_image))
            else:
                X.append(image_features(image))
            y.append(label)
            rows.append({'split': split, 'group': group, 'variant': variant, 'pose': pose,
                         'label': label, 'injected_coverage': hidden.coverage,
                         'domain': seeds, 'image_sha256': hashlib.sha256(image.tobytes()).hexdigest()})
            if keep_examples and group < 3:
                item = (f'{split}-{group}-{variant}', image, rows[-1])
                examples.append(item + (reference_image,) if reference else item)
    return np.asarray(X), np.asarray(y), rows, examples


def make_episode(split, index, background=None, calib=None, stress=False):
    family = FAMILIES[index % len(FAMILIES)]
    seeds = domain_seeds(split, index)
    rng = np.random.default_rng(seeds['seed'])
    anchor, alternate = POSES[2], POSES[4]
    texture = ('woven', 'wood', 'speckle', 'flat')[index % 4]
    common = dict(anchor_pose=anchor, size_uv=(1.3, 1.3), texture=texture,
                  distance_m=float(rng.uniform(.2, .6)))
    objects = []
    if family in ('persistent', 'removed', 'changed_setup', 'unannounced_move'):
        if family == 'removed':
            objects = [Occluder(**common, end_s=CHANGE_S)]
        elif family in ('changed_setup', 'unannounced_move'):
            objects = [Occluder(**common, active_versions=(0,)),
                       Occluder(**dict(common, anchor_pose=alternate), active_versions=(1,), seed=1)]
        else:
            objects = [Occluder(**common)]
    elif family == 'partial':
        objects = [Occluder(**dict(common, size_uv=(.6, .7)))]
    elif family in ('transient', 'long_transient'):
        objects = [Occluder(**common, start_s=80, end_s=245 if family == 'long_transient' else 110, role='transient')]
    scene = ObstructionScene(**seeds, occluders=objects, background=background, calib=calib,
                             width=96, height=56, background_kind='blank' if stress and index % 4 == 0 else 'room',
                             low_light=.02 if family == 'dark' else float(rng.uniform(.4, 1.05)),
                             blur=7 if family == 'blur' else 0,
                             mount_offsets={1: (8, -7, 1)} if family == 'changed_setup' else None)
    return family, scene


def episode(model, config, split, index, background=None, calib=None, stress=False, keep_rows=False):
    family, scene = make_episode(split, index, background, calib, stress)
    memory = ObstructionMemory(config)
    requires_reference = bool(getattr(model, 'requires_reference', False))
    references = {}
    if requires_reference:
        ref_rng = np.random.default_rng(domain_seeds(split + '-reference', index)['seed'])
        for pose in POSES:
            error = ref_rng.uniform(-40, 40, 2) if stress else ref_rng.uniform([-8, -6], [8, 6])
            references[pose] = scene.render(tuple(np.asarray(pose) + error), -60, 0)
    counts = {policy: {'opportunities': 0, 'inspections': 0, 'skips': 0,
                       'useful_skips': 0, 'false_skips': 0, 'deferred_probes': 0,
                       'persistent_skips': 0, 'transient_skips': 0,
                       'missed_pet_opportunities': 0} for policy in POLICIES}
    naive_misses = {p: 0 for p in POSES}
    transitions, samples, recovery = [], [], []
    learned_before_change = False
    change_at = 245 if family == 'long_transient' else CHANGE_S
    changed = False
    for tick in range(5, END_S + 1, TICK_S):
        version = int(tick >= CHANGE_S)
        context = f'geometry:fixed|model:frozen|setup:{version if family == "changed_setup" else 0}'
        if family == 'changed_setup' and version and not changed:
            memory.setup_changed(context, tick)
            changed = True
        proposed = memory.due(POSES, tick, context, PROBE_BUDGET)
        probes = {tuple(d['pose']) for d in proposed}
        for j, pose in enumerate(POSES):
            now = tick + j * .25
            hidden = scene.truth(pose, now, version)
            # Truth never enters the memory/model API. It only evaluates the
            # decisions and generates a separately declared pet-observation test.
            pet = (family == 'empty' and pose == POSES[2] and now >= 245 and int(tick / 15) % 4 == 0)
            decision = memory.decision(pose, now, context)
            if pose == POSES[2] and now < change_at and decision['skip']:
                learned_before_change = True
            naive_skip = naive_misses[pose] >= 3
            decisions = {'always_inspect': False, 'empty_means_blocked': naive_skip,
                         'visual_memory': decision['skip']}
            for policy, skip in decisions.items():
                c = counts[policy]
                c['opportunities'] += 1
                if skip:
                    c['skips'] += 1
                    c['useful_skips' if hidden.coverage >= .85 else 'false_skips'] += 1
                    c['persistent_skips'] += int(hidden.persistent_coverage >= .85)
                    c['transient_skips'] += int(hidden.transient_coverage >= .85)
                    c['missed_pet_opportunities'] += int(pet)
                else:
                    c['inspections'] += 1
            if not naive_skip:
                naive_misses[pose] = 0 if pet else naive_misses[pose] + 1
            deferred = decision['state'] == 'reinspection_due' and pose not in probes
            counts['visual_memory']['deferred_probes'] += int(deferred)
            if deferred:
                counts['visual_memory']['inspections'] -= 1
                counts['visual_memory']['missed_pet_opportunities'] += int(pet)
            probability, accepted, inspected = None, None, False
            if not decision['skip'] and not deferred:
                image = scene.render(pose, now, version)
                inspected = True
                # An authoritative remount invalidates every old reference;
                # no fresh clean frame is conjured after the change.
                reference = None if family == 'changed_setup' and version else references.get(pose)
                probability = (model.predict_pair(image, reference, registered=reference is not None)
                               if requires_reference else model.predict(image))
                observed = memory.observe(Observation(pose=pose, captured_at=now, observed_at=now,
                                                     context=context, obstruction_probability=probability,
                                                     quality=pixel_quality(image), pet_detected=pet))
                accepted = observed['accepted']
                if observed['state'] != decision['state']:
                    transitions.append({'at_s': now, 'pose': pose, 'from': decision['state'],
                                        'to': observed['state'], 'reason': observed['reason']})
            if (family in ('removed', 'changed_setup', 'unannounced_move', 'long_transient')
                    and now >= change_at and pose == POSES[2] and hidden.coverage < .85):
                recovery.append({'at_s': now, 'eligible': not decision['skip'],
                                 'inspected': inspected,
                                 'clear_accepted': accepted is True and probability is not None and probability < config.high_threshold})
            if keep_rows:
                samples.append({'at_s': now, 'pose': pose, 'coverage': hidden.coverage,
                                'truth_blocking': hidden.blocking, 'score': probability,
                                'state': decision['state'], 'skip': decision['skip'],
                                'probe_due': decision['probe_due'], 'deferred': deferred,
                                'context': context, 'accepted': accepted})
    return {'id': f'{split}-{index}', 'family': family, 'domain': domain_seeds(split, index),
            'counts': counts, 'transitions': transitions,
            'learned_before_change': learned_before_change,
            'first_reopened_after_change_s': next((r['at_s'] - change_at for r in recovery if r['eligible']), None),
            'first_inspection_after_change_s': next((r['at_s'] - change_at for r in recovery if r['inspected']), None),
            'clear_accepted_after_change_s': next((r['at_s'] - change_at for r in recovery if r['clear_accepted']), None),
            'samples': samples}


def lifecycle_summary(episodes):
    out = {'episodes': len(episodes), 'policies': {}, 'by_family': {}}
    for policy in POLICIES:
        total = {key: sum(e['counts'][policy][key] for e in episodes)
                 for key in episodes[0]['counts'][policy]}
        total['inspection_reduction'] = 1 - total['inspections'] / max(1, total['opportunities'])
        total['actual_exclusion_reduction'] = total['skips'] / max(1, total['opportunities'])
        out['policies'][policy] = total
    for family in FAMILIES:
        rows = [e for e in episodes if e['family'] == family]
        recovery = [e['first_inspection_after_change_s'] for e in rows if e['first_inspection_after_change_s'] is not None]
        learned = [e for e in rows if e['learned_before_change']]
        expected_recovery = family in ('removed', 'changed_setup', 'unannounced_move', 'long_transient')
        bound = 1.0 if family == 'changed_setup' else 60 + TICK_S * (int(np.ceil(len(POSES) / PROBE_BUDGET)) + 1)
        out['by_family'][family] = {'episodes': len(rows),
                                   'learned_before_change': sum(e['learned_before_change'] for e in rows),
                                   'visual_skips': sum(e['counts']['visual_memory']['skips'] for e in rows),
                                   'false_skips': sum(e['counts']['visual_memory']['false_skips'] for e in rows),
                                   'max_first_inspection_after_change_s': max(recovery) if recovery else None,
                                   'recovery_bound_s': bound if expected_recovery else None,
                                   'learned_cases_recovered_within_bound':
                                       sum(e['first_inspection_after_change_s'] is not None and
                                           e['first_inspection_after_change_s'] <= bound for e in learned)
                                       if expected_recovery else None,
                                   'learned_cases_requiring_recovery': len(learned) if expected_recovery else 0}
    return out


_BANK = None


def _init_bank(model, config, split, background, calib, stress):
    global _BANK
    _BANK = (model, config, split, background, calib, stress)


def _bank_episode(index):
    model, config, split, background, calib, stress = _BANK
    return episode(model, config, split, index, background, calib, stress, keep_rows=index < len(FAMILIES))


def run_bank(model, config, split, count, background=None, calib=None, stress=False, jobs=1):
    if jobs == 1:
        return [episode(model, config, split, i, background, calib, stress, keep_rows=i < len(FAMILIES))
                for i in range(count)]
    with ProcessPoolExecutor(max_workers=jobs, initializer=_init_bank,
                             initargs=(model, config, split, background, calib, stress)) as pool:
        return list(pool.map(_bank_episode, range(count)))


def choose_model(X, y, devX, devY, reference_training=None, reference_development=None):
    from .obstruction_forest import fit as forest_fit
    candidates = [('linear', fit(X, y)),
                  ('forest_depth4', forest_fit(X, y, seed=104, max_depth=4)),
                  ('forest_depth8', forest_fit(X, y, seed=108, max_depth=8))]
    if reference_training is not None and reference_development is not None:
        from .obstruction_reference import fit as reference_fit
        candidates.append(('registered_reference_forest', reference_fit(reference_training, y, seed=204,
                                                                       max_depth=8)))
    selections = []
    for name, model in candidates:
        validation = reference_development if getattr(model, 'requires_reference', False) else devX
        try:
            selection = dict(select_threshold(devY, model.scores(validation)), enabled=True)
        except ValueError as error:
            selection = {'enabled': False, 'reason': str(error), 'threshold': 1.0,
                         'development': classifier_metrics(devY, model.scores(validation), 1.0)}
        selections.append(dict(selection, model=name))
    eligible = [i for i in range(len(candidates)) if selections[i]['enabled']]
    if not eligible:
        raise ValueError('all visual models failed development calibration; leave obstruction learning disabled')
    index = min(eligible, key=lambda i: (-selections[i]['development']['tp'], i))
    chosen = selections[index]
    chosen['model_candidates'] = [{k: v for k, v in row.items() if k != 'candidates'} for row in selections]
    chosen['model_selection_rule'] = 'Maximize development true positives under zero observed FP; ties favour the earlier simpler model.'
    return candidates[index][1], chosen


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--train-groups', type=int, default=160)
    parser.add_argument('--development-groups', type=int, default=64)
    parser.add_argument('--test-groups', type=int, default=128)
    parser.add_argument('--episodes', type=int, default=90)
    parser.add_argument('--test-namespace', default='heldout-v1')
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--room', type=Path, help='Optional private room.json for photo-composite sensitivity only')
    args = parser.parse_args(argv)
    if min(args.train_groups, args.development_groups, args.test_groups, args.episodes) < 1 or not 1 <= args.jobs <= 8:
        parser.error('sample counts must be positive')
    if not args.test_namespace or args.test_namespace in ('train', 'development'):
        parser.error('test namespace must be distinct from training and development')
    args.out.mkdir(parents=True, exist_ok=False)
    inputs = ('sim/obstruction_scene.py', 'sim/obstruction_model.py', 'sim/obstruction_forest.py', 'sim/obstruction_reference.py',
              'sim/view_obstructions.py', 'sim/obstruction_bench.py')
    manifest = {'input_sha256': {p: sha(ROOT / p) for p in inputs},
                'splits': {'train_groups': args.train_groups, 'development_groups': args.development_groups,
                           'test_groups': args.test_groups, 'images_per_group': 8,
                           'test_namespace': args.test_namespace,
                           'sectors': {'train': [250, 1400], 'development': [1450, 2300], 'test': [2350, 3450]}},
                'pixel_label': 'total injected near-object coverage >=0.85; pixels only at prediction',
                'memory_label': 'useful skip when total injected coverage >=0.85; persistent/transient evaluator counts kept separate',
                'lifecycle': {'families': FAMILIES, 'policies': POLICIES, 'change_s': CHANGE_S,
                              'end_s': END_S, 'tick_s': TICK_S, 'max_probes_per_tick': PROBE_BUDGET,
                              'worker_processes': args.jobs},
                'scope': 'Geometric near primitives with schematic/photo RGB backgrounds. Not a validated room/detector simulation.',
                'limits': ['An identical RGB image can represent different physical depths; image confidence cannot prove permanence.',
                           'A floor-bearing view centre is not a visibility label. Empty pet detections never train the visual memory.',
                           'Independent domains/sectors are synthetic holdouts, not independent labelled household recordings.',
                           'Photo-composite labels describe injected objects only; the original background is unlabelled.',
                           'A scheduled view-opportunity experiment does not measure actual motor travel, search latency or live pet recall.',
                           'Pet sightings in the empty-view control are scripted symbolic evidence contracts, not RGB-rendered pet detections.',
                           'Explicit setup events are known inputs; unannounced changes wait for bounded reinspection.',
                           'Repeated visits are temporally distinct, not statistically independent physical worlds.']}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    X, y, training, _ = dataset('train', args.train_groups)
    devX, devY, development, _ = dataset('development', args.development_groups)
    refX, refY, _, _ = dataset('train', args.train_groups, reference=True)
    refDevX, refDevY, _, _ = dataset('development', args.development_groups, reference=True)
    assert np.array_equal(y, refY) and np.array_equal(devY, refDevY)
    model, selection = choose_model(X, y, devX, devY, refX, refDevX)
    manifest['reference_protocol'] = {'candidate_supplied': True, 'selected': bool(getattr(model, 'requires_reference', False)),
                                      'acquired_at_s': -60, 'object_injection_starts_at_s': 0,
                                      'source': 'Pre-acquired RGB before injected objects exist; original background is not certified clear.',
                                      'registration': 'Declared pose match with small simulated acquisition error; stress increases error.',
                                      'invalidation': 'Announced remount drops references; no clean current-time refresh is supplied.',
                                      'production_status': 'No registered-reference cache or validation has been enabled in production.'}
    (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'stage': 'trained', 'images': len(y), 'features': X.shape[1],
                      'model': selection['model']}), flush=True)
    threshold = selection['threshold']
    config = Config(high_threshold=threshold, low_threshold=min(.2, threshold / 2),
                    min_samples=3, min_span_s=45, min_gap_s=15,
                    probe_interval_s=60, exclusion_ttl_s=180)
    selection.update(config=asdict(config), frozen_before_test=True)
    (args.out / 'model.json').write_text(json.dumps(model.to_dict(), indent=2) + '\n')
    (args.out / 'selection.json').write_text(json.dumps(selection, indent=2) + '\n')
    print(json.dumps({'stage': 'selected', **selection['development']}), flush=True)
    testX, testY, testing, examples = dataset('test', args.test_groups, keep_examples=True, namespace=args.test_namespace,
                                           reference=bool(getattr(model, 'requires_reference', False)))
    scores = model.scores(testX)
    metrics = classifier_metrics(testY, scores, threshold)
    (args.out / 'pixel-results.json').write_text(json.dumps({'training': training, 'development': development,
                                                           'test': [dict(row, score=float(score)) for row, score in zip(testing, scores)],
                                                           'metrics': metrics}, indent=1) + '\n')
    from PIL import Image
    example_rows = []
    for item in examples:
        name, rgb, metadata, *extra = item
        Image.fromarray(rgb).resize((384, 224)).save(args.out / (name + '.png'))
        reference = extra[0] if extra else None
        score = model.predict_pair(rgb, reference) if reference is not None else model.predict(rgb)
        if reference is not None:
            Image.fromarray(reference).resize((384, 224)).save(args.out / (name + '-reference.png'))
        example_rows.append(dict(metadata, path=name + '.png', score=score,
                                 reference_path=name + '-reference.png' if reference is not None else None))
    (args.out / 'examples.json').write_text(json.dumps(example_rows, indent=1) + '\n')
    banks = {}
    for split, stress in [('lifecycle-' + args.test_namespace, False), ('stress-' + args.test_namespace, True)]:
        episodes = run_bank(model, config, split, args.episodes, stress=stress, jobs=args.jobs)
        banks[split] = lifecycle_summary(episodes)
        (args.out / (split + '.json')).write_text(json.dumps({'episodes': episodes, 'summary': banks[split]}, indent=1) + '\n')
        print(json.dumps({'stage': split, **banks[split]['policies']['visual_memory']}), flush=True)
    if args.room:
        room = json.loads(args.room.read_text())
        photo = args.room.parent / room['textures']['photo']
        with Image.open(photo) as image:
            background = np.asarray(image.convert('RGB'))
        episodes = run_bank(model, config, 'photo-' + args.test_namespace, args.episodes,
                            background, room['calib'], jobs=args.jobs)
        banks['photo-sensitivity'] = lifecycle_summary(episodes)
        (args.out / 'photo-sensitivity.json').write_text(json.dumps({'episodes': episodes, 'summary': banks['photo-sensitivity'],
                                                                  'background_sha256': sha(photo), 'room_sha256': sha(args.room),
                                                                  'labels': 'injected-object proxy only; existing background unlabelled'}, indent=1) + '\n')
    synthetic_banks = [b for name, b in banks.items() if name != 'photo-sensitivity']
    checks = {'heldout_pixel_fp_zero': metrics['fp'] == 0,
              'heldout_pixel_recall_at_least_half': metrics['recall'] >= .5,
              'no_false_view_exclusions': all(b['policies']['visual_memory']['false_skips'] == 0 for b in synthetic_banks),
              'no_missed_pet_opportunities': all(b['policies']['visual_memory']['missed_pet_opportunities'] == 0 for b in synthetic_banks),
              'useful_savings': synthetic_banks[0]['policies']['visual_memory']['useful_skips'] > 0,
              'bounded_reinspection': all(f['learned_cases_recovered_within_bound'] == f['learned_cases_requiring_recovery']
                                         for b in synthetic_banks for f in b['by_family'].values()
                                         if f['recovery_bound_s'] is not None)}
    unchanged = all(sha(ROOT / p) == h for p, h in manifest['input_sha256'].items())
    report = {'at_unix': time.time(), 'pixel_test': metrics, 'selected_threshold': threshold,
              'banks': banks, 'simulation_checks': checks, 'simulation_gate': all(checks.values()),
              'inputs_unchanged': unchanged, 'production_changed': False,
              'conclusion': 'Offline evidence only; no live suppression or real-camera accuracy claim.'}
    (args.out / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'stage': 'complete', 'pixel_test': metrics, 'checks': checks}), flush=True)
    return 0 if unchanged else 1


if __name__ == '__main__':
    raise SystemExit(main())
