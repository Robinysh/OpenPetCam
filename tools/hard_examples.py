#!/usr/bin/env python3
"""Mine full-frame review examples without treating detector output as ground truth.

    hard_examples.py mine --frames eval/bootstrap/frames --reports eval/reports --out eval/hard_examples/run1
    hard_examples.py validate --manifest .../manifest.json --annotations reviewed.json
    hard_examples.py export --manifest .../manifest.json --annotations reviewed.json --out .../export

Outputs are immutable snapshots. No camera, network, training, deployment or legacy
label writes occur. A review queue is a request for visual labels, not training data.
"""
import argparse
from collections import Counter
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile

from PIL import Image

VERSION = 1
SPLITS = {'train', 'dev', 'test', 'quarantine'}
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png'}
DAY = re.compile(r'^\d{4}-\d{2}-\d{2}$')
REPORT = re.compile(r'^(\d{4})(\d{2})(\d{2})-(\d{2})(\d{2})(\d{2})$')
CLOCK = re.compile(r'^(\d{2})-(\d{2})(?:-(\d{2}))?$|^(\d{2})(\d{2})(\d{2})$')


def canonical(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False, ensure_ascii=False) + '\n').encode()


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def valid_day(value):
    if not isinstance(value, str) or not DAY.fullmatch(value):
        return False
    try:
        return dt.date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def dhash(image):
    values = image.convert('L').resize((9, 8), Image.Resampling.BILINEAR).tobytes()
    n = 0
    for y in range(8):
        for x in range(8):
            n = (n << 1) | (values[y * 9 + x + 1] > values[y * 9 + x])
    return f'{n:016x}'


def infer_time(path, metadata):
    """Only date folders and clock-shaped filenames are interpreted as timestamps."""
    dates = {part for part in path.parts if valid_day(part)}
    stamp = None
    for part in reversed(path.parts):
        match = REPORT.fullmatch(part)
        if match:
            y, mo, d, h, mi, sec = match.groups()
            stamp = f'{y}-{mo}-{d}T{h}:{mi}:{sec}'
            dates.add(stamp[:10])
            break
    if stamp is None and len(dates) == 1:
        match = CLOCK.fullmatch(path.stem)
        if match:
            h, mi, sec = (match.groups()[:3] if match.group(1) else match.groups()[3:])
            stamp = next(iter(dates)) + f'T{h}:{mi}:{sec or "00"}'
    explicit = metadata.get('capture_time')
    if explicit is not None:
        if not isinstance(explicit, str):
            return None, None, ['invalid_capture_time']
        try:
            dt.datetime.fromisoformat(explicit.replace('Z', '+00:00'))
        except ValueError:
            return None, None, ['invalid_capture_time']
        if len(explicit) < 19 or explicit[10] not in ('T', ' '):
            return None, None, ['invalid_capture_time']
        stamp = explicit
        dates.add(stamp[:10])
    provided = metadata.get('day')
    if provided is not None:
        if not valid_day(provided):
            return None, None, ['invalid_day']
        dates.add(provided)
    if len(dates) != 1:
        return None, None, ['conflicting_days' if dates else 'day_unknown']
    day = next(iter(dates))
    if stamp is None:
        return day, None, ['capture_time_unknown']
    try:
        dt.datetime.fromisoformat(stamp.replace('Z', '+00:00'))
    except ValueError:
        return day, None, ['invalid_capture_time']
    return day, stamp, []


def _read_json(path):
    with Path(path).open() as handle:
        return json.load(handle)


def _source_manifest(path):
    value = _read_json(path)
    rows = value.get('frames') if isinstance(value, dict) else value
    if not isinstance(rows, list):
        raise ValueError('source manifest must be a list or an object with frames')
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('path'), str):
            raise ValueError('source records need a path')
        item = dict(row)
        p = Path(item['path']).expanduser()
        item['path'] = str((Path(path).resolve().parent / p).resolve() if not p.is_absolute() else p.resolve())
        item.setdefault('source_type', 'manifest')
        item.setdefault('source_root', str(Path(path).resolve()))
        yield item


def inputs(frame_roots=(), report_roots=(), source_manifest=None):
    records, problems = [], []
    for raw_root in frame_roots:
        root = Path(raw_root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError('a frame source directory does not exist')
        for path in sorted(root.rglob('*')):
            if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            if not path.resolve().is_relative_to(root):
                problems.append({'path': str(path), 'reason': 'source_symlink_escape'})
                continue
            records.append({'path': str(path.resolve()), 'source_type': 'frame', 'source_root': str(root)})
    for raw_root in report_roots:
        root = Path(raw_root).expanduser().resolve()
        if not root.is_dir():
            raise ValueError('a report source directory does not exist')
        for path in sorted(root.glob('*/frame.jpg')):
            if not path.resolve().is_relative_to(root):
                problems.append({'path': str(path), 'reason': 'source_symlink_escape'})
                continue
            metadata = path.parent / 'report.json'
            if not metadata.is_file() or not metadata.resolve().is_relative_to(root):
                problems.append({'path': str(path), 'reason': 'report_metadata_missing_or_unsafe'})
                continue
            try:
                report = _read_json(metadata)
                if not isinstance(report, dict):
                    raise ValueError('report must be an object')
            except (OSError, ValueError):
                problems.append({'path': str(path), 'reason': 'report_metadata_unreadable'})
                continue
            client = report.get('client') if isinstance(report.get('client'), dict) else {}
            records.append({'path': str(path.resolve()), 'source_type': 'report', 'source_root': str(root),
                            'report_id': path.parent.name, 'report_note': str(report.get('note') or ''),
                            'reported_status': str(client.get('shown_status') or '')})
    if source_manifest:
        records.extend(_source_manifest(source_manifest))
    unique = {}
    for item in records:
        key = (item['path'], item.get('source_type'), item.get('source_root'))
        if key in unique and unique[key] != item:
            raise ValueError('conflicting metadata for the same source path')
        unique[key] = item
    return [unique[key] for key in sorted(unique)], problems


def _day_split(day, day_map):
    if day_map is not None:
        return day_map.get(day, 'quarantine')
    value = int(hashlib.sha256(('hard-examples-v1:' + day).encode()).hexdigest()[:8], 16) % 10
    return 'train' if value < 6 else 'dev' if value < 8 else 'test'


def _scores_for(record, scores):
    for key in (record['id'], record['sha256'], record['path']):
        if key in scores:
            value = scores[key]
            if not isinstance(value, dict):
                raise ValueError('score records must be objects')
            return value
    return {}


def _rank(record, scores):
    score = 0
    reasons = []
    if any(alias['source_type'] == 'report' for alias in record['aliases']):
        score += 100
        reasons.append('owner_report_requires_visual_review')
        if any(re.search(r'\b(see|visible|nose|here|seeing)\b', alias.get('report_note', ''), re.I)
               for alias in record['aliases']):
            score += 30
            reasons.append('reported_visibility_is_a_label_lead_not_box_truth')
    if record.get('focused_report'):
        score += 1000
        reasons.append('explicit_report_focus_not_a_label')
    metrics = _scores_for(record, scores)
    if metrics.get('proposal_count') == 0:
        score += 25
        reasons.append('no_proposal_needs_visual_review_not_absence')
    if metrics.get('verifier_rejected') is True:
        score += 40
        reasons.append('rejected_candidate')
    if metrics.get('disagreement') is True:
        score += 40
        reasons.append('model_disagreement')
    value, threshold = metrics.get('verifier_max'), metrics.get('verifier_threshold')
    if all(isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v) for v in (value, threshold)):
        if abs(value - threshold) <= 0.1:
            score += 20
            reasons.append('near_verifier_boundary')
    record['mining_signals'] = metrics
    record['reasons'] = reasons or ['uniform_full_frame_control']
    record['priority'] = score


def build_manifest(records, day_map=None, scores=None, problems=(), focus_reports=()):
    if day_map is not None and (not isinstance(day_map, dict) or any(not valid_day(k) or not isinstance(v, str) or v not in SPLITS for k, v in day_map.items())):
        raise ValueError('day map must map ISO dates to train, dev, test or quarantine')
    if any(not isinstance(ident, str) or not REPORT.fullmatch(ident) for ident in focus_reports):
        raise ValueError('focused report IDs must be timestamp-shaped identifiers')
    scores = scores or {}
    if not isinstance(scores, dict):
        raise ValueError('scores must be an object keyed by frame ID, SHA256 or path')
    by_hash = {}
    excluded = list(problems)
    for item in sorted(records, key=lambda r: (r['path'], r.get('source_type', ''))):
        path = Path(item['path'])
        try:
            sha = digest(path)
            with Image.open(path) as image:
                width, height = image.size
                if width * height > 40_000_000 or min(width, height) < 1:
                    raise ValueError('image dimensions exceed the review bound')
                image.load()
                perceptual = dhash(image)
        except (OSError, ValueError, Image.DecompressionBombError):
            excluded.append({'path': str(path), 'reason': 'unreadable_or_oversize_image'})
            continue
        day, capture, issues = infer_time(path, item)
        group = item.get('group')
        if group is not None and (not isinstance(group, str) or not group.strip() or len(group) > 200):
            issues.append('invalid_recording_group')
            group = None
        group = group or (day + ':hour-' + capture[11:13] if day and capture else None)
        source = str(item.get('source_type') or 'manifest')
        alias = {'path': str(path.resolve()), 'source_type': source,
                 'source_id': hashlib.sha256(str(item.get('source_root', path.parent)).encode()).hexdigest()[:16],
                 'day': day, 'capture_time': capture, 'group': group, 'issues': issues}
        for name in ('scene', 'report_id', 'report_note', 'reported_status'):
            if isinstance(item.get(name), str):
                alias[name] = item[name]
        if sha not in by_hash:
            by_hash[sha] = {'id': 'f' + sha[:24], 'sha256': sha, 'path': str(path.resolve()),
                            'width': width, 'height': height, 'dhash': perceptual, 'aliases': []}
        by_hash[sha]['aliases'].append(alias)
    entries = []
    for sha, record in sorted(by_hash.items()):
        aliases = sorted(record['aliases'], key=lambda a: (a['source_type'] != 'report', a['path']))
        primary = aliases[0]
        record['aliases'] = aliases
        record['path'] = primary['path']
        days = {a['day'] for a in aliases if a['day']}
        proposed = {_day_split(day, day_map) for day in days}
        issues = sorted({problem for a in aliases for problem in a['issues']})
        if len(proposed) > 1:
            issues.append('duplicate_cross_split_conflict')
        if day_map is not None and any(day not in day_map for day in days):
            issues.append('day_missing_from_frozen_map')
        record.update({k: primary.get(k) for k in ('day', 'capture_time', 'group', 'scene', 'source_type', 'source_id')})
        record['issues'] = issues
        record['split'] = next(iter(proposed)) if len(proposed) == 1 and not issues else 'quarantine'
        record['label'] = 'unreviewed'
        record['focused_report'] = any(alias.get('report_id') in focus_reports for alias in aliases)
        _rank(record, scores)
        entries.append(record)
    group_splits = {}
    for record in entries:
        if record['split'] != 'quarantine':
            for alias in record['aliases']:
                group_splits.setdefault(alias['group'], set()).add(record['split'])
    conflicting_groups = {group for group, splits in group_splits.items() if len(splits) > 1}
    for record in entries:
        if any(alias['group'] in conflicting_groups for alias in record['aliases']):
            record['split'] = 'quarantine'
            record['issues'].append('recording_group_cross_split_conflict')
    # Near duplicates affect queue diversity only. Never propagate labels or
    # silently discard a tiny visible part because its background is unchanged.
    scene_buckets = {}
    for record in sorted(entries, key=lambda r: (-r['priority'], r['id'])):
        if not record.get('scene') or not record.get('group'):
            continue
        bucket = scene_buckets.setdefault((record['day'], record['group'], record['scene']), [])
        near = next((other for other in bucket if (int(record['dhash'], 16) ^ int(other['dhash'], 16)).bit_count() <= 2), None)
        if near:
            record['near_duplicate_of'] = near['id']
        bucket.append(record)
    value = {'version': VERSION, 'kind': 'full_frame_review_inventory',
             'split_policy': {'unit': 'day', 'day_map': day_map, 'default': 'sha256 day hash 60/20/20' if day_map is None else 'explicit frozen day map',
                              'unknown': 'quarantine', 'test_queue': 'excluded unless explicitly requested'},
             'focused_reports': sorted(set(focus_reports)),
             'entries': entries, 'excluded': sorted(excluded, key=lambda r: (r['path'], r['reason'])),
             'label_policy': 'All mined examples are unreviewed. Detector outputs and owner report text are leads, never automatic box or absence labels.'}
    value['manifest_sha256'] = hashlib.sha256(canonical(value)).hexdigest()
    return value


def review_queue(manifest, limit=60, per_group=4, per_source=30, splits=('train', 'dev'), control_fraction=0.25):
    if min(limit, per_group, per_source) < 1 or not 0 <= control_fraction <= 1:
        raise ValueError('queue limits must be positive and control fraction must be in 0..1')
    if not set(splits).issubset(SPLITS):
        raise ValueError('unknown review split')
    candidates = [r for r in manifest['entries'] if r['split'] in splits]
    counts, sources, selected, selected_ids = Counter(), Counter(), [], set()

    def add(row, role):
        key = row.get('group') or 'quarantine:' + row['source_id']
        if row['id'] in selected_ids or counts[key] >= per_group or sources[row['source_id']] >= per_source:
            return
        selected.append({'id': row['id'], 'path': row['path'], 'sha256': row['sha256'], 'day': row['day'], 'group': row['group'],
                         'split': row['split'], 'selection': role, 'reasons': row['reasons'], 'label': 'unreviewed',
                         'required_review': ['visible / absent / uncertain', 'box the visible part only when identifiable', 'record reviewer and label source; do not infer hidden body extent']})
        selected_ids.add(row['id']); counts[key] += 1; sources[row['source_id']] += 1

    controls = sorted(candidates, key=lambda r: hashlib.sha256(('control:' + r['id']).encode()).hexdigest())
    target_controls = min(max(0, limit - 1), max(1, round(limit * control_fraction))) if candidates and control_fraction else 0
    ranked = sorted(candidates, key=lambda r: (-r['priority'], bool(r.get('near_duplicate_of')), r['id']))
    for row in ranked:
        if len(selected) >= limit - target_controls:
            break
        add(row, 'hard_example')
    for row in controls:
        if len(selected) >= limit:
            break
        add(row, 'uniform_control')
    # Caps may leave a reserved control slot empty. Fill what is still feasible.
    for row in ranked:
        if len(selected) >= limit:
            break
        add(row, 'hard_example')
    return {'version': VERSION, 'manifest_sha256': manifest['manifest_sha256'], 'limit': limit,
            'review_splits': list(splits), 'entries': selected, 'counts': dict(Counter(r['split'] for r in selected)),
            'instructions': 'Review full frames. Uncertain examples remain excluded from training. Mining priority is not a label.'}


def write_snapshots(directory, files):
    """Never replace a prior dataset, annotation file or evaluation snapshot."""
    directory = Path(directory)
    encoded = {name: canonical(value) for name, value in files.items()}
    for name, body in encoded.items():
        path = directory / name
        if path.exists() and path.read_bytes() != body:
            raise ValueError('output snapshot already exists with different content; choose a new run directory')
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name, body in encoded.items():
        path = directory / name
        if path.exists():
            continue
        fd, temporary = tempfile.mkstemp(prefix='.hard-examples-', dir=directory)
        try:
            with os.fdopen(fd, 'wb') as handle:
                handle.write(body); handle.flush(); os.fsync(handle.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != body:
                    raise ValueError('output snapshot changed concurrently') from None
        finally:
            os.unlink(temporary)


def _box(value):
    return (isinstance(value, list) and len(value) == 4
            and all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) and 0 <= x <= 1 for x in value)
            and value[0] < value[2] and value[1] < value[3])


def validate(manifest, annotations, verify_files=True):
    if not isinstance(manifest, dict) or manifest.get('version') != VERSION or not isinstance(manifest.get('entries'), list):
        raise ValueError('unsupported manifest')
    expected = manifest.get('manifest_sha256'); core = {k: v for k, v in manifest.items() if k != 'manifest_sha256'}
    if expected != hashlib.sha256(canonical(core)).hexdigest():
        raise ValueError('manifest content changed after freeze')
    entries, seen_sha, day_splits, group_splits = {}, {}, {}, {}
    for row in manifest['entries']:
        if row['id'] in entries or row['sha256'] in seen_sha:
            raise ValueError('duplicate manifest identity')
        if row.get('split') not in SPLITS or row['id'] != 'f' + row['sha256'][:24]:
            raise ValueError('invalid manifest identity or split')
        if row['split'] != 'quarantine':
            for alias in row['aliases']:
                day = alias.get('day')
                if not valid_day(day) or alias.get('issues'):
                    raise ValueError('unknown provenance escaped quarantine')
                if day in day_splits and day_splits[day] != row['split']:
                    raise ValueError('a day leaks across dataset partitions')
                day_splits[day] = row['split']
                group = alias.get('group')
                if not isinstance(group, str) or not group:
                    raise ValueError('unknown recording group escaped quarantine')
                if group in group_splits and group_splits[group] != row['split']:
                    raise ValueError('a recording group leaks across dataset partitions')
                group_splits[group] = row['split']
        if verify_files:
            for alias in row['aliases']:
                if digest(alias['path']) != row['sha256']:
                    raise ValueError('a source image changed after freeze')
        entries[row['id']] = row; seen_sha[row['sha256']] = row['id']
    if not isinstance(annotations, dict):
        raise ValueError('annotations must be an object keyed by frame ID')
    if 'annotations' in annotations:
        if annotations.get('manifest_sha256') != expected:
            raise ValueError('annotation manifest hash does not match')
        annotations = annotations['annotations']
    if not isinstance(annotations, dict):
        raise ValueError('annotations must be an object keyed by frame ID')
    for ident, annotation in annotations.items():
        if ident not in entries or not isinstance(annotation, dict):
            raise ValueError('annotation refers to an unknown frame')
        for name, expected_value in [('id', ident), ('sha256', entries[ident]['sha256']),
                                     ('manifest_sha256', expected), ('group', entries[ident]['group']),
                                     ('split', entries[ident]['split'])]:
            if name in annotation and annotation[name] != expected_value:
                raise ValueError('annotation provenance does not match its frozen frame')
        if annotation.get('label') not in ('visible', 'absent', 'uncertain'):
            raise ValueError('annotation needs visible, absent or uncertain')
        if annotation.get('source') not in ('owner', 'review') or not isinstance(annotation.get('reviewer'), str) or not annotation['reviewer'].strip():
            raise ValueError('annotation needs explicit visual-review provenance')
        if annotation['source'] == 'owner' and (annotation.get('attestation') != 'owner-submitted'
                or re.search(r'codex|claude|\bagent\b|model', annotation['reviewer'], re.I)):
            raise ValueError('generated review must not be attributed to the owner')
        boxes = annotation.get('boxes', [])
        if not isinstance(boxes, list) or not all(_box(box) for box in boxes):
            raise ValueError('annotation boxes must be finite normalized visible-part rectangles')
        if annotation['label'] == 'absent' and boxes:
            raise ValueError('absence label conflicts with positive boxes')
    return {'valid': True, 'frames': len(entries), 'reviewed': len(annotations),
            'labels': dict(Counter(r['label'] for r in annotations.values())),
            'partitions': dict(Counter(r['split'] for r in entries.values())),
            'review_provenance': 'operator supplied; owner identity is not authenticated by this offline tool'}, annotations



def load_annotations(path, manifest):
    """Read immutable JSON or an append-only review JSONL; owner edits outrank review labels.

    Invalid/truncated history is an error, never silently skipped. Ordering is
    append order, rather than an untrusted client clock.
    """
    path = Path(path)
    if path.suffix.lower() != '.jsonl':
        return _read_json(path)
    selected = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        annotation = json.loads(line)
        if not isinstance(annotation, dict) or not isinstance(annotation.get('id'), str):
            raise ValueError('JSONL review records need a frame id')
        if annotation.get('manifest_sha256') != manifest['manifest_sha256']:
            raise ValueError('JSONL review belongs to a different manifest')
        ident = annotation['id']
        validate(manifest, {ident: annotation}, verify_files=False)
        old = selected.get(ident)
        rank = {'review': 1, 'owner': 2}
        if old is None or rank[annotation['source']] >= rank[old['source']]:
            selected[ident] = annotation
    return {'manifest_sha256': manifest['manifest_sha256'], 'annotations': selected}


def exports(manifest, annotations, task='localization', verify_files=True):
    if task not in ('localization', 'presence'):
        raise ValueError('export task must be localization or presence')
    audit, labels = validate(manifest, annotations, verify_files)
    result = {split: [] for split in ('train', 'dev', 'test')}
    omitted = Counter()
    for row in manifest['entries']:
        label = labels.get(row['id'])
        if not label:
            omitted['unreviewed'] += 1; continue
        if row['split'] == 'quarantine':
            omitted['quarantine'] += 1; continue
        if label['label'] == 'uncertain':
            omitted['uncertain'] += 1; continue
        if task == 'localization' and label['label'] == 'visible' and not label.get('boxes'):
            omitted['visible_without_box'] += 1; continue
        result[row['split']].append({'id': row['id'], 'path': row['path'], 'sha256': row['sha256'], 'day': row['day'],
                                     'group': row['group'], 'split': row['split'], 'source': label['source'],
                                     'reviewer': label['reviewer'], 'tags': label.get('tags', []),
                                     'label': label['label'], 'boxes': label.get('boxes', []),
                                     'annotation': label})
    counts = {split: dict(Counter(r['label'] for r in rows)) for split, rows in result.items()}
    ready = all(counts[split].get(label, 0) > 0 for split in ('train', 'dev') for label in ('visible', 'absent'))
    files = {split + '.json': {'manifest_sha256': manifest['manifest_sha256'], 'task': task, 'entries': rows} for split, rows in result.items()}
    files['readiness.json'] = {'manifest_sha256': manifest['manifest_sha256'], 'task': task, 'counts': counts,
                               'omitted': dict(omitted), 'training_ready': ready,
                               'independent_test_has_both_classes': all(counts['test'].get(label, 0) > 0 for label in ('visible', 'absent')),
                               'annotation_audit': audit,
                               'claim': 'A data export, not evidence that a model improves detection.'}
    return files


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    mine = commands.add_parser('mine')
    mine.add_argument('--frames', action='append', default=[])
    mine.add_argument('--reports', action='append', default=[])
    mine.add_argument('--source-manifest')
    mine.add_argument('--day-map')
    mine.add_argument('--scores')
    mine.add_argument('--focus-report', action='append', default=[])
    mine.add_argument('--out', default='eval/hard_examples')
    mine.add_argument('--limit', type=int, default=60)
    mine.add_argument('--per-group', type=int, default=4)
    mine.add_argument('--per-source', type=int, default=30)
    mine.add_argument('--review-splits', default='train,dev')
    for name in ('validate', 'export'):
        sub = commands.add_parser(name)
        sub.add_argument('--manifest', required=True)
        sub.add_argument('--annotations', required=True)
        if name == 'export':
            sub.add_argument('--out', required=True)
            sub.add_argument('--task', choices=['localization', 'presence'], default='localization')
    args = parser.parse_args(argv)
    if args.command == 'mine':
        if not (args.frames or args.reports or args.source_manifest):
            raise ValueError('provide at least one frame, report or manifest source')
        records, problems = inputs(args.frames, args.reports, args.source_manifest)
        manifest = build_manifest(records, _read_json(args.day_map) if args.day_map else None,
                                  _read_json(args.scores) if args.scores else None, problems, args.focus_report)
        queue = review_queue(manifest, args.limit, args.per_group, args.per_source, tuple(args.review_splits.split(',')))
        write_snapshots(args.out, {'manifest.json': manifest, 'review_queue.json': queue})
        print(json.dumps({'frames': len(manifest['entries']), 'excluded': len(manifest['excluded']), 'review_queue': len(queue['entries']),
                          'partitions': dict(Counter(r['split'] for r in manifest['entries'])), 'manifest_sha256': manifest['manifest_sha256']}))
    else:
        manifest = _read_json(args.manifest)
        annotations = load_annotations(args.annotations, manifest)
        if args.command == 'validate':
            print(json.dumps(validate(manifest, annotations)[0]))
        else:
            files = exports(manifest, annotations, args.task)
            write_snapshots(args.out, files)
            print(json.dumps(files['readiness.json']))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError) as error:
        raise SystemExit(str(error)) from None
