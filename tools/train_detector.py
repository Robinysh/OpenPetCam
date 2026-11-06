#!/usr/bin/env python3
"""Offline supervised proposal experiment; never installs or deploys a model.

Manifests are JSON lists (or {"records": [...]}) of reviewed images with id,
path, sha256, group, day, split, label (visible/absent), normalized visible-part
boxes, source (owner/review), and reviewer. Unknown/uncertain rows are excluded.
An absent crop only labels that crop, not its parent frame; preserve scope/tags.

Train accepts train and development manifests only. Candidate selection and the
score threshold use development labels. A later test evaluation rejects every
training/development image hash, parent frame, group, day and identifier.
Provenance is hash-bound for integrity, not a cryptographic reviewer signature.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 1
IOU_THRESHOLD = 0.5
MAX_IMAGE_BYTES = 40 * 1024 * 1024
UNCERTAIN = {'unknown', 'uncertain', 'unlabeled'}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def text(value):
    return isinstance(value, str) and bool(value.strip())


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def clean_boxes(value):
    if not isinstance(value, list):
        raise ValueError('boxes must be a list of normalized visible-part boxes')
    boxes = []
    for box in value:
        if not isinstance(box, (list, tuple)) or len(box) != 4 or not all(number(v) for v in box):
            raise ValueError('Each box needs four finite coordinates')
        x0, y0, x1, y1 = map(float, box)
        if not 0 <= x0 < x1 <= 1 or not 0 <= y0 < y1 <= 1:
            raise ValueError('Boxes must have positive area inside the image')
        boxes.append([x0, y0, x1, y1])
    return boxes


def load_records(path, expected_split=None):
    path = Path(path).resolve()
    raw = json.loads(path.read_text())
    rows = raw.get('records', raw.get('entries')) if isinstance(raw, dict) else raw
    if not isinstance(rows, list):
        raise ValueError('A manifest must contain a record list')
    records, excluded, seen = [], [], set()
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('Every record must be an object')
        if row.get('label') in UNCERTAIN:
            excluded.append({'id': row.get('id'), 'label': row['label']})
            continue
        for key in ('id', 'path', 'group', 'day', 'split', 'sha256', 'reviewer'):
            if not text(row.get(key)):
                raise ValueError('Reviewed records require ' + key)
        if row['id'] in seen:
            raise ValueError('Duplicate record id: ' + row['id'])
        if row['split'] not in ('train', 'dev', 'test') or expected_split and row['split'] != expected_split:
            raise ValueError('Record split does not match the requested partition')
        if not re.fullmatch(r'\d{4}-\d{2}-\d{2}', row['day']):
            raise ValueError('day must identify the source day as YYYY-MM-DD')
        if row.get('label') not in ('visible', 'absent') or row.get('source') not in ('owner', 'review'):
            raise ValueError('Only explicitly reviewed visible/absent labels are usable')
        if row.get('label_origin') in ('model', 'pseudo', 'teacher', 'automatic'):
            raise ValueError('Automatic pseudo-labels are not reviewed ground truth')
        annotation = row.get('annotation')
        if isinstance(annotation, dict) and annotation.get('source', row['source']) != row['source']:
            raise ValueError('Conflicting annotation provenance')
        if row.get('parent_frame') is not None and not text(row['parent_frame']):
            raise ValueError('parent_frame must be a nonempty source identifier')
        if row.get('scope') is not None and not text(row['scope']):
            raise ValueError('scope must be nonempty text')
        boxes = clean_boxes(row.get('boxes'))
        if bool(boxes) != (row['label'] == 'visible'):
            raise ValueError('Visible records need boxes; absent records must have none')
        if not re.fullmatch(r'[0-9a-f]{64}', row['sha256']):
            raise ValueError('sha256 must be a lowercase SHA-256 digest')
        image = Path(row['path'])
        image = (path.parent / image).resolve() if not image.is_absolute() else image.resolve()
        if not image.is_file() or image.stat().st_size > MAX_IMAGE_BYTES:
            raise ValueError('Reviewed image is missing or exceeds the size bound')
        if file_hash(image) != row['sha256']:
            raise ValueError('Image content changed: ' + row['id'])
        tags = row.get('tags', [])
        if not isinstance(tags, list) or not all(isinstance(tag, str) for tag in tags):
            raise ValueError('tags must be a list of strings')
        records.append({**row, 'path': str(image), 'boxes': boxes, 'tags': tags,
                        'scope': row.get('scope') or 'reviewed_image'})
        seen.add(row['id'])
    if not records:
        raise ValueError('The manifest contains no usable reviewed records')
    return records, {'manifest_sha256': file_hash(path), 'excluded': excluded}


def partition_summary(records):
    fields = ('id', 'sha256', 'day', 'group', 'split', 'label', 'boxes', 'source', 'reviewer',
              'parent_frame', 'scope', 'tags', 'label_origin', 'label_by', 'localizer_box_source')
    members = [{k: row[k] for k in fields if k in row} for row in sorted(records, key=lambda r: r['id'])]
    return {'members': members, 'members_sha256': digest(members), 'count': len(members),
            'visible': sum(row['label'] == 'visible' for row in records),
            'absent': sum(row['label'] == 'absent' for row in records),
            'unique_image_hashes': len({row['sha256'] for row in records}),
            'unique_groups': len({row['group'] for row in records}),
            'unique_days': len({row['day'] for row in records}),
            'scopes': sorted({row['scope'] for row in records})}


def assert_disjoint(left, right, left_name='train', right_name='dev'):
    for key in ('id', 'sha256', 'group', 'day', 'parent_frame'):
        a = {r[key] for r in left if text(r.get(key))}
        b = {r[key] for r in right if text(r.get(key))}
        if a & b:
            raise ValueError(f'{left_name}/{right_name} overlap in {key}: {len(a & b)} shared values')


def require_classes(records, name):
    if {row['label'] for row in records} != {'visible', 'absent'}:
        raise ValueError(name + ' requires both reviewed visible and absent examples')


def iou(a, b):
    overlap = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0]) * (a[3]-a[1]) + (b[2]-b[0]) * (b[3]-b[1]) - overlap
    return overlap / union if union > 0 else 0.0


def matched_boxes(targets, proposals, threshold=IOU_THRESHOLD):
    """One-to-one greedy matching, with deterministic IoU ordering."""
    pairs = sorted(((iou(t, p['box']), ti, pi) for ti, t in enumerate(targets)
                    for pi, p in enumerate(proposals)), reverse=True)
    used_t, used_p = set(), set()
    for overlap, ti, pi in pairs:
        if overlap < threshold:
            break
        if ti not in used_t and pi not in used_p:
            used_t.add(ti); used_p.add(pi)
    return len(used_t)


def select_predictions(proposals, score_threshold):
    selected = []
    for item in proposals:
        if not isinstance(item, dict) or not number(item.get('score')) or not 0 <= item['score'] <= 1:
            raise ValueError('Detector predictions need finite probability scores')
        box = clean_boxes([item.get('box')])[0]
        if item['score'] >= score_threshold:
            selected.append({**item, 'box': box, 'score': float(item['score'])})
    return sorted(selected, key=lambda p: -p['score'])


def score_predictions(records, predictions, score_threshold):
    if len(records) != len(predictions):
        raise ValueError('There must be exactly one prediction list per input image')
    visible = absent = gt = matched = localized = false_alarms = unmatched_visible = 0
    for row, proposals in zip(records, predictions):
        found = select_predictions(proposals, score_threshold)
        if row['label'] == 'visible':
            visible += 1; gt += len(row['boxes'])
            n = matched_boxes(row['boxes'], found); matched += n; localized += n > 0
            unmatched_visible += len(found) - n
        else:
            absent += 1; false_alarms += bool(found)
    return {'visible_images': visible, 'absent_images': absent, 'ground_truth_boxes': gt,
            'matched_boxes_iou_0_5': matched, 'proposal_box_recall_iou_0_5': matched / gt if gt else None,
            'localized_visible_images': localized, 'localized_frame_recall_iou_0_5': localized / visible if visible else None,
            'false_alarm_images': false_alarms, 'false_alarm_rate': false_alarms / absent if absent else None,
            'unmatched_predictions_on_visible_images': unmatched_visible}


def calibrate_threshold(records, predictions, max_false_alarm_images=0):
    require_classes(records, 'Development calibration')
    if type(max_false_alarm_images) is not int or max_false_alarm_images < 0:
        raise ValueError('The development false-alarm allowance must be a nonnegative integer')
    thresholds = {1.000001}  # a declared reject-all point is preferable to fabricated healthy performance
    for proposals in predictions:
        for p in select_predictions(proposals, 0):
            thresholds.add(p['score'])
    best = None
    for threshold in sorted(thresholds):
        metrics = score_predictions(records, predictions, threshold)
        if metrics['false_alarm_images'] > max_false_alarm_images:
            continue
        key = (metrics['matched_boxes_iou_0_5'], metrics['localized_visible_images'],
               -metrics['false_alarm_images'], -metrics['unmatched_predictions_on_visible_images'], threshold)
        if best is None or key > best[0]:
            best = (key, threshold, metrics)
    _, threshold, metrics = best
    return {'threshold': threshold, 'metrics': metrics,
            'max_false_alarm_images': max_false_alarm_images,
            'reject_all': metrics['matched_boxes_iou_0_5'] == 0,
            'method': 'dev-only localized recall at a bounded absent-image false-alarm count; higher threshold breaks ties'}


def build_model(pretrained=True):
    from torchvision.models.detection import SSDLite320_MobileNet_V3_Large_Weights, ssdlite320_mobilenet_v3_large
    weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
    dog_index = weights.meta['categories'].index('dog')
    model = ssdlite320_mobilenet_v3_large(weights=weights if pretrained else None, weights_backbone=None,
                                         num_classes=len(weights.meta['categories']), progress=False)
    return model, dog_index, weights


def image_tensor(path):
    import numpy as np
    import torch
    from PIL import Image
    with Image.open(path) as source:
        image = source.convert('RGB')
    array = np.array(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1), image.size


def predict_model(model, records, dog_index):
    """CPU inference, matching the intended deployment CPU rather than MPS latency."""
    import torch
    model.to('cpu').eval()
    predictions, elapsed = [], []
    with torch.inference_mode():
        for row in records:
            x, (width, height) = image_tensor(row['path'])
            started = time.perf_counter(); answer = model([x])[0]
            elapsed.append((time.perf_counter() - started) * 1000)
            found = []
            for box, score, label in zip(answer['boxes'].tolist(), answer['scores'].tolist(), answer['labels'].tolist()):
                if label != dog_index:
                    continue
                normalized = [max(0., box[0]/width), max(0., box[1]/height), min(1., box[2]/width), min(1., box[3]/height)]
                if normalized[2] > normalized[0] and normalized[3] > normalized[1]:
                    found.append({'box': normalized, 'score': score})
            predictions.append(found)
    return predictions, elapsed


def seal_metadata(metadata):
    return {**metadata, 'metadata_sha256': digest(metadata)}


def load_candidate(path):
    path = Path(path).resolve()
    metadata = json.loads((path/'metadata.json').read_text())
    stored = metadata.pop('metadata_sha256', None)
    if stored != digest(metadata) or metadata.get('schema') != SCHEMA:
        raise ValueError('Candidate metadata integrity check failed')
    if metadata.get('architecture') != 'torchvision.ssdlite320_mobilenet_v3_large':
        raise ValueError('Unknown candidate architecture')
    if file_hash(path/'checkpoint.pt') != metadata.get('checkpoint_sha256'):
        raise ValueError('Frozen candidate weights changed')
    for part in ('train', 'dev'):
        summary = metadata.get('partitions', {}).get(part, {})
        if digest(summary.get('members')) != summary.get('members_sha256'):
            raise ValueError('Candidate training/development membership changed')
    return metadata


def verify_evaluation_partition(records, metadata):
    splits = {r['split'] for r in records}
    if len(splits) != 1 or not splits <= {'dev', 'test'}:
        raise ValueError('Evaluate only a declared development or test partition')
    split = next(iter(splits))
    train = metadata['partitions']['train']['members']; dev = metadata['partitions']['dev']['members']
    assert_disjoint(train, records, 'train', split)
    if split == 'test':
        assert_disjoint(dev, records, 'dev', 'test')
    elif partition_summary(records)['members_sha256'] != metadata['partitions']['dev']['members_sha256']:
        raise ValueError('Development evaluation must use the frozen development records')
    return split


def verify_unchanged(records):
    for row in records:
        if file_hash(row['path']) != row['sha256']:
            raise ValueError('Image changed during the experiment: ' + row['id'])


def evaluation_output(path):
    path = Path(path).resolve()
    if path == ROOT/'models' or (ROOT/'models') in path.parents:
        raise ValueError('Experiment receipts cannot modify the live model directory')
    if path.exists():
        raise ValueError('Refusing to overwrite an evaluation receipt')
    return path


def immutable_output(path):
    path = Path(path).resolve()
    if path == ROOT/'models' or (ROOT/'models') in path.parents:
        raise ValueError('Experiment output cannot modify the live model directory')
    if path.exists():
        raise ValueError('Candidate output must be a new directory; never overwrite a frozen experiment')
    path.mkdir(parents=True)
    return path


def train(args):
    train_rows, train_manifest = load_records(args.train, 'train')
    dev_rows, dev_manifest = load_records(args.dev, 'dev')
    require_classes(train_rows, 'Training'); require_classes(dev_rows, 'Development')
    assert_disjoint(train_rows, dev_rows)
    output = immutable_output(args.out)
    if args.device == 'mps':
        os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
    import torch
    if args.device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS was requested but is unavailable')
    torch.set_num_threads(args.cpu_threads)
    random.seed(args.seed); torch.manual_seed(args.seed)
    model, dog_index, weights = build_model(pretrained=True)
    model.to(args.device)
    for parameter in model.backbone.parameters():
        parameter.requires_grad = False
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=1e-4)
    history, best, best_state = [], None, None
    inputs = {}
    for row in train_rows:
        image, (width, height) = image_tensor(row['path'])
        boxes = torch.tensor(row['boxes'], dtype=torch.float32).reshape(-1, 4)
        boxes = boxes * torch.tensor([width, height, width, height])
        inputs[row['id']] = (image, {'boxes': boxes, 'labels': torch.full((len(boxes),), dog_index, dtype=torch.int64)})
    for epoch in range(1, args.epochs + 1):
        model.train(); model.backbone.eval()
        # Tiny local batches must not rewrite pretrained normalization statistics.
        for module in model.modules():
            if isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                module.eval()
        order = list(train_rows); random.Random(args.seed + epoch).shuffle(order)
        total_loss = 0.0
        for start in range(0, len(order), args.batch_size):
            batch = [inputs[row['id']] for row in order[start:start+args.batch_size]]
            images = [item[0].to(args.device) for item in batch]
            targets = [{key: val.to(args.device) for key, val in item[1].items()} for item in batch]
            optimizer.zero_grad(set_to_none=True)
            losses = model(images, targets); loss = sum(losses.values())
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss; no candidate checkpoint was frozen')
            loss.backward(); torch.nn.utils.clip_grad_norm_(parameters, 5.0); optimizer.step()
            total_loss += float(loss.detach().cpu()) * len(batch)
        # Do not move the optimizer's parameters between devices for validation.
        dev_model, _, _ = build_model(pretrained=False)
        dev_model.load_state_dict({key: val.detach().cpu() for key, val in model.state_dict().items()})
        predictions, latency = predict_model(dev_model, dev_rows, dog_index)
        del dev_model
        calibration = calibrate_threshold(dev_rows, predictions, args.max_false_alarm_images)
        metrics = calibration['metrics']
        score = (metrics['matched_boxes_iou_0_5'], metrics['localized_visible_images'],
                 -metrics['false_alarm_images'], -metrics['unmatched_predictions_on_visible_images'])
        result = {'epoch': epoch, 'mean_train_loss': total_loss / len(order), 'development': calibration,
                  'development_cpu_ms': latency_summary(latency)}
        history.append(result); print(json.dumps(result), file=sys.stderr, flush=True)
        if best is None or score > best['score']:
            best = {'epoch': epoch, 'score': score, 'calibration': calibration}
            best_state = {key: val.detach().cpu().clone() for key, val in model.state_dict().items()}
    verify_unchanged(train_rows + dev_rows)
    checkpoint = output/'checkpoint.pt'; torch.save(best_state, checkpoint)
    metadata = {'schema': SCHEMA, 'architecture': 'torchvision.ssdlite320_mobilenet_v3_large',
                'official_initial_weights': weights.name, 'dog_index': dog_index,
                'categories': weights.meta['categories'], 'class_count': len(weights.meta['categories']),
                'checkpoint_sha256': file_hash(checkpoint), 'trainer_source_sha256': file_hash(__file__),
                'partitions': {'train': partition_summary(train_rows), 'dev': partition_summary(dev_rows)},
                'manifests': {'train': train_manifest, 'dev': dev_manifest},
                'selected_epoch': best['epoch'], 'calibration': best['calibration'], 'history': history,
                'training': {'epochs': args.epochs, 'device': args.device, 'seed': args.seed,
                             'batch_size': args.batch_size, 'learning_rate': args.learning_rate,
                             'backbone_frozen': True, 'normalization_statistics_frozen': True,
                             'torch_version': torch.__version__},
                'scope': 'Offline reviewed-image pilot; crop results do not establish whole-frame or tiny-part recovery.',
                'runtime': 'torchvision eager CPU evaluation; no serving export or live installation',
                'promotion_eligible': False,
                'promotion_blockers': ['No untouched full-frame test evidence is established by training.',
                                       'No serving export, deployment CPU budget or live validation is established.']}
    (output/'metadata.json').write_text(json.dumps(seal_metadata(metadata), indent=2, allow_nan=False)+'\n')
    return {'candidate': str(output), 'selected_epoch': best['epoch'], 'calibration': best['calibration'],
            'installed': False, 'training_records': len(train_rows), 'development_records': len(dev_rows)}


class Verifier:
    """The baseline verifier's crop semantics, without importing the live watcher."""
    def __init__(self, path):
        import numpy as np
        import onnxruntime as ort
        self.np = np
        path = Path(path).resolve(); self.cfg = json.loads(path.read_text())
        self.threshold = float(self.cfg['threshold'])
        mode = self.cfg.get('mode')
        if mode not in ('finetuned', 'logreg', 'zeroshot') or not 0 <= self.threshold <= 1:
            raise ValueError('Unsupported baseline verifier configuration')
        name = self.cfg['model'] if mode == 'finetuned' else self.cfg.get('feature_model', 'mobilenetv2-12.onnx')
        model_path = (path.parent/name).resolve()
        opts = ort.SessionOptions(); opts.intra_op_num_threads = 1; opts.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), opts, providers=['CPUExecutionProvider'])
        self.input = self.session.get_inputs()[0].name
        self.provenance = {'config_sha256': file_hash(path), 'model_sha256': file_hash(model_path), 'mode': mode}

    def prob(self, image, box):
        from PIL import Image
        np = self.np
        w, h = image.size; x0, y0, x1, y1 = box; mx, my = (x1-x0)*.15, (y1-y0)*.15
        crop = image.crop((int(max(0, (x0-mx)*w)), int(max(0, (y0-my)*h)),
                           int(min(w, (x1+mx)*w)), int(min(h, (y1+my)*h)))).convert('RGB')
        if crop.width < 8 or crop.height < 8:
            return 0.0
        x = np.array(crop.resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32) / 255
        x = ((x-np.array([.485,.456,.406], dtype=np.float32)) / np.array([.229,.224,.225], dtype=np.float32)).transpose(2,0,1)[None]
        z = self.session.run(None, {self.input: x})[0].reshape(-1).astype(np.float64)
        if self.cfg['mode'] == 'finetuned':
            return float(1/(1+np.exp(-np.clip(z[0], -700, 700))))
        e = np.exp(z-z.max()); e /= e.sum(); a, b = self.cfg.get('classes', [151,268]); dog = float(e[a:b+1].sum())
        if self.cfg['mode'] == 'zeroshot':
            return dog
        value = float(((z-np.array(self.cfg['mean']))/np.array(self.cfg['std'])) @ np.array(self.cfg['w']) + self.cfg['b'])
        p = float(1/(1+math.exp(-max(-700, min(700, value)))))
        if self.cfg.get('zeroshot_backup') is not None and dog >= self.cfg['zeroshot_backup']:
            p = max(p, self.threshold)
        return p


def latency_summary(values):
    if not values:
        return {'count': 0, 'median_ms': None, 'p90_ms': None}
    ordered = sorted(values)
    return {'count': len(values), 'median_ms': ordered[len(ordered)//2],
            'p90_ms': ordered[min(len(ordered)-1, math.ceil(len(ordered)*.9)-1)],
            'first_ms': values[0], 'includes_first_inference': True}


def add_verified(records, predictions, verifier, threshold):
    from PIL import Image
    out, elapsed = [], []
    for row, proposals in zip(records, predictions):
        selected = select_predictions(proposals, threshold)[:3]
        with Image.open(row['path']) as source:
            image = source.convert('RGB')
        started = time.perf_counter(); accepted = []
        for p in selected:
            probability = verifier.prob(image, p['box'])
            if probability >= verifier.threshold:
                accepted.append({**p, 'score': probability})
        elapsed.append((time.perf_counter()-started)*1000); out.append(accepted)
    return out, elapsed


def evaluate(args):
    output = evaluation_output(args.out)
    records, manifest = load_records(args.data)
    metadata = load_candidate(args.candidate); split = verify_evaluation_partition(records, metadata)
    import torch
    torch.set_num_threads(args.cpu_threads)
    model, dog_index, _weights = build_model(pretrained=False)
    if dog_index != metadata['dog_index']:
        raise ValueError('Official class mapping differs from the frozen candidate')
    state = torch.load(Path(args.candidate)/'checkpoint.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(state)
    predictions, candidate_ms = predict_model(model, records, dog_index)
    threshold = metadata['calibration']['threshold']
    verifier = Verifier(args.baseline_verifier)
    verified, candidate_verifier_ms = add_verified(records, predictions, verifier, threshold)
    sys.path.insert(0, str(ROOT/'server'))
    from detector import Detector
    baseline_source_sha = file_hash(args.baseline_model)
    detector = Detector(args.baseline_model)
    if file_hash(args.baseline_model) != baseline_source_sha:
        raise ValueError('Baseline detector changed while loading')
    dims = detector.sess.get_inputs()[0].shape[-2:]
    shape = tuple(dims) if all(type(v) is int for v in dims) else (256, 416)
    from PIL import Image
    baseline, baseline_ms = [], []
    for row in records:
        with Image.open(row['path']) as source:
            image = source.convert('RGB')
        started = time.perf_counter()
        found = detector.detect(image, min_score=.10, shape=shape)
        baseline_ms.append((time.perf_counter()-started)*1000)
        baseline.append([{'box': d['box'], 'score': d['score']} for d in found if d['label'] != 'person'])
    baseline_verified, baseline_verifier_ms = add_verified(records, baseline, verifier, .10)
    def measures(rows, predicted, calibrated, confirmed):
        return {'proposals': score_predictions(rows, predicted, calibrated),
                'with_unchanged_baseline_verifier': score_predictions(rows, confirmed, 0)}
    comparison = {'candidate': measures(records, predictions, threshold, verified),
                  'baseline': measures(records, baseline, .10, baseline_verified)}
    slices = {}
    for tag in sorted({tag for row in records for tag in row['tags']}):
        indices = [i for i, row in enumerate(records) if tag in row['tags']]
        sub = [records[i] for i in indices]
        slices[tag] = {'candidate': measures(sub, [predictions[i] for i in indices], threshold, [verified[i] for i in indices]),
                       'baseline': measures(sub, [baseline[i] for i in indices], .10, [baseline_verified[i] for i in indices])}
    audit = [{'id': row['id'], 'sha256': row['sha256'], 'group': row['group'], 'day': row['day'],
              'label': row['label'], 'scope': row['scope'], 'targets': row['boxes'],
              'candidate': select_predictions(predictions[i], threshold), 'candidate_verified': verified[i],
              'baseline': baseline[i], 'baseline_verified': baseline_verified[i],
              'candidate_cpu_ms': candidate_ms[i], 'baseline_cpu_ms': baseline_ms[i]} for i,row in enumerate(records)]
    result = {'schema': SCHEMA, 'partition': split, 'independent_of_train_and_dev': split == 'test' and not args.diagnostic,
              'evaluation_role': 'diagnostic' if args.diagnostic else split,
              'development_selected_threshold': threshold, 'candidate_metadata_sha256': metadata_digest(args.candidate),
              'evaluator_source_sha256': file_hash(__file__),
              'checkpoint_sha256': metadata['checkpoint_sha256'], 'baseline_model_sha256': baseline_source_sha,
              'baseline_verifier': verifier.provenance, 'baseline_input_shape': shape,
              'baseline_known_training_day_overlap': sorted({r['day'] for r in records} & set(verifier.cfg.get('report', {}).get('days', []))),
              'baseline_unseen_status': 'not established: the baseline artifact has no per-image training membership',
              'manifest': manifest, 'input_summary': partition_summary(records), 'metrics': comparison, 'slices': slices,
              'latency_cpu': {'candidate_detector': latency_summary(candidate_ms),
                              'candidate_verifier': latency_summary(candidate_verifier_ms),
                              'baseline_detector': latency_summary(baseline_ms),
                              'baseline_verifier': latency_summary(baseline_verifier_ms)},
              'audit': audit, 'scope': 'Metrics describe input images and their reviewed scopes, not unlabeled parent frames.',
              'timing_scope': 'CPU eager detector calls; image decode is excluded. Candidate tensor creation is excluded; baseline preparation is inside its detector call. This is not deployment CPU validation.',
              'installed': False}
    verify_unchanged(records)
    if file_hash(args.baseline_model) != result['baseline_model_sha256']:
        raise ValueError('Baseline detector changed during evaluation')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as receipt:
        receipt.write(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


def metadata_digest(candidate):
    return file_hash(Path(candidate)/'metadata.json')


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    commands = p.add_subparsers(dest='command', required=True)
    t = commands.add_parser('train'); t.add_argument('--train', required=True); t.add_argument('--dev', required=True)
    t.add_argument('--out', required=True); t.add_argument('--epochs', type=int, default=2)
    t.add_argument('--device', choices=('cpu','mps'), default='cpu'); t.add_argument('--batch-size', type=int, default=4)
    t.add_argument('--learning-rate', type=float, default=1e-4); t.add_argument('--seed', type=int, default=7)
    t.add_argument('--max-false-alarm-images', type=int, default=0); t.add_argument('--cpu-threads', type=int, default=1)
    e = commands.add_parser('evaluate'); e.add_argument('--data', required=True); e.add_argument('--candidate', required=True)
    e.add_argument('--baseline-model', required=True); e.add_argument('--baseline-verifier', required=True)
    e.add_argument('--out', required=True); e.add_argument('--cpu-threads', type=int, default=1)
    e.add_argument('--diagnostic', action='store_true', help='Mark selected regression examples as diagnostic, not confirmatory')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.cpu_threads < 1 or args.cpu_threads > 16:
        raise ValueError('cpu-threads must be 1..16')
    if args.command == 'train':
        if args.max_false_alarm_images < 0:
            raise ValueError('The development false-alarm allowance must be nonnegative')
        if not 1 <= args.epochs <= 100 or not 1 <= args.batch_size <= 128 or not 0 < args.learning_rate <= .1:
            raise ValueError('Training epochs, batch size or learning rate is outside the bounded range')
        result = train(args)
    else:
        result = evaluate(args)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        print('train_detector: '+str(error), file=sys.stderr)
        sys.exit(2)
