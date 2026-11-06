"""Bounded NumPy forest for offline learned RGB-obstruction experiments.

Labels enter fit only. Prediction accepts numeric descriptors or RGB pixels;
pose, renderer truth, seeds and detector misses are not prediction inputs.
Leaf frequencies are smoothed scores, not calibrated real-camera risk.
"""
from dataclasses import dataclass
import math

import numpy as np

from .obstruction_scene import image_features


IMAGE_SCHEMA = 'obstruction_scene.image_features:grid4x4:v1'
NUMERIC_SCHEMA = 'numeric_features_v1'
IMAGE_FEATURE_COUNT = 129
MAX_TREES = 64
MAX_DEPTH = 8
MAX_FEATURES = 1024
MAX_TRAIN_ROWS = 200000


def _integer(value, name, minimum, maximum):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f'{name} must be an integer in {minimum}..{maximum}')
    return value


def _matrix(value):
    source = np.asarray(value)
    if source.dtype.kind not in 'biuf':
        raise ValueError('features must be numeric')
    matrix = np.asarray(source, dtype=np.float64)
    if matrix.ndim != 2 or not matrix.size or not np.isfinite(matrix).all():
        raise ValueError('features must be a nonempty finite matrix')
    if matrix.shape[1] > MAX_FEATURES:
        raise ValueError('feature schema exceeds the bounded feature count')
    return matrix


@dataclass(frozen=True)
class _Node:
    feature: int
    threshold: float
    left: int
    right: int
    count: int
    positives: int

    @property
    def probability(self):
        return (self.positives + .5) / (self.count + 1)


@dataclass(frozen=True)
class ForestModel:
    n_features: int
    trees: tuple
    sample_count: int
    seed: int
    max_depth: int
    min_leaf: int
    max_features: int
    n_thresholds: int
    feature_schema: str

    def __post_init__(self):
        _integer(self.n_features, 'n_features', 1, MAX_FEATURES)
        _integer(self.sample_count, 'sample_count', 2, MAX_TRAIN_ROWS)
        _integer(self.seed, 'seed', 0, 2**63 - 1)
        _integer(self.max_depth, 'max_depth', 1, MAX_DEPTH)
        _integer(self.min_leaf, 'min_leaf', 1, MAX_TRAIN_ROWS)
        _integer(self.max_features, 'max_features', 1, self.n_features)
        _integer(self.n_thresholds, 'n_thresholds', 1, 32)
        if self.feature_schema not in (IMAGE_SCHEMA, NUMERIC_SCHEMA):
            raise ValueError('unsupported feature schema')
        if self.feature_schema == IMAGE_SCHEMA and self.n_features != IMAGE_FEATURE_COUNT:
            raise ValueError('image feature schema requires 129 descriptors')
        try:
            trees = tuple(tuple(tree) for tree in self.trees)
        except TypeError:
            raise ValueError('trees must be sequences of nodes') from None
        _integer(len(trees), 'tree count', 1, MAX_TREES)
        for tree in trees:
            if not 1 <= len(tree) <= 2**(self.max_depth + 1) - 1:
                raise ValueError('tree exceeds declared size bound')
            visited = set()
            stack = [(0, 0)]
            while stack:
                index, depth = stack.pop()
                if index in visited or not 0 <= index < len(tree) or depth > self.max_depth:
                    raise ValueError('tree contains invalid links, cycles or excessive depth')
                visited.add(index)
                node = tree[index]
                if not isinstance(node, _Node):
                    raise ValueError('invalid tree node')
                _integer(node.count, 'node count', 1, self.sample_count)
                _integer(node.positives, 'positive count', 0, node.count)
                _integer(node.feature, 'node feature', -1, self.n_features - 1)
                if isinstance(node.threshold, bool) or not isinstance(node.threshold, (int, float)) or not math.isfinite(node.threshold):
                    raise ValueError('node threshold must be finite')
                if node.feature == -1:
                    if node.left != -1 or node.right != -1 or node.threshold != 0:
                        raise ValueError('leaf contains split parameters')
                    continue
                for child in (node.left, node.right):
                    _integer(child, 'child index', index + 1, len(tree) - 1)
                    if not isinstance(tree[child], _Node):
                        raise ValueError('invalid child node')
                left, right = tree[node.left], tree[node.right]
                if min(left.count, right.count) < self.min_leaf or left.count + right.count != node.count or \
                        left.positives + right.positives != node.positives:
                    raise ValueError('inconsistent split sample counts')
                stack.extend(((node.left, depth + 1), (node.right, depth + 1)))
            if len(visited) != len(tree) or tree[0].count != self.sample_count:
                raise ValueError('unreachable nodes or incorrect bootstrap sample count')
        object.__setattr__(self, 'trees', trees)

    def scores(self, features):
        """Return one probability score per numeric feature row."""
        matrix = _matrix(features)
        if matrix.shape[1] != self.n_features:
            raise ValueError('feature schema differs from fitted forest')
        result = np.zeros(len(matrix), dtype=np.float64)
        for tree in self.trees:
            pending = [(0, np.arange(len(matrix)))]
            while pending:
                index, rows = pending.pop()
                node = tree[index]
                if node.feature == -1:
                    result[rows] += node.probability
                else:
                    left = matrix[rows, node.feature] <= node.threshold
                    if left.any():
                        pending.append((node.left, rows[left]))
                    if (~left).any():
                        pending.append((node.right, rows[~left]))
        return result / len(self.trees)

    def predict(self, rgb):
        if self.feature_schema != IMAGE_SCHEMA:
            raise ValueError('RGB prediction requires the image feature schema')
        return float(self.scores(np.asarray([image_features(rgb)]))[0])

    def to_dict(self):
        return {'version': 1, 'model': 'numpy_obstruction_forest', 'feature_schema': self.feature_schema,
                'n_features': self.n_features, 'sample_count': self.sample_count, 'seed': self.seed,
                'max_depth': self.max_depth, 'min_leaf': self.min_leaf,
                'max_features': self.max_features, 'n_thresholds': self.n_thresholds,
                'trees': [[dict(feature=n.feature, threshold=n.threshold, left=n.left, right=n.right,
                                count=n.count, positives=n.positives) for n in tree] for tree in self.trees],
                'score': 'mean Jeffreys-smoothed bootstrap leaf frequency; not calibrated risk',
                'scope': 'offline learned comparison; real-camera calibration unvalidated'}

    @classmethod
    def from_dict(cls, doc):
        if not isinstance(doc, dict) or doc.get('version') != 1 or doc.get('model') != 'numpy_obstruction_forest':
            raise ValueError('unsupported forest model schema')
        try:
            raw = doc['trees']
            if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_TREES:
                raise ValueError('invalid forest tree count')
            if any(not isinstance(t, list) or not 1 <= len(t) <= 2**(MAX_DEPTH + 1) - 1 for t in raw):
                raise ValueError('invalid forest tree size')
            trees = tuple(tuple(_Node(**node) for node in tree) for tree in raw)
            return cls(trees=trees, **{key: doc[key] for key in ('n_features', 'sample_count', 'seed',
                       'max_depth', 'min_leaf', 'max_features', 'n_thresholds', 'feature_schema')})
        except (KeyError, TypeError, OverflowError):
            raise ValueError('invalid serialized forest') from None


def fit(features, labels, seed=0, n_trees=24, max_depth=6, min_leaf=8,
        max_features=None, n_thresholds=12, feature_schema=None):
    """Bootstrap randomized trees; all tuning/threshold selection belongs on development data.

    A split minimizes weighted binary Gini impurity among random feature
    subsets and fixed quantiles. Zero-gain splits are allowed to learn feature
    interactions such as XOR. Every tree and node has an explicit size bound.
    """
    matrix = _matrix(features)
    n, d = matrix.shape
    _integer(n, 'training rows', 2, MAX_TRAIN_ROWS)
    source_labels = np.asarray(labels)
    if source_labels.dtype.kind not in 'biuf' or source_labels.shape != (n,) or \
            not np.isin(source_labels, (0, 1)).all() or len(np.unique(source_labels)) != 2:
        raise ValueError('training needs both binary classes and one label per image')
    y = source_labels.astype(np.int64)
    _integer(seed, 'seed', 0, 2**63 - 1)
    _integer(n_trees, 'n_trees', 1, MAX_TREES)
    _integer(max_depth, 'max_depth', 1, MAX_DEPTH)
    _integer(min_leaf, 'min_leaf', 1, MAX_TRAIN_ROWS)
    _integer(n_thresholds, 'n_thresholds', 1, 32)
    m = max(1, int(math.sqrt(d))) if max_features is None else _integer(max_features, 'max_features', 1, d)
    schema = (IMAGE_SCHEMA if d == IMAGE_FEATURE_COUNT else NUMERIC_SCHEMA) if feature_schema is None else feature_schema
    if schema not in (IMAGE_SCHEMA, NUMERIC_SCHEMA) or schema == IMAGE_SCHEMA and d != IMAGE_FEATURE_COUNT:
        raise ValueError('unsupported or inconsistent feature schema')
    rng = np.random.default_rng(seed)
    quantiles = np.linspace(0, 1, n_thresholds + 2)[1:-1]
    trees = []
    for _ in range(n_trees):
        nodes = []

        def build(rows, depth):
            count, positives = len(rows), int(y[rows].sum())
            index = len(nodes)
            nodes.append(_Node(-1, 0.0, -1, -1, count, positives))
            if depth == max_depth or count < 2 * min_leaf or positives in (0, count):
                return index
            columns = np.sort(rng.choice(d, size=m, replace=False))
            values = matrix[rows][:, columns]
            thresholds = np.quantile(values, quantiles, axis=0)
            best = None
            for j, feature in enumerate(columns):
                left = values[:, j, None] <= thresholds[:, j]
                nl = left.sum(axis=0)
                valid = (nl >= min_leaf) & (count - nl >= min_leaf)
                if not valid.any():
                    continue
                lp = (left * y[rows, None]).sum(axis=0)
                nr, rp = count - nl, positives - lp
                safe_nl, safe_nr = np.maximum(1, nl), np.maximum(1, nr)
                impurity = 2 * (lp - lp * lp / safe_nl + rp - rp * rp / safe_nr) / count
                for k in np.flatnonzero(valid):
                    choice = (float(impurity[k]), int(feature), float(thresholds[k, j]))
                    if best is None or choice < best:
                        best = choice
            if best is not None:
                _, feature, threshold = best
                mask = matrix[rows, feature] <= threshold
                left_index, right_index = build(rows[mask], depth + 1), build(rows[~mask], depth + 1)
                nodes[index] = _Node(feature, threshold, left_index, right_index, count, positives)
            return index

        build(rng.integers(0, n, size=n), 0)
        trees.append(tuple(nodes))
    return ForestModel(d, tuple(trees), n, seed, max_depth, min_leaf, m, n_thresholds, schema)
