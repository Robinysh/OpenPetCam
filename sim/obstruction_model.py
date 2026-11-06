"""Small pixel-feature classifier for offline simulated obstruction experiments.

The model consumes RGB-derived features only. Renderer truth supplies training
labels separately; pose IDs, seeds, depth, object masks, and detector misses
are not prediction inputs. Its scores need held-out threshold selection and
are not evidence of real-camera calibration.
"""
from dataclasses import dataclass

import numpy as np

from .obstruction_scene import image_features


def sigmoid(value):
    return 1 / (1 + np.exp(-np.clip(value, -40, 40)))


def _matrix(value):
    array = np.asarray(value, dtype=float)
    if array.ndim != 2 or not array.size or not np.isfinite(array).all():
        raise ValueError('features must be a nonempty finite matrix')
    return array


@dataclass(frozen=True)
class PixelModel:
    mean: np.ndarray
    scale: np.ndarray
    coefficients: np.ndarray
    bias: float

    def __post_init__(self):
        mean, scale, coefficients = (np.asarray(getattr(self, key), dtype=float)
                                     for key in ('mean', 'scale', 'coefficients'))
        bias = float(self.bias)
        if mean.ndim != 1 or not len(mean) or scale.shape != mean.shape or coefficients.shape != mean.shape:
            raise ValueError('inconsistent pixel model dimensions')
        if not all(np.isfinite(x).all() for x in (mean, scale, coefficients)) or not np.isfinite(bias) or np.any(scale <= 0):
            raise ValueError('pixel model must be finite with positive scales')
        for name, value in zip(('mean', 'scale', 'coefficients'), (mean, scale, coefficients)):
            value = value.copy()
            value.flags.writeable = False
            object.__setattr__(self, name, value)
        object.__setattr__(self, 'bias', bias)

    def scores(self, features):
        features = _matrix(features)
        if features.shape[1] != len(self.mean):
            raise ValueError('feature schema differs from fitted model')
        return sigmoid((features - self.mean) / self.scale @ self.coefficients + self.bias)

    def predict(self, rgb):
        return float(self.scores(np.asarray([image_features(rgb)]))[0])

    def to_dict(self):
        return {'version': 1, 'features': 'obstruction_scene.image_features',
                'mean': self.mean.tolist(), 'scale': self.scale.tolist(),
                'coefficients': self.coefficients.tolist(), 'bias': self.bias,
                'scope': 'simulated RGB only; real-camera calibration unvalidated'}

    @classmethod
    def from_dict(cls, doc):
        if not isinstance(doc, dict) or doc.get('version') != 1 or doc.get('features') != 'obstruction_scene.image_features':
            raise ValueError('unsupported pixel model schema')
        try:
            return cls(*(doc[k] for k in ('mean', 'scale', 'coefficients', 'bias')))
        except (KeyError, TypeError, OverflowError):
            raise ValueError('invalid pixel model') from None


def fit(features, labels, l2=.02, iterations=60):
    features = _matrix(features)
    labels = np.asarray(labels, dtype=float)
    if labels.shape != (len(features),) or not np.isin(labels, [0, 1]).all() or len(np.unique(labels)) != 2:
        raise ValueError('training needs both binary classes and one label per image')
    if not np.isfinite(l2) or l2 <= 0 or type(iterations) is not int or not 1 <= iterations <= 200:
        raise ValueError('positive regularization and 1..200 iterations required')
    mean, scale = features.mean(0), np.maximum(features.std(0), 1e-6)
    X = np.column_stack(((features - mean) / scale, np.ones(len(features))))
    beta = np.zeros(X.shape[1])
    penalty = np.eye(len(beta)) * l2
    penalty[-1, -1] = 1e-8
    for _ in range(iterations):
        prediction = sigmoid(X @ beta)
        gradient = X.T @ (prediction - labels) / len(labels) + penalty @ beta
        variance = np.maximum(prediction * (1 - prediction), 1e-8)
        hessian = X.T @ (variance[:, None] * X) / len(labels) + penalty
        delta = np.linalg.solve(hessian, gradient)
        # Keep Newton steps bounded on highly separable image features.
        beta -= delta / max(1, float(np.linalg.norm(delta)) / 5)
        if np.max(np.abs(delta)) < 1e-7:
            break
    return PixelModel(mean, scale, beta[:-1], float(beta[-1]))


def classifier_metrics(labels, scores, threshold):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=float)
    if labels.ndim != 1 or scores.shape != labels.shape or not len(labels) or not np.isin(labels, [0, 1]).all():
        raise ValueError('metrics need nonempty paired binary labels and scores')
    if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)) or not 0 <= threshold <= 1:
        raise ValueError('scores and threshold must be probabilities')
    positive, predicted = labels == 1, scores >= threshold
    tp, fp = int(np.sum(positive & predicted)), int(np.sum(~positive & predicted))
    fn, tn = int(np.sum(positive & ~predicted)), int(np.sum(~positive & ~predicted))
    return {'n': len(labels), 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'recall': tp / max(1, tp + fn), 'false_positive_rate': fp / max(1, fp + tn),
            'precision': tp / max(1, tp + fp), 'threshold': float(threshold)}


def select_threshold(labels, scores):
    """Choose on development only; zero observed FP does not imply zero risk."""
    labels = np.asarray(labels)
    if len(np.unique(labels)) != 2:
        raise ValueError('threshold selection needs both positive and negative development images')
    candidates = [.70, .80, .90, .95, .975, .99, .995, 1.0]
    metrics = [classifier_metrics(labels, scores, t) for t in candidates]
    safe = [m for m in metrics if m['fp'] == 0]
    if not safe:
        raise ValueError('no threshold has zero development false positives; disable the classifier')
    selected = min(safe, key=lambda m: (-m['tp'], -m['threshold']))
    return {'threshold': selected['threshold'], 'development': selected, 'candidates': metrics,
            'rule': 'Maximize development true positives with zero observed false positives; tie chooses stricter threshold.',
            'real_camera_calibrated': False}


def pixel_quality(rgb):
    image = np.asarray(rgb, dtype=float)
    if (image.ndim != 3 or image.shape[2] != 3 or not image.size or not np.isfinite(image).all()
            or np.any(image < 0) or np.any(image > 255)):
        return 0.0
    # Fail open in unusable darkness/clipping. Texture is not a validity test:
    # a sharp uniform wall and a smooth obstruction can both have low texture.
    median = float(np.median(image))
    return 0.0 if median < 8 or median > 249 else 1.0
