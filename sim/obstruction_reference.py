"""Offline learned RGB-pair evidence with an explicitly registered reference.

The reference is a preacquired image of the same camera view. A known-clear
reference is an assumption of the synthetic experiment, not a fact inferred
about a real household. Neither pixels alone nor this descriptor establish
registration, unobstructedness, permanence, or real-camera calibration.
"""
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from .obstruction_forest import ForestModel, NUMERIC_SCHEMA, fit as fit_forest
from .obstruction_scene import image_features


PAIR_SCHEMA = 'registered_rgb_pair:grid4x4:photometric_v1'
MAX_SHIFT_PX = 2
LAYOUT = (('current_pixel_features', 129), ('reference_global_features', 17),
          ('raw_difference_global', 7), ('normalized_difference_global', 12),
          ('normalized_difference_tiles', 64), ('edge_change_global', 6),
          ('alignment_and_photometric_parameters', 6))
FEATURE_COUNT = sum(count for _, count in LAYOUT)
NORMALIZED_DIFF_SLICE = slice(153, 165)


def feature_contract():
    """Return a detached description of the frozen pixel-only descriptor."""
    return {'schema': PAIR_SCHEMA, 'feature_count': FEATURE_COUNT,
            'layout': [{'name': name, 'count': count} for name, count in LAYOUT],
            'requires_registered_reference': True,
            'rgb': 'same-size uint8 RGB, at least 8x8; caller establishes same pose/mount',
            'alignment': 'integer offsets -2..2 pixels on a common two-pixel interior; minimum normalized residual',
            'photometric': 'per-channel standard-deviation gain clipped to .5..2 and mean offset clipped to -.25...25',
            'difference': 'absolute RGB residual, quantiles/coverage and 4x4 spatial statistics; normalized luminance edge change',
            'reference_assumption': 'preacquired matched view; known clear only by synthetic experiment construction, unverified on real camera',
            'scope': 'limited small-image registration and global exposure correction; no pose/depth/truth/labels in descriptor'}


def _rgb(image):
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 8:
        raise ValueError('pair images must be uint8 RGB of at least 8x8')
    return image.astype(np.float64) / 255.0


def _normalize(current, reference):
    cm, rm = current.mean((0, 1)), reference.mean((0, 1))
    cs, rs = current.std((0, 1)), reference.std((0, 1))
    gain = np.clip(np.divide(rs, cs, out=np.ones(3), where=cs > .01), .5, 2)
    offset = np.clip(rm - gain * cm, -.25, .25)
    return np.clip(current * gain + offset, 0, 1), gain, offset


def _edge(rgb):
    lum = rgb @ np.array([.2126, .7152, .0722])
    return (np.abs(np.diff(lum, axis=0, append=lum[-1:])) +
            np.abs(np.diff(lum, axis=1, append=lum[:, -1:]))) / 2


def pair_features(current, reference, registered=True):
    """Compute pixels only; missing/unregistered references return None.

    The boolean registration gate is supplied by the acquisition pipeline. The
    bounded pixel shift search cannot validate different camera poses or mounts.
    Raw appearance and residuals remain alongside exposure-corrected residuals
    because normalization can also erase a real, spatially uniform obstruction.
    """
    if registered is not True or current is None or reference is None:
        return None
    cur, ref = _rgb(current), _rgb(reference)
    if cur.shape != ref.shape:
        raise ValueError('registered image pairs must have identical dimensions')
    border = MAX_SHIFT_PX
    h, w = cur.shape[:2]
    anchor = ref[border:h-border, border:w-border]
    best = None
    for dy in range(-border, border + 1):
        for dx in range(-border, border + 1):
            shifted = cur[border+dy:h-border+dy, border+dx:w-border+dx]
            normalized, gain, offset = _normalize(shifted, anchor)
            key = (float(np.abs(normalized - anchor).mean()), abs(dx) + abs(dy), dy, dx)
            if best is None or key < best[0]:
                best = key, shifted, normalized, gain, offset
    key, shifted, normalized, gain, offset = best
    raw_abs, norm_abs = np.abs(shifted - anchor), np.abs(normalized - anchor)
    raw, norm = raw_abs.mean(2), norm_abs.mean(2)
    current_edge, reference_edge = _edge(normalized), _edge(anchor)
    edge_change = np.abs(current_edge - reference_edge)
    values = [*image_features(current), *image_features(reference)[:17],
              *raw_abs.mean((0, 1)), *np.percentile(raw, [50, 90, 99]), float((raw > .15).mean()),
              *norm_abs.mean((0, 1)), *np.percentile(norm, [10, 25, 50, 75, 90, 99]),
              *[float((norm > threshold).mean()) for threshold in (.03, .1, .2)]]
    for rows in np.array_split(np.arange(norm.shape[0]), 4):
        for cols in np.array_split(np.arange(norm.shape[1]), 4):
            tile = norm[rows[:, None], cols[None, :]]
            edges = edge_change[rows[:, None], cols[None, :]]
            values.extend((tile.mean(), (tile > .075).mean(), (tile > .2).mean(), edges.mean()))
    values.extend((edge_change.mean(), *np.percentile(edge_change, [50, 90, 99]),
                   current_edge.mean(), reference_edge.mean(), key[3] / border, key[2] / border,
                   gain.mean(), gain.std(), offset.mean(), offset.std()))
    result = np.asarray(values, dtype=np.float64)
    if result.shape != (FEATURE_COUNT,) or not np.isfinite(result).all():
        raise ValueError('RGB-pair feature contract differs from the frozen schema')
    return result


@dataclass(frozen=True)
class ReferenceModel:
    requires_reference: ClassVar[bool] = True
    inner: ForestModel

    def __post_init__(self):
        if not isinstance(self.inner, ForestModel) or self.inner.n_features != FEATURE_COUNT or \
                self.inner.feature_schema != NUMERIC_SCHEMA:
            raise ValueError('reference model requires the frozen paired feature schema')

    def scores(self, features):
        return self.inner.scores(features)

    def predict_pair(self, current, reference, registered=True):
        try:
            features = pair_features(current, reference, registered=registered)
        except (ValueError, TypeError):
            return None
        return None if features is None else float(self.scores([features])[0])

    def predict(self, current):
        """One RGB image cannot satisfy the required reference contract."""
        return None

    def to_dict(self):
        return {'version': 1, 'model': 'reference_obstruction_forest',
                'requires_registered_reference': True, 'pair_contract': feature_contract(),
                'inner_forest': self.inner.to_dict(),
                'scope': 'offline paired-reference candidate; real reference clarity and calibration unvalidated'}

    @classmethod
    def from_dict(cls, doc):
        if not isinstance(doc, dict) or doc.get('version') != 1 or doc.get('model') != 'reference_obstruction_forest' or \
                doc.get('requires_registered_reference') is not True or doc.get('pair_contract') != feature_contract():
            raise ValueError('unsupported registered-reference feature contract')
        try:
            return cls(ForestModel.from_dict(doc['inner_forest']))
        except (KeyError, TypeError):
            raise ValueError('invalid registered-reference model') from None


def fit(features, labels, **forest_options):
    """Fit on paired training descriptors; select model/threshold on development only."""
    source = np.asarray(features)
    if source.ndim != 2 or source.shape[1] != FEATURE_COUNT:
        raise ValueError('training requires the frozen paired RGB feature schema')
    if 'feature_schema' in forest_options:
        raise ValueError('reference feature schema is fixed')
    return ReferenceModel(fit_forest(features, labels, feature_schema=NUMERIC_SCHEMA, **forest_options))
