"""Private offline search inputs, in the current mount's angular motor coordinates.

This is a planning proxy: a target is the centre of a *floor-bearing view*, not
a measured pet location or a uniformly sampled square metre of floor. Visibility
is a settled angular footprint gated by the saved floor map, not 3-D occlusion,
detector recall, continuous-pan coverage, or a claim of held-out validation.

The all-day histogram in params.json is detector- and incumbent-policy-biased.
It has no mount epoch; by default it receives the grid's approximate pose_shift.
Pass prior_shift=(0, 0) for an explicit unshifted sensitivity comparison. Saved
hourly stops have already been re-aimed and must never receive this shift again.
Nothing in this module reads hardware, changes models, or writes an artifact.
"""
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def _pairs(value, name):
    a = np.asarray(value, dtype=float)
    if a.size == 0:
        return np.empty((0, 2), dtype=float)
    if a.ndim != 2 or a.shape[1] != 2 or not np.isfinite(a).all():
        raise ValueError(f'{name} must be finite [n, 2] coordinates')
    return a


def _pair(value, name, positive=False):
    a = np.asarray(value, dtype=float)
    if a.shape != (2,) or not np.isfinite(a).all() or (positive and np.any(a <= 0)):
        raise ValueError(f'{name} must be a finite {"positive " if positive else ""}pair')
    return tuple(float(x) for x in a)


def _read(path):
    path = Path(path).resolve()
    data = path.read_bytes()
    return json.loads(data), {'path': str(path), 'sha256': hashlib.sha256(data).hexdigest()}


def group_candidates(poses, cell_size):
    """Stable spatial groups of candidate indices; no candidate or mass is lost.

    These groups are a reduction aid, not equivalence classes of visibility or
    route cost. Exact optimization after reducing them is exact only for the
    reduced action set. Motor coordinates have hard limits and do not wrap.
    """
    poses = _pairs(poses, 'poses')
    size = np.asarray(_pair(cell_size, 'cell_size', positive=True))
    groups = {}
    for i, key in enumerate(np.floor(poses / size).astype(int)):
        groups.setdefault(tuple(key), []).append(i)
    return list(groups.values())


@dataclass
class SearchPrior:
    targets: np.ndarray
    weights: np.ndarray
    candidates: np.ndarray
    visibility: np.ndarray
    frame: tuple
    limits: tuple
    provenance: dict
    _grid: dict = field(repr=False)
    _grid_step: tuple = field(repr=False)
    _shift: tuple = field(repr=False)
    _min_floor: float = field(repr=False)
    _stops: dict = field(repr=False)

    def floor_share(self, pan, tilt):
        """Saved floor fraction, or None outside the known shifted grid.

        The reverse shift and nearest-grid rounding match server.roommap.
        This fraction describes the view, not floor at its centre pixel.
        """
        if not np.isfinite([pan, tilt]).all():
            raise ValueError('pose must be finite')
        x = int(round((pan - self._shift[0]) / self._grid_step[0])) * self._grid_step[0]
        y = int(round((tilt - self._shift[1]) / self._grid_step[1])) * self._grid_step[1]
        return self._grid.get((x, y))

    def visibility_for(self, poses, frame=None, floor_mask=True):
        """Return [pose, target] settled angular-footprint visibility.

        Unknown/low-floor views are excluded when floor_mask is true. This is
        deliberately stricter than the live server's unknown-map fallback.
        Frame dimensions may be varied for measured-FOV sensitivity checks.
        """
        poses = _pairs(poses, 'poses')
        fw, fh = self.frame if frame is None else _pair(frame, 'frame', positive=True)
        delta = np.abs(poses[:, None, :] - self.targets[None, :, :])
        visible = (delta[:, :, 0] <= fw / 2) & (delta[:, :, 1] <= fh / 2)
        allowed = ((poses >= 0) & (poses <= np.asarray(self.limits))).all(axis=1)
        if floor_mask:
            allowed &= np.array([((f := self.floor_share(x, y)) is not None and f >= self._min_floor)
                                 for x, y in poses], dtype=bool)
        return visible & allowed[:, None]

    def current_stops(self, hour):
        """Return fresh [pan, tilt, gain] rows exactly as stored, without shift.

        They are an incumbent route, not the histogram used for target weights.
        No clamping is hidden here; visibility_for rejects out-of-bounds poses.
        """
        return [list(row) for row in self._stops.get(int(hour) % 24, [])]

    def candidate_groups(self, cell_size=None):
        return group_candidates(self.candidates, cell_size or (self.frame[0] / 2, self.frame[1] / 2))


def load_prior(params_path=ROOT / 'sim/params.json', map_path=ROOT / 'models/room_map.json',
               uniform_mix=.3, target_step=None, prior_shift=None):
    """Load a normalized all-day angular proxy plus broad floor-bearing mass.

    All floor-bearing grid poses shifted into physical motor limits are retained
    as actions. Out-of-bounds grid poses are filtered, rather than clamped into
    false duplicate boundary measurements. Optional target_step (scalar or pair
    of motor steps) groups only target states, retaining each group's uniform
    mass; it never removes candidate actions. Histogram mass is assigned to the
    nearest retained target in FOV-normalized motor coordinates.

    Source hashes and all approximations are returned in provenance. No date
    split is inferred from either artifact, including map.exclude_day.
    """
    if isinstance(uniform_mix, bool) or not np.isfinite(uniform_mix) or not 0 <= uniform_mix <= 1:
        raise ValueError('uniform_mix must be between zero and one')
    params, psource = _read(params_path)
    room_map, msource = _read(map_path)
    if room_map.get('version') != 1:
        raise ValueError('room map version must be 1')
    geo = params['geometry']
    limits = _pair((geo['pan_max'], geo['tilt_max']), 'limits', positive=True)
    frame = _pair((geo['pan_frame'], geo['tilt_frame']), 'frame', positive=True)
    step = _pair((room_map['pan_step'], room_map['tilt_step']), 'grid step', positive=True)
    shift = _pair(room_map.get('pose_shift') or (0, 0), 'pose_shift')
    histogram_shift = shift if prior_shift is None else _pair(prior_shift, 'prior_shift')
    minimum = float(room_map.get('min_floor', .02))
    if not np.isfinite(minimum) or not 0 <= minimum <= 1:
        raise ValueError('min_floor must be between zero and one')
    grid, candidates = {}, []
    floor_count = rejected = 0
    for row in room_map['poses']:
        x, y = _pair((row['pan'], row['tilt']), 'grid pose')
        fraction = float(row['floor_share'])
        if not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError('floor_share must be between zero and one')
        if (x, y) in grid:
            raise ValueError('duplicate room-map grid pose')
        grid[x, y] = fraction
        if fraction >= minimum:
            floor_count += 1
            q = (x + shift[0], y + shift[1])
            if all(0 <= q[i] <= limits[i] for i in (0, 1)):
                candidates.append(q)
            else:
                rejected += 1
    candidates = _pairs(candidates, 'candidates')
    if not len(candidates):
        raise ValueError('room map has no floor-bearing candidate inside motor limits')

    target_spacing = None
    if target_step is None:
        groups = [[i] for i in range(len(candidates))]
    else:
        target_spacing = _pair((target_step, target_step) if np.isscalar(target_step) else target_step,
                               'target_step', positive=True)
        groups = group_candidates(candidates, target_spacing)
    indices = []
    for group in groups:
        points = candidates[group]
        dist = np.sum(((points - points.mean(axis=0)) / np.asarray(frame)) ** 2, axis=1)
        indices.append(group[int(np.argmin(dist))])
    targets = candidates[indices].copy()
    uniform = np.array([len(group) for group in groups], dtype=float) / len(candidates)

    histogram = np.asarray(params['pet']['room_map'], dtype=float)
    if histogram.size == 0:
        histogram = np.empty((0, 3))
    if histogram.ndim != 2 or histogram.shape[1] != 3 or not np.isfinite(histogram).all() or np.any(histogram[:, 2] < 0):
        raise ValueError('pet.room_map must have finite [pan, tilt, nonnegative mass] rows')
    empirical = np.zeros(len(targets), dtype=float)
    snap_distances = []
    for x, y, mass in histogram:
        point = np.array((x + histogram_shift[0], y + histogram_shift[1]))
        distance = np.linalg.norm((targets - point) / np.asarray(frame), axis=1)
        nearest = int(np.argmin(distance))
        empirical[nearest] += mass
        if mass:
            snap_distances.append(float(distance[nearest]))
    histogram_mass = float(empirical.sum())
    if histogram_mass:
        empirical /= histogram_mass
        weights = (1 - uniform_mix) * empirical + uniform_mix * uniform
    else:
        weights = uniform.copy()
    weights /= weights.sum()

    stops = {int(h): [list(row) for row in rows] for h, rows in room_map.get('stops', {}).items()}
    provenance = {
        'sources': {'params': psource, 'room_map': msource},
        'prior_kind': 'all_day_detector_biased_angular_histogram_plus_floor_bearing_grid_mass',
        'target_kind': 'floor_bearing_view_centres_in_motor_steps',
        'uniform_mix_requested': float(uniform_mix), 'uniform_mix_effective': float(uniform_mix if histogram_mass else 1),
        'uniform_measure': 'equal mass per retained floor-bearing angular grid pose; not uniform physical floor area',
        'histogram_source_mass': histogram_mass, 'histogram_points': len(histogram),
        'histogram_assignment': 'nearest retained target in FOV-normalized motor coordinates',
        'histogram_max_snap_fov_units': max(snap_distances, default=0),
        'histogram_shift': list(histogram_shift),
        'histogram_mount_alignment': 'assumes same pre-remount frame as grid' if prior_shift is None else 'explicit caller override',
        'grid_pose_shift': list(shift), 'stored_stops_shifted_again': False,
        'floor_bearing_grid_poses': floor_count, 'out_of_bounds_candidates_filtered': rejected,
        'candidate_count': len(candidates), 'target_count': len(targets),
        'target_step': list(target_spacing) if target_spacing else None,
        'visibility_model': 'settled rectangular angular footprint AND known floor-bearing view; no 3D occlusion',
        'unknown_floor_policy': 'exclude', 'map_built': room_map.get('built'),
        'map_exclude_day_metadata': room_map.get('exclude_day'), 'map_calib_sha': room_map.get('calib_sha'),
        'held_out_claim': False, 'real_world_validated': False,
        'limitations': [
            'The source histogram is detector-, visibility-, and incumbent-policy-biased and has no date or mount pin.',
            'Grid floor fraction does not establish a target at the view centre or accessible physical floor.',
            'pose_shift is an approximate mount correction; saved stops use their existing re-aimed coordinates.',
            'Angular footprints do not model room occlusion, body extent, detector errors, frame lag, or motion.',
        ],
    }
    result = SearchPrior(targets, weights, candidates, np.empty((0, len(targets)), dtype=bool),
                         frame, limits, provenance, grid, step, shift, minimum, stops)
    result.visibility = result.visibility_for(candidates)
    for array in (result.targets, result.weights, result.candidates, result.visibility):
        array.setflags(write=False)
    return result
