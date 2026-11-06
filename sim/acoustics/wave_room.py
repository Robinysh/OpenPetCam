"""Explicit voxel completion of an uncertain room prior for offline wave tests.

Disconnected saved segments do not define an enclosed footprint. The adapter
adds a declared rigid rectangular box, floor/ceiling, full-height segment walls
and uniform volumetric loss. These are assumptions, not surveyed geometry.
"""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path

import numpy as np

from .wave3d import Grid, _stencils


@dataclass
class VoxelRoom:
    grid: Grid
    metadata: dict


def build_room(path, *, spacing_m=.125, padding_m=.8, sound_speed=343., damping=3.,
               wall_shift_m=0., floor_shift_m=0., ceiling_shift_m=0., seed=0,
               wall_thickness_m=.12):
    values = [spacing_m, padding_m, sound_speed, damping, wall_shift_m,
              floor_shift_m, ceiling_shift_m, wall_thickness_m]
    if (any(isinstance(v, (bool, np.bool_)) for v in values) or not np.isfinite(values).all()
            or min(spacing_m, padding_m, sound_speed, wall_thickness_m) <= 0
            or min(damping, wall_shift_m) < 0):
        raise ValueError('invalid bounded voxel-room parameters')
    path = Path(path)
    raw = path.read_bytes()
    room = json.loads(raw)
    room = room.get('room', room)
    walls = room.get('walls', [])
    if not walls:
        raise ValueError('finite wall segments required')
    ends = np.asarray([[w['a'], w['b']] for w in walls], dtype=float)
    if ends.shape != (len(walls), 2, 2) or not np.isfinite(ends).all():
        raise ValueError('wall endpoints must be finite x,z pairs')
    declared_floor, declared_ceiling = room['floor_height_m'], room.get('ceiling_m', 2.6)
    if (isinstance(declared_floor, bool) or isinstance(declared_ceiling, bool)
            or not np.isfinite([declared_floor, declared_ceiling]).all()
            or not 0 < declared_floor < declared_ceiling):
        raise ValueError('positive lens height below declared ceiling required')
    floor = -float(declared_floor) + floor_shift_m
    ceiling = float(declared_ceiling) - float(declared_floor) + ceiling_shift_m
    if not np.isfinite([floor, ceiling]).all() or ceiling <= floor:
        raise ValueError('invalid floor/ceiling')
    lo = np.floor((ends.min((0, 1)) - padding_m) / spacing_m) * spacing_m
    hi = ends.max((0, 1)) + padding_m
    hn = np.ceil((hi - lo) / spacing_m).astype(int)
    shape = (int(hn[0]), int(np.ceil((ceiling - floor) / spacing_m)), int(hn[1]))
    if min(shape) < 4 or np.prod(shape) > 2_000_000:
        raise ValueError('voxel completion exceeds grid bounds')
    origin = (float(lo[0] + spacing_m / 2), floor + spacing_m / 2, float(lo[1] + spacing_m / 2))
    X, Z = np.meshgrid(origin[0] + np.arange(shape[0]) * spacing_m,
                       origin[2] + np.arange(shape[2]) * spacing_m, indexing='ij')
    solid = np.zeros(shape, dtype=bool)
    rng, realized = np.random.default_rng(seed), []
    radius = wall_thickness_m / 2 + spacing_m / np.sqrt(2)
    for index, segment in enumerate(ends):
        a, b = segment.copy()
        delta = b - a
        length = float(np.linalg.norm(delta))
        if length <= 0:
            raise ValueError('degenerate wall')
        offset = float(rng.uniform(-wall_shift_m, wall_shift_m)) if wall_shift_m else 0.
        normal = np.array([-delta[1], delta[0]]) / length
        a, b = a + offset * normal, b + offset * normal
        u = np.clip(((X - a[0]) * delta[0] + (Z - a[1]) * delta[1]) / length**2, 0, 1)
        mask = np.hypot(X - a[0] - u * delta[0], Z - a[1] - u * delta[1]) <= radius
        solid |= mask[:, None, :]
        realized.append({'index': index, 'a_xz_m': a.tolist(), 'b_xz_m': b.tolist(),
                         'normal_offset_m': offset, 'full_height_extrusion_assumed': True,
                         'horizontal_voxels': int(mask.sum())})
    grid = Grid(origin_m=origin, shape_xyz=shape, spacing_m=spacing_m,
                sound_speed_m_s=sound_speed, courant=.5, solid=solid,
                sigma=np.full(shape, damping, dtype=float))
    metadata = {'source_path': str(path), 'source_sha256': hashlib.sha256(raw).hexdigest(),
                'requested_floor_y_m': floor, 'requested_ceiling_y_m': ceiling,
                'realized_rigid_bounds_xyz_m': [[origin[i] - spacing_m / 2, origin[i] + (shape[i] - .5) * spacing_m] for i in range(3)],
                'shape_xyz': shape, 'spacing_m': spacing_m, 'sound_speed_m_s': sound_speed,
                'uniform_damping_s_inv': damping, 'outer_padding_m': padding_m,
                'wall_thickness_assumed_m': wall_thickness_m,
                'voxel_intersection_radius_m': radius, 'wall_segments': realized,
                'solid_voxels': int(solid.sum()), 'solid_sha256': hashlib.sha256(solid.tobytes()).hexdigest(),
                'declared_segment_bounds_xz_m': np.stack((ends.min((0, 1)), ends.max((0, 1)))).T.tolist(),
                'physical_room_validated': False,
                'assumptions': ['Artificial rectangular rigid completion; no enclosed/walkable footprint inferred.',
                                'Saved segments extruded floor-to-ceiling, including possible furniture faces.',
                                'Acoustic-derived floor scale and assumed ceiling retained as uncertain priors.',
                                'Uniform volumetric damping is not calibrated wall impedance.',
                                'Housing, microphone directionality, transmission and furnishings omitted.',
                                'Refinement changes stair-step geometry and realized outer faces.']}
    return VoxelRoom(grid, metadata)


def air_point(grid, point):
    point = np.asarray(point, dtype=float)
    if point.shape != (3,) or not np.isfinite(point).all():
        return False
    try:
        _stencils(grid, point[None, :])
        return True
    except ValueError:
        return False


def candidates(voxel_room, *, horizontal_spacing=.5, heights_above_floor=(.35, 1.1)):
    if isinstance(horizontal_spacing, (bool, np.bool_)) or not np.isfinite(horizontal_spacing) or horizontal_spacing <= 0:
        raise ValueError('positive spacing required')
    heights = np.asarray(heights_above_floor)
    if (heights.ndim != 1 or not heights.size or heights.dtype.kind not in 'iuf'
            or any(isinstance(v, (bool, np.bool_)) for v in heights_above_floor)
            or not np.isfinite(heights).all() or np.any(heights <= 0)):
        raise ValueError('heights must be nonempty finite positive nonboolean values')
    bounds, floor = voxel_room.metadata['declared_segment_bounds_xz_m'], voxel_room.metadata['requested_floor_y_m']
    points = [(x, floor + h, z)
              for x in np.arange(np.ceil(bounds[0][0] / horizontal_spacing) * horizontal_spacing, bounds[0][1], horizontal_spacing)
              for z in np.arange(np.ceil(bounds[1][0] / horizontal_spacing) * horizontal_spacing, bounds[1][1], horizontal_spacing)
              for h in heights_above_floor]
    return np.asarray([p for p in points if air_point(voxel_room.grid, p)], dtype=float).reshape(-1, 3)
