"""Offline RGB composites for learning camera-view obstruction evidence.

The camera rays and intersections with near 3-D primitives are geometric.
Backgrounds are schematic spherical textures, or an optional equirectangular
RGB photograph. This is not a room reconstruction or photorealistic light
simulation. Labels describe only injected primitives; a photograph's existing
contents are unlabelled.

``render`` returns only RGB pixels. ``truth`` is a separate evaluator API.
``image_features`` and ``image_quality`` accept pixels only, never scene state.
"""
from dataclasses import dataclass
import hashlib
import math
from typing import Optional, Tuple

import numpy as np

from .world.camera import Camera


NOMINAL_CALIB = dict(focal_px=1700.0, k1=0.0, k2=0.0,
                     deg_per_step_pan=0.086, deg_per_step_tilt=0.082,
                     level_tilt=716.0, roll_deg=0.0)


def _seed(*parts):
    encoded = ':'.join(str(p) for p in parts).encode('utf-8')
    return int.from_bytes(hashlib.blake2b(encoded, digest_size=8).digest(), 'big')


def domain_seeds(split, index, master_seed=20261106):
    """Independent background/object/sensor identities for a scene-group split.

Split whole scene groups using this function before rendering their poses or
times. Reusing one scene in several splits would still be leakage; augmenting
images is not a substitute for distinct background and object identities.
    """
    if not isinstance(split, str) or not split or not isinstance(index, (int, np.integer)) or index < 0:
        raise ValueError('split must be nonempty and index a nonnegative integer')
    return {name: _seed('obstruction-domain-v1', master_seed, split, int(index), name)
            for name in ('seed', 'background_seed', 'object_seed')}


@dataclass(frozen=True)
class Occluder:
    anchor_pose: Tuple[float, float] = (1850.0, 716.0)
    center_uv: Tuple[float, float] = (0.5, 0.5)
    size_uv: Tuple[float, float] = (0.7, 0.8)
    distance_m: float = 0.3
    shape: str = 'box'
    role: str = 'persistent'
    texture: str = 'woven'
    color: Optional[Tuple[float, float, float]] = None
    active_versions: Optional[Tuple[int, ...]] = None
    start_s: float = 0.0
    end_s: Optional[float] = None
    seed: int = 0

    def __post_init__(self):
        for name in ('anchor_pose', 'center_uv', 'size_uv'):
            value = tuple(float(x) for x in getattr(self, name))
            if len(value) != 2 or not all(math.isfinite(x) for x in value):
                raise ValueError(f'{name} must contain two finite numbers')
            if name == 'size_uv' and any(x <= 0 for x in value):
                raise ValueError('size_uv must be positive')
            object.__setattr__(self, name, value)
        if self.shape not in ('box', 'sphere'):
            raise ValueError('shape must be box or sphere')
        if self.role not in ('persistent', 'transient'):
            raise ValueError('role must be persistent or transient')
        if self.texture not in ('flat', 'woven', 'wood', 'speckle'):
            raise ValueError('unsupported material texture')
        if not math.isfinite(self.distance_m) or self.distance_m <= 0:
            raise ValueError('distance_m must be positive')
        if not math.isfinite(self.start_s) or (self.end_s is not None and
                (not math.isfinite(self.end_s) or self.end_s <= self.start_s)):
            raise ValueError('invalid active time interval')
        if self.color is not None:
            c = tuple(float(x) for x in self.color)
            if len(c) != 3 or not all(math.isfinite(x) and 0 <= x <= 255 for x in c):
                raise ValueError('color must be three values in 0..255')
            object.__setattr__(self, 'color', c)
        if self.active_versions is not None:
            object.__setattr__(self, 'active_versions', tuple(self.active_versions))

    def active(self, t, setup_version):
        return (self.start_s <= t and (self.end_s is None or t < self.end_s) and
                (self.active_versions is None or setup_version in self.active_versions))


@dataclass(frozen=True)
class ObstructionTruth:
    blocked: bool
    persistent_fraction: float
    transient_fraction: float
    total_injected_fraction: float
    block_threshold: float
    visible_objects: Tuple[int, ...]
    injected_only: bool = True
    background_labelled_clear: bool = False

    @property
    def blocking(self):
        return self.blocked

    @property
    def coverage(self):
        return self.total_injected_fraction

    @property
    def persistent_coverage(self):
        return self.persistent_fraction

    @property
    def transient_coverage(self):
        return self.transient_fraction


def _rgb(image):
    image = np.asarray(image)
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 4 or image.dtype != np.uint8:
        raise ValueError('expected an H x W x 3 uint8 RGB image, at least 4 x 4')
    return image.astype(np.float32) / 255.0


def image_quality(image):
    """Observed RGB quality, with no access to hidden obstruction labels."""
    rgb = _rgb(image)
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    gradient = (np.abs(np.diff(lum, axis=0)).mean() + np.abs(np.diff(lum, axis=1)).mean()) / 2
    return {'brightness': float(lum.mean()), 'contrast': float(lum.std()),
            'gradient': float(gradient)}


def image_features(image, grid=(4, 4)):
    """A fixed RGB descriptor: global colour/texture plus spatial pooled pixels.

No pose, time, seed, setup version, depth, mask, coverage or label is included.
The descriptor can be computed on an actual decoded RGB camera frame. It does
not itself establish that a near object or a permanent obstruction is present.
Default feature count is 129.
    """
    rgb = _rgb(image)
    gy, gx = grid
    if not isinstance(gy, int) or not isinstance(gx, int) or gy < 1 or gx < 1:
        raise ValueError('grid dimensions must be positive integers')
    if gy > rgb.shape[0] or gx > rgb.shape[1]:
        raise ValueError('grid cannot be larger than image')
    lum = rgb @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    saturation = rgb.max(2) - rgb.min(2)
    dx = np.abs(np.diff(lum, axis=1, append=lum[:, -1:]))
    dy = np.abs(np.diff(lum, axis=0, append=lum[-1:]))
    feature = [*rgb.mean((0, 1)), *rgb.std((0, 1)),
               *np.percentile(lum, [5, 25, 50, 75, 95]),
               saturation.mean(), saturation.std(), dx.mean(), dx.std(), dy.mean(), dy.std()]
    for rows in np.array_split(np.arange(rgb.shape[0]), gy):
        for cols in np.array_split(np.arange(rgb.shape[1]), gx):
            tile = rgb[rows[:, None], cols[None, :]]
            edge = (dx[rows[:, None], cols[None, :]] + dy[rows[:, None], cols[None, :]]) / 2
            feature.extend([*tile.mean((0, 1)), *tile.std((0, 1)), edge.mean()])
    return np.asarray(feature, dtype=np.float64)


def _box_blur(image, radius):
    if radius <= 0:
        return image
    width = 2 * radius + 1
    padded = np.pad(image, ((radius, radius), (radius, radius), (0, 0)), mode='edge')
    summed = np.pad(padded.cumsum(0).cumsum(1), ((1, 0), (1, 0), (0, 0)))
    return (summed[width:, width:] - summed[:-width, width:] - summed[width:, :-width] +
            summed[:-width, :-width]) / (width * width)


class ObstructionScene:
    """A rotation-only camera viewing a background plus injected near objects.

``size_uv`` is nominal object width/height at its anchor view; objects are
ray-intersected in 3-D and naturally crop or disappear at other poses. A sphere
uses the smaller nominal half-extent as its radius. Mount offsets are camera
Euler offsets in degrees (pitch, yaw, roll), matching world.camera.Camera.
Blocking requires at least 85% visible persistent-object coverage by default.
Partial obstruction never automatically means the whole view is excluded.
    """

    scope = 'geometric near primitives on schematic/photo spherical RGB composites; no full room/light physics'

    def __init__(self, seed=0, background_seed=None, object_seed=None, occluders=(),
                 background=None, calib=None, mount_offsets=None, width=160, height=90,
                 low_light=1.0, blur=0, background_kind='room', background_color=None,
                 noise_std=1.0, block_threshold=0.85):
        if not isinstance(width, int) or not isinstance(height, int) or min(width, height) < 8:
            raise ValueError('image dimensions must be integers at least 8')
        if not math.isfinite(low_light) or low_light < 0:
            raise ValueError('low_light must be finite and nonnegative')
        if not isinstance(blur, int) or not 0 <= blur <= 20:
            raise ValueError('blur must be an integer in 0..20')
        if not math.isfinite(noise_std) or noise_std < 0:
            raise ValueError('noise_std must be finite and nonnegative')
        if not math.isfinite(block_threshold) or not 0 < block_threshold <= 1:
            raise ValueError('block_threshold must be in (0, 1]')
        if background_kind not in ('room', 'blank', 'texture'):
            raise ValueError('background_kind must be room, blank or texture')
        self.seed = int(seed)
        self.background_seed = int(seed if background_seed is None else background_seed)
        self.object_seed = int(seed if object_seed is None else object_seed)
        self.occluders = tuple(o if isinstance(o, Occluder) else Occluder(**o) for o in occluders)
        self.width, self.height = width, height
        self.low_light, self.blur, self.noise_std = float(low_light), blur, float(noise_std)
        self.block_threshold = float(block_threshold)
        self.background_kind = background_kind
        self.background_color = None if background_color is None else np.asarray(background_color, dtype=float)
        if self.background_color is not None and (self.background_color.shape != (3,) or
                not np.isfinite(self.background_color).all() or (self.background_color < 0).any() or
                (self.background_color > 255).any()):
            raise ValueError('background_color must be an RGB triple in 0..255')
        if background is not None:
            _rgb(background)
            self.background = np.asarray(background).copy()
        else:
            self.background = None
        self.camera = Camera(dict(NOMINAL_CALIB if calib is None else calib))
        self.mount_offsets = dict(mount_offsets or {})
        self._rays_cache = {}
        self._objects = [self._geometry(o, i) for i, o in enumerate(self.occluders)]

    def _geometry(self, obj, index):
        center_ray = self.camera.rays(*obj.anchor_pose,
                                      np.array(obj.center_uv[0] * self.width),
                                      np.array(obj.center_uv[1] * self.height),
                                      w=self.width, h=self.height)
        rotation = self.camera.R(*obj.anchor_pose)
        half = np.array([obj.distance_m * obj.size_uv[0] * Camera.W / (2 * self.camera.f),
                         obj.distance_m * obj.size_uv[1] * Camera.H / (2 * self.camera.f),
                         max(0.005, obj.distance_m * 0.04)])
        rng = np.random.default_rng(_seed('material', self.object_seed, obj.seed, index))
        color = np.asarray(obj.color if obj.color is not None else rng.uniform(35, 215, 3))
        return {'center': center_ray * obj.distance_m, 'rotation': rotation, 'half': half,
                'color': color, 'phase': rng.uniform(0, 2 * np.pi, 3), 'frequency': rng.uniform(30, 160, 3)}

    def _rays(self, pose, setup_version):
        pose = tuple(float(v) for v in pose)
        if len(pose) != 2 or not np.isfinite(pose).all():
            raise ValueError('pose must be two finite motor coordinates')
        off = tuple(self.mount_offsets.get(setup_version, (0, 0, 0)))
        if len(off) != 3 or not np.isfinite(off).all():
            raise ValueError('mount offset must be three finite angles')
        key = pose, off
        if key not in self._rays_cache:
            yy, xx = np.mgrid[0:self.height, 0:self.width]
            self._rays_cache[key] = self.camera.rays(*pose, xx + 0.5, yy + 0.5,
                                                    w=self.width, h=self.height, off=off)
            if len(self._rays_cache) > 256:
                first = next(iter(self._rays_cache))
                if first != key:
                    del self._rays_cache[first]
        return self._rays_cache[key]

    def _background(self, rays):
        u = (np.arctan2(rays[..., 0], rays[..., 2]) / (2 * np.pi) + 0.5) % 1
        v = 0.5 - np.arcsin(np.clip(rays[..., 1], -1, 1)) / np.pi
        if self.background is not None:
            h, w = self.background.shape[:2]
            return self.background[np.minimum(h - 1, (v * h).astype(int)),
                                   (u * w).astype(int) % w].astype(float)
        rng = np.random.default_rng(self.background_seed)
        wall = self.background_color if self.background_color is not None else rng.uniform(95, 225, 3)
        image = np.broadcast_to(wall, (*u.shape, 3)).copy()
        if self.background_kind == 'blank':
            return image
        phase = rng.uniform(0, 2 * np.pi, 5)
        texture = (np.sin(2 * np.pi * (u * 23 + v * 5) + phase[0]) +
                   np.cos(2 * np.pi * (u * 61 - v * 13) + phase[1]) +
                   np.sin(2 * np.pi * (u * 113 + v * 49) + phase[2]))
        image += texture[..., None] * rng.uniform(2, 11, 3)
        if self.background_kind == 'room':
            floor = rays[..., 1] < -0.12
            ceiling = rays[..., 1] > 0.62
            image[floor] = rng.uniform(45, 150, 3) + texture[floor, None] * 7
            image[ceiling] = rng.uniform(165, 235, 3) + texture[ceiling, None] * 2
            # Far wall/furniture patches are image content, not labelled near
            # obstructions. Broad, dark patches are intentional hard negatives.
            for _ in range(12):
                cu, cv = rng.uniform(0, 1), rng.uniform(0.25, 0.78)
                wu, hv = rng.uniform(0.015, 0.12), rng.uniform(0.03, 0.22)
                du = np.minimum(np.abs(u - cu), 1 - np.abs(u - cu))
                patch = (du < wu) & (np.abs(v - cv) < hv)
                image[patch] = rng.uniform(25, 215, 3) + texture[patch, None] * rng.uniform(1, 12)
        return image

    def _intersections(self, rays, obj, geometry):
        direction = rays @ geometry['rotation']
        origin = -geometry['center'] @ geometry['rotation']
        if obj.shape == 'sphere':
            radius = float(min(geometry['half'][:2]))
            b = np.sum(direction * origin, axis=-1)
            discriminant = b * b - (np.sum(origin * origin) - radius * radius)
            distance = -b - np.sqrt(np.maximum(discriminant, 0))
            hit = (discriminant >= 0) & (distance > 0.015)
        else:
            # A thin oriented box; local rays share a fixed origin. Parallel
            # rays outside a slab cannot intersect it.
            parallel = np.abs(direction) < 1e-12
            safe = np.where(parallel, 1.0, direction)
            a, b = (-geometry['half'] - origin) / safe, (geometry['half'] - origin) / safe
            near, far = np.minimum(a, b), np.maximum(a, b)
            within = np.abs(origin) <= geometry['half']
            near = np.where(parallel & within, -np.inf, near)
            far = np.where(parallel & within, np.inf, far)
            impossible = np.any(parallel & ~within, axis=-1)
            distance = near.max(-1)
            hit = ~impossible & (far.min(-1) >= np.maximum(distance, 0.015)) & (distance > 0.015)
        return np.where(hit, distance, np.inf), direction, origin

    def _layers(self, pose, t, setup_version):
        if not math.isfinite(t):
            raise ValueError('time must be finite')
        rays = self._rays(pose, setup_version)
        depth = np.full((self.height, self.width), np.inf)
        owner = np.full((self.height, self.width), -1, dtype=np.int32)
        local = {}
        for index, (obj, geom) in enumerate(zip(self.occluders, self._objects)):
            if not obj.active(t, setup_version):
                continue
            dist, direction, origin = self._intersections(rays, obj, geom)
            front = dist < depth
            owner[front], depth[front] = index, dist[front]
            local[index] = (direction, origin)
        return rays, owner, depth, local

    def truth(self, pose, t, setup_version=0):
        """Hidden evaluator labels for injected visible objects, not a sensor."""
        _, owner, _, _ = self._layers(pose, t, setup_version)
        persistent = np.zeros(owner.shape, dtype=bool)
        transient = np.zeros(owner.shape, dtype=bool)
        visible = []
        for index, obj in enumerate(self.occluders):
            mask = owner == index
            if mask.any():
                visible.append(index)
                if obj.role == 'persistent':
                    persistent |= mask
                else:
                    transient |= mask
        fraction, temporary = float(persistent.mean()), float(transient.mean())
        return ObstructionTruth(fraction >= self.block_threshold, fraction, temporary,
                                fraction + temporary, self.block_threshold, tuple(visible))

    def render(self, pose, t, setup_version=0):
        """Render RGB only. Hidden masks/coverage are never returned with it."""
        rays, owner, depth, local = self._layers(pose, t, setup_version)
        image = self._background(rays)
        for index, (obj, geom) in enumerate(zip(self.occluders, self._objects)):
            mask = owner == index
            if not mask.any():
                continue
            color = np.broadcast_to(geom['color'], (int(mask.sum()), 3)).copy()
            if obj.texture != 'flat':
                direction, origin = local[index]
                points = origin + direction[mask] * depth[mask, None]
                a = points * geom['frequency'] + geom['phase']
                if obj.texture == 'woven':
                    pattern = np.sin(a[:, 0] * 4) + np.sin(a[:, 1] * 4)
                elif obj.texture == 'wood':
                    pattern = np.sin(a[:, 0] + np.sin(a[:, 1] * 0.2) * 2)
                else:
                    pattern = np.sin(a[:, 0] * 7) * np.cos(a[:, 1] * 9) + np.sin(a[:, 2] * 5)
                color += pattern[:, None] * np.array([9, 8, 7])
            image[mask] = color
        image = _box_blur(image * self.low_light, self.blur)
        if self.noise_std:
            frame = int(math.floor(t * 5 + 1e-8))
            noise = np.random.default_rng(_seed('sensor', self.seed, frame)).normal(0, self.noise_std, image.shape)
            image += noise
        return np.rint(np.clip(image, 0, 255)).astype(np.uint8)
