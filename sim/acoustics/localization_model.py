"""Conditional room-aware forward acoustics, with no fitting or hardware access.

Coordinates are metres: x right, y up, z forward at reference pan1850. Use
``Rig.pose_motor(...).receiver_m`` and ``receiver_axis`` for a moving microphone;
the lens origin is not its acoustic centre. Saved room dimensions and the Rig
defaults are uncertain priors, not calibrated measurements.

This model includes point-source direct sound and finite first-order specular
reflections. It omits diffraction, transmission, diffuse/higher-order tails,
housing effects and frequency-dependent boundary impedance. A synthetic inverse
using this same model is an algorithm check, not independent localization proof.
"""
from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path as FilePath
from typing import Mapping

import numpy as np

from .geometry import one_surface_paths


_EPS = 1e-9


def _scalar(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return value


def _vector(value, name):
    value = np.asarray(value, dtype=float)
    if value.shape != (3,) or not np.isfinite(value).all():
        raise ValueError(f"{name} must be a finite xyz vector")
    return value


def _unit(value, name):
    value = _vector(value, name)
    if not np.isclose(np.linalg.norm(value), 1, rtol=1e-9, atol=1e-9):
        raise ValueError(f"{name} must be a unit vector")
    return value


@dataclass(frozen=True)
class Surface:
    """Two-sided plane, optionally bounded by an ordered convex polygon.

    ``reflection`` is a signed pressure coefficient, not an energy absorption
    coefficient. Default +1 is an explicitly assumed rigid reflector. Every
    surface is opaque to geometric direct rays, even if reflection is zero.
    A zero-reflection wall absorbs rays; it does not become an opening.
    """
    surface_id: str
    point_m: tuple
    normal: tuple
    vertices_m: tuple | None = None
    reflection: float = 1.0
    label: str = ""
    provenance: str = "assumed geometry and pressure reflection coefficient"

    def __post_init__(self):
        if not isinstance(self.surface_id, str) or not self.surface_id.strip():
            raise ValueError("surface_id must be a nonempty string")
        p, n = _vector(self.point_m, "point_m"), _unit(self.normal, "normal")
        coefficient = _scalar(self.reflection, "reflection")
        if abs(coefficient) > 1:
            raise ValueError("passive pressure reflection must lie in [-1, 1]")
        object.__setattr__(self, "point_m", tuple(p))
        object.__setattr__(self, "normal", tuple(n))
        object.__setattr__(self, "reflection", coefficient)
        if self.vertices_m is None:
            return
        vertices = np.asarray(self.vertices_m, float)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 3 or not np.isfinite(vertices).all():
            raise ValueError("vertices_m must contain at least three finite xyz vertices")
        if np.max(np.abs((vertices - p) @ n)) > _EPS:
            raise ValueError("surface vertices must lie in its plane")
        edges = np.roll(vertices, -1, axis=0) - vertices
        if np.any(np.linalg.norm(edges, axis=1) <= _EPS):
            raise ValueError("surface polygon cannot have repeated adjacent vertices")
        area = float(np.sum(np.cross(vertices, np.roll(vertices, -1, axis=0)) @ n))
        if abs(area) <= _EPS:
            raise ValueError("surface polygon must have nonzero area")
        orientation = np.sign(area)
        for v, edge in zip(vertices, edges):
            if np.any(orientation * (np.cross(edge, vertices - v) @ n) < -_EPS):
                raise ValueError("surface polygon must be convex and ordered")
        object.__setattr__(self, "vertices_m", tuple(tuple(v) for v in vertices))

    def contains(self, point):
        """Whether a point lies on the plane and inside its finite patch."""
        q = _vector(point, "point")
        n = np.asarray(self.normal)
        if abs(float((q - self.point_m) @ n)) > 10 * _EPS:
            return False
        if self.vertices_m is None:
            return True
        v = np.asarray(self.vertices_m)
        sides = np.cross(np.roll(v, -1, axis=0) - v, q - v) @ n
        return bool(np.all(sides >= -_EPS) or np.all(sides <= _EPS))

    def blocks_segment(self, a, b):
        """Opaque intersection in the open segment; endpoint contacts are allowed.

        A coplanar segment through a surface is a grazing degeneracy and blocked.
        Its propagation would require boundary/diffraction physics absent here.
        """
        a, b = _vector(a, "segment start"), _vector(b, "segment end")
        n, point = np.asarray(self.normal), np.asarray(self.point_m)
        delta = b - a
        denominator = float(delta @ n)
        distance = float((point - a) @ n)
        if abs(denominator) > _EPS:
            fraction = distance / denominator
            return bool(_EPS < fraction < 1 - _EPS and self.contains(a + fraction * delta))
        if abs(distance) > _EPS:
            return False
        if self.vertices_m is None:
            return True
        # Clip a coplanar line segment against the convex polygon's halfplanes.
        vertices = np.asarray(self.vertices_m)
        edges = np.roll(vertices, -1, axis=0) - vertices
        orientation = np.sign(np.sum(np.cross(vertices, np.roll(vertices, -1, axis=0)) @ n))
        lo, hi = 0., 1.
        for v, edge in zip(vertices, edges):
            offset = float(orientation * np.cross(edge, a - v) @ n)
            slope = float(orientation * np.cross(edge, delta) @ n)
            if abs(slope) <= _EPS:
                if offset < -_EPS:
                    return False
            elif slope > 0:
                lo = max(lo, (-_EPS - offset) / slope)
            else:
                hi = min(hi, (-_EPS - offset) / slope)
        return max(lo, _EPS) < min(hi, 1 - _EPS)


@dataclass(frozen=True)
class RoomModel:
    surfaces: tuple
    declared_bounds_m: dict = field(default_factory=dict)
    declared_wall_extents: tuple = ()
    provenance: dict = field(default_factory=dict)

    def __post_init__(self):
        object.__setattr__(self, "surfaces", tuple(self.surfaces))
        if not all(isinstance(s, Surface) for s in self.surfaces):
            raise ValueError("surfaces must be Surface instances")
        if len({s.surface_id for s in self.surfaces}) != len(self.surfaces):
            raise ValueError("surface identifiers must be unique")

    def visible(self, a, b):
        return not any(surface.blocks_segment(a, b) for surface in self.surfaces)


def load_room(path_or_manifest, reflection_coefficients=None, wall_height_limits=None):
    """Load a saved manifest/room dictionary as an explicitly uncertain prior.

    Wall ``a``/``b`` entries are (x,z) metres, never (x,y). Default wall height
    runs from the declared floor to ceiling, an unmeasured extrusion including
    any furniture segments. ``wall_height_limits`` can override each wall's
    world-y (bottom,top) using its id or label. Reflection keys use ids, labels,
    or ``'*'``; omitted coefficients are assumed rigid +1.

    Floor/ceiling have unknown horizontal extent and are infinite here. No
    footprint, closure, free space, or walkable region is inferred from walls.
    The old acoustic estimate builder, overhang and FDTD are never invoked.
    """
    if isinstance(path_or_manifest, Mapping):
        raw = json.dumps(path_or_manifest, sort_keys=True).encode()
        document = json.loads(raw)
        source = "supplied room dictionary"
    else:
        path = FilePath(path_or_manifest)
        raw = path.read_bytes()
        document = json.loads(raw)
        source = str(path)
    room = document.get("room", document)
    floor_height = _scalar(room["floor_height_m"], "floor_height_m")
    ceiling_height = _scalar(room.get("ceiling_m", 2.6), "ceiling_m")
    if not 0 < floor_height < ceiling_height:
        raise ValueError("require positive lens height below floor-to-ceiling height")
    bottom, top = -floor_height, ceiling_height - floor_height
    coefficients, heights = dict(reflection_coefficients or {}), dict(wall_height_limits or {})
    declared = room.get("walls", [])
    if not isinstance(declared, list):
        raise ValueError("walls must be a list")
    wall_names = set()
    ids = []
    for i, wall in enumerate(declared):
        ident = wall.get("id", f"wall_{i}")
        ids.append(ident)
        wall_names.update((ident, wall.get("label", ident)))
    if set(coefficients) - wall_names - {"floor", "ceiling", "*"} or set(heights) - wall_names:
        raise ValueError("unknown surface in coefficient or height overrides")

    def coefficient(ident, label):
        return coefficients.get(ident, coefficients.get(label, coefficients.get("*", 1.)))

    surfaces = [Surface("floor", (0., bottom, 0.), (0., 1., 0.),
                        reflection=coefficient("floor", "floor"),
                        provenance="declared floor height; infinite horizontal extent; material uncalibrated"),
                Surface("ceiling", (0., top, 0.), (0., -1., 0.),
                        reflection=coefficient("ceiling", "ceiling"),
                        provenance="declared floor-to-ceiling height; infinite extent; material uncalibrated")]
    extents, points = [], []
    for ident, wall in zip(ids, declared):
        label = wall.get("label", ident)
        ends = np.asarray([wall["a"], wall["b"]], dtype=float)
        if ends.shape != (2, 2) or not np.isfinite(ends).all() or np.linalg.norm(ends[1] - ends[0]) <= _EPS:
            raise ValueError("each wall needs two distinct finite (x,z) endpoints")
        limits = heights.get(ident, heights.get(label, (bottom, top)))
        if len(limits) != 2:
            raise ValueError("wall vertical limits must be (bottom,top)")
        low, high = (_scalar(v, "wall vertical limit") for v in limits)
        if high <= low:
            raise ValueError("wall top must exceed bottom")
        a, b = ends
        vertices = ((a[0], low, a[1]), (b[0], low, b[1]),
                    (b[0], high, b[1]), (a[0], high, a[1]))
        direction = b - a
        normal = np.array([-direction[1], 0., direction[0]]) / np.linalg.norm(direction)
        surfaces.append(Surface(ident, vertices[0], tuple(normal), vertices,
                                coefficient(ident, label), label,
                                "saved segment; vertical extent/material assumed unless independently supplied"))
        extents.append({"surface_id": ident, "label": label, "a_xz_m": a.tolist(), "b_xz_m": b.tolist(),
                        "bottom_y_m": low, "top_y_m": high,
                        "height_override_supplied": ident in heights or label in heights,
                        "source": wall.get("source", "saved room prior")})
        points.extend(ends)
    bounds = {"floor_y": bottom, "ceiling_y": top, "is_enclosed_or_walkable_region": False}
    if points:
        points = np.asarray(points)
        bounds.update(x_min=float(points[:, 0].min()), x_max=float(points[:, 0].max()),
                      z_min=float(points[:, 1].min()), z_max=float(points[:, 1].max()))
    return RoomModel(tuple(surfaces), bounds, tuple(extents), {
        "source": source, "sha256": hashlib.sha256(raw).hexdigest(),
        "source_manifest_inputs": document.get("inputs", {}),
        "source_manifest_camera": document.get("camera", {}),
        "original_room_provenance": room.get("provenance", {}),
        "geometry_is_uncertain_prior": True, "floor_horizontal_extent_known": False,
        "ceiling_horizontal_extent_known": False, "ceiling_assumed": room.get("ceiling_assumed", True),
        "default_wall_vertical_extent_assumed": True, "acoustic_materials_calibrated": False,
        "default_pressure_reflection": 1., "reflection_overrides": coefficients,
        "overhang_not_included": True, "legacy_acoustic_fit_is_validation": False,
    })


@dataclass(frozen=True)
class AcousticPath:
    order: int
    length_m: float
    delay_s: float
    gain: float
    points_m: tuple
    surface_ids: tuple = ()


@dataclass(frozen=True)
class PathSet:
    paths: tuple
    direct_distance_m: float
    direct_visible: bool
    sound_speed: float
    timing: str
    metadata: dict

    def __iter__(self):
        return iter(self.paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        return self.paths[index]


def build_paths(model, source_m, receiver_m, receiver_axis=None, mic_directivity=0.,
                sound_speed=343., timing="absolute", max_order=1):
    """Direct and first-order specular paths with segment visibility checks.

    Gain is pressure ``reflection * D / (4*pi*path_length)``. The optional
    uncalibrated first-order microphone pattern is D=(1-a)+a*cos(theta), a in
    [0,1]: omni at0, cardioid at.5, signed dipole at1. Source is omnidirectional.
    Direct-relative timing subtracts the *geometric* direct distance even if
    blocked; it is a conditional reference, never a measured emission onset.
    """
    if not isinstance(model, RoomModel):
        raise ValueError("model must be a RoomModel")
    s, r = _vector(source_m, "source_m"), _vector(receiver_m, "receiver_m")
    speed, directivity = _scalar(sound_speed, "sound_speed"), _scalar(mic_directivity, "mic_directivity")
    if speed <= 0 or not 0 <= directivity <= 1:
        raise ValueError("positive speed and microphone directivity in [0,1] required")
    if (timing not in ("absolute", "direct_relative") or isinstance(max_order, (bool, np.bool_))
            or not isinstance(max_order, (int, np.integer)) or max_order not in (0, 1)):
        raise ValueError("timing must be absolute/direct_relative; supported max_order is0 or1")
    axis = _unit(receiver_axis, "receiver_axis") if receiver_axis is not None else None
    if directivity and axis is None:
        raise ValueError("a directional microphone requires receiver_axis")
    direct = float(np.linalg.norm(r - s))
    if direct <= _EPS:
        raise ValueError("point source and receiver must be distinct")
    reference = direct if timing == "direct_relative" else 0.
    paths = []

    def add(points, coefficient, surface_ids):
        length = sum(float(np.linalg.norm(b - a)) for a, b in zip(points, points[1:]))
        direction = (points[-2] - r) / np.linalg.norm(points[-2] - r)
        pattern = (1 - directivity) + directivity * float(direction @ axis) if axis is not None else 1.
        paths.append(AcousticPath(len(surface_ids), length, max(0., (length - reference) / speed),
                                  coefficient * pattern / (4 * math.pi * length),
                                  tuple(tuple(p) for p in points), tuple(surface_ids)))

    direct_visible = model.visible(s, r)
    if direct_visible:
        add((s, r), 1., ())
    if max_order:
        for surface in model.surfaces:
            if not surface.reflection:
                continue
            exact = one_surface_paths(s, r, surface.normal, float(np.dot(surface.point_m, surface.normal)))
            if not exact.valid:
                continue
            point = exact.reflection_points_m
            if surface.contains(point) and model.visible(s, point) and model.visible(point, r):
                add((s, point, r), surface.reflection, (surface.surface_id,))
    paths.sort(key=lambda p: (p.delay_s, p.order, p.surface_ids))
    return PathSet(tuple(paths), direct, direct_visible, speed, timing, {
        "model": "point_source_direct_and_first_order_finite_specular",
        "timing_reference": "physical emission in simulation only" if timing == "absolute" else
                            "conditional geometric direct path; unknown event emission is not measured",
        "direct_reference_observed": False, "direct_path_blocked": not direct_visible,
        "mic_directivity_calibrated": False, "mic_directivity": directivity,
        "source_directivity": "assumed omnidirectional", "acoustic_centres_calibrated": False,
        "max_order": max_order, "diffraction_or_transmission": False,
        "same_model_synthetic_results_are_independent_validation": False,
    })


def render_transfer(paths, frequency_hz):
    """Exact signed complex path sum; no time rounding or gain normalization."""
    frequency = np.asarray(frequency_hz, float)
    if not np.isfinite(frequency).all():
        raise ValueError("frequencies must be finite")
    response = np.zeros(frequency.shape, complex)
    for path in paths:
        response += path.gain * np.exp(-2j * np.pi * frequency * path.delay_s)
    return response


def render_ir(paths, sample_rate=16000, duration_s=.25, onset_s=.02, half_width=32):
    """Signed windowed-sinc fractional delays with an explicit rendering onset.

    Each interpolation kernel has unit DC gain; path gains are never normalized.
    Reject windows that cut off any kernel, including its negative-time ringing.
    ``render_transfer`` is preferred for exact frequency-domain fitting. A finite
    sinc is an approximation near Nyquist, not additional acoustic bandwidth.
    """
    if isinstance(sample_rate, bool) or int(sample_rate) != sample_rate or sample_rate <= 0:
        raise ValueError("sample_rate must be a positive integer")
    if isinstance(half_width, bool) or int(half_width) != half_width or half_width < 8:
        raise ValueError("half_width must be an integer at least8")
    duration, onset = _scalar(duration_s, "duration_s"), _scalar(onset_s, "onset_s")
    if duration <= 0 or onset < 0:
        raise ValueError("positive duration and nonnegative onset required")
    count = math.ceil(duration * sample_rate)
    response = np.zeros(count, float)
    for path in paths:
        position = (onset + path.delay_s) * sample_rate
        indexes = np.arange(math.ceil(position - half_width), math.floor(position + half_width) + 1)
        if indexes[0] < 0 or indexes[-1] >= count:
            raise ValueError("render window truncates a path kernel; increase onset/duration")
        delta = indexes - position
        window = np.i0(8.6 * np.sqrt(np.maximum(0., 1 - (delta / half_width) ** 2))) / np.i0(8.6)
        kernel = np.sinc(delta) * window
        kernel /= kernel.sum()       # fractional-delay interpolation's DC gain, not observed-data normalization
        response[indexes] += path.gain * kernel
    return response
