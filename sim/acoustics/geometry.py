"""Rigid camera acoustics and exact single-plane paths, in metres.

World axes: x right, y up, z forward at the reference pan. Positive yaw turns
toward +x; positive pitch raises the head toward +y. The pan origin is fixed in
the world. Offsets are acoustic centres, not grill edges or voxel locations.

The defaults below are assumptions based on the reference camera's approximate
shell geometry. They are not measured transducer calibration. In particular,
motor coordinates, mount roll/sag and acoustic-centre offsets need independent
calibration before using phase to infer room geometry.
"""
from dataclasses import dataclass
import math
from typing import Mapping

import numpy as np


def _vector(value, name):
    a = np.asarray(value, dtype=float)
    if a.shape != (3,) or not np.isfinite(a).all():
        raise ValueError(f"{name} must be a finite vector of length 3")
    return a


def _normal(value):
    n = _vector(value, "unit_normal")
    if not np.isclose(np.linalg.norm(n), 1., atol=1e-9, rtol=1e-9):
        raise ValueError("unit_normal must have unit length")
    return n


def _ry(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def _head_pitch(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[1., 0., 0.], [0., c, s], [0., -s, c]])


def angles_from_motor(calib: Mapping, pan, tilt):
    """Return (yaw, upward pitch) in radians using the optical calibration.

    ``pan_zero`` defaults to the project's reference motor coordinate, 1850.
    ``roll_deg`` is deliberately not applied: the optical residual roll does
    not by itself specify the fixed acoustic rig's mounting transform.
    """
    yaw = (float(calib.get("pan_zero", 1850.)) - float(pan)) * float(calib["deg_per_step_pan"])
    pitch = (float(tilt) - float(calib["level_tilt"])) * float(calib["deg_per_step_tilt"])
    if not np.isfinite([yaw, pitch]).all():
        raise ValueError("motor coordinates and calibration angles must be finite")
    return math.radians(yaw), math.radians(pitch)


@dataclass(frozen=True)
class RigPose:
    source_m: np.ndarray
    receiver_m: np.ndarray
    lens_m: np.ndarray
    pan_origin_m: np.ndarray
    tilt_origin_m: np.ndarray
    source_axis: np.ndarray
    receiver_axis: np.ndarray
    pan_axis: np.ndarray
    tilt_axis: np.ndarray
    pan_rotation: np.ndarray
    head_rotation: np.ndarray


@dataclass(frozen=True)
class Rig:
    """Fixed-pivot two-axis rig, with configurable assumed acoustic centres.

    ``pan_origin_m`` is fixed in world coordinates. ``tilt_offset_m`` and
    ``speaker_offset_m`` are in the pan-body frame, relative to that origin.
    Mic and lens offsets are in the tilting-head frame, relative to its pivot.
    The default lens is at world zero when yaw=pitch=0.
    """
    pan_origin_m: tuple = (0., 0., -0.029)
    tilt_offset_m: tuple = (0., 0., 0.)
    speaker_offset_m: tuple = (0., 0.048, -0.048)
    mic_offset_m: tuple = (0., 0.012, 0.0404)
    lens_offset_m: tuple = (0., 0., 0.029)

    def __post_init__(self):
        for name in ("pan_origin_m", "tilt_offset_m", "speaker_offset_m", "mic_offset_m", "lens_offset_m"):
            # Store immutable numbers even if a caller supplied a mutable list.
            object.__setattr__(self, name, tuple(_vector(getattr(self, name), name)))

    def pose(self, yaw_rad, pitch_rad):
        if not np.isfinite([yaw_rad, pitch_rad]).all():
            raise ValueError("yaw and pitch must be finite")
        pan_rotation = _ry(float(yaw_rad))
        head_rotation = pan_rotation @ _head_pitch(float(pitch_rad))
        pan = np.asarray(self.pan_origin_m)
        tilt = pan + pan_rotation @ np.asarray(self.tilt_offset_m)
        return RigPose(
            source_m=pan + pan_rotation @ np.asarray(self.speaker_offset_m),
            receiver_m=tilt + head_rotation @ np.asarray(self.mic_offset_m),
            lens_m=tilt + head_rotation @ np.asarray(self.lens_offset_m),
            pan_origin_m=pan.copy(), tilt_origin_m=tilt,
            source_axis=-pan_rotation[:, 2], receiver_axis=head_rotation[:, 2],
            pan_axis=np.array([0., 1., 0.]), tilt_axis=pan_rotation[:, 0],
            pan_rotation=pan_rotation, head_rotation=head_rotation,
        )

    def pose_motor(self, calib: Mapping, pan, tilt):
        return self.pose(*angles_from_motor(calib, pan, tilt))


@dataclass(frozen=True)
class PlanePaths:
    direct_m: np.ndarray
    reflected_m: np.ndarray
    valid: np.ndarray
    reflection_points_m: np.ndarray


def one_surface_paths(source, receiver, unit_normal, distance_m):
    """Exact specular paths for the infinite plane ``n dot x = distance_m``.

    Source/receiver arrays of shape (..., 3) broadcast. A reflection is valid
    only when both endpoints are strictly on the same side of the plane. Its
    length is the distance from the mirrored source to the receiver; the
    specular point is also returned. Invalid reflected lengths/points are NaN.
    Reversing both n and d describes the same plane. Visibility, finite patch
    extent, absorption, diffraction and multiple reflections are not modeled.
    """
    n = _normal(unit_normal)
    d = float(distance_m)
    if not np.isfinite(d):
        raise ValueError("distance_m must be finite")
    s, r = np.asarray(source, float), np.asarray(receiver, float)
    if s.ndim < 1 or r.ndim < 1 or s.shape[-1] != 3 or r.shape[-1] != 3:
        raise ValueError("source and receiver must have shape (..., 3)")
    s, r = np.broadcast_arrays(s, r)
    a, b = s @ n - d, r @ n - d
    finite = np.isfinite(s).all(-1) & np.isfinite(r).all(-1)
    valid = finite & (((a > 1e-12) & (b > 1e-12)) | ((a < -1e-12) & (b < -1e-12)))
    image = s - 2 * a[..., None] * n
    fraction = np.divide(a, a + b, out=np.zeros_like(a), where=valid)
    point = image + fraction[..., None] * (r - image)
    return PlanePaths(
        direct_m=np.linalg.norm(r - s, axis=-1),
        reflected_m=np.where(valid, np.linalg.norm(r - image, axis=-1), np.nan),
        valid=valid,
        reflection_points_m=np.where(valid[..., None], point, np.nan),
    )


def ray_plane_ranges(directions, unit_normal, distance_m, origin_m=(0., 0., 0.)):
    """Forward ranges from an origin to an infinite plane, NaN behind/parallel.

    Rays need not be normalized. Normal orientation is arbitrary: (n,d) and
    (-n,-d) have identical intersections. No front-facing-normal assumption is
    made. Rays starting on the plane are treated as invalid.
    """
    n = _normal(unit_normal)
    origin = _vector(origin_m, "origin_m")
    d = float(distance_m)
    if not np.isfinite(d):
        raise ValueError("distance_m must be finite")
    rays = np.asarray(directions, float)
    if rays.ndim < 1 or rays.shape[-1] != 3:
        raise ValueError("directions must have shape (..., 3)")
    den = rays @ n
    t = np.divide(d - origin @ n, den, out=np.full_like(den, np.nan), where=np.abs(den) > 1e-12)
    valid = np.isfinite(rays).all(-1) & (t > 1e-12)
    return np.where(valid, t * np.linalg.norm(rays, axis=-1), np.nan)
