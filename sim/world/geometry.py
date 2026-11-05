"""Directions from the lens. World frame as in camera.py: y up, z forward (pan 1850), x right; azimuth + to the right."""
import numpy as np


def dir_from(az_deg, el_deg):
    a, e = np.radians(az_deg), np.radians(el_deg)
    return np.stack([np.cos(e) * np.sin(a), np.sin(e) * np.ones_like(a), np.cos(e) * np.cos(a)], -1)


def az_el(d):
    d = np.asarray(d)
    return np.degrees(np.arctan2(d[..., 0], d[..., 2])), np.degrees(np.arcsin(np.clip(d[..., 1], -1, 1)))


def directions(pw, ph):
    """Unit direction of every pixel of a pw x ph equirectangular image (360 x 180 deg, azimuth -180 at the left)."""
    az = (np.arange(pw) + 0.5) / pw * 360 - 180
    el = 90 - (np.arange(ph) + 0.5) / ph * 180
    A, E = np.meshgrid(az, el)
    return dir_from(A, E)


def floor_hit(d, height):
    """Where rays d meet the floor `height` metres below the lens (NaN for rays at or above the horizon)."""
    d = np.asarray(d, float)
    t = np.where(d[..., 1] < -1e-3, height / -np.where(d[..., 1] < -1e-3, d[..., 1], -1), np.nan)
    return d * t[..., None]
