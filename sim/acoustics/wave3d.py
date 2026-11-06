"""Bounded CPU 3-D scalar wave propagation, preserving signed pressure.

The equation is p_tt + 2*sigma*p_t = c**2*Laplacian(p) + q. A symmetric
seven-neighbour graph supplies rigid Neumann faces at solid/free and outer
interfaces. Diagonal damping is a lossy term, never a PML or calibrated room
impedance. This new engine does not reuse the historical nearfield experiment.
Only an explicitly supplied cache directory receives compilation/run files.
"""
from dataclasses import dataclass
import fcntl
import hashlib
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import struct
import subprocess
import sys
import tempfile
import time

import numpy as np


MAX_CELLS = 8_000_000
MAX_STEPS = 100_000
MAX_RECEIVERS = 10_000
MAX_OUTPUT_SAMPLES = 64_000_000
INPUT_HEADER = struct.Struct('<8sIIIQQQdddI')
OUTPUT_HEADER = struct.Struct('<8sQQQd')
KERNEL = Path(__file__).with_name('wave3d_kernel.cpp')


def _scalar(value, name, positive=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be a finite number')
    value = float(value)
    if not math.isfinite(value) or (value <= 0 if positive else value < 0):
        raise ValueError(f'{name} is out of range')
    return value


def _sha(value):
    return hashlib.sha256(value).hexdigest()


@dataclass(frozen=True)
class Grid:
    origin_m: tuple
    shape_xyz: tuple
    spacing_m: float
    sound_speed_m_s: float = 343.0
    courant: float = 0.5
    solid: object = None
    sigma: object = None

    def __post_init__(self):
        origin = np.asarray(self.origin_m, dtype=float)
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError('origin_m must contain three finite coordinates')
        shape = tuple(self.shape_xyz)
        if len(shape) != 3 or any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n < 3 for n in shape):
            raise ValueError('shape_xyz requires three integer dimensions >=3')
        shape = tuple(int(n) for n in shape)
        if math.prod(shape) > MAX_CELLS:
            raise ValueError('grid exceeds bounded cell count')
        h = _scalar(self.spacing_m, 'spacing_m', positive=True)
        c = _scalar(self.sound_speed_m_s, 'sound_speed_m_s', positive=True)
        courant = _scalar(self.courant, 'courant', positive=True)
        if courant > 1 / math.sqrt(3):
            raise ValueError('3-D seven-point CFL requires courant <= 1/sqrt(3)')
        if self.solid is None:
            solid = np.zeros(shape, dtype=bool)
        else:
            solid = np.asarray(self.solid)
            if solid.shape != shape or solid.dtype != np.bool_:
                raise ValueError('solid must be a boolean shape_xyz array')
            solid = solid.copy(order='C')
        sigma = np.zeros(shape) if self.sigma is None else np.asarray(self.sigma, dtype=float)
        if sigma.shape == ():
            sigma = np.full(shape, float(sigma))
        if sigma.shape != shape or not np.isfinite(sigma).all() or (sigma < 0).any():
            raise ValueError('sigma must be a finite nonnegative shape_xyz array or scalar')
        if not np.isfinite(sigma * (courant * h / c)).all():
            raise ValueError('sigma*dt must remain finite')
        sigma = sigma.copy(order='C')
        solid.flags.writeable = sigma.flags.writeable = False
        for key, value in [('origin_m', tuple(origin)), ('shape_xyz', shape), ('spacing_m', h),
                           ('sound_speed_m_s', c), ('courant', courant), ('solid', solid), ('sigma', sigma)]:
            object.__setattr__(self, key, value)

    @property
    def dt_s(self):
        return self.courant * self.spacing_m / self.sound_speed_m_s

    @property
    def sample_rate_hz(self):
        return 1 / self.dt_s

    @property
    def recommended_max_frequency_hz(self):
        return self.sound_speed_m_s / (12 * self.spacing_m)


@dataclass
class WaveResult:
    signals: np.ndarray
    dt_s: float
    sample_rate_hz: float
    metadata: dict
    energy_time_s: np.ndarray
    energy: np.ndarray

    @property
    def fs(self):
        return self.sample_rate_hz


def _positions(value, name):
    positions = np.asarray(value, dtype=float)
    if positions.shape == (3,):
        positions = positions[None, :]
    if positions.ndim != 2 or positions.shape[1] != 3 or not len(positions) or not np.isfinite(positions).all():
        raise ValueError(f'{name} must be a nonempty finite Nx3 array')
    return positions


def _stencils(grid, positions):
    coordinate = (positions - np.asarray(grid.origin_m)) / grid.spacing_m
    maximum = np.asarray(grid.shape_xyz) - 1
    if (coordinate < -1e-10).any() or (coordinate > maximum + 1e-10).any():
        raise ValueError('source/probe lies outside sampled grid nodes')
    coordinate = np.clip(coordinate, 0, maximum)
    # Roundoff in origin + i*h must not manufacture tiny interpolation into
    # an adjacent solid voxel for a source located exactly at a free node.
    nearest = np.rint(coordinate)
    coordinate = np.where(np.abs(coordinate-nearest) < 1e-12, nearest, coordinate)
    base = np.minimum(np.floor(coordinate).astype(np.int64), maximum - 1)
    fraction = coordinate - base
    indexes, weights = [], []
    for dx in (0, 1):
        for dy in (0, 1):
            for dz in (0, 1):
                delta = np.array([dx, dy, dz])
                node = base + delta
                weight = np.prod(np.where(delta, fraction, 1 - fraction), axis=1)
                index = np.ravel_multi_index(node.T, grid.shape_xyz)
                if np.any((weight > 0) & grid.solid.ravel()[index]):
                    raise ValueError('positive interpolation weight touches solid voxel')
                indexes.append(index)
                weights.append(weight)
    return np.asarray(indexes, dtype='<u8').T.copy(), np.asarray(weights, dtype='<f8').T.copy()


def _run_bounded(command, timeout_s):
    """Kill the entire compiler/kernel process group on timeout or interruption."""
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout_s)
    except BaseException as error:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        if isinstance(error, subprocess.TimeoutExpired):
            raise TimeoutError('bounded acoustic compiler/kernel timed out') from error
        raise
    if process.returncode:
        raise RuntimeError('acoustic compiler/kernel failed: ' + stderr.decode(errors='replace')[-3000:])
    return stdout.decode(errors='replace'), stderr.decode(errors='replace')


def _build_kernel(cache_dir, timeout_s):
    cache = Path(cache_dir).resolve()
    cache.mkdir(parents=True, exist_ok=True)
    compiler = shutil.which('clang++') or shutil.which('c++')
    if compiler is None:
        raise RuntimeError('a local clang++ or c++ compiler is required')
    version = _run_bounded([compiler, '--version'], min(timeout_s, 5))[0].splitlines()[0]
    source_hash = _sha(KERNEL.read_bytes())
    flags = ['-std=c++17', '-O3', '-fno-fast-math']
    key = _sha((source_hash + version + platform.machine() + ' '.join(flags)).encode())[:24]
    executable = cache / ('wave3d-' + key)
    deadline = time.monotonic() + timeout_s
    with (cache / ('build-' + key + '.lock')).open('a+b') as lock:
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() >= deadline:
                    raise TimeoutError('acoustic kernel build-lock timed out')
                time.sleep(.02)
        if not executable.exists():
            temporary = executable.with_name(executable.name + f'.{os.getpid()}.tmp')
            try:
                _run_bounded([compiler, *flags, str(KERNEL),
                              '-o', str(temporary)], max(.001, deadline - time.monotonic()))
                temporary.chmod(0o700)
                os.replace(temporary, executable)
            finally:
                temporary.unlink(missing_ok=True)
        fcntl.flock(lock, fcntl.LOCK_UN)
    return executable, {'kernel_source_sha256': source_hash, 'compiler': version, 'compiler_flags': flags,
                        'executable_sha256': _sha(executable.read_bytes())}


def simulate(grid, source_xyz, drive, receiver_xyz, *, cache_dir, timeout_s=60.,
             energy_stride=0, source_weights=None):
    """Return signed signals[K,N] at t=n*dt, without normalization or alignment.

Source coordinates may be one XYZ point or M points with optional signed
weights. Their trilinear node weights are summed, not normalized. drive[n] is
the volume-integrated source q(t); injection is dt**2*drive[n]*weight/h**3.
The state at n is recorded before its update; the final drive sample has no
future output state. Supply trailing zeros for the desired propagation tail.
An impulse of unit time integral is drive[0]=1/dt, not drive[0]=1.
    """
    if not isinstance(grid, Grid):
        raise ValueError('grid must be Grid')
    if sys.byteorder != 'little':
        raise RuntimeError('native binary protocol currently requires little endian')
    timeout = _scalar(timeout_s, 'timeout_s', positive=True)
    if timeout > 600:
        raise ValueError('timeout_s must not exceed 600')
    if type(energy_stride) is not int or not 0 <= energy_stride <= MAX_STEPS:
        raise ValueError('energy_stride must be an integer in 0..MAX_STEPS')
    q = np.asarray(drive, dtype='<f8')
    if q.ndim != 1 or not 2 <= len(q) <= MAX_STEPS or not np.isfinite(q).all():
        raise ValueError('drive must be finite 1D with 2..MAX_STEPS samples')
    source, receivers = _positions(source_xyz, 'source_xyz'), _positions(receiver_xyz, 'receiver_xyz')
    if len(source) > MAX_CELLS or len(receivers) > MAX_RECEIVERS or len(receivers) * len(q) > MAX_OUTPUT_SAMPLES:
        raise ValueError('source/probe/output resource bound exceeded')
    source_index, source_interpolation = _stencils(grid, source)
    weights = np.ones(len(source)) if source_weights is None else np.asarray(source_weights, dtype=float)
    if weights.shape != (len(source),) or not np.isfinite(weights).all():
        raise ValueError('source_weights must have one finite value per source position')
    unique, inverse = np.unique(source_index.ravel(), return_inverse=True)
    combined = np.bincount(inverse, weights=(source_interpolation * weights[:, None]).ravel())
    keep = combined != 0
    source_index, combined = unique[keep].astype('<u8'), combined[keep].astype('<f8')
    if not len(source_index):
        raise ValueError('source weights cancel to an empty source')
    receiver_index, receiver_weight = _stencils(grid, receivers)
    executable, build = _build_kernel(cache_dir, timeout)
    energy_count = (len(q) - 2) // energy_stride + 1 if energy_stride else 0
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='wave3d-run-', dir=Path(cache_dir).resolve()) as temporary:
        input_path, output_path = Path(temporary) / 'input.bin', Path(temporary) / 'output.bin'
        with input_path.open('wb') as stream:
            stream.write(INPUT_HEADER.pack(b'OPCW3D1\0', *grid.shape_xyz, len(q), len(receivers),
                                           len(source_index), grid.spacing_m, grid.sound_speed_m_s,
                                           grid.dt_s, energy_stride))
            for array in (grid.solid.astype('u1'), grid.sigma.astype('<f8'), q,
                          source_index, combined, receiver_index, receiver_weight):
                stream.write(array.tobytes(order='C'))
        _run_bounded([str(executable), str(input_path), str(output_path)], timeout)
        expected = OUTPUT_HEADER.size + 8 * (len(q) * len(receivers) + 2 * energy_count)
        if not output_path.exists() or output_path.stat().st_size != expected:
            raise RuntimeError('acoustic kernel output size differs from declared protocol')
        with output_path.open('rb') as stream:
            magic, count, probes, energies, dt = OUTPUT_HEADER.unpack(stream.read(OUTPUT_HEADER.size))
            if (magic, count, probes, energies, dt) != (b'OPCWOUT1', len(q), len(receivers), energy_count, grid.dt_s):
                raise RuntimeError('acoustic kernel output header differs')
            signals = np.fromfile(stream, '<f8', len(q) * len(receivers)).reshape(len(q), len(receivers)).T.copy()
            energy_time = np.fromfile(stream, '<f8', energy_count)
            energy = np.fromfile(stream, '<f8', energy_count)
    if not all(np.isfinite(a).all() for a in (signals, energy_time, energy)):
        raise RuntimeError('acoustic kernel returned nonfinite values')
    metadata = {**build, 'engine': 'scalar_wave_3d_7point_centered_time_cpu_float64',
                'equation': 'p_tt+2*sigma*p_t=c^2*graph_laplacian(p)+q',
                'shape_xyz': list(grid.shape_xyz), 'origin_m': list(grid.origin_m),
                'spacing_m': grid.spacing_m, 'courant': grid.courant,
                'recommended_max_frequency_hz': grid.recommended_max_frequency_hz,
                'frequency_rule': '12 cells per shortest wavelength; recommendation, not a dispersion certificate',
                'outer_boundary': 'rigid Neumann faces half a cell beyond outer node centres',
                'solid_boundary': 'rigid Neumann at solid/free voxel faces; staircased geometry',
                'damping': 'nonnegative diagonal sigma; optional sponge is not PML or calibrated impedance',
                'source_scaling': 'dt^2*q[n]*summed_trilinear_weight/h^3',
                'receiver_sampling': 'same trilinear weights as emission; signed pressure',
                'time_grid': 'signals[:,n] is p(n*dt); drive[n] first changes p[n+1]',
                'energy_definition': 'half-step discrete PDE energy, not acoustic joules; kinetic plus cross-time edge-gradient potential; no source work removed',
                'solid_sha256': _sha(grid.solid.tobytes()), 'sigma_sha256': _sha(grid.sigma.tobytes()),
                'drive_sha256': _sha(q.tobytes()), 'source_nodes_sha256': _sha(source_index.tobytes()+combined.tobytes()),
                'receiver_nodes_sha256': _sha(receiver_index.tobytes()+receiver_weight.tobytes()),
                'run_wall_s': time.monotonic() - started, 'real_room_validated': False,
                'geometry_gradients_or_adjoint_implemented': False}
    return WaveResult(signals, grid.dt_s, grid.sample_rate_hz, metadata, energy_time, energy)
