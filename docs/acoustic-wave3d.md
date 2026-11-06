# Phase-preserving 3D acoustic experiments

This experiment adds a new wave-equation forward engine and source-position
waveform fitting. It does not reuse the earlier unsupervised room FDTD fit.
Room-geometry optimization and an adjoint remain future work.

```sh
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
MKL_NUM_THREADS=2 timeout 600 nice -n 10 python3 -m sim.acoustics.wave_experiment \
  --room eval/MY_ROOM/manifest.json \
  --sources eval/MY_SOURCES/virtual_source_inventory.json \
  --output eval/MY_NEW_WAVE_RUN --anchors 16 --spacing .5 --seed 2026112701
python3 -m sim.acoustics.wave_report eval/MY_NEW_WAVE_RUN \
  --out eval/MY_WAVE_REPORT.html
```

Use a genuinely fresh seed and output directory for a new evaluation. This seed
identifies an already evaluated experiment; rerunning it is reproduction, not
fresh evidence. NumPy/SciPy are offline analysis dependencies. A local C++17
compiler builds the bounded native solver in the explicit output/cache folder.
There are no downloads, paid calls, camera queries, playback, capture, production
updates or installed-model writes. Private audio and room data stay in `eval/`.

## The forward model

`wave3d.py` and `wave3d_kernel.cpp` integrate

```
p_tt + 2 sigma p_t = c^2 L_h p + q
```

using a seven-point three-dimensional graph Laplacian and centred second-order
time updates. Missing exterior or solid neighbours impose rigid Neumann faces.
Nonnegative diagonal damping is volumetric loss; a damping sponge is not PML
or calibrated boundary impedance. The CFL limit is checked. The recommended
band uses at least twelve cells per shortest wavelength; this is a discretization
rule, not certification of real acoustic accuracy.

Source and receiver use the same trilinear spatial weights. The source update
contains `dt^2 * q[n] / h^3`. Samples retain their signed amplitudes:
`signals[:, n]` is pressure at `n*dt`, before source sample `q[n]` changes the
next state. A unit-sample Green response convolves with a digital drive without
an extra `dt` factor. Source strengths are mathematical forcing units, not a
calibrated loudspeaker or pressure measurement.

Tests independently check absolute 3D free-field delay/amplitude and `1/r`
spreading, exact lattice modes, refinement, reciprocity with solids and varying
damping, positive rigid-wall reflections, blocked transmission, signed linearity,
lossless discrete energy conservation, damping decay and resource/CFL bounds.
The native compiler/kernel have timeouts and bounded allocations. These tests
validate the implemented numerical physics, not the room model or camera housing.

Reciprocity lets one solve emit at the assumed microphone and record many
candidate Green responses. Source-location inversion uses those responses in
the opposite direction. The report's animation shows this reciprocal probe's
actual 3D field on one horizontal slice, convolved with a causal Gaussian-windowed
120 Hz packet. Signed pressure uses one colour scale for every frame, with
native sample times. It is not a two-dimensional propagation approximation.

## The room is an explicit hypothesis

`wave_room.py` reads the saved floor, ceiling and finite wall endpoints. Saved
segments do not certify a closed footprint. The adapter explicitly adds a
rectangular rigid completion, extrudes every segment floor-to-ceiling and
voxelizes finite thickness. Some segments may represent furniture faces. The
floor scale has acoustic-derived provenance and the ceiling is assumed.

All requested and realized bounds, wall shifts, solid masks and quantization
are recorded. Computational air does not mean surveyed free or walkable space.
Housing, microphone directionality, transmission, detailed furnishings and
frequency-dependent wall impedance are omitted. The primary experiment uses
80–220 Hz, below the archived active sweep's 2.1–7.4 kHz band; those sweep fits
cannot validate this new low-frequency engine.

## Source knowledge and phase

The experiment compares six fixed methods:

- First-order ray power ripple, using the unchanged earlier analytic baseline.
- Full-wave power ripple, which retains modal power but discards phase.
- Full-wave complex fitting with the exact prescribed source waveform.
- Complex fitting with another recording from a different source/day group.
- Blind projection onto a real causal 32 ms source FIR with declared ridge.
- An arbitrary complex-source control, which never returns a coordinate.

Known-waveform fitting profiles one real nonnegative gain and one bounded
common onset. It does not fit a separate phase at each frequency. Exact spatial
ties abstain. The arbitrary-source control exposes the ambiguity `Y = G_x S`:
if `S` is unconstrained, many different nonzero transfers explain the same
single-channel observation. A short source FIR or another recording supplies
a source prior; neither is a validated model of an unknown bark.

The four native signed input windows are prescribed virtual emitters. They are
already reverberant observed-room recordings, not dry sources. Machine event
hints do not certify the object that emitted a sound. The exemplars have shared
overlap/day groups and are not four independent source families. Original
samples and scale are checked by hash; a fixed Hann gate and complex DTFT create
derived drives without per-recording normalization. Real physical source and
historical microphone labels remain null.

The source inventory contains `window_archive.path` and its SHA-256, plus four
`entries`. Each entry declares `id`, `npz_array`, `sample_rate_hz`, the original
array-byte `window_sha256`, `source_kind`, `category`, `overlap_group` and
`recommended_split_group`. It must describe verified continuous native windows.
The alternate-template pairs must cross groups. The original recordings are
not uploaded or rewritten.

## Controls and interpretation

Cases, candidates, source/room snapshots, code and hashes freeze before any
evaluation propagation. The comparisons distinguish identical on-grid algorithm
checks, off-grid sources on that same numerical model, finer-grid propagation,
and combined geometry/receiver/speed/loss/completion mismatch. Off-grid scene
comparisons share gain, onset and noise draws. Their continuous onsets have up
to 0.25 ms fitting-grid mismatch; the on-grid control has exactly gridded onsets.
The ray baseline also omits the artificial enclosing walls, so ray-versus-wave
results do not isolate numerical solver order alone.

Observations are complex physical-frequency samples of convolutions with the
finite one-second Green responses. They are not separate time-domain room
recordings. Native grids share the same physical comparison bins without
resampling. Green truncation and different stair-step geometries remain model
assumptions even when propagation is numerically stable.

Silence, absolute background spectra and multiple-source mixtures are retained.
Every single-source case remains in success denominators; missing coordinates
and errors stay null. Multiple-source cases have no invented single true point.
Conditional candidates on backgrounds are counted separately from accepted
events. All operational/real acceptance remains false. Repeated exemplars,
anchors and room variants are correlated, so the report presents no independent
case confidence intervals or real-room accuracy.

This forward model supports subsequent waveform-inversion experiments. Real
localization still needs independent emitter/microphone evidence and verified
room/source assumptions. A successful numerical control supplies neither.
