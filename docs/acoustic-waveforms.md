# Offline acoustic waveform fitting

For new measurements, use the [acquisition and shot-gather workflow](acoustic-acquisition.md)
first. This earlier experiment preserves signed responses, but its acoustic
alignment is not an independently verified emission-time reference. Do not use
its fitted distances as calibrated surface measurements.

This experiment asks whether one plane explains **pose-dependent signed acoustic
responses** better than a shared background response. It runs on saved raw
microphone captures. It does not contact the camera, play sound, update the room
model, or deploy anything.

The implementation starts with analytic image-source geometry. Previous FDTD
fits and evidence ledgers are not calibration or ground truth for this experiment.

## Run

Use Python 3.11 or later with the offline dependencies:

```bash
python3 -m pip install -r sim/acoustics/requirements.txt
python3 -m sim.acoustics.experiment \
  --scan eval/world/raw/acoustic/SESSION \
  --probe eval/world/raw/acoustic/chirp2_16k.npy \
  --calibration eval/world/calib.json \
  --reference SESSION:REFERENCE_INDEX \
  --normal=0,-1,0 \
  --distance-min 0.60 --distance-max 1.05 --distance-step 0.005 \
  --timing-bound-us 250 --timing-steps 11 \
  --window-ms 3.0 7.5 \
  --output eval/acoustic_fit/run01
```

Replace `SESSION` and `REFERENCE_INDEX` with a saved session directory and a
reference recording chosen before fitting. Each session needs the `log.jsonl`
and 48 kHz little-endian signed 16-bit mono PCM files produced by
`sim/world/sonar_scan.py`. The stored transmitted probe is a 16 kHz NumPy array.
Repeat `--scan` to include more sessions. Session directory names must be unique.
Use a fresh output directory for each run; completed runs cannot be overwritten.

The example searches a downward-facing plane, such as a floor. The bounds,
normal, comparison window and timing uncertainty are **experiment assumptions**,
not independently measured room dimensions. Choose them before looking at held-out
scores. Do not try different settings until a held-out score improves and then
report it as a held-out result. The default window excludes the early region used
for alignment, including the lag-search extent and a guard.

Results describe the supplied sessions, not a general camera accuracy guarantee.
Private recordings, outputs and room calibration belong under ignored `eval/`.

## Geometry

`sim/acoustics/geometry.py` defines a fixed pan origin and a tilt pivot rigidly
attached to the pan body. The speaker moves with pan only. The microphone and
lens move with pan and tilt. The default reference lens is at world zero; x is
right, y is up and z is forward. Positive pitch looks upward.

`--rig path.json` overrides these offsets in metres:

```json
{
  "pan_origin_m": [0, 0, -0.029],
  "tilt_offset_m": [0, 0, 0],
  "speaker_offset_m": [0, 0.048, -0.048],
  "mic_offset_m": [0, 0.012, 0.0404],
  "lens_offset_m": [0, 0, 0.029]
}
```

These defaults approximate the reference device; they are **not calibrated
acoustic centres**. Motor angles come from the supplied optical calibration.
Its residual optical roll is not treated as a measured mechanical mount transform.

A plane is `normal · position = distance`. The normal must have unit length.
For source `s`, receiver `r`, normal `n` and distance `d`, the mirrored source is
`s_image = s + 2(d - n·s)n`. Reflection length is `|r - s_image|`; direct length is
`|r - s|`. The modeled relative delay is their difference divided by sound speed,
plus the profiled common timing offset. Both endpoints must be on the same side.

The plane is infinite. The model includes spherical spreading and propagation
phase, but no finite-panel edges, visibility, housing diffraction, absorption,
multiple reflections or unknown hidden objects. A normal's sign does not change
the plane when its signed distance is reversed too.

## What is preserved

`waveforms.py` uses the known probe for regularized deconvolution, including a
smooth 2.1–7.4 kHz taper. It retains signed amplitudes and complex spectra. No
per-recording peak normalization occurs. All four expected sweeps are counted;
missing or malformed captures and alignment-bound hits fail closed. Silence and
other playback volumes remain explicit exclusions in the inventory.

Only the designated training capture defines the early template. Every sweep
receives one bounded fractional-sample lag correction estimated from that early
window and applied to its complete response. Both unaligned and aligned sweeps,
recording means, original matched-filter positions and additional lags are saved.

The early anchor is a prominent acoustic path. It has **not** been established as
the physical direct arrival. Matched-filter extraction also removes each sweep's
integer playback offset. Absolute emission time cannot be recovered from these
metrics. Therefore fitted distances remain conditional on the reference and
acoustic-centre assumptions. Timing profiles expose some of this ambiguity;
the configured timing bound is not a measured confidence limit.

## Fitting and controls

1. Split complete actual pan/tilt groups before reading audio. All repeats at one
   pose stay together across supplied sessions. The reference pose is training.
2. Average captures within each pose, so a repeated reference block does not
   outweigh poses recorded once.
3. Learn a common background waveform from training poses only. This is the
   nuisance-only baseline, including echoes that do not vary with pose.
4. For every candidate distance and common timing offset, generate the reflected
   waveform through the **actual regularized deconvolution transfer kernel**.
   Apply the same fixed window and taper to predictions and observations.
5. Fit one shared complex reflection coefficient on training data. Its real and
   imaginary parts multiply the two phase quadratures. There is no individual
   recording gain, free phase, or candidate-specific alignment.
6. Select the candidate using training error only. Predict held-out poses with
   the frozen background and coefficient. Report improvement against the shared
   background; a negative value means the surface made prediction worse.
7. Shuffle the frozen predicted contributions among held-out poses as a
   direction-association control. This does not reselect the surface. The
   empirical permutation result is a diagnostic, subject to nuisance correlations.

The common background deliberately prevents a strong echo shared by every pose
from being counted as evidence that the model predicts directional geometry.
Identical source/receiver geometry has no pose contrast in this test. This does
not mean that a single echo cannot range a known-normal plane when physical
direct-relative timing is independently known.

The full distance/timing loss grid is retained, including alternative minima,
boundary solutions and design singular values. The reported near-optimal range
band uses 1% of training background-residual energy as a descriptive tolerance;
it is **not a confidence interval**. Distinct path signatures are counted because
symmetry can make many motor poses geometrically equivalent (for example, pan
rotation above a horizontal floor). Pose holdout within one session is not a
new-session test.

## Outputs

| File | Contents |
|---|---|
| `protocol.json` | Pre-audio split, parameter grids, assumptions, dependency versions, code/calibration hashes |
| `waveforms.npz` / `.json` | Signed raw/aligned sweeps and means, exact inverse-filter kernel/spectra, timing metadata, full capture inventory and input hashes |
| `fit.json` | Training-selected candidate, train/held-out scores, complete profiles, permutation control, limitations |
| `fit_arrays.npz` | Grouped signed waveforms/complex spectra, observed/predicted windowed traces, shared background, surface contribution and geometry |

No result is automatically promoted into the room model. Even a positive held-out
result needs independent acoustic calibration and a measured surface or controlled
panel before becoming a physical distance claim.

## Verification

```bash
python3 tests/sim/acoustic_geometry_test.py
python3 tests/sim/acoustic_waveforms_test.py
python3 tests/sim/acoustic_fit_test.py
tools/run_tests.sh
```

Tests cover rigid-pivot invariants, plane intersections and analytic bistatic
delays; raw convolution/deconvolution of signed echoes and amplitudes; fractional
alignment and train-reference isolation; independent synthetic surface recovery;
shared-background and no-pose-contrast controls; fixed-window behavior; and
invariance of all selected fit parameters to changes in held-out waveforms.
CI and releases run these tests with SciPy installed. No private recordings are
needed for tests.
