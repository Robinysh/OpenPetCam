# Offline room-fingerprint localization

This workflow concerns passive **source localization** with a first-order
ray/path dictionary and unknown-source power features. Geometry **Full Waveform
Inversion (FWI)** is a separate research direction: fitting signed or complex
waveforms through a wave-equation forward model. The signed analytic one-plane
diagnostic is also a separate control. Results below assess this source matcher;
they establish neither success nor failure of geometry FWI.

This experiment asks whether one microphone recording can match an uncertain
room model under an explicit smooth-source-spectrum assumption. An arbitrary
unknown source can reproduce a recording through many different room transfers;
one channel does not generally identify its source position. Harmonic sources,
microphone directionality and uncertain room dimensions can break the matching
assumption. Scores and spatial score sets are not probabilities or confidence
regions.

The forward model in `sim/acoustics/localization_model.py` retains signed direct
and finite first-order specular paths. The dictionary sums their complex
transfers before calculating power. The waveform simulator uses signed,
fractional-delay impulse responses. It omits diffraction, transmission and a
validated higher-order furnished-room response. The saved floor scale and some
surfaces have acoustic-derived provenance, so they are model assumptions rather
than independent validation truth.

The unknown-source scorer in `sim/acoustics/localization.py` compares log-power
spectral ripple after removing a fixed smooth envelope. Original audio samples
remain signed and unchanged. The transient Welch/Hann spectrum is only an
approximation to the exact stationary transfer dictionary. Source decay changes
between development and held-out simulations are explicit in the protocol.

## Bounded synthetic benchmark

Run locally with a new private output directory:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
MKL_NUM_THREADS=2 nice -n 10 python3 -m sim.acoustics.localization_experiment \
  --room eval/world_remap1/model/manifest.json \
  --output eval/localization_overnight/NEW_RUN \
  --cases 512 --spacing 0.5 --seed 20261107 --deadline-s 3590
```

Use a separate process supervisor when an absolute wall-time kill is required;
the module checks its deadline at progress boundaries. The overnight job uses a
3600-second supervisor and at most two numerical threads. No camera access or
remote compute is involved.

The module archives source files and their hashes, candidate coordinates,
configuration choices, spatial partitions, seeds and exact case definitions
before generating any evaluation waveform. Whole one-metre horizontal tiles,
including their height variants, share a partition. The candidate dictionary is
a forward-model search space; it contains candidates in both partitions and is
not a fit to source labels. The endpoint bounding box does not certify enclosed,
walkable or physically plausible source space.

The benchmark includes on-grid same-model **algorithm checks**, unseen off-grid
sources, colored and harmonic source spectra, combined 2 cm/10 cm geometry,
material and speed mismatch, diffuse tails, microphone directionality, low SNR,
occluded sources, anechoic and silence controls. These generated bursts are not
validated dog-bark simulations. All accuracy metrics from this benchmark are
**simulated**, not real-room accuracy.

Feature configuration and optional acceptance thresholds use development cases
only. The selection loss includes a frozen penalty for every unlocalized case.
The selected configuration and rule are saved before any held-out waveform is
generated, and the held-out set is evaluated once. Exact score ties always
abstain, including when the calibrated margin threshold is zero. Every case stays
in success and acceptance denominators. Missing positions and their errors stay
null; distance quantiles describe localized cases only. Results include random
and fixed-prior baselines plus a truth-using grid-discretization floor, which is
not an estimator. Repeated draws can share tiles, so the report does not present
unclustered confidence intervals.

## Diagnostics on original recordings

After a benchmark freezes its dictionary and selected configuration:

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
MKL_NUM_THREADS=2 nice -n 10 python3 -m sim.acoustics.localization_real \
  --corpus eval/localization_overnight/corpus_v2/manifest.json \
  --benchmark eval/localization_overnight/NEW_RUN \
  --output eval/localization_overnight/NEW_REAL_DIAGNOSTICS \
  --deadline-s 1200
```

The diagnostic checks corpus hashes and selects complete windows of at least
0.512 seconds wholly inside explicitly declared continuous segments. It never
crosses timestamp gaps, pads short windows or changes original signed scale.
At native 48 kHz, 24,576 samples supply the same physical frequency spacing as
the 16 kHz/8192 dictionary. Power is restricted to dictionary Nyquist before
feature smoothing. No resampling is needed; unsupported exact grids are excluded
with a reason. Highest-RMS window selection is a reproducible diagnostic policy,
not an emitting-source label. Duplicate, estimated overlap and conservative day
group metadata remain attached to every result.

There are no certified source positions or checked historical microphone poses
for this corpus. The output therefore calls a coordinate a
`candidate_model_position_m`, keeps physical source and microphone pose null,
forces real acceptance false and reports no real localization error. Even a
synthetic-rule pass remains an unvalidated conditional ranking. A convincing
candidate cannot establish a real sound source, a calibrated bearing or a room
measurement.

## Behavioral checks

```bash
python3 tests/sim/acoustic_localization_test.py
python3 tests/sim/acoustic_localization_experiment_test.py
python3 tests/sim/acoustic_localization_real_test.py
```

These checks cover missing-transfer accounting, selection penalties, exact-tie
abstention, spatial split preservation, frozen protocol ordering and signed,
continuous native-window provenance. Forward-model and corpus-builder checks
remain in their separate test modules.

## Exploratory development research

`localization_identifiability.py` maps exact and near-equal candidate signatures,
path availability and bearing/range/height confounds without opening held-out
predictions or labels. A known-transfer oracle is an optimistic representation
check. An unknown source recording does not provide that transfer. In particular,
one path has flat power apart from scalar gain, so a delay carried by its complex
phase cannot locate an unknown source through power ripple.

`localization_development.py` compares one, four and sixteen burst spectra at a
fixed simulated source. All settings use nested prefixes of the same sixteen
draws. It averages linear power spectra, without aligning or stacking waveforms.
Room geometry, material, speed, microphone response and diffuse tail stay fixed
within each repeated sequence. Independent/replayed broadband controls share an
identical first waveform; fixed and varying harmonic pitch distinguish averaging
noise from averaging away a persistent source comb. An absolute-background
control has no emitting source or fabricated localization error.

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
MKL_NUM_THREADS=2 nice -n 10 python3 -m sim.acoustics.localization_development \
  --benchmark eval/localization_overnight/FROZEN_RUN \
  --output eval/localization_overnight/NEW_DEVELOPMENT_STUDY \
  --cases 384 --seed 20261108 --deadline-s 590
```

The frozen low/mid/wide bands are crossed with explicit conditional source priors:
the full candidate grid, a nominal low source height, a four-metre receiver range,
and their intersection. These are hypotheses, not a surveyed household footprint
or measured source labels. Each prior is reported on every source case, its
declared valid-support subset, and the wrong-prior complement. Truth never chooses
which mask the matcher uses. Narrower masks can increase score margins even when
the source lies outside them; a larger margin is not calibrated confidence.

These comparisons use development spatial tiles only and remain **exploratory
development evidence**. They do not establish a fresh confirmatory result,
validated repeated-bark performance or real-room accuracy. Repetition can reduce
source-spectrum variance while leaving a systematic model error or directional
alias unchanged. A follow-up method needs a new predeclared evaluation beyond
the already inspected held-out cases.

```bash
python3 tests/sim/acoustic_localization_identifiability_test.py
python3 tests/sim/acoustic_localization_development_test.py
```

`localization_resolution.py` separately tests global candidate-grid spacing under
a fixed height hypothesis. A further private query-only study adds both declared
source heights, so finer spacing can be compared without requiring an external
height cue. Reusing frozen burst spectra keeps the comparisons paired. Every
source family, prior violation and missing prediction stays in the appropriate
denominator; candidate placement never follows the known source coordinates.
Finer search can also increase background-only match scores, so a higher maximum
similarity is not evidence of a sound source.

`localization_bearing.py` examines the camera's bearing decision separately from
range accuracy. It finds the shortest circular arc containing every candidate
within a fixed score tolerance, including azimuth wraparound. Its arc midpoint is
a conditional bearing candidate. Narrow arcs remain descriptive sensitivity
sets: model error or a harmonic source can produce a narrow, wrong arc.

The selective-bearing experiment trains gates on four of five development tile
folds and evaluates decisions on the remaining fold. Whole horizontal tiles keep
height variants and repeated source scenarios together. Background-only cases
have null physical source/error and count as failures if the simulated rule acts.
Training rules require support from at least three source-anchor tiles, and exact
ties always abstain. This is exploratory cross-validation on an already inspected
development domain, with correlated scenario rows. It is not fresh confirmation,
independent-case confidence or permission for a real camera action.

```bash
python3 tests/sim/acoustic_localization_resolution_test.py
python3 tests/sim/acoustic_localization_bearing_test.py
```
