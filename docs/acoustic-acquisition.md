# Acoustic acquisition and shot gathers

This workflow compares excitation signals before fitting room geometry. It
archives the exact digital playback, collects individual shots, checks each shot,
and assembles signed receiver traces with explicit timing limitations. It does
not select an optimal signal or enable inversion automatically.

The earlier recordings remain useful exploratory data, but they do not have an
archived playback packet, a paired source reference, or verified continuous ADC
sample numbering. Preserving their signed samples does not retrospectively supply
those missing measurements. See [the first waveform experiment](acoustic-waveforms.md).

## Keep the output rate fixed

Use native 48 kHz archives for new phase-sensitive experiments with the current
T23 player configured at 48 kHz. A 16 kHz PLAY job switches that output to 16 kHz
and restores 48 kHz afterwards. Controlled recordings reproduced job-to-job
timing steps on that path; native 48 kHz jobs repeated stably within the tested
fixed poses, including three fresh recording connections. A rate change still
starts a new timing epoch. Do not align away that discontinuity or combine epochs
as calibrated phase measurements.

The generator now synthesizes 48 kHz packets directly by default. Live acquisition
rejects a selected 16 kHz packet before touching the camera unless
`--allow-rate-switching` explicitly requests a diagnostic. Older 16 kHz archives
remain readable; `--derive-from` produces an exact threefold sample-held 48 kHz
copy when an existing comparison requires it. That conversion is unnecessary for
new native packets.

Before playback, the adapter checks the live kernel audio report for 48 kHz,
16-bit recording and replay. It also rejects missing or changing SDK output-rate
evidence in the recorded frames. `output_sample_rate` is not a supported query in
this firmware's JSON API; its internal default is insufficient evidence. These
checks do not establish physical DAC emission time, cross-pose acoustic
calibration, or an optimal excitation signal.

## What a gather means here

A common-source gather contains receiver measurements for one unchanged source
position **and orientation**. The planned camera implementation holds pan fixed
within a contiguous block and moves only tilt. The speaker belongs to the pan
body; the microphone belongs to the tilting head. Pan changes therefore start a
different source block. Actual motor positions are recorded before and after each
packet and must agree with the plan.

This is a sequential synthetic receiver gather: each receiver pose gets repeated
emissions. It is not a simultaneous microphone array. The room, playback chain
and source must remain stable during the block. Matching reference receiver visits
bracket every block, and receiver order is randomized with a stored seed.

The default rig produces a centimetre-scale receiver path. Rotating through many
viewing directions does not create a metre-scale acoustic aperture, and symmetry
can make several poses acoustically equivalent. Tilt also changes microphone
orientation and housing response. Source and receiver coordinates in the plan are
explicitly **assumed acoustic centres**, not a physical calibration.

## Timing evidence has distinct levels

| Reference | What it supplies | What it does not establish |
|---|---|---|
| Host command timestamps | Queue and capture bookkeeping | DAC emission sample or microphone first-sample time |
| Archived digital WAV | Exact nominal source samples and event positions | Actual hardware playback timing or acoustic source waveform |
| Two recorded acoustic pilots | A conditional packet-wide offset/rate map, provided the reference acoustic path stays stable | Physical emission time or calibrated phase between receiver poses |
| Paired SDK output reference | A measured output-reference channel on a checked microphone sample grid, when framing and continuity validation pass | The reference's relation to the physical DAC, loudspeaker phase centre, or propagation time |
| Independently calibrated hardware reference | A tested relationship between the reference, emission, ADC and acoustic centres | Accuracy outside the calibration's measured conditions |

The two identical pilots bracket each packet. Two anchors determine an affine
map with no residual degrees of freedom; the remaining shot timing is a separate
diagnostic. Pilots and shots use different waveforms, so their matched-filter
peaks can have different instrument and multipath biases. Their residual is not
pure clock error.

An acoustic pilot origin can vary with receiver position and remove the very
travel-time change needed for geometry. The analyzer consequently keeps
`cross_pose_phase_calibrated` and `physical_dac_timing_calibrated` false. A paired
SDK reference improves the evidence only when its actual sample grid and
continuity have been verified. It must not be relabeled as physical DAC time zero.

## Generate exact, comparable probe packets

Run offline from the repository with the acoustic dependencies installed:

```bash
python3 -m pip install -r sim/acoustics/requirements.txt
python3 -m sim.acoustics.probes eval/acoustic_acquisition/probe_suite_48k \
  --sample-rate 48000 --durations 0.5 2.0 --repetitions 4 --target-rms 0.34
```

The output directory must be new. The default suite contains upward linear and
exponential sweeps at each duration, with:

- 2.0–7.6 kHz nominal sweep band and native 48 kHz mono playback;
- equal duration, target RMS and repetition count within each comparison pair;
- a digital peak ceiling of 0.5, endpoint fades and silent intervals;
- identical downward 80 ms timing pilots before and after the shots;
- exact signed 16-bit `packet.wav`, `probe.wav` and `pilot.wav` archives;
- WAV and PCM hashes, source-sample event positions, actual quantized RMS and energy.

Templates are decoded from the archived quantized WAV. Floating-point recipes
are not substituted for the transmitted digital samples. Energy matches before
quantization; the small quantization differences are recorded. Equal RMS does
not mean equal energy per hertz: the sweep families distribute energy differently.

Default packet durations are 8.66 seconds for the 0.5-second shots and 14.66
seconds for the 2-second shots. A longer sweep consumes more energy and time.
The old fixed 13-second capture script cannot accommodate the longer packet.
The new capture adapter derives its recording interval from the packet manifest.

## Plan a balanced comparison first

Use the existing optical motor calibration to construct assumed poses. It does
not become an acoustic calibration by being included in a plan.

```bash
python3 -m sim.acoustics.survey \
  --suite eval/acoustic_acquisition/probe_suite_48k/suite.json \
  --calibration eval/world/calib.json \
  --mode comparison --pans 1850 --reference-tilt 699 \
  --blocks 3 --volume 30 --seed 731 \
  --output eval/acoustic_acquisition/comparison-plan.json
```

Every packet appears once in each randomized block at the same source/receiver
pose and playback level. Compare linear versus exponential within the **same
duration and block**. Compare duration separately with its time and energy cost.
The plan prohibits per-shot amplitude normalization, cross-pose waveform
alignment, stacking before QC and automatic inversion.

## Capture only after validating the packet and plan

For a separate volume experiment, use one exact packet at one fixed pose:

```bash
python3 -m sim.acoustics.survey \
  --suite eval/acoustic_acquisition/probe_suite_48k/suite.json \
  --calibration eval/world/calib.json \
  --mode volume --packet SELECTED_PACKET_ID_FROM_SUITE_JSON \
  --pans 1850 --reference-tilt 699 --blocks 3 \
  --volumes 20 30 40 --volume 30 --seed 731 \
  --output eval/acoustic_acquisition/volume-plan.json
```

Each block brackets randomized level trials with two visits to the reference
volume (`--volume`). The microphone settings, pose and exact waveform stay fixed.
Levels are device commands, not calibrated sound-pressure levels. A noisy room
can support this robustness experiment, but music and moving people can change
both background noise and the acoustic response. Compare the repeated reference
visits before attributing differences to playback level. This experiment does
not establish a static room response or an optimal probe.

The following preflight is offline and performs no camera operation:

```bash
python3 -m sim.acoustics.capture \
  --plan eval/acoustic_acquisition/comparison-plan.json \
  --suite eval/acoustic_acquisition/probe_suite_48k/suite.json \
  --output eval/acoustic_acquisition/comparison-session
```

Actual capture uses the camera server's existing application environment and
credentials. Copy the repository modules, complete suite and plan there before
running the same command with `--execute`:

```bash
python3 -m sim.acoustics.capture \
  --plan eval/acoustic_acquisition/comparison-plan.json \
  --suite eval/acoustic_acquisition/probe_suite_48k/suite.json \
  --output eval/acoustic_acquisition/comparison-session \
  --deadline-s 900 --execute
```

`--execute` moves the camera and plays audible probes. The adapter validates all
assets before hardware access, copies and revalidates them in the session archive,
and plays that archived copy. It records requested and actual poses, packet and
microphone PCM hashes, audio configuration snapshots and attempted-capture
failures. It refuses an existing output directory, limits the session, checks
privacy, holds automatic following, and attempts to restore the original pose.
Failed or partial sessions are inspected as such; they are not silently resumed.

An outer operator that pauses following must own both its restoration and an
independent fallback. Arm or rearm the watcher fallback after the camera's stock
rollback has been acknowledged, leaving a positive margin after that rollback.
Comparing timeout durations alone is insufficient: activation can consume the
margin between guards started at different times. Verify process identity before
cancelling an owned guardian, and cancel any replacement after verified stock
restoration. Transfer and analysis happen after device restoration.

Capture allows up to 20 seconds for initial raw data before queuing playback.
Each reader runs for the packet duration plus 40 seconds, covering that startup,
a bounded 12-second playback command, settling and the complete source tail.
The session checks for packet duration plus 90 seconds remaining before the next
capture to reserve upload, recording and connection/cleanup time. Insufficient
budget prevents the next capture. These are acquisition scheduling allowances;
they do not establish a live-video latency target or physical emission timing.

The server adapter defaults to the framed reader at `/tmp/opc-audio-stream` and
its socket at `/tmp/opc-audio.sock` (override with `--reader` and `--socket`). It
refuses capture if the adapter is unavailable, if frame validation fails, or if
the SDK reference lacks a continuous paired sample grid. The complete framed
stream, all microphone samples and the common microphone/reference interval
remain archived.

`--legacy-fifo` explicitly enables an exploratory fallback for diagnostic 16 kHz
sources with `--allow-rate-switching`. It cannot satisfy native 48 kHz admission.
That unframed tap
can drop data when its nonblocking writer cannot deliver samples. A byte count,
plausible waveform or pilot fit cannot prove sample continuity. Do not run another
FIFO reader concurrently; the FIFO does not broadcast a copy to every consumer.
The fallback's lock protects cooperating acquisition readers, not arbitrary
legacy readers. It cannot supply verified shot timing.

Device-side framed microphone/reference support is a separate, provisional
instrumentation path in [the interceptor](../sim/acoustics/device/tap.c) and
[stream reader](../sim/acoustics/device/reader.c). Source code or a successful build is not a verified device
capability. Before relying on it, validate the target firmware's reference
meaning, frame sequencing, sample counters, common grid and behavior under reader
backpressure. No preload installation or firmware alteration is implied by the
commands above; the framed adapter must be installed and verified separately.
See the [device build, validation and rollback procedure](../sim/acoustics/device/README.md).

## Check individual shots and assemble signed gathers

Copy a captured session back to the analysis environment, retaining its complete
directory and hashes, then run:

```bash
python3 -m sim.acoustics.gathers \
  eval/acoustic_acquisition/comparison-session \
  eval/acoustic_acquisition/comparison-qc
```

The analyzer validates the capture inventory, raw PCM and playback assets before
processing. It detects the two pilots jointly using their expected separation,
then uses one map for the complete packet. Individual room traces are not shifted
to make them agree. Per-shot checks include PCM-rail clipping, detectable probe
consistency, quiet-window noise and transients, signal-to-noise ratio, search
boundaries, packet timing residuals and repeated-shot coherence. PCM rail tests
cannot rule out analog saturation or nonlinear loudspeaker distortion.

The output contains:

| File | Contents |
|---|---|
| `report.json` | Every planned capture, rejection reasons, shot QC, timing assumptions, gather row identities, paired comparison summaries and hashes |
| `shots.npz` | Unstacked signed raw cuts, source/reference cuts, deconvolved responses, negative-lag support and ADC sample ranges |

Accepted gather matrices retain separate repetitions. Rejected or missing
captures stay visible in the inventory; missing receivers are not filled with
fabricated zero traces. No stacking occurs before QC. Negative-lag responses are
preserved for later exponential-sweep distortion analysis, but a negative-lag
feature or ordinary spectral energy is not itself a validated harmonic diagnosis.

For a paired reference, the capture metadata must name the reference PCM file,
its SHA-256 hash and verified common-grid/continuity status. The reference must
have the same sample count as the microphone. Preserve the original framed
stream as evidence in addition to any decoded microphone and reference PCM.
Setting verification booleans by hand is not a calibration procedure.

Per-band SNR currently means total signal-window power divided by pre-pilot noise
power. It is a descriptive measurement, not a room-reflection SNR or an automatic
selection score. Compare clipping, repeat consistency, bandwise noise, source
energy and capture cost together. The report deliberately leaves `winner` unset;
three blocks at one pose do not establish a universally optimal signal.

For a volume session, run the gather analyzer first, then:

```bash
python3 -m sim.acoustics.volume \
  eval/acoustic_acquisition/volume-qc \
  eval/acoustic_acquisition/volume-analysis --plot
```

The optional plot requires Matplotlib. The volume report pairs trial levels
within each block and reports reference-visit drift separately. It distinguishes
commanded volume, observed submitted-AO-to-SDK-reference gain, absolute microphone
RMS, and microphone response divided by the SDK reference. Digital reference gain
does not measure loudspeaker sound pressure. A fixed response window is used for
all levels; no individual room response is shifted or scaled to improve agreement.
Rejected captures remain in the inventory and cannot contribute accepted gain
estimates. Microphone rail clipping is observable, but its absence cannot prove
absence of analog saturation, speaker distortion, or dynamic processing.
Stable response amplitude does not establish phase repeatability. Check the
unshifted signed response correlation between matching reference visits as well
as its norm. A delay inferred by matching room responses is a diagnostic; applying
that delay would remove possible propagation changes and is not an independent
calibration of the camera's inter-recording latency.

## Plan receiver gathers after choosing a tested probe

Select a packet ID from `probe_suite/suite.json` based on reviewed comparison
evidence. The explicit packet selection prevents accidentally mixing waveform
families in a source gather:

```bash
python3 -m sim.acoustics.survey \
  --suite eval/acoustic_acquisition/probe_suite_48k/suite.json \
  --calibration eval/world/calib.json \
  --mode gather --packet SELECTED_PACKET_ID_FROM_SUITE_JSON \
  --pans 1850 --tilts 520 610 699 790 880 --reference-tilt 699 \
  --blocks 3 --volume 30 --seed 731 \
  --output eval/acoustic_acquisition/gather-plan.json
```

Use the capture preflight, capture command and analyzer above with new plan,
session and QC paths. Additional pan values create additional fixed-source
blocks; they must not be merged as one source. A complete QC report still leaves
`inversion_allowed` false. Physical timing, acoustic-centre and directional
calibration, stable reference visits and controlled reflector evidence remain
separate prerequisites for geometry claims.

## Offline checks

```bash
python3 tests/sim/acoustic_probes_test.py
python3 tests/sim/acoustic_survey_test.py
python3 tests/sim/acoustic_capture_test.py
python3 tests/sim/acoustic_shot_qc_test.py
python3 tests/sim/acoustic_device_native_test.py
```

These checks use generated data and fake cameras. They verify software behavior;
they do not certify a physical source reference or a deployed camera build.
The native interceptor test requires Linux and exercises `LD_PRELOAD` against a
mock SDK with concurrent input/output callers; other platforms explicitly skip
that test. A passing native mock still does not verify the target camera ABI.
