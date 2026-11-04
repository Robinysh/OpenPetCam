# Open work, in order

Worked top to bottom. Each item moves to Done with the commit or date that shipped it.

1. **Live A/B verdict** (gate 3). The corrected report (`sim/ab_report.py`) needs ~3 days of blocks with the
   day-shifted rotation; then promote or remove the learned arm (constants retrained 2026-11-04 on the validated
   simulator). Add labelled spot checks (is she really in view?) as the ground truth the telemetry cannot give.
2. **Find v2: infer where she is before sweeping.** In the calibrated simulator half the automatic searches start while
   she is out of the room. Track where a lost track ended (Claude's exits on the room map): if she left through a
   doorway, watch that doorway and say so ("left through the corridor 12 min ago") instead of sweeping; if she lay down
   by furniture and never came out, say "probably behind the sofa" and look there first. Measure in the simulator and
   on the saved search records before shipping.
3. **Hard-pose detection at search stops**: accept weaker detector proposals during a search (the verifier is the real
   gate), and try the 1080p frame at each stop. Measure offline on the saved search frames, labelled by eye.
4. **Latency, measured on a box that is not throttled** (`docs/performance.md`): the four-way ffmpeg comparison
   (fps filter x output threads), corrected ch1 against the JPEG push on the same moves, the camera's native
   `prudyntctl mjpeg`, then ch1's encoded-output queue on the camera (`IMP_Encoder_GetMaxStreamCnt`/`Query`) only
   if ch1 still lags. 30+ paired trials per condition, day and night, with CPU steal recorded.
5. **Server with sustained CPU** (owner's choice): the $5 nano is throttled most afternoons. Then `FAST_ENABLED`
   (main-stream frames 0.3 s old while Follow is engaged) can be measured and switched on.
6. **PPO back on the camera only after its execution path is shared with the tuned controller** (outside review): one
   execution layer that enforces the manual pause, cancellation, the per-move budget (counting each motor command of a
   search) and frame requirements, used by the server and the simulator alike; the fast-detection action either works
   live or is removed from training; the resting guard works at the real detection cadence.
7. **Re-fit the simulator on logs recorded with DETECT_STILL = 12 s** (changed 2026-11-04; `sim/gate1.py` still mirrors
   30 s, the cadence of the logs it was fitted to). Then **predict a held-out real day** with one frozen code version (`sim/validate_real.py`), calibration exports that
   keep `arm@version`, and a distribution of fits rather than one.
8. **On-camera tracker** (owner, 2026-11-03): a tiny model inside prudynt's frame path (a hardware-scaled ~96x54 NV12
   channel) tracking 5-10 times a second next to the motor. First student: 11,851 params, 1.5 MMAC/frame, held-out recall
   0.88 / precision 0.57 against the server's detector. Ship gate: camera under ~70% busy, beats the server path in the
   simulator and in a live A/B.
9. **Fine-tune the pet classifier on weeks of footage** (owner, 2026-11-03), held out by day, the 118-frame walk set
   at 0 false accepts, a shadow run before it ships.
10. **Agent**: weakness mining over transcripts and reports; a pairwise judge for changes with no metric (copy,
    prompts); a fixed budget per experiment; rebase queued fixes instead of dropping them.
11. Motor loudness per speed (`tools/noise_study.py`), when a safe moment allows.

## Done

- 2026-11-04: Find follows up a single sure glimpse and adds sightings up across looks; a cancel stops a search at
  every motor command and look; "already in view" only from a fresh detection; the simulator models hard poses and the
  watcher's frame-freshness rule and passes gate 1 on held-out seeds (7/7); the A/B report scores visibility over all
  watched time; live frames from the camera's JPEG push; a release keeps the learned policy; one test gate for CI and
  releases.
- 2026-11-03: fine-tuned verifier with a cut-off calibrated from her own photos; Find needs 2 of 4 frames; Follow keeps
  weak tracks; search evidence kept and reviewable; server CPU cut (integer grey, audio floor, one SSH connection).
- 2026-11-02: camera-first Live and visual Today; the report agent; QA suite `tools/qa.py`; Pause turns off the picture
  and the microphone; toy rejection.
