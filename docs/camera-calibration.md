# Calibrate from saved pictures

The room camera uses a shared focal length, two radial distortion coefficients,
and a motor orientation model. Overlapping pictures can estimate these without
a printed target. Natural-scene consistency is not known-target optical truth:
close-object parallax and motor backlash can bias a rotation-only fit.

Keep photographs, per-device profiles and experiment reports in ignored `eval/`
directories. The commands below run locally and never move the camera.

Fit from one saved sweep and its existing ALIKED/LightGlue match archive:

```sh
python3 -m sim.world.calibration_selfcal \
  --sweep eval/twin/train --matches eval/world_train/raw/matches_lightglue.npz \
  --seed-profile eval/world_train/calib.json \
  --out eval/calibration/candidate.json
```

The fitter estimates the optical parameters and free frame orientations, then
fits motor constants with the optics fixed. A positive radial-derivative
parameterization keeps the full sensor inside an invertible lens domain. Every
fixed correspondence contributes symmetric pixel error; an optimizer-only
monotonic continuation penalizes predicted rays outside that domain. Saved
evidence records input/source hashes, convergence, active bounds and outliers.
A budget-limited solve is diagnostic output, not an accepted profile. An active
optical constraint is a reason to be cautious about the fitted edge behaviour.

The original motor-coupled fitter also has bounds, undownweighted physical
constraints, and positive penalties for invalid geometry. Its fixed-match
diagnostic CLI is `sim.world.calibration_fit`; it must report convergence before
its output is treated as a candidate. Neither fitter activates its output.

Then freeze the candidate and compare it with the previous profile on a
different sweep:

```sh
python3 -m sim.world.calibration_validation \
  --training-sweep eval/twin/train --sweep eval/twin/validation \
  --matches eval/world_validation/raw/matches_lightglue.npz \
  --profile previous=eval/world_train/calib.json \
  --profile candidate=eval/calibration/candidate.json \
  --out eval/calibration/validation.json
```

The validator rejects overlapping image hashes. An augmented capture that
reuses old pictures is not an independent sweep. It freezes optical parameters,
fits each capture's nuisance orientations on one set of image pairs, and scores
other pairs with disjoint feature coordinates. The same split and raw-image
outlier filter apply to every profile. Invalid predictions remain in the scored
denominator. Check solver convergence, edge errors, high-percentile errors and
coverage as well as the median; distinguish candidate selection from a final
untouched test when comparing many candidates.

The validator has an independent forward/inverse projection implementation.
Also check the fitted profile with the runtime camera's ray/projection
roundtrip across the entire image, since its iterative inverse has finite
accuracy. A better optical fit does not itself validate motor accuracy or
metric room dimensions.

Before replacing an app bundle's calibration, regenerate its dependent
panoramas, depth, registration and rebased room marks. A new calibration cannot
be substituted into a room whose geometry was built with the old profile.

Regression checks:

```sh
python3 tests/sim/camera_calibration_test.py
python3 tests/sim/calibration_validation_test.py
python3 tests/sim/world_test.py
python3 tests/server/room_test.py
```
