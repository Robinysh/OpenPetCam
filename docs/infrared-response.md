# Active infrared response

The camera measures reflected illumination. It does not measure thermal
emission or body temperature. A surface's near-infrared response can differ
from its visible colour, but one response does not identify a material.

## What the layers mean

| Product | Meaning | Required evidence |
| --- | --- | --- |
| Filter response | Qualitative JPEG change when the IR-cut filter switches | Historical pictures; exposure/processing may differ |
| LED brightness change | Qualitative encoded-pixel difference | Off/on/off pictures at one pose; not radiometry |
| LED-added signal | Signed linear response added by the LEDs | Verified fixed exposure, gain, processing, night mode/filter/pose, and a known camera response |
| IR response | Approximate distance/incidence-corrected signal, normalized once across compatible observations | Qualified signal plus range, surface normals and geometry-quality masks |
| Relative IR reflectivity | Relative material response under a calibrated lighting/camera model | Additional calibrated LED beam, camera shading and source geometry; not supplied by the current capture format |

The current firmware exposes gain **ceilings**, not verified exposure/gain
locks. Its processed JPEG response is also uncalibrated. The controlled runner
therefore refuses to start with this backend. Existing recordings stay
qualitative, and neither processing nor display upgrades them into reflectivity.
Grayscale alone does not verify that the LEDs were on, the IR-cut filter moved,
or the spectrum of the light. The `ir940` command is a driver label, not a
measured wavelength.

## Acquisition

`sim/world/ir_exp.py` is import-safe and captures at the current motor pose. It
does not pan or tilt. A sequence has off/on/off/on/off phases, each with repeated
frames, a shared exclusive capture lock and privacy checks. Night mode is
prepared once; only the LED state changes during the measurement sequence.
The original day/night policy, ISP mode, filter and LED state are restored and
read back even after errors or interruption. A restoration failure is recorded
as a failed capture.

Controlled mode requires a backend that actually supports and verifies exposure,
gain and processing locks and a calibrated linear/lookup-table camera response.
It must return real before/after control readbacks for every image. SDK symbols
not exposed by the installed firmware are not treated as usable controls.
An unchanged configuration signature alone is insufficient if adaptive local
tone mapping remains active: the verified response must describe the complete
recorded image pipeline under the illumination used.

On the server, explicitly qualitative diagnostics remain possible:

```sh
python3 /path/to/ir_exp.py /path/to/new-empty-capture --qualitative
```

This changes the camera's illumination temporarily and should be run when the
scene is suitable. The runner restores its original settings. It sends no camera
commands merely by being imported. Tests use an injected backend and never
access the camera. The experiment script is run separately on the server; it is
not installed by the normal app deployment. Its command-file option points to
the existing camera transport without putting credentials in command arguments.

Logs record the protocol, capabilities, original/restored states, restoration
result, each phase, actual timestamps, pose and control readbacks, and JPEG
hashes. IR-cut readback on the current device is a software latch, not mechanical
feedback, and is labeled accordingly. Missing evidence remains unverified.

## Processing and geometry

For an on frame between two off frames, interpolate the ambient baseline using
the actual capture times, then subtract in **linear** response units:

`signal = linear(on) - interpolated(linear(off_before), linear(off_after))`.

The inverse response is applied to each frame/channel before subtraction.
Automatic gain or tone changes cannot be repaired by assuming faraway surfaces
receive no LED light. Saturation, low signal, off-frame drift, burst motion,
changed controls, missing/mismatched frame hashes and incomplete captures are
masked. The off brackets do not rule out every on-only motion or specular effect.
The noise/drift threshold is a stability heuristic, not a calibrated statistical
confidence interval. With a response LUT, configured noise and drift limits use
its linear output units; their scale must match the response calibration.

For a matte resolved surface illuminated by an approximately point-like source,
the useful geometric correction is proportional to
`signal * light_distance_squared / incidence_cosine`.
It is the distance from the LED to the surface, not an extra return-path inverse
square. The math helper accepts an offset source and a calibrated combined
beam/sensitivity map. The current fusion uses a camera-coincident source
approximation, so unknown beam shape and camera shading remain in its response
proxy. It cannot produce the strict reflectivity layer.

Grazing/backlit normals are rejected instead of replacing their cosine with a
floor. Invalid ranges, depth discontinuities, insufficient multi-view depth
support and excessive relative depth spread are rejected when depth evidence is
available. Missing uncertainty is recorded. Range errors are amplified by the
squared-distance correction, and the geometry is approximate.

The signal, masks, noise estimates and observation counts are saved in
`model/nir_signal.npz`, with acquisition provenance in `nir_signal.json`.
`fuse.py` reads numeric arrays using their recorded grid dimensions. Colourized
JPEG/PNG overlays are display outputs only. Compatible exposure/gain/processing
scales are required before combining observations; the response is normalized
once over the accepted mosaic, not separately for each view.

Old/stale reflectivity textures are excluded by the app-bundle exporter and both
viewers unless the manifest explicitly supplies the required quality evidence.
The historical blog animations are labeled exploratory/qualitative. Rebuilding
the model updates these products without changing raw recordings.

## Checks

```sh
python3 tests/sim/nir_reflectance_test.py
python3 tests/sim/nir_capture_test.py
python3 tests/sim/nir_fusion_test.py
python3 tests/sim/nir_sensor_test.py
python3 tests/sim/nir_ui_test.py
```

These cover independent synthetic illumination physics, response-curve order,
readback qualification, changed controls, invalid geometry, restoration and
legacy-data rejection. Results about an installation belong in ignored `eval/`
folders. Synthetic checks validate implementation, not unmeasured device optics.

## Sources

- [Axis: IR in surveillance](https://whitepapers.axis.com/en-us/ir-in-surveillance): reflected IR, thermal distinction and day/night cameras.
- [Quéau et al.: LED-based photometric stereo](https://arxiv.org/html/1707.01018): near-source geometry, beam anisotropy, linear RAW response, optical falloff and fixed camera settings.
- [Choe et al.: NIR BRDF and surface geometry](https://publications.ri.cmu.edu/simultaneous-estimation-of-near-ir-brdf-and-fine-scale-surface-geometry): richer reflectance estimation uses multiple lighting/viewing directions.
