# World model of a room

Builds a model of the room your camera watches from every sensor it (and the computer next to it) has: calibrated
pictures, near-infrared, the camera's own speaker and microphone, its WiFi link, and where your pet goes. You can view
it in 3D with a time slider, open it in Blender, and Find uses it to search better.

**Everything it writes stays in your world folder (`eval/<world>/`, git-ignored).** It is the inside of your home:
pictures, layout, times. Nothing here uploads it anywhere, and your results belong in `eval/<world>/AUDIT.md`, not in
this repository. `AUDIT.md` here is the procedure for checking each claim.

## Build your own

```bash
python3 sim/sweep.py --tag day                                           # 1. photograph the room (the camera pans/tilts)
python3 -m sim.world.camera eval/twin/day eval/world/calib.json          # 2. calibrate from the sweep (needs opencv)
~/.venvs/da3/bin/python -m sim.world.matches eval/world                  #    optional: better matches (ALIKED+LightGlue)
~/.venvs/da3/bin/python -m sim.world.sensors.geometry eval/world da3     # 3. optional: learned depth (torch, DA3)
python3 -m sim.world.build eval/world --collect [--tape 0.86]           # 4. pull server data, run every sensor, fuse
python3 -m http.server -d eval/world/model 8765                          # 5. open http://localhost:8765 (Dollhouse view)
blender -b --python sim/world/blender_export.py -- eval/world/model      #    optional: room.blend / room.glb
python3 -m sim.world.find_map eval/world                                 # 6. the room map for Find -> model/room_map.json
python3 -m sim.world.app_bundle eval/world                               # 7. the app's Room tab -> models/room/
                                                                         #    (+ acoustic_room.json: the initial acoustic model)
```

Copy `model/room_map.json` to your server's `models/` folder (next to `follow_policy.json`), keep `models/room/`
there too, and deploy: the search uses the map, and the app gains its **Room** tab (your room as measured, every colour view of the model, where the camera points, where
your pet is, where they spent the day, tap any spot). Both stay on your server and are served to signed-in devices
only. Without them the app and the search run as before. `python3 tools/room_dev.py eval/world` previews the Room tab
on this computer from your local data.

Walls are placed from marks you make on the calibrated panorama (`eval/<world>/picks.json`: the floor junction of each
wall, and object windows with a label such as `sofa`, `chair`, `bowls`, `stand`, `stool`). The sweep never moves the
camera while your pet is resting in view unless you pass `--owner-present`.

## The initial acoustic room

Step 4's pano-depth stage (`depthpano.py`) writes the clean metric range, `model/range_pano_metric.f32`: DA3
depth in metres (floor echo or tape), before any snapping to modelled walls. Step 7 builds
`model/acoustic_room.json` from it (`acoustic_room.py`) and copies it to `models/room/`: that range simplified to straight walls on the room's two perpendicular axes, the
floor at its measured height, the ceiling (assumed unless measured) and the furniture boxes. Each wall says
whether the camera imaged it (`imaged`) or it only bounds what was seen (`occlusion`). The wave solver,
echograms and geometry inversions start from this file and refine its numbers; they do not redraw the room.
Only directions the camera actually saw count, and nothing within 0.6 m (learned depth is inflated that close):
the camera's own mount, such as a counter right above the lens, goes in the file's `mount` block.
Rebuild it alone with `python3 -m sim.world.acoustic_room eval/<world>` (a world built before 7 Nov 2026 needs
`python3 -m sim.world.build eval/<world>` first, to write the clean range).

## Sensors (`sensors/`)

Each sensor declares its band, reads what the devices already produce (it never moves the camera or plays sound),
and returns typed layers (`layers.py`: panorama, frames, series, echoes, points, table, matrix, note), each with the
method it was measured by. Sensors with `needs_room = True` run after the room is fused.

| Sensor | Band | Layers |
|---|---|---|
| `VisibleCamera` | 400-700 nm | calibrated sweep panorama, hourly composites, every snapshot at its pose |
| `NightInfrared` | near-IR-sensitive, spectrum unmeasured | qualitative night composite and snapshots (night mode inferred from grayscale) |
| `ActivePassiveNIR` | near-IR, driver label `ir940` | LED off/on/off signal with acquisition evidence; approximate geometry-corrected response only for qualified measurements |
| `ActiveSonar` | 2.1-7.4 kHz | echo profiles per head direction from the raw microphone tap |
| `AmbientSound` | 20 Hz-20 kHz | sound level per minute, spectrum, the passive echo test |
| `WifiLink` | 2.4 GHz | the camera's link RSSI and its per-minute spread against picture motion |
| `WifiEnvironment` | 2.4/5 GHz | one passive scan from this computer (network names dropped) |
| `DepthAnything3Depth` | visible, learned | DA3-BASE depth of all sweep frames jointly, metres from the sound floor echo |
| `Presence` | derived | pet detections on the floor plan, dwell time per detection |

## How the pieces fit

Active IR uses signed numeric measurements and validity masks, never values
decoded from colourized overlays. Legacy captures remain qualitative because
their exposure, gain, processing and camera response were not verified. The
current firmware lacks exposed exposure/gain locks, so controlled acquisition
fails before changing the camera; explicit qualitative capture is available.
See [active IR acquisition and response](../../docs/infrared-response.md).

- **Camera** (`camera.py`): rotation-only self-calibration from the sweep's overlaps; drift of the logged pan/tilt is
  corrected by registering later snapshots to the panorama (`registration.py`). The world frame is left-handed
  (x right, y up, z forward); the viewer and the Blender export mirror one axis.
- **Floor height** from the camera's own floor echo; a tape measure overrides it (`--tape`).
- **Fused surface** (`depthpano.py`): learned depth of every frame merged into one range panorama in the gradient
  domain (each view's unknown scale cancels), scaled by the floor echo, photo-guided, snapped to floor and walls.
- **Objects** (`objects.py`): boxes from labelled windows, placed by where they touch the floor, sized with priors,
  and carved wherever the pet's feet were seen (sightlines are clear to her).
- **Surfels** (`fuse.py`): one record per viewing direction holding every modality and an uncertainty; `nbv.py` plans
  the views that would reduce it most, `remap_capture.py` takes them, `compare.py` checks the result.
- **Find** (`find_map.py`, `find_bench.py`): the room reduced to a search map (likely views per hour, floor share per
  view, tiles worth zooming), and the measurements and offline replay that judge a Find change.
- **Sound** (`acoustic_model.py`, `fdtd.py`, `nearfield.py`, `sonar_scan.py`): an interleaved, seeded scan plan run on
  the server within a permission window you set; a verified 3D FDTD solver (`python3 -m sim.world.fdtd verify`) and a
  near-field model of the camera body. Each must beat a direction-shuffle null before it may change the geometry.

For a separate offline experiment that preserves signed waveforms and fits one plane
with fixed-pivot geometry, see [acoustic waveform fitting](../../docs/acoustic-waveforms.md)
(`sim/acoustics/`). It uses saved raw captures, an analytic image-source model and
pose holdouts; prior FDTD fits are not used as calibration or ground truth.

## The developer viewer (`viewer/`)

The app's Room tab shows what an owner needs. This viewer shows everything the model holds, for building and
checking it. Measured (default): the fused surface, cut away where you look at it from behind. Planes: the fitted
floor, wall planes and furniture boxes (a simplification, for checking the fit). Camera eye and Floor plan. "Colour the room by" shows any surfel
attribute (uncertainty, class, near-IR, sound support, dwell), and a click lists every modality at that point.

## Licences of borrowed methods

Used: ALIKED + LightGlue (BSD/Apache), Depth Anything 3 BASE (Apache), MoGe's gradient merge (MIT). Avoid for
licences: SuperPoint weights, MASt3R, UniK3D, CUT3R, DiffRIR (no licence), misuka (non-commercial); HorizonNet's code
is MIT but its Matterport3D-trained weights inherit non-commercial terms.
