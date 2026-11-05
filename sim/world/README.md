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
```

Copy `model/room_map.json` to your server's `models/` folder (next to `follow_policy.json`) and deploy: the search
then uses it. Without it the search runs as before.

Walls are placed from marks you make on the calibrated panorama (`eval/<world>/picks.json`: the floor junction of each
wall, and object windows with a label such as `sofa`, `chair`, `bowls`, `stand`, `stool`). The sweep never moves the
camera while your pet is resting in view unless you pass `--owner-present`.

## Sensors (`sensors/`)

Each sensor declares its band, reads what the devices already produce (it never moves the camera or plays sound),
and returns typed layers (`layers.py`: panorama, frames, series, echoes, points, table, matrix, note), each with the
method it was measured by. Sensors with `needs_room = True` run after the room is fused.

| Sensor | Band | Layers |
|---|---|---|
| `VisibleCamera` | 400-700 nm | calibrated sweep panorama, hourly composites, every snapshot at its pose |
| `NightInfrared` | ~850 nm | night composite, night snapshots (monochrome frames = IR illuminator on) |
| `ActivePassiveNIR` | 940 nm | the camera's own IR light (active minus passive) and near-IR reflectance |
| `ActiveSonar` | 2.1-7.4 kHz | echo profiles per head direction from the raw microphone tap |
| `AmbientSound` | 20 Hz-20 kHz | sound level per minute, spectrum, the passive echo test |
| `WifiLink` | 2.4 GHz | the camera's link RSSI and its per-minute spread against picture motion |
| `WifiEnvironment` | 2.4/5 GHz | one passive scan from this computer (network names dropped) |
| `DepthAnything3Depth` | visible, learned | DA3-BASE depth of all sweep frames jointly, metres from the sound floor echo |
| `Presence` | derived | pet detections on the floor plan, dwell time per detection |

## How the pieces fit

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

## The viewer

Dollhouse (default): the room's fitted planes with the walls facing you cut away, furniture boxes, the pet's dwell
heatmap on the floor. Orbit: the full fused surface. Camera eye and Floor plan. "Colour the room by" shows any surfel
attribute (uncertainty, class, near-IR, sound support, dwell), and a click lists every modality at that point.

## Licences of borrowed methods

Used: ALIKED + LightGlue (BSD/Apache), Depth Anything 3 BASE (Apache), MoGe's gradient merge (MIT). Avoid for
licences: SuperPoint weights, MASt3R, UniK3D, CUT3R, DiffRIR (no licence), misuka (non-commercial); HorizonNet's code
is MIT but its Matterport3D-trained weights inherit non-commercial terms.
