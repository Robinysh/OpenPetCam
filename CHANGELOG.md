# Changelog

Releases are published from version tags; servers on `UPDATE_CHANNEL=release` install them within a day.

## [Unreleased]

## [0.3.7] - 2026-11-05

### Fixed
- The offline test gate called three acoustic tests that are not in the repository yet, so CI and the 0.3.5 and
  0.3.6 release builds failed (no release archive for either). The gate runs only tests that exist, and now includes
  the scheduled-sweep test. The site gained a blog.

## [0.3.6] - 2026-11-05

### Server
- Scheduled sweeps (`server/scheduled_sweep.py`): the server photographs a list of poses at a set time on its own,
  without the owner's computer, e.g. after a re-mount, at night (infrared) and in daylight. Each run has a hard time
  limit and pauses Follow while it runs. It skips, and says why, when privacy is on or another capture holds the
  camera. alert.py sends one notice when a sweep fails or has no result an hour after its time.

## [0.3.5] - 2026-11-05

### Room model
- Re-mounts are carried through (`sim/world/remount.py`). A pan/tilt pose only means something with the mount's
  rotation, so after a re-mount the new correction (from registering new snapshots to the panorama) starts a dated
  entry in `mount_offsets`: positions logged before the move keep the old correction, later ones get the new, in the
  server's position history and in the Room tab's view cone. Stored poses are re-aimed so each points where it did:
  the room map's search stops (plus a pose shift for its floor grid), the learned search stops and the saved views.

## [0.3.4] - 2026-11-05

### Fixed
- Follow no longer chases motion into a wall and stays there. A chase turn is skipped when an end stop would take
  most of it, or when the room map shows the pose has no floor; a chase that finds nothing goes back to where the
  camera was looking. (Three turns from her resting spot had run the camera to the end of its pan, facing the wall
  beside the mount, until a search minutes later; 5 of one day's 6 three-turn chases ended at an end stop.)

### Room tab
- The scripted-check handle (`window.__room`) also exposes the 3D scene, for recording the room's views.

## [0.3.3] - 2026-11-05

### Fixed
- The size-free detector no longer makes onnxruntime log a shape warning on every run (`tools/make_dyn_detector.py`
  drops the inner shapes recorded for 416x416; rebuild with `rm models/yolox_nano_dyn.onnx; tools/fetch_models.sh`),
  and the detector logs onnxruntime errors only.

## [0.3.2] - 2026-11-05

### Follow
- Follows her between detector runs (`server/fasttrack.py`, `tracker.FAST_FOLLOW`, on). The detector runs only every
  few seconds on a small server, and its box is about a second old when it arrives; the watcher already computes a
  motion image on every frame, and Follow now tracks her through it from the last verified box (frames since that
  box's picture are replayed when it arrives), standing in for a box that has gone stale while she walks or was
  missed. Replay on recorded walks with live detection timing (`sim/world/fastloop_replay.py`): she stayed within
  the re-aim edge 38% of the time while walking, against 17% before, with a small cost while she is still.
- The detector runs on the 16:9 frame without square padding (`watch.DETECT_RECT`): the same answers on recorded
  frames, 37% less detector time (`sim/world/detcost.py`). Needs `models/yolox_nano_dyn.onnx`, which
  `tools/fetch_models.sh` builds when the `onnx` package is installed (`tools/make_dyn_detector.py`); without it the
  square model runs as before.

### Room model and simulator
- Simulator in the measured room (`sim/home.py`), a leave-one-day-out gate (`sim/home_gate.py`), walks from video
  (`sim/world/walks.py`), synthetic camera views, and a local-only report (`sim/world/report.py`).
- Room tab: a colour legend on every view; the Radio view is painted by where the antenna points, with a line toward
  the router.

## [0.3.1] - 2026-11-05

### Room model
- Radio layer: the camera's WiFi link by head direction (`sim/world/radio_map.py`). The antenna turns with the head,
  so the link's strength depends on where it points; the map is drawn only where the head pointed and only if it
  repeats across interleaved halves, and the strongest-link bearing (usually toward the router) is marked
  unconfirmed. The Room tab gains a Radio view with that bearing as a dashed line.
- Radio-shadow test (`sim/world/rti.py`): whether the link changes with where the pet stands, within a fixed camera
  pose (without that control the turning antenna reads as her shadow). With one RSSI link it does not, so no radio
  image of the room is drawn.
- The Room tab's `probe()` lets a script check that the drawn marker, heat and view direction equal the data.

### Tests
- The JavaScript syntax test parses ES modules as modules (a missing bracket in `room.js` passed the old check).

## [0.3.0] - 2026-11-05

### Room model (new, optional: `sim/world/`)
- A measured 3D model of your room from every sensor the camera has: rotation-only pan/tilt self-calibration from
  its own sweep (ALIKED + LightGlue matches), drift of the logged pose corrected by registering later snapshots to
  the panorama, Depth Anything 3 depth of every frame fused in the gradient domain into one range per direction,
  scaled to metres by the camera's own floor echo, furniture placed by floor contact and carved by your pet's
  sightlines, near-infrared material and reflectance (active minus passive), sound coverage, WiFi motion, and a
  per-direction uncertainty with next-best-view planning to improve the weakest parts.
- Sound fills the camera's blind spots where it is shown to hold: the surface above the camera (the nearest echo
  agrees with vision) and the floor under it (marked assumed). Echo ranges elsewhere do not beat a direction-shuffle
  null, so they are not used; a verified 3D wave solver (FDTD) and near-field model are included with that result.
- `AUDIT.md`: the checking procedure for every claim (independent sensor, null, held-out split). Your results go in
  your own git-ignored `eval/<world>/`.
- Developer viewer (`sim/world/viewer/`): every layer in 3D over time, colour by any attribute, click any surface.

### App: Room tab
- Your room as measured, cut away where you look at it from behind; where your pet is or was at the time on the
  day strip; Time spent (that day or the last 7 days); where the camera points; every view of the model in a panel
  beside the room. Built from your room model by `python3 -m sim.world.app_bundle eval/world` into `models/room/`
  (installed on your server only, served to signed-in devices only). Without it the tab says how to build one.
- Server: `roomapi.py` computes your pet's floor positions from the watcher's own records with the room model's
  rule (`roomcam.py`, pinned to the model's camera by a test). three.js is vendored so the tab loads nothing from a
  third party, and it loads only when the tab opens. `tools/room_dev.py` previews the tab from your local data.

### Find
- `sim/world/find_map.py` builds `models/room_map.json` (likeliest views per hour, floor share per view, tiles worth
  zooming) that v0.2.2's search reads; `find_bench.py` measures Find and replays recorded searches offline.

### Privacy
- `tools/check_public.sh` also refuses `eval/`, room-model data or media under `sim/world/`, and any phrase in your
  git-ignored `config/private_terms.txt`.

## [0.2.3] - 2026-11-05

### Follow
- No live A/B: the watcher no longer alternates Follow policies in 2-hour blocks. Today's Follow constants run
  always, with the learned search stops (the v0.2.2 room-map replay measured them together). Over the week before,
  the logs gave the learned Follow constants 18% of watched time with her in view and 48 motor commands an hour,
  against 33% and 29 for today's (hours not matched). An override file can still force the learned constants or the
  PPO controller for a deliberate live test; it expires itself.

## [0.2.2] - 2026-11-05

### Find
- Zoomed look at each stop: when the whole frame shows nothing, the same detector and verifier (same threshold) run
  on half-size tiles of the stopped frame. A curled-up dog is too few pixels in the whole 640 x 360 frame. On 124
  hand-labelled frames of her resting spot: found in 11 of 21 before, 17 of 21 now, no false alarms in the 103
  frames without her. About 25 ms per tile on the server.
- Optional room map (`models/room_map.json`, built from this home's room model; the builder ships with the room
  model in 0.3). With it a search visits the views most likely to show her at this hour first, then the learned
  stops they do not cover, and skips views and sweep-row ends that show no floor (24% of recorded search frames).
  Replay of 56 recorded searches (map built without the searched day): her position in view at a stop in 48 (was
  36), median 16 s to that view (was 20 s). Without the file the search runs exactly as before.
- A failed search says what is likely instead of a flat "not found": "probably out of the room (not seen for N min)"
  and the place she is most often at this hour, marked "not confirmed". The "probably out" rule was right in 51 of
  62 recorded failed searches.

### Install
- `install.sh` installs `roommap.py` and `room_map.json`. A test now fails when a server module imported by another
  is missing from install.sh's list (it was, for 2 minutes of crash-looping after the first deploy of this change).

## [0.2.1] - 2026-11-04

### Find
- A sighting while the camera pans is placed where the camera pointed when that frame was taken, and the search goes
  back to her before confirming (at most once per place, three times per search). Report 2026-11-04: she lay in plain
  view, the verifier was sure three times while panning, but each frame arrived after the camera had panned on and
  every look at the stop showed the next view. Calibrated simulator, 12 held-out days: in view 0.465 -> 0.509.

### Live status
- The app's status is at most 1 s old (was 1-3 s): every new frame refreshes it (at most every 0.5 s, at once when
  motion starts). While nothing needs watching closely the camera pushes 2 frames a second instead of the watcher
  polling a snapshot every 3 s; they are decoded at 1/8 size for motion, so idle CPU stays about the same. Measured:
  median 0.31 s, worst 0.95 s.

### Agent goals
- `app_status_fresh` target 1 s (was 10 s).
- New `video_start`: opening the app to the first video frame, median <= 1 s and 90th percentile <= 2 s over 7 days.
  The app now reports the time since it opened on the first video of each open. Baseline (connect to first frame,
  174 starts): median 1.2 s, 90th percentile 4.6 s.

### From the owner's reports (11 waiting while the agent's daily worker budget was spent)
- The detector runs every 12 s when the picture is still (was 30 s): in 6 reports the app said she was not in view or
  out of sight while the report's own frame showed her, and the detector and verifier see her in it (raw 0.19-0.85,
  verifier 0.82-1.00); the answer was simply 30 s old.
- The app says "seen N s ago" once her last sighting is over 20 s old (was 30 s), instead of still saying "resting".
- Day/night flapping locks night for 20 min, doubling only if it flaps again right after (up to 2 h): a dawn flap held
  a 2 h night lock and the picture stayed infrared in daylight until 08:13.
- The agent's status-age metric reads the clock when it reads the status (it could read as -10 s).

### The self-improvement loop runs again
- It had stalled since 2026-11-03: 26 attempts, 7 shipped, 17 queued and never retried, 11 owner reports waiting up to
  21 h. The nightly job rewrote a tracked file (`agent/lessons.md`), so the checkout was never clean and the agent
  refused to retry or ship ("the code is being edited"). The file is now git-ignored (it also held report times from the
  home).
- A queued fix that no longer applies to the latest code is dropped and redone by a worker on the current code (it was
  retried, and failed the same way, every 2 minutes), and the owner is told.
- Owner reports have their own daily worker budget (12), separate from goal and meta work (10); up to 6 fixes a day go
  live (was 3, which had queued 11 of the 17).
- New goal `fix_queue`: the oldest queued fix ships within 24 h, so a stalled loop shows as a missed goal.
- `tools/agent_plot.py`: every goal over time against its target, and every improvement attempt with its verdict
  (`eval/superagent/agent_timeline.png`); the loop's schedule and what it measured are in docs/superagent.md.

### Tests
- `tests/server/agent_loop_test.py`: the loop cannot stall on its own housekeeping.
- `tests/server/flap_test.py`: the night lock.
- `tests/server/js_syntax_test.py`: every script in the app's pages parses (a syntax error would blank the app).
- `tests/server/search_overshoot_test.py`: the late mid-pan sighting case.

## [0.2.0] - 2026-11-04

The camera learns this camera and this pet, Find was rebuilt from an audit of real searches, and the live picture is
fresher. Everything below was measured on the reference install unless it says simulator.

### Learning (new)
- `sim/`: a simulator of this camera and pet fitted from the watcher's own logs: motor speed, start delay and gear play
  measured on the camera (`sim/sysid_camera.py`), where she rests and walks, how long she stays away, hard poses the
  detector rarely sees, the frame delay, and Claude's map of the room from the camera's own panorama (exits,
  furniture, rest places, bare walls). A fit is accepted only if it reproduces the real metrics on held-out simulated
  days: the current one passes 7/7 and reproduces Find's real success rate (40% vs 39%).
- Follow's constants and Find's search order are trained in it and must beat today's on held-out simulated rooms; the
  learned policy then alternates with today's in 2-hour blocks on the real camera (`track.policy`), scored by
  `sim/ab_report.py`. The current learned constants: in view 0.460 vs 0.437, fewer moves (simulator, 32 held-out rooms).
- A recurrent PPO policy exists but is off the camera: in its live smoke test it had learned to use searches as free
  looks. Every learned policy gets a self-expiring live smoke test (`/var/lib/petcam/policy_override`) first.

### Find
- A single sure sighting is followed up (aim at it, look again) instead of moving on, and sure sightings add up across
  looks by their position in the room; weak ones (a toy at night scored at most 0.956) count for neither.
- Confirms on 2 of up to 4 still frames, not the first frame the verifier accepts; a look waits for the camera's frame
  delay once, not before every frame.
- "Already in view" only from a fresh detection (an old one could answer Find without looking).
- A cancel, or a person moving the camera, stops a search at every motor command and look.
- Every search keeps the frame at each stop and every sighting with its scores (21 days): `tools/search_review.py` and
  the "Searches" queue in `tools/label_ui.py`. The learned search order visits likely places first.

### Follow
- Keeps the track through frames where a moving, blurred dog scores just under the verifier's cut-off (at most 8 s
  since the last confirmed sighting), shifts what it remembers after its own small moves, ignores boxes older than its
  last move, and does not chase motion right after she was seen lying still.

### Live picture and latency
- Live frames come from the camera's JPEG snapshots pushed over one SSH channel: on the same moves 0.95 s vs 1.40 s for
  the H.264 substream by day. The substream stays as the fallback (after a failed push, repeated short sessions, or no
  new frame for 10 s). Measured on the camera: ch1's extra delay is added in its encoder/RTSP output, not the image
  pipeline; no encoder setting removed it (CBR crashes the streamer).
- The substream reader has no fps filter and one output thread (each could hold a frame).
- A faster path (main-stream frames, 0.3 s old, while Follow is engaged) is built behind a CPU governor and off by
  default (`FAST_ENABLED`): the $5 server cannot afford it.
- Camera noise suppression off: the camera's CPU 67% -> 45% busy. Server CPU cut: integer grey for motion, audio floor
  every 2 s, one shared SSH connection.

### Detection
- The fine-tuned verifier's cut-off is calibrated from the pet's own photos (held out by day on 452 owner-labelled
  crops: 9 missed / 1 false accept, against 23 / 9 before).

### App
- Find becomes "Stop looking" while a search runs; Hold to talk sits left of the joystick; the saved view nearest the
  camera is highlighted; the clip player reports a clip it cannot play.

### Releases and tests
- A release install keeps the learned Follow policy (it used to retire it: releases carry no models).
- One offline test gate, `tools/run_tests.sh` (24 server tests and the simulator tests), runs in CI and in the release
  workflow; it needs no camera, server or private config. Checks that need a running install moved to `tests/onserver/`.
- The live A/B report no longer rewards losing her: visibility over all watched time, compared hour of day by hour of
  day, every motor command counted (`track.moves`), and the rotation shifts one block per day.
- The agent's shadow gate holds when its evidence is missing (a failed replay, too few labels) instead of shipping.

### Known limits
- The live A/B verdict needs ~3 days of the corrected blocks. Find still fails most often because she is out of the
  room or behind furniture; inferring that is the next step (docs/TODO.md).
- The $5 Lightsail nano runs out of CPU credits most afternoons (README, Server size).

## [0.1.0] - 2026-11-03

The first public release. Everything below runs on the reference install (Cinnado D1 camera, AWS Lightsail nano).

### Live
- Camera-first live view: full-bleed WebRTC video with what she's doing laid over it, tap to aim, joystick,
  landscape full screen, hold to talk.
- Find: one tap sweeps the room for her, looking where the camera already points first.
- Follow: turns after her when she leaves the picture; quieter motor speed for automatic moves.
- Pause: turns off the picture and the microphone together, and re-applies itself if the camera forgets.

### Her day
- Today: photo mosaic of the moments she was in view, day split, scrubbable timeline with barking, clips.
- Diary, nightly highlights reel, daily timelapse, wellbeing compared with her own normal, zones for bowls and bed.
- Web push alerts for barking, distress sounds, camera offline, blocked view and stopped video.

### Detection
- Two-stage detection: YOLOX-nano proposes, a verifier trained on your own camera decides (rejects plush toys).
- `tools/bootstrap_pet.py` labels crops with Claude, trains and tests on held-out days; `./deploy.sh` refuses to
  replace a trained verifier with the default.

### Runs on $5
- Idle mode: camera snapshots instead of decoding video when nobody is watching; detection every 8 s while she
  moves and every 30 s when still. About 5% of one core on a Lightsail nano (was 47%).
- 6 hours of rolling video; clips, photos and per-minute stats kept much longer.

### Looks after itself
- Freeze watchdog, streamer and audio self-heal, blocked-view check, day/night flap protection, absence alerter.
- Updates itself from GitHub releases daily, with a checksum, a health check and automatic rollback
  (`server/update.py`, `UPDATE_CHANNEL`, `UPDATE_REPO`). Version and last check in Settings.

### Setup and the app
- `./setup`: a guided install from a fresh server to the app on your phone (server, camera WireGuard values,
  passwords typed on the server, a final check).
- The app is OpenPetCam with its own icon set; your pet's name is set in the app on first open or in Settings.
- Themes: Cobalt, Golden and Meadow, each in System, Light or Dark.
- Open phones reload themselves after an update (iOS keeps Home Screen apps resident for days).
- Turning on alerts can no longer freeze the app if iOS stalls while subscribing.

### The agent (optional, runs on your computer)
- "Something wrong?" reports with the frame and the 90 s before it, answered in the app by an agent that ships
  tested fixes.
- Fixes proven on your install are offered upstream as pull requests (`tools/contribute.py`, `CONTRIBUTE=off` to opt out).
- `tools/finetune_pet.py` fine-tunes the pet verifier on Claude-labelled footage; the server runs it as mode `finetuned`.
- Goals measured every 30 min; changes judged by tests, held-out replay, shadow runs and a 24 h re-check; an
  archive of every attempt; the Agent tab shows goals, work and decisions.
