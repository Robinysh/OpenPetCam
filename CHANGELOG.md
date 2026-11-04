# Changelog

Releases are published from version tags; servers on `UPDATE_CHANNEL=release` install them within a day.

## [Unreleased]

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
