# Changelog

Releases are published from version tags; servers on `UPDATE_CHANNEL=release` install them within a day.

## [Unreleased]
### Changed
- The fine-tuned verifier's cut-off is calibrated from the pet's own photos (just below the weakest training photo
  of her) instead of a fixed 0.5. Held out by day on 452 owner-labelled crops: 9 missed / 1 false accept, against
  23 / 9 for the previous logistic-regression verifier and 0 / 16 at 0.5.
- Find says "found her" only when 2 of up to 4 still frames show her in the same place, not on the first frame
  the verifier accepts. Night lookalikes (a clothes pile, a slipper) could end a search on one infrared frame.
### Added
- Every search keeps the frame at each stop and every sighting, with each box and score (kept 21 days), so Find
  can be checked by hand: `tools/search_review.py` and the "Searches" queue in `tools/label_ui.py`.
- `tools/label_ui.py`: label crops by keyboard with an editable box.

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
