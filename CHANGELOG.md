# Changelog

Releases are published from version tags; servers on `UPDATE_CHANNEL=release` install them within a day.

## [1.0.0] - 2026-11-03

The first release. Everything below runs on the reference install (Cinnado D1 camera, AWS Lightsail nano).

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

### The agent (optional, runs on your computer)
- "Something wrong?" reports with the frame and the 90 s before it, answered in the app by an agent that ships
  tested fixes.
- Goals measured every 30 min; changes judged by tests, held-out replay, shadow runs and a 24 h re-check; an
  archive of every attempt; the Agent tab shows goals, work and decisions.
