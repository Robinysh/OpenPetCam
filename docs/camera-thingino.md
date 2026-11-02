# Camera notes: Thingino on a Cinnado D1

The reference camera is a Cinnado D1 (Ingenic T23N SoC, AltoBeam ATBM6012B Wi-Fi, SC2336 sensor,
8 MB flash) running [Thingino](https://thingino.com). Most of this applies to any Thingino pan/tilt camera.
Main stream 1080p 25 fps VBR <= 1600 kbit/s, sub stream 640x360.


- **Thingino's CGIs only parse compact JSON.** `{"username": "root"}` (with spaces, Python's
  default) is rejected as "Username required"; `{"username":"root"}` works. Always
  `json.dumps(..., separators=(",", ":"))` for `/x/login.cgi` and `/x/json-imp.cgi`.
- **Tilt is inverted in the motor config (`invert: 2`), but only for commands.** A tilt step
  moves the reported position the opposite way, and the driver range-checks as if it didn't:
  it accepts a step only while `y + cmd` stays in `[0, 1000]`, yet moves to `y - cmd`.
  Absolute tilt (`d=x`) therefore lands mirrored and can run past the end stop (it reached
  1030 and then refused every tilt move until homed). `app.py` works in reported
  coordinates, sends relative steps with the tilt sign flipped, splits long tilts into
  steps the driver accepts, keeps a 40-step margin, and auto-homes if it ever finds an axis
  out of range.
- **Wait for the motors to stop before computing a move**: positions read mid-move (or
  during homing, which takes ~12 s) send the next move to the wrong place.
- **Geometry** (measured by phase-correlating frames): one frame width ≈ 733 pan steps, one
  frame height ≈ 424 tilt steps. With the image flipped 180°, view-right = x down,
  view-up = y up. Re-measure with `tools/calibrate.py` if the mount changes.
- **Don't run the main stream at 15 fps.** Changing 25 → 15 fps (2026-11-01 23:12) broke
  recording all night without breaking live view: MediaMTX logged ~45/s "sample of track 2
  [audio] received too late, discarding" and 87 "detected drift between recording duration and
  absolute time, resetting". Clips lost their audio and "last hour" playback returned 404s.
  Back at 25 fps (VBR cap kept): 0 warnings, ~1.3 Mbit/s. Check `journalctl -u mediamtx` after
  any stream change, not just the live view.
- **A streamer restart can leave prudynt dead** (RTSP 554 and API 8080 closed, web UI up); a
  second restart fixed it. The watcher now self-heals that too: restart after 2 min of failed
  snapshots while the camera answers, again at 7 min, reboot at 12 min, at most one reboot/hour.
- **Changing stream settings through prudynt's API froze the encoder** (as did saving the
  image flip earlier). The freeze watchdog restarted it after 90 s. Expect ~2 min of outage
  after any streamer change; then `save_config` (see `tools/streamcfg.py`).
- **Day/night flapping when the camera faces something close** (2026-11-02 00:47, after it was
  turned toward a wall/ceiling by hand): its own IR light reads as daylight, so auto mode flips
  every ~34 s (thresholds: night < 25 for 6 s, day > 50 for 4 s, `/x/json-config-daynight.cgi`).
  Each flip clicks the IR-cut filter, which the mic heard as a "loud sound" (58 false events
  in 2 h), and swaps colour/IR, which reads as huge motion. The watcher now locks to night for
  2 h after 4 flips in 10 min (only while on Auto), and the audio detector drops single
  bursts <= 0.5 s.
- The camera burns a ticking clock into the picture, so a bit-identical picture for 90 s
  means the encoder hung. That is the freeze watchdog's signal.

## Audio

- **Camera mic is Opus** (48 kHz). It was AAC, and browsers' WebRTC can't carry AAC, so live view
  was silent. iPhone plays Opus-in-MP4 clips on iOS 17+. Switching took `{"audio":{"mic_format":"OPUS"}}`,
  then `save_config`, then a streamer restart. Until the restart the RTSP description still said
  AAC and MediaMTX logged "unable to decode ADTS".
- Mic volume 100 (was 80) of 120, gain 25 of 31, noise suppression 1 of 3.
- **Live player is our own WHEP client** (`index.html`), so the Sound button can unmute on iPhone. It
  reconnects on failure or an 8 s stall, and disconnects when the page is hidden.
- **Recording segments are 10 min**: each segment boundary drops ~0.4 s of Opus ("received too late").
- **Sound moments come from YAMNet** on episodes found by the loudness gate (dog classes 69-75:
  barking, howling, whining, growling, dog sound). Non-dog sounds become a moment only above
  -28 dBFS, labelled with what YAMNet heard. If the model can't load, the old loudness rule is the fallback.
