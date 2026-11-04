# Where the latency is (profiled 2026-11-02)

The profile was measured on the reference install: a Cinnado D1 running Thingino, WireGuard to an AWS
Lightsail nano in us-east-1, and a laptop on the east coast. Re-run it with:

```bash
scp tests/server/profile_latency.py <server>:/tmp/
ssh <server> 'sudo -u petcam /opt/petcam/venv/bin/python /tmp/profile_latency.py --n 12 --motor --video'
```

`--motor` nudges the camera and `--video` blanks the picture for about 3 s.

## Stage timings

| Stage | Measured | Notes |
|---|---|---|
| Laptop ↔ server | 59 ms TCP connect, 103 ms per HTTPS request | not the problem |
| Server ↔ camera over WireGuard | 16 ms ping, 67 ms for a static file | the camera's web server is slow even for static files |
| Camera control CGI (motor status, move) | 270–430 ms | about 80 ms is `motors` itself and about 90 ms is `auth.sh`; the rest is shell forks on a saturated CPU |
| Camera snapshot `ch1.jpg` | 300 ms to fetch, and the picture is about **1 s older** than the RTSP stream | the privacy-mode test turned the RTSP stream black at 0.74 s and the snapshot at 1.81 s |
| JPEG decode, motion, YOLOX-nano, verifier | 1 ms, 1 ms, **25 ms**, **14 ms** per candidate | Python is not the bottleneck |
| App `/api/live` | about 1 ms | |
| Video, camera → server | ≤ 0.7 s | an upper bound: the privacy command itself takes 0.67 s |
| Opening the live view | waits up to **4 s** for a keyframe | the camera sends one every 4 s; the first frame arrived after 4.6 s |
| Camera CPU | **load 4.8, 0% idle**; prudynt uses 78%; 3 MB of RAM free | everything that runs on the camera waits for the CPU |
| Server CPU | watcher 2.0%, MediaMTX 2.2% of one core; app and Caddy about 0 | 185 MB of 412 MB available |

## End-to-end budgets

**The pet moves, then the app knows it.**

| Step | Delay |
|---|---|
| Snapshot staleness | about 1 s |
| Snapshot fetch | 0.3 s |
| Waiting for the next watcher cycle (3 s period; a still pet is only re-detected every 8 s) | 0–3 s |
| Detector and verifier | 0.04 s |
| Waiting for the app's next check (every 2 s) | 0–2 s |

Typical total is about **4 s**, worst case about **6 s** (11 s for a pet that is lying still).

**A joystick touch, then you see the camera move.**

| Step | Delay |
|---|---|
| Phone to server | 0.06 s |
| Re-reading the position on the first touch after a pause | 0.3 s; **removed on 2026-11-02** |
| Move CGI | 0.28–0.43 s |
| Motor start | — |
| Video back to the phone | 0.2–0.7 s |

That comes to about 0.6–1.2 s now.

## Rewrite in Rust?

No. The server's own compute is about 40 ms out of a four-second budget, and the whole watcher uses
2% of a core. The time goes into:
- polling intervals (3 s, 8 s, 2 s);
- the camera's shell CGIs on an overloaded single-core chip;
- the stale snapshot path;
- a 4-second keyframe interval.

Rewriting the server in Rust would save tens of milliseconds. The streamer on the camera (prudynt)
is already C++.

## Plan, by gain per unit of risk

| # | Change | Expected gain | Cost and risk |
|---|---|---|---|
| A | Keyframe every 1 s instead of every 4 s | live view opens about 2 s sooner on average and recovers faster after packet loss | about 10% more bitrate; changing the stream settings can freeze the encoder for about 2 min (the watchdog recovers it) |
| B | Push the live state to the phone (server-sent events) instead of polling every 2 s | −1 s on average for status, the box and the pill | app-only change, low risk |
| C | Decode the 640x360 RTSP substream on the server instead of fetching snapshots; check motion on every frame; run the detector as soon as there is motion (at most 1/s) | −1.3 s staleness and fetch, −1.5 s waiting for a cycle: detection in about 0.5–1 s | +3.4% of a core at 15 fps (less at 5 fps), +56 MB RAM (tight on 512 MB); medium risk |
| D | A persistent SSH channel (dropbear) to the camera for motor and status commands | about 300 ms → about 100 ms per camera call | needs a key on the camera; medium risk |
| E | Free the camera's CPU: turn off prudynt's own motion detection (unused), check the Opus encoder settings, lower the substream fps | every camera call and snapshot gets faster; fewer frame drops | camera config change; measure the load before and after |

With A–E together: detection goes from about 4 s to about 1 s, the joystick from about 1 s to about 0.5 s, and opening the live view from up to 4 s to up to 1 s.

## Done 2026-11-02: measured after the changes

| What | Before | After |
|---|---|---|
| Camera motor status | 300 ms (CGI) | **53 ms** (persistent SSH shell, `CamShell` in app.py; the CGI is the fallback) |
| Camera move command | 280–430 ms | **about 60 ms** |
| First joystick touch after a pause | position read + move ≈ 0.6 s | move only ≈ 0.06 s on the server |
| Status reaching the phone | polled every 2 s | **pushed** (server-sent events); arrives 0.1 s after the watcher writes it, through Caddy unbuffered |
| Picture change → watcher has noticed and published | 3 s period + 1 s stale snapshot + 0.3 s fetch | **0.6–1.1 s** (substream decoded at 5 fps; motion wakes the loop; at most one cycle per second) |
| Keyframe interval (time to open live view, recovery after loss) | 4 s | **1 s** |
| Camera CPU idle | 0% (prudynt 71–78%) | **15–25%** (prudynt 46–67%): its own motion detection off, substream 15 fps CBR 1 Mbit/s → 5 fps VBR 300 kbit/s |
| Server CPU | watcher 2.0% + MediaMTX 2.2% of a core | watcher about 4% + MediaMTX 2.6% (decoding the substream) |

Lesson: the stream change stalled the camera's RTSP output while `ch1.jpg` kept updating, which no watchdog
noticed (the freeze watchdog looks for identical snapshots). The watcher now restarts the streamer after
90 s without live frames while snapshots work (`check_stream_stall`, `tests/server/stall_test.py`).

## Motor noise (measured with the camera's own mic, 2026-11-02)

| `motors -s` speed (steps/s) | 300-step move | Sound over the room | Strongest tone |
|---|---|---|---|
| 900 (Thingino default) | 0.50 s | +14.9 dB | 900 Hz (the step rate) |
| 600 | 0.68 s | +16.7 dB | 898 Hz |
| 400 | 1.22 s | +16.1 dB | 1199 Hz |
| 250 | 1.52 s | **+10.4 dB** | 1249 Hz |

Only 250 is clearly quieter. Automatic moves (follow, chase, search) use it (`app.QUIET_SPEED`); the
joystick stays at 900 so it feels instant. The mic sits on the camera body, so these numbers overstate
what the pet hears through the air; the comparison between speeds is what counts. `/etc/thingino.json`
has `accel_pan`/`accel_tilt` = 0 (no ramp); whether the legacy motor driver honours a ramp is untested.

## CPU budget on the $5 Lightsail plan (measured 2026-11-02)

The nano plan allows about 5% of its 2 vCPUs sustained, roughly 10% of one core; above that it runs on burst
credits. Those ran out at 18:18 on 2 Nov (balance 0.01%, CPU pinned at 5.0%, steal over 100%), and detection, the
search's looks and the app's status all slowed to a crawl. Measured per service over 5–10 minute windows
(systemd CPUUsageNSec, plus py-spy on the watcher):

| | before | adaptive detection + idle mode | + same-source motion |
|---|---|---|---|
| watcher incl. its ffmpeg decoders | ~37% of a core | 9.0% | 2.2% |
| MediaMTX (relay + 1080p recording) | 9.9% | 3.9% | 2.5% |
| total | ~47% | ~13% | **~4.8%** |
| steal | 105% | 16% | **0.4%** |

What made the difference:
- **Idle mode.** The substream decoder runs only while something needs live frames: motion, an active or
  recently seen pet, a follow move, a search look. Otherwise the watcher takes the camera's JPEG snapshot every 3 s,
  and MediaMTX pulls the substream only while it is read.
- **Adaptive detection.** The detector runs every 30 s on a still picture and every 8 s while she is active; any
  motion still triggers a run at once.
- **Motion only between frames from the same source.** A snapshot compared with a stream frame looked like motion and
  restarted the decoder within 3 s of every switch to idle.

The last window was mostly quiet. An active dog, live viewing and searches cost more and spend the banked credits.
Run `tools/qa.py` and the CPU measurement from this section after changes to the watcher.

## Control latency: from her move to the camera's move (measured 2026-11-04)

Measured as motor command -> first decoded frame that shows the move, alternating moves interleaved across decoders
so every variant sees the same moves, light and load (the hit threshold is 4x the pre-command frame noise, so it does
not depend on texture or light). Motor start itself is 50-120 ms; command send over the camera shell 21-40 ms.

| Path | Median |
|---|---|
| Substream ch1 (640x360, 5 fps), as the watcher decodes it | 1.0-1.1 s |
| Same, read straight from the camera, bypassing MediaMTX | 1.0 s (the relay adds nothing) |
| Same with ffmpeg `nobuffer`/`low_delay` | no change |
| Substream at 10 / 25 fps | 1.3 / 2.5 s: it gets WORSE with fps, a queue in the camera's ch1 pipeline |
| Main stream ch0 (1080p25) through `fps=5` | 0.8 s: ffmpeg's fps filter alone adds ~450 ms |
| Main stream ch0, every frame | **0.33-0.37 s** |

The chain while Follow tracks her today: picture age ~1.0 s, then up to 1 s until the next watcher cycle (MIN_CYCLE),
the detector (37 ms + 11 ms verifier) only when the picture changed, MOVE_EVERY 2.5 s between tracking moves and
POST_MOVE_IGNORE after each, then ~0.2 s of command and motor start. Detection -> motor start is already only
~0.1-0.2 s; the gap is the picture's age, the cycle cadence and the hold-offs.

Fast mode (watch.py, built and tested, **off**: `FAST_ENABLED = False`): while Follow is engaged, frames from the
main stream at 25 fps (every frame read, motion still compared 0.2 s apart) and a detector run every 0.5 s; frames
carry their capture time (`STREAM_LAG`) and the tracker's box time is put on the substream's clock so the tuned
constants keep their meaning. Cost: ~10% of a core to decode (`-skip_loop_filter all` saves ~20%; scaling 25 frames
costs the same as 5) plus ~10% for 2 Hz detection. A governor turns it off for 10 min when CPU steal passes 8%.
It is off because the box cannot pay for it: average use sits at or above the burst baseline, and on 2026-11-04 the
credits ran out (71% steal; the main stream decoded at 6.6 of 25 fps). Turning it on needs either CPU headroom
(cut the baseline load) or a camera-side fix for ch1's lag, which would cost the server nothing.

Where ch1's extra delay is (2026-11-04, from the camera itself): JPEG snapshots taken on the camera and streamed out
show a pan 0.49-0.60 s after the command on ch1 and 0.61-0.69 s on ch0 (each including the ~0.1 s motor start and
its own JPEG encode). ch1's raw frames are therefore fresh, while its H.264 stream is ~1.0 s old: the extra ~0.5 s
is added in ch1's encoder or RTSP output, which is configuration, not the image pipeline. The camera has ~2 MB of
free RAM and /tmp is RAM: never store test files there (a snapshot loop doing so rebooted it once).

**JPEG push (live, 2026-11-04).** No ch1 encoder setting removed its extra delay (measured, each restored: audio off
864 ms, clock overlay off 858, `imp_polling_timeout` 50 ms 1121, `buffers` 1 1618, `buffers` 4 1106, bitrate 1000 990,
against 860-1320 baselines; CBR crashes the streamer and must not be used). Because ch1's JPEG snapshots are fresh, the
watcher's live frames now come from the camera pushing ch1 snapshots over one ssh channel (`PUSH_*` in watch.py): on
the same moves 0.95 s vs 1.40 s for the substream by day, 1.7 s vs 2.6 s at night. ~4 fps, ~0.8 Mbit/s of home
upload while live frames are wanted, a few % of a server core. The camera-side loop holds one ~20 KB file in RAM, is
bounded to 15 min, stops when a write fails and kills a previous loop only if that pid really is one. A push that
yields no frame falls back to the substream for 10 min.
