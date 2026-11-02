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
