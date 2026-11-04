# How it works

## What runs where

| Piece | Where | What it does |
|---|---|---|
| Camera | home Wi-Fi | Any Thingino pan/tilt camera (reference: Cinnado D1, see [camera-thingino.md](camera-thingino.md)). WireGuard client `CAMERA_WG_IP` (10.77.0.2) → server. |
| Server | any Ubuntu 24.04 VM with a public IP (reference: AWS Lightsail `nano_3_0`, $5/month, 512 MB) | Set up and updated by `deploy.sh` → `server/install.sh`. |
| `wg-quick@wg0` | server | WireGuard `10.77.0.1/24`, UDP 51820. Camera dials out; no home ports open. |
| `mediamtx` | server | Pulls RTSP from the camera through the tunnel. WebRTC live view (UDP 8189), 1-minute fMP4 segments, **deletes anything older than 6 h**. Localhost only except WebRTC media. |
| `petcam-app` | server, `127.0.0.1:8080` | Web app (stdlib Python): login, remembered devices, invites, pan/tilt, picture modes, pet status, day reports, diary, push subscriptions. |
| `petcam-watch` | server | Every 3 s: camera JPEG snapshot → freeze watchdog → motion → YOLOX-nano dog detector (on motion or every 8 s) → pet state, events, clips. Audio → loud sound / barking. Nightly timelapse + retention. |
| `petcam-alert.timer` | server, every 3 min | Absence alerter (separate from the watcher): watcher heartbeat, video relay + recording freshness, new events → iPhone web push. |
| `petcam-update.timer` | server, daily ~04:20 | `update.py`: installs a newer GitHub release (checksum, snapshot, `install.sh`, health check, rollback). Leaves development installs alone. Status in `/var/lib/petcam/update.json`, shown in Settings. |
| `caddy` | server | HTTPS (automatic certificate for PUBLIC_HOST). Every video/media/API path goes through `forward_auth` to the app. |

Data on the server: `/var/lib/petcam/` (`petcam.db` SQLite, `snaps/` one per minute ×14 days,
`clips/` 30 s event clips ×30 days / 4 GB cap, `timelapse/` one per day ×365 days,
`highlights/` nightly reel of up to 8 clips ×90 days),
`/var/lib/mediamtx/recordings/` (rolling 6 h). Per-minute stats in `minute` are kept forever.

Secrets (never in git): `/etc/petcam/app.env` (password hash, session secret, camera web login),
`/etc/petcam/camera.env` (RTSP source URL), `/etc/petcam/vapid_*` (push keys).

## Is it really your pet? (two-stage detection + repeatable bootstrap)

On the reference install, YOLOX alone called the dog's **black plush toy** a dog all morning on 2026-11-02 (42 of 51 candidates at
>= 0.30 were the toy; the toy scored up to 0.435 while real dog crops went as low as 0.10, so no
YOLOX threshold separates them). Now:

1. YOLOX (score >= 0.10) proposes candidates; 2. the **verifier** (`/opt/petcam/models/verifier.json`
   + `mobilenetv2-12.onnx`, ~20 ms/crop) decides. Only verified candidates count as the pet.
3. The watcher saves frames with dog-like candidates to `/var/lib/petcam/candidates/` (max 1/30 s).

**Bootstrap (repeatable, on the Mac):**

```bash
cd tools
VENV=/path/to/venv-with-onnxruntime-pillow   # python with onnxruntime, numpy, pillow
$VENV/bin/python bootstrap_pet.py all      # collect -> label with `claude -p` -> train (no deploy)
$VENV/bin/python bootstrap_pet.py deploy   # after reading the train report
```

Labels come from `claude -p` reading numbered contact sheets (cached in `eval/bootstrap/labels.jsonl`;
a re-run only labels new crops). `train` reports the trained classifier, the free zero-shot baseline
(ImageNet P(any dog breed)) and YOLOX-only, on held-out days, and **chooses zero-shot until there are
>= 40 pet and >= 20 other crops over >= 3 days**. Current: zero-shot, threshold 0.34 (toy max 0.294,
dog min 0.393 on 16 Claude-labelled crops; no held-out test yet). See [evaluation.md](evaluation.md).

## Push-to-talk, zones, wellbeing

- **Talk:** Hold to talk (max 15 s) -> `/api/talk` -> ffmpeg (high-pass, level) + soft chime -> the
  camera fetches `http://10.77.0.1:8099/<token>.wav` over the tunnel (served by the app, bound to
  the WireGuard address only) and plays it: `curl … | play -s -f wav -v <talk volume>` via Thingino's
  `/x/run.cgi?cmd=<base64>`. Verified: the camera's own mic heard a test tone, and from a real iPhone (2026-11-02). Refused in privacy
  mode; 1 message / 5 s / device; logged as a "You spoke to <pet>" moment.
- **Zones:** rectangles drawn over a live frame (`/api/frame.jpg`), stored in `/var/lib/petcam/zones.json`
  with the camera pose they were drawn at (counted only within ±40/±25 steps of it). A visit = the
  bottom-centre of the pet's **verified** box inside the zone for 6 s (bowls) / 60 s (bed), one missed
  detection tolerated, ended by 20 s away, 5 min cooldown. Water/food visits get a clip.
- **Wellbeing** (`/api/wellbeing`, Today tab, 7 pm push): active minutes, longest nap, settle-downs,
  night activity (22:00-06:00), water/food visits, compared with the median of up to 14 previous
  days with real data. Says "Learning the pet's normal" until 3 such days exist.

## App design notes

- **Joystick is asynchronous.** `/api/nudge` answers in ~50 ms; one background sender talks to
  the camera (its motor CGI takes 200-400 ms). Ticks that arrive while it is busy are summed
  into one pending move, never queued, so the camera can't lag behind the thumb. Measured:
  8 ticks moved exactly 232 steps and 8 back returned to the same position.
- **Privacy mode** (eye button) blacks out the picture via prudynt `{"privacy":{"enabled":…}}`
  and writes `/var/lib/petcam/privacy`; the watcher stops detecting while it is set. Toggling it
  did not hang the encoder (tested).
- **Follow <pet>** (off by default, toggle under the joystick, stored in
  `/var/lib/petcam/settings.json`): the watcher re-centres on the pet when it is moving and two
  detections in a row put it near the edge; at most one move per 10 s, undershooting by 20%.
- **Blocked view**: the watcher flags "view_blocked" when the picture's sharpness (Laplacian variance,
  160x90) stays under 60 for 80% of 10 minutes (night 1: normal >= 132, wall at close range 14-30).
  Fired live at 03:40 on night 1. Skipped in privacy mode.
- **The alert checker re-reads the last 30 min of events and skips (id, ts) pairs already handled.**
  An id watermark broke when deleting the newest events made SQLite reuse ids; a time watermark
  would miss events stamped in the past ("left", sounds).
- **Days are local midnight to local midnight**, never `+ 86400`: 2026-11-01 (DST end) was 25 h
  and the extra hour crashed the timeline. Timeline slots are by local clock time.
- Chart colours (`--viz-active` `#c98500`, `--viz-rest` `#8a5c00`) pass the dataviz validator on
  the dark surface `#151517`.

## Server quirks

- **Lightsail `nano_3_0` has a ~5% CPU baseline and the box lives near it.** Above it,
  burst credits drain and the VM is throttled (CPU steal). Keep new work cheap; prefer
  the camera's own JPEG snapshots over decoding video; don't run MediaMTX `/list` on a
  timer (2.7 s CPU per call).
- **Lightsail counts inbound data against the 1 TB/month allowance.** At the factory
  4 Mbit/s CBR the camera alone would have used 1.4 TB/month; VBR ≤1.6 Mbit/s is ~0.5 TB.
- Ubuntu's `fwupd`, `snapd`, `multipathd` etc. don't fit in 512 MB; `install.sh` disables them.
- The web app must reset per-request auth state (`handle_one_request`): with HTTP keep-alive,
  Caddy reuses one upstream connection across users. Regression test: `tests/onserver/authleak.py`.

## Detector

YOLOX-nano (Apache-2.0, Megvii), COCO animal classes (a dog in IR is often scored as "cat";
"teddy bear" excluded because of plush toys). Present at score ≥ 0.30. Evidence and
evaluation: see [evaluation.md](evaluation.md).
