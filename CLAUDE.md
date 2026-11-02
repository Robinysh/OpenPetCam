# OpenPetCam: working notes for Claude Code (and humans)

A self-hosted pet camera: Thingino camera → WireGuard → one small Ubuntu server running MediaMTX,
`petcam-watch` (detection), `petcam-app` (web app), `petcam-alert` (absence alerter) and Caddy.
Read `README.md` for the shape, `docs/architecture.md` for the internals, and
`docs/camera-thingino.md` before touching anything that talks to the camera.

## Where things are

- **Your install's settings:** `config/site.env` (git-ignored). The SSH host and key come from
  there, so use `tools/siteconf.py` / `set -a; . config/site.env` and never hard-code them.
- **Health in one command:** `tools/status.sh` (read-only).
- **On the server:**
  - `/opt/petcam/` holds the code;
  - `/etc/petcam/` holds `site.env` plus the secrets (`app.env`, `camera.env`, VAPID keys), readable by root and the `petcam` group only;
  - `/var/lib/petcam/` holds `petcam.db` (tables `obs` every 3 s for 30 days, `minute` forever, `events`, `devices`, `push_subs`), plus `live.json`, `snaps/`, `clips/`, `candidates/` and `reports/`.
- **Logs:** `sudo journalctl -u petcam-watch -u petcam-app -u mediamtx -u petcam-alert`.

## Time zones: check this before reading any timestamp

Three different clocks are involved, and a bug was once investigated in the wrong hour because of it:
- The server's **system clock is UTC**. `journalctl` and a bare `python3` show UTC.
- The services run with `TZ` from `site.env`. Event times, file names under `snaps/` and `clips/`, and the app all use that zone.
- Your phone and laptop may be in yet another zone. The camera burns its own clock into the picture.

Run `date` locally and on the server, and convert explicitly (`TZ=<zone> python3 …`) before matching
a screenshot to the database.

## Fixing a bug

1. **Get the evidence.** In the app, "Something wrong?" (under the live view) saves a report:
   - the frame and the 90 s of video before it;
   - what the phone showed;
   - the watcher's state, 15 min of detections and 3 h of events.

   `tools/triage.py list` shows the reports; `tools/triage.py pull` fetches them into `eval/reports/<id>/`
   with the server logs, and `summary.md` is the place to start. Page JavaScript errors:
   `tools/triage.py errors`. For a screenshot without a report, query `obs` and `events` for that
   minute (see the time zones section).
2. **Look at the pictures.** Open `frame.jpg` and pull frames from `clip.mp4` at the source frame
   rate (`ffprobe` it first). Decide what actually happened from the pixels, not from the logs.
3. **Reproduce in a test first.** Add a test under `tests/server/` that fails on the current code;
   `presence_test.py` is the pattern: a simulated sequence through the real state machine. Most run
   locally (`PETCAM_SITE_ENV=config/site.env python3 tests/server/<x>_test.py`); the ones that need
   the camera or recordings run on the server from `/tmp`.
4. **Fix, then deploy** with `./deploy.sh` (idempotent; restarting services drops live view for a
   few seconds). Confirm on the live system that the original symptom is gone, using the same
   evidence as step 1.
5. **Commit.** The pre-commit hook (`tools/check_public.sh`) must pass. Push only when the owner asks.

## Rules

- **This repository is public.** Never commit frames, clips, reports, labels, IPs, host names,
  e-mail addresses or people's names; all of these live in `eval/` and `config/`, which are git-ignored.
  Don't paste report contents into public GitHub issues; describe the bug in words.
- **Secrets stay on the server.** Never print `/etc/petcam/app.env` or `camera.env`, and never ask
  the owner to paste a password into chat. They run `sudo /opt/petcam/set-secrets.sh` themselves.
- **The server is tiny** (Lightsail nano: 512 MB RAM, about 5% sustained CPU, burst credits).
  - Don't run heavy jobs on it: model training, `bootstrap_pet.py` and evaluation run on the laptop.
  - Check `tools/status.sh` for burst capacity before load tests.
  - Prefer the camera's JPEG snapshots over decoding video.
- **Camera changes can hang the encoder** (stream settings, image flip). The freeze watchdog
  recovers it in about 2 minutes; check `journalctl -u mediamtx` afterwards, not only the live view.
- **Detector thresholds are fit to labels, never chosen by eye.**
  - The labels must come from outside the model: the app's 👍/👎 and Claude-labelled crops via `tools/bootstrap_pet.py`.
  - Split dev and test by day, and compare against the free baseline. See `docs/evaluation.md`.
- **Unattended pieces need a watchdog:** anything new that runs on a timer or loop writes a
  heartbeat that `alert.py` checks.
