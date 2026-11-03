<p align="center"><img src="docs/assets/logo.png" width="140" alt="OpenPetCam logo"></p>

<h1 align="center">OpenPetCam</h1>

**A pet camera that knows your pet, runs on a $5 server you own, and keeps getting better on its own.**

Flash a cheap Wi-Fi pan/tilt camera with open firmware ([Thingino](https://thingino.com)), point it at a small
server, and you get a phone app with live video, your pet's day at a glance, and alerts. The camera finds and
follows your pet, tells it apart from its toys, and an agent you run on your own computer fixes what goes wrong.
Video never goes through a vendor's cloud.

It was built for one golden retriever and then generalized: everything specific to a home lives in one
git-ignored config file, and the pet's name is set in the app.

<!-- docs/assets/openpetcam.gif is a cut of the demo film (docs/demo/). -->

## What you get

**See her, always**
- **Live on your phone.** WebRTC video with sound, low latency, full screen in landscape. Tap the picture to aim,
  or use the joystick. It installs to the Home Screen like an app, with no App Store and no developer account.
- **What she's doing, over the picture.** Resting, moving, at the water bowl, out of view, with how long.
- **Find her.** When she's out of view, one tap sweeps the room and stops on her. It looks where the camera
  already points first, so it doesn't move the motor when she is already there.
- **Follow.** The camera turns after her when she walks out of the picture. Automatic moves use a quieter motor
  speed, and any move you make pauses following for 3 minutes.
- **Hold to talk.** Your voice plays on the camera's speaker.

**Her day, without reading**
- **Today:** a photo mosaic of the moments she was in view, a day split (active, resting, away), a scrubbable
  timeline with barking marked on it, and clips.
- **Diary:** two weeks of days, a nightly highlights reel and a daily timelapse.
- **Wellbeing:** active minutes, naps, restless nights and water visits, compared with her own normal.
- **Zones:** draw the bowls and the bed; visits are counted and bowl visits get a clip.
- **Alerts** as web push: barking, distress sounds, camera offline, view blocked, video stopped.

**It's really her**
- Two-stage detection: YOLOX-nano proposes, and a verifier trained on *your* camera's crops decides. On the
  reference install the plush toy fooled the detector all morning; the verifier rejects it.
- `tools/bootstrap_pet.py` collects crops, has Claude label them, trains, and tests on held-out days before
  anything replaces the default.

**Yours, and private**
- One pause button turns off the picture **and** the microphone.
- The camera dials out over WireGuard, so no ports are open at home. Every video and media path is behind the
  app's login. Family joins with a one-time invite link, not a shared password.

**It looks after itself**
- **Self-healing:** freeze watchdog, streamer and audio self-heal, blocked-view check, day/night flap protection
  and a separate absence alerter that pages when something goes quiet.
- **Updates itself** from GitHub releases once a day, checks it came back healthy, and rolls back if not.
- **Improves itself** (optional): an agent on your computer turns "Something wrong?" reports and missed goals
  into tested fixes. See [The agent](#the-agent-optional).

**Cheap.** It runs on a 512 MB AWS Lightsail nano at about 5% of one CPU core, keeps 6 hours of video and keeps
clips, photos and per-minute stats much longer.

## How it fits together

```
camera (Thingino, home Wi-Fi) ──WireGuard (camera dials out; no ports open at home)──▶ server
server: MediaMTX (RTSP in, WebRTC out, rolling 6 h recording)
        petcam-watch   detection, sounds, zones, clips, watchdogs, search and follow
        petcam-app     web app, auth, camera control, the agent dashboard
        petcam-alert   absence alerter and web push        petcam-update   daily release check
        Caddy          HTTPS; every video and media path is behind the app's login
your computer (optional): the agent, which proposes, tests and ships fixes
```

Details: [docs/architecture.md](docs/architecture.md). Camera quirks: [docs/camera-thingino.md](docs/camera-thingino.md).
CPU budget: [docs/performance.md](docs/performance.md).

## Set it up

You need:
- a Thingino-supported pan/tilt camera (reference: Cinnado D1, Ingenic T23N);
- a Linux VM with a public IP running Ubuntu 24.04 (reference: AWS Lightsail `nano_3_0`, $5/month);
- a Mac or Linux computer with `git`, `ssh` and `rsync`.

1. **Flash the camera** with Thingino ([supported cameras and instructions](https://thingino.com)), join it to
   your Wi-Fi and set its root password in the setup portal.
2. **Create the server.** Open TCP 22, 80 and 443, plus UDP 51820 (WireGuard) and UDP 8189 (WebRTC media), in
   the cloud firewall. Note its public IP and SSH key.
3. **Get the latest release and configure it:**
   ```bash
   git clone https://github.com/crimeacs/OpenPetCam.git && cd OpenPetCam
   git checkout "$(git describe --tags --abbrev=0)"     # the latest release, so it can update itself
   cp config/site.env.example config/site.env           # then fill it in
   ```
   Fill in the pet's name, pronoun and description, the server IP, a host name and the time zone. A free host
   name that needs no DNS is `petcam.<ip-with-dashes>.sslip.io`.
4. **Deploy:** `./deploy.sh`. It downloads the models (pinned by SHA-256), installs MediaMTX, Caddy and WireGuard,
   and starts everything. It prints the server's WireGuard public key.
5. **Connect the camera.** In Thingino's WireGuard page fill in address `10.77.0.2/24`, endpoint
   `<server IP>:51820`, peer public key (the key printed in step 4), allowed IPs `10.77.0.1/32` and keepalive
   `25`. Copy the camera's public key into `CAMERA_WG_PUBKEY` in `config/site.env`, then run `./deploy.sh` again.
6. **Set the passwords** on the server, interactively: `ssh -t <server> 'sudo /opt/petcam/set-secrets.sh'`.
   It asks for the web app password and the camera's web login, and optionally a new RTSP login (change the
   default `thingino`/`thingino`).
7. **Open** `https://<PUBLIC_HOST>` on your phone, sign in once, and *Share → Add to Home Screen*. Opened from
   the Home Screen it runs full screen and can send notifications. Invite others from Settings.

## Updates

Once a day (around 04:20 server time) the server checks this repository's
[releases](https://github.com/crimeacs/OpenPetCam/releases). When there is a newer one it:
1. downloads it and checks its SHA-256;
2. keeps a copy of what is running;
3. installs it with the same `install.sh` that `./deploy.sh` uses;
4. checks that the app answers and the watcher sees pictures again within 3 minutes;
5. puts the previous version back if not, and never retries that release.

Settings in the app shows the version and the last check. Your settings, passwords, keys, trained pet model,
recordings and history are not part of a release, so an update never touches them.

- **Turn it off:** `UPDATE_CHANNEL=off` in `config/site.env`, then `./deploy.sh`. The app still tells you when a
  release is out.
- **Your own fork:** `UPDATE_REPO=you/your-fork`.
- **Running your own changes:** a server deployed from a checkout that is not exactly a release tag (for example
  `v1.0.0-3-gabc1234`, or with uncommitted changes) is never overwritten automatically. `./deploy.sh` updates it.
- **Publishing a release** (maintainers): add a `## [X.Y.Z]` section to `CHANGELOG.md`, then
  `git tag vX.Y.Z && git push --tags`. The release workflow runs the checks and attaches the archive.

## Teach it your pet

```bash
python3 -m venv .venv && .venv/bin/pip install onnxruntime numpy pillow
.venv/bin/python tools/bootstrap_pet.py all      # collect crops, label them with `claude -p`, train, report
.venv/bin/python tools/bootstrap_pet.py deploy   # after reading the report
```

Out of the box the verifier asks "is this any dog breed?". The labeller is told about your pet and its lookalikes
(`PET_DESCRIPTION`, `PET_LOOKALIKES`); labels are cached, so a re-run only labels new crops. A trained verifier
replaces the default only once there are enough labels across at least 3 days to test it honestly on held-out
days, and `./deploy.sh` refuses to replace a trained verifier with the default. The 👍/👎 answers in the app are the
human labels used to check it. Method and numbers: [docs/evaluation.md](docs/evaluation.md).

## The agent (optional)

If you use [Claude Code](https://claude.com/claude-code), `bash tools/install_autopilot.sh` runs an agent on your
computer that treats what goes wrong as feedback and fixes it:

- **"Something wrong?"** in the app files a report with the frame, the 90 s of video before it and what the phone
  showed. The agent diagnoses it, writes a fix with a test, ships it through the checks below and answers you in
  the app.
- **Goals** in `agent/goals.json` (find success, CPU, freshness, reports answered, QA) are measured every 30 min.
  A missed goal becomes work.
- **Every change is judged by things it can't edit:** tests, a guard against removed assertions, a held-out replay
  of detection and search, a shadow run for anything that changes how frames become "your pet", a health check
  with rollback, and a check 24 h later. Rejected attempts are kept and shown to the next attempt.
- **Watch it** in the app's **Agent** tab: goals with 7-day trends, what it's working on, every decision.
  **Steer it** in plain words: edit `agent/steer.md`, or tell Claude Code.

Design and the research it draws on: [docs/superagent.md](docs/superagent.md).

## Secrets and privacy

| What | Where | In git? |
|---|---|---|
| Pet name, server address, time zone, SSH settings | `config/site.env` on your computer; non-SSH parts copied to `/etc/petcam/site.env` | no (git-ignored) |
| Web app password (PBKDF2 hash only), session key, camera web login | `/etc/petcam/app.env` on the server, mode 640, set by `set-secrets.sh` | no, never leaves the server |
| Camera RTSP login | `/etc/petcam/camera.env` | no |
| Web push keys, WireGuard private key | generated on the server by `install.sh` | no |
| Frames, crops, labels and reports from your home | `eval/` on your computer | no (git-ignored) |

Enable the guard once per clone: `git config core.hooksPath .githooks`. The pre-commit hook
(`tools/check_public.sh`) refuses any commit that contains your server's IP or host name, your camera's address,
a private key or other secret-shaped strings. Run `tools/check_public.sh --all` before publishing a fork.

Phones stay signed in with a device cookie; the server stores only its SHA-256 hash, and a device can be revoked
from Settings. Login attempts are rate-limited.

## Repository

| Path | What |
|---|---|
| `server/` | everything that runs on the server, plus `install.sh` and `update.py` |
| `config/site.env.example` | the one file you fill in |
| `deploy.sh` | sync and install from your computer |
| `tools/` | run on your computer: `bootstrap_pet.py`, `status.sh` (health report), `triage.py` (reports), `calibrate.py`, `replay.py` (held-out replay), `superagent.py` (the agent), `make_release.sh` |
| `agent/` | the agent's goals, steering, policies and prompts |
| `tests/server/` | simulations that run anywhere (tracker, presence, search, updates) and checks that run against a live install |
| `docs/` | architecture, camera notes, evaluation, performance, the agent, research |

## Limits

- **Dogs first.** The sound model listens for dog classes and the default verifier asks "is this any dog breed?".
  Cats would need different classes in `verifier.json` and `watch.py`.
- **One camera per server.**
- **Tested on one camera model** (Cinnado D1, Ingenic T23N). Pan/tilt geometry is calibrated per mount
  (`tools/calibrate.py`); some motor quirks may be specific to that model.
- **Detection runs every 8 s while she moves and every 30 s when the picture is still**, not on every frame, to
  fit the $5 server's CPU budget. Searches and follows detect continuously while the camera moves.

## Credits and licenses

OpenPetCam is Apache-2.0 (see `LICENSE`). It builds on:
- [Thingino](https://github.com/themactep/thingino-firmware) (camera firmware);
- [MediaMTX](https://github.com/bluenviron/mediamtx) (MIT);
- [Caddy](https://caddyserver.com) (Apache-2.0);
- [WireGuard](https://www.wireguard.com);
- [YOLOX](https://github.com/Megvii-BaseDetection/YOLOX) (Apache-2.0);
- [MobileNetV2 from the ONNX model zoo](https://github.com/onnx/models) (Apache-2.0);
- [YAMNet](https://github.com/tensorflow/models/tree/master/research/audioset/yamnet) (Apache-2.0);
- [pywebpush](https://github.com/web-push-libs/pywebpush).

The models are downloaded at deploy time, not stored in this repository.
