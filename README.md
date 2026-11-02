# OpenPetCam

A private, self-hosted pet camera that knows your pet. Flash a cheap Wi-Fi pan/tilt camera with
open firmware ([Thingino](https://thingino.com)), point it at a $5/month server you own, and get
a phone app with live video, your pet's day, and alerts. Video never goes through a vendor's cloud.

Built for one dog (the pet) and then generalized: everything specific to a home lives in one
git-ignored config file.

## What you get

- **Live view on your phone.** WebRTC with sound and low latency. Tap the picture to
  aim the camera, or use the joystick; landscape goes full screen. Installs to the home screen.
- **Hold to talk.** Your voice plays on the camera's speaker after a soft chime.
- **Is it really your pet?** A small detector (YOLOX-nano) proposes candidates and a second-stage
  verifier rejects plush toys. A repeatable bootstrap (`tools/bootstrap_pet.py`) has Claude label
  crops from your own camera and retrains the verifier for your pet.
- **Your pet's day.** Timeline, moments feed (comings and goings, barking, whining), clips, a
  nightly highlights reel, a daily timelapse and a two-week diary. Per-minute stats are kept forever.
- **Zones.** Draw the water bowl, food bowl and bed. Visits are counted and bowl visits get a clip.
- **Wellbeing.** Active minutes, naps, restless nights and water visits, compared with your
  pet's own normal over the last two weeks. A daily summary arrives at 7 pm.
- **Alerts** as web push: barking, distress sounds, camera offline, view blocked, video stopped.
- **Stays up unattended.** A freeze watchdog, a streamer self-heal, a blocked-view check,
  day/night flap protection and a separate absence alerter.
- **Remembers your phones.** Each phone gets a long-lived device session, and you invite family
  with a link instead of sharing a password. Privacy mode blacks out the picture.
- **Cheap.** It runs on a 512 MB / 5%-CPU VM. Video is kept for a rolling hour; clips and stats
  are kept longer.

## How it fits together

```
camera (Thingino, home Wi-Fi) ──WireGuard (camera dials out; no ports open at home)──▶ server
server: MediaMTX (RTSP in, WebRTC out, rolling 1 h recording)
        petcam-watch (detection, sounds, zones, clips, watchdogs)
        petcam-app (web app, auth, camera control)   petcam-alert (absence alerter, push)
        Caddy (HTTPS; every video/media path is behind the app's login)
```

Details: [docs/architecture.md](docs/architecture.md). Camera quirks: [docs/camera-thingino.md](docs/camera-thingino.md).

## Set it up

You need:
- a Thingino-supported pan/tilt camera (reference: Cinnado D1);
- a Linux VM with a public IP running Ubuntu 24.04 (reference: AWS Lightsail `nano_3_0`);
- a Mac or Linux laptop with `ssh` and `rsync`.

1. **Flash the camera** with Thingino ([supported cameras and instructions](https://thingino.com)),
   join it to your Wi-Fi and set its root password in the setup portal.
2. **Create the server.** Open TCP 22, 80 and 443, plus UDP 51820 (WireGuard) and UDP 8189
   (WebRTC media), in the cloud firewall. Note its public IP and SSH key.
3. **Configure:** `cp config/site.env.example config/site.env` and fill it in: the pet's name,
   pronoun and description, the server IP, a host name and the timezone. A free host name that
   needs no DNS is `petcam.<ip-with-dashes>.sslip.io`.
4. **Deploy:** run `./deploy.sh`. It downloads the models (pinned by SHA-256), installs MediaMTX,
   Caddy and WireGuard, and starts everything. It prints the server's WireGuard public key.
5. **Connect the camera:** in Thingino's WireGuard page, fill in:
   - address `10.77.0.2/24`;
   - endpoint `<server IP>:51820`;
   - peer public key = the server key printed in step 4;
   - allowed IPs `10.77.0.1/32`, keepalive `25`.

   Copy the camera's public key into `CAMERA_WG_PUBKEY` in `config/site.env`, then run `./deploy.sh` again.
6. **Set the passwords**, interactively, on the server:
   `ssh -t <server> 'sudo /opt/petcam/set-secrets.sh'`. It asks for the web app password and the
   camera's web login, and optionally a new RTSP login (change the default `thingino`/`thingino`).
7. **Open** `https://<PUBLIC_HOST>`, sign in once, and add it to your home screen. Invite others from
   Settings.

`./deploy.sh` is also how every later change reaches the server. It is idempotent and never
overwrites existing secrets or keys.

## Secrets and privacy

| What | Where | In git? |
|---|---|---|
| Pet name, server address, timezone, SSH settings | `config/site.env` on your laptop; non-SSH parts copied to `/etc/petcam/site.env` | no (git-ignored) |
| Web app password (PBKDF2 hash only), session key, camera web login | `/etc/petcam/app.env` on the server, mode 640, set by `set-secrets.sh` | no, never leaves the server |
| Camera RTSP login | `/etc/petcam/camera.env` | no |
| Web push keys, WireGuard private key | generated on the server by `install.sh` | no |
| Frames, crops and labels from your home | `eval/` on your laptop | no (git-ignored) |

Enable the guard once per clone: `git config core.hooksPath .githooks`. The pre-commit hook
(`tools/check_public.sh`) refuses any commit that contains your server's IP or host name, your
camera's address, a private key or other secret-shaped strings. Run `tools/check_public.sh --all`
before publishing a fork.

Everything that serves video or media sits behind the app's login, using Caddy `forward_auth`.
Phones stay signed in with a device cookie; the server stores only its SHA-256 hash, and a device
can be revoked from Settings. Login attempts are rate-limited.

## Teach it your pet

```bash
python3 -m venv .venv && .venv/bin/pip install onnxruntime numpy pillow
.venv/bin/python tools/bootstrap_pet.py all      # collect crops, label them with `claude -p`, train, report
.venv/bin/python tools/bootstrap_pet.py deploy   # after reading the report
```

The labeller is told about your pet and its lookalikes (`PET_DESCRIPTION`, `PET_LOOKALIKES`).
Labels are cached, so a re-run only labels new crops. A trained classifier replaces the zero-shot
default only once there are enough labels across at least 3 days to test it honestly on held-out
days. The 👍/👎 answers in the app are the human labels used to check it. Method and numbers:
[docs/evaluation.md](docs/evaluation.md).

## Repository

| Path | What |
|---|---|
| `server/` | everything that runs on the server, plus `install.sh` |
| `config/site.env.example` | the one file you fill in |
| `deploy.sh` | sync and install |
| `tools/` | run on your laptop: `bootstrap_pet.py`, `evaluate.py`, `calibrate.py` (motor steps per frame), `status.sh` (health report), `streamcfg.py`, `fetch_models.sh`, `check_public.sh` |
| `tests/server/` | checks that run on the server against the live install (auth leak, zones, follow, self-heal, privacy, end to end) |
| `docs/` | architecture, camera notes, evaluation, research |

## Limits

- **Dogs first.** The sound model listens for dog classes and the default verifier asks "is this
  any dog breed?". Cats would need different classes in `verifier.json` and `watch.py`.
- **One camera per server.**
- **Tested on one camera model** (Cinnado D1, Ingenic T23N). Pan/tilt geometry is calibrated per
  mount (`tools/calibrate.py`, `PAN_STEPS_PER_FRAME` and `TILT_STEPS_PER_FRAME`); some motor quirks
  may be specific to that model.

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
