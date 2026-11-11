# Raspberry Pi camera, no motors

Run all of OpenPetCam on one Raspberry Pi with a Pi camera module. No Thingino camera, no cloud server and no
WireGuard. The camera is fixed, so everything that moves it is turned off.

Tested on a Pi 5 (8 GB, Raspberry Pi OS Bookworm, 64-bit) with a Camera Module v2 (IMX219).

## What works, and what does not

| Works | Turned off (hidden in the app, refused by the server) |
|---|---|
| Live view, 6 h rolling recording, clips, timelapse | Find, Follow, joystick, tap-to-aim, saved places, recalibrate |
| Detecting your pet, naps and activity, zones, the day timeline | Scheduled sweeps (nothing to sweep) |
| Push alerts for the camera and for absence | Sound detection, barking, find by sound (no microphone) |
| Pause (releases the camera) | Hold to talk (no speaker), day/night and IR controls |

Tapping the picture still lets you **Mark** your pet, which helps the verifier learn.

## Settings (`config/site.env`)

```
CAMERA=rpi
PTZ=0                  # the default for rpi; PTZ=1 only if you wire up your own motors (not supported)
AUDIO=0                # the default for rpi
SSH_HOST=local         # this machine is the server: deploy.sh runs rsync + sudo locally, no SSH
UPDATE_CHANNEL=off     # only if you have changed the code; releases would overwrite it
PUBLIC_HOST=raspberrypi.<your-tailnet>.ts.net
PUBLIC_IP=100.x.y.z    # this Pi's Tailscale address (used for WebRTC)
```

The optional `RPI_CAM_*` settings (size, frame rate, bitrate, sensor mode, flips) are in `config/site.env.example`.
The Pi 5 has no hardware H.264 encoder, so MediaMTX encodes in software; keep the size and bitrate modest.

## Remote access with Tailscale

1. Install Tailscale on the Pi and on your phone, signed in to the same tailnet.
2. In the Tailscale admin console, under DNS, turn on **HTTPS Certificates**.
3. Put the Pi's `*.ts.net` name in `PUBLIC_HOST` and its `100.x` address in `PUBLIC_IP`.

`install.sh` sets `TS_PERMIT_CERT_UID=caddy` in `/etc/default/tailscaled`, so Caddy can fetch the certificate. Nothing is
opened on your router. Open `https://<PUBLIC_HOST>` on your phone while Tailscale is on.

## Install

```bash
sudo apt install rsync
cp config/site.env.example config/site.env     # then edit as above
./deploy.sh
sudo /opt/petcam/set-secrets.sh                # your app password (it does not ask for a camera password)
tools/status.sh
```

## How it differs from the Thingino setup

- MediaMTX reads the camera itself: path `cam` is the recorded H.264 stream with a clock overlay (the freeze watchdog
  watches it tick), path `sub` is a 640x360 MJPEG copy for the detector. The settings come from `/etc/petcam/camera.env`,
  rewritten on every install.
- The snapshot for the app and reports is one frame from `sub`; the watcher decodes `sub` only while it needs live frames.
- The watchdog's "restart the encoder" becomes "restart MediaMTX": the watcher touches `/var/lib/petcam/restart_camera`
  and `petcam-camrestart.path` (root) does it. "Reboot the camera" is not offered.
- Pause stops the `cam` and `sub` paths through MediaMTX's control API (localhost only), so there is no picture and
  nothing is recorded. The watcher re-confirms it every cycle, so a MediaMTX restart cannot quietly reopen the camera.
