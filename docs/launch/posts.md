# Launch posts

Posted: Thingino Discussions 2026-11-03 (https://github.com/themactep/thingino-firmware/discussions/1914). Approved, waiting for the browser: Show HN, r/selfhosted. Baseline before launch: 9 stars, 0 forks.

Facts used, each checkable in the repo: $5 Lightsail nano server; Thingino-flashed pan/tilt camera (tested on one
model, Cinnado D1); YOLOX-nano + a MobileNetV2 verifier fine-tuned on your own camera's pictures; 26 ms + 11 ms per
frame on the $5 box; about 5% of one core; Apache-2.0; WebRTC to the phone; no vendor cloud; v0.1.0.

## 1. Show HN

**Title:** Show HN: OpenPetCam – a self-hosted pet camera that goes looking for your dog

**URL:** https://github.com/crimeacs/OpenPetCam

**First comment:**

I wanted a camera for our dog that didn't stream my living room through a vendor's cloud, and that could answer
"where is she?" without me steering a joystick. OpenPetCam is what came out of that.

The camera is a cheap pan/tilt model reflashed with Thingino (open firmware). It dials out over WireGuard to a $5
VPS, so nothing is open on the home network. The server relays WebRTC to your phone, records a rolling 6 hours,
and runs the detection.

The part I find interesting is the "is that really her?" problem. YOLOX-nano happily calls a plush toy a dog, so
every box goes through a second model, a MobileNetV2 fine-tuned on your own camera's crops. The cut-off is set
from the weakest photo of your pet rather than a fixed 0.5: on held-out days a fixed 0.5 let 12 of 25 night
lookalikes through, and this rule let none through. A search only says "found her" when 2 of 4 still frames agree. Box plus check is 37 ms
on the $5 box, and the whole thing idles at about 5% of one core.

When she's out of view it sweeps the room (saved views first, then the pan rows) and stops on her. There's also
an optional agent: tap "Something wrong?" in the app, and Claude Code on your laptop reads the report and the
video around it, writes a fix with a test, and ships it only if the tests and a held-out replay pass.

Honest limits: tested on one camera model so far, the setup assumes you can run a VPS, and when she lies right
under the camera with her head out of frame the detector still misses her. I'd love to hear from people with
other Thingino cameras.

## 2. r/selfhosted

**Title:** I built a self-hosted pet camera: Thingino camera + $5 VPS, no cloud, finds the dog on its own

**Body:**

Repo: https://github.com/crimeacs/OpenPetCam (Apache-2.0)

- Camera: any Thingino-flashed pan/tilt camera (tested on a Cinnado D1). It connects out over WireGuard, so no
  ports are open at home.
- Server: Ubuntu 24.04 VPS with 512 MB of RAM (Lightsail nano, $5/month). MediaMTX for WebRTC, Caddy for HTTPS, a
  Python watcher for detection, all at about 5% of one core.
- App: a PWA (Add to Home Screen). Live video with sound, push alerts for barking or camera offline, a daily
  timeline and timelapse, hold-to-talk, invite links for family instead of a shared password.
- Privacy toggle turns off the picture and the microphone on the camera itself.
- `./setup` asks for the server IP and SSH key and does the rest. Updates install nightly from GitHub releases.

The detector runs on the server's CPU (YOLOX-nano, then a small classifier trained on your own pet so toys stop
counting as the dog).

Feedback very welcome, especially if you have a different Thingino camera.

## 3. Thingino community (GitHub Discussions, "Show and tell")

**Title:** OpenPetCam: pan/tilt pet camera app on top of Thingino (WireGuard + WebRTC + detection on a VPS)

**Body:**

Thanks for Thingino. It made this possible. OpenPetCam uses a Thingino camera's RTSP stream, the motor HTTP API
and the WireGuard client to build a pet camera that finds and follows the dog. Tested on a Cinnado D1 (T23N,
ATBM6012B). The camera notes cover the quirks we hit: inverted tilt commands, CGIs that only parse compact JSON, and
prudynt freezing after a streamer restart, at https://github.com/crimeacs/OpenPetCam/blob/main/docs/camera-thingino.md

If anyone tries it on another pan/tilt model, I'd like to add it to the supported list.
