# Launch posts

Posted: Thingino Discussions 2026-11-03 (https://github.com/themactep/thingino-firmware/discussions/1914). Show HN 2026-11-03 (https://news.ycombinator.com/item?id=50360077): auto-flagged within a minute (account karma 4); emailed hn@ycombinator.com for a review 2026-11-03 ~19:40 ET (owner approved); never repost. r/selfhosted: the browser extension blocks Reddit; owner posts by hand. 2026-11-04: Self-Host Weekly Project Launch submitted (owner's own text) and the same text emailed to WLTechBlog (josh@wltechblog.com). Reddit replies (section 6) and directory listings (section 7) wait for the owner. Baseline before launch: 9 stars, 0 forks.

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

## 4. Self-Host Weekly (selfh.st/submit → Newsletter → Project Launch)

- **Project Name:** OpenPetCam
- **Link:** https://github.com/crimeacs/OpenPetCam
- **Does the project leverage AI to assist with development?** Yes
- **Description:**
  OpenPetCam is a self-hosted pet camera for Thingino-flashed pan/tilt cameras. The camera connects out over
  WireGuard to a small server (it runs on a $5 Lightsail nano), which relays live WebRTC video to a phone web app,
  keeps a rolling recording and runs the detection: YOLOX-nano, then a small classifier fine-tuned on your own pet
  so toys don't count as the dog. When the pet walks out of view, the camera searches the room and stops on them.
  Barking alerts, a daily timeline, and a privacy switch that turns off the picture and the microphone.
  Apache-2.0, v0.1.0.

## 5. Email to WLTechBlog (Josh), from the owner's Gmail

**To:** josh@wltechblog.com (his address in public Thingino commits)
**Subject:** A pet camera built on Thingino (Cinnado D1)

Hi Josh,

Your no-tools installer videos are how I got into Thingino, so thank you. I built something on top of it that you
might find interesting: OpenPetCam, an open-source pet camera for Thingino pan/tilt cameras.

The camera dials out over WireGuard to a $5 VPS, which serves live video to a phone app and runs dog detection.
When our dog walks out of view, the camera searches the room and finds her. It runs on a Cinnado D1 today. I'd like
it to work on more Thingino cameras, so I wrote up what I hit along the way (inverted tilt commands, CGIs that need
compact JSON, the prudynt restart freeze) at
https://github.com/crimeacs/OpenPetCam/blob/main/docs/camera-thingino.md

Repo: https://github.com/crimeacs/OpenPetCam

If it's ever useful for a video, I'm happy to answer questions or test a camera you're working on. No reply needed
either way.

Artemii

## 6. Reddit replies (the owner posts them; say you built it)

**r/homeassistant, "Help setting up Cinnado D1 cam running Thingino"**
(https://www.reddit.com/r/homeassistant/comments/1up2nbu/)

I run a Cinnado D1 on Thingino every day, so here's what fixed the same two things for me.

Choppy stream: the D1's main stream defaults to a long keyframe interval. In Thingino's Streamer settings set
stream0 to GOP 25 with max GOP 25 (one keyframe a second, at 25 fps) and VBR. If HA only needs a preview, point it at
the substream (ch1, 640x360), which is much lighter.

Pan/tilt: the D1's motors aren't exposed through ONVIF PTZ in a way HA picks up. What works is the camera's own
`motors` command over SSH (`motors -d g -x <steps> -y <steps>` for a relative move), which you can wrap in a
shell_command in HA. Heads-up: on the D1 the tilt direction is inverted for commands, so flip the sign of y.

I wrote these quirks up while building an open-source pet camera on this exact camera, in case it helps:
https://github.com/crimeacs/OpenPetCam/blob/main/docs/camera-thingino.md (I'm the author.)

**r/puppy101, "Pet camera recommendations" (no subscription, see and hear the pup)**
(https://www.reddit.com/r/puppy101/comments/1w18ops/)

If you're comfortable with a weekend tinkering project: a cheap pan/tilt camera reflashed with Thingino (free, open
firmware) plus a small server gives you live video and sound with no subscription and nothing going through a
vendor's cloud. I built an open-source app for exactly this, OpenPetCam (I'm the author), which also points the
camera at the dog on its own: https://github.com/crimeacs/OpenPetCam
If you'd rather not tinker, a camera with local SD-card recording is the simpler no-subscription route.

**r/Wyze, "Remove Wyze's grasp on your cameras for free forever with Thingino!"**
(https://www.reddit.com/r/Wyze/comments/1t00r28/)

One more thing you can do once a camera is on Thingino: I've been using a Thingino pan/tilt cam as a pet camera that
follows our dog around the room, with live video on the phone and barking alerts, all self-hosted. It's open source
if anyone wants to try it on their own camera: https://github.com/crimeacs/OpenPetCam (I'm the author; tested on a
Cinnado D1 so far, reports from other cams very welcome.)

## 7. Directory listings (the owner creates the accounts; these are the fields)

- **Name:** OpenPetCam
- **Tagline:** Self-hosted pet camera that finds your dog. No subscription, no vendor cloud.
- **URL:** https://github.com/crimeacs/OpenPetCam · **Website:** https://crimeacs.github.io/OpenPetCam/
- **License:** Apache-2.0 · **Platforms:** Self-hosted (Linux server), web (PWA on iPhone/Android)
- **Categories / tags:** Home & pets, video surveillance, self-hosted, privacy, computer vision
- **Alternative to:** Furbo, Petcube, Wyze Cam (with Cam Plus), Ring indoor cam
- **Description:** (same as the Self-Host Weekly description above)
- **Where:** selfhost.directory/submit (free account), alternativeto.net (add as an alternative to Furbo and
  Petcube), openalternative.co (free queue).
