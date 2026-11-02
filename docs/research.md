# Open-source pet-cam research: gaps versus the pet cam

Researched 2026-11-02. Scope: open-source projects relevant to a self-hosted pet camera, compared with
the pet cam (Thingino/prudynt on a Cinnado D1 T23N, MediaMTX + Python on a Lightsail nano with a ~5% CPU
baseline). Each claim cites the page or file it came from. **(unverified)** marks anything I could not
confirm from a primary source. **(measured)** marks numbers I measured myself in this session. Star
counts come from the GitHub API on 2026-11-02.

## Status after this report (2026-11-02 morning)

- **Gap 6 done before the report landed:** the camera's mic is now Opus (it *was* AAC, so live
  view had no sound: 83 WebRTC sessions logged "skipping track 2 (MPEG-4 Audio)"). Live view
  has sound behind a Sound button.
- **Gap 1 done:** YAMNet (this report's recommendation, `ai-edge-litert`, 4,126,810-byte model,
  pinned SHA-256) now classifies every sound episode the loudness gate finds. Measured on the
  Lightsail box: **3.3 ms per window**, ~60 ms per episode. See evaluation.md for what it
  changed. Threshold 0.55 is provisional until the app's 👍/👎 labels are in.
- **Gaps 2, 3, 4 done (2026-11-02):** push-to-talk (camera-side `play` via the command runner, not the
  RTSP backchannel), zones bound to camera poses, wellbeing vs her own 14-day median. Gap 5 (live
  two-way talk) dropped at the owner's request. Gap 7 partly done: a toy-vs-the pet verifier (see README).

## TL;DR

1. **Two-way audio is already in the firmware.** Thingino's prudynt implements the ONVIF RTSP audio
   backchannel and offers **AAC, OPUS/48000/2, PCMU and PCMA**. Browser mic Opus can pass through
   without transcoding. **MediaMTX cannot do backchannel** (it skips the track). **go2rtc can.** There is
   also an on-camera `play` command (FIFO `/run/prudynt/audio_out`) for pre-recorded clips: wav, aac,
   opus, mp3, flac.
2. **YAMNet is the right bark model, and it fits the box.** It is what Frigate ships (Apache-2.0). The
   TFLite file is 4.1 MB and takes a 15600-sample float32 waveform (0.975 s at 16 kHz), giving 521
   scores. It measured **0.88 ms per window** single-threaded on an M1 Pro (≈40 MB RSS). Your watcher
   already decodes exactly that format (16 kHz mono s16le), so YAMNet can run only on loud episodes, at
   near-zero average cost.
3. **Zones (bowl, water, bed) are the cheapest new insight.** Monitora Pet does exactly this with
   YOLOX-Tiny ONNX (polygon zones, dwell time, cooldown). Frigate tests the box's bottom-centre point
   against the zone and uses inertia and loitering time. You already have YOLOX-nano boxes every
   3–8 s, so a zone needs no extra inference.

## 1. Projects studied

| Project | Link | Licence | Stars / last push | What it does | Worth borrowing |
|---|---|---|---|---|---|
| Thingino firmware | https://github.com/themactep/thingino-firmware | MIT | 2303 / 2026-11-01 | Open firmware for Ingenic cameras. Ships prudynt from `themactep/prudynt-t` (stable branch) per `package/prudynt-t/prudynt-t.mk` | `play` and `speaker` helpers, `playonspeaker` on motion (`package/prudynt-t/files/`), Opus enabled by default (`package/prudynt-t/Config.in`: `BR2_PACKAGE_PRUDYNT_T_OPUS default y`, AAC default y) |
| prudynt-t (Thingino fork) | https://github.com/themactep/prudynt-t (commit `2fe05c7` pinned by Thingino) | no licence file reported by the GitHub API | 3 / 2026-10-31 | The camera's RTSP streamer | `docs/webrtc.md` (backchannel and gateway matrix), `docs/audio.md` (audio-out FIFO, mic tap FIFO) |
| prudynt-t (upstream) | https://github.com/gtxaspec/prudynt-t | none reported | 56 / 2026-04-16 | Original streamer. README: "Two-Way Audio … using AAC and PCMU codecs" | Mic codec options in `prudynt.cfg.example`: OPUS, AAC, PCM, G711A, G711U, G726 |
| Frigate | https://github.com/blakeblackshear/frigate | MIT | 37150 / 2026-11-02 | NVR with object and audio detection | Audio pipeline (YAMNet TFLite, RMS gate, per-label threshold 0.8); zone semantics (bottom-centre, inertia, loitering) |
| go2rtc | https://github.com/AlexxIT/go2rtc | MIT | 14555 / 2026-10-26 | Camera streaming gateway | Two-way audio: browser WebRTC mic sent to the RTSP backchannel; `POST /api/streams?dst=cam&src=…` plays files on the camera speaker |
| MediaMTX | https://github.com/bluenviron/mediamtx | MIT | 20584 | Your current gateway | WebRTC read audio is Opus, G722 or G711 only (no AAC). fMP4 recording accepts Opus/AAC/G711. WHIP publish exists. No backchannel |
| Monitora Pet | https://github.com/lucascodebr20/monitora-pet | Apache-2.0 | 13 / 2026-11-01 | Local app logging cat/dog visits to food, water and litter zones from IP cameras | Polygon zones over video; dwell ("permanência"), tolerance and cooldown; YOLOX-Tiny ONNX; snapshot per confirmed visit; human review queue |
| Doggy Detector | https://github.com/dhabedank/doggy-detector | MIT | 3 / 2026-06-25 | Raspberry Pi bark monitor | YAMNet scores dog sounds; nearby barks grouped into incidents; clips with pre-roll; manual relabel. Uses full TensorFlow (too heavy for 512 MB) |
| Raspberry Pi Bark Logger | https://github.com/JakeSiddall/raspberrypi_bark_logger | none | 3 / 2025-06-27 | YAMNet bark logger | Uses `yamnet_float16.tflite` with TFLite on a Pi. Confirms the TFLite-only route |
| Dog Barking Detector | https://github.com/jatacid/dogbarkingdetector | MIT | 13 / 2025-12-12 | In-browser YAMNet (TF.js) | 1 s chunks at 16 kHz, dog classes 70–75, prefers bark/yip/howl over the generic "Dog" class |
| bark-detector | https://github.com/aidanmehal/bark-detector | Apache-2.0 | 1 / 2026-10-21 | Planned Rust bark/whine monitor on a Pi Zero 2 W | Design ideas (planned, not built): bark/whine/howl/uncertain labels, adaptive noise floor, manual relabel, hourly/daily vocalisation summary |
| TattleTail | https://github.com/emilyemorehouse/tattle-tail | GPL-3.0 | 14 / 2022-06-21 | Loudness-threshold bark push alerts | Same approach as your current heuristic (ambient-dB calibration, push throttling). Nothing new |
| PetCam | https://github.com/whiteSHADOW1234/PetCam | none | 24 / 2025-02-10 | WebRTC pet cam on an old phone, ESP32 stepper pan, "Talk to Pet" | Talk-to-pet over WebRTC on HTTPS. No licence, so read it but don't copy code |
| ha-furbo | https://github.com/willbeeching/ha-furbo | MIT | 5 / 2026-10-28 | Home Assistant bridge for Furbo (treat toss, pan, barking alerts) | Lesson: talkback on a second camera connection knocked out video, so it was disabled in 1.2.0 |
| Doggy Cam | https://github.com/danshagen/Doggy_Cam_Computer_Vision | MIT | 2 / 2022-02-04 | Pi + webcam motion "calm vs stressed" classifier with a paper and an evaluation harness | Hand-labelled reference + `evaluate_algorithm.py` loop for tuning an activity metric |
| Cat Prey Analyzer | https://github.com/niciBume/Cat_Prey_Analyzer | MIT | 168 / 2025-08-03 | Cascade of CNNs for prey detection at a cat flap | Cascade pattern: cheap stage gates the expensive one (you already do this with motion → YOLOX) |
| DogFaceNet | https://github.com/GuillaumeMougeot/DogFaceNet | MIT | 164 / 2026-04-29 | Dog face verification (triplet loss, ResNet) | Dataset on Zenodo/HF. Code needs TF 1.12, so it is not deployable as-is |
| wildlife-tools / MegaDescriptor | https://github.com/WildlifeDatasets/wildlife-tools, https://huggingface.co/BVRA/MegaDescriptor-T-224 | MIT (tools); **CC-BY-NC-4.0** (model, OK for personal use) | 81 / 2026-10-18 | Animal re-identification toolkit; MegaDescriptor-T is 28.3M params at 224×224 | Embedding + kNN gallery idea for "is it the pet". Too heavy to run continuously (see gaps) |

Not studied in depth: Scrypted (https://github.com/koush/scrypted, 6034 stars, licence NOASSERTION per
the GitHub API), Viseron (https://github.com/roflcoopter/viseron, MIT; its docs host failed TLS when
fetched) and motionEye (https://github.com/motioneye-project/motioneye, GPL-3.0). From what I could
find, none adds a pet-specific feature that Frigate and go2rtc don't already cover **(unverified)**.

## 2. Key findings

### 2a. Thingino two-way audio and speaker playback

Sources: https://github.com/themactep/prudynt-t/blob/2fe05c7dea9e35d8b7a6f13e9f683c9a7cb27a5e/docs/webrtc.md,
`…/docs/audio.md`, `…/src/audio/IMPBackchannel.hpp`, `…/src/network/RTSP.cpp`, and
https://github.com/themactep/thingino-firmware/tree/master/package/prudynt-t/files.

- **RTSP backchannel (ONVIF).** prudynt implements `Require: www.onvif.org/ver20/backchannel`. The
  sendonly `track0` appears in the SDP only when the client sends that header, and it activates on
  PLAY. It is enabled when `cfg->audio.output_enabled` is true (`RTSP.cpp`). Offered formats:
  **AAC (PT 97, at the output sample rate), OPUS/48000/2 (PT 102), PCMU/8000, PCMA/8000**. Opus and
  AAC appear only when built in (`IMPBackchannel.hpp` `#if USE_AAC` / `USE_OPUS`). Thingino builds
  both by default.
- **Gateway support** (prudynt `docs/webrtc.md`): go2rtc **yes**. It sends the Require header and
  passes the browser mic's Opus straight through with no transcoding (use
  `media=video+audio+microphone`). MediaMTX **no** "as of v1.19.2": feature request
  https://github.com/bluenviron/mediamtx/issues/941 is open. I confirmed that MediaMTX's RTSP source
  skips `IsBackChannel` medias (`internal/staticsources/rtsp/source.go`). The same doc sketches a
  "paired path" design for MediaMTX that is **not implemented**.
- **Mic (camera → server) codecs:** OPUS, AAC, PCM, G711A, G711U, G726 (`gtxaspec/prudynt-t`
  `prudynt.cfg.example` and `IMPAudio.cpp`). The Thingino fork names the key
  `"audio": {"mic_format": "OPUS"}` (`docs/webrtc.md`). AAC does not cross WebRTC: MediaMTX logs
  "skipping track (MPEG-4 Audio)" and serves video only. MediaMTX WebRTC read supports Opus, G722 and
  G711 (`mediamtx/docs/5-read/03-webrtc.md`). **Note:** your brief says AAC, but
  `server/mediamtx.yml` comments on "Opus audio", and the live player has a sound button. The camera
  is probably on Thingino's Opus default already. Check with the `DESCRIBE` one-liner in prudynt's
  `docs/webrtc.md`.
- **Playing a file on the speaker (no RTSP needed):** with `audio.spk_enabled` true, prudynt exposes
  the FIFO `/run/prudynt/audio_out`. It takes commands such as
  `PLAY url=<path|url> [format=pcm|wav|aac|opus|mp3|flac] [vol=] [gain=] [loop=] [delay=]`, plus
  `STOP`, `MUTE`, `VOLUME` and `GAIN` (`docs/audio.md`). The `play` wrapper script does the same and
  skips the local existence check for `*://` URLs. Whether prudynt actually fetches http URLs on your
  build is **(unverified)**. MP3 is built only when `FLASH_SIZE_MB > 8` (`Config.in`), so on your
  8 MB D1 use **opus or wav**. `speaker on|off` toggles `audio.spk_enabled` through
  `prudyntctl json`.
- **Mic tap:** `audio.tap_enabled` exposes raw mic PCM (s16le mono, 16 kHz by default) at
  `/run/prudynt/audio_in.pcm` on the camera (`docs/audio.md`). It is not needed, because the server
  already decodes the RTSP audio.
- **Caveats seen in the wild:** the Thingino Frigate wiki says that with AAC "activating the microphone
  deactivates the speaker temporarily"
  (https://github-wiki-see.page/m/themactep/thingino-firmware/wiki/Integration:-Frigate). ha-furbo
  disabled talkback because a second connection to the camera knocked out video
  (https://github.com/willbeeching/ha-furbo). go2rtc users report the speaker clipping the first
  second of audio (https://github.com/AlexxIT/go2rtc/issues/2084, unconfirmed fix). Expect streamer
  hiccups like the ones in your README: test on the device.

### 2b. Bark detection: YAMNet and alternatives

- **Frigate uses YAMNet.** Its Dockerfile downloads
  `https://www.kaggle.com/api/v1/models/google/yamnet/tfLite/classification-tflite/1/download` and
  renames it `cpu_audio_model.tflite` (https://github.com/blakeblackshear/frigate/blob/dev/docker/main/Dockerfile).
  `frigate/const.py`: `AUDIO_DURATION = 0.975`, `AUDIO_FORMAT = "s16le"`, `AUDIO_SAMPLE_RATE = 16000`,
  `AUDIO_MIN_CONFIDENCE = 0.5`. `frigate/events/audio.py` reads mono s16le from ffmpeg and runs the
  model only if RMS ≥ `min_volume`. Each label needs score > threshold (default 0.8). It uses
  `tflite_runtime`, falling back to `ai_edge_litert`. Docs: "relatively lightweight", CPU only,
  `bark` is in the default `listen` list (https://docs.frigate.video/configuration/audio_detectors).
- **Model facts.** About 3.7M weights, 69.2M multiplies per 0.96 s frame, 521 AudioSet classes, 16 kHz
  mono (https://github.com/tensorflow/models/tree/master/research/audioset/yamnet). Licence:
  **Apache-2.0** (tensorflow/models `LICENSE`). Kaggle TFLite archive **(measured)**: 3.24 MB download,
  `1.tflite` = **4,126,810 bytes**. Input `waveform_binary` float32 `[15600]` (raw waveform in [-1, 1];
  the mel frontend is inside the model). Output `[1, 521]`. Dog-related class indices from
  `yamnet_class_map.csv`: **69 Dog, 70 Bark, 71 Yip, 72 Howl, 73 Bow-wow, 74 Growling,
  75 Whimper (dog)**. Also 76 Cat, 77 Purr, 78 Meow.
- **CPU cost (measured, with caveat).** `ai-edge-litert` 2.3.0, 1 thread, XNNPACK, on an Apple M1 Pro:
  **0.88 ms per invoke**, peak RSS ≈ **40 MB** including numpy. The runtime package is 39 MB on disk.
  I did not measure a Lightsail vCPU. Assuming it is 3–6× slower per core, expect **≈3–5 ms per
  0.975 s window**, which is ≈0.3–0.5% of one vCPU if run continuously **(estimate)**. Gated to loud
  episodes only, the average is ≪0.1%. Decoding is already paid for: `watch.py` already runs
  `ffmpeg … -allowed_media_types audio -ac 1 -ar 16000 -f s16le`.
- **Runtime on Ubuntu 24.04 (Python 3.12):** `tflite-runtime` 2.14.0 has no cp312 wheel on PyPI. Use
  **`ai-edge-litert`**, which ships `cp312 manylinux_2_27_x86_64` wheels (https://pypi.org/pypi/ai-edge-litert/json).
  Alternatively, convert to ONNX to reuse the onnxruntime you already run for YOLOX. That conversion
  is untested **(unverified)**.
- **Alternatives:** the other bark repos are a hand-trained spectrogram CNN
  (https://github.com/t04glovern/dog-bark-detection, no licence) or darknet
  (lincolnhard/dog-bark-detector, no licence) and offer nothing over YAMNet. PANNs (larger AudioSet
  CNNs) would be more accurate but much heavier. I did not study them **(unverified)**. Two cheap
  upgrades are fine-tuning a tiny head on YAMNet embeddings with your own labelled clips, or keeping
  YAMNet as is.

### 2c. Pet-wellbeing insights that fit the box

- **Zones / bowl visits.** Monitora Pet: polygon zones drawn over video, a dwell time before a visit
  counts, tolerance for brief misses, cooldown between visits, one snapshot per confirmed visit, and a
  review queue. It uses YOLOX-Tiny ONNX, the same family as your YOLOX-nano
  (https://github.com/lucascodebr20/monitora-pet). Frigate's rules: test the **bottom-centre** of the
  box, inertia default 3 frames, optional `loitering_time`, per-zone object filter
  (https://docs.frigate.video/configuration/zones). **Pan/tilt caveat:** zones are in image
  coordinates, so they hold only at a saved view. Store each zone with its view (pan, tilt), and
  evaluate it only when the camera is at that view (or with Follow the pet off).
- **Activity and sleep.** Doggy Cam scores calm vs stressed from motion, and evaluates it against
  hand-labelled reference video (https://github.com/danshagen/Doggy_Cam_Computer_Vision). You already
  have per-minute "moving/resting". The cheap additions are derived metrics: longest uninterrupted rest
  bout, number of settle-downs, restlessness at night (state changes per hour, 22:00–06:00), and
  pacing (box-centre path length per minute while "moving"). All are SQL over your existing per-minute
  table, with zero inference.
- **Identity ("is it the pet").** DogFaceNet needs aligned dog faces and TF 1.12. MegaDescriptor-T is
  28.3M params (Swin-T), CC-BY-NC-4.0. Its timm/torch stack will not fit 512 MB alongside everything
  else, and an ONNX export is untested **(unverified)**. For a single-dog home, the realistic cheap
  version is: (1) COCO class (dog vs cat/person) from YOLOX, which you have; (2) a colour/size
  signature of the dog crop (HSV histogram + box area at a given view) against a gallery of the pet
  crops. That is my suggestion, not taken from a project.

## 3. Gap table (ranked by value for "feel comfortable about the pet at home" ÷ cost)

CPU is expressed as a percentage of one vCPU, averaged; the box's budget is ≈5%. Effort: S ≈ ½ day,
M ≈ 1–2 days, L ≈ a week+.

| # | Gap | Who shows how | CPU / RAM on this box | Effort | Value | Notes |
|---|---|---|---|---|---|---|
| 1 | **ML bark classification** (bark / whine / howl / growl vs door, TV, IR click) | Frigate audio pipeline; Doggy Detector (incident grouping); jatacid (dog classes 70–75) | ≈0.0–0.1% avg (runs only on loud episodes, ~3–5 ms per 1 s window, estimate); +40 MB RSS, +39 MB disk | S | High: fewer false "barking" pushes, and whining/howling (a distress sign) becomes visible | Model file 4.1 MB, Apache-2.0 |
| 2 | **Talk to the pet, push-to-talk clip** (record in browser, play on camera speaker) | prudynt `docs/audio.md` FIFO `PLAY`; Thingino `play`; PetCam's talk-to-pet UX | ~0 continuous; one ffmpeg remux/transcode of a few seconds per press (<1 CPU-second) | S–M | High (comfort for you, and reassurance for her, though some dogs get confused by a voice with no person) | No backchannel and no new daemon. Latency of seconds |
| 3 | **Zones: bowl / water / bed visits** | Monitora Pet (dwell, cooldown, snapshot); Frigate zones (bottom-centre, inertia, loitering) | ~0 (point-in-polygon on boxes you already compute) | S–M | High: "drank water at 14:10", "hasn't eaten since morning" | Bind zones to saved views because of pan/tilt |
| 4 | **Wellbeing metrics from existing data** (rest bouts, night restlessness, pacing, anomaly vs her 14-day baseline in the daily summary) | Doggy Cam (motion → calm/stressed + evaluation loop); bark-detector (hourly/daily vocal summary) | ~0 (nightly SQL) | S | Medium–high: turns logs into "normal day / unusual day" | Needs no new sensing |
| 5 | **Live two-way talk** (full duplex) | go2rtc → prudynt ONVIF backchannel, Opus passthrough | go2rtc ~0% idle, packet forwarding while talking; RAM ~15–30 MB **(unverified)**; a second RTSP session to the camera while talking | M | Medium (nice, but #2 covers most of the value) | MediaMTX can't do it. Run go2rtc only as the talk path, or move live view to it. Watch for streamer freezes, which your watchdog already handles |
| 6 | **Opus mic audio confirmed end to end** | prudynt `docs/webrtc.md` "AAC does not cross WebRTC" | 0 | S | Medium **if** the camera is really on AAC (then live view has no sound) | Check with `DESCRIBE`; the config comment suggests Opus already |
| 7 | **"Is it the pet" (other dog / person)** | YOLOX COCO classes (have); MegaDescriptor / DogFaceNet (too heavy) | Colour signature ≈0; MegaDescriptor on a new-arrival event only: ~0.3–1 s CPU per call and >100 MB RAM **(estimate, unverified)** | M (colour signature) / L (embeddings) | Low–medium for a single-dog flat; person detection matters more (YOLOX already has `person`) | Push "person seen while you're out" is cheap: same detector |
| 8 | **Treat dispenser** | ha-furbo (treat toss via the Furbo bridge); PetCam (ESP32 servo over HTTP) | 0 server | L (hardware) | Medium | Needs an ESP32 + servo/auger feeder on Wi-Fi, reached over WireGuard or LAN. No subscription cost |
| 9 | HomeKit | Scrypted / go2rtc HomeKit **(not studied)** | go2rtc/Scrypted RAM | M | Low (you already have web push) | Skip |

## 4. Recommended next 3

### 1) YAMNet on loud episodes (gap 1)

- **Model:** Kaggle `google/yamnet/tfLite/classification-tflite/1` (Frigate's source URL above).
  Extract `1.tflite` (4,126,810 bytes) to `models/yamnet.tflite` and pin its SHA-256 in `install.sh`.
  Take the class names from `yamnet_class_map.csv` (tensorflow/models) or Frigate's
  `audio-labelmap.txt`.
- **Runtime:** `pip install ai-edge-litert` (cp312 wheel), then
  `Interpreter(model_path=…, num_threads=1)`. Load it once, lazily, in the Audio thread.
- **Input:** `watch.py` already gets s16le 16 kHz mono in 4000-sample blocks. Keep a ring buffer of
  the last ~6 s. When an episode closes (your existing `bursts`/`quiet >= 12` point), take windows of
  **15600 samples, hop 7800** across the episode (start −0.5 s, end +0.5 s). Convert with
  `x.astype(np.float32) / 32768.0`, then `set_tensor` and `invoke`, giving 521 scores per window.
- **Decision:** episode `max` over windows of score[70 Bark], [71 Yip], [73 Bow-wow] → `barking`;
  [72 Howl] → `howling`; [75 Whimper (dog)] → `whining`; [74 Growling] → `growling`. Use Frigate's 0.8
  per-label threshold as a starting point only. Fit the real threshold on your own labelled clips. Your
  30-day event clips are the dataset: label ~100 episodes by ear before choosing it (dev/test split by
  day). Keep the loudness detector as the gate and as a fallback when the model errors.
- **Cost guard:** log `invoke` time per call for the first day. On a 10-window episode, expect tens of
  ms. Never run it on every block.
- **Output:** add `label` and `scores` to the existing `sound` event. Push only `barking ≥ N/5 min`,
  and always push `whining`/`howling` (separation-distress signal).

### 2) Push-to-talk to the speaker (gap 2), no backchannel

- **Camera, once:** set `audio.spk_enabled=true` (`speaker on`, which writes it through
  `prudyntctl json` and restarts audio thread 4), and check `/run/prudynt/audio_out` exists. Test
  first over SSH: `play -v 70 /usr/share/sounds/chime_1.opus` (path from the `play` usage text).
- **Browser:** hold-to-talk button → `getUserMedia` (HTTPS via Caddy is already in place) →
  `MediaRecorder`. iPhone Safari typically yields `audio/mp4` (AAC) **(unverified per iOS version)**.
  POST the blob to `/api/talk`, capped at 15 s and 1 MB.
- **Server:** `ffmpeg -i in -ac 1 -ar 16000 -f wav out.wav`, under `timeout -k 5 20` and `nice` (wav
  avoids depending on which decoders are present). Then deliver it with `scp out.wav cam:/tmp/t.wav &&
  ssh cam 'play -v 80 /tmp/t.wav'` over WireGuard, behind a hard timeout. The `play url=http://…` route
  would avoid scp but is unverified.
- **Guards:** rate-limit per device; refuse in privacy mode; log each use as an event so the timeline
  shows "you spoke". Add a 2 s chime first (`playonspeaker` pattern) so the pet isn't startled. Watch
  the encoder: your README notes prudynt changes can freeze it. Speaker playback via the FIFO did not
  touch the encoder in prudynt's design, but test it.
- **Later:** if you want live talk, add go2rtc with
  `cam_talk: rtsp://…/ch0#media=audio` (or the `media=video+audio+microphone` WebRTC page) for the talk
  session only, and keep MediaMTX for view and recording. Cost: one extra RTSP session while talking.

### 3) Zones bound to saved views (gap 3), plus derived wellbeing metrics (gap 4)

- **Data:** `zones` table with `(id, name, view_id, polygon[normalised 0–1], dwell_s, cooldown_s)`.
  Draw the zones in the existing saved-views UI over a still snapshot.
- **Logic, per detection:** if the camera is at `view_id` (pan/tilt within a few steps) and Follow is
  not moving it, test the **bottom-centre** of the pet's box against the polygon (ray casting). Require
  presence for ≥ `dwell_s` (bowl ≈ 5–10 s; at a 3 s cadence that is 2–4 consecutive hits, the same
  idea as Frigate's `inertia: 3`), then emit `zone_visit(name, start, end)`. Allow one missed detection
  inside a visit (Monitora Pet's tolerance), and apply a cooldown (≈5 min) before counting a new visit.
- **Summary:** "water ×6, food ×2 (last 08:12)". Alert if there is no water visit in N hours while
  she was home. Add nightly metrics from the per-minute table: longest rest bout, settle-downs, night
  restlessness, and a z-score against the trailing 14 days. One line in the daily push, e.g. "calmer
  than usual".

## 5. Not verified

- Lightsail vCPU timing for YAMNet (measured only on an M1 Pro; the Lightsail figure is a scaling
  estimate).
- Whether prudynt's FIFO `PLAY url=http://…` fetches remote URLs on your build.
- Your camera's actual mic codec (AAC per the brief vs Opus per the `mediamtx.yml` comment).
- Whether your D1 build has `USE_OPUS` for the backchannel (Thingino's default says yes).
- go2rtc RAM footprint; MegaDescriptor CPU time; YAMNet → ONNX conversion; iOS MediaRecorder output
  format.
- Scrypted, Viseron and motionEye pet-specific features.
