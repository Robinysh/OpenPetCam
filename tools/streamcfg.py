"""Read or change the camera's prudynt stream settings. Runs on the server (uses the app's camera session).

  streamcfg.py show
  streamcfg.py set stream0 gop=25 max_gop=25          any prudynt keys, values parsed as JSON when possible
  streamcfg.py save                                    persist on the camera (save_config)

Stream changes can stall the encoder; the watcher restarts it (freeze / stream-stall watchdogs).
Check `journalctl -u mediamtx` afterwards, not only the live view (docs/camera-thingino.md).
Current reference settings (2026-11-02): stream0 25 fps VBR 1600 kbit/s gop=max_gop=25 (keyframe
every 1 s, so live view opens fast); stream1 5 fps VBR 300 kbit/s gop 5 / max_gop 10; motion off.
2026-11-03: audio mic_noise_suppression=0 (software noise suppression was the camera's single biggest cost:
camera 67% -> 45% busy, mic thread 19% -> 0%, measured over 20 s each; the server analyses raw sound anyway).
"""
import json
import sys

sys.path.insert(0, "/opt/petcam")
import app  # noqa: E402

SHOW = {"stream0": dict.fromkeys(("mode", "bitrate", "fps", "gop", "max_gop")),
        "stream1": dict.fromkeys(("enabled", "mode", "bitrate", "fps", "gop", "max_gop")),
        "motion": {"enabled": None}, "audio": {"mic_format": None, "mic_noise_suppression": None, "mic_hq": None}, "privacy": {"enabled": None}}


def value(v):
    try:
        return json.loads(v)
    except ValueError:
        return v


cmd = sys.argv[1] if len(sys.argv) > 1 else "show"
if cmd == "show":
    print(json.dumps(app.prudynt(SHOW), indent=1))
elif cmd == "set":
    section, pairs = sys.argv[2], sys.argv[3:]
    body = {section: {k: value(v) for k, v in (p.split("=", 1) for p in pairs)}}
    print("set ->", json.dumps(app.prudynt(body)))
elif cmd == "save":
    print("save ->", json.dumps(app.prudynt({"action": {"save_config": None}})))
else:
    sys.exit(__doc__)
