#!/usr/bin/env bash
# Download the open models into models/, each pinned by SHA-256. Safe to re-run.
#   YOLOX-nano (Apache-2.0, Megvii)      pet/person detector
#   MobileNetV2 (Apache-2.0, ONNX zoo)   "is this really the pet?" verifier features
#   YAMNet (Apache-2.0, Google)          dog-sound classifier, + its class map
set -euo pipefail
cd "$(dirname "$0")/../models" 2>/dev/null || { mkdir -p "$(dirname "$0")/../models"; cd "$(dirname "$0")/../models"; }
get() {  # name url sha256
  if [ -f "$1" ] && echo "$3  $1" | shasum -a 256 -c --quiet - 2>/dev/null; then echo "ok    $1"; return; fi
  curl -fsSL -m 300 -o "$1.part" "$2"
  echo "$3  $1.part" | shasum -a 256 -c --quiet - && mv "$1.part" "$1" && echo "got   $1"
}
get yolox_nano.onnx https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx \
  c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d
get mobilenetv2-12.onnx https://github.com/onnx/models/raw/main/validated/vision/classification/mobilenet/model/mobilenetv2-12.onnx \
  c0c3f76d93fa3fd6580652a45618618a220fced18babf65774ed169de0432ad5
get yamnet_class_map.csv https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv \
  cdf24d193e196d9e95912a2667051ae203e92a2ba09449218ccb40ef787c6df2
if ! { [ -f yamnet.tflite ] && echo "10c95ea3eb9a7bb4cb8bddf6feb023250381008177ac162ce169694d05c317de  yamnet.tflite" | shasum -a 256 -c --quiet - 2>/dev/null; }; then
  get yamnet.tgz https://www.kaggle.com/api/v1/models/google/yamnet/tfLite/classification-tflite/1/download \
    1b971b2132760273a5fb8f5dc504e86783b27853bb3135d555ae06f1e1e365f3
  tar -xzf yamnet.tgz 1.tflite && mv 1.tflite yamnet.tflite && rm -f yamnet.tgz
  echo "10c95ea3eb9a7bb4cb8bddf6feb023250381008177ac162ce169694d05c317de  yamnet.tflite" | shasum -a 256 -c --quiet -
fi
echo "ok    yamnet.tflite"
# The same detector with a free input size, so the watcher runs it on the 16:9 frame without square padding
# (37% less detector time). Needs the onnx package here; without it the watcher uses the square model as before.
if [ ! -f yolox_nano_dyn.onnx ]; then
  if python3 -c "import onnx" 2>/dev/null; then
    python3 ../tools/make_dyn_detector.py yolox_nano.onnx yolox_nano_dyn.onnx
  else
    # No onnx package here: build the file in a throwaway virtualenv (it is derived from the pinned weights above).
    tmpenv=$(mktemp -d)
    if python3 -m venv "$tmpenv" >/dev/null 2>&1 && "$tmpenv/bin/pip" install -q onnx >/dev/null 2>&1; then
      "$tmpenv/bin/python" ../tools/make_dyn_detector.py yolox_nano.onnx yolox_nano_dyn.onnx
    fi
    rm -rf "$tmpenv"
  fi
  [ -f yolox_nano_dyn.onnx ] || echo "WARNING: no yolox_nano_dyn.onnx (needs the onnx package): the watcher uses the plain square detector and skips its native-size second look, which is less sensitive to small pets."
fi
# The verifier config: zero-shot "any dog breed" (PET_KIND=cat: "any domestic cat") until tools/bootstrap_pet.py trains
# one for your pet. An untrained verifier follows PET_KIND (so changing it takes effect); a trained one is never replaced.
DEFAULT=../server/verifier.default.json
[ "${PET_KIND:-dog}" = cat ] && DEFAULT=../server/verifier.cat.json
if [ ! -f verifier.json ] || python3 - "$DEFAULT" <<'PY'
import json, sys
cur, want = json.load(open("verifier.json")), json.load(open(sys.argv[1]))
sys.exit(0 if cur.get("mode") == "zeroshot" and cur.get("classes") != want["classes"] else 1)
PY
then cp "$DEFAULT" verifier.json; fi
