import sys, time
sys.path.insert(0, sys.argv[1] + "/petcam")
from PIL import Image
from detector import Detector
D, I = sys.argv[1], sys.argv[2]
labels = {  # hand labels, written before running any model
    f"{D}/flipped.jpg": 1, f"{I}/5.png": 1, f"{I}/9.png": 1,
    f"{D}/thingino.jpg": 0, f"{D}/fresh.jpg": 0, f"{D}/frames/now.jpg": 0, f"{D}/frames/sweep_1400.jpg": 0,
    f"{D}/frames/sweep_900.jpg": 0, f"{D}/frames/sweep_2500.jpg": 0, f"{D}/cam.jpg": 0,
}
for model in ("yolox_nano", "yolox_tiny"):
    det = Detector(f"{D}/models/{model}.onnx")
    print(f"== {model}")
    for path, y in labels.items():
        img = Image.open(path)
        t = time.time(); found = det.detect(img, min_score=0.05); ms = (time.time() - t) * 1000
        animals = [d for d in found if d["label"] != "person"]
        top = max((d["score"] for d in animals), default=0.0)
        best = max(animals, key=lambda d: d["score"])["label"] if animals else "-"
        people = max((d["score"] for d in found if d["label"] == "person"), default=0.0)
        print(f"  {'DOG ' if y else 'none'}  pet={top:.2f} ({best:<10}) person={people:.2f}  {ms:5.0f}ms  {path.split('/')[-1]}")
