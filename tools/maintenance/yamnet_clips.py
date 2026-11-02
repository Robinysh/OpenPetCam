"""Run YAMNet over each event clip's audio (16 kHz mono s16le) and report what it hears.

YAMNet's opinion is NOT ground truth: it is a different model than the loudness rule, which is
useful, but labels for threshold-setting must come from a person listening.
"""
import csv, glob, os, sys
import numpy as np
from ai_edge_litert.interpreter import Interpreter

HERE = os.path.dirname(os.path.abspath(__file__))
M = os.path.join(HERE, "..", "models")
names = [r["display_name"] for r in csv.DictReader(open(f"{M}/yamnet_class_map.csv"))]
DOG = {69: "Dog", 70: "Bark", 71: "Yip", 72: "Howl", 73: "Bow-wow", 74: "Growling", 75: "Whimper"}
it = Interpreter(model_path=f"{M}/yamnet.tflite", num_threads=1)
it.allocate_tensors()
inp, out = it.get_input_details()[0]["index"], it.get_output_details()[0]["index"]

def windows(x, win=15600, hop=7800):
    for s in range(0, max(1, len(x) - win + 1), hop):
        yield x[s:s + win]

d = sys.argv[1]
index = {}
for line in open(f"{d}/index.txt"):
    i, kind, t, detail = line.rstrip("\n").split("|", 3)
    index[i] = (kind, t)
rows = []
for p in sorted(glob.glob(f"{d}/*.raw"), key=lambda p: int(os.path.basename(p).split("_")[0])):
    eid = os.path.basename(p).split("_")[0]
    x = np.fromfile(p, dtype=np.int16).astype(np.float32) / 32768.0
    best = np.zeros(521, dtype=np.float32)
    for w in windows(x):
        it.set_tensor(inp, w)
        it.invoke()
        best = np.maximum(best, it.get_tensor(out).reshape(-1))
    dog_i = max(DOG, key=lambda k: best[k])
    top = np.argsort(best)[::-1][:3]
    kind, t = index.get(eid, ("?", "?"))
    rows.append((t, kind, best[dog_i], DOG[dog_i], ", ".join(f"{names[k]} {best[k]:.2f}" for k in top)))
for t, kind, ds, dl, top in rows:
    print(f"{t}  rule={kind:<10} dog-max {ds:.2f} ({dl:<8})  top: {top}")
dogs = [r for r in rows if r[2] >= 0.5]
print(f"\n{len(rows)} clips: YAMNet dog-sound >= 0.5 in {len(dogs)}, < 0.5 in {len(rows) - len(dogs)}")
