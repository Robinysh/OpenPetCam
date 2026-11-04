#!/usr/bin/env python3
"""Evaluate the pet detector against hand labels, without letting scores bias the labels.

  evaluate.py pull   DIR            copy per-minute snapshots from the server into DIR/frames
  evaluate.py sheets DIR [N]        contact sheets of frames, numbered only (no scores), for labelling;
                                    includes every frame + a random sample if there are many
  evaluate.py score  DIR            run the detector on every frame -> DIR/scores.json
  evaluate.py report DIR            join DIR/labels.json with scores: dev/test split by hour,
                                    threshold chosen on dev, reported on test

labels.json is written by hand BEFORE `score` is run: {"<frame name>": 1 or 0, ...}
"""
import glob
import json
import os
import random
import subprocess
import sys

from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.join(HERE, "..", "server")
sys.path.insert(0, HERE)
from siteconf import HOST, KEY, SSH  # noqa: E402


def pull(d):
    os.makedirs(f"{d}/frames", exist_ok=True)
    subprocess.run(SSH + [
                    "sudo tar -C /var/lib/petcam/snaps -czf /tmp/snaps.tgz . && sudo chmod 644 /tmp/snaps.tgz"], check=True)
    subprocess.run(["scp", "-q", "-i", KEY, f"{HOST}:/tmp/snaps.tgz", f"{d}/snaps.tgz"], check=True)
    subprocess.run(["tar", "-xzf", f"{d}/snaps.tgz", "-C", f"{d}/frames"], check=True)
    files = sorted(glob.glob(f"{d}/frames/*/*.jpg"))
    print(f"{len(files)} frames in {d}/frames")


def frames(d):
    return sorted(os.path.relpath(p, f"{d}/frames") for p in glob.glob(f"{d}/frames/*/*.jpg"))


def sheets(d, limit=400):
    names = frames(d)
    random.seed(7)
    pick = names if len(names) <= limit else sorted(random.sample(names, limit))
    with open(f"{d}/sheet_index.json", "w") as f:
        json.dump(pick, f)
    cols, rows, w, h = 6, 5, 320, 180
    for s in range(0, len(pick), cols * rows):
        sheet = Image.new("RGB", (cols * w, rows * (h + 18)), (20, 20, 20))
        dr = ImageDraw.Draw(sheet)
        for i, name in enumerate(pick[s:s + cols * rows]):
            im = Image.open(f"{d}/frames/{name}").convert("RGB").resize((w, h))
            x, y = (i % cols) * w, (i // cols) * (h + 18)
            sheet.paste(im, (x, y + 18))
            dr.text((x + 4, y + 3), f"#{s + i}", fill=(255, 214, 10))
        out = f"{d}/sheet_{s // (cols * rows):02d}.jpg"
        sheet.save(out, quality=85)
        print(out)


def score(d):
    sys.path.insert(0, SERVER)
    from detector import Detector

    det = Detector(os.path.join(HERE, "..", "models", "yolox_nano.onnx"))
    out = {}
    for name in frames(d):
        found = det.detect(Image.open(f"{d}/frames/{name}"), min_score=0.05)
        animals = [x for x in found if x["label"] != "person"]
        out[name] = round(max((x["score"] for x in animals), default=0.0), 3)
    with open(f"{d}/scores.json", "w") as f:
        json.dump(out, f, indent=0)
    print(f"scored {len(out)} frames")


def report(d):
    labels = json.load(open(f"{d}/labels.json"))
    scores = json.load(open(f"{d}/scores.json"))
    items = [(n, labels[n], scores[n]) for n in labels if n in scores]
    hour = lambda n: n.split("/")[0] + n.split("/")[1][:2]           # group = calendar hour
    groups = sorted({hour(n) for n, _, _ in items})
    dev_groups = set(groups[0::2])
    dev = [x for x in items if hour(x[0]) in dev_groups]
    test = [x for x in items if hour(x[0]) not in dev_groups]

    def stats(rows, th):
        tp = sum(1 for _, y, s in rows if y and s >= th)
        fp = sum(1 for _, y, s in rows if not y and s >= th)
        fn = sum(1 for _, y, s in rows if y and s < th)
        tn = sum(1 for _, y, s in rows if not y and s < th)
        return tp, fp, fn, tn

    best = max((round(t * 0.01, 2) for t in range(5, 80)),
               key=lambda th: (lambda tp, fp, fn, tn: (2 * tp) / max(1, 2 * tp + fp + fn) - 0.001 * th)(*stats(dev, th)))
    print(f"labelled frames: {len(items)} ({sum(y for _, y, _ in items)} with the pet), "
          f"{len(groups)} hour-groups, dev {len(dev)} / test {len(test)}")
    for name, rows in (("dev", dev), ("test", test)):
        for th in sorted({best, 0.30}):
            tp, fp, fn, tn = stats(rows, th)
            prec = tp / max(1, tp + fp)
            rec = tp / max(1, tp + fn)
            print(f"  {name:<4} threshold {th:.2f}: TP {tp} FP {fp} FN {fn} TN {tn}  precision {prec:.2f}  recall {rec:.2f}")
    print(f"threshold chosen on dev: {best:.2f} (deployed: 0.30)")
    misses = [(n, s) for n, y, s in items if y and s < 0.30]
    false = [(n, s) for n, y, s in items if not y and s >= 0.30]
    print("missed pet frames:", misses[:12])
    print("false alarms:", false[:12])


if __name__ == "__main__":
    cmd, d = sys.argv[1], sys.argv[2]
    {"pull": lambda: pull(d), "sheets": lambda: sheets(d, int(sys.argv[3]) if len(sys.argv) > 3 else 400),
     "score": lambda: score(d), "report": lambda: report(d)}[cmd]()
