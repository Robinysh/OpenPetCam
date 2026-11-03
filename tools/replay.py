#!/usr/bin/env python3
"""Offline replay: judge a code change on held-out data before it ships (part of the super-agent's evaluator, which
workers may not edit). Two suites, each run on a source tree (the current code or a candidate worktree):

  replay.py build                    (re)build the held-out set in eval/heldout/ from labelled crops + owner ground truth
  replay.py detect --tree PATH       detector + verifier of that tree on held-out frames and crops -> JSON metrics
  replay.py search --tree PATH       that tree's tracker over a fixed grid of simulated rooms -> JSON metrics
  replay.py compare CUR CAND         both suites on both trees; exit 0 if the candidate is no worse on any metric

Held-out data: labelled crops (eval/bootstrap) grouped by day+hour; a group is held out when its hash says so (about a
third), so no hand-picking. Owner ground truth adds whole frames: the walk on 2 Nov 14:28-15:45 (owner: "we are not at
home") is a set of frames where any "the dog" is a false alarm. Workers never get this folder; the harness also rejects a
worker whose transcript touched it.
"""
import glob
import hashlib
import json
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from siteconf import ROOT  # noqa: E402

BOOT = os.path.join(ROOT, "eval", "bootstrap")
HELD = os.path.join(ROOT, "eval", "heldout")
MODELS = os.path.join(ROOT, "models")
RANK = {"owner": 2, "human": 2, "review": 1}
# frames with owner ground truth "she is not home" (reports 2026-11-02 14:45 and 15:50)
ABSENT_WINDOWS = [("2026-11-02", "142830", "154500")]


def held_out(group):
    return int(hashlib.sha1(group.encode()).hexdigest(), 16) % 3 == 0


def build():
    man = {json.loads(x)["id"]: json.loads(x) for x in open(os.path.join(BOOT, "manifest.jsonl"))}
    lab = {}
    for x in open(os.path.join(BOOT, "labels.jsonl")):
        r = json.loads(x)
        if r["id"] not in lab or RANK.get(r["source"], 0) >= RANK.get(lab[r["id"]]["source"], 0):
            lab[r["id"]] = r
    if os.path.exists(HELD):
        shutil.rmtree(HELD)
    os.makedirs(os.path.join(HELD, "crops"))
    os.makedirs(os.path.join(HELD, "absent"))
    crops, n_in = [], 0
    for cid, r in lab.items():
        m = man.get(cid)
        if not m:
            continue
        if not held_out(m["day"] + m.get("hour", "")):
            n_in += 1
            continue
        src = os.path.join(BOOT, "crops", f"{cid}.jpg")
        if os.path.exists(src):
            shutil.copy(src, os.path.join(HELD, "crops", f"{cid}.jpg"))
            crops.append({"id": cid, "pet": r["label"] in ("pet", "pet"), "label": r["label"], "group": m["day"] + m.get("hour", "")})
    absent = []
    for day, a, b in ABSENT_WINDOWS:
        for f in sorted(glob.glob(os.path.join(BOOT, "frames", "cand", day, "*.jpg"))):
            t = os.path.basename(f)[:6]
            if a <= t <= b:
                dst = f"{day}_{t}.jpg"
                shutil.copy(f, os.path.join(HELD, "absent", dst))
                absent.append(dst)
    json.dump({"crops": crops, "absent": absent, "held_in_crops": n_in}, open(os.path.join(HELD, "index.json"), "w"), indent=1)
    return {"held_out_crops": len(crops), "pets": sum(c["pet"] for c in crops), "held_in_crops": n_in, "absent_frames": len(absent)}


def load_tree(tree):
    """Import that tree's server modules (detector, watch) fresh, with the shared models."""
    import importlib.util
    for name in ("petconf", "app", "watch", "detector", "tracker"):
        sys.modules.pop(name, None)
    sys.path.insert(0, os.path.join(tree, "server"))

    def mod(name):
        spec = importlib.util.spec_from_file_location(name, os.path.join(tree, "server", f"{name}.py"))
        m = importlib.util.module_from_spec(spec)
        sys.modules[name] = m
        spec.loader.exec_module(m)
        return m
    return mod


def detect(tree):
    from PIL import Image
    idx = json.load(open(os.path.join(HELD, "index.json")))
    mod = load_tree(tree)
    det = mod("detector").Detector(os.path.join(MODELS, "yolox_nano.onnx"))
    os.environ["PETCAM_VERIFIER"] = os.path.join(MODELS, "verifier.json")
    w = mod("watch")
    ver = w.Verifier(os.path.join(MODELS, "verifier.json"))
    th = ver.threshold
    tp = fp = fn = tn = 0
    for c in idx["crops"]:                       # verifier on labelled crops: is this the dog?
        im = Image.open(os.path.join(HELD, "crops", f"{c['id']}.jpg")).convert("RGB")
        p = ver.prob(im, [0.0, 0.0, 1.0, 1.0], margin=0.0)
        yes = p >= th
        tp += yes and c["pet"]
        fp += yes and not c["pet"]
        fn += (not yes) and c["pet"]
        tn += (not yes) and not c["pet"]
    false_alarms = 0                             # whole pipeline on frames where she was not home
    for f in idx["absent"]:
        im = Image.open(os.path.join(HELD, "absent", f)).convert("RGB")
        found = [d for d in det.detect(im, min_score=0.10) if d["label"] != "person"]
        if any(ver.prob(im, d["box"]) >= th for d in sorted(found, key=lambda d: -d["score"])[:3]):
            false_alarms += 1
    n = len(idx["crops"])
    return {"crops": n, "recall": round(tp / max(1, tp + fn), 3), "precision": round(tp / max(1, tp + fp), 3),
            "accuracy": round((tp + tn) / max(1, n), 3), "absent_frames": len(idx["absent"]),
            "false_alarm_rate": round(false_alarms / max(1, len(idx["absent"])), 3)}


def search(tree):
    """The tracker of that tree in a fixed grid of rooms: pet at every pan/tilt cell, camera starting far away."""
    mod = load_tree(tree)
    T = mod("tracker")
    FW, FH, XM, YM = 733.0, 424.0, 3700, 1000

    class World:
        SPEED, LAG = 250.0, 0.7

        def __init__(self, X, pet, pet_y, Y=500):
            self.X, self.Y, self.tx, self.pet, self.pet_y, self.t, self.moved = float(X), float(Y), float(X), pet, pet_y, 0.0, -1e9
            self.travel, self.stops, self.hist = 0.0, 0, [(0.0, float(X), float(Y))]

        def advance(self, dt):
            step = self.SPEED * dt
            nx = self.tx if abs(self.tx - self.X) <= step else self.X + step * (1 if self.tx > self.X else -1)
            self.travel += abs(nx - self.X)
            self.X = nx
            self.t += dt
            self.hist.append((self.t, self.X, self.Y))
            self.hist = self.hist[-400:]

        def sleep(self, s):
            for _ in range(max(1, int(round(s / 0.1)))):
                self.advance(0.1)

        def position(self): return round(self.X), round(self.Y)
        def limits(self): return XM, YM
        def frame_steps(self): return FW, FH
        def presets(self): return []
        def home(self): return 1850, 500
        def moved_at(self): return self.moved
        def busy(self): return self.X != self.tx
        def stop(self): self.tx = self.X
        def wait_idle(self):
            while self.busy():
                self.advance(0.1)

        def goto(self, x, y):
            self.stops += 1
            self.tx, self.Y = float(max(0, min(XM, x))), float(max(0, min(YM, y)))
            self.travel += abs(self.Y - y) * 0

        def aim(self, fx, fy):
            self.stops += 1
            self.travel += abs(fx * FW)
            self.X = self.tx = float(max(0, min(XM, self.X + fx * FW)))
            self.Y = float(max(0, min(YM, self.Y + fy * FH)))

        def box_at(self, X, Y):
            fx, fy = (X - self.pet) / FW, (self.pet_y - Y) / FH
            if abs(fx) > 0.45 or abs(fy) > 0.45:
                return None
            return [0.5 + fx - 0.08, 0.5 + fy - 0.08, 0.5 + fx + 0.08, 0.5 + fy + 0.08]

        def box(self): return self.box_at(self.X, self.Y)
        # Realism (2026-11-02 sweeps): judging a frame costs ~0.3 s of detector time, a frame taken mid-pan
        # catches her only about half the time (blur, partly in frame), a settled one about 9 in 10.
        def find_pet(self, after):
            self.sleep(0.4 + 0.3)
            b = self.box()
            return b if b is not None and self.rng.random() < 0.9 else None

        def find_now(self, since):
            self.sleep(0.3)
            past = [h for h in self.hist if h[0] <= self.t - self.LAG]
            _, X, Y = past[-1] if past else self.hist[0]
            b = self.box_at(X, Y)
            return b if b is not None and self.rng.random() < 0.5 else None

    found = times = travel = stops = n = 0
    for pet in range(150, 3700, 450):
        for pet_y in (300, 500, 720):
            for start in (200, 3500):
                if abs(start - pet) < FW:
                    continue
                w = World(start, pet, pet_y)
                import random
                w.rng = random.Random(pet * 7919 + pet_y * 31 + start)
                tr = T.Tracker(w, w.find_pet, clock=lambda w=w: w.t, log=lambda m: None, sleep=w.sleep, find_now=w.find_now)
                tr.last_seen_t = w.t - 100
                res = tr.search("replay")
                n += 1
                if "found her" in res and w.box() is not None:
                    found += 1
                    times += w.t
                travel += w.travel
                stops += w.stops
    return {"rooms": n, "found_rate": round(found / max(1, n), 3), "mean_time_s": round(times / max(1, found), 1),
            "motor_travel_per_search": round(travel / max(1, n)), "moves_per_search": round(stops / max(1, n), 1)}


def pipeline(tree, frames):
    """Shadow runs: that tree's detector + verifier on each frame -> the verified box or None."""
    from PIL import Image
    mod = load_tree(tree)
    det = mod("detector").Detector(os.path.join(MODELS, "yolox_nano.onnx"))
    ver = mod("watch").Verifier(os.path.join(MODELS, "verifier.json"))
    out = {}
    for f in sorted(os.listdir(frames)):
        if not f.endswith(".jpg"):
            continue
        im = Image.open(os.path.join(frames, f)).convert("RGB")
        best = None
        for d in sorted([d for d in det.detect(im, min_score=0.10) if d["label"] != "person"], key=lambda d: -d["score"])[:3]:
            p = ver.prob(im, d["box"])
            if p >= ver.threshold and (best is None or p > best[1]):
                best = ([round(v, 4) for v in d["box"]], p)
        out[f] = best[0] if best else None
    return out


# metric -> (direction, tolerance): the candidate may not be worse than current by more than the tolerance
RULES = {"recall": ("up", 0.0), "precision": ("up", 0.0), "accuracy": ("up", 0.0), "false_alarm_rate": ("down", 0.0),
         "found_rate": ("up", 0.0), "mean_time_s": ("down", 2.0), "moves_per_search": ("down", 1.0),
         "motor_travel_per_search": ("down", 300)}


def compare(cur, cand):
    out = {"current": {}, "candidate": {}, "worse": [], "better": []}
    for suite in (detect, search):
        a, b = suite(cur), suite(cand)
        out["current"].update(a)
        out["candidate"].update(b)
        for k, (d, tol) in RULES.items():
            if k in a and k in b:
                delta = b[k] - a[k]
                if (d == "up" and delta < -tol) or (d == "down" and delta > tol):
                    out["worse"].append(f"{k} {a[k]} -> {b[k]}")
                elif (d == "up" and delta > tol) or (d == "down" and delta < -tol):
                    out["better"].append(f"{k} {a[k]} -> {b[k]}")
    out["ok"] = not out["worse"]
    return out


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    tree = sys.argv[sys.argv.index("--tree") + 1] if "--tree" in sys.argv else ROOT
    if cmd == "build":
        print(json.dumps(build()))
    elif cmd == "detect":
        print(json.dumps(detect(tree)))
    elif cmd == "search":
        print(json.dumps(search(tree)))
    elif cmd == "pipeline":
        print(json.dumps(pipeline(tree, sys.argv[sys.argv.index("--frames") + 1])))
    elif cmd == "compare":
        r = compare(sys.argv[2], sys.argv[3])
        print(json.dumps(r, indent=1))
        sys.exit(0 if r["ok"] else 1)
    else:
        print(__doc__)
        sys.exit(2)
