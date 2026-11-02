#!/usr/bin/env python3
"""Repeatable bootstrap of the "is this really my pet?" verifier.

YOLOX (the first stage) finds dog-like things, including plush toys. This trains a
second stage that knows what your pet looks like, from crops labelled by Claude
(described by PET_DESCRIPTION / PET_LOOKALIKES in config/site.env):

  bootstrap_pet.py collect   pull snapshots (+ crops the watcher saved), run YOLOX loosely,
                               cut candidate crops, drop near-duplicates
  bootstrap_pet.py label     label NEW crops with `claude -p` via numbered contact sheets
                               (cached in labels.jsonl; re-runs only label what is new)
  bootstrap_pet.py train     MobileNetV2 features -> logistic regression; threshold chosen on the
                               dev days, reported on held-out test days, against the YOLOX-only baseline
  bootstrap_pet.py deploy    copy verifier.json (+ the feature model) to the server, restart watcher
  bootstrap_pet.py all       collect, label, train (deploy stays a separate, deliberate step)

Labels come from a different model family than the detector (Claude vs YOLOX/MobileNet), as the
evaluation rules require. Human answers from the app (events.label) override Claude's when present.
"""
import glob
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import time

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
WORK = os.path.join(ROOT, "eval", "bootstrap")
MODELS = os.path.join(ROOT, "models")
from siteconf import HOST, KEY, PET, SITE  # noqa: E402

LABELS = ("pet", "toy", "other_animal", "person", "nothing")
SHEET_N = 40
PROMPT_VERSION = "v2"   # v1 used the label "the pet"; read as "pet"

sys.path.insert(0, os.path.join(ROOT, "server"))
sys.path.insert(0, HERE)


def ssh(cmd):
    return subprocess.run(["ssh", "-i", KEY, "-o", "BatchMode=yes", HOST, cmd], check=True, capture_output=True, text=True).stdout


def load_jsonl(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def append_jsonl(path, rows):
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


# ---------------------------------------------------------------- collect

def dhash(img, size=8):
    g = np.asarray(img.convert("L").resize((size + 1, size)), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def crop_box(im, box, margin=0.15):
    w, h = im.size
    x0, y0, x1, y1 = box
    mx, my = (x1 - x0) * margin, (y1 - y0) * margin
    return im.crop((int(max(0, (x0 - mx) * w)), int(max(0, (y0 - my) * h)),
                    int(min(w, (x1 + mx) * w)), int(min(h, (y1 + my) * h))))


def collect():
    from detector import Detector

    frames_dir = os.path.join(WORK, "frames")
    crops_dir = os.path.join(WORK, "crops")
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(crops_dir, exist_ok=True)
    # 1) per-minute snapshots from the server (14 days kept there)
    ssh("sudo tar -C /var/lib/petcam/snaps -czf /tmp/bs_snaps.tgz . && sudo chmod 644 /tmp/bs_snaps.tgz")
    subprocess.run(["scp", "-q", "-i", KEY, f"{HOST}:/tmp/bs_snaps.tgz", f"{WORK}/snaps.tgz"], check=True)
    subprocess.run(["tar", "-xzf", f"{WORK}/snaps.tgz", "-C", frames_dir], check=True)
    # Candidate frames the watcher saved when it saw something dog-like (more than one a minute).
    ssh("sudo mkdir -p /var/lib/petcam/candidates && sudo tar -C /var/lib/petcam/candidates -czf /tmp/bs_cands.tgz . "
        "&& sudo chmod 644 /tmp/bs_cands.tgz")
    subprocess.run(["scp", "-q", "-i", KEY, f"{HOST}:/tmp/bs_cands.tgz", f"{WORK}/cands.tgz"], check=True)
    cand_dir = os.path.join(frames_dir, "cand")
    os.makedirs(cand_dir, exist_ok=True)
    subprocess.run(["tar", "-xzf", f"{WORK}/cands.tgz", "-C", cand_dir], check=True)
    # 2) extra positive sources dropped by hand into eval/bootstrap/extra/<label>/*.jpg (optional)
    manifest_path = os.path.join(WORK, "manifest.jsonl")
    manifest = load_jsonl(manifest_path)
    seen_frames = {m["frame"] for m in manifest}
    hashes = [m["hash"] for m in manifest]
    det = Detector(os.path.join(MODELS, "yolox_nano.onnx"))
    new = []
    frames = sorted(glob.glob(f"{frames_dir}/*/*.jpg") + glob.glob(f"{frames_dir}/cand/*/*.jpg"))
    for p in frames:
        rel = os.path.relpath(p, frames_dir)
        if rel.startswith("cand/"):
            rel = rel[5:]                       # cand/<day>/<HHMMSS>.jpg -> <day>/<HHMMSS>.jpg
        if rel in seen_frames:
            continue
        im = Image.open(p).convert("RGB")
        for f in det.detect(im, min_score=0.10):
            if f["label"] == "person":
                continue
            c = crop_box(im, f["box"])
            if c.width < 12 or c.height < 12:
                continue
            h = dhash(c)
            # near-duplicate: a still toy produces the same crop every minute for hours
            if any(bin(h ^ o).count("1") <= 5 for o in hashes[-400:]):
                continue
            hashes.append(h)
            cid = hashlib.sha1(f"{rel}{f['box']}".encode()).hexdigest()[:12]
            c.save(os.path.join(crops_dir, f"{cid}.jpg"), quality=92)
            new.append({"id": cid, "frame": rel, "day": rel[:10], "hour": rel[11:13], "box": f["box"],
                        "yolo_label": f["label"], "yolo_score": round(f["score"], 3), "hash": h})
        seen_frames.add(rel)
    append_jsonl(manifest_path, new)
    print(f"{len(frames)} frames scanned, {len(new)} new unique candidate crops, {len(manifest) + len(new)} total")


# ---------------------------------------------------------------- label (Claude)

PROMPT = """Read the image {sheet}. It is a grid of numbered crops (#{a} to #{b}) cut from a home pet camera
(day colour or night infrared black-and-white). The household's pet is {pet}.
Things in the home that can look like it: {lookalikes}.
For EACH number, choose exactly one label:
- pet: any visible part of {name} (even partial, blurry, lying down, infrared)
- toy: a toy or plush, not a living animal
- other_animal: a real animal that is not {name}
- person: a human
- nothing: furniture, floor, shadow, or nothing identifiable
Reply with ONLY a JSON object mapping each number (as a string) to its label, e.g. {{"{a}": "pet"}}."""


def make_sheet(crop_ids, path):
    cols, W, H = 8, 200, 150
    rows = (len(crop_ids) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * W, rows * (H + 18)), (20, 20, 20))
    dr = ImageDraw.Draw(sheet)
    for i, cid in enumerate(crop_ids):
        c = Image.open(os.path.join(WORK, "crops", f"{cid}.jpg")).convert("RGB")
        c.thumbnail((W - 4, H - 4))
        x, y = (i % cols) * W, (i // cols) * (H + 18)
        sheet.paste(c, (x + (W - c.width) // 2, y + 18))
        dr.text((x + 4, y + 3), f"#{i}", fill=(255, 214, 10))
    sheet.save(path, quality=88)


def claude_label(sheet_path, n):
    prompt = PROMPT.format(sheet=sheet_path, a=0, b=n - 1, name=PET, pet=SITE.get("PET_DESCRIPTION") or PET,
                           lookalikes=SITE.get("PET_LOOKALIKES") or "plush toys, balls, rope toys")
    env = dict(os.environ, CLAUDE_CODE_HOOKS_ENABLED="false")
    r = subprocess.run(["timeout", "-k", "5", "180", "claude", "-p", "--allowedTools", "Read", "--output-format", "json", prompt],
                       capture_output=True, text=True, env=env, cwd="/tmp")
    if r.returncode != 0:
        raise RuntimeError(f"claude -p failed ({r.returncode}): {r.stderr[-300:]}")
    out = json.loads(r.stdout)
    if out.get("is_error"):
        raise RuntimeError(f"claude -p error: {out.get('result', '')[:300]}")
    text = out["result"].strip()
    text = text[text.index("{"): text.rindex("}") + 1]
    labels = json.loads(text)
    return {int(k): ("pet" if v == "the pet" else v) for k, v in labels.items() if v in LABELS + ("the pet",)}, out.get("total_cost_usd")


def label():
    manifest = load_jsonl(os.path.join(WORK, "manifest.jsonl"))
    labels_path = os.path.join(WORK, "labels.jsonl")
    done = {r["id"] for r in load_jsonl(labels_path) if r["source"] == "claude"}
    todo = [m["id"] for m in manifest if m["id"] not in done]
    print(f"{len(todo)} crops to label")
    os.makedirs(os.path.join(WORK, "sheets"), exist_ok=True)
    for s in range(0, len(todo), SHEET_N):
        batch = todo[s:s + SHEET_N]
        path = os.path.join(WORK, "sheets", f"sheet_{int(time.time())}_{s // SHEET_N:03d}.jpg")
        make_sheet(batch, path)
        labels, cost = claude_label(path, len(batch))
        rows = [{"id": cid, "label": labels[i], "source": "claude", "prompt": PROMPT_VERSION, "sheet": os.path.basename(path),
                 "ts": time.time()} for i, cid in enumerate(batch) if i in labels]
        append_jsonl(labels_path, rows)
        missing = len(batch) - len(rows)
        print(f"  sheet {s // SHEET_N}: {len(rows)} labelled{f', {missing} missing' if missing else ''} (notional ${cost})")


# ---------------------------------------------------------------- features + train

class Features:
    """MobileNetV2 (ONNX model zoo, Apache-2.0) ImageNet logits of a crop: a 1000-d description
    that already separates 'golden retriever' from 'teddy bear' / 'ball'. One stage, CPU, ~20 ms."""

    MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    def __init__(self, path):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(path, opts, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name

    def __call__(self, img):
        x = np.asarray(img.convert("RGB").resize((224, 224), Image.BILINEAR), dtype=np.float32) / 255.0
        x = ((x - self.MEAN) / self.STD).transpose(2, 0, 1)[None]
        return self.sess.run(None, {self.inp: x})[0].reshape(-1)


def fit_logreg(X, y, l2=1.0, iters=600, lr=0.1):
    w = np.zeros(X.shape[1], dtype=np.float64)
    b = 0.0
    pos = max(1, y.sum())
    neg = max(1, len(y) - y.sum())
    sw = np.where(y == 1, len(y) / (2 * pos), len(y) / (2 * neg))   # balance classes
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b)))
        g = (p - y) * sw
        w -= lr * (X.T @ g / len(y) + l2 * w / len(y))
        b -= lr * g.mean()
    return w, b


def prf(y, pred):
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": round(p, 3), "recall": round(r, 3),
            "f1": round(2 * p * r / max(1e-9, p + r), 3)}


def train():
    manifest = {m["id"]: m for m in load_jsonl(os.path.join(WORK, "manifest.jsonl"))}
    labels = {}
    for r in load_jsonl(os.path.join(WORK, "labels.jsonl")):
        if r["source"] == "human" or r["id"] not in labels or labels[r["id"]]["source"] != "human":
            labels[r["id"]] = r
    items = [(manifest[i], 1.0 if r["label"] in ("pet", "the pet") else 0.0) for i, r in labels.items() if i in manifest]
    feats_path = os.path.join(WORK, "features.npz")
    cache = dict(np.load(feats_path)) if os.path.exists(feats_path) else {}
    fx = None
    for m, _ in items:
        if m["id"] not in cache:
            fx = fx or Features(os.path.join(MODELS, "mobilenetv2-12.onnx"))
            cache[m["id"]] = fx(Image.open(os.path.join(WORK, "crops", f"{m['id']}.jpg")))
    np.savez_compressed(feats_path, **cache)
    X = np.stack([cache[m["id"]] for m, _ in items]).astype(np.float64)
    y = np.array([lab for _, lab in items])
    yolo = np.array([m["yolo_score"] for m, _ in items])
    # split by day when there are >= 2 labelled days with the pet in them, else by hour
    groups = sorted({m["day"] for m, _ in items})
    key = (lambda m: m["day"]) if len(groups) >= 3 else (lambda m: m["day"] + m["hour"])
    gs = sorted({key(m) for m, _ in items})
    random.Random(7).shuffle(gs)
    test_g = set(gs[: max(1, len(gs) // 3)])
    test = np.array([key(m) in test_g for m, _ in items])
    dev = ~test
    mu, sd = X[dev].mean(0), X[dev].std(0) + 1e-6
    Z = (X - mu) / sd
    w, b = fit_logreg(Z[dev], y[dev])
    prob = 1 / (1 + np.exp(-(Z @ w + b)))
    # Free alternative (no training): ImageNet probability of any dog breed (classes 151-268).
    sm = np.exp(X - X.max(1, keepdims=True))
    sm /= sm.sum(1, keepdims=True)
    zs = sm[:, 151:269].sum(1)
    ths = np.round(np.arange(0.05, 0.96, 0.01), 2)
    th = max(ths, key=lambda t: (prf(y[dev], (prob[dev] >= t).astype(int))["f1"], -abs(t - 0.5)))
    base_th = max(ths, key=lambda t: prf(y[dev], (yolo[dev] >= t).astype(int))["f1"])
    zs_th = max(ths, key=lambda t: (prf(y[dev], (zs[dev] >= t).astype(int))["f1"], -abs(t - 0.4)))
    neg_max = float(zs[y == 0].max()) if (y == 0).any() else 0.0
    pos_min = float(zs[y == 1].min()) if (y == 1).any() else 1.0
    enough = y.sum() >= 40 and (len(y) - y.sum()) >= 20 and len(groups) >= 3
    report = {
        "items": len(items), "pet": int(y.sum()), "not_pet": int(len(y) - y.sum()),
        "split": f"by {'day' if len(groups) >= 3 else 'hour'}: dev {int(dev.sum())} / test {int(test.sum())}",
        "verifier_threshold": float(th),
        "verifier_test": prf(y[test], (prob[test] >= th).astype(int)),
        "yolox_only_current_0.30_test": prf(y[test], (yolo[test] >= 0.30).astype(int)),
        "yolox_only_best_threshold": float(base_th),
        "yolox_only_best_test": prf(y[test], (yolo[test] >= base_th).astype(int)),
        "zeroshot_dog_threshold": float(zs_th),
        "zeroshot_test": prf(y[test], (zs[test] >= zs_th).astype(int)),
        "zeroshot_gap_all_labels": {"not_pet_max": round(neg_max, 3), "pet_min": round(pos_min, 3)},
        "mode_chosen": "logreg" if enough else "zeroshot",
        "why": "enough labelled data for a held-out test" if enough else
               "too few labels for a meaningful held-out test (need >=40 pet, >=20 not, >=3 days); using the untrained baseline",
        "trained_at": time.strftime("%Y-%m-%d %H:%M"),
    }
    print(json.dumps(report, indent=1))
    if enough:
        out = {"mode": "logreg", "mean": mu.tolist(), "std": sd.tolist(), "w": w.tolist(), "b": float(b),
               "threshold": float(th), "feature_model": "mobilenetv2-12.onnx", "report": report}
    else:
        # midpoint of the gap between the classes when they separate, else the dev-fitted threshold
        zt = round((neg_max + pos_min) / 2, 2) if pos_min > neg_max else float(zs_th)
        report["zeroshot_threshold_used"] = zt
        out = {"mode": "zeroshot", "classes": [151, 268], "threshold": zt, "feature_model": "mobilenetv2-12.onnx",
               "report": report}
    with open(os.path.join(WORK, "verifier.json"), "w") as f:
        json.dump(out, f)
    errs = [(m["id"], m["frame"], round(float(p), 2), int(t)) for (m, t), p in zip(items, prob) if (p >= th) != bool(t)]
    print(f"misclassified ({len(errs)}):", errs[:20])


def deploy():
    v = os.path.join(WORK, "verifier.json")
    shutil.copy(v, os.path.join(MODELS, "verifier.json"))   # so later install.sh runs keep it
    subprocess.run(["scp", "-q", "-i", KEY, v, os.path.join(MODELS, "mobilenetv2-12.onnx"), f"{HOST}:/tmp/"], check=True)
    print(ssh("sudo install -m 644 /tmp/verifier.json /tmp/mobilenetv2-12.onnx /opt/petcam/models/ "
              "&& sudo systemctl restart petcam-watch && sleep 3 && systemctl is-active petcam-watch"))


if __name__ == "__main__":
    os.makedirs(WORK, exist_ok=True)
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    steps = {"collect": [collect], "label": [label], "train": [train], "deploy": [deploy],
             "all": [collect, label, train]}[cmd]
    for step in steps:
        step()
