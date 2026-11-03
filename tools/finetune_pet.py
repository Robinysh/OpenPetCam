#!/usr/bin/env python3
"""Fine-tune the "is this really my pet?" verifier on your own camera's footage (owner, 2026-11-03).

The logistic regression on frozen MobileNetV2 features memorised scenes: on 2026-11-02 it gave the pet 0.11 on a
rug it had never seen her on. This unfreezes the last MobileNetV2 blocks and trains them on every labelled crop
(tools/bootstrap_pet.py collect + label: Claude labels, the owner's 👍/👎 and report words override), with
augmentation for this camera's world: infrared grayscale, sun and shade, partial and off-centre crops.

    python3 tools/finetune_pet.py train      leave-one-day-out report + final model -> eval/finetune/
    python3 tools/finetune_pet.py install    copy the candidate into models/ (verifier.json mode "finetuned")

Judging (nothing here deploys):
  - every reported score comes from a model that never saw that day (leave-one-day-out), against the current
    logistic regression on the same folds, both with thresholds chosen without the test day;
  - then tools/replay.py compare (held-out crops + the 118 walk frames where she was out) and a shadow run on live
    frames, which the agent harness runs for any change to how frames become "the pet".

Input is identical to the server's: the detector box widened by 15%, resized to 224x224 bilinear, ImageNet
normalisation. Output is one logit; the server applies the sigmoid (server/watch.py Verifier, mode "finetuned").
"""
import json
import os
import random
import shutil
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
BOOT = os.path.join(ROOT, "eval", "bootstrap")
OUT = os.path.join(ROOT, "eval", "finetune")
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
RANK = {"owner": 2, "human": 2, "review": 1}
UNFREEZE = 5            # last MobileNetV2 feature blocks trained (of 19); the rest stay ImageNet features
EPOCHS, BATCH, SEED = 12, 32, 7


def load_jsonl(p):
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def dataset():
    manifest = {m["id"]: m for m in load_jsonl(os.path.join(BOOT, "manifest.jsonl"))}
    labels = {}
    for r in load_jsonl(os.path.join(BOOT, "labels.jsonl")):
        if r["id"] not in labels or RANK.get(r["source"], 0) >= RANK.get(labels[r["id"]]["source"], 0):
            labels[r["id"]] = r
    items = []
    for i, r in labels.items():
        p = os.path.join(BOOT, "crops", f"{i}.jpg")
        if i in manifest and os.path.exists(p):
            items.append({"id": i, "path": p, "day": manifest[i]["day"], "y": 1.0 if r["label"] == "pet" else 0.0,
                          "source": r["source"]})
    return items


def torch_bits():
    import torch
    import torchvision
    from torchvision import transforms as T
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    aug = T.Compose([
        T.RandomResizedCrop(224, scale=(0.65, 1.0), ratio=(0.75, 1.33), interpolation=T.InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(),
        T.RandomApply([T.ColorJitter(0.35, 0.35, 0.3, 0.04)], p=0.8),
        T.RandomGrayscale(p=0.25),                                   # infrared night frames are grayscale
        T.RandomApply([T.GaussianBlur(5, (0.1, 1.6))], p=0.25),     # motion and focus blur
        T.ToTensor(), T.Normalize(MEAN, STD)])
    plain = T.Compose([T.Resize((224, 224), interpolation=T.InterpolationMode.BILINEAR), T.ToTensor(), T.Normalize(MEAN, STD)])
    return torch, torchvision, dev, aug, plain


def build(torch, torchvision):
    m = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for p in m.features[:-UNFREEZE].parameters():
        p.requires_grad = False
    m.classifier = torch.nn.Sequential(torch.nn.Dropout(0.3), torch.nn.Linear(1280, 1))
    return m


def fit(items, torch, torchvision, dev, aug, plain, epochs=EPOCHS):
    random.seed(SEED); torch.manual_seed(SEED)
    m = build(torch, torchvision).to(dev)
    imgs = {it["id"]: Image.open(it["path"]).convert("RGB") for it in items}
    y = torch.tensor([it["y"] for it in items])
    pos = float(y.sum()); neg = float(len(y) - pos)
    lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(neg / max(pos, 1)).to(dev))
    opt = torch.optim.AdamW([{"params": m.classifier.parameters(), "lr": 1e-3},
                             {"params": [p for p in m.features.parameters() if p.requires_grad], "lr": 2e-4}], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    idx = list(range(len(items)))
    for _ in range(epochs):
        m.train(); random.shuffle(idx)
        for s in range(0, len(idx), BATCH):
            b = idx[s:s + BATCH]
            x = torch.stack([aug(imgs[items[i]["id"]]) for i in b]).to(dev)
            loss = lossf(m(x).squeeze(1), y[b].to(dev))
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    m.eval()
    return m


def predict(m, items, torch, dev, plain):
    out = []
    with torch.no_grad():
        for s in range(0, len(items), 64):
            x = torch.stack([plain(Image.open(it["path"]).convert("RGB")) for it in items[s:s + 64]]).to(dev)
            out += torch.sigmoid(m(x).squeeze(1)).cpu().tolist()
    return np.array(out)


def scores(y, p, t):
    pred = p >= t
    tp, fp = int((pred & (y == 1)).sum()), int((pred & (y == 0)).sum())
    fn, tn = int((~pred & (y == 1)).sum()), int((~pred & (y == 0)).sum())
    return {"recall": round(tp / max(tp + fn, 1), 3), "precision": round(tp / max(tp + fp, 1), 3),
            "false_accepts": fp, "missed": fn, "n_pet": tp + fn, "n_not": fp + tn}


def pick_threshold(y, p):
    """No false accepts if possible (a toy passing as the pet is the costly error), then highest recall."""
    best = None
    for t in np.round(np.arange(0.05, 0.96, 0.01), 2):
        s = scores(y, p, t)
        key = (-s["false_accepts"], s["recall"], -abs(t - 0.5))
        if best is None or key > best[0]:
            best = (key, float(t))
    return best[1]


def logreg_baseline(items):
    """The deployed method on the same folds: frozen features + logistic regression (tools/bootstrap_pet.py)."""
    sys.path.insert(0, HERE)
    import bootstrap_pet as bp
    cache = dict(np.load(os.path.join(BOOT, "features.npz"))) if os.path.exists(os.path.join(BOOT, "features.npz")) else {}
    fx = None
    for it in items:
        if it["id"] not in cache:
            fx = fx or bp.Features(os.path.join(ROOT, "models", "mobilenetv2-12.onnx"))
            cache[it["id"]] = fx(Image.open(it["path"]))
    np.savez_compressed(os.path.join(BOOT, "features.npz"), **cache)
    X = np.stack([cache[it["id"]] for it in items]).astype(np.float64)
    return X, bp.fit_logreg


def train():
    items = dataset()
    days = sorted({it["day"] for it in items})
    y = np.array([it["y"] for it in items])
    print(f"{len(items)} labelled crops ({int(y.sum())} pet, {int(len(y) - y.sum())} not) over {len(days)} days: {days}")
    if len(days) < 2:
        sys.exit("need at least 2 days of labelled crops for a held-out report")
    torch, torchvision, dev, aug, plain = torch_bits()
    X, fit_logreg = logreg_baseline(items)
    oof_ft, oof_lr, fold_th = np.zeros(len(items)), np.zeros(len(items)), {}
    for d in days:                                   # leave one day out: that day is never seen in training
        tr = [i for i, it in enumerate(items) if it["day"] != d]
        te = [i for i, it in enumerate(items) if it["day"] == d]
        if len({items[i]["y"] for i in tr}) < 2:
            continue
        m = fit([items[i] for i in tr], torch, torchvision, dev, aug, plain)
        oof_ft[te] = predict(m, [items[i] for i in te], torch, dev, plain)
        trp = predict(m, [items[i] for i in tr], torch, dev, plain)
        fold_th[d] = pick_threshold(y[tr], trp)
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        w, b = fit_logreg((X[tr] - mu) / sd, y[tr])
        oof_lr[te] = 1 / (1 + np.exp(-(((X[te] - mu) / sd) @ w + b)))
        print(f"  held out {d}: fine-tuned {scores(y[te], oof_ft[te], fold_th[d])}  logreg@0.47 {scores(y[te], oof_lr[te], 0.47)}")
    # one threshold for the report, chosen on out-of-fold scores of the OTHER days is not possible with 2-3 days,
    # so report each held-out day at the threshold picked on its training days, and pooled at their median.
    th = float(np.median(list(fold_th.values())))
    rep = {"days": days, "n": len(items), "threshold": th,
           "finetuned_heldout": scores(y, oof_ft, th), "logreg_heldout_at_0.47": scores(y, oof_lr, 0.47),
           "per_day_threshold": fold_th}
    print(json.dumps(rep, indent=1))
    final = fit(items, torch, torchvision, dev, aug, plain)
    os.makedirs(OUT, exist_ok=True)
    final.cpu().eval()
    torch.onnx.export(final, torch.zeros(1, 3, 224, 224), os.path.join(OUT, "petnet.onnx"), input_names=["input"],
                      output_names=["logit"], opset_version=17, dynamo=False)
    json.dump({"mode": "finetuned", "model": "petnet.onnx", "threshold": th, "report": rep},
              open(os.path.join(OUT, "verifier.json"), "w"), indent=1)
    print("candidate written to", os.path.relpath(OUT, ROOT), "(not installed)")


def install():
    for f in ("petnet.onnx", "verifier.json"):
        shutil.copy(os.path.join(OUT, f), os.path.join(ROOT, "models", f))
    print("installed into models/; judge it with tools/replay.py compare before deploying")


if __name__ == "__main__":
    {"train": train, "install": install}.get((sys.argv[1:] or [""])[0], lambda: sys.exit(__doc__))()
