#!/usr/bin/env python3
"""Label crops by hand, fast: a local web page with the full frame, an editable box and one-key labels.

    python3 tools/label_ui.py            then open http://127.0.0.1:8777

Reads eval/bootstrap/ (manifest.jsonl, labels.jsonl, crops/, frames/), the same data tools/bootstrap_pet.py and
tools/finetune_pet.py train on. Every answer is appended to labels.jsonl as source "owner", which outranks
Claude's labels in training. Moving a box re-cuts that crop (the original is kept as <id>.orig.jpg) and drops its
cached features so the next training run recomputes them. Drawing a box on a frame adds a new candidate, for a pet
the detector missed. Frames you mark "no pet" go to frame_labels.jsonl.

Listens on 127.0.0.1 only: these are pictures from inside your home.
"""
import hashlib
import json
import os
import shutil
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
WORK = os.environ.get("LABEL_WORK") or os.path.join(ROOT, "eval", "bootstrap")   # LABEL_WORK: label a copy
PAGE = os.path.join(HERE, "label_ui.html")
RANK = {"owner": 3, "human": 3, "review": 2, "claude": 1}
LABELS = ("pet", "toy", "person", "nothing")
sys.path.insert(0, HERE)
from bootstrap_pet import crop_box  # noqa: E402  same 15% margin the watcher and training use


def jl(name):
    p = os.path.join(WORK, name)
    return [json.loads(line) for line in open(p) if line.strip()] if os.path.exists(p) else []


def append(name, row):
    with open(os.path.join(WORK, name), "a") as f:
        f.write(json.dumps(row) + "\n")


def state():
    manifest = jl("manifest.jsonl")
    edits = {}
    for r in jl("box_edits.jsonl"):
        edits[r["id"]] = r["box"]
    best, owner = {}, {}
    for r in jl("labels.jsonl"):
        if r["id"] not in best or RANK.get(r["source"], 0) >= RANK.get(best[r["id"]]["source"], 0):
            best[r["id"]] = r
        if r["source"] in ("owner", "human"):
            owner[r["id"]] = r["label"]
    cv = {}
    try:
        cv = json.load(open(os.path.join(WORK, "cv_scores.json"))).get("items", {})
    except (OSError, ValueError):
        pass
    items = []
    for m in manifest:
        lab = best.get(m["id"], {})
        score = cv.get(m["id"], [None])[0]
        try:
            frame_path(m["frame"]); has_frame = True
        except ValueError:
            has_frame = False                     # crops from in-app reports keep no full frame
        items.append({"id": m["id"], "frame": m["frame"], "has_frame": has_frame, "day": m.get("day"), "hour": m.get("hour"),
                      "box": edits.get(m["id"], m["box"]), "box_orig": m["box"], "yolo": m.get("yolo_score"),
                      "label": lab.get("label"), "source": lab.get("source"), "owner": owner.get(m["id"]),
                      "score": score})
    framed = {m["frame"] for m in manifest}
    marked = {r["frame"] for r in jl("frame_labels.jsonl")}
    fdir = os.path.join(WORK, "frames")
    frames = sorted({os.path.relpath(os.path.join(dp, f), fdir).removeprefix("cand/")
                     for dp, _, fs in os.walk(fdir) for f in fs if f.endswith(".jpg")})
    empty = [f for f in frames if f not in framed]
    return {"items": items, "empty_frames": empty, "frames_marked": sorted(marked)}


def frame_path(rel):
    """Snapshots live in frames/<day>/, the watcher's candidate frames in frames/cand/<day>/ (the manifest drops cand/)."""
    base = os.path.realpath(os.path.join(WORK, "frames"))
    for p in (os.path.join(base, rel), os.path.join(base, "cand", rel)):
        p = os.path.realpath(p)
        if p.startswith(base + os.sep) and os.path.exists(p):
            return p
    raise ValueError("no such frame")


def drop_features(cid):
    p = os.path.join(WORK, "features.npz")
    if not os.path.exists(p):
        return
    import numpy as np
    d = dict(np.load(p))
    if d.pop(cid, None) is not None:
        np.savez_compressed(p, **d)


def recut(cid, frame, box):
    crops = os.path.join(WORK, "crops")
    cur, orig = os.path.join(crops, f"{cid}.jpg"), os.path.join(crops, f"{cid}.orig.jpg")
    if os.path.exists(cur) and not os.path.exists(orig):
        shutil.copy(cur, orig)
    im = Image.open(frame_path(frame)).convert("RGB")
    crop_box(im, box).save(cur, quality=92)
    drop_features(cid)


def clean_box(b):
    x0, y0, x1, y1 = [min(1.0, max(0.0, float(v))) for v in b]
    x0, x1 = sorted((x0, x1)); y0, y1 = sorted((y0, y1))
    if x1 - x0 < 0.01 or y1 - y0 < 0.01:
        raise ValueError("box too small")
    return [round(v, 4) for v in (x0, y0, x1, y1)]


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, code, body, ctype="application/json"):
        b = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/":
            return self.send(200, open(PAGE, "rb").read(), "text/html; charset=utf-8")
        if u.path == "/api/state":
            return self.send(200, state())
        if u.path in ("/frame", "/crop"):
            q = parse_qs(u.query)
            try:
                if u.path == "/frame":
                    p = frame_path(q.get("f", [""])[0])
                else:
                    base = os.path.realpath(os.path.join(WORK, "crops"))
                    p = os.path.realpath(os.path.join(base, q.get("id", [""])[0] + ".jpg"))
                    if not p.startswith(base + os.sep) or not os.path.exists(p):
                        raise ValueError("not found")
            except ValueError:
                return self.send(404, {"error": "not found"})
            return self.send(200, open(p, "rb").read(), "image/jpeg")
        self.send(404, {"error": "not found"})

    def do_POST(self):
        try:
            data = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            u = urlparse(self.path).path
            now = time.time()
            if u == "/api/label":
                if data["label"] not in LABELS:
                    raise ValueError("unknown label")
                row = {"id": data["id"], "label": data["label"], "source": "owner", "by": "label_ui", "ts": now}
                if data.get("box"):
                    box = clean_box(data["box"])
                    append("box_edits.jsonl", {"id": data["id"], "box": box, "ts": now})
                    recut(data["id"], data["frame"], box)
                    row["box"] = box
                append("labels.jsonl", row)
                return self.send(200, {"ok": True})
            if u == "/api/new":
                if data["label"] not in LABELS:
                    raise ValueError("unknown label")
                box, frame = clean_box(data["box"]), data["frame"]
                cid = "o" + hashlib.sha1(f"{frame}{box}".encode()).hexdigest()[:11]
                im = Image.open(frame_path(frame)).convert("RGB")
                crop_box(im, box).save(os.path.join(WORK, "crops", f"{cid}.jpg"), quality=92)
                day, hhmm = frame.split("/")[0], frame.split("/")[-1]
                append("manifest.jsonl", {"id": cid, "frame": frame, "day": day, "hour": hhmm[:2], "box": box,
                                          "yolo_label": "owner-drawn", "yolo_score": None, "hash": 0})
                append("labels.jsonl", {"id": cid, "label": data["label"], "source": "owner", "by": "label_ui",
                                        "drawn": True, "ts": now})
                return self.send(200, {"ok": True, "id": cid})
            if u == "/api/frame":
                append("frame_labels.jsonl", {"frame": data["frame"], "has_pet": False, "source": "owner", "ts": now})
                return self.send(200, {"ok": True})
            self.send(404, {"error": "not found"})
        except (KeyError, ValueError, OSError) as e:
            self.send(400, {"error": str(e)})


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8777
    if not os.path.exists(os.path.join(WORK, "manifest.jsonl")):
        sys.exit("no eval/bootstrap/manifest.jsonl yet: run tools/bootstrap_pet.py collect first")
    print(f"labelling at http://127.0.0.1:{port}  (Ctrl-C to stop)")
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
