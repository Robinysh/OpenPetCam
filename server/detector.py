"""Tiny CPU object detector for petcam: YOLOX (Apache-2.0, Megvii) via onnxruntime.

Returns animals and people found in a frame. Runs on a single thread so it never starves the
web app or the video relay on a 2-vCPU burstable box.
"""
import numpy as np
from PIL import Image

COCO_ANIMALS = {15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow", 21: "bear"}  # no "teddy bear": her plush toys would count as the pet
PERSON = 0


class Detector:
    def __init__(self, model_path, size=416, shape=None):
        """size: the square input (the exported model's fixed shape). shape=(h, w): any multiple of 32 instead, for a
        model with free input dimensions (tools/make_dyn_detector.py): (256, 416) fits a 16:9 frame without padding."""
        import onnxruntime as ort

        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1
        opts.inter_op_num_threads = 1
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.sess = ort.InferenceSession(model_path, opts, providers=["CPUExecutionProvider"])
        self.input = self.sess.get_inputs()[0].name
        self.size = size
        self.shape = tuple(shape) if shape else (size, size)
        self._grids = self._make_grids(*self.shape)
        self._grid_cache = {self.shape: self._grids}

    @staticmethod
    def _make_grids(h, w=None):
        w = h if w is None else w
        grids, strides = [], []
        for s in (8, 16, 32):
            yv, xv = np.meshgrid(np.arange(h // s), np.arange(w // s), indexing="ij")
            grids.append(np.stack((xv, yv), 2).reshape(-1, 2))
            strides.append(np.full((len(xv.ravel()), 1), s))
        return np.concatenate(grids, 0).astype(np.float32), np.concatenate(strides, 0).astype(np.float32)

    def _prep(self, img, shape=None):
        img = img.convert("RGB")
        sh, sw = shape or self.shape
        w, h = img.size
        r = min(sw / w, sh / h)
        nw, nh = int(round(w * r)), int(round(h * r))
        canvas = np.full((sh, sw, 3), 114, dtype=np.uint8)
        canvas[:nh, :nw] = np.asarray(img.resize((nw, nh), Image.BILINEAR))
        bgr = canvas[:, :, ::-1].astype(np.float32)          # YOLOX was trained on BGR, 0..255
        return np.ascontiguousarray(bgr.transpose(2, 0, 1))[None], r

    def detect(self, img, min_score=0.15, shape=None):
        """Returns [{'label', 'score', 'box': [x0,y0,x1,y1] in 0..1 of the frame}], best first.
        shape: a one-off (h, w) input, multiples of 32 (only with a free-shape model)."""
        shape = tuple(shape) if shape else self.shape
        blob, r = self._prep(img, shape)
        out = self.sess.run(None, {self.input: blob})[0][0]
        if shape not in self._grid_cache:
            self._grid_cache[shape] = self._make_grids(*shape)
        grids, strides = self._grid_cache[shape]
        xy = (out[:, :2] + grids) * strides
        wh = np.exp(out[:, 2:4]) * strides
        scores = out[:, 4:5] * out[:, 5:]
        cls = scores.argmax(1)
        best = scores[np.arange(len(scores)), cls]
        keep = best >= min_score
        w, h = img.size
        found = []
        for (cx, cy), (bw, bh), c, s in zip(xy[keep], wh[keep], cls[keep], best[keep]):
            label = COCO_ANIMALS.get(int(c)) or ("person" if int(c) == PERSON else None)
            if not label:
                continue
            x0, y0 = (cx - bw / 2) / r / w, (cy - bh / 2) / r / h
            x1, y1 = (cx + bw / 2) / r / w, (cy + bh / 2) / r / h
            found.append({"label": label, "score": float(s),
                          "box": [max(0.0, float(x0)), max(0.0, float(y0)), min(1.0, float(x1)), min(1.0, float(y1))]})
        return nms(found)


def detect_crop(det, img, box, side=256, min_score=0.10):
    """Detect inside a square window of the frame: box = (x0, y0, x1, y1) in 0..1, run at `side` pixels; boxes are
    returned in the whole frame's 0..1 coordinates."""
    w, h = img.size
    x0, y0, x1, y1 = (int(round(box[0] * w)), int(round(box[1] * h)), int(round(box[2] * w)), int(round(box[3] * h)))
    crop = img.crop((x0, y0, x1, y1))
    cw, ch = crop.size
    out = []
    for f in det.detect(crop, min_score=min_score, shape=(side, side)):
        b = f["box"]
        out.append({**f, "box": [(x0 + b[0] * cw) / w, (y0 + b[1] * ch) / h, (x0 + b[2] * cw) / w, (y0 + b[3] * ch) / h]})
    return out


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def nms(found, thresh=0.5):
    """Class-agnostic: an IR dog is often scored as dog AND cat/sheep; keep the best box."""
    found = sorted(found, key=lambda d: -d["score"])
    kept = []
    for d in found:
        if all(iou(d["box"], k["box"]) < thresh for k in kept):
            kept.append(d)
    return kept
