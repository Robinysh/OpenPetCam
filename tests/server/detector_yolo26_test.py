"""The YOLO26 detector (DETECTOR=yolo26): its output decode, letterboxing and class filter, with a stub model so no
weights are needed. Run: PETCAM_SITE_ENV=config/site.env python3 tests/server/detector_yolo26_test.py"""
import os
import sys
import unittest

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "server"))
import detector  # noqa: E402


class Stub:
    """An onnxruntime session that returns one prepared head: (1, 84, N) = cx, cy, w, h in input pixels, then 80
    class probabilities, as the Ultralytics export produces."""

    def __init__(self, rows):
        self.rows, self.seen = rows, None

    def get_inputs(self):
        return [type("I", (), {"name": "images"})()]

    def run(self, _outs, feed):
        self.seen = feed["images"]
        n = len(self.rows)
        out = np.zeros((1, 84, n), dtype=np.float32)
        for i, (cx, cy, w, h, cls, p) in enumerate(self.rows):
            out[0, 0:4, i] = (cx, cy, w, h)
            out[0, 4 + cls, i] = p
        return [out]


def make(rows, shape):
    return detector.Yolo26Detector("unused.onnx", shape=shape, session=Stub(rows))


class Yolo26Test(unittest.TestCase):
    def test_box_is_scaled_back_to_the_frame(self):
        # A 1280x720 frame into a 576x1024 input (scale 0.8, no padding): a cat centred at (512, 288), 160x80 px.
        d = make([(512, 288, 160, 80, 15, 0.9)], (576, 1024))
        found = d.detect(Image.new("RGB", (1280, 720)), min_score=0.1)
        self.assertEqual([f["label"] for f in found], ["cat"])
        x0, y0, x1, y1 = found[0]["box"]
        self.assertAlmostEqual(x0, (512 - 80) / 1024, places=3)
        self.assertAlmostEqual(x1, (512 + 80) / 1024, places=3)
        self.assertAlmostEqual(y0, (288 - 40) / 576, places=3)
        self.assertAlmostEqual(y1, (288 + 40) / 576, places=3)
        self.assertAlmostEqual(found[0]["score"], 0.9, places=3)

    def test_letterbox_padding_is_removed(self):
        # A square 400x400 frame into 576x1024: scale 1.44, centred, so 224 px of padding on the left and right.
        d = make([(512, 288, 144, 144, 16, 0.8)], (576, 1024))
        f = d.detect(Image.new("RGB", (400, 400)), min_score=0.1)[0]
        self.assertEqual(f["label"], "dog")
        self.assertAlmostEqual(f["box"][0], (512 - 72 - 224) / 576, places=3)
        self.assertAlmostEqual(f["box"][2], (512 + 72 - 224) / 576, places=3)

    def test_input_is_rgb_zero_to_one(self):
        d = make([], (64, 64))
        img = Image.new("RGB", (64, 64), (255, 0, 0))
        d.detect(img)
        blob = d.sess.seen
        self.assertEqual(blob.shape, (1, 3, 64, 64))
        self.assertAlmostEqual(float(blob[0, 0].max()), 1.0, places=5)     # red channel first (RGB, not BGR)
        self.assertAlmostEqual(float(blob[0, 2].max()), 0.0, places=5)

    def test_only_pets_and_people_and_threshold(self):
        d = make([(100, 100, 50, 50, 56, 0.95),      # chair: dropped
                  (300, 300, 50, 50, 21, 0.6),       # bear: an animal, kept
                  (500, 200, 50, 50, 0, 0.7),        # person
                  (500, 400, 50, 50, 15, 0.05)],     # below min_score
                 (576, 1024))
        got = {f["label"] for f in d.detect(Image.new("RGB", (1280, 720)), min_score=0.1)}
        self.assertEqual(got, {"bear", "person"})

    def test_overlapping_boxes_collapse_to_the_best(self):
        d = make([(512, 288, 160, 80, 15, 0.9), (514, 290, 160, 80, 16, 0.5)], (576, 1024))
        found = d.detect(Image.new("RGB", (1280, 720)), min_score=0.1)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]["label"], "cat")

    def test_free_input_shape_per_call(self):
        d = make([], (576, 1024))
        d.detect(Image.new("RGB", (1280, 720)), shape=(384, 640))
        self.assertEqual(d.sess.seen.shape, (1, 3, 384, 640))


if __name__ == "__main__":
    unittest.main()
