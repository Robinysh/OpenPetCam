"""A second detector pass at the frame's native size runs only on a miss while she was seen recently (or a search is
on), and its boxes still go through the verifier (server/watch.py NATIVE_RECHECK).
  python3 tests/server/native_recheck_test.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "server"))
os.environ.setdefault("PETCAM_SITE_ENV", os.path.join(HERE, "..", "..", "config", "site.env.example"))
import watch  # noqa: E402


class Det:
    def __init__(self):
        self.calls = []

    def detect(self, img, min_score=0.1, shape=None):
        self.calls.append(shape)
        if shape == watch.NATIVE_SHAPE:
            return [{"label": "dog", "score": 0.3, "box": [0.4, 0.4, 0.6, 0.7]}]
        return []


class Ver:
    threshold = 0.8

    def __init__(self, p):
        self.p = p

    def prob(self, img, box):
        return self.p


class Tr:
    searching = False


clock = [10_000.0]
watch.now = lambda: clock[0]
watch.log = lambda *a: None


def run(last_seen, p=0.95, searching=False):
    w = watch.Watch.__new__(watch.Watch)
    w.privacy_on = lambda: False
    w.det, w.det_dynamic, w.verifier = Det(), True, Ver(p)
    w.tracker = Tr()
    w.tracker.searching = searching
    w.last_seen = last_seen
    w.record_search_frame = lambda *a: None
    return w.verified_box(object()), w.det.calls


box, calls = run(clock[0] - 30)
assert box is not None and watch.NATIVE_SHAPE in calls, (box, calls)
box, calls = run(clock[0] - 3600)
assert box is None and calls == [None], calls                  # long gone: no second look
box, calls = run(clock[0] - 3600, searching=True)
assert box is not None, calls                                  # a search looks harder
box, calls = run(clock[0] - 30, p=0.2)
assert box is None and watch.NATIVE_SHAPE in calls, calls       # the verifier still decides
print("native second look: recent miss found, old miss skipped, search looks, verifier decides")
print("native recheck ok")
