"""Scheduled sweeps: alert.py hears about a failed sweep, and about one with no result an hour after its time."""
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "server"))
import scheduled_sweep as SS  # noqa: E402


def test_check():
    root = tempfile.mkdtemp()
    SS.ROOT = root
    t0 = 1_800_000_000
    for tag, files in (("ok", {"done.json": {}}), ("bad", {"failed.json": {"reason": "privacy is on"}}), ("late", {}), ("soon", {})):
        d = os.path.join(root, tag); os.makedirs(d)
        json.dump({"tag": tag, "at": t0 if tag != "soon" else t0 + 3000, "when": "x", "poses": []}, open(os.path.join(d, "plan.json"), "w"))
        for f, v in files.items():
            json.dump(v, open(os.path.join(d, f), "w"))
    probs = dict(SS.check(now=t0 + 4000))
    assert set(probs) == {"bad", "late"}, probs
    assert "privacy" in probs["bad"] and "no result" in probs["late"]
    assert SS.check(now=t0 + 10 * 86400) == [("bad", "failed: privacy is on")]      # old misses stop nagging


if __name__ == "__main__":
    test_check(); print("ok test_check")
