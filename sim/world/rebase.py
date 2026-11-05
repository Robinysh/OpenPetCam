"""Keep hand-made marks (picks.json: wall junctions, object windows, in azimuth/elevation) valid across recalibration.

Marks are read off a panorama made with one calibration (picks.json "calib_ref"). A new calibration can shift the
whole frame (e.g. the level tilt moved 728 -> 706 steps between 2026-11-04 and 11-05, ~1.8 deg). This renders the
reference panorama, phase-correlates its gradients against the current one, and moves every mark by the measured
azimuth/elevation shift. Writes picks_rebased.json next to picks.json; room.fuse() prefers it when it matches the
current calibration."""
import hashlib
import json
import os
import sys

import numpy as np


def _h(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()[:16]


def shift_between(ref_pano, cur_pano, PW=1024):
    import cv2
    rd = lambda p: cv2.resize(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2GRAY), (PW, PW // 2)).astype(np.float32)
    g = lambda a: cv2.magnitude(cv2.Sobel(a, cv2.CV_32F, 1, 0), cv2.Sobel(a, cv2.CV_32F, 0, 1))
    A, B = g(rd(ref_pano)), g(rd(cur_pano))
    band = slice(int(PW // 2 * 0.25), int(PW // 2 * 0.80))           # the band both see (elevation ~ +45 to -54)
    win = cv2.createHanningWindow((PW, band.stop - band.start), cv2.CV_32F)
    (dx, dy), resp = cv2.phaseCorrelate(A[band] * win, B[band] * win)
    ppd = PW / 360
    return dx / ppd, -dy / ppd, resp                                   # azimuth (deg, right +), elevation (deg, up +)


def rebase(world):
    from .camera import Camera
    from .pano import stitch_best
    from PIL import Image
    picks_p = os.path.join(world, "picks.json")
    picks = json.load(open(picks_p))
    ref = picks.get("calib_ref", "calib_v1.json")
    cur = os.path.join(world, "calib.json")
    ref_twin = os.path.join(os.path.dirname(os.path.abspath(world)), "twin", picks.get("twin_ref", "day"))
    ref_path = os.path.join(os.path.dirname(os.path.abspath(world)), "world", ref) if not os.path.exists(os.path.join(world, ref)) else os.path.join(world, ref)
    out = os.path.join(world, "raw", "pano_ref.jpg")
    if not os.path.exists(out):
        cam = Camera(ref_path)
        P = json.load(open(os.path.join(ref_twin, "poses.json")))["poses"]
        off = dict(zip(cam.c.get("frames", []), cam.c.get("offsets_deg", [])))
        img, _ = stitch_best(cam, [(os.path.join(ref_twin, p["file"]), p["pan"], p["tilt"], off.get(p["file"], (0, 0, 0))) for p in P])
        Image.fromarray(img).save(out, quality=88)
    daz, del_, resp = shift_between(out, os.path.join(world, "model", "pano_visible_sweep.jpg"))
    moved = json.loads(json.dumps(picks))
    for s in moved.get("surfaces", []):
        s["pts"] = [[round(a + daz, 2), round(e + del_, 2)] for a, e in s["pts"]]
    for o in moved.get("objects", []):
        if "az" in o:
            o["az"] = [round(a + daz, 2) for a in o["az"]]
            o["el"] = [round(e + del_, 2) for e in o["el"]]
    for s in moved.get("acoustic_surfaces", []):
        s["az"] = [round(a + daz, 2) for a in s["az"]]
    moved["rebased"] = {"from_calib": ref, "to_calib_sha": _h(cur), "shift_az_deg": round(daz, 2), "shift_el_deg": round(del_, 2),
                        "phase_corr_peak": round(float(resp), 3)}
    json.dump(moved, open(os.path.join(world, "picks_rebased.json"), "w"), indent=1)
    print(f"marks moved by {daz:+.2f} deg azimuth, {del_:+.2f} deg elevation (match strength {resp:.2f})")
    return moved


if __name__ == "__main__":
    rebase(sys.argv[1])
