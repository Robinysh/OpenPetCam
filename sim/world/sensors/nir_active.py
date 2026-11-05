"""Near-infrared, active vs passive (raw/ir/exp*/ from the IR experiment runner: sim/world/ir_exp.py on the server).

At one pose the camera takes: A visible (IR-cut filter in, LEDs off), B passive (filter out, LEDs off), C active
(filter out, its 940 nm LEDs on), B2 passive again. Two products, both projected into the panorama:
  near-IR excess (B - A, exposure-normalised on neutral surfaces): what reflects near-infrared more than visible light.
    Plants (chlorophyll), many dyed fabrics, fur: bright. Screens, black plastics: dark. A material cue no photo gives.
  LED light (C - B, exposure-normalised on surfaces beyond 3 m, which the LEDs barely reach): the camera's own light
    only, ~ reflectance x cos(incidence) / range^2. With the fused range it gives near-IR reflectance per pixel; its
    fall-off with range is a physical check on the learned depth (the camera is the light source, at the lens).
Silicon at 850-940 nm does not see room-temperature heat: 'passive IR' here is reflected ambient near-IR, not thermal."""
import json
import os

import numpy as np
from PIL import Image

from ..geometry import directions
from ..layers import Band, Layer
from .base import Sensor


class ActivePassiveNIR(Sensor):
    name = "infrared.active"
    band = Band("infrared", 800e-9, 1000e-9, "940 nm LEDs at the lens (active) vs ambient near-IR (passive), IR-cut filter out")
    needs_room = True

    def layers(self):
        ctx = self.ctx
        root = ctx.path("raw", "ir")
        if not os.path.isdir(root):
            return []
        rp = os.path.join(ctx.model, "range_pano.f32")
        PW, PH = 1024, 512
        D = directions(PW, PH)
        nir_acc = np.zeros((PH, PW)); led_acc = np.zeros((PH, PW)); n = np.zeros((PH, PW))
        stats = []
        for exp in sorted(os.listdir(root)):
            lp = os.path.join(root, exp, "log.json")
            if not os.path.exists(lp):
                continue
            L = json.load(open(lp))
            ld = lambda f: np.asarray(Image.open(os.path.join(root, exp, f + ".jpg")).convert("RGB"), np.float32).mean(-1)
            try:
                A, B, C, B2 = ld("A_visible_1"), ld("B_passive_1"), ld("C_active_1"), ld("B2_passive_1")
            except FileNotFoundError:
                continue
            Bm = (B + B2) / 2
            pan, tilt = L["pos"]
            h, w = A.shape
            u, v, ok = ctx.camera.project(pan, tilt, D, w, h)
            ok &= ~((v < 0.06 * h) & (u < 0.36 * w))
            ui, vi = u.astype(int).clip(0, w - 1), v.astype(int).clip(0, h - 1)
            a, b, c = A[vi, ui], Bm[vi, ui], C[vi, ui]
            sat = (a > 245) | (b > 245) | (c > 245)
            good = ok & ~sat
            k = np.median(a[good]) / max(np.median(b[good]), 1)
            nir = b * k - a
            R = np.fromfile(rp, np.float32).reshape(384, 768) if os.path.exists(rp) else None
            r = R[(np.arange(PH)[:, None] * 384 // PH).repeat(PW, 1), (np.arange(PW)[None] * 768 // PW).repeat(PH, 0)] if R is not None else np.full((PH, PW), 0)
            far = good & (r > 3.0)
            g = np.median(c[far]) / max(np.median(b[far]), 1) if far.sum() > 500 else 1.0
            led = c - g * b
            nir_acc[good] += nir[good]; led_acc[good] += led[good]; n[good] += 1
            rows = []
            for lo, hi in ((0.8, 1.5), (1.5, 2.5), (2.5, 4.0), (4.0, 8.0)):
                s = good & (r >= lo) & (r < hi)
                if s.sum() > 200:
                    rows.append({"range_m": [lo, hi], "led_added": round(float(np.median(led[s])), 2),
                                 "led_x_r2": round(float(np.median(led[s] * r[s] ** 2)), 2), "pixels": int(s.sum())})
            stats.append({"experiment": exp, "pan": pan, "tilt": tilt, "exposure_ratio": round(float(g), 3), "falloff": rows})
        if not stats:
            return []
        with np.errstate(invalid="ignore"):
            nm, lm = nir_acc / n, led_acc / n
        seen = n > 0

        def save(arr, name, gain):
            rgb = np.zeros((PH, PW, 3), np.uint8)
            v = np.clip(arr * gain, -1, 1)
            # diverging: magenta = more near-IR than visible, teal = less; grey midpoint (never black, which means unseen)
            pos, neg = np.clip(v, 0, 1), np.clip(-v, 0, 1)
            rgb[..., 0] = (110 + 145 * pos - 80 * neg).clip(1, 255)
            rgb[..., 1] = (110 - 80 * pos + 60 * neg).clip(1, 255)
            rgb[..., 2] = (110 + 120 * pos + 100 * neg).clip(1, 255)
            rgb[~seen] = 0
            Image.fromarray(rgb).save(os.path.join(ctx.model, name), quality=90)
        save(nm, "pano_nir_excess.jpg", 1 / 40)
        save(lm, "pano_led_light.jpg", 1 / 8)
        return [Layer("infrared.nir_excess", self.name, self.band, "panorama", "Near-IR minus visible (materials)",
                      "Frame with the IR-cut filter out minus the visible frame, exposure-normalised: magenta reflects more "
                      "near-infrared than visible (plants, many fabrics, fur), teal less (screens, black plastics).",
                      None, None, {"image": "pano_nir_excess.jpg", "overlay": True, "experiments": stats}),
                Layer("infrared.led", self.name, self.band, "panorama", "The camera's own IR light (active minus passive)",
                      "LEDs on minus LEDs off, normalised on surfaces beyond 3 m. Should fall off as 1/range^2 from the lens: "
                      + "; ".join(f"{r['range_m'][0]}-{r['range_m'][1]} m: +{r['led_added']} (x r^2 {r['led_x_r2']})" for r in stats[0]["falloff"])
                      + ". Taken at dawn, so the signal is a few grey levels; at night it dominates.",
                      None, None, {"image": "pano_led_light.jpg", "overlay": True, "experiments": stats})]
