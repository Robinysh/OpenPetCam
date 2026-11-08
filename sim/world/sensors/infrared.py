"""Qualitative night-sensitive pictures, inferred from monochrome snapshots.

Night mode can include reflected visible and near-infrared light. Grayscale
alone does not verify LED state, spectrum, exposure, or a material property.
"""
import os

from PIL import Image

from ..layers import Band, Layer
from ..pano import is_monochrome, stitch_mean
from .base import Sensor
from .visible import frames_layer, snapshots


class NightInfrared(Sensor):
    name = "infrared"
    band = Band("infrared", 800e-9, 1000e-9,
                "night-sensitive mode inferred from grayscale; LED state and spectrum unverified")

    def layers(self):
        ctx = self.ctx
        snaps = [s for s in snapshots(ctx) if is_monochrome(s[3])]
        if not snaps:
            return []
        im, n = stitch_mean(ctx.camera, [(p, pan, tilt) for _t, pan, tilt, p in snaps])
        Image.fromarray(im).save(os.path.join(ctx.model, "pano_infrared.jpg"), quality=88)
        return [Layer("infrared.night", self.name, self.band, "panorama", "Near-infrared: every night snapshot",
                      f"{len(snaps)} inferred night-mode snapshots averaged at their camera poses. Moving things blur; "
                      "brightness is qualitative, not calibrated infrared reflectivity or temperature.",
                      snaps[0][0], snaps[-1][0], {"image": "pano_infrared.jpg", "seen_share": round(float((n > 0).mean()), 3)}),
                frames_layer(ctx, "infrared.frames", self.name, self.band, "Near-infrared: minute snapshots", snaps)]
