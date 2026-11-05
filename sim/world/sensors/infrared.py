"""The camera at night: near-infrared (~850 nm) from its own illuminator, IR-cut filter out, monochrome picture.

Detected per snapshot (a monochrome picture means night mode). Same geometry as visible light; the room reads
differently (fabrics and the dog's coat reflect IR strongly, screens and some plastics go dark)."""
import os

from PIL import Image

from ..layers import Band, Layer
from ..pano import is_monochrome, stitch_mean
from .base import Sensor
from .visible import frames_layer, snapshots


class NightInfrared(Sensor):
    name = "infrared"
    band = Band("infrared", 800e-9, 900e-9, "850 nm LED illuminator, IR-cut filter out (inferred from monochrome frames)")

    def layers(self):
        ctx = self.ctx
        snaps = [s for s in snapshots(ctx) if is_monochrome(s[3])]
        if not snaps:
            return []
        im, n = stitch_mean(ctx.camera, [(p, pan, tilt) for _t, pan, tilt, p in snaps])
        Image.fromarray(im).save(os.path.join(ctx.model, "pano_infrared.jpg"), quality=88)
        return [Layer("infrared.night", self.name, self.band, "panorama", "Near-infrared: every night snapshot",
                      f"{len(snaps)} night-mode snapshots averaged at the poses the camera held. Moving things blur.",
                      snaps[0][0], snaps[-1][0], {"image": "pano_infrared.jpg", "seen_share": round(float((n > 0).mean()), 3)}),
                frames_layer(ctx, "infrared.frames", self.name, self.band, "Near-infrared: minute snapshots", snaps)]
