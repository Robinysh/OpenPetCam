#!/usr/bin/env python3
"""Clean the OpenPetCam logo and make every icon from it.

    python3 tools/brand/make_logo.py <source.png>

The source has a transparent background with faint stray specks. Keep only solid, large shapes (the ring and the
dog's head with its eyes and nose), then write:
  docs/assets/logo.png         transparent, 1024 px, for the README and the film
  server/icon.png              512 px app icon on a tile (manifest, login page)
  server/apple-touch-icon.png  180 px, opaque (iOS fills transparency with black)
  server/favicon.png           64 px
"""
import sys

import numpy as np
from PIL import Image
from scipy import ndimage

src = Image.open(sys.argv[1]).convert("RGBA")
a = np.array(src).astype(np.float32)
alpha = a[..., 3]
solid = alpha > 140
lab, n = ndimage.label(solid)
sizes = ndimage.sum(solid, lab, range(1, n + 1))
keep = np.isin(lab, 1 + np.flatnonzero(sizes > 0.002 * solid.size))      # the ring and the head; specks are tiny
keep = ndimage.binary_fill_holes(keep)                                    # eyes and nose sit inside the head
keep = ndimage.binary_dilation(keep, iterations=2)                        # keep the soft anti-aliased edge
out = a.copy()
out[..., 3] = np.where(keep, alpha, 0)
# Opaque inside the shapes: the source's interior is partly translucent from compression.
core = ndimage.binary_erosion(keep, iterations=3)
out[..., 3] = np.where(core, 255, out[..., 3])
logo = Image.fromarray(out.astype(np.uint8), "RGBA")
bbox = logo.getbbox()
logo = logo.crop(bbox)
side = max(logo.size)
sq = Image.new("RGBA", (side, side), (0, 0, 0, 0))
sq.paste(logo, ((side - logo.width) // 2, (side - logo.height) // 2))
sq.resize((1024, 1024), Image.LANCZOS).save("docs/assets/logo.png", optimize=True)


def tile(size, bg, scale):
    t = Image.new("RGBA", (size, size), bg)
    inner = int(size * scale)
    mark = sq.resize((inner, inner), Image.LANCZOS)
    t.alpha_composite(mark, ((size - inner) // 2, (size - inner) // 2))
    return t.convert("RGB")


BG = (255, 255, 255, 255)
tile(512, BG, 0.74).save("server/icon.png", optimize=True)
tile(180, BG, 0.74).save("server/apple-touch-icon.png", optimize=True)
tile(64, BG, 0.86).save("server/favicon.png", optimize=True)
print("specks removed:", int(n - (sizes > 0.002 * solid.size).sum()), "kept shapes:", int((sizes > 0.002 * solid.size).sum()))
