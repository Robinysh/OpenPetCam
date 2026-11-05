"""Every colour scale the room model paints with, in one place, so a legend always matches its texture.

Each map takes values already scaled to 0..1 and returns RGB in 0..1. `legend()` gives what the app needs to draw a
colour bar: stops sampled from the SAME function, and the labels for its ends (or swatches for a categorical view)."""
import numpy as np


def seq(v):                       # one-hue sequential: dark violet -> warm cream
    v = np.asarray(v, float)
    return np.stack([0.15 + 0.85 * v, 0.1 + 0.6 * v ** 1.5, 0.35 - 0.2 * v], -1)


def div(v):                       # diverging around grey at 0.5: green-teal below, magenta above
    v = np.asarray(v, float)
    return np.stack([np.where(v > .5, 0.43 + 1.14 * (v - .5), 0.43 - 0.6 * (.5 - v)),
                     np.where(v > .5, 0.43 - 0.5 * (v - .5), 0.43 + 0.2 * (.5 - v)),
                     np.where(v > .5, 0.43 + 0.9 * (v - .5), 0.43 + 0.9 * (.5 - v))], -1).clip(0, 1)


def classes(v):                   # floor (0) / wall or other (0.5) / object (1)
    v = np.asarray(v, float)
    return np.stack([0.3 + 0.6 * v, 0.6 - 0.2 * v, 0.9 - 0.6 * v], -1)


def radio(v):                     # weak link: dim violet -> strong link: warm
    v = np.asarray(v, float)
    return np.stack([0.25 + 0.7 * v, 0.15 + 0.55 * v ** 1.3, 0.55 - 0.35 * v], -1)


def hexes(fn, n=6):
    return ["#%02x%02x%02x" % tuple(int(round(255 * c)) for c in np.clip(fn(np.array([k / (n - 1)]))[0], 0, 1)) for k in range(n)]


def gradient(fn, lo, hi, title):
    return {"kind": "gradient", "stops": hexes(fn), "lo": lo, "hi": hi, "title": title}


def swatches(items, title):
    return {"kind": "swatches", "items": [{"color": "#%02x%02x%02x" % tuple(int(round(255 * c)) for c in col), "label": lab} for col, lab in items],
            "title": title}
