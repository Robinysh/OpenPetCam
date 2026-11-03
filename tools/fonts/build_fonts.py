#!/usr/bin/env python3
"""Draw OpenPetCam's two typefaces from one set of letter skeletons.

    python3 tools/fonts/build_fonts.py        writes site/fonts/{snout,snout-mono}.{ttf,woff2} + a specimen PNG

Every letter is a centreline (lines and elliptical arcs) on a 1000-unit grid. The same skeletons are stroked twice:
  Snout       heavy rounded strokes drawn with an oval pen (verticals 1.6x the horizontals), proportional;
              the dots of i and j are paw prints.
  Snout Mono  thin strokes, every glyph the same width; the camera-readout face (labels, timestamps, scores).
Round caps and joins come from buffering the centreline (shapely); round letters get tighter sidebearings.
Needs: fonttools, shapely, brotli, pillow.
"""
import math
import os
import sys

from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from shapely.affinity import scale as sscale
from shapely.affinity import rotate, translate
from shapely.geometry import LineString, Point
from shapely.geometry.polygon import orient
from shapely.ops import unary_union

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
OUT = os.path.join(ROOT, "site", "fonts")
XH, CH, ASC, DSC = 565, 740, 790, -230


def arc(cx, cy, rx, ry, a0, a1):
    n = max(8, int(abs(a1 - a0) / 4))
    return [(cx + rx * math.cos(math.radians(a0 + (a1 - a0) * i / n)),
             cy + ry * math.sin(math.radians(a0 + (a1 - a0) * i / n))) for i in range(n + 1)]


def ell(cx, cy, rx, ry):
    return arc(cx, cy, rx, ry, 0, 360)


class K:
    """Metrics for one stroke weight: centreline positions so the OUTER edge lands on the grid line.
    pen > 1 is an oval pen, wider than tall: verticals come out heavier than horizontals, which keeps joins
    and counters open at heavy weights (what a hand-drawn rounded face does by eye)."""
    def __init__(self, s, pen=1.0):
        self.s, self.pen = s, pen
        h = s / 2
        self.b, self.x, self.c, self.a, self.d = h, XH - h, CH - h, ASC - h, DSC + h
        self.m, self.cm = (self.b + self.x) / 2, (self.b + self.c) / 2


ROUND_L = set("acdeogqCGOQS0368@")
ROUND_R = set("bcdeopDOQS0369@")
R = 210          # lowercase bowl half-width
RC = 250         # capital bowl half-width


def glyphs(k, paw):
    b, x, c, a, d, m, cm, s = k.b, k.x, k.c, k.a, k.d, k.m, k.cm, k.s
    ry = (x - b) / 2
    bowl = ell(R, m, R, ry)
    arch = R * 0.95                                   # n/h/m/u arch height
    tail = R * 0.62
    g = {}

    def dot(px, py, big=False):
        return ("paw", px, py) if (paw and big) else ("dot", px, py)

    idot = x + s * 0.62 + (128 if paw else 115)
    # lowercase
    g["o"] = [bowl]
    g["a"] = [bowl, [(2 * R, x), (2 * R, b)]]
    g["b"] = [bowl, [(0, a), (0, b)]]
    g["d"] = [bowl, [(2 * R, a), (2 * R, b)]]
    g["p"] = [bowl, [(0, x), (0, d)]]
    g["q"] = [bowl, [(2 * R, x), (2 * R, d)]]
    g["c"] = [arc(R, m, R, ry, 42, 318)]
    g["e"] = [[(0, m + s * 0.02), (2 * R, m + s * 0.02)], arc(R, m, R, ry, 4, 300)]
    na = [(0, b)] + arc(R, x - arch, R, arch, 180, 0) + [(2 * R, b)]
    g["n"] = [[(0, x), (0, b)], na]
    g["h"] = [[(0, a), (0, b)], na]
    R2 = 155
    g["m"] = [[(0, x), (0, b)], [(0, b)] + arc(R2, x - arch, R2, arch, 180, 0) + [(2 * R2, b)],
              [(2 * R2, b)] + arc(3 * R2, x - arch, R2, arch, 180, 0) + [(4 * R2, b)]]
    ua = [(0, x)] + arc(R, b + arch, R, arch, 180, 360) + [(2 * R, x)]
    g["u"] = [ua, [(2 * R, x), (2 * R, b)]]
    g["r"] = [[(0, x), (0, b)], [(0, b + 10)] + arc(R * 0.9, x - arch, R * 0.9, arch, 180, 55)]
    g["i"] = [[(0, x), (0, b)], dot(0, idot, True)]
    hook = arc(2 * R - tail, d + tail, tail, tail, 0, -200)
    g["j"] = [[(0, x), (0, d + tail)] + arc(-tail * 0.9, d + tail, tail * 0.9, tail, 0, -165), dot(0, idot, True)]
    g["g"] = [bowl, [(2 * R, x), (2 * R, d + tail)] + arc(R, d + tail, R, tail, 0, -158)]
    g["y"] = [ua, [(2 * R, x), (2 * R, d + tail)] + arc(R, d + tail, R, tail, 0, -158)]
    q = 120
    g["l"] = [[(0, a), (0, b + q)] + arc(q, b + q, q, q, 180, 270) + [(q + 20, b)]]
    g["t"] = [[(0, a - 70), (0, b + q)] + arc(q, b + q, q, q, 180, 270) + [(q + 40, b)], [(-115, x), (2 * R * 0.62, x)]]
    g["f"] = [[(0, b), (0, a - q)] + arc(q, a - q, q, q, 180, 90) + [(q + 70, a)], [(-115, x), (2 * R * 0.6, x)]]
    rs = (x - b) / 4
    g["s"] = [arc(R * 0.85, x - rs, R * 0.85, rs, 15, 270) + arc(R * 0.85, b + rs, R * 0.85, rs, 90, -165)]
    g["v"] = [[(0, x), (R, b), (2 * R, x)]]
    g["w"] = [[(0, x), (150, b), (300, x), (450, b), (600, x)]]
    g["x"] = [[(0, x), (2 * R, b)], [(2 * R, x), (0, b)]]
    g["z"] = [[(0, x), (2 * R, x), (0, b), (2 * R, b)]]
    g["k"] = [[(0, a), (0, b)], [(2 * R * 0.9, x), (0, m * 0.85)], [(R * 0.55, m * 0.95), (2 * R * 0.95, b)]]

    # capitals
    cry = (c - b) / 2
    g["O"] = [ell(RC, cm, RC, cry)]
    g["C"] = [arc(RC, cm, RC, cry, 45, 315)]
    g["G"] = [arc(RC, cm, RC, cry, 42, 360) + [(2 * RC, cm), (RC * 1.15, cm)]]
    g["Q"] = [ell(RC, cm, RC, cry), [(RC * 1.3, b + 120), (2 * RC + 50, b - 50)]]
    p, Rr = 100, 260
    g["D"] = [[(0, b), (0, c), (p, c)] + arc(p, cm, Rr, cry, 90, -90) + [(0, b)]]
    m2 = cm + 25
    t1, t2 = (c - m2) / 2, (m2 - b) / 2
    g["B"] = [[(0, b), (0, c), (p, c)] + arc(p, c - t1, 190, t1, 90, -90) + [(0, m2)],
              [(0, m2), (p + 20, m2)] + arc(p + 20, b + t2, 215, t2, 90, -90) + [(0, b)]]
    m3 = cm - 30
    t3 = (c - m3) / 2
    g["P"] = [[(0, b), (0, c), (p, c)] + arc(p, c - t3, 230, t3, 90, -90) + [(0, m3)]]
    g["R"] = g["P"] + [[(p + 90, m3), (p + 250, b)]]
    W = 370
    g["E"] = [[(W, c), (0, c), (0, b), (W, b)], [(0, cm), (W * 0.82, cm)]]
    g["F"] = [[(W, c), (0, c), (0, b)], [(0, cm), (W * 0.8, cm)]]
    g["H"] = [[(0, c), (0, b)], [(450, c), (450, b)], [(0, cm), (450, cm)]]
    g["I"] = [[(0, c), (0, b)]]
    jr = 170
    g["J"] = [[(340, c), (340, b + jr)] + arc(340 - jr, b + jr, jr, jr, 0, -180)]
    g["K"] = [[(0, c), (0, b)], [(400, c), (0, cm * 0.8)], [(150, cm * 1.03), (420, b)]]
    g["L"] = [[(0, c), (0, b), (360, b)]]
    g["M"] = [[(0, b), (0, c), (290, cm * 0.62), (580, c), (580, b)]]
    g["N"] = [[(0, b), (0, c), (450, b), (450, c)]]
    rS = (c - b) / 4
    g["S"] = [arc(RC * 0.88, c - rS, RC * 0.88, rS, 15, 270) + arc(RC * 0.88, b + rS, RC * 0.92, rS, 90, -165)]
    g["T"] = [[(0, c), (480, c)], [(240, c), (240, b)]]
    ur = 210
    g["U"] = [[(0, c), (0, b + ur)] + arc(ur, b + ur, ur, ur, 180, 360) + [(2 * ur, c)]]
    g["V"] = [[(0, c), (250, b), (500, c)]]
    g["W"] = [[(0, c), (190, b), (390, c * 0.9), (590, b), (780, c)]]
    g["X"] = [[(0, c), (470, b)], [(470, c), (0, b)]]
    g["Y"] = [[(0, c), (240, cm * 0.95), (480, c)], [(240, cm * 0.95), (240, b)]]
    g["Z"] = [[(0, c), (450, c), (0, b), (450, b)]]
    g["A"] = [[(0, b), (260, c), (520, b)], [(110, cm * 0.72), (410, cm * 0.72)]]

    # figures
    g["0"] = [ell(225, cm, 225, cry)]
    g["1"] = [[(0, c - 150), (150, c), (150, b)]]
    r2 = 205
    g["2"] = [arc(r2, c - r2 * 0.95, r2, r2 * 0.95, 155, -25) + [(0, b), (2 * r2 + 10, b)]]
    r3a, r3b = (c - b) * 0.235, (c - b) * 0.265
    g["3"] = [arc(200, c - r3a, 195, r3a, 150, -90) + arc(200, b + r3b, 215, r3b, 90, -150)]
    g["4"] = [[(320, b), (320, c), (0, b + (c - b) * 0.32), (440, b + (c - b) * 0.32)]]
    r5 = (c - b) * 0.31
    g["5"] = [[(400, c), (40, c), (20, b + 2 * r5 + 30)] + arc(190, b + r5, 210, r5, 140, -160)]
    r6 = (c - b) * 0.3
    six = [ell(205, b + r6, 205, r6), arc(390, b + r6, 390, c - b - r6, 90, 180)]
    g["6"] = six
    g["9"] = [("rot", six, 205, cm)]
    g["7"] = [[(0, c), (420, c), (150, b)]]
    r8a, r8b = (c - b) * 0.23, (c - b) * 0.27
    g["8"] = [ell(210, c - r8a, 175, r8a), ell(210, b + r8b, 210, r8b)]

    # punctuation and symbols
    g["."] = [dot(0, b)]
    g[","] = [dot(0, b), [(0, b), (-40, b - 150)]]
    g[":"] = [dot(0, b), dot(0, x - s * 0.1)]
    g[";"] = [dot(0, x - s * 0.1), dot(0, b), [(0, b), (-40, b - 150)]]
    g["!"] = [[(0, c), (0, b + s * 1.7)], dot(0, b)]
    g["?"] = [arc(190, c - 185, 190, 185, 160, -60) + [(170, cm - 40), (170, b + s * 1.7)], dot(170, b)]
    g["'"] = [[(0, c), (0, c - 170)]]
    g['"'] = [[(0, c), (0, c - 170)], [(160, c), (160, c - 170)]]
    g["’"] = [[(0, c), (-30, c - 170)]]
    g["-"] = [[(0, m), (230, m)]]
    g["–"] = [[(0, m), (400, m)]]
    g["—"] = [[(0, m), (720, m)]]
    g["/"] = [[(0, b - 60), (320, c + 40)]]
    g["("] = [arc(260, cm, 260, cry + 110, 115, 245)]
    g[")"] = [arc(-150, cm, 260, cry + 110, 65, -65)]
    g["+"] = [[(0, m), (360, m)], [(180, m - 180), (180, m + 180)]]
    g["="] = [[(0, m + 90), (360, m + 90)], [(0, m - 90), (360, m - 90)]]
    g["×"] = [[(0, m - 160), (320, m + 160)], [(0, m + 160), (320, m - 160)]]
    g["*"] = [[(150, c), (150, c - 300)], [(20, c - 75), (280, c - 225)], [(20, c - 225), (280, c - 75)]]
    g["#"] = [[(120, c), (60, b)], [(340, c), (280, b)], [(0, cm + 110), (420, cm + 110)], [(0, cm - 110), (400, cm - 110)]]
    g["$"] = g["S"] + [[(RC * 0.88, c + 90), (RC * 0.88, b - 90)]]
    g["%"] = [ell(110, c - 120, 110, 120), ell(470, b + 120, 110, 120), [(80, b), (500, c)]]
    g["→"] = [[(0, m), (520, m)], [(340, m + 180), (520, m), (340, m - 180)]]
    g["←"] = [[(520, m), (0, m)], [(180, m + 180), (0, m), (180, m - 180)]]
    g["↓"] = [[(180, c), (180, b)], [(0, b + 180), (180, b), (360, b + 180)]]
    g["✓"] = [[(0, m), (170, b), (480, c)]]
    g["•"] = [dot(0, m)] if not paw else [("dotbig", 0, m)]
    g["@"] = [ell(310, cm - 20, 120, 150), arc(310, cm - 20, 300, 330, -10, 330) , [(430, cm + 130), (430, cm - 120)] + arc(500, cm - 120, 70, 60, 180, 340)]
    g["&"] = [arc(170, c - 140, 140, 140, 330, -40) + [(30, b + 210)] + arc(200, b + 190, 190, 190, 180, 300) + [(470, cm - 20)], [(170, c - 280 + 60), (480, b)]]
    g["_"] = [[(0, d + 40), (500, d + 40)]]
    g["<"] = [[(340, m + 200), (0, m), (340, m - 200)]]
    g[">"] = [[(0, m + 200), (340, m), (0, m - 200)]]
    g["|"] = [[(0, a), (0, d)]]
    g["["] = [[(160, c + 80), (0, c + 80), (0, b - 80), (160, b - 80)]]
    g["]"] = [[(0, c + 80), (160, c + 80), (160, b - 80), (0, b - 80)]]
    g["°"] = [ell(110, c - 110, 95, 95)]
    g["●"] = [("rec", 230, m)]
    return g


def shape(k, item, paw):
    s = k.s
    if isinstance(item, tuple):
        kind = item[0]
        if kind == "rot":
            _, sub, cx, cy = item
            geo = unary_union([shape(k, it, paw) for it in sub])
            return sscale(geo, -1, -1, origin=(cx, cy))
        _, px, py = item
        if kind == "dot":
            return Point(px, py).buffer(s * 0.56, quad_segs=16)
        if kind == "dotbig":
            return Point(px, py).buffer(s * 0.75, quad_segs=16)
        if kind == "rec":
            return Point(px, py).buffer(230, quad_segs=24)
        if kind == "paw":                             # a pad and four toes
            pad = sscale(Point(px, py).buffer(1, quad_segs=16), s * 0.74, s * 0.6)
            toes = [Point(px + s * 0.9 * math.cos(math.radians(t)), py + s * 0.12 + s * 0.8 * math.sin(math.radians(t)))
                    .buffer(s * 0.24, quad_segs=12) for t in (158, 113, 67, 22)]
            return unary_union([pad] + toes)
    kp = k.pen
    line = LineString([(px / kp, py) for px, py in item])
    return sscale(line.buffer(s / 2, cap_style="round", join_style="round", quad_segs=16), kp, 1, origin=(0, 0))



def build(name, s, paw, mono, sp, wobble=0.0, pen=1.0):
    k = K(s, pen)
    defs = glyphs(k, paw)
    order = [".notdef", "space"] + [f"u{ord(ch):04X}" for ch in defs]
    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder(order)
    fb.setupCharacterMap({32: "space", 160: "space", **{ord(ch): f"u{ord(ch):04X}" for ch in defs}})
    glyf, metrics = {}, {}
    empty = TTGlyphPen(None).glyph()
    glyf[".notdef"] = empty; metrics[".notdef"] = (500, 0)
    glyf["space"] = empty; metrics["space"] = (580 if mono else 240, 0)
    ADV = 580
    for ch, items in defs.items():
        geo = unary_union([shape(k, it, paw) for it in items])
        if wobble and ch.isalnum():                 # felt letters pinned by hand: each sits a little askew
            hsh = (ord(ch) * 2654435761) & 0xFFFFFFFF
            ang = ((hsh % 1000) / 1000 - 0.5) * 2 * wobble
            dy = (((hsh >> 10) % 1000) / 1000 - 0.5) * 2 * wobble * 6
            cx0, cy0 = (geo.bounds[0] + geo.bounds[2]) / 2, XH / 2
            geo = translate(rotate(geo, ang, origin=(cx0, cy0)), 0, dy)
        minx, _, maxx, _ = geo.bounds
        if mono:
            avail = ADV - 2 * 55
            ink = maxx - minx
            if ink > avail:                        # squeeze the skeleton, then stroke again at full weight
                f = (avail - s) / max(ink - s, 1)
                items2 = [scale_items(it, f) for it in items]
                geo = unary_union([shape(k, it, paw) for it in items2])
                minx, _, maxx, _ = geo.bounds
            geo = translate(geo, (ADV - (maxx - minx)) / 2 - minx, 0)
            adv = ADV
        else:
            lsb = sp * (0.62 if ch in ROUND_L else 1.0)      # round sides sit closer: optical spacing
            rsb = sp * (0.62 if ch in ROUND_R else 1.0)
            geo = translate(geo, lsb - minx, 0)
            adv = int(round(maxx - minx + lsb + rsb))
        geo = geo.simplify(0.6)
        pen = TTGlyphPen(None)
        polys = getattr(geo, "geoms", [geo])
        for poly in polys:
            poly = orient(poly, sign=-1.0)            # TrueType: outer clockwise, holes counter-clockwise
            for ring in [poly.exterior] + list(poly.interiors):
                pts = [(int(round(px)), int(round(py))) for px, py in list(ring.coords)[:-1]]
                dedup = [pt for i, pt in enumerate(pts) if pt != pts[i - 1]]
                if len(dedup) < 3:
                    continue
                pen.moveTo(dedup[0])
                for pt in dedup[1:]:
                    pen.lineTo(pt)
                pen.closePath()
        gname = f"u{ord(ch):04X}"
        glyf[gname] = pen.glyph()
        glyf[gname].recalcBounds(None)
        metrics[gname] = (adv, int(geo.bounds[0]))
    fb.setupGlyf(glyf)
    fb.setupHorizontalMetrics(metrics)
    fb.setupHorizontalHeader(ascent=930, descent=-280)
    fam = name
    fb.setupNameTable({"familyName": fam, "styleName": "Regular", "uniqueFontIdentifier": f"OpenPetCam:{fam}:0.1",
                       "fullName": fam, "psName": fam.replace(" ", ""), "version": "Version 0.100",
                       "copyright": "OpenPetCam contributors, Apache-2.0",
                       "licenseDescription": "Apache License 2.0", "designer": "OpenPetCam"})
    fb.setupOS2(sTypoAscender=930, sTypoDescender=-280, sTypoLineGap=0, usWinAscent=1000, usWinDescent=320,
                sxHeight=XH, sCapHeight=CH, achVendID="OPCM", fsType=0,
                ulCodePageRange1=1, usWeightClass=800 if not mono else 400,
                xAvgCharWidth=ADV if mono else 520)
    fb.setupPost(isFixedPitch=1 if mono else 0)
    fb.setupMaxp() if hasattr(fb, "setupMaxp") else None
    os.makedirs(OUT, exist_ok=True)
    slug = fam.lower().replace(" ", "-")
    ttf = os.path.join(OUT, f"{slug}.ttf")
    fb.save(ttf)
    from fontTools.ttLib import TTFont
    f = TTFont(ttf)
    f.flavor = "woff2"
    f.save(os.path.join(OUT, f"{slug}.woff2"))
    print(fam, len(defs), "glyphs ->", os.path.relpath(OUT, ROOT))
    return ttf


def scale_items(it, f):
    if isinstance(it, tuple):
        if it[0] == "rot":
            return ("rot", [scale_items(x, f) for x in it[1]], it[2] * f, it[3])
        return (it[0], it[1] * f, it[2])
    return [(px * f, py) for px, py in it]


def specimen(ttf_display, ttf_mono):
    from PIL import Image, ImageDraw, ImageFont
    im = Image.new("RGB", (1800, 1100), "#FCFBF7")
    dr = ImageDraw.Draw(im)
    D = lambda n: ImageFont.truetype(ttf_display, n)
    M = lambda n: ImageFont.truetype(ttf_mono, n)
    dr.text((60, 40), "OpenPetCam", font=D(190), fill="#202332")
    dr.text((60, 270), "Where's the dog? Found her.", font=D(96), fill="#355BEB")
    dk = Image.new("RGB", (1800, 330), "#1B1D2A"); dd = ImageDraw.Draw(dk)
    dd.text((50, 20), "The pet cam that", font=D(130), fill="#F1EBDF"); dd.text((50, 170), "finds your dog.", font=D(130), fill="#F1EBDF")
    dk.save(os.path.join(ROOT, "docs", "brand", "fonts-headline.png"))
    dr.text((60, 410), "ABCDEFGHIJKLMNOPQRSTUVWXYZ", font=D(78), fill="#202332")
    dr.text((60, 520), "abcdefghijklmnopqrstuvwxyz", font=D(78), fill="#202332")
    dr.text((60, 630), "0123456789 $5 0.99 5% → ✓ ! ? & @ #", font=D(78), fill="#202332")
    dr.text((60, 780), "● REC  CAM 01  PAN 112°  TILT 34°  dog 0.99  toy 0.00", font=M(44), fill="#202332")
    dr.text((60, 850), "ABCDEFGHIJKLMNOPQRSTUVWXYZ abcdefghijklmnopqrstuvwxyz", font=M(30), fill="#4A4F66")
    dr.text((60, 900), "0123456789 ./:;!?-+=×*#$%()[]<>@&", font=M(30), fill="#4A4F66")
    path = os.path.join(ROOT, "docs", "brand", "fonts-specimen.png")
    im.save(path)
    print("specimen ->", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    a = build("Snout", 128, paw=True, mono=False, sp=24, pen=1.62)
    build("Snout Text", 118, paw=False, mono=False, sp=30, pen=1.5)   # for sizes under ~48px: plain dots
    b = build("Snout Mono", 64, paw=False, mono=True, sp=0)
    if "--no-specimen" not in sys.argv:
        specimen(a, b)
