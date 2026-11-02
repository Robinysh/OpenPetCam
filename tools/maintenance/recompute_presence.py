"""Recompute per-minute 'present' for a day from that minute's saved snapshot, using the
detector + verifier (fixes minutes where the old logic counted a toy as the pet)."""
import os, sqlite3, sys, time
sys.path.insert(0, "/opt/petcam")
from PIL import Image
import app  # noqa: E402
import watch  # noqa: E402
from detector import Detector

day = sys.argv[1]
v, det = watch.Verifier(), Detector(watch.MODEL)
con = sqlite3.connect(watch.DB)
_, start, end = app.day_bounds(day)           # local midnight to midnight (DST days are 23 or 25 h)
rows = con.execute("SELECT ts, n, present, moving, snap FROM minute WHERE ts >= ? AND ts < ? ORDER BY ts", (start, end)).fetchall()
changed = was = now_ = 0
for ts, n, present, moving, snap in rows:
    path = f"{watch.SNAPS}/{snap}" if snap else None
    if not path or not os.path.exists(path):
        continue
    img = Image.open(path)
    seen = any(v.prob(img, f["box"]) >= v.threshold for f in det.detect(img, min_score=0.10) if f["label"] != "person")
    new_present = n if seen else 0
    was += present > 0
    now_ += new_present > 0
    if (present > 0) != seen:
        changed += 1
        con.execute("UPDATE minute SET present=?, moving=? WHERE ts=?", (new_present, moving if seen else 0, ts))
con.commit()
print(f"{day}: {len(rows)} minutes; pet-present minutes before {was}, after {now_}; {changed} changed")
