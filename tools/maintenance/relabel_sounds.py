"""Re-classify this morning's loudness-rule sound events with the SoundModel (same rules as live)."""
import json, os, sqlite3, sys
import numpy as np
sys.path.insert(0, "/opt/petcam")
import watch

m = watch.SoundModel()
con = sqlite3.connect(watch.DB)
kept, relabelled, removed = 0, 0, 0
for line in open("/tmp/snd/index.txt"):
    eid, kind, t, detail = line.rstrip("\n").split("|", 3)
    raw = f"/tmp/snd/{eid}_{kind}.raw"
    if not os.path.exists(raw):
        continue
    d = json.loads(detail)
    y = m.classify(np.fromfile(raw, dtype=np.int16))
    d["yamnet"] = y
    d["relabelled"] = "2026-11-02 from loudness rule"
    if y["dog"] >= watch.DOG_SOUND_THRESHOLD:
        con.execute("UPDATE events SET kind=?, detail=? WHERE id=?", (y["dog_kind"], json.dumps(d), int(eid))); kept += 1
        print(f"{t} -> {y['dog_kind']} (dog {y['dog']:.2f})")
    elif d.get("peak_db", -99) >= watch.LOUD_NON_DOG_DB:
        con.execute("UPDATE events SET kind='loud_sound', detail=? WHERE id=?", (json.dumps(d), int(eid))); relabelled += 1
        print(f"{t} -> loud_sound: {y['top']} (dog {y['dog']:.2f}, peak {d['peak_db']} dB)")
    else:
        row = con.execute("SELECT clip, thumb FROM events WHERE id=?", (int(eid),)).fetchone()
        for f in row or ():
            if f and os.path.exists(f"{watch.CLIPS}/{f}"):
                os.remove(f"{watch.CLIPS}/{f}")
        con.execute("DELETE FROM events WHERE id=?", (int(eid),)); removed += 1
con.commit()
print(f"dog sounds kept {kept}, relabelled loud {relabelled}, removed (quiet household sounds) {removed}")
