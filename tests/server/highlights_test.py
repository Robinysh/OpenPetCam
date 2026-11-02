"""Cut two test clips from the rolling recording, stitch them with make_highlights, verify, clean up."""
import os, sys, time, subprocess
sys.path.insert(0, "/opt/petcam")
import watch
day = time.strftime("%Y-%m-%d")
out = f"{watch.HIGHLIGHTS}/{day}.mp4"
con = watch.db()
ids = [watch.add_event(con, "active", {"test": True}, None, time.time() - 150),
       watch.add_event(con, "barking", {"test": True}, None, time.time() - 90)]
try:
    c = watch.Clips()
    c.cut(ids[0], time.time() - 150 - 8)
    c.cut(ids[1], time.time() - 90 - 8)
    if os.path.exists(out):
        os.rename(out, out + ".keep")
    watch.make_highlights(day)
    info = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration,size", "-of", "compact", out], capture_output=True, text=True).stdout.strip()
    print("highlights:", info)
finally:
    for (clip,) in con.execute(f"SELECT clip FROM events WHERE id IN ({ids[0]},{ids[1]})").fetchall():
        if clip and os.path.exists(f"{watch.CLIPS}/{clip}"):
            os.remove(f"{watch.CLIPS}/{clip}")
    con.execute(f"DELETE FROM events WHERE id IN ({ids[0]},{ids[1]})")
    if os.path.exists(out):
        os.remove(out)
    if os.path.exists(out + ".keep"):
        os.rename(out + ".keep", out)
    print("cleaned up")
