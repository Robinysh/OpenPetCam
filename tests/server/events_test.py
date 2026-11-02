"""Event rules found wrong in the 2026-11-02 clip review, replayed with a temporary database.
  python3 tests/server/events_test.py   (locally or on the server)"""
import collections
import json
import os
import queue
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "server"))
sys.path.insert(0, "/opt/petcam")
import watch  # noqa: E402

tmp = tempfile.mkdtemp()
watch.DB, watch.MOVED, watch.TALKING = f"{tmp}/t.db", f"{tmp}/moved_at", f"{tmp}/talking"
watch.init_db()
con = watch.db()


def fresh():
    w = watch.Watch.__new__(watch.Watch)
    w.present, w.present_since, w.rest_since, w.moving_since, w.move_box0 = False, None, None, None, None
    w.last_seen, w.gone_since, w.cam_moves, w.events = 0.0, None, collections.deque(maxlen=20), queue.Queue()
    w.dog_sound_pending = False
    w.save_thumb = lambda *a: None
    w.clips = type("C", (), {"want": lambda *a: None})()
    return w


def kinds():
    return [r[0] for r in con.execute("SELECT kind FROM events ORDER BY ts")]


def clear():
    con.execute("DELETE FROM events")
    for f in (watch.MOVED, watch.TALKING):
        if os.path.exists(f):
            os.remove(f)


def touch_moved(t):
    open(watch.MOVED, "w").close()
    os.utime(watch.MOVED, (t, t))


BOX = [0.3, 0.4, 0.5, 0.6]
# 1. The owner turns the camera onto her (10:19:51 'appeared' false positive): no event. Without a move: event.
clear(); w = fresh(); t = 10_000.0
touch_moved(t - 3)
w.update_pet(con, t, 1.0, False, b"", BOX)
assert kinds() == [], kinds()
clear(); w = fresh()
w.update_pet(con, t, 1.0, False, b"", BOX)
assert kinds() == ["appeared"], kinds()
print("appeared: camera turned onto her -> no event; she walked in -> event")

# 2. The camera is turned away from her: no 'left' 3 minutes later.
clear(); w = fresh(); w.present, w.present_since, w.last_seen = True, t - 600, t
touch_moved(t + 4)
w.update_pet(con, t + 200, 0.0, False, b"")
assert not w.present and kinds() == [], kinds()
print("left: camera turned away -> no event")

# 3. 'active' needs her to move: head turn / exposure jump (10:33:58) no; standing up and walking yes.
clear(); w = fresh(); w.present, w.present_since, w.last_seen, w.rest_since = True, t - 900, t, t - 900
for k in range(8):
    w.update_pet(con, t + k, 1.0, True, b"", [0.30, 0.40, 0.50 + 0.002 * k, 0.60])
assert kinds() == [], kinds()
w = fresh(); w.present, w.present_since, w.last_seen, w.rest_since = True, t - 900, t, t - 900
for k in range(8):
    w.update_pet(con, t + k, 1.0, True, b"", [0.30 + 0.02 * k, 0.30, 0.50 + 0.02 * k, 0.60])
assert kinds() == ["active"], kinds()
print("active: head turn -> no event; gets up and walks -> event")

# 4. Five barks within a minute are one episode (events 80-84), with the count on the event.
clear(); w = fresh()
for k in range(5):
    w.events.put(("sound", {"kind": "dog sound", "peak_db": -10 - k, "yamnet": {"dog": 0.9}}, t + 10 * k))
w.drain_events(con, b"")
rows = con.execute("SELECT kind, detail FROM events").fetchall()
assert len(rows) == 1 and json.loads(rows[0][1])["episode_barks"] == 5, rows
print("barking episode: 5 barks -> 1 event (episode_barks=5)")

# 5. The owner's voice through the speaker (87, 89, 92) is not a 'loud sound'; a dog answering still is.
clear(); w = fresh()
with open(watch.TALKING, "w") as f:
    f.write(f"{t} {t + 8}")
w.events.put(("sound", {"kind": "loud sound", "yamnet": {"top": "Speech", "dog": 0.02, "dog_label": "Growling"}}, t + 2))
w.events.put(("sound", {"kind": "dog sound", "yamnet": {"dog": 0.95}}, t + 5))
w.drain_events(con, b"")
assert kinds() == ["dog_sound"], kinds()
print("owner talking: their voice dropped, the dog barking back kept")

# 6. The loud-sound cap counts the database, so a restart can't reset it.
clear()
for k in range(6):
    con.execute("INSERT INTO events(ts, kind, detail) VALUES(?,?,?)", (t + k, "loud_sound", "{}"))
w = fresh()                                     # a restarted watcher
w.events.put(("sound", {"kind": "loud sound", "yamnet": {"dog": 0.1, "dog_label": "Growling"}}, t + 100))
w.drain_events(con, b"")
assert kinds().count("loud_sound") == 6, kinds()
print("loud-sound cap: holds across restarts (6/hour)")

# 7. A loud non-dog sound does not carry a misleading dog label.
clear(); w = fresh()
w.events.put(("sound", {"kind": "loud sound", "yamnet": {"dog": 0.08, "dog_label": "Growling", "dog_kind": "growling"}}, t))
w.drain_events(con, b"")
d = json.loads(con.execute("SELECT detail FROM events").fetchone()[0])
assert "dog_label" not in d["yamnet"], d
print("loud sound: no 'Growling' label at dog score 0.08")
print("events ok")
