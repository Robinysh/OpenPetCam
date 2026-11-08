"""After a search that FOUND her, losing her again searches after SEARCH_AFTER s, not SEARCH_EVERY (report 20261108-090206,
the 4th report of this gap: found at 8:55, walked off at 8:59, no search for 10 min). Bounded: at most AFTER_FIND_MAX such
searches in AFTER_FIND_WINDOW; an empty search falls back to the 10-min wait.
  python3 tests/server/search_after_find_test.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "server"))
import tracker as T  # noqa: E402
from tracker_test import World, tracker_for  # noqa: E402


def lost_then_wait(tr, w, seconds):
    """Step with no box for `seconds`; return the search results that happened."""
    out = []
    for _ in range(int(seconds)):
        w.advance(1.0)
        a = tr.step(None, moving=False)
        if a and a.startswith("search"):
            out.append((round(w.t), a))
    return out


# 1. A search finds her; she walks off out of the room's reach of that view; a new search starts ~SEARCH_AFTER later.
w = World(X=1850, pet=1850)
tr = tracker_for(w)
res = tr.search("test")
assert "found her" in res and tr.last_search_found, res
w.pet = 99999                                   # she walked out of every view
t_lost = w.t
s = lost_then_wait(tr, w, T.SEARCH_AFTER + 30)
assert s, "no search after losing her right after a find"
import re  # noqa: E402
gone = int(re.search(r"out of sight (\d+) s", s[0][1]).group(1))
assert T.SEARCH_AFTER < gone <= T.SEARCH_AFTER + 5, s     # started right after SEARCH_AFTER, not SEARCH_EVERY
print("re-search after a find:", s[0])

# 2. That search was empty: back to the 10-min wait (no search for the next 5 minutes).
s2 = lost_then_wait(tr, w, 300)
assert not s2, s2
print("after an empty search: waits", s2)

# 3. Bounded: found, lost, found, lost... the early searches stop after AFTER_FIND_MAX in the window.
w = World(X=1850, pet=1850)
tr = tracker_for(w)
early = 0
for i in range(6):
    w.pet = 1850
    res = tr.search("test")                      # found her where the camera is
    assert "found her" in res, res
    w.pet = 99999
    early += len(lost_then_wait(tr, w, T.SEARCH_AFTER + 5))
assert early == T.AFTER_FIND_MAX, early
print(f"bounded: {early} early searches in a burst (max {T.AFTER_FIND_MAX})")
print("search after find ok")

# 4. Report 20261107-095133: chases that keep ending "not found" (she is behind the sofa) stop after CHASE_MISS_LIMIT
#    in CHASE_MISS_WINDOW; a first chase still happens (report 2026-11-05 17:47 needed it).
w = World(X=1850, pet=1850)
tr = tracker_for(w)
assert not tr._chases_missing(w.t)
for _ in range(T.CHASE_MISS_LIMIT):
    tr.chase_misses.append(w.t)
assert tr._chases_missing(w.t) and not tr._chases_missing(w.t + T.CHASE_MISS_WINDOW + 1)
tr.chase_misses.clear()
tr.last_seen_c, tr.last_moving, tr.last_vel = (0.3, 0.0), True, (0.1, 0.0)
tr.last_seen_t = w.t - 2
for _ in range(T.CHASE_MISS_LIMIT):
    tr.chase_misses.append(w.t)
tr.chased_for = None
w.pet = 99999
a = tr.step(None, moving=False)
assert not (a or "").startswith("chase"), a
print("repeated chase misses: holds still")
print("chase hold ok")
