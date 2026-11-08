"""Delay-aware Follow switches in server/tracker.py (sim/follow_delay.py), without hardware.

  1. RETURN_AFTER_PAUSE: a person turns the camera away while she is in view; when the pause ends and she is not in
     the picture, Follow goes back to where she was, once. Off, or if she was not in view when the person moved it,
     there is no return move (the usual lost-pet logic still runs).
  2. SETTLE_LEAD: a trotting dog is aimed further ahead (to when the move will have finished) than with LEAD alone.
  3. CALM_FOLLOW: a dog ambling off-centre (slow, inside EDGE_SLOW) is left alone; one past it is brought back to KEEP
     from the centre on her own side, not re-centred, so ambling back and forth doesn't swing the camera across her.
  python3 tests/server/follow_delay_test.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "server"))
import tracker as T  # noqa: E402
from tracker_test import World, tracker_for  # noqa: E402


def person_moves_away(on, seen_before=True):
    T.RETURN_AFTER_PAUSE = on
    w = World(X=1850, pet=1850)
    tr = tracker_for(w)
    if seen_before:
        for _ in range(4):
            w.advance(3.0)
            tr.step(w.box(), moving=False, box_t=w.t)
    else:
        w.advance(30.0)
    w.advance(5.0)
    w.X = w.tx = 600.0                       # a person turns the camera to another view
    w.moved = w.t
    acts = []
    for _ in range(int(T.USER_PAUSE // 3) + 4):
        w.advance(3.0)
        a = tr.step(w.box(), moving=False, box_t=w.t)
        if a:
            acts.append(a)
    return w, acts


try:
    w, acts = person_moves_away(True)
    assert any(a.startswith("back to where she was") for a in acts), acts
    assert abs(w.X - 1850) < 40 and w.box() is not None, (w.X, acts)
    assert sum(a.startswith("back to") for a in acts) == 1, acts
    print("return after a person's move:", [a for a in acts if a.startswith("back")][0])
    w, acts = person_moves_away(False)
    assert not any(a.startswith("back") for a in acts), acts            # off: only the usual lost-pet logic
    w, acts = person_moves_away(True, seen_before=False)
    assert not any(a.startswith("back") for a in acts), acts            # she wasn't in view: nothing to return to
    print("off, or not in view when the person moved it: no return move")
finally:
    T.RETURN_AFTER_PAUSE = False


def first_aim(settle):
    T.SETTLE_LEAD = settle
    w = World(X=1850, pet=1850)
    tr = tracker_for(w)
    tr.last_move = tr.last_aim_t = -1e9
    for _ in range(12):
        w.pet -= 60                          # trotting toward view-right (lower pan) at 60 steps per second
        w.advance(1.0)
        a = tr.step(w.box(), moving=True, box_t=w.t)
        if a and a.startswith("track"):
            return float(a.split("aim (")[1].split(",")[0])
    return None


try:
    plain, settle = first_aim(False), first_aim(True)
    assert plain is not None and settle is not None, (plain, settle)
    assert abs(settle) > abs(plain), (plain, settle)
    print(f"settle lead aims further ahead: {plain:+.2f} -> {settle:+.2f}")
finally:
    T.SETTLE_LEAD = False


def amble(calm):
    T.CALM_FOLLOW = calm
    w = World(X=1850, pet=1850)
    tr = tracker_for(w)
    tr.last_move = tr.last_aim_t = -1e9
    acts = []
    for k in range(40):                      # ambling 0.28 frame either side of centre, about 0.1 frame/s
        side = 1 if (k // 6) % 2 else -1          # relative to where the camera points now
        w.pet = w.X + side * (0.24 + 0.01 * (k % 6)) * 733
        w.advance(1.0)
        a = tr.step(w.box(), moving=True, box_t=w.t)
        if a and a.startswith("track"):
            acts.append(float(a.split("aim (")[1].split(",")[0]))
    return acts


try:
    plain, calm = amble(False), amble(True)
    assert len(calm) < len(plain), (plain, calm)
    flips = lambda a: sum(1 for x, y in zip(a, a[1:]) if x * y < 0)      # noqa: E731
    assert flips(calm) <= flips(plain), (plain, calm)
    print(f"calm follow while she ambles: {len(plain)} aims -> {len(calm)}")
    T.CALM_FOLLOW = True
    w = World(X=1850, pet=1850 - int(0.30 * 733))      # off-centre and walking out: still followed, to KEEP not 0
    tr = tracker_for(w)
    tr.last_move = tr.last_aim_t = -1e9
    a = None
    for _ in range(4):
        w.pet -= 25
        w.advance(1.0)
        a = tr.step(w.box(), moving=True, box_t=w.t) or a
    assert a and a.startswith("track"), a
    fx = float(a.split("aim (")[1].split(",")[0])
    assert fx > 0.1 and 0.08 < abs(w.fx()) < 0.45, (a, w.fx())     # followed, kept in view, not re-centred
    print(f"calm follow past the edge: {a.split(':', 1)[1].strip()}, she ends {w.fx():+.2f} from centre")
finally:
    T.CALM_FOLLOW = False
print("follow delay ok")
