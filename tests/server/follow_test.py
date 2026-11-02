"""Follow-mode decisions with simulated detections (camera move mocked)."""
import sys
sys.path.insert(0, "/opt/petcam")
import watch
moves = []
watch.camapi.aim = lambda fx, fy: moves.append((round(fx, 2), round(fy, 2)))
w = watch.Watch.__new__(watch.Watch)
w.setting = lambda k, d=None: True                     # follow on
edge = {"box": [0.80, 0.40, 0.98, 0.70], "score": 0.6}   # near the right edge
mid = {"box": [0.40, 0.40, 0.60, 0.70], "score": 0.6}
cases = [("centred, moving", mid, True, 0), ("edge, still", edge, False, 1), ("edge, moving #1", edge, True, 2),
         ("edge, moving #2", edge, True, 3), ("edge, moving #3 (rate limit)", edge, True, 6), ("edge, moving after 12 s", edge, True, 16),
         ("edge, moving after 12 s #2", edge, True, 19)]
for label, best, moving, t in cases:
    before = len(moves)
    w.maybe_follow(1000 + t, best, best["score"], moving)
    print(f"{label:<28} -> {'MOVE ' + str(moves[-1]) if len(moves) > before else 'no move'}")
w.setting = lambda k, d=None: False
w.maybe_follow(2000, edge, 0.6, True); w.maybe_follow(2003, edge, 0.6, True)
print("follow OFF, edge moving x2       ->", "no move" if len(moves) == 2 else "MOVED (bug)")
