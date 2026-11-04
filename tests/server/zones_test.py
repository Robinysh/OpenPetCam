"""Zone-visit decisions with simulated detections (camera, events and clips mocked)."""
import os
import sys, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "server"))
sys.path.insert(0, "/opt/petcam")
import watch
events, updates = [], []
watch.add_event = lambda con, kind, detail=None, thumb=None, ts=None: events.append((kind, detail, ts)) or len(events)
class Con:
    def execute(self, sql, args): updates.append(json.loads(args[0])); return self
w = watch.Watch.__new__(watch.Watch)
w.zones = [{"id": "z1", "name": "Water bowl", "kind": "water", "rect": [0.6, 0.6, 0.9, 0.95], "pose": [1700, 450], "dwell_s": 6, "cooldown_s": 300}]
w.load_zones = lambda: w.zones
w.pose = (1700, 450); w.current_pose = lambda t: w.pose
w.save_thumb = lambda jpg, t, kind: None
class Clips:
    def want(self, *a): pass
w.clips = Clips()
inside = {"box": [0.65, 0.5, 0.8, 0.8]}          # bottom-centre (0.725, 0.8) is in the zone
outside = {"box": [0.1, 0.2, 0.3, 0.5]}
def step(t, best, label):
    n0 = len(events)
    w.check_zones(Con(), t, best, b"")
    if len(events) > n0: print(f"t={t:4d}s {label:<34} -> visit STARTS {events[-1][1]}")
for t in (0, 3):            step(t, inside, "in zone")
step(6, None, "one missed detection")
for t in (9, 12, 30):       step(t, inside, "in zone")
for t in (33, 45, 60):      step(t, outside, "walked away")
print("visit closed with:", updates[-1] if updates else None)
for t in (100, 110):        step(t, inside, "back at bowl inside cooldown")
w.pose = (2600, 450)
for t in (500, 510, 520):   step(t, inside, "camera panned away")
w.pose = (1700, 450)
for t in (600, 603, 606, 609): step(t, inside, "back at view, after cooldown")
print("visits started:", len(events))
assert updates, "the first visit must close when she walks away"
assert len(events) == 2, ("one visit, none inside the cooldown or with the camera elsewhere, one after the cooldown", events)
print("zones ok")
