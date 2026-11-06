# Find routes: offline experiment

`sim/search_bench.py` compares routes by running the production
`server/tracker.py` search against controlled simulated scenes. It does not
write `models/follow_policy.json`, change the live search, or contact a camera.

```sh
python3 -m sim.search_bench --out eval/route_search/run \
  --dev 192 --test 1200 --stress 600
python3 -m sim.search_report eval/route_search/run \
  --out eval/route_search/index.html
```

The default input paths use this installation's existing room map, motor
measurements and detector-derived angular histogram. Results and household
inputs stay in ignored `eval/`. Unit tests generate their own fixtures and do
not require recordings, a room model or a running server.

## What is compared

The incumbent is the real `Tracker.search_stops()` order: room-map stops for
the scenario hour, merged with the learned policy's stops and filtered through
the floor map. A camera pose and hour are explicitly set for each trial.

Alternatives reorder that same action set by nearest travel or exact finite
planning cost; or select up to eight stops from all reachable floor-bearing map
poses using remaining prior mass per travel/observation cost. Exact reordering
of those eight selected stops and two wider fallback row spacings are also
compared. Only the stop provider and declared row spacing change. Initial-view
confirmation, fresh-frame checks during sweeps, backtracking, glimpse handling,
confirmation rules and parking behavior remain the real controller's code.

`sim/search_routes.py` scores travel time, start/settle/observation delays,
missed prior mass and a terminal failure cost. Its dynamic program exhausts
orders (and, optionally, subsets) of at most twelve supplied actions under
binary, stationary visibility. Tests compare it with independent exhaustive
enumeration. This proves an optimum for that finite planning model. It does
not prove an optimum for all camera movements, moving targets, room geometry
or the executed controller. The controller may skip stops near its initial
pose and observe a target during a pan; the planning surrogate does neither.

## Reproducibility and scoring

`sim/search_scene.py` retains the existing motor physics and camera interface.
Target motion is an absolute function of scenario time. Sensor outcomes are
keyed by scenario seed and frame number. Policies share the same world and
detector draws; extra looks cannot change the target's subsequent motion.
Returned boxes retain exposure-time truth, including camera pose and false
detections. Being physically in view when a false box is accepted does not
make a false confirmation correct.

The banks balance six declared strata: prior-weighted stationary locations,
broad placements that weaken the prior, prescribed motion, low detection
probability, persistent occlusion, and absence. They are stress coverage, not
estimated household frequencies. The stress bank increases latency and false
accepts and varies speed, measured field of view and backlash. A separate
absent-target bank checks cancellation. Random owner movements are disabled
during route ranking.

Every trial has the same 120-second deadline. Correct finds pay their elapsed
time; wrong finds and misses pay 120 seconds, including early failed searches.
Absent and occluded cases remain in the denominator. Movement is integrated
from actual motor-counter motion, including interrupted pans. It ends when
search returns; a pending centering move is outside this measurement. Planning
CPU time is excluded, as routes are prepared offline.

The explicit development preference is capped loss plus two seconds per
thousand motor steps. Development selection permits no extra wrong finds and
at most a two-percentage-point regression in correct or 60-second finds.
Candidate selection is written before either independent simulation bank is
run. Test/stress acceptance repeats these guards and requires a lower mean
score. The runner records the guard outcomes; it never installs a winner.

Outputs include input hashes, full cases and outcomes, the frozen selection,
routes, failure counts, paired bootstrap intervals, cancellation checks and
sampled camera traces. Intervals describe simulation sampling variation under
these assumptions. They do not include uncertainty in geometry, detector
fidelity or household habits. New test seeds can be supplied explicitly; they
must remain distinct from development seeds. Do not keep adjusting policies
against an exposed test bank and call it held out.

## Room-model boundary

The map's floor fraction describes how much of a view contains floor. The
view's center is not a measured pet position or a guaranteed unobstructed
floor point. The prior is therefore an angular proxy with a broad-mass mixture,
not an occupancy map. Histogram alignment after remount is an explicit
assumption recorded in provenance. Existing saved stops are already in current
coordinates; only the old grid and the assumed histogram alignment receive
the remount shift.

The current room map includes all recorded days. Independent simulation seeds
do not make its geometry or habit prior independently validated. Existing 3D
and archived replay tools also have limitations: weak failed-search labels,
pre-remount traces, and behavior resampling that can cross held-out-day
boundaries. Their old scores must not be used as an independent accuracy gate.

Before a live route change, resolve any simulation accuracy regression and
check the frozen candidate against independently annotated recorded searches.
Use matching mount coordinates, expose uncertain/unknown locations, and test
room visibility against actual frames before trusting it as geometric truth.
The existing Find-success and Find-time goals remain the live outcome measures;
an offline speed improvement does not complete either goal.
