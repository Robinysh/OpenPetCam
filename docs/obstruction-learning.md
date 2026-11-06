# Learning obstructed views in simulation

```sh
python3 -m sim.obstruction_bench --out eval/obstruction_learning/run \
  --train-groups 160 --development-groups 64 --test-groups 128 --episodes 100 --jobs 4
python3 -m sim.obstruction_report eval/obstruction_learning/run \
  --out eval/obstruction_learning/index.html
```

Optionally supply this installation's private `--room models/room/room.json`
for a photographic-background sensitivity bank. All fitted models, frames,
labels and results go into the requested ignored output directory. Nothing
contacts a camera, changes the production watcher, modifies a search policy,
or writes the installed room model.

## The experiment

The operational target is to avoid repeatedly inspecting an obstructed view
while reopening it when conditions change. It is not possible to establish
physical permanence from a finite sequence of images. The memory therefore
uses temporary exclusions with explicit expiry and mandatory reinspection.

`sim/obstruction_scene.py` generates camera rays using the repository's camera
model and intersects near boxes and spheres in three dimensions. Procedural
spherical backgrounds and optional room photographs supply the RGB background.
This is an image-composite experiment with geometric near objects, not a
validated room reconstruction or light-transport simulation.

The renderer returns RGB only. A separate evaluator reports injected-object
coverage. A per-frame blocking label means an injected object occupies at
least 85% of the image. Partial obstructions do not make an entire view
unsearchable. Persistent and transient objects retain separate evaluator
counts; either can temporarily make inspection unhelpful.

Training, development and test sets use separate background, object and
sensor identities, and nonoverlapping angular sectors. Supervised labels
train the models; predictions consume RGB-derived features only. Models and
thresholds are selected on development scenes and saved before the fresh test
namespace is rendered. A zero observed false-positive count on development
does not establish zero future risk. An unsafe calibration is disabled rather
than silently selecting a threshold that still produces false positives.

Candidates include a regularized linear classifier, bounded forests with two
depths, and a forest using a current/reference RGB pair. References are
pre-acquired observations from before the injected objects appear. They are
not regenerated as clean current-time counterfactuals. Small acquisition-pose
errors, sensor noise and exposure differences vary the input. Stress increases
registration error. A known remount invalidates old references; missing or
unregistered references produce no obstruction prediction. This conditional
experiment does not establish that an actual saved household image is clear
or accurately registered.

Lifecycle banks compare inspecting every scheduled view, the deliberately
unsafe rule “repeated pet misses mean blocked,” and the learned visual memory.
They include empty views, partial objects, short and long temporary objects,
removed objects, moved objects, announced remounts, darkness and blur. The
empty-view control eventually supplies a scripted positive pet observation.
That checks the evidence contract; it is not a rendered pet-detector benchmark.

Counts include every scheduled opportunity, deferred mandatory probes and
missed scripted positives. Savings count actual exclusions separately from
budget deferrals. These are view-opportunity counts, not measured motor travel,
search time or live detection recall. Recovery denominators retain cases that
never recover. Cases that never learned an exclusion are identified rather
than presented as successful recovery from a learned obstruction.

## The memory contract

`sim/view_obstructions.py` is an offline component. Its inputs are an exact
pose, capture/observation times, setup fingerprint, image validity/quality,
visual obstruction probability, and any verified pet detection from that
frame. A missing pet detection alone supplies no visual obstruction evidence.

Repeated fresh, settled, valid high-scoring observations over a minimum span
can temporarily exclude a pose. Clear or ambiguous visual evidence and fresh
verified pet evidence reopen it. Due probes and expired exclusions also reopen
it. Invalid, stale, moving or duplicated frames cannot extend an exclusion.
Repeated visits are distinct in time; they are not statistically independent
proofs against a systematic visual classifier error.

Probe proposals prioritize mandatory reinspections and rotate attempted
unusable views so they cannot starve others. Scheduling alone consumes no
inspection; attempts never renew obstruction evidence or expiry. A due view
remains eligible even if its probe is deferred by the budget.

The caller supplies the authoritative current context. Geometry, active mount,
map/calibration, classifier and preprocessing signatures belong in that
fingerprint. An explicit setup change clears all beliefs and probe history,
even if the fingerprint string is reused. Delayed old-context frames cannot
restore an older setup or advance the current observation clock. Persisted
state is validated and remains subject to context and expiry checks.

## What a pass would establish

The recorded simulation checks require no held-out pixel false positives,
useful recall, no false whole-view exclusions or missed scripted pet
opportunities, useful exclusions, and bounded reinspection of previously
learned views. Unit tests separately check context invalidation, freshness,
expiry, queue fairness, serialization, label separation and invalid calibration.

A photo-background bank is sensitivity evidence only. Its labels describe
newly injected geometry; the original photograph is unlabelled. Exclusions
outside the injected mask are uncertainty signals, not independently verified
household false positives. No synthetic pass authorizes live suppression.

Real integration must first validate captured pose and image registration,
classify annotated real views, retain frame-linked evidence, and run in shadow
mode before it can influence the search. Existing `pose_observation` rows
retain image hashes rather than RGB, so a classifier must consume the existing
in-memory image or verified retained frames. The current low-sharpness alert
is a separate generic signal and does not establish a permanent obstacle.
