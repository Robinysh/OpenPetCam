# World model: how each claim is checked

This is the procedure, not anyone's results. Your own ledger (status and numbers for your room) belongs in your world
folder, `eval/<world>/AUDIT.md`, which git ignores: it describes the inside of a home and stays on your computer.

Status words: **supported** (an independent check passed, with its null or baseline), **consistent** (agrees with
another measurement that is not independent), **unvalidated** (no independent check yet), **invalidated** (a check
failed: not used). A number is only a result with its null or baseline beside it. Invalid trials are counted, never
dropped.

Rules:
1. Independent checks: a sensor is judged by a different sensor, never by itself.
2. Every result has a null that should centre on chance, and the null's own centre is reported (a null that does not
   centre on chance is a broken test, not a strong result).
3. Collection order is randomised and interleaved; "same direction" must never also mean "same time".
4. The instrument is modelled before the room: device response, firmware processing, pose drift.
5. Inputs are named exactly: each build records sha256 of every input and the code revision (`manifest.inputs`).
6. Causal filters for picking arrivals; transients or band-averaged delays, never single-tone phase.
7. Splits for held-out tests follow the grouping unit (a day, a 2-hour block, a search), never single rows.

| ID | Claim | Check (independent evidence and its null) | Where |
|---|---|---|---|
| CAM-1 | The camera is calibrated | reprojection error of matched sweep pairs; independently, learned depth of overlapping frames must agree better than with the previous calibration | `camera.py`, `matches.py` |
| CAM-2 | Logged pan/tilt drift is corrected | snapshots registered to the panorama; dog positions beyond the walls before vs after correction | `registration.py` |
| GEO-1 | Floor height | the floor echo across recordings; a tape measure when available (`--tape`) | `sensors/acoustic.py`, `room.py` |
| GEO-2 | The fused surface has the room's shape | unsnapped depth where the dog has stood (the floor, seen by a different sensor) | `depthpano.py`, `fuse.py` |
| GEO-3 | Object boxes | sightline carving held out by 2-hour blocks: dog positions of the held-out blocks must not fall inside a box | `objects.py` |
| GEO-4 | Wall positions | marks by eye against learned depth, after rebasing into the current calibration | `room.py`, `rebase.py` |
| ACO-1 | Echo profiles are repeatable and direction-specific | interleaved, seeded plan: same direction (still and moved) vs different directions vs across days | `sonar_plan.py`, `sonar_scan.py` |
| ACO-2 | Sound refines the geometry | forward model vs recordings, scored leave-one-out, against a direction-shuffle null (stricter than a turned room) | `acoustic_model.py`, `nearfield.py` |
| ACO-3 | Passive echoes from room noise | correlation of ambient noise vs a control segment | `sensors/acoustic.py` |
| ACO-4 | Head directivity | floor echo over direct path vs head pitch across tilt rows | `acoustic_model.py` |
| ACO-7 | Sound fills the camera's blind spots where it holds (`blindfill.py`) | above the lens: the nearest echo's spread over head directions (a flat surface returns one delay), and vision's height of the surface it sees near the top of its sweep, against that echo; elsewhere: the strongest and first echo beyond 0.3 m against vision's range on the head axis, with a direction-shuffle null; the near cabinet echo's strength against facing it, with a shuffle null. Only what passes is filled; the floor under the camera is marked assumed | `python3 -m sim.world.blindfill eval/<world>` |
| SIM-1 | The FDTD solver is correct | closed forms: 1/r spreading, second-order convergence, rigid-wall image source | `python3 -m sim.world.fdtd verify` |
| RF-1 | The camera's WiFi signal tracks motion | per-minute RSSI spread vs picture motion; block bootstrap and circular-shift null | `sensors/wifi.py` |
| NIR-1 | Active minus passive near-IR is the camera's own light | LED light x r^2 roughly constant over range | `sensors/nir_active.py`, `ir_exp.py` |
| PRES-1 | Dog positions on the floor | cached repeats dropped; share beyond the modelled walls | `sensors/presence.py` |
| UNC-1 | The uncertainty map says where the model is weakest | after a next-best-view round, error falls most where the map said it was highest, judged by evidence the new views did not produce | `fuse.py`, `nbv.py`, `remap_capture.py`, `compare.py` |
| FIND-1 | Find performance | every recorded search: truth from the confirmed box or her next resting detection; outcome found / missed in view / hidden / probably out / unknown | `find_bench.py` |
| FIND-2 | Zoomed tiles find a resting pet | hand-labelled frames of the resting spot: recall and false alarms, whole frame vs whole frame or tiles, at the unchanged verifier threshold | `find_bench.py` |
| FIND-3 | Room-map stop order | replay, map built without the searched day: share of searches with her in view at a stop, time to that view, paired difference with a bootstrap interval | `find_map.py`, `find_bench.py --replay` |
| FIND-4 | "Probably out" | share of failed searches followed by >= 30 min without a detection; hand labels where available | `find_bench.py` |

## Find: how a change is judged (no A/B)

A change to Find is judged before it ships by offline replay on the recorded searches (`find_bench.py --replay`,
maps built without the searched day), and after it ships by these numbers on the searches it ran, compared with the
same numbers before it:
- success when she is home: found / (searches minus those followed by >= 30 min without her)
- time and views to a find (median, p90)
- missed in view: searches where the room model puts her in a stopped frame and nothing confirmed her
- "probably out" correctness: share followed by >= 30 min without a detection
