# Follow that learns your camera and your pet

Every camera is different: its motor speed, its lag, how far one motor step turns the picture, how much slack the gears
have. So is every pet: where they lie, how fast they walk, how long they stay. A Follow policy tuned for one home is a
guess in the next. OpenPetCam therefore learns both from the camera's own logs, builds a simulator of that home, trains a
policy there, and only keeps it if it wins, first in simulation on rooms it never trained on, then live.

```bash
python3 sim/selfcal.py --sysid --deploy      # the whole loop for this install
python3 sim/ab_report.py 3                    # after a few days: does the learned policy stay?
```

## The loop

| Step | What it does | Check that must pass |
|---|---|---|
| 1. Export | Pulls the watcher's own logs: one row per cycle (detection, box), the camera pose and Follow's action per cycle, searches and comings and goings. Numbers only; pictures never leave the server. | |
| 2. Fit (`sim/fit_from_logs.py`) | Cycle timing by state, detection persistence, where the pet is found (a map in motor coordinates), rest and move durations. Also measures the real metrics: share of time in view while home, losses per hour, how fast she is seen again, actions per hour. | |
| 3. Measure the camera (`sim/sysid_camera.py`) | Motor delay and speed, whether a new command queues, frame size in motor steps and gear backlash from image shifts (phase correlation). Refuses while the pet is resting in view; pauses Follow; undoes every move. | Each measurement repeated; accepted only if trials agree within 5% |
| 4. Calibrate (`sim/calibrate.py`) | What logs cannot show directly (true walking speed, time away, how often people set off motion) is fitted by running the real controller in the simulator and matching the statistics it would have logged to the real ones (CMA-ES, the same seeds for every candidate). | |
| 5. Gate 1 (`sim/gate1.py`) | The unchanged controller, in the simulator, on seeds never used for calibration. | Real metrics inside the simulator's range |
| 6. Train (`sim/train_policy.py`) | Direct policy search (CMA-ES) over the controller's tuning constants, on randomised rooms around the calibrated ones, scored by the worst quarter of rooms, with no more motor moves than today. A recurrent PPO policy (`sim/train_ppo.py`) is trained on the same environment as a challenger. | |
| 7. Gate 2 | Learned vs today's constants on 32 held-out rooms (different seeds and parameter draws). | More time in view, better worst quarter, no more moves |
| 8. Live check (gate 3, `sim/ab_report.py`) | No rotation since 0.2.3 (owner: no live A/B). The report reads the logs before and after a change, hour-of-day matched; the 2026-10/11 rotation logs remain readable. | Logged in-view share not lower and moves per home hour not higher, over 8+ home hours per arm; else remove `models/follow_policy.json` |

The policy is the existing controller with learned constants, so it costs nothing to run on a $5 server, stays readable,
and keeps every hard rule: the server accepts only whitelisted constants clamped to fixed bounds
(`server/tracker.py` `POLICY_BOUNDS`), and "no automatic moves near a resting pet" is enforced in code, never learned.

## Measured on the reference install (Cinnado D1, 2026-11-04)

- **Camera (sysid):** 216-218 steps/s at the quiet speed, 50-120 ms from command to first motion, a new command waits for
  the move in progress; frame 740 steps wide (configured 733); **27 steps of gear play**: after a reversal the step
  counter moves at once and the picture only after the play (the simulator models counter and optics separately).
  Pan frame and play differ by pose (707-740 steps, 27-41 play; the lens is likely off the pan axis), so training
  randomises over the measured range. Tilt: 431 steps per frame and essentially no play (1 step), measured only once
  every compared frame sat in the textured band (above tilt ~650 this camera sees a blank ceiling).
- **The driver's tilt range check is inverted:** a relative tilt step is accepted only while y + cmd stays in 0..1000,
  yet the head moves to y - cmd. A naive sweep left tilt stuck at 0 (fixed by homing); calibration and sweeps now move
  through the app's own `goto()`, which splits long tilts.
- **Sensing:** frames 0.7 s late; the detector runs every 30 s while she rests (8 s while lively, at once on motion),
  so one missed detection becomes a 30-second "loss" in the logs, and the simulator runs the same schedule.
- **You:** people turn the camera by hand about 3 times an hour, which pauses Follow for 3 minutes; measured from the
  logs and part of the simulated room.
- **Gate 1:** 7 of 7 real metrics inside the simulator's range on held-out seeds (logged in-view 0.756 real, 0.731 sim).
- **Gate 2:** learned constants vs today's on 32 held-out rooms: true in-view while home 0.624 -> 0.659, worst
  quarter 0.431 -> 0.484, motor moves 31.2 -> 26.7 per hour, better in 15 of 32 rooms. The policy re-aims at most
  every ~5 s, chases only from a clear edge exit, and searches later.

- **RL challenger:** recurrent PPO (64-unit LSTM, 600k steps, 18 min on the laptop CPU) on the same environment and
  the same 32 held-out rooms: in-view 0.596, worst quarter 0.251, **237 motor moves per hour**. It lost to both, so the
  deployed policy is the tuned controller. Before trying again: a much higher per-move cost (or the move budget as a
  hard constraint), the resting shield plus a move budget in the action mask, and longer training (Modal CPU sweeps).

## The pixel twin (in progress)

The policy only sees detector output, so pixels matter in one place: when and where the real detector misses or
false-accepts. Plan (research memo 2026-11-04): a panorama from the camera's own sweeps (`sim/sweep.py`, done for day),
depth for it from a phone walkthrough (VGGT / MASt3R), the dog inserted at the right scale and occlusion, the real YOLOX
and verifier run on those frames, and a perception error model fitted from them and fed into the fast simulator, which
stays the training engine. Cosmos-Transfer adds lighting and appearance variety as augmentation; it is too slow and has
no pan/tilt control to be the simulator itself (~$2k per million steps). Needed from the owner: a 2-3 minute phone
walkthrough including views toward the camera, a night sweep, and ~300 labelled frames.

## Training across days

The two real days fit very differently (calm: rests ~24 min, walks 37 steps/s; active: rests ~1 min, walks 322 steps/s),
so `sim/per_day.py` calibrates one simulator per day and training draws rooms from all of them. A policy searched over
both days failed gate 2 (0.612 vs 0.645); the deployed one still wins on cross-day held-out rooms: in-view 0.645 -> 0.662,
moves 44.5 -> 40.4 per hour (calm-day rooms 0.645 -> 0.679, active-day rooms unchanged with 10% fewer moves).

## PPO, honestly

- With a hard move budget (6 per 10 min, enforced in code), recurrent PPO first scored in-view 0.704 on cross-day
  held-out rooms, beating the tuned constants (0.662), and passed every parity test (ONNX actor 0/500 mismatches,
  server controller 1,144/1,144 identical decisions, simulated watcher loop 0.705).
- A forced 10-minute **live** smoke test (`/var/lib/petcam/policy_override`, self-expiring) then showed it choosing
  "search" on 183 of 186 cycles with the dog in view: in the simulator a search with her visible is a free 4-frame look
  that costs no move budget. Live it wrote 201 fake Find searches (relabelled `search_smoketest`, not deleted).
- With searches allowed only after 10 s unseen (`server/follow_obs.py`, shared by training and the server), retrained
  PPO scores 0.518: the whole advantage was the loophole. PPO is off the camera; the tuned constants stay the learned
  arm. Next RL attempt: invalid actions masked in the policy itself (it still wastes ~200 choices per room on refused
  searches), occlusion-aware visibility, longer training.
- The lesson kept as a rule: every learned policy gets a forced live smoke test before its A/B block.

## Finding: where to look first, learned

- `sim/train_search.py` learns the search order from where she is actually found (a 2D sighting map from the logs, 20%
  uniform, Claude's rest places), greedy by probability covered per second of travel and look, skipping blank views.
  Judged by the real tracker search in the simulator on 120 held-out trials: median time to find 15.2 -> 11.0 s, found
  within 60 s 0.908 -> 0.950; with her placed uniformly anywhere 36.4 -> 29.6 s, 0.683 -> 0.742.
- Live check 2026-11-04: turned away from her, Follow's own search visited the learned stops and found her at the 5th
  (the Door area) in 44 s. One trial; the live A/B decides whether it is faster on average.

## Claude builds the scene

`sim/scene_from_claude.py` gives Claude the camera's own panorama and gets back exits, occluders, rest places and blank
views, converted to motor coordinates. Without being told, it put the furniture and doorways where the owner's saved
views are. The simulator walks her out through those doorways, hides her for a rest behind or under
furniture (calibrated: 76% of rests inside an occluder), and the search skips blank walls. With the scene, gate 1 holds
(6/7), the tracking constants tuned without occlusion LOST to today's (0.535 vs 0.564), and constants retrained on it
win on 32 held-out rooms: in-view 0.564 -> 0.583, worst quarter 0.275 -> 0.383, moves 42.6 -> 40.5 per hour. Those, with
the learned search, are the live learned arm.

## 2026-11-04: what an audit of real searches and an outside review changed

- **The simulator thought Find was easy.** On its own held-out trials Find succeeded 98% of the time; the 84 real
  searches since 2026-11-03 succeeded 39% (automatic ones). Labelling the failures by eye: 8 of 12 failures that had a
  sighting were really her, two lying fully in view; the detector's raw score on a lying dog sat at 0.10-0.53, so it
  proposed her on few frames while the verifier was sure whenever it did. The 99%-per-frame detection the simulator
  used was measured while she was already being tracked, i.e. on the poses it sees well.
- **Hard poses** are now part of the model: each lie-down is hard with `q_hard` and then seen on `p_hard` of frames,
  calibrated together with the real search success rate (fit: about half the lie-downs, 4% of frames).
- **Frame timing parity** (outside review): the simulated settled look returned a frame exposed before the requested
  time and ignored "since" mid-pan; it now follows the watcher's rule (a frame taken after the request, a new one per
  look, on the 5 fps grid). Sensing has its own random stream, so policies are compared on the same dog.
- **Result:** the re-fit passes gate 1 on held-out seeds (7/7, 8 seeds x 4 h; the ranges are wide) and reproduces Find
  (40% vs 39% real). On it, the previously deployed learned constants were no better than today's (robust 0.132 vs
  0.141); constants retrained on it pass gate 2 against both (in-view 0.460 vs 0.437 / 0.440, robust 0.152, moves 52.8 vs
  53.6 per hour; 13 of 32 rooms won, so the gain is on average and in the worst rooms, not everywhere). They are the
  learned arm from 2026-11-04 14:02. Two earlier retrains on intermediate fits failed gate 2 and were not shipped.
- **Where Find's failures are** (calibrated simulator, 169 searches on held-out days): she was out of the room for 85
  (found 2), behind furniture for 15 (found 0), in a hard pose for 13 (found 4), lying normally for 55 (found 55). The
  next Find work is therefore inference (where did she go: an exit, behind the sofa) before sweeping, not a faster
  sweep. Shipped the same day: a single sure glimpse is followed up (aim at it, look again), and sure sightings add up
  across looks by room position; in the simulator Find went 31% -> 35%.
- **The live A/B was not a valid gate** (outside review). "Home" was inferred from the detections being judged, so an
  arm that lost her for long stretches dropped them from its own denominator (a 25%-visible arm scored 95%); a search
  counted as one move; and two arms in 2-hour blocks got the same hours every day. `sim/ab_report.py` now scores
  visibility over all watched time, hour of day by hour of day on hours both arms covered, counts every motor command
  (`track.moves`), and the rotation shifts one block per day. The A/B restarted with these on 2026-11-04.

## 2026-11-08: what a live walk showed that the simulators can't

An outside review proposed predicting her to when an aim will have finished, and learning only a small correction on
top. `sim/follow_delay.py` tested that on both simulators, before any reinforcement learning:

- **Following a moving dog is nearly solved in simulation.** Of home time, she is out of the frame while moving 0.9%,
  hidden behind something 8.7%, and lying in the open outside the frame 38.5%. The loss is reacquiring a resting dog.
- **Counterfactuals, same random draws:** hard lying poses always detected 0.52 -> 0.63 in view; nobody ever turning
  the camera by hand 0.52 -> 0.58 (most of it inside the 3-minute pause a person's move earns); searching every 2 min
  instead of 10 makes it worse (0.49).
- **Delay-aware prediction (`SETTLE_LEAD`), capture-time stamping (`CAPTURE_TIME`) and returning to her after a
  person's move (`RETURN_AFTER_PAUSE`)** did not beat today's constants on held-out rooms in either simulator, nor after
  re-tuning both arms the same way (re-tuning on 24 rooms overfits: both tuned arms lost to the deployed constants).
  They stay in `server/tracker.py`, off.
- **A supervised live walk (6 min, owner walking the dog)** showed what neither simulator reproduces: 35 tracking aims,
  many undoing the previous one while she ambled at 0.05-0.16 frame/s 0.2-0.4 from the centre. Stills from the
  recording at 12 logged misses: 7 had her outside the frame, 3 had her cut off at an edge, 2 lying belly-up in view.
- **Calm Follow (`CALM_FOLLOW`, on since 2026-11-08):** speed can't tell ambling from walking (a steady walk across
  the room is ~0.15 frame/s, the walk's ambling 0.05-0.16), direction can. Once the camera aimed the other way within
  `AMBLE_WINDOW` (20 s) she is ambling: she must be predicted past `EDGE_SLOW` (0.32) before it moves, and an aim leaves
  her `KEEP` (0.15) from the centre on her own side. A first version gated on speed alone lost the steady-walk
  benchmark (77% -> 61% in view) and was replaced. The direction version holds every follow benchmark scenario (walks
  across 77 -> 76%, walks past a lookalike 73 -> 79% and 1.6 -> 0.1 s on the toy) and on the walk's logged decisions
  would have made 25 of the 35 aims with 24% less travel (10.4 -> 7.9 frame widths) and 11 instead of 17 swings across
  her. It changes nothing in either simulator, which don't model ambling.
- **Live measure:** `python3 tools/follow_live.py EXPORT.json --since/--until`: aims, reversals, travel and the
  share of cycles she is verified, per active hour (6 days before: 13.5 aims, 2 reversals, 3.8 frames, 36.6% seen).



- **A live verdict.** The corrected A/B needs about three days of blocks before it can decide.

- **Predicting a held-out real day.** Leave-one-day-out (fit on one real day, predict the other) put only 1-3 of 7
  real metrics inside the simulator's range. Two reasons: the two days differed (a calm day, 1.8 tracking aims an hour;
  an active day with walks, 7.8 an hour), and about 20 versions of the tracker ran in those two days, so the logs
  cannot be replayed with one controller. Since 2026-11-04 every cycle logs `arm@version`
  (`track.policy`), and `sim/gate1.py load_tracker(rev=...)` replays the exact code; `sim/validate_real.py` becomes a
  fair test once a week of logs from one version exists. Until then the live A/B (gate 3) is the deciding test.
- **One parameter set is not enough.** Day-to-day variation is real, so calibration should produce a distribution
  (one fit per day or per block) and training should randomise across it, as BayesSim and SimOpt do.
- **A camera other than the reference one** has not been run through the loop yet.

