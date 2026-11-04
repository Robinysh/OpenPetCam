# Owner requests, in order

Worked top to bottom. Each item moves to Done with the commit that shipped it.

1. ~~**Live and Today redesign (prototype)**~~ done, canvas v22: Live is camera-first: full-bleed video that keeps the dog framed, what she is doing laid over it, joystick below. Today is visual: photo mosaic, day split, scrubbable timeline, moments as photos. (2 Nov)
2. ~~**Report agent:**~~ done (every 5 min, probe answered 18:30): every "Something wrong?" report is handled as continuous feedback, as the owner's agent: labels from the owner's words, diagnosis, a gated fix and deploy, and a reply in the app and on the phone. A report unanswered for 6 h pages. (2 Nov)
3. **Exhaustive QA** of the setup, software and hardware: check every assumption, test every behaviour. The checks
   become a repeatable suite (tools/qa.py) that the loop re-runs nightly, so a regression is caught without the owner. (2 Nov)
4. **Self-improving loop.** The owner's prompts are signals the system should catch itself. North star: the owner
   always knows what the dog is doing and can see her. Nightly: measure it (detector errors against owner labels,
   time to find her, searches that failed, video start time, reports, usage friction), compare with the trend, and
   have an agent make one improvement under the same gates as the report agent, then report what changed. (2 Nov)
   Rigour (owner: "measure and improve rigorously"): every metric has ground truth from outside the system being
   measured (owner labels, owner reports, hand-checked frames, the motor's own position), a fixed definition and a
   history; a change is scored on held-out days against the current version AND the free baseline, ships only if it
   wins on the north-star metrics without regressing the others, and is re-measured after it ships (kept or reverted).
5. **Learn her hotspots:** infer where she likes to be from where the camera finds her (pan/tilt pose + box over days);
   use them for search order, suggested places, and Today. The first improvement the loop makes from its own data. (2 Nov)
6. ~~Build the approved prototype into the real app~~ done 2 Nov 19:50 (camera-first Live, visual Today). Still open: views and
   zones as one 'place' in the backend (the app shows them in one row for now).
7. Report agent: a fix queued because the code moved on is never retried; rebase it in its worktree, re-run the gates, ship.
8. Motor loudness per speed (tools/noise_study.py loudness): waits for a safe moment; then decide resting moves and search style.

9. **Weakness mining** (Self-Harness, arXiv 2606.09498): a nightly pass over worker and router transcripts, the
   archive and owner reports that names recurring failure patterns before a goal misses, and opens threads for
   them. Today only goal misses and reports start work.
10. **Pairwise gate for changes with no metric** (`~/Documents/auto-improve`, after karpathy/autoresearch): app copy,
   push wording, prompts. An independent judge compares champion and candidate in both orders and keeps the candidate
   only on a 2-0 win; it never grades its own work. Router prompt changes would use it alongside the router cases.
11. **Fixed budget per experiment** (autoresearch): every worker run gets the same wall-clock and token budget so attempts
   in the archive are comparable, and the archive records cost per accepted change.

12. **Fine-tune the pet classifier on footage collected over weeks** (owner, 2026-11-03), replacing the
   logistic regression on frozen MobileNetV2 features, which memorises scenes (rug 0.11 on 2 Nov).
   - Collect: keep every candidate crop with its context (time, pan/tilt, IR or day), deduplicated, bounded in size.
   - Label: bootstrap with `claude -p` as today, owner 👍/👎 as the independent labels, plus hard negatives (toys,
     empty rug, people) and hard positives (curled up, IR, partial).
   - Train on the laptop or the Jetson: unfreeze the last MobileNetV2 blocks (or a small EfficientNet), strong
     augmentation (IR-like grayscale, crops, light), ONNX export at the same speed budget.
   - Judge: held-out by day (never seen days), the 118-frame walk set must stay at 0 false accepts, then a
     shadow run on live frames before it ships; retrain weekly as footage grows.

## Done

- QA suite tools/qa.py: 20 pass, 0 fail (2 Nov 19:10)
- Pause = picture and mic off; sound self-heal; Find <pet> button; searches keep the app fresh (2 Nov)
- Toy accepted as the dog, "moving" while asleep, searching an empty flat every 10 min: `1db6fba` (2 Nov)

13. **On-camera tracker (owner, 2026-11-03: "squeeze as much as possible from the camera").** A tiny model baked into a
   static MIPS binary on the camera tracks her 5-10 times a second next to the motor; the server keeps the "is it her"
   check every few seconds. Measured constraints: Ingenic T23N, one MIPS32 XBurst core (~1185 BogoMIPS), no NN
   accelerator, 37 MB RAM (~7 MB free); camera busy 45% after noise suppression off (was 67%), so ~25-35% is spendable.
   - Data: ~7,500 frames (server candidates/ with same-cycle detections, per-minute snaps, eval/bootstrap/frames) +
     452 owner/Claude-labelled crops. Teacher: the server's YOLOX-nano + verifier (owner labels override).
   - Model: grayscale ~96x54 input, int8 or binary weights compiled into C, a few MMAC per frame (YOLOX-nano is ~1 GMAC);
     output: her box or "not in view". Train on the laptop; judge held-out by day against the teacher.
   - First student (2026-11-03, eval/tiny/): 11,851 params, 1.5 MMAC/frame (~700x less than YOLOX-nano). Teacher-labelled
     3,457 frames (1,434 with her). Held out 11-03: recall 0.88 / precision 0.57 at 0.3, median centre error 0.08 frame.
     Weak on identity (expected: the camera only follows near the server's last confirmed box).
   - Blocker: frames on the camera. prudynt owns the ISP; JPEG/H.264 decode would cost more than the model. Path: run
     the model inside prudynt's IVS/motion frame path (open source, Thingino toolchain), output a box to the motor loop.
   - Ship gate: camera stays under ~70% busy (per-thread /proc profile), tracking beats today's 2.5 s detections in
     the Follow sim fitted to the track table (pan/tilt per cycle, logged since 2026-11-03), and a live A/B.
14. **Server CPU allowance.** The $5 box ran pinned at its 5% cap all day (credits 0%). Shipped 2026-11-03: integer grey
   for motion (18x cheaper), audio floor every 2 s, one multiplexed SSH connection (~3,700 logins/day before). If the
   credits still don't refill: camera motion detection + webhook instead of server polling, keyframe-only decode when
   idle, microSD recording on the camera; int8 on this CPU is likely slower (VNNI hidden by the hypervisor).
