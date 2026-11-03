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

## Done

- QA suite tools/qa.py: 20 pass, 0 fail (2 Nov 19:10)
- Pause = picture and mic off; sound self-heal; Find the dog button; searches keep the app fresh (2 Nov)
- Toy accepted as the dog, "moving" while asleep, searching an empty flat every 10 min: `1db6fba` (2 Nov)
