# The super-agent: how OpenPetCam improves itself

The owners asked for a system that runs and improves itself without them invoking it, stays observable (a
dashboard) and stays steerable (by talking to Claude Code). This is that system. Code: `tools/superagent.py`
(brain), `server/agentdb.py` (state), `server/goalmonitor.py` (read-only monitoring),
`server/agent.html` (work and decisions at `/agent`), `agent/` (goals, steering, policies).

## Running it

If you use [Claude Code](https://claude.com/claude-code), `bash tools/install_autopilot.sh` runs an agent on your
computer that treats what goes wrong as feedback and fixes it:

- **"Something wrong?"** in the app files a report with the frame, the 90 s of video before it and what the phone
  showed. The agent diagnoses it, writes a fix with a test, ships it through the checks below and answers you in
  the app.
- **Goals** in `agent/goals.json` (find success, CPU, freshness, reports answered, QA) are measured every 30 min.
  A missed goal becomes work.
- **Every change is judged by things it can't edit:** tests, a guard against removed assertions, a held-out replay
  of detection and search, a shadow run for anything that changes how frames become "your pet", a health check
  with rollback, and a check 24 h later. Rejected attempts are kept and shown to the next attempt.
- **Fixes are shared.** When a fix has run live and healthy on your install, the agent opens a pull request to this
  repository so every install gets it. The PR carries a public summary and the checks it passed; nothing from your
  home (frames, clips, times, names, addresses) goes in, and the public-repo guard and your pet's name are checked
  again before anything is pushed. Needs a fork as `origin` and `gh` signed in; `CONTRIBUTE=off` keeps fixes local.
- **Watch it** in **Monitoring**: measured goals beside the behavior you want to keep an eye on.
  **Agent activity** shows its work and recorded decisions.
  **Steer it** in plain words: edit `agent/steer.md`, or tell Claude Code.

Design and the research it draws on: [docs/superagent.md](superagent.md).

## Request board states

Backlog is work or investigation the agent still needs to do. Monitoring holds continuously measured goals
and behavior the owner wants to keep an eye on during normal use, with no decision requested now. Needs review is reserved for a concrete
question and supporting material the owner can inspect now. Missing validation, an expired investigation,
or a failed attempt belongs in Backlog. A legacy freeform `needs_owner` prompt does not make a review ready.

For a resolution component, use `state: "monitoring"` for observation. Use `state: "needs_review"` only with
`review: {"question": "...", "evidence": ["..."]}`. The evidence should be inspectable within the card,
such as a concise comparison or complete policy tradeoff; a private filesystem path alone is not useful
to an owner on a phone. A policy decision can be ready without an implementation prototype.

Without a component resolution, an answered reply may supply the same bundle in `fix.review` for a
`live`, `no_change`, or `planned` outcome. Failed or still-running attempts do not enter the review queue.
Manual moves to Needs review ask what can be reviewed or decided. Moving a card never creates completion
evidence, and Done still requires every requested component to be verified.

Goal cards are read-only monitors, not request cards. Their targets and definitions come from
`agent/goals.json`, synchronized into the server's goal catalog with the existing metric batch. An on-target
reading or a closed work thread never moves a goal to Done. A card shows its measurement, target, source,
notes, recent recorded values and existing linked work. Viewing it does not collect metrics or start work.

Health is separate from work state. Missing, invalid and stale observations cannot show On target.
Most readings expire after 90 minutes; nightly QA expires after 36 hours based on the original QA result's
timestamp, not the time a cached result was copied. Old QA and report scans without trustworthy source
timestamps stay unverified until the existing collector supplies a new reading. Search and Follow metrics
retain their detector-proxy limitations. The authenticated `/api/monitoring` endpoint only reads existing
state; it never creates a database or migrates it. Report history, workflow revisions and counts remain
separate from goal monitors.

Each unresolved component can supply `next_action` with `assignee` (`owner` or `agent`), `task`,
`steps`, `measure` and `done_when`. An owner check also needs `on_failure`: the observable event
that should be reported, without asking the owner to diagnose the cause. Use `record_template`
to offer an editable report draft. Opening that draft never submits it.

For example, an observation can cover the next three ordinary uses, count successful, failed and
inconclusive outcomes, and stop at the first confirmed failure. It should not require extra camera
movement, staged pet behavior or an open-ended watch. Completing that small check does not complete
the underlying request or continuous goal. If a fix or comparison is still missing, the card should
say no owner action is needed and give the agent's next test and acceptance criteria instead.

Retained report photos are available through authenticated `/api/reports/frame?id=…` and shown with
the phone's reported status. The photo and phone state can have different capture times; they do not
prove a synchronized sequence. A missing photo has an explicit unavailable state. Opening evidence
never takes a new camera snapshot.

## Design, after "Exploring Super-agent Design" (DeepSeek Harness, 2026-11-01)

| Idea in the article | Here |
|---|---|
| **Two levels.** A long-lived Router does all the talking and dispatching and never executes. Isolated Workers do one task each. | The Router is one Claude session, resumed across runs and compacted after 40 turns; its only tools are Read, Write and recall. Workers are Claude sessions in their own git worktree, one per thread. |
| **Event-driven Triggers, filtered on a fast path (code) and then a slow path (model).** | Code turns reports and goal misses into deduplicated triggers, and sends reports straight to a report worker. Only triggers that need judgement reach the Router, batched into one call. |
| **Threads, subscription and pass-through.** | Every trigger belongs to a thread. A thread whose worker is running receives its follow-ups directly. |
| **Inline and spill; Chain of Recall.** | Every object has an ID. Long text is cut at 400 characters with a pointer to the rest. `superagent.py recall <id>` follows those references one hop at a time. |
| **Full-duplex; per-channel tone.** | Workers run in the background. Messages to the owners (app chat, push) stay under 100 words; commits and the dashboard carry the full detail. |
| **Compute, state and environment kept apart.** | State lives in `agent.db` on the always-on server. Compute runs on the owner's computer, which may sleep while triggers wait for it. The camera and server are the environment, reached over SSH. |
| **An external Watchdog for liveness and progress.** | `superagent.py watchdog` runs as its own launchd job. It catches stuck workers, a stalled tick, stale goals and triggers that pile up. The server also pages the owners if the agent goes silent for 6 hours. |

## Self-improvement: what the research says, and what we do

The sources are a research pass on 2026-11-02:
- [Darwin Gödel Machine](https://arxiv.org/html/2505.22954v3) ([Sakana blog](https://sakana.ai/dgm/))
- [ADAS](https://arxiv.org/abs/2408.08435)
- [AlphaEvolve](https://deepmind.google/discover/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/)
- [STOP](https://arxiv.org/abs/2310.02304)
- [Voyager](https://arxiv.org/abs/2305.16291)
- [Reflexion](https://arxiv.org/abs/2303.11366)
- [autoresearch](https://github.com/karpathy/autoresearch)
- [METR on reward hacking](https://metr.org/blog/2025-06-05-recent-reward-hacking)
- [Anthropic on evals](https://anthropic.com/engineering/demystifying-evals-for-ai-agents)
- the owner's notes on Lilian Weng's harness engineering, the Tomoro meta-harness, and our own auto-improve and dream loops

What we apply:

| Mechanism | Here |
|---|---|
| **The evaluator stays outside the proposer's reach** (DGM faked its test logs; METR saw models edit scoring code) | Workers may not change the gates, metrics, QA, the agent, goals, steering or policies. They may not remove test assertions, and their changes must stay inside `server/`, `tools/`, `tests/` and `docs/`. All of this is enforced in `tools/report_agent.py:gates`. |
| **Ground truth from outside the system** | Goals in `agent/goals.json` say what they measure and where its ground truth comes from: owner reports, detections after a search, systemd CPU accounting, the AWS credit metric. |
| **An archive of every attempt, fed back to the proposer** | `eval/superagent/archive.jsonl` is append-only and keeps rejected attempts too. Each worker sees the previous attempts on its thread. |
| **Small, single-hypothesis changes; budgets that fail closed** | One thread per worker, at most 2 workers at a time, 10 worker runs and 60 router calls a day. |
| **Measure before and after; revert regressions** | Workers record a baseline and an expected value. The health check rolls back a bad deploy. Twenty-four hours after shipping, the goal is checked again, and if it still misses, the router is told. |
| **The loop cannot tune its own authority** (Weng) | Caps, gates, goals and steering change only when the owner says so in chat. |

### How a change is judged (all built 2026-11-02)
1. **Static gates**: tests, compile and undefined names, the public-repo guard, allowed paths, the protected evaluator, and the test assertion count.
2. **Held-out replay** (`tools/replay.py compare`), against the current code:
   - **Detection:** the verifier on held-out labelled crops, where about a third of the day+hour groups are held out by hash, plus the whole detector and verifier on frames the owner said had no dog (the 2 Nov walk).
   - **Search:** the tracker over a fixed grid of 36 simulated rooms, with realistic detection costs and misses.
   - The candidate may not be worse on any metric beyond a small tolerance. Workers never get `eval/heldout`, and a transcript that touches it is rejected.
   - **Probe:** re-enabling the 2 Nov zero-shot backup and removing the tilt rows was caught (false alarms 0 -> 19.5%, found 0.89 -> 0.28).
3. **Shadow run** for changes to how frames become "your pet":
   - For 6 h and at least 30 new live frames, both versions judge the same live candidate frames on the owner's computer, so the server's CPU budget is untouched.
   - Disagreements go to an independent labeller that sees only the picture and the box.
   - The candidate ships only if it is right at least as often as the current version.
4. **Ship**: the existing deploy, health check and rollback.
5. **Post-ship check**: 24 h later the goal is measured again; a miss goes back to the router.
6. **Share it**: a fix that is live and healthy is offered upstream as a pull request (`tools/contribute.py`): cherry-picked
   onto upstream main, checked again there, scrubbed of anything private (paths, the pet's name; the public guard runs on the
   diff with this install's values) and described by the worker's `public_summary`. On the maintainer's own install it is
   skipped, because the fix is already on main. Open source in the age of agents: every install's agent is a contributor.

### Lessons and the agent's own prompts
- **Lessons:** each night a deterministic pass over the archive writes `agent/lessons.md`. A failure class seen twice becomes a lesson that cites its archive rows, and the router and workers read the file.
- **Prompts:** they live in `agent/prompts/`. Once a week a meta worker may change them, but only when no other worker runs, so a result can be pinned on the change. The change ships only if it scores at least as well on `agent/evals/router_cases.json`: cases with known correct decisions, which the meta worker can't edit.
- **Fixes that wait** because the code moved on are rebased and re-judged on a later tick.

## Operating it
- **Watch:** open **Monitoring** in the app for goal readings and watched behavior. Open `/agent` for linked work,
  recent decisions and the steering. Links between the two keep a selected goal in context.
- **Steer:** tell Claude Code. It edits `agent/steer.md` (read every turn), `agent/goals.json` or `agent/policies.md`, and pushes the change.
- **Pause or resume:** `tools/install_autopilot.sh --remove` / `tools/install_autopilot.sh`.
- **Logs:** `eval/superagent/superagent.log`, `launchd-*.log` and `archive.jsonl`.

## Schedule and measurement (where to look)

Four launchd jobs on the owner's computer (`tools/install_autopilot.sh`):

| Job | Every | Does |
|---|---|---|
| `ai.openpetcam.agent.tick` | 2 min | collects triggers (owner reports, goal misses), routes them, starts at most 2 workers (10 worker runs and 60 router calls a day) |
| `ai.openpetcam.agent.metrics` | 30 min | measures every goal in `agent/goals.json` and writes it to `metrics` in the server's `agent.db` |
| `ai.openpetcam.agent.watchdog` | 15 min | stuck workers, a stalled tick, stale goals, triggers piling up |
| `ai.openpetcam.autopilot` | 03:30 | nightly: detector retraining, the check-in message, QA |

The jobs run only while that computer is awake: 55 measurements in the first 44 hours instead of ~88.

- **Goals over time and every attempt:** `python3 tools/agent_plot.py [hours]` writes `eval/superagent/agent_timeline.png`
  (each goal against its target, misses marked; the attempts and their verdicts below). The app's Agent tab shows
  7-day trends.
- **Every attempt, kept:** `eval/superagent/archive.jsonl` (append-only, verdicts live / queued / rejected / duplicate).

Measured 2026-11-04, first 44 hours: 26 attempts, 7 shipped live (all on 2026-11-03), 17 queued and never retried,
1 rejected; the daily worker budget ran out at 07:05 and 11 owner reports waited up to 21 h (docs/TODO.md item 10).
