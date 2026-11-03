# The super-agent: how OpenPetCam improves itself

The owners asked for a system that runs and improves itself without them invoking it, stays observable (a
dashboard) and stays steerable (by talking to Claude Code). This is that system. Code: `tools/superagent.py`
(brain), `server/agentdb.py` (state), `server/agent.html` (dashboard at `/agent`), `agent/` (goals, steering, policies).

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

### Not built yet (in order)
1. **Offline replay on held-out data**, with clips and owner labels split by day and a set the worker cannot read, plus an A/A comparison to measure the noise band before calling a win.
2. **A shadow A/B run** for detector and search changes: run the candidate beside the current version on the same live input.
3. **A nightly lessons miner**: a failure class seen twice in the archive becomes an entry in `agent/lessons.md`, citing its archive rows.
4. **Router and worker prompts as improvable files**, one change in flight at a time, judged against past triggers with known correct decisions.

## Operating it
- **Watch:** open `/agent` in the app. It shows goals with 7-day trends, what the agent is working on, its recent decisions and the steering.
- **Steer:** tell Claude Code. It edits `agent/steer.md` (read every turn), `agent/goals.json` or `agent/policies.md`, and pushes the change.
- **Pause or resume:** `tools/install_autopilot.sh --remove` / `tools/install_autopilot.sh`.
- **Logs:** `eval/superagent/superagent.log`, `launchd-*.log` and `archive.jsonl`.
