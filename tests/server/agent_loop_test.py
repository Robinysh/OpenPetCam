"""The self-improvement loop cannot stall on its own housekeeping (2026-11-03/04: 26 attempts, 7 shipped, 17 queued and
never retried; 11 owner reports waited up to 21 h).
  1. agent/lessons.md, rewritten nightly by the agent, is git-ignored: tracked, it kept the checkout "being edited"
     forever and retry_queued() / ship() refused to act.
  2. A queued fix that no longer applies to the latest code is dropped and a redo is filed (it used to be retried, and
     fail the same way, every tick), and the report it answers is told.
  3. Owner reports have their own worker budget, separate from goal and meta work.
  4. The fix_queue goal measures the oldest queued fix by when it was queued.
  python3 tests/server/agent_loop_test.py
"""
import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
os.environ.setdefault("PETCAM_CONFIG", os.path.join(ROOT, "config", "site.env.example"))
sys.path.insert(0, os.path.join(ROOT, "tools"))
import superagent as sa  # noqa: E402

# 1. the nightly lessons file never makes the checkout dirty
r = subprocess.run(["git", "check-ignore", "-q", "agent/lessons.md"], cwd=ROOT)
assert r.returncode == 0, "agent/lessons.md must be git-ignored"
tracked = subprocess.run(["git", "ls-files", "agent/lessons.md"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
assert not tracked, "agent/lessons.md must not be tracked"

# 2. a fix that no longer applies: dropped, redo filed, report told
run = tempfile.mkdtemp()
sa.RUN = run
calls, replies = [], []
sa.adb = lambda op, a=None: calls.append((op, a)) or {}
sa.note = lambda *a, **k: None
sa.running_workers = lambda: []
sa.replay_compare = lambda wt: {"ok": True}
sa.ra.git = lambda *a, **k: ""
sa.ra.load_state = lambda: {"deploys": {}}
sa.ra.save_state = lambda st: None
sa.ra.ship = lambda thread, wt, branch, st: ("queued", "the fix no longer applies to the latest code; it will be redone")
sa.ra.write_reply = lambda rid, status, text="", fix=None: replies.append((rid, status, text))
sa.subprocess.run = lambda *a, **k: subprocess.CompletedProcess(a, 0, "", "")
wt = tempfile.mkdtemp()
q = {"thread": "report:20261104-124359", "wt": wt, "branch": "fix/x", "report": "20261104-124359", "commit": "abc1234",
     "queued_at": time.time() - 30 * 3600}
open(os.path.join(run, "queued_report_20261104-124359.json"), "w").write(json.dumps(q))
sa.ARCHIVE = os.path.join(run, "archive.jsonl")
sa.retry_queued()
assert not any(f.startswith("queued_") for f in os.listdir(run)), "the stale fix stays queued"
trig = [a for op, a in calls if op == "add_trigger"]
assert trig and trig[0]["kind"] == "report" and trig[0]["dedupe"].startswith("redo:report:20261104-124359"), trig
assert trig[0]["payload"] == {"rid": "20261104-124359"}
assert replies and replies[0][0] == "20261104-124359" and "written again" in replies[0][2], replies

# 3. separate budgets: a report worker spends report_runs, a goal worker worker_runs
spent = []
sa.spend = lambda kind: spent.append(kind)
sa.budget_left = lambda kind: (1, {}, "")
sa.subprocess.Popen = lambda *a, **k: type("P", (), {"pid": 4242})()
os.makedirs(run, exist_ok=True)
assert sa.start_worker("report:r1", "report", "answer owner report")
assert sa.start_worker("goal:qa", "improve", "fix qa")
assert spent == ["report_runs", "worker_runs"], spent
assert sa.BUDGET["report_runs"] >= 10, sa.BUDGET

# 4. fix_queue: by when it was queued
open(os.path.join(run, "queued_goal_x.json"), "w").write(json.dumps({"thread": "goal:x", "queued_at": time.time() - 30 * 3600}))
sa.subprocess.run = lambda *a, **k: subprocess.CompletedProcess(a, 1, "", "")   # metrics(): no server here
rows = sa.metrics()
fq = [r for r in rows if r["goal"] == "fix_queue"]
assert fq and 29.5 <= fq[0]["value"] <= 30.5 and fq[0]["ok"] is False, fq
print("agent loop ok: lessons ignored, stale fix redone, separate report budget, fix_queue measured")
