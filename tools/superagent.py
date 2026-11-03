#!/usr/bin/env python3
"""The pet cam's super-agent: always on, event driven, self-improving, observable and steerable.

Design (after DeepSeek Harness, "Exploring Super-agent Design", 2026-11-01), mapped to this project:
  state     on the always-on server (server/agentdb.py: triggers, threads, log, goal metrics, heartbeats)
  compute   here, on the owner's computer, through `claude -p`; it may sleep, triggers wait on the server
  collect   fast path, code only: reports, goal misses, QA failures become triggers (deduped)
  route     a long-lived Router session (resumed across runs) decides what needs judgement, batched;
            it never executes. Follow-ups on a thread with a worker go straight to it (pass-through)
  work      isolated Worker sessions, one per thread, in their own git worktree; changes ship only through
            the deterministic gates of tools/report_agent.py (tests, public guard, health check, rollback, cap)
  watchdog  a separate loop: liveness AND progress (stuck workers, stale goals, queue not draining)
  steer     agent/steer.md, agent/goals.json, agent/policies.md: plain files the owner changes through Claude Code

  superagent.py tick        collect + route + start workers (launchd, every 2 min; cheap when nothing new)
  superagent.py metrics     measure every goal (launchd, every 30 min)
  superagent.py watchdog    the control loop (launchd, every 15 min)
  superagent.py work THREAD run one worker (started by tick, in the background)
  superagent.py recall REF  an object by ID (for sessions: Chain of Recall)
"""
import fcntl
import json
import os
import re
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import report_agent as ra  # noqa: E402  (gates, ship, health check, replies, the report worker)
from siteconf import PET, ROOT, SITE, SSH  # noqa: E402

AGENT = os.path.join(ROOT, "agent")
RUN = os.path.join(ROOT, "eval", "superagent")
ARCHIVE = os.path.join(RUN, "archive.jsonl")
ROUTER_TIMEOUT = 300
WORKER_TIMEOUT = 40 * 60
MAX_WORKERS = 2
BUDGET = {"router_calls": 60, "worker_runs": 10}          # per day: a spend cap that fails closed
ENV = dict(os.environ, GIT_TERMINAL_PROMPT="0", CLAUDE_CODE_HOOKS_ENABLED="false")


def log(*a):
    os.makedirs(RUN, exist_ok=True)
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(os.path.join(RUN, "superagent.log"), "a") as f:
        f.write(line + "\n")


def adb(cmd, payload=None, timeout=60):
    r = subprocess.run(SSH + [f"sudo -u petcam python3 /opt/petcam/agentdb.py {cmd}"], input=json.dumps(payload or {}),
                       capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"agentdb {cmd}: {r.stderr[-300:]}")
    return json.loads(r.stdout)


def note(actor, text, thread=None, kind="note"):
    try:
        adb("log", {"actor": actor, "thread": thread, "kind": kind, "text": text})
    except Exception as e:  # noqa: BLE001
        log("log failed:", e)


def beat(name, detail=None):
    try:
        adb("beat", {"name": name, "detail": detail})
    except Exception as e:  # noqa: BLE001
        log("beat failed:", e)


def budget_left(kind):
    p = os.path.join(RUN, "budget.json")
    day = time.strftime("%Y-%m-%d")
    try:
        b = json.load(open(p))
    except (OSError, ValueError):
        b = {}
    if b.get("day") != day:
        b = {"day": day}
    return BUDGET[kind] - b.get(kind, 0), b, p


def spend(kind):
    left, b, p = budget_left(kind)
    b[kind] = b.get(kind, 0) + 1
    json.dump(b, open(p, "w"))


def read(path, limit=6000):
    try:
        return open(os.path.join(AGENT, path)).read()[:limit]
    except OSError:
        return ""


# ------------------------------------------------------------------ collect (fast path: code only)

def collect():
    """Environment events -> triggers. Deterministic and cheap; dedupe keys make it idempotent."""
    n = 0
    for rid in ra.unanswered():                       # reports with no reply yet
        r = adb("add_trigger", {"source": "report", "kind": "report", "thread": f"report:{rid}", "dedupe": f"report:{rid}",
                                "summary": f"owner report {rid}", "payload": {"rid": rid}, "title": f"Report {rid}"})
        n += r["new"]
    day = time.strftime("%Y-%m-%d")
    for g in adb("digest")["goals"]:                  # latest measurement of each goal
        if g["ok"] or g["value"] is None:
            continue
        r = adb("add_trigger", {"source": "metrics", "kind": "goal_miss", "thread": f"goal:{g['goal']}",
                                "dedupe": f"goal:{g['goal']}:{day}", "title": f"Goal: {g['goal']}",
                                "summary": f"goal {g['goal']} missed: value {g['value']} target {g['target']}",
                                "payload": {"goal": g["goal"], "value": g["value"], "target": g["target"], "detail": g["detail"]}})
        n += r["new"]
    return n


# ------------------------------------------------------------------ route

ROUTER_PROMPT = """You are the router of {pet}'s pet-cam super-agent. You run continuously on the owners' behalf.
You never execute tasks or change code. You decide, for each new trigger, one action:
  "ignore"    nothing to do (say why in one line)
  "note"      record an observation on its thread, no work
  "dispatch"  start an isolated worker on the trigger's thread: kind "improve" (raise a goal metric with a measured,
              gated change), "investigate" (find the cause, propose; no shipping), with a brief: goal, evidence,
              what done looks like, how to measure before and after
  "ask"       the owners must decide (money, hardware, more motor moves or sounds near the dog, privacy, deleting data):
              a question of at most 100 words
To see more about any object, run: python3 tools/superagent.py recall <id-or-thread> (Chain of Recall).

North star and goals (agent/goals.json):
{goals}

Owner steering (agent/steer.md, newest first; obey it):
{steer}

Policies (agent/policies.md):
{policies}

Budget left today: {budget}. Workers running now: {running}.

Current state (open threads, recent decisions, latest goal values):
{digest}

New triggers (batched since your last turn):
{triggers}

Write your decisions as JSON to {out}, exactly:
{{"actions": [{{"trigger_ids": [1], "action": "dispatch", "thread": "goal:cpu", "kind": "improve", "brief": "...",
  "ask": "", "why": "one line"}}]}}
Every new trigger id must appear in exactly one action."""


def route(triggers):
    """The slow path: one batched router call for every trigger that needs judgement."""
    if not triggers:
        return []
    left, _, _ = budget_left("router_calls")
    if left <= 0:
        log("router budget spent for today; triggers wait")
        return []
    th = adb("thread", {"id": "router", "title": "Router", "kind": "router"})
    out = os.path.join(RUN, f"router_{int(time.time())}.json")
    running = [t["id"] for t in adb("digest")["open_threads"] if t.get("worker") and t["worker"].startswith("pid:")]
    prompt = ROUTER_PROMPT.format(
        pet=PET, goals=read("goals.json"), steer=read("steer.md"), policies=read("policies.md"),
        budget=json.dumps({k: budget_left(k)[0] for k in BUDGET}), running=running or "none",
        digest=json.dumps(adb("digest"), default=str)[:12000],
        triggers=json.dumps([{k: t[k] for k in ("id", "ts", "source", "kind", "thread", "summary", "payload")} for t in triggers], default=str)[:12000],
        out=out)
    cmd = ["timeout", "-k", "10", str(ROUTER_TIMEOUT), "claude", "-p", prompt, "--output-format", "json",
           "--allowedTools", "Read", "Write", "Bash(python3 tools/superagent.py recall:*)"]
    if th.get("session"):
        cmd += ["--resume", th["session"]]                      # one long-lived router: context continues
    spend("router_calls")
    r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=ROUTER_TIMEOUT + 30, env=ENV)
    try:
        meta = json.loads(r.stdout)
        sid = meta.get("session_id")
        # compaction: a router session that grew long restarts fresh; the digest carries the state
        n = int((th.get("summary") or "0").split(":")[-1] or 0) + 1
        adb("thread", {"id": "router", "session": sid if n < 40 else "", "summary": f"turns:{0 if n >= 40 else n}"})
    except (ValueError, TypeError):
        log("router output unreadable:", r.stdout[-300:], r.stderr[-300:])
        return []
    try:
        acts = json.load(open(out))["actions"]
    except (OSError, ValueError, KeyError) as e:
        log("router wrote no decisions:", e)
        return []
    return acts


def running_workers():
    out = []
    for f in os.listdir(RUN) if os.path.isdir(RUN) else []:
        if f.startswith("worker_") and f.endswith(".pid"):
            pid = int(open(os.path.join(RUN, f)).read() or 0)
            try:
                os.kill(pid, 0)
                out.append((f[7:-4], pid))
            except OSError:
                os.remove(os.path.join(RUN, f))
    return out


def start_worker(thread, kind, brief):
    left, _, _ = budget_left("worker_runs")
    if left <= 0 or len(running_workers()) >= MAX_WORKERS:
        return False
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", thread)
    with open(os.path.join(RUN, f"brief_{safe}.json"), "w") as f:
        json.dump({"thread": thread, "kind": kind, "brief": brief}, f)
    p = subprocess.Popen([sys.executable, os.path.abspath(__file__), "work", thread], cwd=ROOT, env=ENV,
                         stdout=open(os.path.join(RUN, f"worker_{safe}.out"), "a"), stderr=subprocess.STDOUT, start_new_session=True)
    with open(os.path.join(RUN, f"worker_{safe}.pid"), "w") as f:
        f.write(str(p.pid))
    adb("thread", {"id": thread, "worker": f"pid:{p.pid}", "kind": kind})
    spend("worker_runs")
    note("router", f"started a {kind} worker: {brief[:300]}", thread, "dispatch")
    return True


def tick():
    os.makedirs(RUN, exist_ok=True)
    lock = open(os.path.join(RUN, ".tick.lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return
    try:                                                   # the steering the owners see on the dashboard
        north = json.loads(read("goals.json"))["north_star"]
    except (ValueError, KeyError):
        north = ""
    adb("thread", {"id": "steer", "title": north, "kind": "steer", "status": "closed", "summary": read("steer.md", 4000)})
    new = collect()
    pend = adb("pending")
    slow, done = [], []
    for t in pend:
        if t["kind"] == "report":                          # fast path policy: reports go straight to a report worker
            if not t["subscriber"] and start_worker(t["thread"], "report", f"answer owner report {t['payload']}"):
                done.append(t["id"])
            continue
        if t.get("subscriber") and t["subscriber"].startswith("pid:"):
            continue                                       # pass-through: its worker picks it up when it next runs
        slow.append(t)
    if done:
        adb("mark", {"ids": done, "status": "routed", "routed_to": "report-worker"})
    for a in route(slow):
        ids = [i for i in a.get("trigger_ids", []) if isinstance(i, int)]
        act, thread = a.get("action"), a.get("thread")
        if act == "dispatch" and thread:
            ok = start_worker(thread, a.get("kind", "investigate"), a.get("brief", ""))
            adb("mark", {"ids": ids, "status": "routed" if ok else "new", "routed_to": "worker" if ok else None})
            if not ok:
                note("router", f"wanted to dispatch but no slot or budget: {a.get('brief', '')[:200]}", thread)
        elif act == "ask":
            note("router", a.get("ask", ""), thread, "ask")
            ra.notify(None, f"{PET} cam needs you", a.get("ask", "")[:280])
            adb("mark", {"ids": ids, "status": "asked"})
        else:
            note("router", f"{act}: {a.get('why', '')}", thread, act or "note")
            adb("mark", {"ids": ids, "status": "handled"})
    beat("tick", {"new": new, "pending": len(pend), "slow": len(slow), "workers": len(running_workers())})


# ------------------------------------------------------------------ work

WORKER_PROMPT = """You are a worker of {pet}'s pet-cam super-agent, in your own git worktree of the project. You own one
thread: {thread}. Your task ({kind}):
{brief}

Earlier attempts on this thread (the archive; learn from what failed, don't repeat it):
{archive}

Context: the router's view of this thread (recall more with: python3 {tools}/superagent.py recall <id>):
{recall}
Goals: {goals}
Owner steering (obey): {steer}

Rules that the harness enforces: you may not edit the evaluator (tools/qa.py, tools/superagent.py, tools/report_agent.py,
agent/*, server/agentdb.py, deploy and install scripts) or remove test assertions; such changes are rejected. One hypothesis
per change, small diff.
How to work: measure first (the current value and how you measured it, with ground truth from outside the system),
then change, then measure the same way again. Small change, with a test that fails without it. Never deploy, push,
commit or ssh to change anything: the harness decides what ships. Read-only checks on the server are fine through
python3 tools/qa.py or tools/*.py read-only commands. If the right change needs the owners (money, hardware, more motor
moves or sounds near the dog, privacy), propose it instead.

Write {out}, exactly:
{{"summary": "what you found, with numbers", "baseline": "metric before, how measured", "fixed": true|false,
 "fix_summary": "", "tests_run": "", "expected_after": "what the metric should become", "needs_owner": "",
 "reply": "to the owners, at most 100 words, only if something changed for them; else empty"}}"""


def work(thread):
    safe = re.sub(r"[^a-zA-Z0-9_.-]", "_", thread)
    try:
        b = json.load(open(os.path.join(RUN, f"brief_{safe}.json")))
        beat(f"worker:{thread}", {"started": time.time()})
        if b["kind"] == "report":                         # the specialised report worker (labels, reply in the app)
            st = ra.load_state()
            ra.handle(thread.split(":", 1)[1], st)
            ra.save_state(st)
            adb("thread", {"id": thread, "worker": "", "status": "closed", "summary": "answered"})
            return
        wt = os.path.join(ROOT, "eval", "agent-worktrees", safe)
        branch = f"agent/{safe}"
        if os.path.exists(wt):
            subprocess.run(["git", "worktree", "remove", "--force", wt], cwd=ROOT, capture_output=True)
            subprocess.run(["git", "branch", "-D", branch], cwd=ROOT, capture_output=True)
        ra.git("worktree", "add", "-q", "-b", branch, wt, "HEAD")
        out = os.path.join(RUN, f"result_{safe}.json")
        if os.path.exists(out):
            os.remove(out)
        past = [json.loads(x) for x in open(ARCHIVE)] if os.path.exists(ARCHIVE) else []
        mine = [x for x in past if x.get("thread") == thread][-8:]
        prompt = WORKER_PROMPT.format(pet=PET, thread=thread, kind=b["kind"], brief=b["brief"], tools=HERE,
                                      archive=json.dumps(mine, indent=0)[:4000] or "none yet",
                                      recall=json.dumps(adb("recall", {"ref": thread}), default=str)[:8000],
                                      goals=read("goals.json", 3000), steer=read("steer.md", 3000), out=out)
        r = subprocess.run(["timeout", "-k", "10", str(WORKER_TIMEOUT), "claude", "-p", prompt, "--output-format", "json",
                            "--add-dir", RUN, "--add-dir", os.path.join(ROOT, "models"), "--add-dir", os.path.join(ROOT, "eval"),
                            "--allowedTools", "Read", "Grep", "Glob", "Edit", "Write", "Bash(python3:*)", "Bash(git diff:*)",
                            "Bash(git status:*)", "Bash(ls:*)", "Bash(ffmpeg:*)", "Bash(ffprobe:*)"],
                           cwd=wt, capture_output=True, text=True, timeout=WORKER_TIMEOUT + 60, env=ENV)
        if not os.path.exists(out):
            raise RuntimeError(f"worker wrote no result (rc {r.returncode}): {(r.stderr or r.stdout)[-300:]}")
        res = json.load(open(out))
        status, detail, commit = "no_change", "", None
        if ra.git("status", "--porcelain", cwd=wt):
            ok, why = ra.gates(wt)
            if ok and b["kind"] == "improve":
                ra.git("commit", "-q", "-m", f"Agent {thread}: {res.get('fix_summary') or 'improvement'}\n\n{res.get('summary', '')[:1500]}\n"
                       f"Baseline: {res.get('baseline', '')[:300]}\nExpected: {res.get('expected_after', '')[:300]}\n\n"
                       "Made by the super-agent (tools/superagent.py).\n\nCo-Authored-By: Claude <noreply@anthropic.com>", cwd=wt)
                commit = ra.git("rev-parse", "--short", "HEAD", cwd=wt)
                st = ra.load_state()
                status, detail = ra.ship(thread, wt, branch, st)
                ra.save_state(st)
            elif not ok:
                status, detail = "failed_checks", why
            else:
                status = "proposal"              # investigate workers never ship
        with open(ARCHIVE, "a") as f:            # append-only lineage of every attempt, kept or not
            f.write(json.dumps({"ts": time.time(), "thread": thread, "kind": b["kind"], "parent": ra.git("rev-parse", "--short", "HEAD"),
                                "commit": commit, "verdict": status, "detail": detail[:300], "summary": res.get("summary", "")[:500],
                                "baseline": res.get("baseline", "")[:300], "expected_after": res.get("expected_after", "")[:300],
                                "fix_summary": res.get("fix_summary", "")[:300]}) + "\n")
        if status == "live" and thread.startswith("goal:"):
            json.dump({"thread": thread, "shipped": time.time(), "commit": commit},
                      open(os.path.join(RUN, f"watch_{safe}.json"), "w"))
        text = f"[{status}] {res.get('summary', '')[:600]} | baseline: {res.get('baseline', '')[:200]} | " \
               f"expected: {res.get('expected_after', '')[:200]}" + (f" | commit {commit}" if commit else "") + (f" | {detail[:200]}" if detail else "")
        note(f"worker:{b['kind']}", text, thread, "result")
        if res.get("needs_owner"):
            note(f"worker:{b['kind']}", res["needs_owner"], thread, "ask")
        if res.get("reply") or res.get("needs_owner"):
            ra.notify(None, f"{PET} cam", (res.get("reply") or res.get("needs_owner"))[:280])
        adb("thread", {"id": thread, "worker": "", "status": "closed" if status in ("live", "no_change", "proposal") else "open",
                       "summary": f"{status}: {res.get('fix_summary') or res.get('summary', '')[:200]}"})
        if status not in ("queued",):
            subprocess.run(["git", "worktree", "remove", "--force", wt], cwd=ROOT, capture_output=True)
            if status in ("no_change", "proposal", "rolled_back"):
                subprocess.run(["git", "branch", "-D", branch], cwd=ROOT, capture_output=True)
    except Exception as e:  # noqa: BLE001
        log("worker", thread, "failed:", type(e).__name__, str(e)[:300])
        note("worker", f"failed: {type(e).__name__}: {str(e)[:300]}", thread, "error")
        adb("thread", {"id": thread, "worker": "", "summary": f"error: {str(e)[:200]}"})
    finally:
        try:
            os.remove(os.path.join(RUN, f"worker_{safe}.pid"))
        except OSError:
            pass
        beat(f"worker:{thread}", {"finished": time.time()})


# ------------------------------------------------------------------ metrics (goals, measured)

METRICS_REMOTE = r'''
import json, os, sqlite3, time, subprocess
S = "/var/lib/petcam"; now = time.time(); out = {}
units = "petcam-watch mediamtx petcam-app caddy".split()
cur = {u: int(subprocess.run(["systemctl", "show", "-p", "CPUUsageNSec", "--value", u], capture_output=True, text=True).stdout.strip() or 0) for u in units}
p = f"{S}/agent_cpu.json"
try:
    prev = json.load(open(p))
except Exception:
    prev = None
json.dump({"t": now, "cpu": cur}, open(p, "w"))
if prev and now - prev["t"] > 300:
    out["cpu"] = round(sum(cur[u] - prev["cpu"].get(u, cur[u]) for u in units) / 1e9 / (now - prev["t"]) * 100, 1)
out["live_age"] = round(now - json.load(open(f"{S}/live.json"))["t"], 1)
c = sqlite3.connect(f"{S}/petcam.db")
ev = c.execute("select ts, detail from events where kind='search' and ts > ?", (now - 86400,)).fetchall()
home = found = 0
for ts, d in ev:
    seen_after = c.execute("select count(*) from obs where ts between ? and ? and score > 0", (ts, ts + 600)).fetchone()[0]
    if "found her" in (d or "") or seen_after:
        home += 1
        found += "found her" in (d or "")
out["search"] = [found, home]
reps = sorted(d for d in os.listdir(f"{S}/reports") if d[:8].isdigit())
waiting = [now - os.path.getmtime(f"{S}/reports/{d}/report.json") for d in reps if not os.path.exists(f"{S}/reports/{d}/reply.json")]
out["oldest_unanswered_min"] = round(max(waiting) / 60, 1) if waiting else 0
week = [d for d in reps if os.path.getmtime(f"{S}/reports/{d}/report.json") > now - 7 * 86400]
owner = [d for d in week if "probe" not in open(f"{S}/reports/{d}/report.json").read(400).lower()]
out["owner_reports_per_day"] = round(len(owner) / 7, 2)
print(json.dumps(out))
'''


def metrics():
    r = subprocess.run(SSH + ["sudo python3 -"], input=METRICS_REMOTE, capture_output=True, text=True, timeout=120)
    m = json.loads(r.stdout) if r.returncode == 0 else {}
    rows = []

    def add(goal, value, target, ok, detail=None):
        rows.append({"goal": goal, "value": value, "target": target, "ok": ok, "detail": detail})
    if "cpu" in m:
        add("cpu", m["cpu"], "<= 8", m["cpu"] <= 8, "% of one core, camera services, since last measure")
    add("app_status_fresh", m.get("live_age"), "<= 10", (m.get("live_age") or 999) <= 10, "seconds since live.json")
    f, h = m.get("search", [0, 0])
    if h:
        add("search_success", round(f / h, 2), ">= 0.8", f / h >= 0.8, f"{f} found of {h} searches when she was home (24 h)")
    add("reports_answered", m.get("oldest_unanswered_min", 0), "<= 15", m.get("oldest_unanswered_min", 0) <= 15, "minutes")
    add("owner_reports", m.get("owner_reports_per_day", 0), "<= 1", m.get("owner_reports_per_day", 0) <= 1, "per day, 7-day average")
    try:
        ins, reg = SITE.get("LIGHTSAIL_INSTANCE"), SITE.get("LIGHTSAIL_REGION", "us-east-1")
        if ins:
            q = subprocess.run(["timeout", "-k", "2", "40", "aws", "lightsail", "get-instance-metric-data", "--region", reg,
                                "--instance-name", ins, "--metric-name", "BurstCapacityPercentage", "--period", "300",
                                "--start-time", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time() - 1800)),
                                "--end-time", time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "--unit", "Percent",
                                "--statistics", "Average", "--output", "json"], capture_output=True, text=True, env=ENV)
            pts = sorted(json.loads(q.stdout)["metricData"], key=lambda x: x["timestamp"])
            if pts:
                v = round(pts[-1]["average"], 1)
                add("burst_credits", v, ">= 20", v >= 20, "% (it refills while CPU stays under the baseline)")
    except Exception as e:  # noqa: BLE001
        log("burst metric:", e)
    qa = os.path.join(RUN, "qa_last.json")
    if time.localtime().tm_hour == 3 or not os.path.exists(qa) or time.time() - os.path.getmtime(qa) > 26 * 3600:
        q = subprocess.run([sys.executable, os.path.join(HERE, "qa.py"), "--json"], capture_output=True, text=True, timeout=900, env=ENV)
        try:
            json.loads(q.stdout)
            open(qa, "w").write(q.stdout)
        except ValueError:
            pass
    try:
        res = json.load(open(qa))
        fails = [x["name"] for x in res if x["status"] == "FAIL"]
        add("qa", len(fails), "== 0", not fails, ", ".join(fails)[:300] or "all pass")
    except (OSError, ValueError):
        pass
    adb("metric", {"metrics": rows})
    by = {r["goal"]: r for r in rows}
    for f in [x for x in os.listdir(RUN) if x.startswith("watch_")]:
        w = json.load(open(os.path.join(RUN, f)))
        goal = w["thread"].split(":", 1)[1]
        if time.time() - w["shipped"] > 86400:
            r = by.get(goal)
            verdict = "kept: goal on target" if r and r["ok"] else "did not reach the goal"
            note("metrics", f"post-ship check of {w.get('commit')}: {verdict} (now {r['value'] if r else '?'})", w["thread"], "result")
            if r and not r["ok"]:
                adb("add_trigger", {"source": "metrics", "kind": "post_ship_miss", "thread": w["thread"], "dedupe": f"postship:{w.get('commit')}",
                                    "summary": f"shipped {w.get('commit')} 24 h ago, goal {goal} still misses ({r['value']} vs {r['target']})"})
            os.remove(os.path.join(RUN, f))
    beat("metrics", {"n": len(rows)})
    return rows


# ------------------------------------------------------------------ watchdog (independent control loop)

def watchdog():
    """Liveness and progress, outside the main loop: stuck workers, a tick that stopped, goals not measured,
    triggers that never drain, the watcher on the server. Fixes what it can; pages only for what needs a person."""
    d = adb("dashboard")
    now = time.time()
    beats = {b["name"]: b for b in d["beats"]}
    problems = []
    for name, limit in (("tick", 15 * 60), ("metrics", 75 * 60)):
        b = beats.get(name)
        if not b or now - b["ts"] > limit:
            problems.append(f"{name} has not run for {round((now - (b['ts'] if b else 0)) / 60)} min")
    for thread, pid in running_workers():
        started = os.path.getmtime(os.path.join(RUN, f"worker_{thread}.pid"))
        if now - started > WORKER_TIMEOUT + 600:
            os.killpg(pid, 15)
            note("watchdog", f"stopped a worker stuck for {round((now - started) / 60)} min; its thread stays open", thread, "fix")
    if d["queue"] and beats.get("tick") and now - beats["tick"]["ts"] < 600:
        old = [t for t in adb("pending") if now - t["ts"] > 3600]
        if old:
            problems.append(f"{len(old)} triggers waiting over an hour (budget or worker slots?)")
    for t in d["threads"]:                     # a worker marked as running whose process is gone
        if (t.get("worker") or "").startswith("pid:"):
            try:
                os.kill(int(t["worker"][4:]), 0)
            except (OSError, ValueError):
                adb("thread", {"id": t["id"], "worker": ""})
                note("watchdog", "cleared a worker that had exited without reporting", t["id"], "fix")
    state = os.path.join(RUN, "watchdog.json")
    try:
        prev = json.load(open(state))
    except (OSError, ValueError):
        prev = {}
    key = "|".join(sorted(problems))
    if problems and key != prev.get("key"):
        note("watchdog", "; ".join(problems), None, "alert")
        ra.notify(None, f"{PET} cam agent", "; ".join(problems)[:280])
    json.dump({"key": key, "t": now}, open(state, "w"))
    beat("watchdog", {"problems": problems})
    return problems


def recall(ref):
    print(json.dumps(adb("recall", {"ref": ref}), indent=1, default=str))


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "tick":
        tick()
    elif cmd == "metrics":
        print(json.dumps(metrics(), indent=1))
    elif cmd == "watchdog":
        print(watchdog())
    elif cmd == "work":
        work(sys.argv[2])
    elif cmd == "recall":
        recall(sys.argv[2])
    else:
        print(__doc__)
        sys.exit(2)
