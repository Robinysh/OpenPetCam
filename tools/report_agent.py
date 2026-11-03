#!/usr/bin/env python3
"""The owner's agent for "Something wrong?" reports: every report is feedback, handled within minutes.

  report_agent.py run           handle every report that has no reply yet (what launchd runs every 5 min)
  report_agent.py run ID        handle one report again
  report_agent.py --dry-run run everything except deploying, writing replies and notifying

Per report:
  1. mark it "looking" in the app, pull the case file (tools/triage.py)
  2. crop what the detector saw at that moment, so the owner's words can become labels
  3. Claude Code, as the owner's agent, reads the case and the code in a separate git worktree. It
     labels the crops from what the owner said, diagnoses, and fixes the code with a test when the
     cause is clear. It writes agent_result.json and never deploys anything itself.
  4. this harness, not the model, decides what ships:
       - labels go into the verifier's training labels (source "owner", outranks the auto labeller)
       - a code change ships only if: the tests, the public-repo guard and a name check pass; the main
         checkout is clean (nobody is editing it); fewer than MAX_DEPLOYS_PER_DAY today. Then fast-forward,
         ./deploy.sh, check the server is healthy for 60 s, and revert + redeploy if it is not
  5. reply in the app (reply.json) and on the reporter's phone

Unattended rules: one run at a time (lock), a hard deadline on every step, no interactive prompts.
Heartbeat: /var/lib/petcam/report_agent_ok on the server each run; alert.py pages when a report waits 6 h.
"""
import fcntl
import glob
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from siteconf import PET, ROOT, SSH  # noqa: E402

DRY = "--dry-run" in sys.argv
REMOTE = "/var/lib/petcam/reports"
OUT = os.path.join(ROOT, "eval", "reports")
WT_DIR = os.path.join(ROOT, "eval", "agent-worktrees")
STATE = os.path.join(ROOT, "eval", "report_agent_state.json")
LOG = os.path.join(ROOT, "eval", "report_agent.log")
BOOT = os.path.join(ROOT, "eval", "bootstrap")
AGENT_TIMEOUT = 25 * 60
MAX_DEPLOYS_PER_DAY = 3
ENV = dict(os.environ, GIT_TERMINAL_PROMPT="0", CLAUDE_CODE_HOOKS_ENABLED="false")


def log(*a):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
    print(line, flush=True)
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def run(cmd, timeout=120, cwd=None, check=False, **kw):
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd, env=ENV, **kw)
    if check and r.returncode != 0:
        raise RuntimeError(f"{cmd if isinstance(cmd, str) else ' '.join(cmd)[:120]} -> {r.returncode}: {(r.stderr or r.stdout)[-400:]}")
    return r


def ssh(cmd, timeout=120):
    return run(SSH + [cmd], timeout=timeout)


def git(*a, cwd=ROOT, check=True, timeout=60):
    return run(["git", *a], cwd=cwd, check=check, timeout=timeout).stdout.strip()


def load_state():
    try:
        with open(STATE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"deploys": {}, "pending": {}}


def save_state(st):
    with open(STATE + ".tmp", "w") as f:
        json.dump(st, f, indent=1)
    os.replace(STATE + ".tmp", STATE)


# ---------------------------------------------------------------- server side

def unanswered():
    """Report ids with no reply.json, oldest first; touches the heartbeat in the same call."""
    out = ssh(f"sudo touch /var/lib/petcam/report_agent_ok; cd {REMOTE} 2>/dev/null && "
              "for d in 2*; do [ -f $d/report.json ] && [ ! -f $d/reply.json ] && echo $d; done; true")
    if out.returncode != 0:
        raise RuntimeError(f"listing reports failed: {out.stderr[-300:]}")
    return sorted(x for x in out.stdout.split() if x[:8].isdigit())


def write_reply(rid, status, text="", fix=None):
    body = {"status": status, "text": text, "ts": time.time(), "fix": fix}
    if DRY:
        log("dry-run reply", rid, json.dumps(body)[:300])
        return
    q = json.dumps(body).replace("'", "'\\''")
    r = ssh(f"echo '{q}' | sudo tee {REMOTE}/{rid}/reply.json >/dev/null && sudo chmod 644 {REMOTE}/{rid}/reply.json")
    if r.returncode != 0:
        raise RuntimeError(f"writing reply failed: {r.stderr[-200:]}")


def notify(device, title, body):
    if DRY or not device:
        return
    q = lambda s: "'" + s.replace("'", "'\\''") + "'"   # noqa: E731
    ssh(f"sudo -u petcam /opt/petcam/venv/bin/python /opt/petcam/alert.py --notify {q(title)} {q(body[:300])} report {q(device)}")


# ---------------------------------------------------------------- case file + crops

def pull(rid):
    run([sys.executable, os.path.join(HERE, "triage.py"), "pull", rid], timeout=600, check=True)
    return os.path.join(OUT, rid)


def crops(d):
    """Detector proposals on the report frame and on clip frames every 10 s: what the owner's words
    can label. Saved as crop_N.jpg with crops.json (frame, box, detector label/score)."""
    sys.path.insert(0, os.path.join(ROOT, "server"))
    from detector import Detector
    from PIL import Image
    det = Detector(os.path.join(ROOT, "models", "yolox_nano.onnx"))
    frames = [("frame", os.path.join(d, "frame.jpg"))]
    clip = os.path.join(d, "clip.mp4")
    if os.path.exists(clip):
        fd = os.path.join(d, "clipframes")
        os.makedirs(fd, exist_ok=True)
        run(["ffmpeg", "-v", "error", "-y", "-i", clip, "-vf", "fps=0.1,scale=640:360", os.path.join(fd, "%02d.jpg")], timeout=180)
        frames += [(f"clip+{(int(os.path.basename(p)[:2]) - 1) * 10}s", p) for p in sorted(glob.glob(os.path.join(fd, "[0-9][0-9].jpg")))]
    out = []
    for name, path in frames:
        if not os.path.exists(path):
            continue
        im = Image.open(path).convert("RGB")
        for f in det.detect(im, min_score=0.10):
            if f["label"] == "person":
                continue
            x0, y0, x1, y1 = f["box"]
            W, H = im.size
            mx, my = (x1 - x0) * 0.15, (y1 - y0) * 0.15
            c = im.crop((int(max(0, (x0 - mx) * W)), int(max(0, (y0 - my) * H)), int(min(W, (x1 + mx) * W)), int(min(H, (y1 + my) * H))))
            if c.width < 12 or c.height < 12:
                continue
            n = len(out)
            c.save(os.path.join(d, f"crop_{n}.jpg"), quality=92)
            out.append({"n": n, "frame": name, "frame_path": path, "box": [round(v, 4) for v in f["box"]],
                        "yolo_label": f["label"], "yolo_score": round(f["score"], 3)})
    with open(os.path.join(d, "crops.json"), "w") as f:
        json.dump(out, f, indent=1)
    return out


# ---------------------------------------------------------------- the agent

PROMPT = """You are the owner's agent for their pet camera (OpenPetCam: a pan/tilt camera watching their dog {pet}).
They filed a report from the app. Treat it as feedback from the person you work for: find out what went wrong,
teach the system from what they said, and fix the cause if you can do that safely. You have the code in this
directory (a git worktree; README.md and docs/ explain the system) and the case file in {d}:
  summary.md     start here: what the app showed, the watcher's state, presence and events around the report
  report.json    the same, raw;  journal.txt  server logs;  frame.jpg  the picture at that moment
  clip.mp4       the 90 s before (if present);  crop_N.jpg + crops.json  every detector proposal near that moment
Look at the pictures yourself. Read code before blaming it. Earlier reports and their outcomes are in {reports}
(each has summary.md and, once handled, agent_result.json) and docs/evaluation.md; check whether this is a repeat.

The owner wrote: "{note}"
The app showed: "{shown}"

Do, in order:
1. Labels. For each crop_N.jpg decide what it is, using the owner's words as ground truth where they apply
   ("this is a toy" -> the crop the app treated as the pet is "toy"; "you are looking at her" -> the crop of her is
   "pet"). Labels: pet, toy, other_animal, person, nothing. Only label crops you are sure about.
2. Diagnosis: what happened and why, in a few sentences, citing evidence (times, scores, log lines, code lines).
3. Fix, only if the cause is clear and the change is small: edit the code here, and add or extend a test that fails
   without the fix (tests/server/tracker_test.py is a simulator; there may be others). Run the tests with python3.
   Never touch config/, secrets, models/, or anything that sends data elsewhere. Never deploy, ssh, push or commit:
   the harness decides what ships. If the right fix needs the owner's decision or is large, describe it instead.
   Things that stress the dog (sounds, frequent motor moves) need the owner's say: propose, don't ship.
4. Write {d}/agent_result.json, exactly this shape:
   {{"labels": [{{"n": 0, "label": "toy"}}], "diagnosis": "...", "fixed": true|false,
    "fix_summary": "one sentence or empty", "tests_run": "command and result or empty",
    "reply": "to the owner, at most 280 characters, plain words, no jargon: what happened and what you did or propose",
    "needs_owner": "a decision you need from them, or empty"}}
"""


def run_agent(rid, d, wt, rep):
    note = (rep.get("note") or "").replace('"', "'")
    prompt = PROMPT.format(pet=PET, d=d, reports=OUT, note=note,
                           shown=((rep.get("client") or {}).get("shown_status") or "").replace('"', "'"))
    cmd = ["timeout", "-k", "10", str(AGENT_TIMEOUT), "claude", "-p", prompt, "--add-dir", d, "--add-dir", OUT, "--add-dir", os.path.join(ROOT, "models"),
           "--allowedTools", "Read", "Grep", "Glob", "Edit", "Write", "Bash(python3:*)", "Bash(ffmpeg:*)", "Bash(ffprobe:*)",
           "Bash(git diff:*)", "Bash(git status:*)", "Bash(ls:*)", "--output-format", "json"]
    r = run(cmd, cwd=wt, timeout=AGENT_TIMEOUT + 60)
    with open(os.path.join(d, "agent_transcript.json"), "w") as f:
        f.write(r.stdout or "")
    res_path = os.path.join(d, "agent_result.json")
    if r.returncode != 0 or not os.path.exists(res_path):
        raise RuntimeError(f"agent failed (rc {r.returncode}): {(r.stderr or r.stdout)[-300:]}")
    with open(res_path) as f:
        return json.load(f)


# ---------------------------------------------------------------- labels -> training data

def apply_labels(rid, d, res):
    """Owner-grounded labels join the verifier's training data (eval/bootstrap manifest + labels)."""
    from PIL import Image
    sys.path.insert(0, HERE)
    import bootstrap_pet as bp
    cs = {c["n"]: c for c in json.load(open(os.path.join(d, "crops.json")))}
    man, lab = [], []
    for l in res.get("labels") or []:
        c = cs.get(l.get("n"))
        if not c or l.get("label") not in bp.LABELS:
            continue
        cid = f"r{rid.replace('-', '')}{c['n']:02d}"
        crop = Image.open(os.path.join(d, f"crop_{c['n']}.jpg"))
        os.makedirs(os.path.join(BOOT, "crops"), exist_ok=True)
        crop.save(os.path.join(BOOT, "crops", f"{cid}.jpg"), quality=92)
        day = f"{rid[:4]}-{rid[4:6]}-{rid[6:8]}"
        man.append({"id": cid, "frame": f"report/{rid}/{c['frame']}", "day": day, "hour": rid[9:11], "box": c["box"],
                    "yolo_label": c["yolo_label"], "yolo_score": c["yolo_score"], "hash": bp.dhash(crop)})
        lab.append({"id": cid, "label": l["label"], "source": "owner", "report": rid, "ts": time.time()})
    if man and not DRY:
        known = {m["id"] for m in bp.load_jsonl(os.path.join(BOOT, "manifest.jsonl"))}
        bp.append_jsonl(os.path.join(BOOT, "manifest.jsonl"), [m for m in man if m["id"] not in known])
        bp.append_jsonl(os.path.join(BOOT, "labels.jsonl"), lab)
    return len(lab)


# ---------------------------------------------------------------- fix -> gated deploy

ALLOWED = ("server/", "tools/", "tests/", "docs/", "README.md")
# The evaluator stays out of the proposer's reach (DGM faked its test logs; METR saw models edit scoring code): the gates,
# goal metrics, QA, the agent itself, goals, steering and policies change only through the owner.
NEVER = ("tools/report_agent.py", "tools/superagent.py", "tools/qa.py", "tools/check_public.sh", "tools/noise_study.py",
         "deploy.sh", "server/install.sh", "server/set-secrets.sh", "server/agentdb.py")
NEVER_PREFIX = ("agent/",)


def assert_count(root):
    n = 0
    for dirpath, _, files in os.walk(os.path.join(root, "tests")):
        for f in files:
            if f.endswith(".py"):
                n += open(os.path.join(dirpath, f), errors="replace").read().count("assert ")
    return n


def gates(wt):
    """Every check a change must pass before it may ship. Returns (ok, text)."""
    git("add", "-A", cwd=wt)
    changed = git("diff", "--cached", "--name-only", cwd=wt).split()
    kinds = git("diff", "--cached", "--summary", cwd=wt)
    bad = [f for f in changed if not f.startswith(ALLOWED) or f in NEVER or f.startswith(NEVER_PREFIX)]
    if bad or "120000" in kinds:                                # 120000 = a symlink
        return False, f"the change touches files the agent may not ship: {bad or 'a symlink'}"
    if assert_count(wt) < assert_count(ROOT):
        return False, f"the change removes test assertions ({assert_count(ROOT)} -> {assert_count(wt)})"
    # Only tests that run on this computer: the others import the live install (/opt/petcam) and run in tools/qa.py.
    checks = [[sys.executable, "tests/server/tracker_test.py"], [sys.executable, "tests/server/events_test.py"],
              [sys.executable, "tests/server/presence_test.py"], [sys.executable, "tests/server/stall_test.py"],
              ["bash", "-c", f"{sys.executable} -m py_compile server/*.py tools/*.py"],
              ["bash", "-c", f"{sys.executable} -m pyflakes server tools 2>/dev/null | grep -E 'undefined name|unable to detect' && exit 1 || exit 0"],
              ["bash", "-c", "git add -A && tools/check_public.sh"]]
    for c in checks:
        r = run(c, cwd=wt, timeout=600)
        if r.returncode != 0:
            return False, f"{' '.join(c)[:80]} failed: {(r.stdout + r.stderr)[-400:]}"
    return True, "tests, compile, names and public-repo guard passed"


def healthy(t_deploy):
    """Services up, the watcher heartbeating, and no traceback since the deploy, for 60 s."""
    time.sleep(60)
    r = ssh("systemctl is-active petcam-watch petcam-app mediamtx caddy | sort -u; "
            "echo hb $(( $(date +%s) - $(stat -c %Y /var/lib/petcam/heartbeat) )); "
            f"journalctl -u petcam-watch -u petcam-app --since @{int(t_deploy)} --no-pager | grep -c Traceback; true")   # grep -c exits 1 on 0
    try:
        v = json.load(open(os.path.join(ROOT, "models", "verifier.json")))
        want = f"verifier: {v['mode']} threshold {v['threshold']}"
    except (OSError, ValueError, KeyError):
        return False, "models/verifier.json missing or unreadable"
    vl = ssh(f"journalctl -u petcam-watch --since @{int(t_deploy)} --no-pager | grep 'verifier:' | tail -1").stdout
    if want not in vl or v["mode"] != "logreg":
        return False, f"watcher loaded '{vl.strip()[-60:]}', expected the trained '{want}'"
    lines = r.stdout.split()
    ok = len(lines) >= 4 and lines[:1] == ["active"] and "hb" in lines and int(lines[lines.index("hb") + 1]) < 30 \
        and lines[-1] == "0"
    return ok, r.stdout.strip().replace("\n", " ")


def ship(rid, wt, branch, st):
    today = time.strftime("%Y-%m-%d")
    if st["deploys"].get(today, 0) >= MAX_DEPLOYS_PER_DAY:
        return "queued", f"{MAX_DEPLOYS_PER_DAY} fixes already went live today; this one waits for tomorrow"
    if git("status", "--porcelain"):
        return "queued", "the code is being edited on the computer right now; it ships once that is committed"
    head = git("rev-parse", "HEAD")
    if git("merge-base", head, branch) != head:
        return "queued", "the code moved on since the fix was written; it will be redone"
    if DRY:
        return "dry", "would deploy"
    git("merge", "--ff-only", branch)
    t = time.time()
    r = run(["./deploy.sh"], cwd=ROOT, timeout=1200)
    ok, why = (r.returncode == 0, r.stdout[-300:] + r.stderr[-300:])
    if ok:
        ok, why = healthy(t)
    if ok:
        st["deploys"][today] = st["deploys"].get(today, 0) + 1
        return "live", why
    log("deploy unhealthy, rolling back:", why)
    git("revert", "--no-edit", "HEAD")
    run(["./deploy.sh"], cwd=ROOT, timeout=1200)
    return "rolled_back", why


def handle(rid, st):
    log("report", rid)
    write_reply(rid, "looking")
    d = pull(rid)
    rep = json.load(open(os.path.join(d, "report.json")))
    crops(d)
    branch = f"agent/report-{rid}"
    wt = os.path.join(WT_DIR, rid)
    if os.path.exists(wt):
        run(["git", "worktree", "remove", "--force", wt], cwd=ROOT)
        run(["git", "branch", "-D", branch], cwd=ROOT)
    os.makedirs(WT_DIR, exist_ok=True)
    git("worktree", "add", "-q", "-b", branch, wt, "HEAD")
    # No links into the worktree: on 2026-11-02 a 'models' link got committed with a fix and, merged, replaced the real
    # models folder (the deploy then shipped the untrained default verifier). The agent reads models via --add-dir.
    res = run_agent(rid, d, wt, rep)
    n_labels = apply_labels(rid, d, res)
    status, detail, commit = "no_change", "", None
    if git("status", "--porcelain", cwd=wt):
        ok, why = gates(wt)
        if ok:
            git("commit", "-q", "-m", f"Report {rid}: {res.get('fix_summary') or 'fix'}\n\n{res.get('diagnosis', '')[:1500]}\n\n"
                "Made by the report agent (tools/report_agent.py).\n\nCo-Authored-By: Claude <noreply@anthropic.com>", cwd=wt)
            commit = git("rev-parse", "--short", "HEAD", cwd=wt)
            status, detail = ship(rid, wt, branch, st)
            if status == "queued":
                st["pending"][rid] = branch
        else:
            status, detail = "failed_checks", why
            # keep the draft: commit it on its branch so a person (or the next run) can pick it up
            git("add", "-A", cwd=wt)
            git("commit", "-q", "--no-verify", "-m", f"UNSHIPPED report {rid}: {res.get('fix_summary') or 'fix'} (failed checks)\n\n{why[:1500]}", cwd=wt)
            commit = git("rev-parse", "--short", "HEAD", cwd=wt)
    reply = (res.get("reply") or "Looked at it.").strip()
    tail = {"live": " The fix is live now.", "queued": " A fix is ready and goes live shortly.",
            "rolled_back": " A fix was tried but undone because the server didn't look healthy; noted for review.",
            "failed_checks": " A fix was drafted but failed its checks, so nothing changed yet.",
            "dry": "", "no_change": ""}[status]
    if res.get("needs_owner"):
        tail += f" Your call: {res['needs_owner']}"
    write_reply(rid, "answered", (reply + tail)[:600],
                {"status": status, "commit": commit, "labels": n_labels, "detail": detail[:300]})
    notify(rep.get("device"), "About your report", reply + tail)
    log(rid, "->", status, f"labels={n_labels}", detail[:200])
    if status not in ("queued",):
        run(["git", "worktree", "remove", "--force", wt], cwd=ROOT)
        if status in ("no_change", "dry", "rolled_back"):      # failed_checks keeps its branch for review
            run(["git", "branch", "-D", branch], cwd=ROOT)


def main():
    os.makedirs(os.path.join(ROOT, "eval"), exist_ok=True)
    lock = open(os.path.join(ROOT, "eval", ".report_agent.lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another run is in progress")
        return
    st = load_state()
    args = [a for a in sys.argv[2:] if not a.startswith("--")]
    ids = args or unanswered()
    if not ids:
        return
    for rid in ids:
        try:
            handle(rid, st)
        except Exception as e:  # noqa: BLE001 - one bad report must not block the others
            log(rid, "failed:", type(e).__name__, str(e)[:400])
            try:
                write_reply(rid, "answered", "I couldn't finish looking at this one automatically; it's flagged for a closer look.",
                            {"status": "error", "detail": str(e)[:300]})
            except Exception:  # noqa: BLE001
                pass
        finally:
            save_state(st)


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in ("run", "--dry-run"):
        print(__doc__)
        sys.exit(2)
    if sys.argv[1] == "--dry-run":
        sys.argv.pop(1)
    main()
