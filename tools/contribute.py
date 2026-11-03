#!/usr/bin/env python3
"""Offer a fix that an agent shipped on this install upstream, as a pull request everyone can benefit from.

Open source in the age of agents: every install runs an agent that turns its owners' reports into tested fixes.
A fix that is proven here (tests, held-out replay, live and healthy on this server) is usually good for every
other install too, so the agent opens a pull request to the upstream repository instead of keeping it private.

What goes into the PR is written for strangers. The worker writes a public summary (what was wrong and what
changed, in general terms), and nothing from this home goes in: no report frames, clips, times, names, places
or server addresses. The message and the diff both pass tools/check_public.sh, and the pet's name is replaced.

    python3 tools/contribute.py --check            # what would happen on this install (no network writes)

Settings (config/site.env):
  CONTRIBUTE=pr | off        default pr
  UPSTREAM_REPO=owner/name   default crimeacs/OpenPetCam

On the upstream maintainer's own install (origin is the upstream repository) nothing is opened: the fix is
already on main and reaches GitHub when main is pushed. Everyone else needs a fork as `origin` and `gh` signed
in; without them the agent says why it skipped and the fix stays local.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from siteconf import PET, ROOT, SITE  # noqa: E402

UPSTREAM = SITE.get("UPSTREAM_REPO", "crimeacs/OpenPetCam")
MODE = SITE.get("CONTRIBUTE", "pr").lower()
DRY = os.environ.get("PETCAM_CONTRIB_DRY") == "1"     # tests: plan everything, write nothing to GitHub


def run(cmd, cwd=ROOT, timeout=300, check=False):
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL,
                       env=dict(os.environ, GIT_TERMINAL_PROMPT="0", GH_PROMPT_DISABLED="1"))
    if check and r.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd[:3])}: {(r.stderr or r.stdout)[-300:]}")
    return r


def slug(url):
    """'https://github.com/a/b.git' or 'git@github.com:a/b.git' -> 'a/b' (lowercase); anything else -> ''."""
    m = re.search(r"github\.com[:/]([^/]+)/([^/]+?)(?:\.git)?/?$", (url or "").strip())
    return f"{m.group(1)}/{m.group(2)}".lower() if m else ""


def scrub(text):
    """Public text: no local paths, no worktree names, no pet name."""
    text = re.sub(re.escape(ROOT) + r"/eval/agent-worktrees/[^/\s]+/", "", text or "")
    text = text.replace(ROOT + "/", "").replace(os.path.expanduser("~") + "/", "~/")
    if PET and PET.lower() not in ("the pet", "your pet") and len(PET) > 1:
        text = re.sub(rf"\b{re.escape(PET)}\b", "the pet", text, flags=re.IGNORECASE)
    return text[:1].upper() + text[1:] if text else text


def plan():
    """Where a contribution would go from this install, or why not. Pure reads."""
    if MODE != "pr":
        return {"go": False, "why": "CONTRIBUTE is off in config/site.env"}
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", UPSTREAM):
        return {"go": False, "why": f"UPSTREAM_REPO '{UPSTREAM}' is not owner/name"}
    origin = slug(run(["git", "remote", "get-url", "origin"]).stdout)
    if not origin:
        return {"go": False, "why": "no GitHub 'origin' remote; fork the upstream repository and set it as origin"}
    if origin == UPSTREAM.lower():
        return {"go": False, "maintainer": True,
                "why": "this install is the upstream repository: the fix reaches GitHub when main is pushed"}
    if run(["gh", "auth", "status"], timeout=30).returncode != 0:
        return {"go": False, "why": "gh is not signed in (run `gh auth login` once)"}
    return {"go": True, "origin": origin, "fork_owner": origin.split("/")[0]}


def body(title, public_summary, evidence, tests):
    lines = [public_summary.strip() or title, "",
             "### How it was checked",
             f"- Tests: {tests.strip() or 'the repository tests'}",
             "- The agent's gates on the install that found it: tests, compile and undefined names, the public-repo "
             "guard, protected-evaluator and assertion-count checks.",
             ]
    if evidence:
        lines.append(f"- Held-out replay: {evidence}")
    lines += ["- It ran live on that install and passed the health check before this PR was opened.", "",
              "Opened by the OpenPetCam agent (tools/contribute.py) after an owner's report or a missed goal on their "
              "install. The install's frames, clips and settings stay private; only the change and this summary are "
              "shared. A maintainer reviews and merges as usual."]
    return scrub("\n".join(lines))


def contribute(ident, commit, title, public_summary, evidence="", tests="", gates=None):
    """Cherry-pick `commit` onto upstream/main, check it, push to the fork and open a PR. Returns (status, detail)."""
    p = plan()
    if not p["go"]:
        return ("skipped", p["why"])
    branch = "agent-fix/" + re.sub(r"[^a-zA-Z0-9_.-]", "-", ident)[:60]
    wt = tempfile.mkdtemp(prefix="contrib-", dir=os.path.join(ROOT, "eval"))
    try:
        run(["git", "fetch", "-q", f"https://github.com/{UPSTREAM}.git", "main:refs/contrib/upstream-main"], check=True, timeout=180)
        run(["git", "worktree", "add", "-q", "--detach", wt, "refs/contrib/upstream-main"], check=True)
        if run(["git", "cherry-pick", commit], cwd=wt).returncode != 0:
            run(["git", "cherry-pick", "--abort"], cwd=wt)
            return ("skipped", "the fix does not apply cleanly to upstream main; it stays on this install")
        msg = scrub(f"{title}\n\n{public_summary}\n\nOpened by the OpenPetCam agent from a fix proven on a user's install.")
        run(["git", "commit", "-q", "--amend", "-m", msg], cwd=wt, check=True)
        if gates:
            ok, why = gates(wt)
            if not ok:
                return ("skipped", f"on upstream main the checks fail: {why[:200]}")
        # The guard needs this install's private values (server address, pet's name) to know what to refuse.
        os.makedirs(os.path.join(wt, "config"), exist_ok=True)
        shutil.copy(os.path.join(ROOT, "config", "site.env"), os.path.join(wt, "config", "site.env"))
        guard = run(["bash", "-c", "git diff --name-only HEAD~1 | xargs -r tools/check_public.sh --files"], cwd=wt)
        if guard.returncode != 0:
            return ("skipped", "the public-repo guard refused the change: " + (guard.stdout or guard.stderr)[-200:])
        if DRY:
            return ("dry", json.dumps({"branch": branch, "head": f"{p['fork_owner']}:{branch}", "repo": UPSTREAM,
                                       "title": scrub(title)[:120], "body": body(title, public_summary, evidence, tests)}))
        run(["git", "push", "-q", "--force", "origin", f"HEAD:refs/heads/{branch}"], cwd=wt, check=True, timeout=180)
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
            f.write(body(title, public_summary, evidence, tests))
        r = run(["gh", "pr", "create", "--repo", UPSTREAM, "--head", f"{p['fork_owner']}:{branch}", "--base", "main",
                 "--title", scrub(title)[:120], "--body-file", f.name], cwd=wt, timeout=120)
        os.unlink(f.name)
        if r.returncode != 0:
            if "already exists" in (r.stderr or ""):
                return ("pr_updated", "the pull request for this fix was updated")
            return ("pr_failed", (r.stderr or r.stdout)[-300:])
        return ("pr_opened", r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "")
    except Exception as e:      # noqa: BLE001  a failed contribution never undoes the local fix
        return ("pr_failed", str(e)[:300])
    finally:
        run(["git", "worktree", "remove", "--force", wt])


if __name__ == "__main__":
    if "--check" in sys.argv:
        print(json.dumps(plan(), indent=1))
