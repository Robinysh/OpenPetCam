#!/usr/bin/env python3
"""The unattended improvement loop: retrains the "is it really my pet?" verifier every night and
checks in on the whole system, so the owner only hears about it when something changed or broke.

  autopilot.py nightly      collect -> label (claude -p) -> train -> gate -> deploy/keep -> check-in
  autopilot.py train        just the training half
  autopilot.py checkin      just the check-in half
  autopilot.py --dry-run …  everything except deploying and notifying

Runs on the owner's computer (the labeller is `claude -p`; the server is too small), started by
launchd (tools/install_autopilot.sh). Every run touches /var/lib/petcam/autopilot_ok on the server;
alert.py pages the phone if that goes 36 h stale, so a dead loop never fails silently.

Deploy gate. The candidate is scored out-of-fold (every crop by a model that never saw its day, or
hour while there is only one day). The deployed model is scored on the same crops, which flatters
it (it trained on most of them). The candidate replaces it only if it makes no more errors overall
AND no more on the newest day, which the deployed model has never seen. Otherwise the current
model stays. After a deploy the watcher must log the new verifier within 30 s, else it is rolled back.
"""
import json
import os
import shutil
import subprocess
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import bootstrap_pet as bp  # noqa: E402
from siteconf import PET, ROOT, SSH  # noqa: E402

DRY = "--dry-run" in sys.argv
LOGDIR = os.path.join(ROOT, "eval", "autopilot")
MODELS = os.path.join(ROOT, "models")
DEPLOYED = os.path.join(MODELS, "verifier.json")
PREVIOUS = os.path.join(MODELS, "verifier.prev.json")
CLAUDE_TIMEOUT = 300


def log(*a):
    line = time.strftime("%Y-%m-%d %H:%M:%S ") + " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(os.path.join(LOGDIR, "autopilot.log"), "a") as f:
        f.write(line + "\n")


def ssh(cmd, timeout=120):
    return subprocess.run(SSH + [cmd], capture_output=True, text=True, timeout=timeout)


def notify(title, body):
    log("notify:", title, "|", body)
    if DRY:
        return
    q = lambda s: "'" + s.replace("'", "'\\''") + "'"   # noqa: E731
    ssh(f"sudo -u petcam /opt/petcam/venv/bin/python /opt/petcam/alert.py --notify {q(title)} {q(body)}")


def heartbeat():
    if not DRY:
        ssh("sudo -u petcam touch /var/lib/petcam/autopilot_ok")


# ------------------------------------------------------------------ scoring

def scores(cfg, X):
    """The verifier's probability for each feature row (MobileNetV2 logits), as watch.Verifier computes it."""
    if cfg["mode"] == "logreg":
        z = (X - np.array(cfg["mean"])) / np.array(cfg["std"])
        return 1 / (1 + np.exp(-(z @ np.array(cfg["w"]) + float(cfg["b"]))))
    sm = np.exp(X - X.max(1, keepdims=True))
    sm /= sm.sum(1, keepdims=True)
    a, b = cfg.get("classes", [151, 268])
    return sm[:, a:b + 1].sum(1)


def errors(p, y, th):
    return {"toys_accepted": int(((p >= th) & (y == 0)).sum()), "pets_rejected": int(((p < th) & (y == 1)).sum())}


def is_ir(crop_path):
    """Night infrared frames are grey: almost no colour saturation."""
    from PIL import Image
    hsv = np.asarray(Image.open(crop_path).convert("HSV"), dtype=np.float32)
    return float(hsv[..., 1].mean()) < 20


def train_and_gate():
    os.environ.setdefault("PETCAM_LABEL_MAX", "400")       # at most 10 sheets a night
    bp.collect()
    bp.label()
    sys.argv = [sys.argv[0], "train", "--allow-single-day"]
    bp.train()
    cand = json.load(open(os.path.join(bp.WORK, "verifier.json")))
    cv = json.load(open(os.path.join(bp.WORK, "cv_scores.json")))
    ids = list(cv["items"])
    cache = dict(np.load(os.path.join(bp.WORK, "features.npz")))
    ids = [i for i in ids if i in cache]
    X = np.stack([cache[i] for i in ids]).astype(np.float64)
    y = np.array([cv["items"][i][1] for i in ids])
    days = np.array([cv["items"][i][2] for i in ids])
    newest = max(d for d in days if d[:4].isdigit())
    new = days == newest
    oof = np.array([cv["items"][i][0] for i in ids])
    cur = json.load(open(DEPLOYED)) if os.path.exists(DEPLOYED) else {"mode": "zeroshot", "threshold": 0.34}
    pcur = scores(cur, X)
    ir = np.array([is_ir(os.path.join(bp.WORK, "crops", f"{i}.jpg")) for i in ids])
    res = {
        "labels": {"total": len(ids), "pet": int(y.sum()), "not_pet": int((y == 0).sum()), "days": sorted(set(days)),
                   "night_ir": {"pet": int((ir & (y == 1)).sum()), "not_pet": int((ir & (y == 0)).sum())}},
        "newest_day": newest,
        "candidate": {"mode": cand["mode"], "cv_by": cv["by"], "threshold": cand["threshold"],
                      "errors_all_out_of_fold": errors(oof, y, cv["threshold"]),
                      "errors_newest_day": errors(oof[new], y[new], cv["threshold"])},
        "deployed": {"mode": cur["mode"], "threshold": cur["threshold"],
                     "errors_all_in_sample": errors(pcur, y, cur["threshold"]),
                     "errors_newest_day": errors(pcur[new], y[new], cur["threshold"]),
                     "errors_night_ir": errors(pcur[ir], y[ir], cur["threshold"])},
    }
    total = lambda e: sum(e.values())   # noqa: E731
    c_all, d_all = total(res["candidate"]["errors_all_out_of_fold"]), total(res["deployed"]["errors_all_in_sample"])
    c_new, d_new = total(res["candidate"]["errors_newest_day"]), total(res["deployed"]["errors_newest_day"])
    better = cand["mode"] == "logreg" and c_all <= d_all and c_new <= d_new and (c_all < d_all or c_new < d_new or
                                                                               cand.get("report", {}).get("items", 0) >= 1.1 * cur.get("report", {}).get("items", 0))
    res["decision"] = "deploy" if better else "keep"
    res["why"] = (f"candidate errors {c_all} (held-out) / {c_new} on {newest} vs deployed {d_all} / {d_new}")
    return cand, res


def deploy(cand):
    shutil.copy(DEPLOYED, PREVIOUS) if os.path.exists(DEPLOYED) else None
    bp.deploy()                                            # copies to models/ and the server, restarts the watcher
    t0 = time.time()
    while time.time() - t0 < 45:
        j = ssh("journalctl -u petcam-watch --since '-60 s' --no-pager | grep 'verifier:' | tail -1").stdout
        if f"verifier: {cand['mode']} threshold {cand['threshold']}" in j:
            return True
        time.sleep(3)
    return False


def rollback():
    if os.path.exists(PREVIOUS):
        shutil.copy(PREVIOUS, os.path.join(bp.WORK, "verifier.json"))
        bp.deploy()


# ------------------------------------------------------------------ check-in

def gather():
    """Facts for the check-in, collected by plain code (the model only reads and summarises)."""
    st = subprocess.run(["bash", os.path.join(HERE, "status.sh")], capture_output=True, text=True, timeout=120).stdout
    reports = subprocess.run([sys.executable, os.path.join(HERE, "triage.py"), "list"], capture_output=True, text=True,
                             timeout=120).stdout
    if "NEW" in reports and not DRY:
        subprocess.run([sys.executable, os.path.join(HERE, "triage.py"), "pull"], capture_output=True, timeout=600)
    errs = subprocess.run([sys.executable, os.path.join(HERE, "triage.py"), "errors"], capture_output=True, text=True,
                          timeout=120).stdout
    day = ssh("sudo -u petcam TZ=$(grep ^TZ= /etc/petcam/site.env | cut -d= -f2) /opt/petcam/venv/bin/python -c \""
              "import sys,json; sys.path.insert(0,'/opt/petcam'); import app; "
              "print(json.dumps({'wellbeing': app.wellbeing(None), 'diary': app.diary()[:3]}, default=str))\"").stdout
    return {"status": st, "reports": reports, "page_errors": errs[-3000:], "today": day[-6000:]}


def checkin(train_result):
    facts = gather()
    facts["training"] = train_result
    day = time.strftime("%Y-%m-%d")
    fpath = os.path.join(LOGDIR, f"facts-{day}.json")
    with open(fpath, "w") as f:
        json.dump(facts, f, indent=1, default=str)
    prompt = f"""You are the nightly check-in for OpenPetCam, a pet camera watching {PET}. Read {fpath}.
It has: a health report of the server (status), bug reports the owner filed from the app (reports; NEW = not
seen before), page JavaScript errors, today's numbers, and tonight's detector training result.
Decide whether the owner needs to know or do anything. Things that matter: a service not active, the camera
offline or the view blocked, low disk or memory, low CPU burst capacity, NEW bug reports, repeated page errors,
the detector deployed or rolled back, the detector missing the pet at night (training.deployed.errors_night_ir),
zero time with the pet in view all day while sounds show the dog is home. Things that do not matter: routine
numbers. Reply with ONLY a JSON object: {{"needs_owner": true|false, "headline": "<= 70 chars", "detail":
"<= 240 chars, plain words, what happened and what (if anything) to do", "for_claude_code": "notes for the
next debugging session, or empty"}}"""
    env = dict(os.environ, CLAUDE_CODE_HOOKS_ENABLED="false")
    r = subprocess.run(["timeout", "-k", "5", str(CLAUDE_TIMEOUT), "claude", "-p", "--allowedTools", "Read",
                        "--output-format", "json", prompt], capture_output=True, text=True, env=env, cwd=LOGDIR)
    if r.returncode != 0:
        raise RuntimeError(f"claude -p failed ({r.returncode}): {r.stderr[-300:]}")
    text = json.loads(r.stdout)["result"]
    verdict = json.loads(text[text.index("{"): text.rindex("}") + 1])
    with open(os.path.join(LOGDIR, f"checkin-{day}.md"), "w") as f:
        f.write(f"# Check-in {day}\n\n**{verdict['headline']}**\n\n{verdict['detail']}\n\n"
                f"For Claude Code: {verdict.get('for_claude_code') or '-'}\n\nFacts: {os.path.basename(fpath)}\n")
    log("check-in:", json.dumps(verdict))
    return verdict


# ------------------------------------------------------------------ main

def main():
    os.makedirs(LOGDIR, exist_ok=True)
    cmd = next((a for a in sys.argv[1:] if not a.startswith("--")), "nightly")
    log(f"== {cmd}{' (dry run)' if DRY else ''}")
    result = None
    try:
        if cmd in ("nightly", "train"):
            cand, result = train_and_gate()
            log("training:", json.dumps(result))
            if result["decision"] == "deploy" and not DRY:
                if deploy(cand):
                    notify(f"{PET}'s detector got better", f"Retrained on {result['labels']['total']} labelled crops. "
                           f"{result['why']}.")
                else:
                    rollback()
                    result["decision"] = "rolled back"
                    notify(f"{PET}'s detector: update rolled back", "The new model didn't load on the server, so the "
                           "previous one is back. Nothing to do; it will try again tomorrow.")
            with open(os.path.join(LOGDIR, f"train-{time.strftime('%Y-%m-%d')}.json"), "w") as f:
                json.dump(result, f, indent=1)
        if cmd in ("nightly", "checkin"):
            v = checkin(result)
            if v.get("needs_owner"):
                notify(v["headline"], v["detail"])
        heartbeat()
        log("done")
    except Exception as e:  # noqa: BLE001 - report, never die silently
        log("FAILED:", repr(e))
        log(traceback.format_exc())
        notify("Nightly check-up failed", f"{type(e).__name__}: {str(e)[:200]}. The camera keeps working; "
               "details in eval/autopilot/autopilot.log.")
        sys.exit(1)


if __name__ == "__main__":
    main()
