#!/usr/bin/env python3
"""Measure Find honestly: pull what each search saw, label it, and report how often Find was right.

    python3 tools/search_review.py pull     copy the server's searches (/var/lib/petcam/searches) to eval/searches/
    python3 tools/label_ui.py               queue "Searches: was Find right?" (keys 1-4)
    python3 tools/search_review.py report   Find precision, recall and time to find, day vs night

Each search the watcher runs (Follow's automatic ones and the owner's Find taps) keeps the frame at every stop
and every frame the verifier accepted, with each box and score (server/watch.py record_search_frame). The owner's
answer per search goes to eval/searches/outcomes.jsonl:
    right_find   it said "found her" and it was her
    false_find   it said "found her" and it was not her
    missed       it said "not found" but she is in one of its frames
    empty        it said "not found" and she was not in any of them
Before this (2026-11-03) "found her" could only be checked against the verifier's own opinion.
"""
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
OUT = os.environ.get("LABEL_SEARCHES") or os.path.join(ROOT, "eval", "searches")
REMOTE = "/var/lib/petcam/searches"
OUTCOMES = ("right_find", "false_find", "missed", "empty")


def searches():
    rows = []
    for day in sorted(os.listdir(OUT)) if os.path.isdir(OUT) else []:
        p = os.path.join(OUT, day)
        if not os.path.isdir(p):
            continue
        for hms in sorted(os.listdir(p)):
            m = os.path.join(p, hms, "search.json")
            if os.path.exists(m):
                try:
                    rows.append({"id": f"{day}/{hms}", **json.load(open(m))})
                except ValueError:
                    pass
    return rows


def outcomes():
    p = os.path.join(OUT, "outcomes.jsonl")
    got = {}
    if os.path.exists(p):
        for line in open(p):
            if line.strip():
                r = json.loads(line)
                got[r["search"]] = r["outcome"]          # the newest answer wins
    return got


def pull():
    sys.path.insert(0, HERE)
    from siteconf import SSH             # same host, key and options the other tools use
    os.makedirs(OUT, exist_ok=True)
    r = subprocess.run(SSH + [f"sudo test -d {REMOTE} && sudo tar -C {REMOTE} -cf - . || true"], capture_output=True, timeout=600)
    if r.returncode != 0:
        sys.exit(f"pull failed: {r.stderr[-300:]!r}")
    if r.stdout:
        subprocess.run(["tar", "-xf", "-", "-C", OUT], input=r.stdout, check=True)
    print(f"{len(searches())} searches in {os.path.relpath(OUT, ROOT)}, {len(outcomes())} labelled")


def report():
    rows, got = searches(), outcomes()
    lab = [r for r in rows if r["id"] in got]
    print(f"{len(rows)} searches recorded, {len(lab)} labelled")
    for name, part in (("all", lab), ("day", [r for r in lab if r.get("night") is False]),
                       ("night", [r for r in lab if r.get("night")])):
        o = [got[r["id"]] for r in part]
        rf, ff, mi, em = (o.count(k) for k in OUTCOMES)
        prec = rf / (rf + ff) if rf + ff else None
        rec = rf / (rf + mi) if rf + mi else None
        times = sorted(r["t1"] - r["t0"] for r in part if got[r["id"]] == "right_find")
        med = times[len(times) // 2] if times else None
        f = lambda v: "–" if v is None else f"{v:.0%}"
        print(f"  {name:<5} n={len(part):<3} found-and-right {rf}, found-but-wrong {ff}, missed {mi}, empty {em}"
              f" | precision {f(prec)} (was 'found her' really her)  recall {f(rec)} (found her when she was there)"
              f" | median time to find {'–' if med is None else f'{med:.0f} s'}")


if __name__ == "__main__":
    {"pull": pull, "report": report}.get((sys.argv[1:] or [""])[0], lambda: sys.exit(__doc__))()
