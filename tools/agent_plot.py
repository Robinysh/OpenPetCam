#!/usr/bin/env python3
"""A timestamped picture of the self-improvement loop: every goal's measurement over time against its target (from the
server's agent.db `metrics` table, written every 30 min by `superagent.py metrics`), and every improvement attempt
with its verdict (eval/superagent/archive.jsonl).

    python3 tools/agent_plot.py [hours]    ->  eval/superagent/agent_timeline.png   (default: all history)

Needs matplotlib. The picture stays in eval/ (git-ignored): it carries times from your home.
"""
import datetime as dt
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from siteconf import ROOT, SSH  # noqa: E402

LABEL = {"search_success": "Find success (share)", "find_time": "Find time (median s)", "app_status_fresh": "Status age (s)",
         "video_start": "Video start (median s)", "cpu": "CPU (% of a core)", "burst_credits": "CPU credits (%)",
         "qa": "QA failures", "reports_answered": "Oldest unanswered report (min)", "owner_reports": "Owner reports / day"}


def target_value(t):
    """The number a target line is drawn at: 'p50 <= 1, p90 <= 2' -> 1, '>= 0.8' -> 0.8, '== 0' -> 0."""
    m = re.search(r"(?:<=|>=|==|<|>)\s*([\d.]+)", t or "")
    return float(m.group(1)) if m else None


def main(hours=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    since = 0 if hours is None else dt.datetime.now().timestamp() - hours * 3600
    raw = subprocess.run(SSH + [f"sudo sqlite3 -json /var/lib/petcam/agent.db \"select ts, goal, value, target, ok from metrics "
                                f"where ts > {since} order by ts\""], capture_output=True, text=True, timeout=60).stdout
    m = json.loads(raw or "[]")
    arch_p = os.path.join(ROOT, "eval", "superagent", "archive.jsonl")
    arch = [a for a in (json.loads(l) for l in open(arch_p)) if a.get("ts", 0) > since] if os.path.exists(arch_p) else []
    goals = [g for g in LABEL if any(r["goal"] == g for r in m)]
    T = dt.datetime.fromtimestamp
    INK, MUTED, GRID, OK, MISS = "#1F2430", "#6B7280", "#E5E7EB", "#355BEB", "#C2410C"
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": GRID, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, axes = plt.subplots(len(goals) + 1, 1, figsize=(10, 1.9 * (len(goals) + 1)), sharex=True,
                             gridspec_kw={"height_ratios": [1] * len(goals) + [0.8]})
    for ax, g in zip(axes, goals):
        rows = [r for r in m if r["goal"] == g and r["value"] is not None]
        xs, ys = [T(r["ts"]) for r in rows], [r["value"] for r in rows]
        ax.plot(xs, ys, color=OK, lw=1.4, zorder=2)
        bad = [(x, y) for x, y, r in zip(xs, ys, rows) if not r["ok"]]
        if bad:
            ax.scatter(*zip(*bad), s=12, color=MISS, zorder=3)
        tv = target_value(rows[-1]["target"])
        if tv is not None:
            ax.axhline(tv, color=MUTED, ls="--", lw=0.9, zorder=1)
        missed = sum(1 for r in rows if not r["ok"])
        ax.text(0.005, 0.93, f"{LABEL[g]}   target {rows[-1]['target']}   {len(rows)} samples, {missed} missed (orange)",
                transform=ax.transAxes, va="top", color=INK, fontsize=9, fontweight="bold")
        ax.grid(axis="y", color=GRID, lw=0.6)
        ax.spines[["top", "right"]].set_visible(False)
    ax = axes[-1]
    lanes = [("live", "shipped live", OK, "o"), ("queued", "queued (not retried)", "#A3A3A3", "."),
             ("rejected", "rejected", MISS, "x"), ("duplicate", "duplicate", "#D4D4D4", ".")]
    for y, (k, name, col, mk) in zip(range(len(lanes) - 1, -1, -1), lanes):
        xs = [T(a["ts"]) for a in arch if a.get("verdict") == k]
        ax.scatter(xs, [y] * len(xs), s=40, color=col, marker=mk, zorder=3)
    ax.set_yticks(list(range(len(lanes) - 1, -1, -1)),
                  [f"{name} ({sum(a.get('verdict') == k for a in arch)})" for k, name, _, _ in lanes])
    ax.set_ylim(-0.6, len(lanes) - 0.4)
    ax.text(0.005, 1.02, "Improvement attempts (eval/superagent/archive.jsonl)", transform=ax.transAxes, color=INK, fontweight="bold")
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%a %H:%M"))
    fig.suptitle("OpenPetCam agent: goals over time and what the loop shipped (local time)", x=0.01, ha="left",
                 color=INK, fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    out = os.path.join(ROOT, "eval", "superagent", "agent_timeline.png")
    fig.savefig(out, dpi=110)
    print(out)


if __name__ == "__main__":
    main(float(sys.argv[1]) if len(sys.argv) > 1 else None)
