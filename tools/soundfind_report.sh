#!/usr/bin/env bash
# What find by sound has learned and whether it is being used (read-only; server/soundfind.py).
#   tools/soundfind_report.sh
set -u
set -a; . "$(dirname "$0")/../config/site.env"; set +a
KEY=${SSH_KEY/#\~/$HOME}
ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=20 "$SSH_HOST" 'sudo -n /opt/petcam/venv/bin/python - 2>/dev/null || sudo -n python3 -' <<'PY'
import collections, glob, json, os, sys, time
sys.path.insert(0, "/opt/petcam")
import soundfind as S
sf = S.SoundFind("/var/lib/petcam", log=lambda *a: None)
ok, g = sf.gate()
settings = {}
try:
    settings = json.load(open("/var/lib/petcam/settings.json"))
except OSError:
    pass
mode = settings.get("sound_find", "auto")
print(f"mode {mode}; guess {'USED' if sf.use_guess(mode) else 'not used'}; gate {g}")
print(f"(gate: >= {S.GATE_MIN_N} search labels, >= {S.GATE_MARGIN} more top-2 hits than the baseline, >= {S.GATE_MIN_RATE:.0%} hits)")
by = collections.defaultdict(lambda: [0, 0, 0, 0])
for r in sf.ledger:
    k = by[r.get("source")]
    k[0] += 1
    if r.get("hit2") is not None and r.get("base_hit2") is not None:
        k[1] += 1; k[2] += bool(r["hit2"]); k[3] += bool(r["base_hit2"])
for src, (n, both, h, b) in by.items():
    print(f"{src:7s} labels {n:5d}; with a guess {both:5d}: guess top-2 {h} ({h / max(1, both):.0%}), baseline top-2 {b} ({b / max(1, both):.0%})")
snips = glob.glob("/var/lib/petcam/sounds/*/*.wav")
print(f"snippets {len(snips)}, {sum(os.path.getsize(p) for p in snips) / 1e6:.1f} MB")
for r in sf.ledger[-5:]:
    print(time.strftime("%m-%d %H:%M", time.localtime(r["t"])), r.get("source"), r.get("kind"), "her view", r["aim"],
          "guess", r.get("guess")[:2], "hit" if r.get("hit2") else "-", "| baseline", "hit" if r.get("base_hit2") else "-")
PY
