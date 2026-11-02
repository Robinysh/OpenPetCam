#!/usr/bin/env python3
"""Review a day's event clips: did each event really happen as labelled?

  clipreview.py [YYYY-MM-DD]      default: yesterday (the nightly loop's day)

For every event with a clip: pull the clip, make a 12-frame sheet with the event header, have
`claude -p` judge it (correct / false positive / wrong label / misses the action / unclear), and
write eval/clipreview/<day>/verdicts.json + summary.md. The model only reads pictures and metadata.
It can't hear, so sound verdicts are judged from the pictures and the audio model's scores. Nothing
is changed on the server: a person (or a Claude Code session) decides what to fix.
Run by tools/autopilot.py nightly; its findings go into the check-in.
"""
import datetime as dt
import io
import json
import os
import subprocess
import sys
from zoneinfo import ZoneInfo

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from siteconf import PET, ROOT, SITE, SSH  # noqa: E402

TZ = ZoneInfo(SITE.get("TZ", "UTC"))
BATCH = 8            # sheets per claude -p call


def day_arg():
    if len(sys.argv) > 1 and sys.argv[1][:4].isdigit():
        return sys.argv[1]
    return (dt.datetime.now(TZ) - dt.timedelta(days=1)).strftime("%Y-%m-%d")


def pull(day, out):
    start = dt.datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=TZ)
    t0, t1 = start.timestamp(), (start + dt.timedelta(days=1)).timestamp()
    q = (f"import sqlite3,json; c=sqlite3.connect('/var/lib/petcam/petcam.db'); c.row_factory=sqlite3.Row; "
         f"print(json.dumps([dict(r) for r in c.execute('select id,ts,kind,detail,clip from events where ts>=? and ts<? order by ts',({t0},{t1}))]))")
    events = json.loads(subprocess.run(SSH + [f'sudo python3 -c "{q}"'], capture_output=True, text=True, check=True, timeout=60).stdout)
    clips = [e["clip"] for e in events if e.get("clip")]
    os.makedirs(f"{out}/clips", exist_ok=True)
    if clips:
        tar = subprocess.run(SSH + ["sudo tar -C /var/lib/petcam/clips -cf - " + " ".join(clips)], capture_output=True,
                             timeout=900).stdout
        subprocess.run(["tar", "-xf", "-", "-C", f"{out}/clips"], input=tar, check=True)
    return events


def sheet(e, path, out):
    from PIL import Image, ImageDraw
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                               capture_output=True, text=True).stdout or 0)
    frames = []
    for k in range(12):
        t = dur * (k + 0.5) / 12
        png = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.2f}", "-i", path, "-frames:v", "1", "-vf", "scale=480:-1",
                              "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True).stdout
        try:
            frames.append((t, Image.open(io.BytesIO(png)).convert("RGB")))
        except Exception:  # noqa: BLE001
            pass
    if not frames:
        return None
    w, h = frames[0][1].size
    img = Image.new("RGB", (4 * w, 3 * (h + 20) + 60), (15, 15, 15))
    d = ImageDraw.Draw(img)
    when = dt.datetime.fromtimestamp(e["ts"], TZ).strftime("%H:%M:%S")
    d.text((6, 6), f"event #{e['id']} {e['kind']} at {when} (clip starts 8 s before)", fill=(255, 214, 10))
    d.text((6, 26), (e.get("detail") or "")[:170], fill=(220, 220, 220))
    for i, (t, f) in enumerate(frames):
        x, y = (i % 4) * w, 60 + (i // 4) * (h + 20)
        img.paste(f, (x, y + 20))
        d.text((x + 4, y + 3), f"t={t:.1f}s", fill=(255, 255, 255))
    p = f"{out}/sheets/{e['id']:05d}-{e['kind']}.jpg"
    img.save(p, quality=80)
    return p


PROMPT = """You review event clips from a home pet camera watching {pet}. Read each of these sheet images: {files}
Each sheet shows 12 frames of a 30 s clip that starts 8 s before the event, with the event header (kind, time,
detail JSON). Kinds:
- appeared: the pet came into view after 3+ min away.
- active: got up after resting 5+ min.
- visit: spent time at a marked zone.
- dog_sound and similar: an audio model heard the dog (detail.yamnet.dog is its score).
- loud_sound: a loud non-dog sound.
- search: the camera looked for the pet.
You cannot hear: judge sounds by what the pictures show plus the scores, and say "unclear" when they can't tell.
Things that make an event wrong:
- a toy taken for the pet;
- the camera being moved or handheld;
- "appeared" while the pet was already there;
- "active" for a head turn;
- duplicates.
Reply with ONLY a JSON list: [{{"id": <event id>, "verdict": "correct"|"false_positive"|"wrong_label"|"misses_action"|"unclear",
"evidence": "<= 120 chars: what you saw at which t"}}]"""


def judge(paths):
    env = dict(os.environ, CLAUDE_CODE_HOOKS_ENABLED="false")
    r = subprocess.run(["timeout", "-k", "5", "420", "claude", "-p", "--allowedTools", "Read", "--output-format", "json",
                        PROMPT.format(pet=PET, files=" ".join(paths))], capture_output=True, text=True, env=env, cwd="/tmp")
    if r.returncode != 0:
        raise RuntimeError(f"claude -p failed ({r.returncode}): {r.stderr[-300:]}")
    text = json.loads(r.stdout)["result"]
    return json.loads(text[text.index("["): text.rindex("]") + 1])


def review(day):
    out = os.path.join(ROOT, "eval", "clipreview", day)
    os.makedirs(f"{out}/sheets", exist_ok=True)
    events = pull(day, out)
    sheets = {}
    for e in events:
        if e.get("clip") and os.path.exists(f"{out}/clips/{e['clip']}"):
            p = sheet(e, f"{out}/clips/{e['clip']}", out)
            if p:
                sheets[e["id"]] = p
    verdicts = []
    ids = list(sheets)
    for i in range(0, len(ids), BATCH):
        verdicts += judge([sheets[k] for k in ids[i:i + BATCH]])
    with open(f"{out}/verdicts.json", "w") as f:
        json.dump(verdicts, f, indent=1)
    counts = {}
    for v in verdicts:
        counts[v["verdict"]] = counts.get(v["verdict"], 0) + 1
    kinds = {e["id"]: e["kind"] for e in events}
    bad = [v for v in verdicts if v["verdict"] in ("false_positive", "wrong_label", "misses_action")]
    lines = [f"# Clip review {day}", "", f"{len(events)} events, {len(sheets)} clips reviewed: {counts}", ""]
    lines += [f"- #{v['id']} {kinds.get(v['id'])}: **{v['verdict']}**, {v['evidence']}" for v in bad]
    with open(f"{out}/summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")
    return {"day": day, "events": len(events), "clips": len(sheets), "counts": counts,
            "problems": [{"id": v["id"], "kind": kinds.get(v["id"]), "verdict": v["verdict"], "evidence": v["evidence"]} for v in bad]}


if __name__ == "__main__":
    print(json.dumps(review(day_arg()), indent=1))
