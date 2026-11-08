#!/usr/bin/env python3
"""Find by sound: learn from the rolling recording (the last hours of audio already on the server), with the same rules
as the live watcher, so the memory does not start empty after an install.

  episodes  250 ms blocks at least 18 dB over the rolling noise floor (and above -45 dBFS), ended by 3 s of quiet; a
            single burst of <= 0.5 s is dropped (server/watch.py Audio)
  kept      YAMNet: a dog sound, or no speech/music/TV class >= SOUNDFIND_SPEECH_MAX (privacy)
  label     a verified box (score >= 0.5) at a settled pose within 4 s of the episode, no person (score < 0.5) in any of
            those frames: the pose that centres her (server/soundfind.py aim_pose)
Rows are written to the same memory and ledger with "backfill": true. In-view labels never move the gate.
Run on the server (root), then restart petcam-watch so it loads them:
    sudo /opt/petcam/venv/bin/python /opt/petcam/soundfind_backfill.py [--dry-run]"""
import datetime as dt
import glob
import json
import os
import sqlite3
import subprocess
import sys

import numpy as np

sys.path.insert(0, "/opt/petcam")
import petconf  # noqa: E402
import soundfind  # noqa: E402

STATE = petconf.STATE
REC = "/var/lib/mediamtx/recordings/cam"
BLOCK = 4000


def episodes(x):
    """(start_index, pcm) per sound episode, as the watcher's Audio thread closes them."""
    levels, floor, out = [], -60.0, []
    burst, bursts, quiet, start = 0, [], 0, None
    for i in range(0, len(x) - BLOCK + 1, BLOCK):
        b = x[i:i + BLOCK].astype(np.float32)
        db = 20 * np.log10(max(1e-9, float(np.sqrt(np.mean(b * b))) / 32768))
        levels.append(db)
        levels = levels[-1200:]
        if len(levels) % 8 == 1:
            floor = float(np.percentile(levels, 20)) if len(levels) > 40 else -60.0
        if db > max(floor + 18, -45):
            if start is None:
                start = i
            burst += 1
            quiet = 0
            continue
        if burst:
            bursts.append(burst)
            burst = 0
        quiet += 1
        if bursts and quiet >= 12:
            if not (len(bursts) == 1 and bursts[0] <= 2):
                a = max(0, start - BLOCK * 2)
                out.append((a, x[a:i - BLOCK * 10]))
            bursts, start = [], None
    return out


def main(dry):
    import watch                                              # SoundModel, frame size and privacy check, exactly as live
    if watch.privacy_flag():
        print("privacy is on: nothing done")
        return
    model = watch.SoundModel()
    frame = (watch.camapi.PAN_STEPS_PER_FRAME, watch.camapi.TILT_STEPS_PER_FRAME)
    con = sqlite3.connect(f"file:{STATE}/petcam.db?mode=ro", uri=True)
    sf = soundfind.SoundFind(STATE, log=print)
    done = {r.get("snippet") for r in sf.ledger}
    n_ep = n_kept = n_lab = 0
    for f in sorted(glob.glob(f"{REC}/*.mp4"))[:-1]:          # the newest segment is still being written
        st = dt.datetime.strptime(os.path.basename(f)[:19], "%Y-%m-%d_%H-%M-%S").replace(tzinfo=dt.timezone.utc).timestamp()
        raw = subprocess.run(["nice", "-n", "15", "ffmpeg", "-nostdin", "-loglevel", "error", "-threads", "1", "-i", f,
                              "-vn", "-ac", "1", "-ar", "16000", "-f", "s16le", "-"], capture_output=True, timeout=300).stdout
        x = np.frombuffer(raw, dtype=np.int16)
        for a, pcm in episodes(x):
            n_ep += 1
            t0 = st + a / soundfind.SR
            key = f"backfill:{t0:.2f}"
            if key in done or len(pcm) < soundfind.SR // 4:
                continue
            y = model.classify(pcm)
            dog = y["dog"] >= watch.DOG_SOUND_THRESHOLD
            if not dog and y.get("speech", 1.0) >= watch.SOUNDFIND_SPEECH_MAX:
                continue
            fp = soundfind.fingerprint(pcm)
            if fp is None:
                continue
            n_kept += 1
            t1 = t0 + len(pcm) / soundfind.SR
            rows = con.execute("SELECT o.ts, o.score, o.box, o.person, p.pan, p.tilt, p.pose_settled FROM obs o "
                               "JOIN pose_observation p ON p.ts = o.ts WHERE o.ts BETWEEN ? AND ?", (t0 - 4, t1 + 4)).fetchall()
            seen = [r for r in rows if r[1] and r[1] >= 0.5 and r[2] and r[6] and r[4] is not None]
            if not seen or any((r[3] or 0) >= 0.5 for r in rows):
                continue
            r = min(seen, key=lambda r: abs(r[0] - t0))
            aim = soundfind.aim_pose((r[4], r[5]), json.loads(r[2]), frame)
            n_lab += 1
            if not dry:
                sf.learn(fp, aim, "view", frame, [], {"kind": y["dog_kind"] if dog else "sound", "dog": dog, "top": y.get("top"),
                                                       "snippet": key, "backfill": True, "pose": [r[4], r[5]]})
    print(f"episodes {n_ep}, kept (dog or no speech) {n_kept}, labelled with her in view {n_lab}{' (dry run)' if dry else ''}; "
          f"memory now {len(sf.memory)}")


if __name__ == "__main__":
    main("--dry-run" in sys.argv)
