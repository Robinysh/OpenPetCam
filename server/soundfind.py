"""Find by sound: where to look first when she is heard but not seen, learned from her own sounds.

No calibration and no simulator. A spot in a room colours a sound the same way every time (the room's echoes put a
fixed ripple on the spectrum, whatever the sound), and a dog makes most of her sounds in a few places (the bowls,
the door, her bed). So:

  learn     every sound heard while she is in view is stored with WHERE she was: the motor pose that centres her
            (pose + box offset). A search that finds her within SEARCH_LINK s of a sound stores the same.
  guess     a sound heard while she is out of view is compared with the stored ones (cosine of the ripple and a
            coarse spectral shape); the poses of the most similar sounds, grouped by view, are where to look first.
  verify    before a new label is stored, the guess it would have got is written to the ledger beside the free
            baseline (the room map's stops for that hour). Only labels from searches (she was NOT in view, so the
            camera was not already pointing at her) decide whether the guess is used: in view, the camera pose and
            her position coincide and a guess could just learn "where the camera points".
  switch    mode "auto" uses the guess only while the ledger's search labels show it beating the baseline
            (gate() below, declared before any data); "shadow" never uses it, only logs; "on" always.

Pure numpy (the server has no scipy). Snippets are kept for offline study: dog sounds always, other sounds only
when YAMNet heard no speech or music; deleted after RETAIN_DAYS or beyond RETAIN_MB, and never while privacy is on
(the watcher drops the audio before it gets here)."""
import json
import math
import os
import threading
import time
import wave

import numpy as np

SR = 16000
NFFT = 512                       # 32 ms segments, 31.25 Hz bins
BAND = (250.0, 7500.0)
SMOOTH_HZ = 300.0                # the smooth source/colour envelope removed to leave the room's ripple
SHAPE_EDGES = np.geomspace(250, 7500, 13)
MAX_SNIPPET_S = 6.0
MIN_SIGNAL_S = 0.1

K = 7                            # neighbours that vote
TAU = 0.08                       # similarity temperature
MIN_MEMORY = 12                  # fewer labels than this: no guess
SAME_VIEW = (0.5, 0.5)           # two poses are the same view within half a frame on both axes
SEARCH_LINK = 90.0               # a search that starts this soon after a sound is "about" that sound
MEMORY_MAX = 4000

# gate(): declared 2026-11-08 before any search label existed. Counted over ledger rows from searches only.
GATE_MIN_N = 15                  # at least this many sound-linked searches that found her
GATE_MARGIN = 3                  # the guess's top-2 must contain her view in >= this many MORE of them than the
                                 # baseline's top-2 does...
GATE_MIN_RATE = 0.40             # ...and in at least this share of them
GATE_RECENT = 20                 # judged on the most recent rows, so a guess that stops working is switched off

RETAIN_DAYS = 30
RETAIN_MB = 400

SPEECH_CLASSES = {0, 1, 2, 3, 4, 5, 6, 9, 12, 13, 24, 27, 31, 32, 132, 133, 243, 247, 251, 255, 257, 261, 518, 519}


# ---------------------------------------------------------------- the fingerprint

def _smooth(y, width_bins):
    """Gaussian smoothing along the last axis with reflected edges (numpy only)."""
    s = max(1e-6, width_bins)
    r = int(math.ceil(3 * s))
    k = np.exp(-0.5 * (np.arange(-r, r + 1) / s) ** 2)
    k /= k.sum()
    pad = np.pad(y, (r, r), mode="reflect")
    return np.convolve(pad, k, mode="valid")


def active_part(x, sr=SR):
    """The loud part of the snippet: 20 ms frames within 20 dB of the loudest one (the sound, not the quiet
    around it)."""
    n = int(0.02 * sr)
    m = len(x) // n
    if m < 2:
        return x
    e = (x[:m * n].reshape(m, n) ** 2).mean(1) + 1e-12
    db = 10 * np.log10(e)
    keep = np.repeat(db >= db.max() - 20, n)
    return x[:m * n][keep]


def fingerprint(pcm16, sr=SR):
    """{'ripple': unit vector, 'shape': unit vector} for a sound, or None when there is too little of it.

    ripple   log power (Welch, 32 ms Hann, half overlap) minus its 300 Hz-smoothed envelope, 250-7500 Hz:
             the comb the room's early reflections put on any broadband sound, independent of its loudness
    shape    the smoothed envelope in 12 log-spaced bands, mean removed: the sound's colour (what made it and
             what the path between filtered out)."""
    x = np.asarray(pcm16, dtype=np.float64) / 32768.0
    if x.ndim != 1 or not np.isfinite(x).all():
        return None
    x = active_part(x - x.mean(), sr)
    if len(x) < max(NFFT, MIN_SIGNAL_S * sr):
        return None
    w = np.hanning(NFFT)
    hop = NFFT // 2
    segs = np.lib.stride_tricks.sliding_window_view(x, NFFT)[::hop]
    p = (np.abs(np.fft.rfft(segs * w, axis=1)) ** 2).mean(0)
    f = np.fft.rfftfreq(NFFT, 1 / sr)
    logp = np.log(np.maximum(p, p.max() * 1e-10 + 1e-30))
    env = _smooth(logp, SMOOTH_HZ / (f[1] - f[0]))
    keep = (f >= BAND[0]) & (f <= BAND[1])
    ripple = (logp - env)[keep]
    ripple -= ripple.mean()
    shape = np.array([env[(f >= a) & (f < b)].mean() for a, b in zip(SHAPE_EDGES[:-1], SHAPE_EDGES[1:])])
    shape -= shape.mean()
    rn, sn = np.linalg.norm(ripple), np.linalg.norm(shape)
    if rn < 1e-9 or sn < 1e-9:
        return None
    return {"ripple": (ripple / rn).round(4).tolist(), "shape": (shape / sn).round(4).tolist()}


def similarity(a, b):
    """0..1-ish: cosine of the ripples (the room) weighted 2:1 over the cosine of the shapes (the sound)."""
    r = float(np.dot(a["ripple"], b["ripple"]))
    s = float(np.dot(a["shape"], b["shape"]))
    return (2 * r + s) / 3


def speech_score(scores):
    """Highest YAMNet score among speech, singing, music, TV and radio classes."""
    return float(max(scores[i] for i in SPEECH_CLASSES))


# ---------------------------------------------------------------- views

def aim_pose(pose, box, frame):
    """The motor pose that would centre a box seen at `pose` (the same sign convention as app.aim)."""
    fw, fh = frame
    cx, cy = (box[0] + box[2]) / 2 - 0.5, (box[1] + box[3]) / 2 - 0.5
    return [int(round(pose[0] - cx * fw)), int(round(pose[1] - cy * fh))]


def same_view(a, b, frame):
    return abs(a[0] - b[0]) <= SAME_VIEW[0] * frame[0] and abs(a[1] - b[1]) <= SAME_VIEW[1] * frame[1]


def contains(views, aim, frame, top):
    return any(same_view(v, aim, frame) for v in views[:top])


# ---------------------------------------------------------------- memory, guess, ledger

class SoundFind:
    def __init__(self, state_dir, log=print, clock=time.time):
        self.dir = os.path.join(state_dir, "soundfind")
        self.snips = os.path.join(state_dir, "sounds")
        self.memory_path = os.path.join(self.dir, "memory.jsonl")
        self.ledger_path = os.path.join(self.dir, "ledger.jsonl")
        self.log, self.clock = log, clock
        self.lock = threading.Lock()
        self.R = np.zeros((0, 0), np.float32)    # ripples, one row per label (float32: ~4 MB at MEMORY_MAX)
        self.Sh = np.zeros((0, 0), np.float32)   # shapes
        self.aims = []
        self.ledger = []
        os.makedirs(self.dir, exist_ok=True)
        self._load()

    @property
    def memory(self):
        return self.aims

    def _add(self, fp, aim):
        r = np.asarray(fp["ripple"], np.float32)[None]
        sh = np.asarray(fp["shape"], np.float32)[None]
        if self.R.shape[0] and (self.R.shape[1] != r.shape[1] or self.Sh.shape[1] != sh.shape[1]):
            return                                # a fingerprint of another layout (older code): skip it
        self.R = np.concatenate([self.R, r]) if self.R.shape[0] else r
        self.Sh = np.concatenate([self.Sh, sh]) if self.Sh.shape[0] else sh
        self.aims.append(list(aim))
        if len(self.aims) > MEMORY_MAX:
            self.R, self.Sh, self.aims = self.R[-MEMORY_MAX:], self.Sh[-MEMORY_MAX:], self.aims[-MEMORY_MAX:]

    def _load(self):
        for path, kind in ((self.memory_path, "memory"), (self.ledger_path, "ledger")):
            try:
                with open(path) as f:
                    for line in f:
                        try:
                            row = json.loads(line)
                        except ValueError:
                            continue
                        if kind == "memory":
                            self._add(row["fp"], row["aim"])
                        else:
                            self.ledger.append(row)
            except (OSError, KeyError, TypeError):
                pass

    def _append(self, path, row):
        with open(path, "a") as f:
            f.write(json.dumps(row) + "\n")

    def guess(self, fp, frame):
        """Views to look at first, best first: [{'pose': [pan, tilt], 'score': s, 'n': labels}], [] when the memory
        is too small or nothing is similar."""
        if fp is None:
            return []
        with self.lock:
            R, Sh, aims = self.R, self.Sh, list(self.aims)
        if len(aims) < MIN_MEMORY or R.shape[1] != len(fp["ripple"]):
            return []
        sims = (2 * (R @ np.asarray(fp["ripple"], np.float32)) + Sh @ np.asarray(fp["shape"], np.float32)) / 3
        order = np.argsort(sims)[::-1][:K]
        w = np.exp((sims[order] - sims[order[0]]) / TAU)
        views = []
        for i, wi in zip(order, w):
            aim = aims[i]
            for v in views:
                if same_view(v["pose"], aim, frame):
                    v["score"] += float(wi)
                    v["n"] += 1
                    break
            else:
                views.append({"pose": list(aim), "score": float(wi), "n": 1})
        tot = sum(v["score"] for v in views) or 1.0
        for v in views:
            v["score"] = round(v["score"] / tot, 3)
        return sorted(views, key=lambda v: -v["score"])

    def learn(self, fp, aim, source, frame, baseline=(), meta=None):
        """Store a label (source 'view' or 'search'), writing first what the guess and the baseline would have said."""
        if fp is None or aim is None:
            return None
        g = self.guess(fp, frame)
        views = [v["pose"] for v in g]
        base = [list(b) for b in baseline]
        row = {"t": round(self.clock(), 2), "source": source, "aim": list(aim),
               "guess": views[:3], "baseline": base[:3],
               "hit1": contains(views, aim, frame, 1) if views else None,
               "hit2": contains(views, aim, frame, 2) if views else None,
               "base_hit1": contains(base, aim, frame, 1) if base else None,
               "base_hit2": contains(base, aim, frame, 2) if base else None,
               "memory": len(self.memory), **(meta or {})}
        with self.lock:
            self._add(fp, aim)
            self.ledger.append(row)
            del self.ledger[:-MEMORY_MAX]
            try:
                self._append(self.memory_path, {"t": row["t"], "fp": fp, "aim": list(aim), "source": source})
                self._append(self.ledger_path, row)
            except OSError as e:
                self.log("soundfind: could not save:", e)
        return row

    def gate(self):
        """(use the guess?, summary). Search labels only, most recent GATE_RECENT, rows that had both a guess and a
        baseline."""
        with self.lock:
            rows = [r for r in self.ledger if r.get("source") == "search" and r.get("hit2") is not None
                    and r.get("base_hit2") is not None][-GATE_RECENT:]
        n = len(rows)
        hits = sum(1 for r in rows if r["hit2"])
        base = sum(1 for r in rows if r["base_hit2"])
        ok = n >= GATE_MIN_N and hits - base >= GATE_MARGIN and hits >= GATE_MIN_RATE * n
        return ok, {"n": n, "guess_top2": hits, "baseline_top2": base, "views_learned": len(self.memory)}

    def use_guess(self, mode):
        if mode == "on":
            return True
        if mode == "auto":
            return self.gate()[0]
        return False

    # ------------------------------------------------------------ snippets

    def keep_snippet(self, pcm16, t0, kind, meta):
        """Write the sound as a 16 kHz mono WAV with a JSON beside it; returns the path or None."""
        try:
            x = np.asarray(pcm16, dtype=np.int16)[-int(MAX_SNIPPET_S * SR):]
            day = time.strftime("%Y-%m-%d", time.localtime(t0))
            d = os.path.join(self.snips, day)
            os.makedirs(d, exist_ok=True)
            base = os.path.join(d, f"{time.strftime('%H%M%S', time.localtime(t0))}-{int(t0 * 1000) % 1000:03d}-{kind}")
            with wave.open(base + ".wav", "wb") as w:
                w.setnchannels(1)
                w.setsampwidth(2)
                w.setframerate(SR)
                w.writeframes(x.tobytes())
            with open(base + ".json", "w") as f:
                json.dump({"t0": round(t0, 3), "kind": kind, **meta}, f)
            return base + ".wav"
        except Exception as e:  # noqa: BLE001 - evidence must never break listening
            self.log("soundfind snippet:", e)
            return None

    def compact(self):
        """Keep the memory and ledger files to their last MEMORY_MAX lines (they are appended to on every label)."""
        with self.lock:
            for path in (self.memory_path, self.ledger_path):
                try:
                    with open(path) as f:
                        lines = f.readlines()
                except OSError:
                    continue
                if len(lines) > MEMORY_MAX:
                    with open(path + ".tmp", "w") as f:
                        f.writelines(lines[-MEMORY_MAX:])
                    os.replace(path + ".tmp", path)

    def prune(self, now=None):
        """Housekeeping: drop snippet days older than RETAIN_DAYS, then the oldest files beyond RETAIN_MB."""
        now = now or self.clock()
        self.compact()
        if not os.path.isdir(self.snips):
            return
        cutoff = time.strftime("%Y-%m-%d", time.localtime(now - RETAIN_DAYS * 86400))
        files = []
        for day in sorted(os.listdir(self.snips)):
            p = os.path.join(self.snips, day)
            if day[:10] < cutoff:
                for n in os.listdir(p):
                    os.remove(os.path.join(p, n))
                os.rmdir(p)
                continue
            for n in os.listdir(p):
                q = os.path.join(p, n)
                files.append((os.path.getmtime(q), os.path.getsize(q), q))
        total = sum(f[1] for f in files)
        for _, size, q in sorted(files):
            if total <= RETAIN_MB * 1e6:
                break
            os.remove(q)
            total -= size
