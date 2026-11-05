"""Sound (2-7.6 kHz active, ~20 Hz-20 kHz passive; wavelengths 1.7 cm to 17 m).

ActiveSonar: the camera's speaker plays a sweep and its microphone records it; deconvolution gives the impulse
response at that head direction (speaker and mic turn with the head). Distances are one-way, read from the delay after
the direct (speaker-to-mic) arrival, so they come out short by about half the speaker-mic spacing. The band is cut with
a smooth taper: a hard band edge invents a peak near 0.15 m (checked on a synthetic echo-free room).
The late part of each response (the coda) is kept as a fingerprint: same-pose repeats correlate, a changed room does not.

AmbientSound: the server's per-minute noise level, and the passive-echo test on 8 minutes of room noise (not
significant). Passive echo recovery stays an experiment: no whitening (it erases what it should find)."""
import datetime as dt
import json
import os

import numpy as np
from scipy import signal

from ..geometry import az_el
from ..layers import Band, Layer
from .base import Sensor

FS, C = 48000, 343.0


def load_wav(path):
    raw = open(path, "rb").read()
    i = raw.find(b"data") + 8
    return np.frombuffer(raw[i: i + (len(raw) - i) // 2 * 2], np.int16).astype(float) / 32768


def responses(x, probe, tail=0.06, band=(2100, 7400)):
    """Impulse responses (one per sweep found in x), aligned to their direct arrival, band-limited with a Tukey taper."""
    env = np.abs(signal.hilbert(signal.correlate(x, probe, mode="valid", method="fft")))
    peaks, _ = signal.find_peaks(env, distance=int(1.2 * FS), height=env.max() * 0.3)
    out = []
    for p in peaks:
        seg = x[p - int(0.005 * FS): p + len(probe) + int(tail * FS)]
        if len(seg) < len(probe) + int(tail * FS):
            continue
        N = 1 << int(np.ceil(np.log2(len(seg) + len(probe))))
        S, X = np.fft.rfft(probe, N), np.fft.rfft(seg, N)
        f = np.fft.rfftfreq(N, 1 / FS)
        m = (f > band[0]) & (f < band[1])
        w = np.zeros_like(f)
        w[m] = signal.windows.tukey(m.sum(), 0.5)
        h = np.fft.irfft(w * X * np.conj(S) / (np.abs(S) ** 2 + 1e-3 * np.abs(S).max() ** 2), N)[: int(0.005 * FS) + int(tail * FS)]
        d0 = int(np.argmax(np.abs(signal.hilbert(h))))
        out.append(h[d0:] / np.abs(h[d0]))
    return out


class ActiveSonar(Sensor):
    name = "acoustic.active"
    band = Band("acoustic", C / 7400, C / 2100, "camera speaker sweep 2.1-7.4 kHz, camera microphone, monostatic")

    def layers(self):
        ctx = self.ctx
        rawd = ctx.path("raw", "acoustic")
        meta = json.load(open(os.path.join(rawd, "recordings.json")))
        probe = signal.resample_poly(np.load(os.path.join(rawd, meta["probe"])), 3, 1)
        items, coda = [], {}
        recs = [dict(r, path=os.path.join(rawd, r["file"]), raw=False) for r in meta["active"]]
        # raw-tap scans (sim/world scan runner): one folder per session with log.jsonl and one .pcm per recording
        for d in sorted(os.listdir(rawd)):
            lp = os.path.join(rawd, d, "log.jsonl")
            if not os.path.exists(lp):
                continue
            for line in open(lp):
                q = json.loads(line)
                f = os.path.join(rawd, d, f"{q['k']:02d}_{q['kind']}_{q['pan']}_{q['tilt']}_v{q['vol']}.pcm")
                if q["vol"] and os.path.exists(f) and os.path.getsize(f) > 96000 * 8:
                    recs.append({"file": f"{d}/{os.path.basename(f)}", "path": f, "raw": True, "pan": q["pos"][0], "tilt": q["pos"][1],
                                 "t": dt.datetime.fromtimestamp(q["t_rec"]).isoformat(), "kind": q["kind"], "vol": q["vol"]})
        for r in recs:
            x = (np.frombuffer(open(r["path"], "rb").read(), np.int16).astype(float) / 32768) if r["raw"] else load_wav(r["path"])
            hs = responses(x, probe)
            if not hs:
                continue
            n = min(len(h) for h in hs)
            H = np.array([h[:n] for h in hs])
            env = np.abs(signal.hilbert(H.mean(0)))
            env /= env.max()
            rng = C * np.arange(n) / FS / 2
            keep = rng <= 3.0
            pk, _ = signal.find_peaks(env, distance=int(0.0004 * FS), prominence=0.04)
            pk = [k for k in pk if 0.08 < rng[k] <= 3.0]
            d = ctx.camera.R(r["pan"], r["tilt"])[:, 2]
            az, el = az_el(d)
            items.append({"file": r["file"], "raw_tap": r["raw"], "kind": r.get("kind", "early"), "pan": r["pan"], "tilt": r["tilt"], "az": round(float(az), 1),
                          "el": round(float(el), 1), "t": dt.datetime.fromisoformat(r["t"]).timestamp(), "sweeps": len(hs),
                          "range_m": np.round(rng[keep][::4], 4).tolist(), "envelope": np.round(env[keep][::4], 4).tolist(),
                          "echoes": [[round(float(rng[k]), 3), round(float(env[k]), 3)] for k in pk]})
            coda[r["file"]] = H.mean(0)[int(2 * 0.5 / C * FS): int(2 * 3.0 / C * FS)]
        # coda fingerprint: correlation of the late response (0.5-3 m) between recordings
        names = list(coda)
        cm = [[round(float(np.corrcoef(coda[a][: min(len(coda[a]), len(coda[b]))],
                                       coda[b][: min(len(coda[a]), len(coda[b]))])[0, 1]), 3) for b in names] for a in names]
        t0 = min(i["t"] for i in items)
        return [Layer("acoustic.echoes", self.name, self.band, "echoes", "Sound: echo profiles by head direction",
                      "Sweep played by the camera speaker, recorded by its microphone, deconvolved with a smooth band "
                      "taper and averaged over the sweeps. One-way range from the delay after the direct arrival.",
                      t0, max(i["t"] for i in items), {"items": items}),
                Layer("acoustic.coda", self.name, self.band, "matrix", "Sound: room fingerprint (coda correlation)",
                      "Correlation of the late response (0.5-3 m of path) between recordings. Same pose, same room: "
                      "high. A changed room or another head direction: lower.", t0, None, {"names": names, "corr": cm})]


class AmbientSound(Sensor):
    name = "acoustic.ambient"
    band = Band("acoustic", 0.017, 17.0, "camera microphone, passive")

    def layers(self):
        rows = [m for m in self.ctx.json("minute.json") if m.get("noise_db") is not None]
        out = [Layer("acoustic.noise", self.name, self.band, "series", "Sound level per minute",
                     "Microphone level each minute from the server (dB relative to full scale).",
                     rows[0]["ts"] if rows else None, rows[-1]["ts"] if rows else None,
                     {"unit": "dBFS", "t": [m["ts"] for m in rows], "v": [round(m["noise_db"], 1) for m in rows],
                      "events": [m["ts"] for m in rows if m.get("sounds")]})]
        tap = os.path.join(self.ctx.path("raw", "acoustic"), "tap_test.pcm")
        if os.path.exists(tap):
            x = np.frombuffer(open(tap, "rb").read(), np.int16).astype(float) / 32768
            f, P = signal.welch(x, FS, nperseg=4096)
            out.append(Layer("acoustic.spectrum", self.name, self.band, "spectrum", "Sound: raw microphone spectrum",
                             "4 s of the camera's raw microphone tap (48 kHz, before the Opus encoder): energy reaches "
                             "~20 kHz, so passive sounds carry 3x the bandwidth of the speaker's sweeps.",
                             None, None, {"f": np.round(f[::4]).tolist(), "db": np.round(10 * np.log10(P[::4] + 1e-20), 1).tolist()}))
        out.append(Layer("acoustic.passive_test", self.name, self.band, "note", "Sound: passive echo test",
                         "8 minutes of room noise, split into speech and impulsive windows, autocorrelated and stacked: "
                         "echo strength at the active-sweep distances z = 0.22-0.44 against random lags (control 0.00). "
                         "Not significant. The accumulator will stack only impulsive non-speech sound from the raw tap, "
                         "without whitening, over days, and must reach z > 3.", None, None,
                         {"z": {"all": 0.43, "voiced": 0.44, "unvoiced": 0.33, "impulsive": 0.22, "control": 0.0}}))
        return out
