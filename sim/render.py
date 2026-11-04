#!/usr/bin/env python3
"""Render a simulated run (real server/tracker.py in the calibrated room) to a video: the camera's view on the pan/tilt
map, her true position, what the watcher saw, and a timeline. Local viewing only (eval/ is git-ignored).

    python3 sim/render.py [seed] [minutes] [out.mp4]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import animation  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gate1  # noqa: E402
from room import Room, load_params  # noqa: E402


def record(seed, minutes):
    P = load_params()
    room = Room(seed=seed, free=P.get("free"))
    T = gate1.load_tracker()
    frames = []
    orig_advance = room.advance

    def advance(dt):                                   # sample the world every 0.25 s of sim time
        n = max(1, int(round(dt / 0.25)))
        for _ in range(n):
            orig_advance(dt / n)
            frames.append((room.t, room.X, room.Y, room.px, room.py, room.dog_state, room.in_view(), len(room.moves)))
    room.advance = advance
    room.sleep = lambda s: advance(max(room.dt, s))
    obs, a = gate1.run_watcher(room, T, minutes / 60)
    return room, frames, obs, a


def main(seed=11, minutes=6, out=None):
    room, frames, obs, acts = record(seed, minutes)
    out = out or os.path.join(HERE, "..", "eval", "rl", f"sim_seed{seed}.mp4")
    seen_t = [o[0] for o in obs if o[1] > 0]
    fig = plt.figure(figsize=(10, 6.2), dpi=110)
    ax = fig.add_axes([0.06, 0.34, 0.9, 0.6])
    tl = fig.add_axes([0.06, 0.08, 0.9, 0.18])
    fw, fh = room.FW, room.FH
    ax.set_xlim(room.XM + 400, -400)                   # view right = lower pan: draw it the way the camera sees the room
    ax.set_ylim(0, room.YM)
    ax.set_xlabel("pan (motor steps)  -  the room as the camera sees it, left to right")
    ax.set_ylabel("tilt")
    for p in room.presets():
        ax.plot(p["x"], p["y"], marker="s", ms=5, color="#8aa0ff", alpha=0.6)
        ax.text(p["x"], p["y"] + 25, p["name"], fontsize=7, color="#5a6fd6", ha="center")
    view = plt.Rectangle((0, 0), fw, fh, fill=False, lw=2, ec="#355BEB")
    ax.add_patch(view)
    dog, = ax.plot([], [], "o", ms=12)
    title = ax.set_title("")
    T_end = frames[-1][0]
    tl.set_xlim(0, T_end / 60)
    tl.set_ylim(0, 3)
    tl.set_yticks([0.5, 1.5, 2.5], ["camera moves", "watcher saw her", "truly in view"], fontsize=8)
    tl.set_xlabel("minutes")
    iv = [(f[0] / 60) for f in frames if f[6] and f[5] != "away"]
    tl.scatter(iv, [2.5] * len(iv), s=2, color="#3cb371", marker="|")
    tl.scatter([t / 60 for t in seen_t], [1.5] * len(seen_t), s=8, color="#355BEB", marker="|")
    tl.scatter([m[0] / 60 for m in room.moves], [0.5] * len(room.moves), s=20, color="#F2B33D", marker="v")
    cursor = tl.axvline(0, color="k", lw=1)
    step = 4                                           # 1 s of sim per video frame at 0.25 s samples -> 4x speed at 4 fps... use 8 fps
    idx = list(range(0, len(frames), step))

    def draw(i):
        t, X, Y, px, py, st, inv, nm = frames[i]
        view.set_xy((X - fw / 2, Y - fh / 2))
        view.set_edgecolor("#3cb371" if inv else "#355BEB")
        if st == "away":
            dog.set_data([], [])
        else:
            dog.set_data([px], [py])
            dog.set_color("#F2B33D" if st == "move" else "#777777")
        recent = [a for a in acts if t - 6 <= a[0] <= t]
        title.set_text(f"t = {t/60:5.1f} min   dog: {st:<4}   {'IN VIEW' if inv else 'not in view'}   "
                       f"moves so far {nm}   {('last action: ' + recent[-1][1]) if recent else ''}")
        cursor.set_xdata([t / 60, t / 60])
        return view, dog, title, cursor

    anim = animation.FuncAnimation(fig, draw, frames=idx, blit=False)
    anim.save(out, writer=animation.FFMpegWriter(fps=8, bitrate=1800))
    print("wrote", os.path.relpath(out), f"({len(idx)} frames, {T_end/60:.1f} sim min at {step * 0.25 * 8:.0f}x)")
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    main(int(a[0]) if a else 11, float(a[1]) if len(a) > 1 else 6, a[2] if len(a) > 2 else None)
