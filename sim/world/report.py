"""A local, step-by-step visual report of the room-model simulator and Follow v2, so a person can check every part.

It shows the inside of a home and the pet's movements: it is written to eval/<world>/report/ (git-ignored) and served
on localhost only. Each step: what it is, the picture that lets you check it, and the numbers with their nulls.
    python3 -m sim.world.report eval/world      -> eval/world/report/index.html"""
import html
import json
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RL = os.path.join(ROOT, "eval", "rl")
BG, FG, MUTED, ACC, DOG, WARN = "#0f141b", "#e8e6e1", "#8b93a1", "#7aa7ff", "#5fd3a0", "#ff8a5b"
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "axes.edgecolor": "#39404c", "axes.labelcolor": FG,
                     "xtick.color": MUTED, "ytick.color": MUTED, "text.color": FG, "font.size": 10, "savefig.facecolor": BG})


def _load(name):
    p = os.path.join(RL, name)
    return json.load(open(p)) if os.path.exists(p) else None


def floor_plan(ax, R, floor_h, max_r=7.0):
    """Top-down view of the measured room: every measured surface point, floor light, walls and objects dark."""
    H, W = R.shape
    j, i = np.mgrid[0:H, 0:W]
    az = np.radians((i + 0.5) / W * 360 - 180); el = np.radians(90 - (j + 0.5) / H * 180)
    ok = np.isfinite(R) & (R > 0) & (R < max_r)
    x = (np.cos(el) * np.sin(az) * R)[ok]; y = (np.sin(el) * R)[ok]; z = (np.cos(el) * np.cos(az) * R)[ok]
    fl = y < -floor_h + 0.08
    ax.scatter(x[fl][::3], z[fl][::3], s=0.3, c="#3a4250", lw=0)
    ax.scatter(x[~fl & (y < 1.5)][::2], z[~fl & (y < 1.5)][::2], s=0.4, c="#9aa3b2", lw=0)
    ax.plot([0], [0], "o", ms=7, c=ACC); ax.annotate("camera", (0, 0), (6, 6), textcoords="offset points", color=ACC, fontsize=8)
    ax.set_aspect("equal"); ax.set_xlabel("x, m (right)"); ax.set_ylabel("z, m (forward at pan 1850)")


def save(fig, out, name):
    fig.tight_layout(); fig.savefig(os.path.join(out, name), dpi=130); plt.close(fig)
    return name


def main(world):
    sys.path.insert(0, os.path.join(ROOT, "sim")); sys.path.insert(0, os.path.join(ROOT, "server"))
    from zoneinfo import ZoneInfo
    import gate1
    import roomfollow
    from home import HomeData, HomeRoom
    from room import Room, load_params
    out = os.path.join(world, "report"); os.makedirs(out, exist_ok=True)
    tz = ZoneInfo("America/New_York")
    data = HomeData.from_world(world, os.path.join(RL, "rl_export.json"), tz=tz, walks=os.path.join(RL, "walks.json"))
    doc = json.load(open(os.path.join(ROOT, "models", "room", "room.json")))
    rf = roomfollow.RoomFollow.load(doc, os.path.join(ROOT, "models", "room", doc["textures"]["range"]))
    R = data.R; fh = data.floor_h
    S = []                                                    # sections: (title, html)

    # 1. her positions from the watcher's records
    fig, ax = plt.subplots(figsize=(7, 6)); floor_plan(ax, R, fh)
    rows = np.array(data.rows)
    days = sorted({__import__("datetime").datetime.fromtimestamp(t, tz).strftime("%m-%d") for t in rows[:, 0]})
    cols = [DOG, WARN, ACC, "#c58bff"]
    for k, dname in enumerate(days):
        m = np.array([__import__("datetime").datetime.fromtimestamp(t, tz).strftime("%m-%d") == dname for t in rows[:, 0]])
        ax.scatter(rows[m, 1], rows[m, 2], s=4, c=cols[k % 4], label=f"Nov {dname[-2:]} ({m.sum()})", alpha=0.6, lw=0)
    ax.legend(loc="upper left", fontsize=8, facecolor=BG); ax.set_title("Step 1 · Where she was: every detection placed on the floor")
    S.append(("1. Her positions, from the camera's own records", f"<img src='{save(fig, out, 's1_positions.png')}'>"
              f"<p>Each verified box's bottom centre (her feet) along the calibrated camera ray at that moment's pose and drift, "
              f"met with the floor 0.81 m below the lens (the server's own rule, <code>roomapi.py</code>). "
              f"{len(rows)} detections from 3 to 5 Nov (the pose log starts on 3 Nov). <b>Check:</b> dots should lie on the floor "
              f"(dark-grey points), not inside walls or furniture (light points).</p>"))

    # 2. behaviour: stays, absences, gait
    fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
    st = np.array([s["end"] - s["start"] for s in data.stays])
    axs[0].hist(st / 60, bins=np.logspace(-1, 2.3, 30), color=DOG); axs[0].set_xscale("log"); axs[0].set_xlabel("stay length, min"); axs[0].set_title(f"{len(st)} stays")
    hrs = np.array([s["hour"] for s in data.stays]); w = np.array(st)
    axs[1].bar(range(24), [w[hrs == h].sum() / 3600 for h in range(24)], color=DOG); axs[1].set_xlabel("hour of day"); axs[1].set_ylabel("hours resting"); axs[1].set_title("when she rests (training days)")
    g = np.array(data.gait)
    axs[2].hist(g, bins=30, color=WARN); axs[2].axvline(0.09, color=MUTED, ls="--"); axs[2].text(0.1, axs[2].get_ylim()[1] * 0.85, "box jitter\n(90th pct)", color=MUTED, fontsize=8)
    axs[2].set_xlabel("walking speed, m/s (from video)"); axs[2].set_title(f"gait: median {np.median(g):.2f} m/s, n={len(g)}")
    S.append(("2. Her behaviour model", f"<img src='{save(fig, out, 's2_behaviour.png')}'>"
              f"<p>Stays: consecutive detections within 0.5 m with gaps under 5 min. Absences: no detection for 30 min or more "
              f"({len(data.absences)}, median {np.median([a['dur'] for a in data.absences]) / 60:.0f} min), with where she left and came back. "
              f"Walking speed cannot come from the records (a walk leaves the frame within a detector cycle), so it comes from the "
              f"continuous video (step 3). Within a stay she shifts in place: {data.fidget_share:.0%} of her detections there are 'moving'. "
              f"<b>Check:</b> do the resting hours match your day with her?</p>"))

    # 3. walks from video
    wk = _load("walks.json")
    if wk:
        mv = [it for it in wk["items"] if it["moving"]]
        fig, ax = plt.subplots(figsize=(7, 6)); floor_plan(ax, R, fh)
        for it in sorted(mv, key=lambda x: -x["path_m"])[:8]:
            P = np.array(it["points"]); ax.plot(P[:, 1], P[:, 2], "-", lw=1.2, alpha=0.9); ax.plot(P[0, 1], P[0, 2], "o", ms=3, c=FG)
        ax.set_title("Step 3 · Her real walks, from 2 h of video at 5 frames/s (8 longest)")
        S.append(("3. Walks from continuous video", f"<img src='{save(fig, out, 's3_walks.png')}'>"
                  f"<p>The server's rolling recording, judged at 5 fps by the same detector and verifier: {wk['frames_with_her']:,} frames with her, "
                  f"{wk['placed']:,} placed on the floor, recording clock {wk['clock_lag_s']:+d} s from the watcher's records (cross-correlated). "
                  f"{wk['walks']} walks (median path {wk['walk_path_m_median']} m). <b>Check:</b> the paths should follow places she really walks.</p>"))

    # 4. one simulated hour
    P = load_params(); free = P.get("free") or {}
    Room.HARD_CORRELATED = True
    T = gate1.load_tracker()
    room = HomeRoom(seed=4242, params=P, free=free, home=data, start_hour=12)
    trace = []
    tr = T.Tracker(room, room.find_pet, clock=lambda: room.t, log=lambda m: None, sleep=room.sleep, find_now=room.find_now)
    last_detect, result, last_motion = -1e9, (None, False, []), -1e9
    acts = []
    while room.t < 3600:
        motion = room.motion()
        if motion:
            last_motion = room.t
        fresh = bool(motion) or room.t - last_detect >= (8 if room.t - last_motion < 60 else 30)
        if fresh:
            result, last_detect = room.frame(), room.t
        box, moving, weak = result
        trace.append((room.t, room.px, room.py, room._state, room.in_view(), box is not None, room.X, room.Y))
        a = tr.step(box, moving=bool(moving and box), sound=False, motion=motion, weak=(weak if fresh else []) or None, box_t=last_detect)
        if a:
            acts.append((room.t, a.split(":")[0]))
        room.advance(max(0.05, (1.0 if motion else 3.0)))
    tr_ = np.array([(t, x, z, iv, dt_, X, Y) for t, x, z, st_, iv, dt_, X, Y in trace], dtype=float)
    states = [r[3] for r in trace]
    fig = plt.figure(figsize=(13, 6)); ax = fig.add_subplot(1, 2, 1); floor_plan(ax, R, fh)
    home_m = np.array([s != "away" for s in states])
    ax.plot(tr_[home_m, 1], tr_[home_m, 2], "-", c=DOG, lw=0.8, alpha=0.7)
    for k in range(0, len(tr_), max(1, len(tr_) // 25)):          # where the camera points (every ~2 min)
        d = rf.cam.ray(tr_[k, 5], tr_[k, 6], 0.5, 0.5, rf.off)
        ax.plot([0, d[0] * 2.5], [0, d[2] * 2.5], c=ACC, lw=0.6, alpha=0.5)
    ax.set_title("Step 4 · One simulated hour: her path (green), where the camera points (blue)")
    ax2 = fig.add_subplot(1, 2, 2)
    t_min = tr_[:, 0] / 60
    ax2.fill_between(t_min, 2, 3, where=home_m, color="#2b3340", step="post", label="home")
    ax2.fill_between(t_min, 1, 2, where=tr_[:, 3] > 0, color=DOG, step="post", label="truly in view")
    ax2.fill_between(t_min, 0, 1, where=tr_[:, 4] > 0, color=WARN, step="post", label="detected")
    for t, k in acts:
        ax2.plot([t / 60, t / 60], [3.1, 3.5], c={"track": ACC, "chase": "#c58bff", "search": "#ffd166"}.get(k, MUTED), lw=1.2)
    ax2.set_yticks([0.5, 1.5, 2.5, 3.3]); ax2.set_yticklabels(["detected", "truly in view", "home", "actions"]); ax2.set_xlabel("minute")
    ax2.set_title("the same hour on a timeline (actions: blue track, violet chase, yellow search)")
    S.append(("4. The simulator: one hour in your room", f"<img src='{save(fig, out, 's4_sim_hour.png')}'>"
              f"<p>The real controller (<code>server/tracker.py</code>, unchanged) drives the simulated camera; her body is projected "
              f"through the calibrated camera with drift, hidden where the measured room blocks the line from the lens. "
              f"This hour: home {home_m.mean():.0%}, truly in view {tr_[home_m, 3].mean():.0%} of home time, "
              f"{len(room.moves)} motor moves, {sum(1 for _, k in acts if k == 'chase')} chases, {sum(1 for _, k in acts if k == 'search')} searches. "
              f"<b>Check:</b> does her path look like her? Does the camera follow her the way it does at home?</p>"))

    # 4b. what the simulated camera sees: synthetic views with her projected body
    from world.synthview import load as sv_load, render as sv_render
    ctx, pano = sv_load(world)
    fig, axs = plt.subplots(1, 3, figsize=(13, 2.8))
    picks = [k for k in range(len(trace)) if trace[k][4]][:: max(1, sum(1 for r in trace if r[4]) // 3)][:3]
    for ax, k in zip(axs, picks):
        t, x, z, st_, iv, dt_, X, Y = trace[k]
        img = sv_render(ctx, pano, X, Y)
        seen, box, _c = room._view(X, Y, x, z, st_)
        ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        if box and seen:
            ax.add_patch(plt.Rectangle((box[0] * 640, box[1] * 360), (box[2] - box[0]) * 640, (box[3] - box[1]) * 360, fill=False, ec=DOG, lw=2))
        ax.set_title(f"minute {t / 60:.0f}: pose ({X:.0f}, {Y:.0f}), {st_}", fontsize=9)
    S.append(("4b. What the simulated camera sees", f"<img src='{save(fig, out, 's4b_views.png')}'>"
              "<p>The camera's view at the simulated pose, rendered from the room model, with her simulated body's visible box (green): "
              "this box is what the simulated detector judges. <b>Check:</b> is the box where a dog standing there would appear, "
              "and the size a dog would be?</p>"))

    # 5. simulator vs reality (held out by day)
    gates = [("v1", _load("home_gate_v1.json")), ("v2 (partial)", _load("home_gate_v2_partial.json")), ("v3", _load("home_gate.json"))]
    rows_html, chart = [], {}
    for tag, g in gates:
        if not g:
            continue
        for day, v in g.items():
            if day.startswith("_"):
                continue
            for name in ("schematic", "room_model"):
                if name in v:
                    rows_html.append(f"<tr><td>{tag}</td><td>{day}</td><td>{name.replace('_', ' ')}</td><td>{v[name]['inside']}</td>"
                                     f"<td>{v[name]['mean_error_in_tolerances']}</td><td>{v[name].get('range_width_in_tolerances', '')}</td></tr>")
                    chart.setdefault((tag, name), []).append(v[name]["mean_error_in_tolerances"])
    fig, ax = plt.subplots(figsize=(7, 3.2))
    labels = sorted({k[0] for k in chart}, key=lambda s: s)
    for j, name in enumerate(("schematic", "room_model")):
        vals = [np.mean(chart.get((l, name), [np.nan])) for l in labels]
        ax.bar(np.arange(len(labels)) + j * 0.38, vals, 0.36, label=name.replace("_", " "), color=[MUTED, DOG][j])
    ax.set_xticks(np.arange(len(labels)) + 0.19); ax.set_xticklabels(labels); ax.set_ylabel("mean error, tolerances (lower is better)"); ax.legend(facecolor=BG)
    ax.set_title("Step 5 · Held-out days: which simulator predicts reality")
    S.append(("5. Simulator against reality, on days it never saw", f"<img src='{save(fig, out, 's5_gate.png')}'>"
              "<table><tr><th>run</th><th>held-out day</th><th>simulator</th><th>real metrics inside range</th><th>mean error</th><th>range width</th></tr>"
              + "".join(rows_html) + "</table><p>Each simulator is fitted and calibrated on the other days only, runs the real controller "
              "with 8 new seeds, and predicts the held-out day's logged statistics (in-view share, losses, re-finds, chases, searches, aims). "
              "Mean error is in calibration tolerances; 'inside' alone rewards a vague simulator (a wide range), so read it with the width. "
              "v1: first version. v2: her gait from video and in-place shifts; it predicted many short losses the real data does not have, "
              "because detector misses were independent per frame. v3: a pose the detector cannot see stays unseen until she moves (both simulators).</p>"))

    # 6. synthetic views against real snapshots
    svc = None
    p = os.path.join(world, "raw", "synthview_check.json")
    if os.path.exists(p):
        svc = json.load(open(p))
        fig, ax = plt.subplots(figsize=(7, 3))
        ax.hist([r["shift_px"] for r in svc["rows"]], bins=20, range=(0, 45), color=DOG, alpha=0.9, label="same pose")
        ax.hist([r["null_shift_px"] for r in svc["rows"]], bins=20, range=(0, 45), color=MUTED, alpha=0.6, label="a different pose (null)")
        ax.set_xlabel("misalignment, px of a 640-wide frame"); ax.legend(facecolor=BG); ax.set_title("Step 6 · Synthetic views line up with real snapshots")
        S.append(("6. Synthetic camera views", f"<img src='{save(fig, out, 's6_synthview.png')}'>"
                  f"<p>Every pixel of a synthetic view is a ray of the calibrated camera at that pose, coloured by the room model's panorama. "
                  f"Against {svc['n']} real snapshots at tilts the panorama covers: median misalignment {svc['median_shift_px']} px, "
                  f"versus {svc['null_median_shift_px']} px when the render is of a different pose. The worst tenth "
                  f"({svc['p90_shift_px']} px) are where the panorama is patchy. Lighting is the daytime sweep's, not the hour's.</p>"))

    # 7. Follow v2 parts
    # 7a hidden map
    fig, ax = plt.subplots(figsize=(7, 6)); floor_plan(ax, R, fh)
    xs = np.arange(-4, 4.01, 0.1); zs = np.arange(-3, 7.01, 0.1)
    def kind(x, z):                                       # 0 visible, 1 behind furniture, 2 beyond a wall
        if not rf.hidden(x, z):
            return 0
        # furniture is low: a line to the same spot 1.3 m above the floor clears it; a wall blocks that line too
        p = np.array([x, -fh + 1.3, z]); r = float(np.linalg.norm(p)); rr = rf.range_at(p / r)
        return 1 if (rr <= 0 or rr >= r * 0.92) else 2
    hid = np.array([[kind(x, z) for x in xs] for z in zs], float)
    from matplotlib.colors import ListedColormap
    ax.imshow(np.ma.masked_where(hid < 1, hid), extent=[xs[0], xs[-1], zs[0], zs[-1]], origin="lower",
              cmap=ListedColormap([WARN, "#3b1f25"]), vmin=1, vmax=2, alpha=0.55)
    ax.scatter(rows[:, 1], rows[:, 2], s=2, c=DOG, alpha=0.4, lw=0)
    ax.set_xlim(-4, 4); ax.set_ylim(-3, 7)
    ax.set_title("Step 7a · Hidden from every pose: behind furniture (orange), beyond a wall (dark); her sightings (green)", fontsize=9)
    behind = int((hid == 1).sum())
    seen_hidden = float(np.mean([rf.hidden(x, z) for x, z in rows[:, 1:3]]))
    S.append(("7a. Follow v2: areas hidden from every pose", f"<img src='{save(fig, out, 's7a_hidden.png')}'>"
              f"<p>The camera only rotates, so a floor spot is hidden from every pose exactly when the measured room blocks the line from "
              f"the lens to her body (centre and top). <b>Self-check:</b> {seen_hidden:.1%} of the places she was actually SEEN count as hidden "
              f"(should be near 0: where it is not, the room model is wrong there, and Follow ignores 'hidden' for that sighting). "
              f"{behind * 0.01:.1f} m\u00b2 of floor is behind furniture: those are the spots Follow should wait outside of.</p>"))
    # 7b prediction
    pt = _load("prediction_test.json")
    if pt and pt["rows"]:
        fig, ax = plt.subplots(figsize=(7, 3.2))
        for k, c in (("none", MUTED), ("pixel", ACC), ("metric", DOG)):
            v = np.sort([r[k] for r in pt["rows"]]); ax.plot(v, np.linspace(0, 1, len(v)), c=c, label=f"{k}: median {np.median(v):.3f}")
        ax.set_xlabel(f"error of where she'll be in {pt['lead_s']:.0f} s, frame widths"); ax.set_ylabel("share of moments"); ax.legend(facecolor=BG)
        ax.set_title("Step 7b · Predicting her position ahead: metres vs pixels vs no prediction")
        d = np.array([r["pixel"] - r["metric"] for r in pt["rows"]]); rng = np.random.default_rng(0)
        bs = [np.median(rng.choice(d, len(d))) for _ in range(2000)]
        S.append(("7b. Follow v2: predicting where she will be", f"<img src='{save(fig, out, 's7b_predict.png')}'>"
                  f"<p>{len(pt['rows'])} moments while she walks, in simulated hours where the true future position is known. Pixel minus "
                  f"metric error: median {np.median(d):+.3f} frame widths (95% CI {np.percentile(bs, 2.5):+.3f} to {np.percentile(bs, 97.5):+.3f}). "
                  "Her walks are short and stop-start, so predicting ahead buys little over aiming at her last box.</p>"))
    # 7c framing by size
    fig, axs = plt.subplots(1, 2, figsize=(12, 3.2))
    dist = np.hypot(rows[:, 1], rows[:, 2])
    far = rows[np.argmin(np.abs(dist - 2.4) + 10 * np.array([rf.hidden(x, z) for x, z in rows[:, 1:3]]))]   # a real sighting ~2.4 m away
    for ax, (x, z) in zip(axs, ((0.35, 0.9), (float(far[1]), float(far[2])))):
        pan = 1850 - math.degrees(math.atan2(x, z)) / doc["calib"]["deg_per_step_pan"]
        rm = HomeRoom(seed=1, params=P, free=free, home=data); rm.heading = 0.0
        tilt, first_seen = None, None
        for tl in range(760, 300, -10):                     # the tilt where her box is cut at the bottom, else any tilt that sees her
            seen, box, _c = rm._view(pan, tl, x, z, "move")
            if seen and box and first_seen is None:
                first_seen = tl
            if seen and box and box[3] > 0.97:
                tilt = tl
                break
        tilt = tilt or first_seen or 640
        img = sv_render(ctx, pano, pan, tilt); ax.imshow(img); ax.set_xticks([]); ax.set_yticks([])
        seen, box, _c = rm._view(pan, tilt, x, z, "move")
        if box and seen:
            ax.add_patch(plt.Rectangle((box[0] * 640, box[1] * 360), (box[2] - box[0]) * 640, (box[3] - box[1]) * 360, fill=False, ec=WARN, lw=2))
            bc = ((box[0] + box[2]) / 2 * 640, (box[1] + box[3]) / 2 * 360); ax.plot(*bc, "x", c=WARN, ms=10, mew=2)
        o = rf.offset(pan, tilt, rf.body_centre(x, z))
        if o:
            ax.plot((o[0] + 0.5) * 640, (o[1] + 0.5) * 360, "+", c=DOG, ms=14, mew=2)
        ax.set_xlim(0, 640); ax.set_ylim(420, 0)
        ax.set_title(f"dog {math.hypot(x, z):.1f} m away, tilt {tilt}: box centre (orange x) vs body centre (green +)", fontsize=9)
    S.append(("7c. Follow v2: framing her by her real size", f"<img src='{save(fig, out, 's7c_framing.png')}'>"
              "<p>A dog close to the camera is cut off by the frame's bottom edge, so her box centre (orange) sits high on her body and an "
              "aim at it keeps her cropped. From her distance and height (0.55 m) the body centre (green) is known even when the box is cut. "
              "<b>Check:</b> the green cross should sit at the middle of her body.</p>"))
    # 7d controller comparison
    fe = _load("follow_eval.json")
    if fe:
        fig, axs = plt.subplots(1, 3, figsize=(13, 3))
        for ax, k, lab in zip(axs, ("true_in_view", "moves_per_hour", "losses_per_home_hour"), ("truly in view (higher is better)", "motor moves / h (lower)", "losses / home h (lower)")):
            a = [r[k] for r in fe["rows"]["today"]]; b = [r[k] for r in fe["rows"]["room_aware"]]
            for p_, q_ in zip(a, b):
                ax.plot([0, 1], [p_, q_], c=MUTED, lw=0.8)
            ax.plot([0, 1], [np.mean(a), np.mean(b)], "-o", c=DOG, lw=2)
            ax.set_xticks([0, 1]); ax.set_xticklabels(["today", "room-aware"]); ax.set_title(lab, fontsize=9)
        d = fe["room_aware_minus_today"]
        S.append(("7d. Follow v2 as a whole, in the simulator", f"<img src='{save(fig, out, 's7d_controller.png')}'>"
                  f"<p>The same seeds and start hours for both (grey lines pair them). Room-aware minus today: truly in view "
                  f"{d['true_in_view']['mean']:+.3f} ({d['true_in_view']['ci95'][0]:+.3f} to {d['true_in_view']['ci95'][1]:+.3f}), motor moves "
                  f"{d['moves_per_hour']['mean']:+.1f}/h, losses {d['losses_per_home_hour']['mean']:+.1f}/h, over {fe['seeds']} x {fe['hours']} h. "
                  "Not better yet: that is the next work, tuned against these exact measures.</p>"))
    page = ("<!doctype html><meta charset=utf-8><meta name=viewport content='width=device-width'><title>Room simulator report</title>"
            f"<style>body{{background:{BG};color:{FG};font:15px/1.5 -apple-system,Segoe UI,sans-serif;max-width:1100px;margin:0 auto;padding:16px}}"
            f"h1{{font-size:24px}}h2{{font-size:18px;margin-top:34px;border-top:1px solid #2b3340;padding-top:18px}}img{{max-width:100%;border-radius:8px}}"
            f"p{{color:{MUTED}}}b{{color:{FG}}}code{{color:{FG}}}table{{border-collapse:collapse;font-size:13px;margin:10px 0}}td,th{{border-bottom:1px solid #2b3340;padding:4px 10px;text-align:left}}</style>"
            "<h1>The room-model simulator and Follow v2, step by step</h1><p>Local only: this page shows the inside of your home.</p>"
            + "".join(f"<h2>{html.escape(t)}</h2>{b}" for t, b in S))
    open(os.path.join(out, "index.html"), "w").write(page)
    return os.path.join(out, "index.html")


if __name__ == "__main__":
    print(main(sys.argv[1]))
