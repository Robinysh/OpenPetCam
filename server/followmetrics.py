"""Observational Follow measures; a recent sighting is not proof the pet is home."""
import json
import math
import time


def status_age(snapshot, measured_at=None):
    """Call after reading the snapshot; invalid/future clocks are unavailable."""
    measured_at = time.time() if measured_at is None else measured_at
    try:
        stamp = float(snapshot["t"])
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(stamp) or stamp > measured_at + 1:
        return None
    return round(max(0., measured_at - stamp), 3)


def retention(obs_rows, track_rows, *, recent_s=1800., max_gap_s=6.):
    """Time-weighted visibility during logged Follow, within 30 min of a sighting.

    Long telemetry gaps and periods without Follow records are excluded. Detector
    scores may repeat between detector runs; this is operational evidence, not
    independently labeled at-home time or a measure of distress.
    """
    active = {float(r["ts"]) for r in track_rows}
    rows = sorted(obs_rows, key=lambda r: r["ts"])
    seen = None
    eligible = visible = logged = 0.
    def detected(row):
        try:
            box = json.loads(row["box"]) if isinstance(row["box"], str) else row["box"]
            return (float(row["score"] or 0) >= .15 and isinstance(box, (tuple, list)) and len(box) == 4
                    and all(isinstance(v, (int, float)) and math.isfinite(v) for v in box)
                    and 0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1)
        except (ValueError, TypeError, KeyError):
            return False
    for row, nxt in zip(rows, rows[1:]):
        t, end = float(row["ts"]), float(nxt["ts"])
        yes = detected(row)
        if yes:
            seen = t
        dt = end - t
        if t not in active or end not in active or not 0 < dt <= max_gap_s:
            continue
        logged += dt
        if seen is not None and t - seen < recent_s:
            dt = min(dt, max(0., seen + recent_s - t))
            eligible += dt
            if yes:
                visible += dt
    return {"visible_percent": round(100 * visible / eligible, 2) if eligible else None,
            "visible_seconds": round(visible, 2), "eligible_seconds": round(eligible, 2),
            "logged_follow_seconds": round(logged, 2), "recent_sighting_window_s": recent_s,
            "scope": "logged Follow intervals within 30 min of a detector sighting; not known at-home time",
            "independent_ground_truth": False}
