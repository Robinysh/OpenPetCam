#!/usr/bin/env python3
"""Prepare checked browser clips, then switch matching event links; dry-run unless --execute.

Run on the server with its existing FFmpeg. Sources and prior outputs are never
overwritten. --limit and --deadline bound each explicitly requested batch.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "server"))
sys.path.insert(0, "/opt/petcam")
from clipmedia import audio_window_status, browser_copy, probe, validate_video_timeline, write_receipt  # noqa: E402

PARTIAL_AUDIO_WARNING = "Audio does not cover this recording's full video window. The original is retained."


def update_events(con, original, output, *, audio_partial):
    """Publish the pointer and its warning together, preserving current event detail."""
    # Take the writer lock only after media work, then read the latest details.
    # A different writer's replacement pointer is excluded by the old-clip match.
    con.execute("BEGIN IMMEDIATE")
    if not audio_partial:
        return con.execute("UPDATE events SET clip=? WHERE clip=?", (output, original)).rowcount
    updates = []
    for ident, raw in con.execute("SELECT rowid, detail FROM events WHERE clip=?", (original,)):
        detail = json.loads(raw or "{}")
        if not isinstance(detail, dict):
            raise ValueError("event detail must be an object to preserve clip warnings")
        warning = detail.get("clip_warning")
        if warning and not isinstance(warning, str):
            raise ValueError("existing clip warning must be text")
        # An existing warning can describe a separate video issue. Never clear
        # it or make a previously hidden clip playable through an audio repair.
        if warning:
            if PARTIAL_AUDIO_WARNING not in warning:
                detail["clip_warning"] = warning + " " + PARTIAL_AUDIO_WARNING
            detail.setdefault("clip_unusable", True)
        else:
            detail["clip_warning"] = PARTIAL_AUDIO_WARNING
            detail.setdefault("clip_unusable", False)
        updates.append((output, json.dumps(detail), ident, original))
    return con.executemany("UPDATE events SET clip=?, detail=? WHERE rowid=? AND clip=?", updates).rowcount


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def backfill(state, *, limit=5, deadline_s=120, execute=False, after=None):
    if not 1 <= limit <= 20 or not 1 <= deadline_s <= 600:
        raise ValueError("limit must be1..20 and deadline1..600 seconds")
    state = Path(state).resolve()
    root = state / "clips"
    con = sqlite3.connect((state / "petcam.db").as_uri() + ("?mode=rw" if execute else "?mode=ro"), uri=True)
    started = time.monotonic()
    result = {"execute": execute, "items": [], "scanned": 0, "attempted": 0, "limit": limit, "deadline_s": deadline_s,
              "deadline_semantics": "no new asset starts after budget; an in-flight probe/conversion remains separately bounded"}
    try:
        rows = con.execute("SELECT clip FROM events WHERE clip IS NOT NULL AND clip>? GROUP BY clip ORDER BY clip", (after or "",)).fetchall()
        last_scanned = None
        for (rel,) in rows:
            if result["attempted"] >= limit or time.monotonic() - started >= deadline_s or result["scanned"] >= 200:
                break
            result["scanned"] += 1
            last_scanned = rel
            source = (root / rel).resolve()
            if not source.is_relative_to(root) or not source.is_file():
                result["items"].append({"clip": rel, "state": "missing_or_invalid"})
                continue
            try:
                media = probe(source)
                source_timeline = validate_video_timeline(media)
                source_audio = audio_window_status(media)
                audio = [s["codec_name"] for s in media["streams"] if s["codec_type"] == "audio"]
                if source.name.endswith(".browser.mp4") and any(codec != "aac" for codec in audio):
                    raise ValueError("browser clip contains incompatible audio")
                if not audio or all(codec == "aac" for codec in audio):
                    if source_audio["audio_window_complete"] is False:
                        result["attempted"] += 1
                        item = {"clip": rel, "state": "planned_warning", "audio_partial": True, "source_audio": source_audio}
                        result["items"].append(item)
                        if execute:
                            count = update_events(con, rel, rel, audio_partial=True)
                            con.commit()
                            item.update(state="warning_updated", events_updated=count)
                    continue
                result["attempted"] += 1
                output = source.with_name(source.stem + ".browser.mp4")
                item = {"clip": rel, "browser_clip": str(output.relative_to(root)), "state": "planned",
                        "source_audio": source_audio, "audio_partial": source_audio["audio_window_complete"] is False}
                result["items"].append(item)
                if not execute:
                    continue
                before = sha(source)
                receipt_path = output.with_suffix(".json")
                if output.exists():
                    saved = json.loads(receipt_path.read_text())
                    if saved.get("source_sha256") != before or saved.get("browser_sha256") != sha(output):
                        raise ValueError("existing browser copy has no matching source/output provenance")
                    checked = probe(output)
                else:
                    checked = browser_copy(source, output)["output"]
                if sha(source) != before:
                    raise ValueError("source changed during conversion")
                output_timeline = validate_video_timeline(checked)
                output_audio = audio_window_status(checked)
                if abs(output_timeline["video_duration_s"] - source_timeline["video_duration_s"]) > .25:
                    raise ValueError("browser video duration differs from original")
                if not any(s["codec_type"] == "audio" and s["codec_name"] == "aac" for s in checked["streams"]):
                    raise ValueError("browser output has no AAC audio")
                write_receipt(receipt_path, {"source_sha256": before, "browser_sha256": sha(output),
                                            "source_file": source.name, "browser_file": output.name,
                                            "source_video_timeline": source_timeline, "browser_video_timeline": output_timeline,
                                            "source_audio": source_audio, "browser_audio": output_audio})
                partial = item["audio_partial"] or output_audio["audio_window_complete"] is False
                count = update_events(con, rel, item["browser_clip"], audio_partial=partial)
                con.commit()
                item.update(state="converted", events_updated=count, source_sha256=before, browser_sha256=sha(output),
                            output_audio=output_audio, audio_partial=partial)
            except Exception as exc:  # noqa: BLE001 - preserve source/pointers and continue this bounded batch
                con.rollback()
                if result["items"] and result["items"][-1].get("clip") == rel:
                    result["items"][-1].update(state="error", error=str(exc))
                else:
                    result["items"].append({"clip": rel, "state": "error", "error": str(exc)})
        result["next_after"] = last_scanned if result["scanned"] < len(rows) else None
    finally:
        con.close()
    result["elapsed_s"] = round(time.monotonic() - started, 3)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state", required=True, type=Path)
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--deadline", type=float, default=120)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--after", help="Continue after next_after from the preceding batch of the same mode")
    args = parser.parse_args()
    result = backfill(args.state, limit=args.limit, deadline_s=args.deadline, execute=args.execute, after=args.after)
    print(json.dumps(result, indent=2))
    if args.execute and any(item["state"] not in ("converted", "warning_updated") for item in result["items"]):
        sys.exit(1)
