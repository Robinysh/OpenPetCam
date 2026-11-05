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
from clipmedia import browser_copy, probe, write_receipt  # noqa: E402


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
            if source.name.endswith(".browser.mp4"):
                continue
            try:
                media = probe(source)
                audio = [s["codec_name"] for s in media["streams"] if s["codec_type"] == "audio"]
                if not audio or all(codec == "aac" for codec in audio):
                    continue
                result["attempted"] += 1
                output = source.with_name(source.stem + ".browser.mp4")
                item = {"clip": rel, "browser_clip": str(output.relative_to(root)), "state": "planned"}
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
                if not any(s["codec_type"] == "audio" and s["codec_name"] == "aac" for s in checked["streams"]):
                    raise ValueError("browser output has no AAC audio")
                write_receipt(receipt_path, {"source_sha256": before, "browser_sha256": sha(output),
                                            "source_file": source.name, "browser_file": output.name})
                # Compare the old pointer as well: don't replace an independently updated event.
                count = con.execute("UPDATE events SET clip=? WHERE clip=?", (item["browser_clip"], rel)).rowcount
                con.commit()
                item.update(state="converted", events_updated=count, source_sha256=before, browser_sha256=sha(output))
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
    if args.execute and any(item["state"] != "converted" for item in result["items"]):
        sys.exit(1)
