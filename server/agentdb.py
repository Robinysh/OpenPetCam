#!/usr/bin/env python3
"""The super-agent's shared state, kept on the always-on server (state separated from compute: the agent's
brain runs on the owner's computer and may sleep; triggers keep arriving here and wait). One SQLite file,
/var/lib/petcam/agent.db. The owner's computer talks to it over SSH:

  agentdb.py <command> < json-on-stdin   ->  json on stdout

Objects (each has an ID so sessions can recall context by reference, "Chain of Recall"):
  trigger  an event from the environment: a report, a chat message, a goal miss, a QA failure, a tick
  thread   a stream of related triggers (one report, one goal), owned by at most one worker
  log      what the router, workers and watchdog decided or did, per thread
  metric   one measurement of a goal (value, target, ok)
  beat     heartbeats of the agent's own loops (router, workers, watchdog, collector)
"""
import json
import math
import os
import re
import sqlite3
import sys
import time

DB = os.environ.get("PETCAM_AGENT_DB", "/var/lib/petcam/agent.db")
SCHEMA = """
CREATE TABLE IF NOT EXISTS triggers (id INTEGER PRIMARY KEY, ts REAL, source TEXT, thread TEXT, kind TEXT,
  summary TEXT, payload TEXT, dedupe TEXT UNIQUE, status TEXT DEFAULT 'new', routed_to TEXT, handled_ts REAL);
CREATE INDEX IF NOT EXISTS triggers_status ON triggers(status, ts);
CREATE TABLE IF NOT EXISTS threads (id TEXT PRIMARY KEY, title TEXT, kind TEXT, status TEXT DEFAULT 'open',
  worker TEXT, session TEXT, created REAL, updated REAL, summary TEXT);
CREATE TABLE IF NOT EXISTS log (id INTEGER PRIMARY KEY, ts REAL, actor TEXT, thread TEXT, kind TEXT, text TEXT);
CREATE INDEX IF NOT EXISTS log_ts ON log(ts);
CREATE TABLE IF NOT EXISTS metrics (ts REAL, goal TEXT, value REAL, target TEXT, ok INTEGER, detail TEXT);
CREATE INDEX IF NOT EXISTS metrics_goal ON metrics(goal, ts);
CREATE TABLE IF NOT EXISTS beats (name TEXT PRIMARY KEY, ts REAL, detail TEXT);
CREATE TABLE IF NOT EXISTS goal_catalog (id TEXT PRIMARY KEY, definition TEXT NOT NULL, updated REAL NOT NULL);
CREATE TABLE IF NOT EXISTS goal_catalog_state (id INTEGER PRIMARY KEY CHECK(id=1), updated REAL NOT NULL);
"""
SPILL = 400          # characters inlined per text field; the rest is recalled by ID ("Inline & Spill")


def db():
    c = sqlite3.connect(DB, timeout=10)
    c.row_factory = sqlite3.Row
    c.executescript(SCHEMA)
    return c


def rows(cur):
    return [dict(r) for r in cur]


def spill(d, keep=("payload",)):
    out = dict(d)
    for k, v in d.items():
        if isinstance(v, str) and len(v) > SPILL:
            out[k] = v[:SPILL] + f"… [{len(v) - SPILL} more chars: recall {d.get('id', '')}]"
    return out


def cmd_add_trigger(c, a):
    """Fast path dedupe: the same dedupe key is stored once. Returns the trigger id (new or existing)."""
    now = time.time()
    status = a.get('status', 'new')
    if status not in ('new', 'review'):
        raise ValueError('new triggers must be new or review')
    try:
        cur = c.execute("INSERT INTO triggers(ts,source,thread,kind,summary,payload,dedupe,status) VALUES(?,?,?,?,?,?,?,?)",
                        (a.get("ts", now), a["source"], a.get("thread"), a.get("kind", ""), a.get("summary", "")[:2000],
                         json.dumps(a.get("payload"))[:20000], a.get("dedupe"), status))
        tid, new = cur.lastrowid, True
    except sqlite3.IntegrityError:
        tid, new = c.execute("SELECT id FROM triggers WHERE dedupe=?", (a.get("dedupe"),)).fetchone()[0], False
    if a.get("thread") and new:
        c.execute("INSERT INTO threads(id,title,kind,created,updated) VALUES(?,?,?,?,?) ON CONFLICT(id) DO UPDATE SET "
                  "updated=excluded.updated, status=CASE WHEN threads.status='closed' THEN 'open' ELSE threads.status END",
                  (a["thread"], a.get("title", a.get("summary", "")[:80]), a.get("kind", ""), now, now))
    c.commit()
    return {"id": tid, "new": new}


def cmd_pending(c, a):
    return rows(c.execute("SELECT t.*, th.worker AS subscriber FROM triggers t LEFT JOIN threads th ON th.id=t.thread "
                          "WHERE t.status='new' ORDER BY t.ts LIMIT ?", (a.get("limit", 50),)))


def cmd_mark(c, a):
    c.executemany("UPDATE triggers SET status=?, routed_to=?, handled_ts=? WHERE id=?",
                  [(a["status"], a.get("routed_to"), time.time(), i) for i in a["ids"]])
    c.commit()
    return {"ok": True}


def cmd_resolve_reviews(c, a):
    """Retire historical review entries only after component evidence is complete."""
    threads = list(dict.fromkeys(a.get('threads', [])))
    if any(not isinstance(thread, str) or not thread.startswith('report:') for thread in threads):
        raise ValueError('review resolution accepts report threads only')
    resolved = 0
    for thread in threads:
        cur = c.execute("UPDATE triggers SET status='handled', handled_ts=? "
                        "WHERE thread=? AND kind='report_review' AND status='review'", (time.time(), thread))
        resolved += cur.rowcount
        # Do not interfere with a live worker or newly filed follow-up work.
        c.execute("UPDATE threads SET status='closed', updated=? WHERE id=? AND status!='closed' AND COALESCE(worker,'')='' "
                  "AND NOT EXISTS (SELECT 1 FROM triggers WHERE thread=? AND status='new')",
                  (time.time(), thread, thread))
    c.commit()
    return {'resolved': resolved}


def cmd_thread(c, a):
    sets, vals = [], []
    for k in ("title", "kind", "status", "worker", "session", "summary"):
        if k in a:
            sets.append(f"{k}=?")
            vals.append(a[k])
    c.execute("INSERT OR IGNORE INTO threads(id,title,created,updated) VALUES(?,?,?,?)", (a["id"], a.get("title", a["id"]), time.time(), time.time()))
    if sets:
        c.execute(f"UPDATE threads SET {', '.join(sets)}, updated=? WHERE id=?", (*vals, time.time(), a["id"]))
    c.commit()
    return dict(c.execute("SELECT * FROM threads WHERE id=?", (a["id"],)).fetchone())


def cmd_log(c, a):
    c.execute("INSERT INTO log(ts,actor,thread,kind,text) VALUES(?,?,?,?,?)",
              (time.time(), a["actor"], a.get("thread"), a.get("kind", "note"), a["text"][:4000]))
    c.commit()
    return {"ok": True}


def _goal_definitions(value):
    """Validate an entire owner-authored catalog before changing its snapshot."""
    if not isinstance(value, list) or len(value) > 200:
        raise ValueError('goals must be a list of at most 200 definitions')
    out, seen = [], set()
    for item in value:
        if not isinstance(item, dict) or not isinstance(item.get('id'), str) \
                or not re.fullmatch(r'[a-z0-9][a-z0-9_.-]{0,79}', item['id']) or item['id'] in seen:
            raise ValueError('goal identifiers must be unique and valid')
        definition = {'id': item['id']}
        for field in ('what', 'metric', 'target', 'ground_truth'):
            text = item.get(field)
            if not isinstance(text, str) or not text.strip() or len(text) > 4000:
                raise ValueError('goal definitions need bounded title, metric, target and source text')
            definition[field] = text
        note = item.get('note', '')
        if not isinstance(note, str) or len(note) > 8000:
            raise ValueError('goal note must be bounded text')
        definition['note'] = note
        out.append(definition)
        seen.add(item['id'])
    return out


def _save_goal_catalog(c, definitions):
    c.execute('DELETE FROM goal_catalog')
    c.executemany('INSERT INTO goal_catalog(id,definition,updated) VALUES(?,?,?)',
                  [(item['id'], json.dumps(item, sort_keys=True, allow_nan=False), time.time()) for item in definitions])
    c.execute('INSERT INTO goal_catalog_state(id,updated) VALUES(1,?) ON CONFLICT(id) DO UPDATE SET updated=excluded.updated',
              (time.time(),))


def cmd_goal_catalog(c, a):
    """Replace only goal descriptions; no metrics, thresholds, threads or triggers are changed."""
    definitions = _goal_definitions(a.get('goals'))
    with c:
        _save_goal_catalog(c, definitions)
    return {'goals': len(definitions)}


def _metric_detail(metric):
    detail = metric.get('detail')
    observed = metric.get('observed_at')
    if isinstance(observed, (int, float)) and not isinstance(observed, bool) and math.isfinite(observed):
        # Keep provenance parseable even when a producer emits a long detail.
        text = detail if isinstance(detail, str) else json.dumps(detail, allow_nan=False)
        text = text[:1700]
        while True:
            encoded = json.dumps({'measurement': text, 'observed_at': observed}, allow_nan=False)
            if len(encoded) <= 2000:
                return encoded
            text = text[:len(text) // 2]
    return json.dumps(detail, allow_nan=False)[:2000]


def cmd_metric(c, a):
    definitions = _goal_definitions(a['goals']) if 'goals' in a else None
    values = [(time.time(), m['goal'], m.get('value'), str(m.get('target', '')),
               int(bool(m.get('ok'))), _metric_detail(m)) for m in a['metrics']]
    # Catalog and samples belong to the same producer snapshot. Older callers
    # without a catalog remain compatible and cannot erase its definitions.
    with c:
        if definitions is not None:
            _save_goal_catalog(c, definitions)
        c.executemany("INSERT INTO metrics(ts,goal,value,target,ok,detail) VALUES(?,?,?,?,?,?)", values)
        c.execute("DELETE FROM metrics WHERE ts < ?", (time.time() - 90 * 86400,))
    return {"ok": True}


def cmd_beat(c, a):
    c.execute("INSERT INTO beats(name,ts,detail) VALUES(?,?,?) ON CONFLICT(name) DO UPDATE SET ts=excluded.ts, detail=excluded.detail",
              (a["name"], time.time(), json.dumps(a.get("detail"))[:2000]))
    c.commit()
    return {"ok": True}


def cmd_recall(c, a):
    """Chain of Recall: an object by ID with its neighbours' IDs and summaries inlined (one hop)."""
    ref = str(a["ref"])
    if ref.isdigit():
        t = c.execute("SELECT * FROM triggers WHERE id=?", (int(ref),)).fetchone()
        if not t:
            return {"error": "no such trigger"}
        t = dict(t)
        t["thread_obj"] = spill(dict(c.execute("SELECT * FROM threads WHERE id=?", (t["thread"],)).fetchone() or {}))
        return t
    th = c.execute("SELECT * FROM threads WHERE id=?", (ref,)).fetchone()
    if not th:
        return {"error": "no such thread"}
    out = dict(th)
    out["triggers"] = [spill(r) for r in rows(c.execute("SELECT id,ts,kind,summary,status FROM triggers WHERE thread=? ORDER BY ts DESC LIMIT 20", (ref,)))]
    out["log"] = [spill(r) for r in rows(c.execute("SELECT ts,actor,kind,text FROM log WHERE thread=? ORDER BY ts DESC LIMIT 20", (ref,)))]
    return out


def cmd_digest(c, a):
    """What the router keeps 'in mind': open threads (inlined) and recent decisions; the rest by recall."""
    return {"open_threads": [spill(r) for r in rows(c.execute(
                "SELECT id,title,kind,status,worker,summary,updated FROM threads WHERE status!='closed' ORDER BY updated DESC LIMIT 30"))],
            "recent_log": [spill(r) for r in rows(c.execute("SELECT ts,actor,thread,kind,text FROM log ORDER BY ts DESC LIMIT 15"))],
            "goals": latest_goals(c)}


def latest_goals(c):
    return rows(c.execute("SELECT m.goal,m.value,m.target,m.ok,m.ts,m.detail FROM (SELECT DISTINCT goal FROM metrics) g "
                          "JOIN metrics m ON m.rowid=(SELECT n.rowid FROM metrics n WHERE n.goal=g.goal "
                          "ORDER BY n.ts DESC,n.rowid DESC LIMIT 1) "
                          "ORDER BY m.goal"))


def cmd_dashboard(c, a):
    since = time.time() - 7 * 86400
    series = {}
    for r in c.execute("SELECT goal, ts, value, ok FROM metrics WHERE ts > ? ORDER BY ts", (since,)):
        series.setdefault(r["goal"], []).append([round(r["ts"]), r["value"], r["ok"]])
    return {"now": time.time(), "goals": latest_goals(c), "series": series,
            "threads": rows(c.execute("SELECT id,title,kind,status,worker,summary,created,updated FROM threads ORDER BY updated DESC LIMIT 40")),
            "log": rows(c.execute("SELECT ts,actor,thread,kind,text FROM log ORDER BY ts DESC LIMIT 60")),
            "beats": rows(c.execute("SELECT name,ts,detail FROM beats")),
            "queue": c.execute("SELECT COUNT(*) FROM triggers WHERE status='new'").fetchone()[0],
            "review_queue": c.execute("SELECT COUNT(*) FROM triggers WHERE status='review'").fetchone()[0],
            "triggers_24h": rows(c.execute("SELECT kind, status, COUNT(*) n FROM triggers WHERE ts > ? GROUP BY kind, status", (time.time() - 86400,)))}


COMMANDS = {k[4:]: v for k, v in globals().items() if k.startswith("cmd_")}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(json.dumps({"error": f"commands: {sorted(COMMANDS)}"}))
        sys.exit(2)
    arg = json.loads(sys.stdin.read() or "{}")
    with db() as conn:
        print(json.dumps(COMMANDS[sys.argv[1]](conn, arg), default=str))
