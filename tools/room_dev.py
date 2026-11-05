#!/usr/bin/env python3
"""Preview the app's Room tab on this computer, without the server: serves the real index.html (placeholders filled
from config/site.env), room.js and the vendored three.js, and answers /api/room* from your local room bundle
(models/room, from sim/world/app_bundle.py) and the watcher records the room model pulled (eval/<world>/obs.json,
track.json). Other /api calls get an empty answer. Listens on 127.0.0.1 only; nothing leaves the machine.
    python3 tools/room_dev.py eval/world [8790]    -> http://127.0.0.1:8790/#room"""
import json
import os
import sqlite3
import sys
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SERVER = os.path.join(ROOT, "server")
os.environ.setdefault("PETCAM_ROOM_DIR", os.path.join(ROOT, "models", "room"))
os.environ.setdefault("PETCAM_SITE_ENV", os.path.join(ROOT, "config", "site.env"))
sys.path.insert(0, SERVER)
import petconf  # noqa: E402
import roomapi  # noqa: E402


def load_db(world):
    con = sqlite3.connect(":memory:", check_same_thread=False)
    con.executescript("""CREATE TABLE obs(ts REAL, score REAL, person REAL, box TEXT, motion REAL, present INT, moving INT);
                         CREATE TABLE track(ts REAL, pan INT, tilt INT, act TEXT, weak INT);""")
    for o in json.load(open(os.path.join(world, "obs.json"))):
        con.execute("INSERT INTO obs VALUES(?,?,?,?,?,?,?)", (o["ts"], o.get("score"), 0, o["box"], 0, o["present"], o.get("moving")))
    for t in json.load(open(os.path.join(world, "track.json"))):
        con.execute("INSERT INTO track VALUES(?,?,?,?,?)", (t["ts"], t["pan"], t["tilt"], "", 0))
    return con


class H(BaseHTTPRequestHandler):
    def send(self, code, body, ctype="application/json"):
        body = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):
        pass

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        p, q = u.path, dict(urllib.parse.parse_qsl(u.query))
        if p == "/":
            body = open(os.path.join(SERVER, "index.html"), "rb").read()
            for k, v in (petconf.template_vars() | {"BUILD": "dev"}).items():
                body = body.replace(f"{{{{{k}}}}}".encode(), v.encode())
            return self.send(200, body, "text/html; charset=utf-8")
        if p.endswith((".js", ".css", ".png", ".svg", ".webmanifest")) and ".." not in p:
            f = os.path.join(SERVER, p.lstrip("/"))
            if os.path.isfile(f):
                ct = {"js": "text/javascript", "css": "text/css", "png": "image/png", "svg": "image/svg+xml"}.get(p.rsplit(".", 1)[1], "application/octet-stream")
                return self.send(200, open(f, "rb").read(), ct)
        if p == "/api/room":
            return self.send(200, roomapi.info())
        if p == "/api/room/tex":
            f = roomapi.texture(q.get("f", ""))
            return self.send(200, open(f, "rb").read(), "image/png" if f.endswith(".png") else "image/jpeg") if f else self.send(404, b"no")
        if p == "/api/room/positions":
            # the preview's "now" is the end of the recorded data, so today shows the last recorded day
            until = min(float(q.get("until", END)), END)
            since = float(q.get("since", until - 86400))
            return self.send(200, {"since": since, "until": until, "rows": roomapi.positions(CON, since, until)})
        if p.startswith("/api/"):
            return self.send(200, {})
        return self.send(404, b"not found", "text/plain")


if __name__ == "__main__":
    a = sys.argv[1:]
    CON = load_db(a[0])
    END = CON.execute("SELECT max(ts) FROM obs").fetchone()[0]
    port = int(a[1]) if len(a) > 1 else 8790
    print(f"Room preview on http://127.0.0.1:{port}/ (records up to {END:.0f})")
    ThreadingHTTPServer(("127.0.0.1", port), H).serve_forever()
