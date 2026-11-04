import sys, json, http.client, time
sys.path.insert(0, "/opt/petcam")
import app
app.init_auth_db()
tok = app.new_device("claude-test", "127.0.0.1", "test")
H = {"Cookie": f"petcam_device={tok}", "X-Forwarded-For": "127.0.0.1"}
PUB = app.PUBLIC_HOST


def req(method, path, body=None, headers=None, public=False):
    c = http.client.HTTPSConnection(PUB, 443, timeout=20) if public else http.client.HTTPConnection("127.0.0.1", 8080, timeout=20)
    h = dict(H if headers is None else headers)
    data = None
    if body is not None:
        data = json.dumps(body).encode()
        h["Content-Type"] = "application/json"
    c.request(method, path, body=data, headers=h)
    r = c.getresponse()
    b = r.read()
    return r.status, r.getheader("Set-Cookie"), b


try:
    for p in ("/api/live", "/api/day", "/api/diary", "/api/devices", "/api/state"):
        s, _, b = req("GET", p)
        print(f"{p:<15} {s} {b[:140].decode(errors='replace')}")
    print("no cookie live ", req("GET", "/api/live", headers={})[0])
    s, ck, _ = req("GET", "/")
    print("index          ", s, "cookie refreshed:", bool(ck and "Max-Age=34560000" in ck))
    s, _, b = req("POST", "/api/invites", {"name": "Test Guest"})
    url = json.loads(b)["url"]
    path = "/join/" + url.split("/join/")[1]
    print("invite         ", s, url.split("/join/")[0] + "/join/<token>")
    s, _, b = req("GET", path, headers={})
    print("join page      ", s, b"Join " in b)
    s, ck, _ = req("POST", path, headers={"User-Agent": "Mozilla iPhone"})
    print("join accept    ", s, "device cookie set:", bool(ck and ck.startswith("petcam_device=")))
    print("join reuse     ", req("POST", path, headers={})[0], "(410 = single use)")
    snap = json.loads(req("GET", "/api/snap?ts=%d" % (time.time() - 120))[2])["snap"]
    print("snap path      ", snap)
    s, _, b = req("GET", snap, public=True)
    print("media + cookie ", s, len(b), "bytes")
    print("media no cookie", req("GET", snap, headers={}, public=True)[0])
    print("db via /media  ", req("GET", "/media/petcam.db", public=True)[0])
    print("live video auth", req("GET", "/cam/", headers={}, public=True)[0], "(401 without cookie)")
finally:
    con = app.dbc()
    con.execute("UPDATE devices SET revoked=1 WHERE name IN ('claude-test', 'Test Guest''s iPhone')")
    print("active devices ", [r[0] for r in con.execute("SELECT name FROM devices WHERE revoked=0")])
