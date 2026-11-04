"""Regression: auth must not carry over between requests on one keep-alive connection."""
import sys, http.client
sys.path.insert(0, "/opt/petcam")
import app
tok = app.new_device("claude-leaktest", "127.0.0.1", "test")
try:
    c = http.client.HTTPConnection("127.0.0.1", 8080, timeout=10)
    for label, hdrs in (("signed in", {"Cookie": f"petcam_device={tok}"}), ("anonymous", {}), ("signed in", {"Cookie": f"petcam_device={tok}"}), ("anonymous", {})):
        c.request("GET", "/api/live", headers=hdrs); r = c.getresponse(); r.read()
        print(f"same connection, {label:<9} -> {r.status}")
finally:
    con = app.dbc(); con.execute("UPDATE devices SET revoked=1 WHERE name='claude-leaktest'")
