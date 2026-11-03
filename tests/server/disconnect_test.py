"""A client that hangs up mid-request must not be logged as a crash; a real bug still must be.

Background (2026-11-02): tools/qa.py's "no crashes" check failed on 2 tracebacks in petcam-app,
both ending in `ConnectionResetError: [Errno 104] Connection reset by peer`. socketserver's default
handle_error prints a full traceback for every exception that escapes a handler, including a phone
or Caddy dropping the connection, which is normal and not a bug.

Runs locally: PETCAM_SITE_ENV=config/site.env python3 tests/server/disconnect_test.py
"""
import contextlib
import io
import os
import socket
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "server"))
import app  # noqa: E402

Server = getattr(app, "AppServer", ThreadingHTTPServer)


def stderr_of(exc):
    class H(BaseHTTPRequestHandler):
        def do_GET(self):
            raise exc

        def log_message(self, *a):
            pass

    srv = Server(("127.0.0.1", 0), H)
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        t = threading.Thread(target=srv.handle_request)
        t.start()
        with socket.create_connection(srv.server_address, timeout=5) as c:
            c.sendall(b"GET / HTTP/1.0\r\n\r\n")
            with contextlib.suppress(OSError):
                c.recv(100)
        t.join(5)
    srv.server_close()
    return buf.getvalue()


fails = 0
for exc in (ConnectionResetError(104, "Connection reset by peer"), BrokenPipeError(32, "Broken pipe")):
    out = stderr_of(exc)
    ok = "Traceback" not in out
    fails += not ok
    print(("PASS" if ok else "FAIL"), f"client disconnect {type(exc).__name__} is not logged as a crash", "" if ok else repr(out[-200:]))

out = stderr_of(ValueError("a real bug"))
ok = "Traceback" in out and "ValueError" in out
fails += not ok
print(("PASS" if ok else "FAIL"), "a real exception in a handler is still logged with its traceback")

src = open(app.__file__).read()
ok = "ThreadingHTTPServer(" not in src.replace("class AppServer(ThreadingHTTPServer)", "")
fails += not ok
print(("PASS" if ok else "FAIL"), "both of the app's HTTP servers use the quiet-disconnect server")

sys.exit(1 if fails else 0)
