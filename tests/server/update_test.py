#!/usr/bin/env python3
"""server/update.py against a fake GitHub, in a scratch root. Runs anywhere: python3 tests/server/update_test.py

Covers: a newer release installs and records itself; the same or an older release is left alone; a development
install (not an exact tag) is never overwritten; a bad checksum, a tag/archive mismatch and a path-escaping archive
are refused without touching the install; a release whose install fails, or that comes up unhealthy, is rolled
back to the previous source and never retried; UPDATE_CHANNEL=off only reports.
"""
import hashlib
import http.server
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
UPDATE = os.path.join(HERE, "..", "..", "server", "update.py")

# A stand-in for install.sh: installs VERSION and keeps a copy of its source, like the real one.
FAKE_INSTALL = r'''#!/usr/bin/env bash
set -e
SRC=$(cd "$(dirname "$0")" && pwd)
[ -f "$SRC/FAIL" ] && { echo "install failed on purpose"; exit 3; }
mkdir -p "$PETCAM_UPDATE_ROOT/opt/petcam"
cp "$SRC/VERSION" "$PETCAM_UPDATE_ROOT/opt/petcam/VERSION"
cp "$SRC/app.py" "$PETCAM_UPDATE_ROOT/opt/petcam/app.py"
if [ "$SRC" != "$PETCAM_UPDATE_ROOT/opt/petcam/source" ]; then
  rm -rf "$PETCAM_UPDATE_ROOT/opt/petcam/source"; mkdir -p "$PETCAM_UPDATE_ROOT/opt/petcam/source"
  (cd "$SRC" && tar --exclude=./site.env -cf - .) | tar -xf - -C "$PETCAM_UPDATE_ROOT/opt/petcam/source"
fi
[ -f "$SRC/UNHEALTHY" ] && touch "$PETCAM_UPDATE_ROOT/unhealthy" || rm -f "$PETCAM_UPDATE_ROOT/unhealthy"
'''


def archive(version, extra=None, evil=False, tag_in_file=None):
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as t:
        files = {"VERSION": (tag_in_file or version) + "\n", "install.sh": FAKE_INSTALL, "app.py": f"# app {version}\n"}
        files.update(extra or {})
        for name, body in files.items():
            info = tarfile.TarInfo(name)
            data = body.encode()
            info.size, info.mode = len(data), 0o755
            t.addfile(info, io.BytesIO(data))
        if evil:
            info = tarfile.TarInfo("../../escaped.txt")
            info.size = 3
            t.addfile(info, io.BytesIO(b"bad"))
    return buf.getvalue()


class Fake:
    """GitHub releases API + downloads + the app's /healthz, all on one local port."""
    def __init__(self, root):
        self.root, self.release, self.files = root, None, {}
        fake = self

        class H(http.server.BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_GET(self):
                if self.path == "/healthz":
                    bad = os.path.exists(os.path.join(fake.root, "unhealthy"))
                    self.send_response(500 if bad else 200); self.end_headers(); return
                if self.path.endswith("/releases/latest") and fake.release is None:
                    self.send_response(404); self.end_headers(); return
                if self.path.endswith("/releases/latest"):
                    body = json.dumps(fake.release).encode()
                elif self.path.startswith("/dl/") and self.path[4:] in fake.files:
                    body = fake.files[self.path[4:]]
                else:
                    self.send_response(404); self.end_headers(); return
                self.send_response(200); self.end_headers(); self.wfile.write(body)

        self.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.port = self.srv.server_address[1]
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()

    def publish(self, tag, data, sums_override=None):
        name = f"openpetcam-{tag.lstrip('v')}.tar.gz"
        self.files = {name: data, "SHA256SUMS": (sums_override or f"{hashlib.sha256(data).hexdigest()}  {name}\n").encode()}
        base = f"http://127.0.0.1:{self.port}/dl/"
        self.release = {"tag_name": tag, "body": f"notes for {tag}", "html_url": "https://example/rel",
                        "assets": [{"name": n, "browser_download_url": base + n} for n in self.files]}


def setup(root, version, channel="release"):
    for d in ("etc/petcam", "opt/petcam/source", "var/lib/petcam"):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    with open(os.path.join(root, "etc/petcam/site.env"), "w") as f:
        f.write(f"PET_NAME=Biscuit\nUPDATE_CHANNEL={channel}\n")
    for d in ("opt/petcam", "opt/petcam/source"):
        open(os.path.join(root, d, "VERSION"), "w").write(version + "\n")
        open(os.path.join(root, d, "app.py"), "w").write(f"# app {version}\n")
    with open(os.path.join(root, "opt/petcam/source/install.sh"), "w") as f:
        f.write(FAKE_INSTALL)


def run(root, fake, *args):
    env = dict(os.environ, PETCAM_UPDATE_ROOT=root, PETCAM_UPDATE_API=f"http://127.0.0.1:{fake.port}",
               PETCAM_HEALTH_URL=f"http://127.0.0.1:{fake.port}/healthz", PETCAM_HEALTH_WAIT="4")
    r = subprocess.run([sys.executable, UPDATE, *args], env=env, capture_output=True, text=True, timeout=120)
    st = json.load(open(os.path.join(root, "var/lib/petcam/update.json")))
    return r, st


def installed(root):
    return open(os.path.join(root, "opt/petcam/VERSION")).read().strip(), open(os.path.join(root, "opt/petcam/app.py")).read().strip()


fails = 0


def check(name, cond, extra=""):
    global fails
    print(("PASS " if cond else "FAIL ") + name + (f"  ({extra})" if extra and not cond else ""))
    fails += 0 if cond else 1


def case(version, channel="release"):
    root = tempfile.mkdtemp(prefix="upd-")
    setup(root, version, channel)
    return root, Fake(root)


# 1. A newer release installs.
root, f = case("v1.0.0"); f.publish("v1.1.0", archive("v1.1.0"))
r, st = run(root, f)
check("newer release installs", installed(root) == ("v1.1.0", "# app v1.1.0") and st["result"] == "updated", r.stdout + r.stderr)
check("source copy follows the install", open(os.path.join(root, "opt/petcam/source/VERSION")).read().strip() == "v1.1.0")
r, st = run(root, f)
check("then reports up to date", st["result"] == "up_to_date", st)

# 2. Same or older release: nothing happens.
root, f = case("v1.2.0"); f.publish("v1.1.0", archive("v1.1.0"))
r, st = run(root, f)
check("older release ignored", installed(root)[0] == "v1.2.0" and st["result"] == "up_to_date", st)

# 3. Development install is never overwritten.
for dev in ("v1.0.0-3-gabc1234", "v1.0.0-dirty", "abc1234", ""):
    root, f = case(dev); f.publish("v9.0.0", archive("v9.0.0"))
    r, st = run(root, f)
    check(f"development install '{dev}' left alone", installed(root)[0] == dev and st["result"] == "development", st)

# 4. Refusals that never touch the install.
root, f = case("v1.0.0"); f.publish("v1.1.0", archive("v1.1.0"), sums_override="0" * 64 + "  openpetcam-1.1.0.tar.gz\n")
r, st = run(root, f)
check("bad checksum refused", installed(root)[0] == "v1.0.0" and st["result"] == "error" and "checksum" in st["detail"], st)
root, f = case("v1.0.0"); f.publish("v1.1.0", archive("v1.1.0", tag_in_file="v6.6.6"))
r, st = run(root, f)
check("archive not matching its tag refused", installed(root)[0] == "v1.0.0" and st["result"] == "error", st)
root, f = case("v1.0.0"); f.publish("v1.1.0", archive("v1.1.0", evil=True))
r, st = run(root, f)
check("path-escaping archive refused", installed(root)[0] == "v1.0.0" and st["result"] == "error"
      and not any(os.path.exists(os.path.join(root, p, "escaped.txt")) for p in ("", "var", "var/lib")), st)

# 5. Failed install / unhealthy start roll back, and that version is not retried.
root, f = case("v1.0.0"); f.publish("v1.1.0", archive("v1.1.0", extra={"FAIL": "1"}))
r, st = run(root, f)
check("failed install rolls back", installed(root) == ("v1.0.0", "# app v1.0.0") and st["result"] == "rolled_back"
      and "v1.1.0" in st["failed"], r.stdout + r.stderr)
r, st = run(root, f)
check("failed version not retried", st["result"] == "skipped", st)
root, f = case("v1.0.0"); f.publish("v1.1.0", archive("v1.1.0", extra={"UNHEALTHY": "1"}))
r, st = run(root, f)
check("unhealthy start rolls back", installed(root) == ("v1.0.0", "# app v1.0.0") and st["result"] == "rolled_back", r.stdout + r.stderr)

# 6. UPDATE_CHANNEL=off and --check only report.
root, f = case("v1.0.0", channel="off"); f.publish("v1.1.0", archive("v1.1.0"))
r, st = run(root, f)
check("channel off only reports", installed(root)[0] == "v1.0.0" and st["result"] == "available" and st["latest"] == "v1.1.0", st)
root, f = case("v1.0.0"); f.publish("v1.1.0", archive("v1.1.0"))
r, st = run(root, f, "--check")
check("--check only reports", installed(root)[0] == "v1.0.0" and st["result"] == "available", st)

# 7. GitHub unreachable is not an error worth paging about.
root, f = case("v1.0.0"); f.release = None; f.srv.shutdown(); f.srv.server_close()
r, st = run(root, f)
check("offline check is recorded, install untouched", installed(root)[0] == "v1.0.0" and st["result"] == "error" and r.returncode == 0, st)

# 8. A repository with no releases yet (GitHub answers 404) is not an error.
root, f = case("v1.0.0"); f.release = None
r, st = run(root, f)
check("no releases yet", st["result"] == "no_release" and installed(root)[0] == "v1.0.0", st)

print(f"\n{'all passed' if not fails else f'{fails} failed'}")
sys.exit(1 if fails else 0)
