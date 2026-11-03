#!/usr/bin/env python3
"""Keep an OpenPetCam server up to date from GitHub releases (run daily by petcam-update.timer, as root).

What it does, in order:
1. Reads UPDATE_CHANNEL and UPDATE_REPO from /etc/petcam/site.env.
   - `release` (the default): install a newer release when one appears.
   - `off`: only check, and show "update available" in the app.
2. Reads the installed version from /opt/petcam/VERSION. An install deployed from a git checkout that is
   not exactly a release tag (for example `v1.0.0-12-gabc1234`, or `-dirty`) is somebody's development
   install, so it is never overwritten automatically.
3. Asks GitHub for the latest release. If it is newer, downloads `openpetcam-<version>.tar.gz` and
   `SHA256SUMS`, checks the hash, and unpacks it into a staging folder.
4. Saves a copy of the current code, runs the release's install.sh (the same one ./deploy.sh runs), and
   checks the result: the app answers /healthz, and the watcher writes a fresh heartbeat within 3 minutes.
   On any failure it puts the saved code back, reinstalls it, and never tries that version again.
5. Writes the outcome to /var/lib/petcam/update.json, which the app shows in Settings. alert.py pages the
   owners if the check itself stops running.

Your settings, passwords, keys, trained pet model, recordings and history are not in a release, so an update
never touches them. `python3 update.py --check` only reports; `--now` updates even outside the timer.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import urllib.error
import urllib.request

R = os.environ.get("PETCAM_UPDATE_ROOT", "")    # tests run everything under a scratch root
ETC = f"{R}/etc/petcam"
OPT = f"{R}/opt/petcam"
STATE = f"{R}/var/lib/petcam"
WORK = f"{R}/var/lib/petcam-update"
API = os.environ.get("PETCAM_UPDATE_API", "https://api.github.com")
HEALTH_URL = os.environ.get("PETCAM_HEALTH_URL", "http://127.0.0.1:8080/healthz")
STATUS = f"{STATE}/update.json"
HEARTBEAT = f"{STATE}/heartbeat"
DEFAULT_REPO = "crimeacs/OpenPetCam"
HTTP_DEADLINE = 120          # seconds, wall clock, per download
HEALTH_WAIT = int(os.environ.get("PETCAM_HEALTH_WAIT", 180))            # seconds for the watcher to heartbeat after an install
INSTALL_TIMEOUT = 1500       # install.sh may install packages on a slow nano
SOURCE = f"{OPT}/source"     # install.sh keeps a copy of the source it installed (no settings, no models)


def log(*a):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), *a, flush=True)


def site():
    conf = {}
    try:
        for line in open(f"{ETC}/site.env"):
            m = re.match(r"\s*([A-Z_]+)=(.*)", line)
            if m:
                conf[m.group(1)] = m.group(2).split("#")[0].strip().strip('"').strip("'")
    except OSError:
        pass
    return conf


def parse(v):
    """'v1.2.3' -> (1, 2, 3); anything that is not exactly a release (e.g. 'v1.2.3-4-gabc', 'dirty') -> None."""
    m = re.fullmatch(r"v?(\d+)\.(\d+)\.(\d+)", (v or "").strip())
    return tuple(int(x) for x in m.groups()) if m else None


def installed():
    try:
        return open(f"{OPT}/VERSION").read().strip()
    except OSError:
        return ""


def read_status():
    try:
        return json.load(open(STATUS))
    except (OSError, ValueError):
        return {}


def write_status(**kw):
    st = read_status()
    st.update(kw, checked=time.time())
    tmp = STATUS + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f)
    os.chmod(tmp, 0o644)
    os.replace(tmp, STATUS)
    return st


def fetch(url, dest=None, limit=50 << 20):
    """GET with a real wall-clock deadline (a socket timeout only bounds silence, not a slow trickle)."""
    out, err = {}, {}

    def run():
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "openpetcam-updater",
                                                       "Accept": "application/vnd.github+json"})
            with urllib.request.urlopen(req, timeout=30) as r:
                data = r.read(limit + 1)
            if len(data) > limit:
                raise ValueError("download too large")
            out["data"] = data
        except Exception as e:      # noqa: BLE001  reported to the caller
            err["e"] = e

    t = threading.Thread(target=run, daemon=True)
    t.start()
    t.join(HTTP_DEADLINE)
    if t.is_alive():
        raise TimeoutError(f"{url} took longer than {HTTP_DEADLINE} s")
    if "e" in err:
        raise err["e"]
    if dest:
        with open(dest, "wb") as f:
            f.write(out["data"])
    return out["data"]


def latest(repo):
    rel = json.loads(fetch(f"{API}/repos/{repo}/releases/latest", limit=2 << 20))
    tag = rel.get("tag_name", "")
    assets = {a["name"]: a["browser_download_url"] for a in rel.get("assets", [])}
    return tag, assets, (rel.get("body") or "")[:2000], rel.get("html_url", "")


def healthy(since, was_beating):
    """The app answers and the watcher has heartbeated since `since`."""
    deadline = time.time() + HEALTH_WAIT
    app_ok = False
    while time.time() < deadline:
        if not app_ok:
            try:
                with urllib.request.urlopen(HEALTH_URL, timeout=5) as r:
                    app_ok = r.status == 200
            except OSError:
                pass
        try:
            beat = os.path.getmtime(HEARTBEAT) > since
        except OSError:
            beat = False
        # A paused camera (privacy) still heartbeats; an offline camera does not, so accept the app alone
        # when the heartbeat was already stale before the update.
        if app_ok and (beat or not was_beating):
            return True
        time.sleep(1 if R else 5)
    return False


def snapshot(dest):
    """Copy the source of the running install so a failed update can be undone exactly (configs included)."""
    if not os.path.exists(f"{SOURCE}/install.sh"):
        raise RuntimeError(f"no {SOURCE}/install.sh to roll back to; run ./deploy.sh once to create it")
    shutil.copytree(SOURCE, dest, ignore=shutil.ignore_patterns("__pycache__"))


def run_install(src):
    shutil.copy(f"{ETC}/site.env", f"{src}/site.env")
    r = subprocess.run(["timeout", "-k", "30", str(INSTALL_TIMEOUT), "bash", f"{src}/install.sh"],
                       capture_output=True, text=True, stdin=subprocess.DEVNULL,
                       env=dict(os.environ, DEBIAN_FRONTEND="noninteractive"))
    return r.returncode == 0, (r.stdout + r.stderr)[-1500:]


def restore(snap):
    """Put the saved source back: its install.sh re-renders every config and restarts the services."""
    return run_install(snap)


def main(argv):
    check_only = "--check" in argv
    conf = site()
    channel = conf.get("UPDATE_CHANNEL", "release").lower()
    repo = conf.get("UPDATE_REPO", DEFAULT_REPO)
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        write_status(result="error", detail=f"UPDATE_REPO '{repo}' is not owner/name")
        return 1
    cur = installed()
    try:
        tag, assets, notes, url = latest(repo)
    except urllib.error.HTTPError as e:
        if e.code != 404:
            log("could not check:", e)
            write_status(result="error", detail=f"GitHub answered {e.code}", current=cur)
            return 0
        log("no releases published yet in", repo)
        write_status(result="no_release", detail="", current=cur, latest="", channel=channel)
        return 0
    except Exception as e:      # noqa: BLE001
        log("could not check:", e)
        write_status(result="error", detail=f"could not reach GitHub: {e}"[:300], current=cur)
        return 0                # offline is normal; alert.py notices only if checks stop entirely
    new, now = parse(tag), parse(cur)
    st = read_status()
    base = dict(current=cur, latest=tag, notes=notes, url=url, channel=channel)
    if not new:
        write_status(result="error", detail=f"latest release tag '{tag}' is not a version", **base)
        return 1
    if now and new <= now:
        log("up to date:", cur)
        write_status(result="up_to_date", detail="", **base)
        return 0
    if not now:
        log(f"installed '{cur or 'unknown'}' is not a release; leaving it alone (latest is {tag})")
        write_status(result="development", detail="installed from a git checkout; update with ./deploy.sh", **base)
        return 0
    if tag in st.get("failed", []):
        write_status(result="skipped", detail=f"{tag} failed to install earlier; waiting for the next release", **base)
        return 0
    if check_only or channel != "release":
        log("update available:", cur, "->", tag)
        write_status(result="available", detail="", **base)
        return 0

    ver = tag.lstrip("v")
    tgz_name = f"openpetcam-{ver}.tar.gz"
    if tgz_name not in assets or "SHA256SUMS" not in assets:
        write_status(result="error", detail=f"release {tag} has no {tgz_name} or SHA256SUMS", **base)
        return 1
    os.makedirs(WORK, mode=0o700, exist_ok=True)
    stage = tempfile.mkdtemp(prefix=f"stage-{ver}-", dir=WORK)
    snap = os.path.join(WORK, "previous")
    try:
        log("downloading", tag)
        tgz = os.path.join(stage, tgz_name)
        fetch(assets[tgz_name], tgz)
        sums = fetch(assets["SHA256SUMS"], limit=1 << 16).decode()
        want = next((l.split()[0] for l in sums.splitlines() if l.strip().endswith(tgz_name)), None)
        got = subprocess.run(["sha256sum", tgz], capture_output=True, text=True).stdout.split()[0]
        if not want or want != got:
            raise RuntimeError(f"checksum mismatch for {tgz_name}")
        src = os.path.join(stage, "src")
        with tarfile.open(tgz) as t:
            t.extractall(src, filter="data")       # refuses absolute paths, links out of the tree, devices
        if open(os.path.join(src, "VERSION")).read().strip() != tag or not os.path.exists(f"{src}/install.sh"):
            raise RuntimeError("the release archive does not match its tag")
        shutil.rmtree(snap, ignore_errors=True)
        snapshot(snap)
        was_beating = os.path.exists(HEARTBEAT) and time.time() - os.path.getmtime(HEARTBEAT) < 120
        started = time.time()
        log("installing", tag)
        ok, out = run_install(src)
        if ok and healthy(started, was_beating):
            log("updated", cur, "->", tag)
            write_status(result="updated", detail="", previous=cur, updated_at=time.time(), **dict(base, current=tag))
            return 0
        log("update failed, rolling back:", out[-400:])
        rok, rout = restore(snap)
        log("rollback", "ok" if rok else "FAILED: " + rout[-300:])
        failed = (st.get("failed", []) + [tag])[-10:]
        why = out if not ok else "the app did not come back healthy"
        write_status(result="rolled_back" if rok else "broken", detail=why[-300:],
                     failed=failed, **base)
        return 1
    except Exception as e:      # noqa: BLE001
        log("update aborted:", e)
        write_status(result="error", detail=str(e)[:300], **base)
        return 1
    finally:
        shutil.rmtree(stage, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
