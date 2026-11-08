"""Fixed-pose IR LED bursts with recorded controls and state restoration.

Current Thingino/Prudynt has no exposed, verified exposure/gain locks or
calibrated JPEG response. Controlled capture fails before mutation; explicit
--qualitative captures diagnostic data only. Importing this module is safe.
"""
import argparse
from contextlib import contextmanager, nullcontext
import hashlib
from io import BytesIO
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

from PIL import Image


class CaptureError(RuntimeError):
    pass


def _write_log(out, log):
    tmp = out / "log.json.tmp"
    tmp.write_text(json.dumps(log, indent=2) + "\n")
    tmp.replace(out / "log.json")


def _missing_capabilities(caps):
    missing = [k for k in ("exposure_lock", "gain_lock", "processing_lock") if caps.get(k) is not True]
    r = caps.get("radiometry", {})
    if not isinstance(r, dict):
        return missing + ["invalid radiometry structure"]
    if r.get("verified") is not True or r.get("response_model") not in ("linear", "lut"):
        missing.append("verified radiometric response")
    black, white = r.get("black_level"), r.get("white_level")
    if not _number(black) or not _number(white) or white <= black:
        missing.append("valid radiometric black/white bounds")
    for key in ("noise_floor", "max_off_drift"):
        if key in r and (not _number(r[key]) or r[key] < 0):
            missing.append(f"valid radiometric {key}")
    if r.get("response_model") == "lut":
        lut = r.get("lut")
        if (not isinstance(lut, (list, tuple)) or len(lut) < 2 or not all(_number(v) for v in lut)
                or any(b < a for a, b in zip(lut, lut[1:])) or lut[-1] <= lut[0]):
            missing.append("valid monotonic finite radiometric LUT with positive range")
    return missing


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _check(controls, led, reference, controlled):
    if controls.get("night_mode") != "night" or controls.get("ircut") != "out":
        raise CaptureError("Night mode and removed IR-cut filter must remain unchanged")
    if controls.get("led") != led:
        raise CaptureError("LED readback does not match requested state")
    if controls.get("moving") is not False:
        raise CaptureError("Camera is moving or idle readback is unknown")
    pose = (controls.get("pan"), controls.get("tilt"))
    if any(not _number(v) for v in pose):
        raise CaptureError("Missing camera pose readback")
    if reference is not None and pose != (reference["pan"], reference["tilt"]):
        raise CaptureError("Camera pose changed during fixed-pose capture")
    if controlled:
        if controls.get("verified") is not True:
            raise CaptureError("Radiometric readbacks are unverified")
        for key in ("exposure_us", "analog_gain", "digital_gain"):
            v = controls.get(key)
            if not _number(v) or v <= 0:
                raise CaptureError(f"Missing valid {key} readback")
            if reference is not None and not math.isclose(float(v), float(reference[key]), rel_tol=1e-6, abs_tol=1e-9):
                raise CaptureError(f"Locked {key} changed")
        if not isinstance(controls.get("processing_signature"), str) or not controls["processing_signature"]:
            raise CaptureError("Missing processing-state signature")
        if reference is not None and controls["processing_signature"] != reference["processing_signature"]:
            raise CaptureError("Locked image processing changed")


def run_capture(backend, out, *, cycles=2, burst=3, settle_s=1., frame_interval_s=.15,
                qualitative=False, sleep=time.sleep, now=time.time):
    """Capture off/on/off/on/off at the current pose with an injected backend.

    Controlled backends must expose actual fixed exposure/gain/processing
    readbacks and a verified linear or LUT response before they can be used.
    """
    if (type(cycles) is not int or type(burst) is not int or cycles < 1 or burst < 1
            or not _number(settle_s) or not _number(frame_interval_s)
            or not 0 <= settle_s <= 30 or not 0 <= frame_interval_s <= 10):
        raise ValueError("Invalid capture count or interval")
    if not qualitative and burst % 2 == 0:
        raise ValueError("Controlled capture requires an odd burst count before nonlinear response correction")
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise ValueError("Capture output directory must be empty")
    caps = backend.capabilities()
    log = {"schema_version": 2, "protocol": "nir-led-off-on-off", "started": now(),
           "mode": "qualitative" if qualitative else "controlled", "status": "preflight", "capabilities": caps,
           "frames": [], "bursts": [], "controls_readbacks": [], "controls_verified": False,
           "exposure_locked": False, "gain_locked": False, "processing_locked": False,
           "same_night_mode": False, "ircut_unchanged": False, "pose_fixed": False,
           "restoration": {"attempted": False, "success": False},
           "radiometry": {"response_model": "unknown", "verified": False, "black_level": 0, "white_level": 255}}
    missing = _missing_capabilities(caps)
    if missing and not qualitative:
        log.update(status="failed", error="Controlled capture unsupported: " + ", ".join(missing))
        _write_log(out, log)
        raise CaptureError(log["error"])
    if qualitative:
        log["limitations"] = ["Diagnostic JPEG differences only; not quantitative reflectivity", *missing]
    else:
        log["radiometry"].update(caps["radiometry"])
    original = reference = failure = None
    session = backend.session() if hasattr(backend, "session") else nullcontext()
    try:
        with session:
            try:
                original = backend.snapshot_state()
                log["original_state"] = original
                log["restoration"]["attempted"] = True
                backend.prepare_night(qualitative)
                log["status"] = "capturing"
                for group in range(cycles * 2 + 1):
                    led = "on" if group % 2 else "off"
                    backend.set_led(led == "on")
                    sleep(settle_s)
                    record = {"id": group, "state": led, "files": []}
                    log["bursts"].append(record)
                    for k in range(burst):
                        before = backend.read_controls()
                        _check(before, led, reference, not qualitative)
                        if reference is None:
                            reference = dict(before)
                            log.update(pan=before["pan"], tilt=before["tilt"], pos=[before["pan"], before["tilt"]])
                        captured = now()
                        jpg = backend.snapshot_jpeg()
                        with Image.open(BytesIO(jpg)) as im:
                            size = list(im.size)
                            im.verify()
                        after = backend.read_controls()
                        _check(after, led, reference, not qualitative)
                        name = f"{led}_{group:02d}_{k:02d}.jpg"
                        (out / name).write_bytes(jpg)
                        log["frames"].append({"file": name, "state": led, "led": led, "ircut": "out", "phase": group, "burst": group,
                                              "t": captured, "bytes": len(jpg), "image_size": size,
                                              "sha256": hashlib.sha256(jpg).hexdigest(), "controls": before, "controls_after": after})
                        log["controls_readbacks"].extend([dict(before, when="before", file=name), dict(after, when="after", file=name)])
                        record["files"].append(name)
                        _write_log(out, log)
                        sleep(frame_interval_s)
                log.update(same_night_mode=True, ircut_unchanged=True, pose_fixed=True,
                           controls_verified=not qualitative, exposure_locked=not qualitative,
                           gain_locked=not qualitative, processing_locked=not qualitative)
            except BaseException as exc:  # interruptions still restore state and persist a failure record
                failure = exc
                log.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            finally:
                if original is not None:
                    try:
                        backend.restore(original)
                        log["restored_state"] = backend.snapshot_state()
                        if log["restored_state"] != original:
                            raise CaptureError("Restoration readback differs from original settings")
                        log["restoration"]["success"] = True
                        if original.get("auto_enabled") is True:
                            log["restoration"]["note"] = "Original auto policy resumed; subsequent normal mode transitions remain possible"
                    except Exception as exc:
                        log["restoration"]["error"] = f"{type(exc).__name__}: {exc}"
                        failure = CaptureError(f"{log.get('error', 'Capture completed')}; restoration failed: {exc}")
                        log["status"] = "failed"
    except BaseException as exc:
        failure = exc
        log.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    log["status"] = "complete" if failure is None else "failed"
    if failure is not None:
        log["controls_verified"] = False
    log["finished"] = now()
    _write_log(out, log)
    if failure is not None:
        raise CaptureError(str(failure)) from failure
    return log


class ThinginoBackend:
    """Only source-verified shell controls; no invented exposure-lock API."""
    def __init__(self, command, state_dir="/var/lib/petcam", app_module=None):
        self.command, self.state_dir, self.app = list(command), Path(state_dir), app_module

    def capabilities(self):
        return {"backend": "thingino-prudynt", "exposure_lock": False, "gain_lock": False, "processing_lock": False,
                "radiometry": {"response_model": "unknown", "verified": False},
                "reason": "Prudynt exposes gain ceilings, not manual exposure/gain locks; JPEG response is uncalibrated"}

    def _run(self, args):
        return subprocess.run(self.command + [shlex.join(args)], capture_output=True, timeout=30, check=True).stdout

    def _text(self, args):
        return self._run(args).decode().strip()

    def _config(self, key):
        v = self._text(["jct", "/etc/thingino.json", "get", key])
        try:
            return json.loads(v)
        except json.JSONDecodeError:
            return v

    def _reload(self):
        self._run(["sh", "-c", 'test -s /run/daynightd.pid && kill -HUP "$(cat /run/daynightd.pid)"'])

    def _wait_policy(self, expected=None, timeout=15.):
        """Wait for two fresh daemon loops after HUP, not just a config write.

        daynightd processes HUP asynchronously. Its telemetry reflects its
        internal applied mode; waiting prevents a pending force-mode transition
        from re-enabling the LEDs in the middle of an off burst. This is still
        software state, not radiometric or mechanical feedback.
        """
        deadline = time.monotonic() + timeout
        baseline = None
        samples = set()
        while time.monotonic() < deadline:
            self._guard()
            try:
                data = json.loads(self._text(["cat", "/run/thingino/daynight_sensors"]))
            except json.JSONDecodeError:  # daemon rewrites this file without atomic replacement
                time.sleep(.2)
                continue
            stamp = data.get("time_now")
            mode = data.get("mode")
            # Camera timestamps establish camera freshness; the server's wall
            # clock may have a different offset. Only the local timeout uses
            # the server clock, and that clock is monotonic.
            if _number(stamp) and baseline is None:
                baseline = stamp
            elif (_number(stamp) and stamp > baseline and mode in ("day", "night")
                    and (expected is None or mode == expected)):
                samples.add(stamp)
                if len(samples) >= 2:
                    return
            elif mode not in ("day", "night") or (expected is not None and mode != expected):
                samples.clear()
            time.sleep(.2)
        raise CaptureError("Day/night daemon did not acknowledge two fresh policy samples")

    def _guard(self):
        self.app.require_camera_active()
        (self.state_dir / "moved_at").touch()
        if (self.state_dir / "capture.lock").exists():
            (self.state_dir / "capture.lock").touch()

    @contextmanager
    def session(self):
        if self.app is None:
            sys.path.insert(0, "/opt/petcam")
            import app
            self.app = app
        self.app.require_camera_active()
        lock = self.state_dir / "capture.lock"
        fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        identity = os.fstat(fd).st_ino
        os.write(fd, f"nir_capture pid={os.getpid()}".encode())
        os.close(fd)
        try:
            self._guard()
            yield
        finally:
            if lock.exists() and lock.stat().st_ino == identity:
                lock.unlink()

    def snapshot_state(self):
        mode = json.loads(self._text(["prudyntctl", "json", '{"image":{"running_mode":null}}']))["image"]["running_mode"]
        state = {"auto_enabled": self._config("daynight.enabled"), "force_mode": self._config("daynight.force_mode"),
                 "running_mode": mode, "ircut": int(self._text(["ircut", "read"])),
                 "led": int(self._text(["light", "ir940", "read"]))}
        if type(state["auto_enabled"]) is not bool or state["force_mode"] not in ("", "day", "night") or mode not in (0, 1):
            raise CaptureError("Cannot snapshot exact original day/night settings")
        if state["ircut"] not in (0, 1) or state["led"] not in (0, 1):
            raise CaptureError("Original LED/filter state is unknown; capture refused")
        if state["auto_enabled"] is False and not state["force_mode"]:
            raise CaptureError("Disabled photosensing without an explicit force mode cannot be synchronized for restoration")
        return state

    def prepare_night(self, qualitative):
        self._guard()
        if not qualitative:
            raise CaptureError("Current Prudynt backend cannot verify locked radiometry")
        self._run(["jct", "/etc/thingino.json", "set", "daynight.enabled", "false"])
        self._run(["jct", "/etc/thingino.json", "set", "daynight.force_mode", "night"])
        self._reload()
        self._wait_policy("night")
        self._run(["daynight", "night"])

    def set_led(self, on):
        self._guard()
        self._run(["light", "ir940", "on" if on else "off"])

    def read_controls(self):
        self._guard()
        state = self.snapshot_state()
        if state["auto_enabled"] is not False or state["force_mode"] != "night":
            raise CaptureError("Night policy changed during capture")
        motors = json.loads(self._text(["motors", "-j"]))
        return {"night_mode": "night" if state["running_mode"] == 1 else "day",
                "ircut": "out" if state["ircut"] == 0 else "in", "led": "on" if state["led"] else "off",
                "pan": int(motors["xpos"]), "tilt": int(motors["ypos"]), "moving": str(motors["status"]) != "0",
                "verified": False, "exposure_us": None, "analog_gain": None, "digital_gain": None,
                "processing_signature": None, "readback_kind": "software mode/filter latch and LED GPIO; not per-frame radiometry",
                "raw_state": state}

    def snapshot_jpeg(self):
        self._guard()
        return self._run(["prudyntctl", "snapshot", "-c", "0", "-q", "92"])

    def restore(self, state):
        policy = [["jct", "/etc/thingino.json", "set", "daynight.enabled", "true" if state["auto_enabled"] else "false"],
                  ["jct", "/etc/thingino.json", "set", "daynight.force_mode", state["force_mode"]]]
        errors = []
        for args in policy:
            try:
                self._run(args)
            except Exception as exc:
                errors.append(str(exc))
        try:
            self._reload()
            self._wait_policy(state["force_mode"] or None)
        except Exception as exc:
            errors.append(str(exc))
        # Apply exact hardware settings only after queued daemon policy changes
        # settle. Returning to auto allows subsequent normal mode transitions;
        # the caller verifies the immediate state and must not promise a freeze.
        hardware = [["prudyntctl", "json", json.dumps({"image": {"running_mode": state["running_mode"]}}, separators=(",", ":"))],
                    ["ircut", "on" if state["ircut"] else "off"], ["light", "ir940", "on" if state["led"] else "off"]]
        for args in hardware:
            try:
                self._run(args)
            except Exception as exc:
                errors.append(str(exc))
        if errors:
            raise CaptureError("; ".join(errors))
        for _ in range(2):
            time.sleep(.2)
            if self.snapshot_state() != state:
                raise CaptureError("Restored hardware/policy readback is not stable")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out")
    ap.add_argument("--qualitative", action="store_true", help="Explicitly allow diagnostic JPEG differences")
    ap.add_argument("--cycles", type=int, default=2)
    ap.add_argument("--burst", type=int, default=3)
    ap.add_argument("--settle-s", type=float, default=1.)
    ap.add_argument("--cam-command-file", default="/tmp/camcmd")
    args = ap.parse_args()
    backend = ThinginoBackend(shlex.split(Path(args.cam_command_file).read_text()))
    log = run_capture(backend, args.out, cycles=args.cycles, burst=args.burst, settle_s=args.settle_s, qualitative=args.qualitative)
    print(json.dumps({"status": log["status"], "frames": len(log["frames"]), "controls_verified": log["controls_verified"],
                      "restored": log["restoration"]["success"]}))


if __name__ == "__main__":
    main()
