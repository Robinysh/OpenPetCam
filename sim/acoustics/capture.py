"""Bounded, resumeless acquisition on the existing camera server.

Run this module on the server with its application environment. Planning and
analysis are separate laptop commands. This adapter archives FIFO PCM; it does
not manufacture DAC/ADC timing from host timestamps or acoustic correlation.
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
import uuid
import wave


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def assets(plan_path, suite_path):
    """Validate every playback byte before any camera operation (stdlib only)."""
    plan = json.loads(Path(plan_path).read_text())
    suite_path = Path(suite_path)
    suite = json.loads(suite_path.read_text())
    if plan.get("schema") != "openpetcam_acquisition_v1" or plan.get("suite_sha256") != sha(suite_path):
        raise ValueError("plan schema or suite hash mismatch")
    packets = {}
    for entry in suite["packets"]:
        manifest_path = (suite_path.parent / entry["manifest"]).resolve()
        if not manifest_path.is_relative_to(suite_path.parent.resolve()):
            raise ValueError("manifest escapes suite directory")
        if sha(manifest_path) != entry["manifest_sha256"]:
            raise ValueError("manifest hash mismatch")
        m = json.loads(manifest_path.read_text())
        fs = m.get("sample_rate_hz")
        if (m.get("schema") != "openpetcam.acoustic.probe_packet.v1"
                or type(fs) is not int or fs not in (16000, 48000)
                or m.get("config", {}).get("sample_rate_hz") != fs):
            raise ValueError("invalid packet schema or sample rate")
        payloads = {}
        for kind in ("packet", "probe", "pilot"):
            info = m[kind]
            if type(info.get("samples")) is not int or info["samples"] <= 0:
                raise ValueError("invalid asset sample count")
            if Path(info["file"]).name != info["file"]:
                raise ValueError("asset must be basename")
            path = manifest_path.parent / info["file"]
            if sha(path) != info["sha256"]:
                raise ValueError("WAV hash mismatch")
            with wave.open(str(path), "rb") as w:
                if (w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getcomptype()) != (1, 2, fs, "NONE"):
                    raise ValueError("unsupported playback format")
                if w.getnframes() != info["samples"]:
                    raise ValueError("WAV header sample count differs from manifest")
                raw = w.readframes(w.getnframes())
                if len(raw) != 2 * info["samples"] or hashlib.sha256(raw).hexdigest() != info["pcm_sha256"]:
                    raise ValueError("PCM sample inventory mismatch")
                payloads[kind] = raw
        events = m["events"]
        count = m["config"]["repetitions"]
        if type(count) is not int or count < 2 or [q["type"] for q in events] != ["pilot"] + ["shot"] * count + ["pilot"]:
            raise ValueError("invalid event inventory")
        if events[0]["event_id"] != "pilot_start" or events[-1]["event_id"] != "pilot_end":
            raise ValueError("invalid bracketing pilot identities")
        previous = 0
        for j, event in enumerate(events):
            start, end = event["sample_start"], event["sample_end"]
            if type(start) is not int or type(end) is not int or not previous <= start < end <= m["packet"]["samples"]:
                raise ValueError("invalid event sample positions")
            template = payloads["probe" if event["type"] == "shot" else "pilot"]
            if payloads["packet"][2 * start:2 * end] != template:
                raise ValueError("event differs from exact template")
            if event["type"] == "shot" and event["repeat_id"] != j - 1:
                raise ValueError("invalid repetition inventory")
            previous = end
        if m["packet_id"] != entry["packet_id"]:
            raise ValueError("packet identity mismatch")
        if m["packet_id"] in packets:
            raise ValueError("duplicate suite packet ID")
        packets[m["packet_id"]] = (m, manifest_path.parent / m["packet"]["file"])
    ids = set()
    gathers = {}
    closed_gathers = set()
    last_gather = None
    comparison_pose = None
    geometry = {}
    gather_roles = {}
    volume_blocks = {}
    volume_pose_packet = None
    if plan.get("mode") not in ("comparison", "gather", "volume"):
        raise ValueError("invalid acquisition mode")
    if plan["mode"] == "volume":
        levels, reference_volume, blocks = plan.get("volumes"), plan.get("reference_volume"), plan.get("blocks")
        if (plan.get("purpose") != "noisy_room_volume_robustness"
                or not isinstance(levels, list) or len(levels) < 2
                or any(type(v) is not int or not 1 <= v <= 40 for v in levels)
                or len(set(levels)) != len(levels)
                or type(reference_volume) is not int or not 1 <= reference_volume <= 40
                or reference_volume not in levels
                or type(blocks) is not int or not 1 <= blocks <= 20):
            raise ValueError("invalid volume-study design metadata")
    for e in plan["entries"]:
        if e["capture_id"] in ids or not re.fullmatch(r"[A-Za-z0-9_-]+", e["capture_id"]):
            raise ValueError("invalid or duplicate capture ID")
        ids.add(e["capture_id"])
        if any(type(e[k]) is not int for k in ("pan", "tilt", "volume", "block")):
            raise ValueError("pose, volume and block must be integers")
        if e["packet_id"] not in packets or not 1 <= e["volume"] <= 40:
            raise ValueError("invalid packet or playback volume")
        if not 0 <= e["pan"] <= 3700 or not 40 <= e["tilt"] <= 960:
            raise ValueError("unsafe planned motor coordinates")
        if e["source_id"] != f"pan_{e['pan']}" or e["receiver_id"] != f"pan_{e['pan']}_tilt_{e['tilt']}":
            raise ValueError("source/receiver identity disagrees with pose")
        if e["gather_id"] != f"pan_{e['pan']}_block_{e['block']}":
            raise ValueError("gather identity disagrees with source/block")
        for label, identity in (("source", e["source_id"]), ("receiver", e["receiver_id"])):
            position, axis = e[f"{label}_m_assumed"], e[f"{label}_axis_assumed"]
            if (len(position) != 3 or len(axis) != 3
                    or any(not isinstance(v, (int, float)) or not math.isfinite(v) for v in (*position, *axis))
                    or abs(sum(v * v for v in axis) - 1) > 1e-8):
                raise ValueError("invalid assumed transducer geometry")
            values = tuple(position) + tuple(axis)
            key = (label, identity)
            if key in geometry and any(abs(a - b) > 1e-9 for a, b in zip(geometry[key], values)):
                raise ValueError("transducer geometry changed under the same identity")
            geometry[key] = values
        if plan["mode"] == "comparison":
            pose = (e["pan"], e["tilt"], e["volume"])
            comparison_pose = comparison_pose or pose
            if pose != comparison_pose:
                raise ValueError("probe comparisons require identical pose and volume")
        elif plan["mode"] == "volume":
            signature = (e["pan"], e["tilt"], e["packet_id"])
            if volume_pose_packet is not None and signature != volume_pose_packet:
                raise ValueError("volume study must keep one fixed pose and exact probe")
            volume_pose_packet = signature
            volume_blocks.setdefault(e["block"], []).append(e)
        else:
            key = e["gather_id"]
            if last_gather is not None and last_gather != key:
                closed_gathers.add(last_gather)
            if key in closed_gathers:
                raise ValueError("source gather must be contiguous")
            last_gather = key
            source = (e["pan"], e["volume"], e["packet_id"])
            if key in gathers and gathers[key] != source:
                raise ValueError("source/probe/level changed within gather")
            gathers[key] = source
            gather_roles.setdefault(key, []).append(e)
    if not ids:
        raise ValueError("empty capture plan")
    for entries in gather_roles.values():
        if (len(entries) < 3 or entries[0]["role"] != "reference_before"
                or entries[-1]["role"] != "reference_after"
                or entries[0]["receiver_id"] != entries[-1]["receiver_id"]
                or any(e["role"] != "receiver" for e in entries[1:-1])):
            raise ValueError("gather requires matching bracketing reference poses")
    if plan["mode"] == "volume":
        expected_blocks = [block for block in range(blocks) for _ in range(len(levels) + 2)]
        if [e["block"] for e in plan["entries"]] != expected_blocks:
            raise ValueError("volume-study blocks must be complete, ordered and contiguous")
        for entries in volume_blocks.values():
            if (entries[0]["role"] != "reference_before" or entries[-1]["role"] != "reference_after"
                    or entries[0]["volume"] != reference_volume or entries[-1]["volume"] != reference_volume
                    or any(e["role"] != "volume_trial" for e in entries[1:-1])
                    or sorted(e["volume"] for e in entries[1:-1]) != sorted(levels)):
                raise ValueError("volume-study block requires matching anchors and exactly one trial per declared level")
    return plan, packets


def verify_constant_output_rate(device):
    """Require observed SDK AO attributes to stay at48k, including idle frames.

    Call only on metadata freshly produced by the framed decoder. This does not
    infer physical clock stability from an SDK attribute or sample timestamps.
    """
    frames, submissions = device.get("frames"), device.get("ao_submissions")
    if (device.get("complete") is not True or device.get("source_rate_hz") != 48000
            or not frames or not submissions):
        raise ValueError("native48k playback requires complete framed output-rate evidence")
    if (any(frame.get("reference", {}).get("ao_input_rate_hz") != 48000 for frame in frames)
            or any(row.get("ao", {}).get("rate_hz") != 48000 for row in submissions)):
        raise ValueError("native48k output-rate contract failed: missing or changing SDK AO rate")
    return {"required_rate_hz": 48000, "reported_rate_constant": True,
            "mic_frames_checked": len(frames), "ao_submissions_checked": len(submissions),
            "physical_clock_verified": False,
            "scope": "Reported SDK AO attributes throughout captured interval, not physical sample-clock or phase calibration."}


def acquisition_rate_policy(plan, packets, *, allow_rate_switching=False):
    """Check selected live packets without restricting historical archive readers."""
    if type(allow_rate_switching) is not bool:
        raise ValueError("allow_rate_switching must be an explicit boolean")
    rates = sorted({packets[e['packet_id']][0]['sample_rate_hz'] for e in plan['entries']})
    if not allow_rate_switching and rates != [48000]:
        raise ValueError("acquisition requires native48k packets; use --allow-rate-switching only for diagnostic16k playback")
    return {"required_source_rate_hz": None if allow_rate_switching else 48000,
            "selected_source_rates_hz": rates, "allow_rate_switching": allow_rate_switching,
            "purpose": "diagnostic_rate_switching" if allow_rate_switching else "fixed_native48_acquisition",
            "physical_clock_verified": False}


def parse_audio_aic_format(text):
    """Read only stable driver-reported rates/formats, never infer defaults."""
    fields = {}
    for kind, label in (('record', 'record'), ('replay', 'output')):
        for source, target in (('rate', 'rate_hz'), ('format', 'format_bits')):
            values = re.findall(r'^\s*The living ' + source + ' of ' + kind + r'\s*:\s*([0-9]+)\s*$',
                                text, re.MULTILINE)
            if len(values) != 1:
                raise RuntimeError('missing or ambiguous live audio driver ' + kind + ' ' + source)
            fields[label + '_' + target] = int(values[0])
    return fields


def verify_configured_output_rate(snapshot):
    """Require live48k driver format; this is not a frequency measurement."""
    fmt = snapshot.get('runtime_audio_format', {})
    expected = {'record_rate_hz': 48000, 'output_rate_hz': 48000,
                'record_format_bits': 16, 'output_format_bits': 16}
    if any(type(fmt.get(k)) is not int or fmt[k] != v for k, v in expected.items()):
        raise RuntimeError("cannot verify live48k driver format; acquisition will not reconfigure it")
    return {'configured_rate_hz': 48000, 'basis': 'kernel_audio_aic_reported_format',
            'format': dict(fmt), 'physical_clock_verified': False}


class Camera:
    def __init__(self, *, legacy_fifo=False, reader="/tmp/opc-audio-stream", socket="/tmp/opc-audio.sock",
                 allow_rate_switching=False):
        if type(allow_rate_switching) is not bool:
            raise ValueError("allow_rate_switching must be an explicit boolean")
        sys.path.insert(0, "/opt/petcam")
        import app
        import petconf
        self.app = app
        self.state = Path(petconf.STATE)
        self.legacy_fifo = legacy_fifo
        self.allow_rate_switching = allow_rate_switching
        self.reader, self.socket = reader, socket
        host = petconf.SITE.get("CAMERA_WG_IP", "10.77.0.2")
        self.ssh = ["ssh", "-T", "-i", str(self.state / "ssh/id_ed25519"),
                    "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
                    "-o", f"UserKnownHostsFile={self.state / 'ssh/known_hosts'}", f"root@{host}"]

    def command(self, command, data=None, timeout=30):
        r = subprocess.run(self.ssh + [command], input=data, capture_output=True, timeout=timeout)
        if r.returncode:
            raise RuntimeError(f"camera command failed ({r.returncode}): {r.stderr.decode(errors='replace')[-300:]}")
        return r.stdout

    def check_privacy(self):
        p = self.state / "privacy"
        if p.exists() and p.read_text().strip() not in ("", "0"):
            raise RuntimeError("camera privacy is active; acquisition stopped")

    def live_audio_format(self):
        return parse_audio_aic_format(self.command('cat /proc/jz/audio/audio_aic_info').decode())

    def snapshot(self):
        self.check_privacy()
        if not self.legacy_fifo:
            self.command(f"test -x {shlex.quote(self.reader)} && test -S {shlex.quote(self.socket)}")
        config = json.loads(self.command("cat /etc/prudynt.json"))
        # Select only audio settings; never archive API keys or network secrets.
        audio = config.get("audio", {})
        # Only keys registered by the current JsonAPI audio handler. In
        # particular output_sample_rate is not a supported JSON query.
        registered = ('mic_enabled', 'mic_hq', 'mic_format', 'mic_vol', 'mic_gain',
                      'mic_alc_gain', 'mic_noise_suppression', 'mic_high_pass_filter',
                      'mic_agc_enabled', 'mic_agc_target_level_dbfs',
                      'mic_agc_compression_gain_db', 'force_stereo', 'spk_enabled',
                      'spk_vol', 'spk_gain')
        runtime = self.app.prudynt({"audio": dict.fromkeys(k for k in registered if k in audio)})
        audio_format = self.live_audio_format()
        return {"audio": audio, "runtime_audio": runtime.get("audio", {}),
                "runtime_audio_format": audio_format,
                "runtime_output_rate_hz": audio_format['output_rate_hz'],
                "runtime_output_rate_basis": "kernel_audio_aic_reported_format",
                "config_origin": "/etc/prudynt.json plus live prudynt API",
                "build": self.command("cat /etc/os-release").decode(),
                "binary_sha256": self.command("sha256sum /usr/bin/prudynt").decode().split()[0],
                "timing_capability": ("unframed_pcm_no_hardware_sample_reference" if self.legacy_fifo
                                      else "framed_sdk_observations_require_per_capture_validation")}

    def pose(self):
        self.check_privacy()
        status, pan, tilt = self.app.motor_status()
        if status != "0":
            raise RuntimeError("camera is moving during acquisition")
        return [pan, tilt]

    def goto(self, pan, tilt):
        self.check_privacy()
        self.app.goto(pan, tilt, 250)
        self.app.wait_idle(30)
        time.sleep(.8)

    @contextlib.contextmanager
    def hold(self):
        stop = threading.Event()
        self.check_privacy()

        def heartbeat():
            while not stop.is_set():
                (self.state / "moved_at").touch()
                stop.wait(.5)

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=2)
            (self.state / "moved_at").touch()

    def record(self, wav_path, output, volume, duration_s):
        """One continuous reader; wait for incoming PCM before queuing playback."""
        self.check_privacy()
        with wave.open(str(wav_path), "rb") as wav:
            source_rate = wav.getframerate()
        if source_rate not in (16000, 48000):
            raise ValueError("unsupported playback source rate")
        allow_switching = getattr(self, 'allow_rate_switching', False)
        if source_rate != 48000 and not allow_switching:
            raise ValueError("acquisition requires native48k playback; diagnostic16k requires explicit rate-switching opt-in")
        if source_rate == 48000 and self.legacy_fifo:
            raise ValueError("native48k output-rate contract requires framed capture")
        configured_output = None
        if not allow_switching:
            configured_output = verify_configured_output_rate({'runtime_audio_format': self.live_audio_format()})
        token = uuid.uuid4().hex
        remote = f"/tmp/openpetcam-acoustic-{token}.wav"
        self.command(f"cat > {remote}", Path(wav_path).read_bytes())
        # Budget initial delivery (20s), PLAY command (12s), settle and source tail.
        seconds = math.ceil(duration_s + 40)
        (self.state / "talking").write_text(f"{time.time()} {time.time() + seconds + 10}")
        rec = None
        try:
            capture_file = Path(output) if self.legacy_fifo else Path(output).with_suffix(".opcaud")
            if self.legacy_fifo:
                # FIFO has no broadcast semantics. This fallback cannot verify
                # producer sample loss and is explicitly diagnostic only.
                cmd = ("mkdir /tmp/openpetcam-acoustic-reader.lock || exit 73; "
                       "trap 'rmdir /tmp/openpetcam-acoustic-reader.lock' EXIT; "
                       f"timeout {seconds} cat /run/prudynt/audio_mic.pcm")
            else:
                cmd = (f"timeout {seconds + 5} {shlex.quote(self.reader)} "
                       f"{shlex.quote(self.socket)} {seconds}")
            with capture_file.open("xb") as f:
                rec = subprocess.Popen(self.ssh + [cmd], stdout=f, stderr=subprocess.PIPE)
                started = time.monotonic()
                while capture_file.stat().st_size < 9600:
                    self.check_privacy()
                    if rec.poll() is not None or time.monotonic() - started > 20:
                        raise RuntimeError("raw microphone did not deliver initial PCM")
                    time.sleep(.05)
                time.sleep(.4)
                self.check_privacy()
                queued_before = time.monotonic()
                self.command(f"play -f wav -v {int(volume)} {shlex.quote(remote)}", timeout=12)
                queued_after = time.monotonic()
                limit = time.monotonic() + seconds + 12
                while rec.poll() is None:
                    self.check_privacy()
                    if time.monotonic() >= limit:
                        raise RuntimeError("microphone reader exceeded capture deadline")
                    time.sleep(.25)
                _, error = rec.communicate(timeout=5)
                allowed_codes = (0, 124) if self.legacy_fifo else (0,)
                if rec.returncode not in allowed_codes:
                    raise RuntimeError(f"raw microphone reader failed: {error.decode(errors='replace')[-200:]}")
            device = None
            if not self.legacy_fifo:
                from .device.decode import export_capture
                device = export_capture(capture_file, Path(output).parent, prefix=Path(output).stem + ".device")
                if not device["reference_sample_grid_verified"] or not device["reference_continuity_verified"]:
                    raise RuntimeError("SDK output reference did not establish a continuous paired sample grid")
                if source_rate == 48000:
                    output_rate_contract = verify_constant_output_rate(device)
                shutil.copyfile(device["mic_file"], output)
            size = Path(output).stat().st_size
            minimum = duration_s + 2 if self.legacy_fifo else duration_s - .25
            if size % 2 or size < minimum * 96000:
                raise RuntimeError("truncated or malformed raw microphone capture")
            result = {"sample_rate_hz": 48000, "pcm_format": "s16le mono", "samples": size // 2,
                    "queue_before_monotonic": queued_before, "queue_after_monotonic": queued_after,
                    "host_capture_started_monotonic": started,
                    "timing_status": "unreferenced_adc_samples; host_times_are_not_emission_times",
                    "hardware_timing_verified": False, "sample_continuity_verified": False,
                    "allow_rate_switching": allow_switching,
                    "configured_output_rate_check": configured_output}
            if device:
                result.update(timing_status="paired_sdk_output_reference; physical_DAC_mapping_unverified",
                              sample_continuity_verified=device["mic_continuity_verified"],
                              reference_file=Path(device["reference_file"]).name,
                              reference_sha256=device["reference_sha256"],
                              reference_sample_grid_verified=device["reference_sample_grid_verified"],
                              reference_continuity_verified=device["reference_continuity_verified"],
                              device_metadata_file=Path(output).stem + ".device.json",
                              device_metadata_sha256=sha(Path(output).parent / (Path(output).stem + ".device.json")),
                              active_ao_reference_verified=device["active_ao_reference_check"]["verified"],
                              framed_file=capture_file.name, framed_sha256=sha(capture_file),
                              common_mic_sample_interval=device["common_mic_sample_interval"])
                if source_rate == 48000:
                    result["output_rate_contract"] = output_rate_contract
            return result
        except BaseException:
            # Installed play utility's STOP clears the queue. This acquisition
            # owns playback; privacy or abort must stop it, not merely stop PCM.
            self.command("play stop", timeout=10)
            raise
        finally:
            if rec is not None and rec.poll() is None:
                rec.terminate()
                try:
                    rec.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    rec.kill()
            # Delete only our own file, after the bounded reader has finished.
            self.command(f"rm -f {remote}")


def run_plan(plan_path, suite_path, output, camera, *, deadline_s=900, allow_rate_switching=False):
    if not math.isfinite(deadline_s) or not 0 < deadline_s <= 3600:
        raise ValueError("deadline must be finite, positive and at most one hour")
    plan, packets = assets(plan_path, suite_path)
    rate_policy = acquisition_rate_policy(plan, packets, allow_rate_switching=allow_rate_switching)
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    shutil.copy2(plan_path, out / "plan.json")
    shutil.copytree(Path(suite_path).parent, out / "suite")
    code = Path(__file__).parent
    for path in code.rglob("*.py"):
        archived = out / "source" / path.relative_to(code)
        archived.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, archived)
    # Execute exactly the archived assets, revalidating after the copy.
    plan, packets = assets(out / "plan.json", out / "suite" / Path(suite_path).name)
    started = time.monotonic()
    original = None
    summary = {"schema": "openpetcam_capture_v1", "status": "running", "completed": 0,
               "inversion_allowed": False, "hardware_timing_verified": False,
               "acquisition_rate_policy": rate_policy,
               "plan_sha256": sha(plan_path), "suite_sha256": sha(suite_path),
               "source_sha256": {str(p.relative_to(out / "source")): sha(p)
                                 for p in (out / "source").rglob("*.py")},
               "python_version": sys.version}
    save(out / "session.json", summary)
    try:
        initial = camera.snapshot()
        save(out / "device_before.json", initial)
        if not allow_rate_switching:
            summary['configured_output_rate_check'] = verify_configured_output_rate(initial)
        original = camera.pose()
        with camera.hold():
            for e in plan["entries"]:
                manifest, wav_path = packets[e["packet_id"]]
                duration = manifest["packet"]["samples"] / manifest["sample_rate_hz"]
                # Reserve upload (30s), recording margin (40s), connection/cleanup (20s).
                if time.monotonic() - started + duration + 90 > deadline_s:
                    raise RuntimeError("session deadline prevents next capture")
                filename = e["capture_id"] + ".pcm"
                row = dict(e, file=filename, status="attempted", pose_stable=False,
                           packet_sha256=manifest["packet"]["sha256"])
                try:
                    camera.goto(e["pan"], e["tilt"])
                    before = camera.pose()
                    row["actual_pose_before"] = before
                    if before != [e["pan"], e["tilt"]]:
                        raise RuntimeError("actual motor pose differs from plan")
                    metadata = camera.record(wav_path, out / filename, e["volume"], duration)
                    after = camera.pose()
                    row.update(actual_pose_after=after, pcm_sha256=sha(out / filename),
                               acquisition=metadata, pose_stable=before == after)
                    if not row["pose_stable"]:
                        raise RuntimeError("camera moved during packet; capture rejected")
                    row["status"] = "recorded"
                except BaseException as exc:
                    row.update(status="failed", error=f"{type(exc).__name__}: {exc}")
                    raise
                finally:
                    with (out / "captures.jsonl").open("a") as log:
                        log.write(json.dumps(row, allow_nan=False) + "\n")
                        log.flush()
                        os.fsync(log.fileno())
                summary["completed"] += 1
                save(out / "session.json", summary)
                print(json.dumps({"capture": e["capture_id"], "packet": e["packet_id"],
                                  "source": e["source_id"], "receiver": e["receiver_id"]}), flush=True)
        final = camera.snapshot()
        save(out / "device_after.json", final)
        if final != initial:
            raise RuntimeError("device configuration changed during capture")
        summary["status"] = "captured_requires_qc_and_timing_calibration"
    except BaseException as exc:
        summary.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if original is not None:
            try:
                with camera.hold():
                    camera.goto(*original)
                summary["restored_pose"] = camera.pose()
                if summary["restored_pose"] != original:
                    raise RuntimeError("restored pose differs from initial pose")
            except Exception as exc:
                summary["restore_error"] = str(exc)
                if summary["status"] != "failed":
                    summary["status"] = "failed_restore"
        save(out / "session.json", summary)
    if summary.get("restore_error"):
        raise RuntimeError("camera pose restoration failed: " + summary["restore_error"])
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--suite", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--deadline-s", type=float, default=900)
    p.add_argument("--execute", action="store_true", help="move camera and play planned probes")
    p.add_argument("--legacy-fifo", action="store_true", help="diagnostic only: allow unframed, unverified microphone FIFO")
    p.add_argument("--allow-rate-switching", action="store_true",
                   help="diagnostic only: explicitly allow16k packets and output-rate reconfiguration")
    p.add_argument("--reader", default="/tmp/opc-audio-stream")
    p.add_argument("--socket", default="/tmp/opc-audio.sock")
    a = p.parse_args()
    plan, packets = assets(a.plan, a.suite)
    policy = acquisition_rate_policy(plan, packets, allow_rate_switching=a.allow_rate_switching)
    if not a.execute:
        print(json.dumps({"captures": len(plan["entries"]), "mode": plan["mode"],
                          "status": "validated_only; pass --execute on camera server to record",
                          "acquisition_rate_policy": policy}))
        return
    run_plan(a.plan, a.suite, a.output, Camera(legacy_fifo=a.legacy_fifo, reader=a.reader, socket=a.socket,
             allow_rate_switching=a.allow_rate_switching), deadline_s=a.deadline_s,
             allow_rate_switching=a.allow_rate_switching)


if __name__ == "__main__":
    main()
