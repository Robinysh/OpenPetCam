"""Byte-exact native reader burst/backpressure regressions; no SDK or hardware.

The tests compile the current repository reader into a temporary directory and
use only local AF_UNIX sockets, synthetic CRC-framed records and stdout pipes.
Strict burst tests request the same bounded socket capacity as the producer.
The partial-output test uses a deadline-bounded reliable producer to isolate
reader ordering from admission failures caused by host scheduling.
"""
from pathlib import Path
import errno
import hashlib
import json
import os
import select
import shlex
import shutil
import socket
import struct
import subprocess
import tempfile
import threading
import time
import unittest
import zlib

try:
    import fcntl
except ImportError:
    fcntl = None

READER = None


def sha(data):
    return hashlib.sha256(data).hexdigest()


def frame(meta, payload=b""):
    b = json.dumps(meta, separators=(",", ":")).encode()
    return (
        struct.pack(
            "<8sIII", b"OPCAUD01", len(b), len(payload), zlib.crc32(b + payload)
        )
        + b
        + payload
    )


def verify_frames(data, require_end=False):
    pos = 0
    count = 0
    while pos < len(data):
        if len(data) - pos < 20:
            raise ValueError("truncated framing prefix")
        magic, nj, nb, crc = struct.unpack_from("<8sIII", data, pos)
        end = pos + 20 + nj + nb
        if (
            magic != b"OPCAUD01"
            or end > len(data)
            or zlib.crc32(data[pos + 20 : end]) != crc
        ):
            raise ValueError("bad framing/CRC")
        m = json.loads(data[pos + 20 : pos + 20 + nj])
        if m["record_index"] != count:
            raise ValueError("record order/index changed")
        pos = end
        count += 1
    if require_end and (count < 2 or m.get("kind") != "end"):
        raise ValueError("successful reader output lacks complete terminal END")
    return count


def measured_pipe():
    r, w = os.pipe()
    os.set_blocking(r, False)
    os.set_blocking(w, False)
    n = 0
    try:
        while True:
            n += os.write(w, b"P" * 4096)
    except BlockingIOError:
        pass
    drained = 0
    while drained < n:
        drained += len(os.read(r, n - drained))
    return r, w, n


def io_case(
    name,
    *,
    burst_frames=80,
    burst_span_s=0.0067,
    release_after_s=0.85,
    duration_s=0.4,
    throttle_bytes=65536,
    throttle_pause_s=0.0,
    long_stall=False,
    early_eof=False,
    producer_send_buffer_bytes=262144,
    reliable_producer=False,
):
    result = {
        "name": name,
        "producer_send_buffer_request_bytes": producer_send_buffer_bytes,
        "producer_send_policy": "bounded_retry_for_output_ordering"
        if reliable_producer
        else "single_nonblocking_send_per_record",
        "producer_eagain_retries": 0,
        "producer_partial_sends": 0,
        "native_AO_payload_bytes": 1920,
        "burst_frames": burst_frames,
        "burst_span_target_s": burst_span_s,
        "stdout_release_after_accept_s": None if long_stall else release_after_s,
        "reader_duration_s": duration_s,
        "sender_fault": None,
        "stop_observed": False,
        "sender_error": None,
    }
    r, w, capacity = measured_pipe()
    result["measured_stdout_pipe_capacity_bytes"] = capacity
    # Build the COMPLETE planned stream before a child, socket or send exists.
    # An attempted/sent prefix is distinct from the expected lossless stream.
    record = 0
    aoindex = 0

    def make(kind, payload=b""):
        nonlocal record, aoindex
        m = {
            "schema": "openpetcam.audio.v1",
            "kind": kind,
            "stream_id": "loopback-transport-fixture",
            "record_index": record,
            "dropped_records": 0,
            "producer_pid": 12345,
        }
        if kind == "ao":
            m.update(
                rc=0,
                monotonic_before_ns=1 + aoindex * 20000000,
                monotonic_after_ns=2 + aoindex * 20000000,
                ao={
                    "submission_index": aoindex,
                    "bytes": len(payload),
                    "rate_hz": 48000,
                    "bitwidth": 16,
                    "channels": 1,
                },
            )
            aoindex += 1
        record += 1
        return frame(m, payload)

    hello = make("hello")
    warm = []
    n = 0
    while n < capacity + 8192:
        b = make("ao", bytes((aoindex + i) % 256 for i in range(1920)))
        warm.append(b)
        n += len(b)
    burst = [
        make("ao", bytes((aoindex + i) % 256 for i in range(1920)))
        for _ in range(burst_frames)
    ]
    extra = [
        make("ao", bytes((aoindex + i) % 256 for i in range(1920)))
        for _ in range(512 if long_stall else 0)
    ]
    end = make("end")
    planned = b"".join([hello, *warm, *burst, *extra, end])
    result.update(
        expected_bytes=len(planned),
        expected_sha256=sha(planned),
        expected_framed_records=record,
        warmup_bytes=n,
        planned_stream_built_before_child=True,
    )
    captured = bytearray()
    sent_prefix = bytearray()
    accepted = threading.Event()
    done = threading.Event()
    shared = {}
    proc = None
    with tempfile.TemporaryDirectory(prefix="opcio-", dir="/tmp") as td:
        path = str(Path(td) / "s")
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        listener.bind(path)
        listener.listen(1)
        listener.settimeout(15)

        def serve():
            conn = None
            try:
                conn, _ = listener.accept()
                conn.setsockopt(
                    socket.SOL_SOCKET, socket.SO_SNDBUF, producer_send_buffer_bytes
                )
                result["producer_send_buffer_actual_bytes"] = conn.getsockopt(
                    socket.SOL_SOCKET, socket.SO_SNDBUF
                )
                conn.setblocking(False)
                shared["accepted_at"] = time.monotonic()
                accepted.set()

                def send_one(b):
                    # Only the partial-output test uses a reliable producer.
                    # It isolates stdout ordering from upstream admission. The
                    # burst/stop/overflow cases retain one-shot sends.
                    offset = 0
                    deadline = shared["accepted_at"] + duration_s + 1.0
                    while offset < len(b):
                        try:
                            n = conn.sendmsg(
                                [memoryview(b)[offset:]],
                                [],
                                getattr(socket, "MSG_DONTWAIT", 0),
                            )
                        except OSError as ex:
                            if reliable_producer and ex.errno in (
                                errno.EAGAIN,
                                errno.EWOULDBLOCK,
                                errno.EINTR,
                            ):
                                result["producer_eagain_retries"] += 1
                                remaining = deadline - time.monotonic()
                                if remaining > 0:
                                    select.select([], [conn], [], min(0.02, remaining))
                                    continue
                                ex = TimeoutError(
                                    errno.ETIMEDOUT,
                                    "bounded producer admission deadline",
                                )
                            result["sender_fault"] = {
                                "errno": ex.errno,
                                "name": errno.errorcode.get(ex.errno),
                                "stage": shared.get("stage"),
                                "attempt_bytes": len(b) - offset,
                                "record_offset": offset,
                            }
                            return False
                        if n > 0:
                            sent_prefix.extend(b[offset : offset + n])
                            offset += n
                        if offset != len(b):
                            result["producer_partial_sends"] += 1
                            if (
                                reliable_producer
                                and n > 0
                                and time.monotonic() < deadline
                            ):
                                continue
                            result["sender_fault"] = {
                                "errno": 0,
                                "name": "SHORT_SEND",
                                "stage": shared.get("stage"),
                                "attempt_bytes": len(b),
                                "sent_bytes": offset,
                            }
                            return False
                    return True

                shared["stage"] = "hello"
                if not send_one(hello):
                    return
                if early_eof:
                    return
                shared["stage"] = "warmup"
                for b in warm:
                    if not send_one(b):
                        return
                    time.sleep(0.001)
                time.sleep(0.06)
                shared["stage"] = "burst"
                t = time.monotonic()
                for i, b in enumerate(burst):
                    target = t + i * burst_span_s / max(1, burst_frames - 1)
                    while time.monotonic() < target:
                        time.sleep(max(0.0, min(0.00002, target - time.monotonic())))
                    if not send_one(b):
                        return
                result["burst_actual_span_s"] = time.monotonic() - t
                shared["stage"] = "long_stall" if long_stall else "wait_stop"
                for b in extra:
                    if not send_one(b):
                        return
                    try:
                        control = conn.recv(1)
                        if control == b"S":
                            result["stop_observed"] = True
                            result["stop_after_accept_s"] = (
                                time.monotonic() - shared["accepted_at"]
                            )
                    except BlockingIOError:
                        pass
                    time.sleep(0.001)
                deadline = shared["accepted_at"] + duration_s + 1.0
                while not result["stop_observed"] and time.monotonic() < deadline:
                    ready, _, _ = select.select([conn], [], [], 0.02)
                    if ready:
                        try:
                            b = conn.recv(1)
                        except BlockingIOError:
                            continue
                        if b == b"S":
                            result["stop_observed"] = True
                            result["stop_after_accept_s"] = (
                                time.monotonic() - shared["accepted_at"]
                            )
                            break
                        if b == b"":
                            break
                if result["stop_observed"]:
                    shared["stage"] = "end"
                    send_one(end)
                else:
                    result[
                        "sender_error"
                    ] = "stop request not observed before bounded producer timeout"
            except BaseException as ex:
                result["sender_error"] = repr(ex)
            finally:
                if conn is not None:
                    conn.close()
                done.set()

        thread = threading.Thread(target=serve, daemon=True)
        thread.start()
        started = time.monotonic()
        proc = subprocess.Popen(
            [str(READER), path, str(duration_s)], stdout=w, stderr=subprocess.PIPE
        )
        os.close(w)
        w = None
        try:
            while True:
                elapsed = time.monotonic() - started
                if accepted.is_set():
                    since = time.monotonic() - shared["accepted_at"]
                    deadline_exceeded = since > duration_s + 5.0
                    can_read = not long_stall and since >= release_after_s
                else:
                    deadline_exceeded = elapsed > 15.0
                    can_read = False
                if deadline_exceeded:
                    result["supervisor_timeout"] = True
                    proc.terminate()
                    break
                # Reader consumption is clock-driven and independent of sender completion.
                if can_read:
                    ready, _, _ = select.select([r], [], [], 0.01)
                    if ready:
                        try:
                            b = os.read(r, throttle_bytes)
                        except BlockingIOError:
                            b = b""
                        captured.extend(b)
                        if throttle_pause_s:
                            time.sleep(throttle_pause_s)
                else:
                    time.sleep(0.005)
                if proc.poll() is not None:
                    if not long_stall:
                        while True:
                            try:
                                b = os.read(r, 65536)
                            except BlockingIOError:
                                break
                            if not b:
                                break
                            captured.extend(b)
                    break
            try:
                _, stderr = proc.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()
                _, stderr = proc.communicate(timeout=2)
            thread.join(timeout=2)
            result.update(
                connection_accepted=accepted.is_set(),
                exit_code=proc.returncode,
                elapsed_s=time.monotonic() - started,
                sent_bytes=len(sent_prefix),
                sent_prefix_sha256=sha(sent_prefix),
                captured_bytes=len(captured),
                captured_sha256=sha(captured),
                byte_exact=bytes(captured) == planned,
                stderr=stderr.decode(errors="replace"),
            )
            telemetry = []
            for line in result["stderr"].splitlines():
                try:
                    j = json.loads(line)
                    if (
                        isinstance(j, dict)
                        and j.get("schema") == "openpetcam.reader_io.v1"
                    ):
                        telemetry.append(j)
                except ValueError:
                    pass
            result["reader_telemetry"] = telemetry[-1] if telemetry else None
            if captured:
                try:
                    result["verified_framed_records"] = verify_frames(
                        captured, require_end=result["exit_code"] == 0
                    )
                except ValueError as ex:
                    result["framing_error"] = str(ex)
        finally:
            os.close(r)
            listener.close()
        if thread.is_alive():
            result["server_thread_still_running"] = True
    return result


@unittest.skipUnless(
    os.name == "posix" and fcntl is not None and hasattr(socket, "AF_UNIX"),
    "requires POSIX pipes and local AF_UNIX sockets",
)
class AcousticDeviceIOTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global READER
        repo = Path(__file__).resolve().parents[2]
        source = repo / "sim/acoustics/device/reader.c"
        if not source.is_file():
            raise RuntimeError("repository reader source is missing")
        compiler = shlex.split(os.environ.get("CC", "cc"))
        if not compiler or shutil.which(compiler[0]) is None:
            raise unittest.SkipTest("a native C compiler is required")
        cls.build = tempfile.TemporaryDirectory(prefix="opc-device-io-build-")
        cls.addClassCleanup(cls.build.cleanup)
        READER = Path(cls.build.name) / "opc-audio-stream"
        compiled = subprocess.run(
            compiler
            + [
                "-std=c11",
                "-O2",
                "-Wall",
                "-Wextra",
                "-Werror",
                str(source),
                "-o",
                str(READER),
                "-lm",
            ],
            capture_output=True,
            timeout=60,
        )
        if compiled.returncode:
            raise RuntimeError(
                "native reader build failed: "
                + compiled.stderr.decode(errors="replace")
            )
        # Resolve any first-launch overhead before the measured local socket test.
        preflight = subprocess.run(
            [str(READER), "/tmp/opcio-no-endpoint-" + str(os.getpid()), ".1"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            timeout=15,
        )
        if preflight.returncode != 3:
            raise RuntimeError(
                "native reader startup precondition failed: "
                + repr((preflight.returncode, preflight.stderr))
            )

    def test_transient_native48_burst_is_lossless(self):
        r = io_case("transient_native48_80frame_burst")
        self.assertTrue(
            r["connection_accepted"],
            "Fixture connection not established; no instrument regression measured",
        )
        self.assertGreater(r["expected_bytes"], 80 * 1920)
        self.assertGreater(r["expected_framed_records"], 80)
        self.assertIsNone(
            r["sender_fault"],
            "Nonblocking producer lost a framed record during transient stdout stall: "
            + str(r["sender_fault"]),
        )
        self.assertEqual(r["exit_code"], 0, r["stderr"])
        self.assertTrue(r["byte_exact"], r)
        self.assertNotIn("framing_error", r)
        self.assertGreater(r.get("verified_framed_records", 0), 80)
        t = r["reader_telemetry"]
        self.assertIsNotNone(t, "bounded-buffer telemetry missing")
        self.assertGreater(r["producer_send_buffer_actual_bytes"], 0)
        self.assertLessEqual(r["producer_send_buffer_actual_bytes"], 1048576)
        self.assertEqual(
            r["producer_send_policy"], "single_nonblocking_send_per_record"
        )
        self.assertLessEqual(t["ring_capacity_bytes"], 1048576)
        self.assertLessEqual(t["high_water_bytes"], t["ring_capacity_bytes"])
        self.assertGreater(t["stdout_eagain"], 0)

    def test_stop_request_progresses_while_stdout_stalled(self):
        r = io_case(
            "stop_while_stdout_blocked",
            burst_frames=0,
            duration_s=0.20,
            release_after_s=0.85,
        )
        self.assertIsNone(r["sender_fault"], r)
        self.assertNotIn("framing_error", r)
        self.assertTrue(r["stop_observed"], r)
        self.assertLess(
            r["stop_after_accept_s"],
            0.75,
            "Stop request delayed behind stdout instead of progressing independently",
        )
        self.assertEqual(r["exit_code"], 0, r["stderr"])
        self.assertTrue(r["byte_exact"])

    def test_throttled_output_preserves_partial_write_order(self):
        r = io_case(
            "throttled_partial_output",
            release_after_s=0.30,
            duration_s=0.50,
            throttle_bytes=257,
            throttle_pause_s=0.0007,
            producer_send_buffer_bytes=32768,
            reliable_producer=True,
        )
        self.assertIsNone(r["sender_fault"], r)
        self.assertEqual(r["exit_code"], 0, r["stderr"])
        self.assertTrue(r["byte_exact"], r)
        self.assertNotIn("framing_error", r)
        self.assertEqual(r["producer_send_policy"], "bounded_retry_for_output_ordering")
        self.assertEqual(r["sent_bytes"], r["expected_bytes"])
        self.assertGreater(r["reader_telemetry"]["partial_writes"], 0)

    def test_permanent_stdout_stall_is_bounded_and_nonzero(self):
        r = io_case("permanent_stdout_stall", long_stall=True, duration_s=0.20)
        self.assertNotEqual(r["exit_code"], 0, r)
        self.assertFalse(
            r.get("supervisor_timeout", False),
            "Reader failed to enforce its own bounded deadline",
        )
        self.assertLess(r["elapsed_s"], 4.7, r)
        if r["reader_telemetry"]:
            t = r["reader_telemetry"]
            self.assertLessEqual(t["high_water_bytes"], t["ring_capacity_bytes"])
            self.assertLessEqual(t["ring_capacity_bytes"], 1048576)

    def test_unrequested_producer_eof_is_failure(self):
        r = io_case(
            "early_producer_eof", early_eof=True, release_after_s=0, duration_s=0.50
        )
        self.assertEqual(r["exit_code"], 7, r)
        self.assertLess(r["elapsed_s"], 2, r)


if __name__ == "__main__":
    unittest.main()
