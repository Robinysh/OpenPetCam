"""Passive view evidence preserves timestamps, unknown poses and movement provenance."""
import hashlib
import os
import sys
import tempfile
import types

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "server"))
import watch  # noqa: E402


with tempfile.TemporaryDirectory() as d:
    watch.DB, watch.MOVED = os.path.join(d, "db.sqlite"), os.path.join(d, "moved")
    watch.init_db()
    con = watch.db()
    con.row_factory = __import__("sqlite3").Row
    w = watch.Watch.__new__(watch.Watch)
    w.pose, w.pose_t = (1850, 500), 995.0
    w.tracker = types.SimpleNamespace(last_move=990.0)
    w.frame_capture_t, w.frame_mode, w.last_detect_capture = 999.0, "push", 997.0
    w.observation_model_fingerprint, w.observation_map_fingerprint = "weights", "map1"
    w.privacy_on = lambda: False
    w.sharp_samples, w.view_blocked = [], False
    gray = np.arange(160 * 90, dtype=np.int16).reshape(90, 160) % 255
    w.check_blocked(con, gray, 1000.0)
    assert w.frame_sharpness > 0
    # Instrumentation may use ONLY existing state; an accidental camera lookup fails this test.
    def unexpected(*args, **kwargs):
        raise AssertionError("passive observation must not query camera hardware")
    watch.camapi.motor_status = unexpected
    w.current_pose = unexpected
    w.record_pose_observation(con, 1000.0, b"existing jpeg", 1.0)
    r = dict(con.execute("SELECT * FROM pose_observation").fetchone())
    assert r["capture_ts"] == 999.0 and r["pose_ts"] == 995.0 and r["detection_capture_ts"] == 997.0
    assert r["pose_settled"] == 1 and r["pan"] == 1850 and r["tilt"] == 500
    assert r["frame_sha256"] == hashlib.sha256(b"existing jpeg").hexdigest()
    assert r["sharpness"] == w.frame_sharpness and r["score"] == 1.0
    assert r["model_fingerprint"] == "weights" and r["map_fingerprint"] == "map1"
    for t, capture, pose_t, moved, pose in (
        (1041, 1040, 995, 990, (1850, 500)),  # cache too old
        (1042, 1000, 995, 997, (1850, 500)),  # camera moved after cached pose
        (1043, 1000, 1001, 990, (1850, 500)), # later pose cannot prove earlier geometry
        (1044, 1000, 995, 990, None),        # no pose: unknown, not home/default
    ):
        w.frame_capture_t, w.pose_t, w.pose, w.tracker.last_move = capture, pose_t, pose, moved
        w.record_pose_observation(con, t, b"existing jpeg", 0.0)
        r = con.execute("SELECT pose_settled FROM pose_observation WHERE ts=?", (t,)).fetchone()
        assert r[0] == 0, (t, r[0])
    # Manual movement also invalidates geometry, without refreshing the cached motor position.
    w.pose, w.pose_t, w.frame_capture_t = (1850, 500), 995.0, 1000.0
    with open(watch.MOVED, "w") as f:
        f.write("manual")
    os.utime(watch.MOVED, (998, 998))
    w.record_pose_observation(con, 1045, b"existing jpeg", 0.0)
    r = dict(con.execute("SELECT * FROM pose_observation WHERE ts=1045").fetchone())
    assert r["moved_ts"] == 998 and r["pose_settled"] == 0
    assert con.execute("SELECT count(*) FROM events").fetchone()[0] == 0, "observing a view makes no obstruction/presence claim"
    con.close()
    os.unlink(watch.MOVED)
    # A failed motor read must not make the old cache look freshly measured.
    stale = watch.Watch.__new__(watch.Watch)
    stale.pose, stale.pose_t = (1200, 500), 1990.0
    attempts = []
    def failed_read():
        attempts.append(True)
        raise OSError("offline")
    watch.camapi.motor_status = failed_read
    assert stale.current_pose(2021.0) == (1200, 500)
    assert stale.pose_t == 1990.0, "failed position read must preserve original measurement timestamp"
    stale.current_pose(2022.0)
    assert len(attempts) == 1, "failed reads retain a bounded retry cooldown"
    watch.camapi.motor_status = lambda: ("0", 1300, 550)
    assert stale.current_pose(2052.0) == (1300, 550) and stale.pose_t == 2052.0
print("pose observation: ok")
