"""Privacy does not depend on a live image or trigger camera recovery."""
from pathlib import Path
import queue
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'server'))
import watch  # noqa: E402


def forbidden(*args, **kwargs):
    raise AssertionError('camera acquisition/recovery or observation while paused')


class PrivacyWatchTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='opc-privacy-watch-')
        self.addCleanup(self.temp.cleanup)
        self.flag = Path(self.temp.name) / 'privacy'
        self.flag.write_text('1')
        self.heartbeat = Path(self.temp.name) / 'heartbeat'
        for name, value in [('PRIVACY', str(self.flag)), ('HEARTBEAT', str(self.heartbeat))]:
            p = patch.object(watch, name, value)
            p.start(); self.addCleanup(p.stop)
        p = patch.object(watch.camapi, 'STATE_DIR', self.temp.name)
        p.start(); self.addCleanup(p.stop)
        p = patch.object(watch, 'now', lambda: 10000.0)
        p.start(); self.addCleanup(p.stop)

    def watcher(self):
        w = watch.Watch.__new__(watch.Watch)
        w.frames = watch.FrameSource()
        w.audio = watch.Audio(queue.Queue())
        w.events = queue.Queue()
        w.privacy_check_thread = types.SimpleNamespace(is_alive=lambda: True)
        w.last_privacy_check = 10000.0
        w.imaging, w.find_result, w.last_bark = {}, None, None
        w.present, w.last_seen, w.gone_since = True, 9900.0, None
        w.rest_since, w.moving_since = 9900.0, None
        w.last_ok, w.offline = 9000.0, False
        w.last_fix, w.last_audio_fix = 0.0, 0.0
        w.view_blocked = False
        w.last_detect = 9900.0
        w.prev_small, w.prev_box = object(), [0.1, 0.1, 0.4, 0.4]
        w.dog_sound_pending, w.activity_sound_pending = True, 9900.0
        w.minute, w.acc = 165, {'n': 1}
        w.frames.latest = forbidden
        w.published = []
        w.publish = lambda **kw: w.published.append(kw)
        return w

    def test_paused_missing_stream_never_acquires_or_opens_observations(self):
        w = self.watcher()
        with patch.object(watch, 'snapshot', forbidden), patch.object(watch, 'db', forbidden):
            w.cycle()
        self.assertEqual(w.published[-1]['status'], 'privacy')
        self.assertFalse(w.published[-1]['audio_ok'])
        self.assertEqual(float(self.heartbeat.read_text()), 10000.0)
        self.assertFalse(w.dog_sound_pending)
        self.assertEqual(w.activity_sound_pending, 0.0)
        self.assertIsNone(w.prev_small)

    def test_paused_failure_never_heals_or_marks_camera_offline(self):
        w = self.watcher()
        w.heal_streamer = forbidden
        with patch.object(watch, 'db', forbidden):
            w.on_failure()
        self.assertEqual(w.published[-1]['status'], 'privacy')
        self.assertFalse(w.offline)

    def test_pause_during_frame_lookup_cannot_fall_back_to_a_camera_snapshot(self):
        w = self.watcher()
        self.flag.write_text('0')

        def pause_during_lookup():
            self.flag.write_text('1')
            return None

        w.frames.latest = pause_during_lookup
        with patch.object(watch, 'snapshot', forbidden), patch.object(watch, 'db', forbidden):
            w.cycle()
        self.assertEqual(w.published[-1]['status'], 'privacy')

    def test_all_camera_recovery_paths_are_inert_while_paused(self):
        w = self.watcher()
        w.last_fix, w.last_audio_fix = 0, 0
        w.stream_down_since, w.audio_watch_from = 9000, 9000
        w.same_since, w.fixes = 9000, 3
        w.prev_small = watch.np.zeros((2, 2), dtype=watch.np.int16)
        with patch.object(watch, 'db', forbidden), patch.object(watch, 'cam_action', forbidden):
            w.check_stream_stall()
            w.check_audio_lost(False, {'mic_enabled': 1})
            w.heal_streamer()
            self.assertFalse(w.check_frozen(None, w.prev_small.copy(), 10000))

    def test_frame_source_cannot_start_or_return_cached_room_in_privacy(self):
        frames = watch.FrameSource()
        frames.want(60)
        frames.frame = watch.np.ones((2, 2, 3), dtype=watch.np.uint8)
        frames.t = 10000
        self.assertIsNone(frames.desired_mode())
        self.assertIsNone(frames.latest())

    def test_search_and_frame_callbacks_cannot_override_paused_status(self):
        w = self.watcher()
        w.last_tick, w.live_last = 0, {'status': 'ok'}
        self.assertTrue(w.find_cancelled(9999))
        with patch.object(watch, 'write_json', forbidden), patch.object(watch, 'snapshot', forbidden):
            w.publish_tick(10000, 5)
            self.assertIsNone(w.find_pet_now(9999))
            self.assertIsNone(w.find_pet(9999))
            w.search_tick()
        self.assertEqual(w.published[-1]['status'], 'privacy')

    def test_missing_camera_reconciliation_is_bounded_and_does_not_need_a_frame(self):
        w = self.watcher()
        w.privacy_check_thread = None
        w.last_privacy_check = 0
        calls = []

        def ensure():
            calls.append(True)
            raise OSError('camera unreachable')

        with patch.object(watch.camapi, 'ensure_privacy_paused', ensure, create=True), \
                patch.object(watch, 'snapshot', forbidden), patch.object(watch, 'db', forbidden):
            w.cycle()
            w.privacy_check_thread.join(2)
            self.assertFalse(w.privacy_check_thread.is_alive())
            w.cycle()
        self.assertEqual(calls, [True], 'same interval must not create another camera action')
        self.assertEqual(w.published[-1]['status'], 'privacy')
        self.assertEqual(float(self.heartbeat.read_text()), 10000)

    def test_resume_resets_outage_and_motion_without_inventing_presence(self):
        w = self.watcher()
        w.cycle()
        self.flag.write_text('0')

        class AcquisitionResumed(Exception):
            pass

        def resumed():
            raise AcquisitionResumed

        w.frames.latest = resumed
        with self.assertRaises(AcquisitionResumed):
            w.cycle()
        self.assertFalse(w.privacy_active)
        self.assertEqual(w.last_ok, 10000)
        self.assertEqual(w.audio_watch_from, 10000)
        self.assertIsNone(w.same_since)
        self.assertFalse(w.present)
        self.assertEqual(w.last_seen, 0)
        self.assertEqual(w.tracker.last_seen_t, 10000)
        with patch.object(watch, 'db', forbidden), patch.object(watch, 'cam_action', forbidden):
            w.check_stream_stall()
            w.check_audio_lost(False, {'mic_enabled': 1})
            w.heal_streamer()

    def test_audio_pause_closes_a_blocked_reader_and_discards_buffered_episode(self):
        self.flag.write_text('0')
        reading, killed = threading.Event(), threading.Event()

        class Process:
            def __init__(self):
                self.stdout = self
                self.waited = False

            def read(self, count):
                reading.set()
                killed.wait(3)
                return b''

            def kill(self):
                killed.set()

            def wait(self):
                self.waited = True
                return 0

        process = Process()
        audio = watch.Audio(queue.Queue())
        audio.ring.append(watch.np.ones(10))
        audio.minute_peak, audio.minute_sounds = -10, 2
        with patch.object(watch.subprocess, 'Popen', lambda *a, **kw: process):
            thread = threading.Thread(target=audio.listen, daemon=True)
            thread.start()
            self.assertTrue(reading.wait(1))
            self.flag.write_text('1')
            thread.join(2)
            killed.set()
        self.assertFalse(thread.is_alive(), 'privacy must interrupt a silent blocked subscription')
        self.assertTrue(process.waited)
        self.assertEqual(list(audio.ring), [])
        self.assertTrue(audio.events.empty())
        self.assertEqual(audio.minute_sounds, 0)

    def test_video_pause_interrupts_a_blocked_decoder_without_counting_source_failure(self):
        self.flag.write_text('0')
        reading, killed = threading.Event(), threading.Event()

        class StopLoop(Exception):
            pass

        class PausedWake:
            def set(self):
                pass

            def clear(self):
                pass

            def wait(self, timeout):
                raise StopLoop

        class Process:
            def __init__(self):
                self.stdout = self
                self.waited = False

            def read(self, count):
                reading.set()
                killed.wait(3)
                return b''

            def kill(self):
                killed.set()

            def wait(self):
                self.waited = True
                return 0

        frames = watch.FrameSource()
        frames.want(60)
        frames.wake_decoder = PausedWake()
        process = Process()
        errors = []

        def run():
            try:
                frames.run()
            except StopLoop:
                pass
            except Exception as exc:
                errors.append(exc)

        with patch.object(watch, 'PUSH_ENABLED', False), \
                patch.object(watch.subprocess, 'Popen', lambda *a, **kw: process):
            thread = threading.Thread(target=run, daemon=True)
            thread.start()
            self.assertTrue(reading.wait(1))
            self.flag.write_text('1')
            thread.join(2.5)
            killed.set()
        self.assertFalse(thread.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(process.waited)
        self.assertIsNone(frames.frame)
        self.assertEqual(frames.push_down_until, 0, 'intentional pause is not a failed source')

    def test_corrupt_or_unreadable_privacy_flag_does_not_enable_acquisition(self):
        self.flag.write_text('corrupt')
        self.assertTrue(watch.privacy_flag())
        self.assertIsNone(watch.FrameSource().desired_mode())

    def test_motor_and_recovery_boundary_rechecks_privacy_during_a_search(self):
        with patch.object(watch.camapi, 'goto', forbidden), patch.object(watch.camapi, 'aim', forbidden), \
                patch.object(watch.urllib.request, 'urlopen', forbidden):
            with self.assertRaises(watch.camapi.Busy):
                watch.CamAdapter.goto(100, 100)
            with self.assertRaises(watch.camapi.Busy):
                watch.CamAdapter.aim(0.2, 0)
            self.assertFalse(watch.cam_action('/x/reboot.cgi'))

    def test_pause_while_sound_model_loads_discards_episode_without_classification(self):
        self.flag.write_text('0')
        audio = watch.Audio(queue.Queue())
        audio.levels = [-10.0] * 10
        audio.ring.append(watch.np.ones(audio.BLOCK, dtype=watch.np.int16))

        def loading_model():
            self.flag.write_text('1')
            audio.pause()
            return types.SimpleNamespace(classify=forbidden)

        audio.get_model = loading_model
        audio.close_episode([(9998, 4, -10)], 4, -60, -10)
        self.assertTrue(audio.events.empty())


if __name__ == '__main__':
    unittest.main()
