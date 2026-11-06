"""View aggregation uses bounded database evidence and never calls the camera."""
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import unittest
from unittest.mock import MagicMock, patch

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('PETCAM_SITE_ENV', str(ROOT / 'config/site.env.example'))
sys.path.insert(0, str(ROOT / 'server'))
import app


class ViewAPI(unittest.TestCase):
    def setUp(self):
        self.wait_worker()
        app._view_insights_cache.clear()
        app._view_insights_errors.clear()
        app._view_insights_fingerprint = None

    def tearDown(self):
        self.wait_worker()
        app._view_insights_cache.clear()
        app._view_insights_errors.clear()

    def wait_worker(self):
        worker = app._view_insights_worker
        if worker is not None:
            worker.join(2)
            self.assertFalse(worker.is_alive(), 'test leaked a statistics worker')

    def test_cold_request_returns_pending_while_database_read_is_blocked(self):
        entered, release = threading.Event(), threading.Event()
        con = MagicMock()
        con.execute.side_effect = lambda *args: iter(())
        result = {}

        def blocked_db():
            entered.set()
            release.wait(3)
            return con

        app._view_insights_cache.clear()
        with patch.object(app, 'dbc', side_effect=blocked_db), patch.object(app, 'load_presets', return_value=[]):
            request = threading.Thread(target=lambda: result.update(app.view_insights(7)), daemon=True)
            request.start()
            try:
                self.assertTrue(entered.wait(2), 'database read never started')
                request.join(.2)
                self.assertFalse(request.is_alive(), 'cold HTTP request blocks on full history scan')
                self.assertEqual(result['status'], 'pending')
            finally:
                release.set()
                request.join(2)
                worker = getattr(app, '_view_insights_worker', None)
                if worker is not None:
                    worker.join(2)
        app._view_insights_cache.clear()

    def test_bounded_query_cache_and_preset_invalidation(self):
        with tempfile.TemporaryDirectory() as td:
            db = str(Path(td) / 'petcam.db')
            c = sqlite3.connect(db)
            c.executescript('CREATE TABLE obs(ts REAL, score REAL, box TEXT); CREATE TABLE track(ts REAL, pan INT, tilt INT, act TEXT, moves INT);')
            for t in range(800, 1001, 3):
                c.execute('INSERT INTO obs VALUES (?, .8, ?)', (t, '[0.1,0.2,0.3,0.4]'))
                c.execute("INSERT INTO track VALUES (?,100,100,'idle',0)", (t,))
            c.commit()
            c.close()
            app._view_insights_cache.clear()
            with patch.object(app, 'DB_PATH', db), patch.object(app, 'load_presets', return_value=[{'name': 'Corner', 'x': 100, 'y': 100}]) as presets, \
                    patch.object(app.time, 'time', return_value=1000), patch.object(app, 'position', side_effect=AssertionError('camera queried')):
                self.assertEqual(app.view_insights('1')['status'], 'pending')
                self.wait_worker()
                result = app.view_insights('1')
                self.assertEqual(result['days'], 1)
                self.assertEqual(result['status'], 'ready')
                self.assertGreater(result['observed_seconds'], 0)
                self.assertEqual(result['views'][0]['name'], 'Corner')
                with patch.object(app, 'dbc', side_effect=AssertionError('cache missed')):
                    self.assertEqual(app.view_insights(1), result)
                presets.return_value = [{'name': 'Renamed', 'x': 100, 'y': 100}]
                self.assertEqual(app.view_insights(1)['status'], 'pending')
                self.wait_worker()
                self.assertEqual(app.view_insights(1)['views'][0]['name'], 'Renamed')
                for invalid in (0, 8, 'bad'):
                    with self.assertRaises(ValueError):
                        app.view_insights(invalid)
            app._view_insights_cache.clear()

    def test_one_worker_serves_duplicate_requests_without_parallel_day_scans(self):
        entered, release = threading.Event(), threading.Event()

        def compute(days, presets):
            entered.set()
            release.wait(3)
            return {'days': days, 'until': 1000}

        with patch.object(app, 'load_presets', return_value=[]), \
                patch.object(app, '_compute_view_insights', side_effect=compute) as run:
            try:
                self.assertEqual(app.view_insights(7)['status'], 'pending')
                self.assertTrue(entered.wait(2))
                for days in (7, 7, 1, 3, 7):
                    self.assertEqual(app.view_insights(days)['status'], 'pending')
                self.assertEqual(run.call_count, 1)
                self.assertTrue(app._view_insights_worker.daemon)
            finally:
                release.set()
                self.wait_worker()
            self.assertEqual(app.view_insights(7)['status'], 'ready')
            self.assertEqual(app.view_insights(1)['status'], 'pending')
            self.wait_worker()
            self.assertEqual(run.call_count, 2)

    def test_stale_result_remains_available_during_refresh(self):
        entered, release = threading.Event(), threading.Event()

        def refresh(days, presets):
            entered.set()
            release.wait(3)
            return {'days': days, 'until': 2000}

        with patch.object(app, 'load_presets', return_value=[]), \
                patch.object(app.time, 'monotonic', return_value=100) as clock, \
                patch.object(app, '_compute_view_insights', return_value={'days': 7, 'until': 1000}) as compute:
            app.view_insights(7)
            self.wait_worker()
            clock.return_value = 401
            compute.side_effect = refresh
            try:
                stale = app.view_insights(7)
                self.assertEqual((stale['status'], stale['until'], stale['refreshing']), ('ready', 1000, True))
                self.assertTrue(entered.wait(2))
                self.assertEqual(app.view_insights(7), stale)
                self.assertEqual(compute.call_count, 2)
            finally:
                release.set()
                self.wait_worker()
            fresh = app.view_insights(7)
            self.assertEqual((fresh['until'], fresh['refreshing']), (2000, False))

    def test_preset_change_discards_inflight_old_geometry(self):
        entered, release = threading.Event(), threading.Event()

        def compute(days, presets):
            entered.set()
            release.wait(3)
            return {'days': days, 'views': presets}

        with patch.object(app, 'load_presets', return_value=[{'name': 'Old', 'x': 1, 'y': 1}]) as presets, \
                patch.object(app, '_compute_view_insights', side_effect=compute) as run:
            try:
                app.view_insights(7)
                self.assertTrue(entered.wait(2))
                presets.return_value = [{'name': 'New', 'x': 2, 'y': 2}]
                self.assertEqual(app.view_insights(7)['status'], 'pending')
                self.assertEqual(run.call_count, 1)
            finally:
                release.set()
                self.wait_worker()
            self.assertEqual(app._view_insights_cache, {})
            self.assertEqual(app.view_insights(7)['status'], 'pending')
            self.wait_worker()
            self.assertEqual(app.view_insights(7)['views'][0]['name'], 'New')

    def test_stale_window_keeps_polling_while_another_window_owns_worker(self):
        entered, release = threading.Event(), threading.Event()

        def compute(days, presets):
            entered.set()
            release.wait(3)
            return {'days': days, 'until': 2000}

        with patch.object(app, 'load_presets', return_value=[]), \
                patch.object(app.time, 'monotonic', return_value=100) as clock, \
                patch.object(app, '_compute_view_insights', return_value={'days': 7, 'until': 1000}) as run:
            app.view_insights(7)
            self.wait_worker()
            clock.return_value = 401
            run.side_effect = compute
            try:
                app.view_insights(1)
                self.assertTrue(entered.wait(2))
                stale = app.view_insights(7)
                self.assertEqual((stale['until'], stale['refreshing']), (1000, True))
                self.assertEqual(stale['retry_after_s'], 5)
                self.assertEqual(run.call_count, 2)
            finally:
                release.set()
                self.wait_worker()
            self.assertTrue(app.view_insights(7)['refreshing'])
            self.wait_worker()
            self.assertEqual(app.view_insights(7)['until'], 2000)

    def test_failed_refresh_preserves_completed_summary_and_cools_down(self):
        with patch.object(app, 'load_presets', return_value=[]), \
                patch.object(app.time, 'monotonic', return_value=100) as clock, \
                patch.object(app, '_compute_view_insights', return_value={'days': 7, 'until': 1000}) as run:
            app.view_insights(7)
            self.wait_worker()
            clock.return_value = 401
            run.side_effect = RuntimeError('private SQL error')
            self.assertTrue(app.view_insights(7)['refreshing'])
            self.wait_worker()
            for stamp in (401, 440, 460):
                clock.return_value = stamp
                result = app.view_insights(7)
                self.assertEqual((result['status'], result['until'], result['refreshing']), ('ready', 1000, False))
                self.assertEqual(result['retry_after_s'], 60)
                self.assertTrue(result['refresh_error'])
                self.assertNotIn('private', result['refresh_error'])
            self.assertEqual(run.call_count, 2)
            clock.return_value = 461
            run.side_effect = None
            run.return_value = {'days': 7, 'until': 2000}
            self.assertTrue(app.view_insights(7)['refreshing'])
            self.wait_worker()
            self.assertEqual(app.view_insights(7)['until'], 2000)
            self.assertNotIn('refresh_error', app.view_insights(7))

    def test_errors_are_generic_and_retry_only_after_cooldown(self):
        with patch.object(app, 'load_presets', return_value=[]), \
                patch.object(app.time, 'monotonic', return_value=100) as clock, \
                patch.object(app, '_compute_view_insights', side_effect=RuntimeError('private database path')) as run:
            self.assertEqual(app.view_insights(7)['status'], 'pending')
            self.wait_worker()
            for stamp in (100, 120, 159):
                clock.return_value = stamp
                result = app.view_insights(7)
                self.assertEqual((result['status'], result['retry_after_s']), ('error', 60))
                self.assertNotIn('private', result['error'])
            self.assertEqual(run.call_count, 1)
            clock.return_value = 160
            run.side_effect = None
            run.return_value = {'days': 7, 'until': 1000}
            self.assertEqual(app.view_insights(7)['status'], 'pending')
            self.wait_worker()
            self.assertEqual(app.view_insights(7)['status'], 'ready')
            self.assertEqual(run.call_count, 2)

    def test_cache_retains_all_seven_windows_for_current_presets(self):
        with patch.object(app, 'load_presets', return_value=[]), \
                patch.object(app, '_compute_view_insights', side_effect=lambda days, presets: {'days': days}) as run:
            for days in range(1, 8):
                self.assertEqual(app.view_insights(days)['status'], 'pending')
                self.wait_worker()
            self.assertEqual(len(app._view_insights_cache), 7)
            for days in range(1, 8):
                self.assertEqual(app.view_insights(days)['status'], 'ready')
            self.assertEqual(run.call_count, 7)


if __name__ == '__main__':
    unittest.main()
