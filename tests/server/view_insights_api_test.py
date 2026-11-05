"""View aggregation uses bounded database evidence and never calls the camera."""
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('PETCAM_SITE_ENV', str(ROOT / 'config/site.env.example'))
sys.path.insert(0, str(ROOT / 'server'))
import app


class ViewAPI(unittest.TestCase):
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
                result = app.view_insights('1')
                self.assertEqual(result['days'], 1)
                self.assertGreater(result['observed_seconds'], 0)
                self.assertEqual(result['views'][0]['name'], 'Corner')
                with patch.object(app, 'dbc', side_effect=AssertionError('cache missed')):
                    self.assertIs(app.view_insights(1), result)
                presets.return_value = [{'name': 'Renamed', 'x': 100, 'y': 100}]
                self.assertEqual(app.view_insights(1)['views'][0]['name'], 'Renamed')
                for invalid in (0, 8, 'bad'):
                    with self.assertRaises(ValueError):
                        app.view_insights(invalid)
            app._view_insights_cache.clear()


if __name__ == '__main__':
    unittest.main()
