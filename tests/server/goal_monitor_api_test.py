"""Monitoring is authenticated, read-only, and independent of camera acquisition."""
import http.client
import json
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('PETCAM_SITE_ENV', str(ROOT/'config/site.env.example'))
sys.path.insert(0, str(ROOT/'server'))
import app
import agentdb


class MonitoringApiTest(unittest.TestCase):
    def serve(self, authenticated):
        class Handler(app.Handler):
            def authed(self):
                return authenticated

            def log_message(self, *_args):
                pass

        server = app.AppServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join, 2)
        self.addCleanup(server.shutdown)
        return server.server_address[1]

    def get(self, port, path):
        connection = http.client.HTTPConnection('127.0.0.1', port, timeout=3)
        try:
            connection.request('GET', path)
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), json.loads(response.read())
        finally:
            connection.close()

    def test_anonymous_cannot_read_goal_monitoring(self):
        port = self.serve(False)
        with patch.object(app, 'usage'), patch.object(app, 'monitoring_goals') as projection:
            code, _, body = self.get(port, '/api/monitoring')
        self.assertEqual(code, 401)
        self.assertIn('login', body['error'])
        projection.assert_not_called()

    def test_signed_in_reads_existing_projection_even_when_camera_is_paused(self):
        payload = {'goals': [{'id': 'goal:cpu', 'kind': 'goal', 'status': {'state': 'monitoring'}}], 'available': True}
        port = self.serve(True)
        with patch.object(app, 'usage'), patch.object(app, 'privacy_requested', return_value=True), \
             patch.object(app, 'monitoring_goals', return_value=payload), \
             patch.object(app, 'prudynt', side_effect=AssertionError('camera accessed')), \
             patch.object(app._camsh, 'run', side_effect=AssertionError('camera accessed')):
            code, headers, body = self.get(port, '/api/monitoring')
        self.assertEqual(code, 200)
        self.assertEqual(body, payload)
        self.assertEqual(headers.get('Cache-Control'), 'no-store')

    def test_missing_database_is_reported_without_creating_it(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td)/'never-created.db'
            with patch.object(agentdb, 'DB', str(path)):
                result = app.monitoring_goals()
            self.assertFalse(result['available'])
            self.assertFalse(path.exists())

    def test_linked_closed_goal_work_outside_dashboard_limit_is_readable(self):
        with tempfile.TemporaryDirectory() as td:
            path = str(Path(td)/'agent.db')
            with patch.object(agentdb, 'DB', path):
                with agentdb.db() as c:
                    c.execute("INSERT INTO threads(id,title,kind,status,created,updated) VALUES('goal:cpu','CPU investigation','improve','closed',1,1)")
                    for n in range(45):
                        c.execute('INSERT INTO threads(id,title,created,updated) VALUES(?,?,?,?)', (f'report:{n}', 'Newer request', n+2, n+2))
                    c.execute("INSERT INTO log(ts,actor,thread,kind,text) VALUES(1,'worker','goal:cpu','result','The attempt was rejected; goal remains monitored.')")
                before = Path(path).read_bytes()
                result = app.agent_dashboard('goal:cpu')
                self.assertFalse(any(t['id']=='goal:cpu' for t in result['threads']))
                self.assertEqual(result['selected_thread']['status'], 'closed')
                self.assertIn('rejected', result['selected_thread']['log'][0]['text'])
                self.assertEqual(Path(path).read_bytes(), before, 'activity read mutated existing state')
            c = sqlite3.connect(path)
            self.assertEqual(c.execute('SELECT count(*) FROM threads').fetchone()[0], 46)
            c.close()


if __name__ == '__main__':
    unittest.main()
