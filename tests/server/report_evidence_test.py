"""Retained report photos are authenticated, bounded evidence, never live capture."""
import hashlib
import http.client
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault('PETCAM_SITE_ENV', str(ROOT / 'config/site.env.example'))
sys.path.insert(0, str(ROOT / 'server'))
import app
import requestboard


REPORT_ID = '20260101-120000'
JPEG = b'\xff\xd8\xff\xe0retained report image\xff\xd9'


class ReportEvidenceTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'reports'
        self.root.mkdir()
        self.folder = self.root / REPORT_ID
        self.folder.mkdir()
        self.report = {'id': REPORT_ID, 'ts': 1, 'note': 'Pet visible, status says absent',
                       'client': {'shown_status': 'Out of sight'}}
        (self.folder / 'report.json').write_text(json.dumps(self.report))
        (self.folder / 'reply.json').write_text(json.dumps({'status': 'answered', 'text': 'Investigating'}))
        self.patch = patch.object(app, 'REPORTS_DIR', str(self.root))
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def serve(self, authenticated=True):
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

    def get(self, port, query='id=' + REPORT_ID):
        connection = http.client.HTTPConnection('127.0.0.1', port, timeout=3)
        try:
            connection.request('GET', '/api/reports/frame?' + query)
            response = connection.getresponse()
            return response.status, dict(response.getheaders()), response.read()
        finally:
            connection.close()

    def original_hashes(self):
        return {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in self.folder.iterdir() if path.is_file()}

    def test_authenticated_photo_round_trip_is_read_only_and_independent_of_camera(self):
        (self.folder / 'frame.jpg').write_bytes(JPEG)
        before = self.original_hashes()
        port = self.serve()
        with patch.object(app, 'usage') as usage, \
             patch.object(app.Handler, 'camera_call', side_effect=AssertionError('camera pool used')), \
             patch.object(app, 'live_frame', side_effect=AssertionError('live capture used')), \
             patch.object(app._camsh, 'run', side_effect=AssertionError('camera accessed')):
            code, headers, body = self.get(port)
        self.assertEqual((code, body), (200, JPEG))
        self.assertEqual(headers['Content-Type'], 'image/jpeg')
        self.assertEqual(headers['Cache-Control'], 'no-store')
        usage.assert_not_called()
        self.assertEqual(self.original_hashes(), before)
        self.assertEqual(requestboard.load(self.root, REPORT_ID)['evidence']['frame'],
                         '/api/reports/frame?id=' + REPORT_ID)

    def test_anonymous_request_is_denied_before_evidence_is_opened(self):
        port = self.serve(False)
        with patch.object(requestboard, 'frame', side_effect=AssertionError('private file opened')):
            code, _, body = self.get(port)
        self.assertEqual(code, 401)
        self.assertIn('login', json.loads(body)['error'])

    def test_missing_photo_is_404_and_not_advertised(self):
        port = self.serve()
        code, _, body = self.get(port)
        self.assertEqual(code, 404)
        self.assertIn('unavailable', json.loads(body)['error'])
        self.assertNotIn('evidence', requestboard.load(self.root, REPORT_ID))

    def test_missing_report_is_404_even_when_a_photo_exists(self):
        (self.folder / 'frame.jpg').write_bytes(JPEG)
        (self.folder / 'report.json').unlink()
        port = self.serve()
        self.assertEqual(self.get(port)[0], 404)
        self.assertEqual(self.get(port, 'id=20260101-235959')[0], 404)

    def test_invalid_or_traversing_identifiers_are_400(self):
        port = self.serve()
        for query in ('', 'id=../../outside', 'id=20260101-120000%2f..',
                      'id=goal%3Acpu', 'id=20260101-120000extra',
                      'id=%EF%BC%92%EF%BC%90%EF%BC%92%EF%BC%96%EF%BC%90%EF%BC%91%EF%BC%90%EF%BC%91-120000'):
            with self.subTest(query=query):
                code, _, body = self.get(port, query)
                self.assertEqual(code, 400)
                self.assertIn('Invalid request', json.loads(body)['error'])

    def test_frame_symlink_escape_is_400_and_not_advertised(self):
        outside = Path(self.temp.name) / 'outside.jpg'
        outside.write_bytes(JPEG)
        (self.folder / 'frame.jpg').symlink_to(outside)
        port = self.serve()
        self.assertEqual(self.get(port)[0], 400)
        self.assertNotIn('evidence', requestboard.load(self.root, REPORT_ID))

    def test_symlink_to_another_report_photo_is_not_advertised(self):
        other = self.root / '20260101-130000'
        other.mkdir()
        (other / 'frame.jpg').write_bytes(JPEG)
        (self.folder / 'frame.jpg').symlink_to(other / 'frame.jpg')
        self.assertNotIn('evidence', requestboard.load(self.root, REPORT_ID))
        with self.assertRaises(ValueError):
            requestboard.frame(self.root, REPORT_ID)

    def test_report_directory_symlink_escape_is_400(self):
        outside = Path(self.temp.name) / 'outside'
        outside.mkdir()
        (outside / 'report.json').write_text(json.dumps(self.report))
        (outside / 'frame.jpg').write_bytes(JPEG)
        alias = self.root / '20260101-140000'
        alias.symlink_to(outside, target_is_directory=True)
        port = self.serve()
        self.assertEqual(self.get(port, 'id=' + alias.name)[0], 400)

    def test_oversized_photo_is_not_read_or_advertised(self):
        path = self.folder / 'frame.jpg'
        with path.open('wb') as handle:
            handle.write(JPEG)
            handle.truncate(requestboard.FRAME_MAX_BYTES + 1)
        port = self.serve()
        self.assertEqual(self.get(port)[0], 404)
        self.assertNotIn('evidence', requestboard.load(self.root, REPORT_ID))

    def test_wrong_magic_and_non_regular_file_are_not_served(self):
        path = self.folder / 'frame.jpg'
        path.write_bytes(b'not a JPEG file')
        port = self.serve()
        self.assertEqual(self.get(port)[0], 404)
        self.assertNotIn('evidence', requestboard.load(self.root, REPORT_ID))
        path.unlink()
        os.mkfifo(path)
        self.assertEqual(self.get(port)[0], 404)
        self.assertNotIn('evidence', requestboard.load(self.root, REPORT_ID))

    def test_frame_availability_does_not_change_workflow_revision(self):
        initial = requestboard.load(self.root, REPORT_ID)
        moved = requestboard.move(self.root, REPORT_ID, 'monitoring', initial['revision'], 'Owner',
                                  'Observe during ordinary use')
        before = self.original_hashes()
        (self.folder / 'frame.jpg').write_bytes(JPEG)
        available = requestboard.load(self.root, REPORT_ID)
        self.assertEqual(available['revision'], moved['revision'])
        self.assertEqual(available['workflow'], moved['workflow'])
        self.assertIn('evidence', available)
        (self.folder / 'frame.jpg').unlink()
        absent = requestboard.load(self.root, REPORT_ID)
        self.assertEqual(absent['revision'], moved['revision'])
        self.assertNotIn('evidence', absent)
        self.assertEqual(self.original_hashes(), before)


if __name__ == '__main__':
    unittest.main()
