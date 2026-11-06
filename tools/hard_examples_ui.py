#!/usr/bin/env python3
"""Review frozen whole-frame hard examples on localhost; submit labels explicitly.

    python3 tools/hard_examples_ui.py --manifest RUN/manifest.json --queue RUN/review_queue.json \
        --annotations RUN/owner_annotations.jsonl --port 8788

No labels are written on page load or navigation. Test partitions are omitted
unless --include-sealed is supplied. This tool never trains, uploads or deploys.
"""
import argparse
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import mimetypes
import os
from pathlib import Path
import secrets
import stat
import threading
import time
from urllib.parse import parse_qs, urlsplit

from hard_examples import canonical, load_annotations, validate

PAGE = Path(__file__).with_suffix('.html')
TAGS = {'tiny', 'occluded', 'closeup', 'resting', 'day', 'ir'}
IMAGE_LIMIT = 32 * 1024 * 1024
BODY_LIMIT = 65536


class Conflict(ValueError):
    """A replacement needs the latest saved review revision."""


class Review:
    def __init__(self, manifest, queue, annotations, include_sealed=False):
        self.manifest_path = Path(manifest).resolve()
        self.manifest = json.loads(self.manifest_path.read_text())
        validate(self.manifest, {}, verify_files=False)
        queue = json.loads(Path(queue).read_text())
        if not isinstance(queue, dict) or queue.get('manifest_sha256') != self.manifest['manifest_sha256']:
            raise ValueError('Review queue does not match the frozen manifest.')
        if not isinstance(queue.get('entries'), list):
            raise ValueError('Review queue needs entries.')
        self.annotations_path = Path(annotations).absolute()
        if self.annotations_path.suffix.lower() != '.jsonl':
            raise ValueError('Use a .jsonl path for append-only owner review history.')
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        entries = {row['id']: row for row in self.manifest['entries']}
        self.items, self.by_id = [], {}
        self.excluded = 0
        allowed = {'train', 'dev', 'test'} if include_sealed else {'train', 'dev'}
        self.include_sealed = include_sealed
        seen_queue = set()
        for proposed in queue['entries']:
            ident = proposed.get('id') if isinstance(proposed, dict) else None
            if ident not in entries or ident in seen_queue:
                raise ValueError('Review queue contains an unknown or duplicate identity.')
            seen_queue.add(ident)
            row = dict(entries[ident])
            for key in ('sha256', 'path', 'group', 'split'):
                if key in proposed and proposed[key] != row[key]:
                    raise ValueError('Review queue changed frozen frame provenance.')
            if row['split'] not in allowed:
                self.excluded += 1
                continue
            row['reasons'] = proposed.get('reasons', row.get('reasons', []))
            self.items.append(row)
            self.by_id[ident] = row
        self._reviews()  # Validate existing history without creating any path.

    def _reviews(self):
        if self.annotations_path.is_symlink():
            raise ValueError('Annotation history must be a regular file, not a link.')
        if not self.annotations_path.exists():
            return {}
        if not self.annotations_path.is_file():
            raise ValueError('Annotation history must be a regular file, not a link.')
        result = load_annotations(self.annotations_path, self.manifest)['annotations']
        return {ident: row for ident, row in result.items() if row.get('source') == 'owner'}

    @staticmethod
    def revision(annotation):
        return hashlib.sha256(canonical(annotation)).hexdigest() if annotation else None

    def state(self):
        with self.lock:
            reviews = self._reviews()
            items = []
            for row in self.items:
                annotation = reviews.get(row['id'])
                items.append({key: row.get(key) for key in
                              ('id', 'sha256', 'width', 'height', 'day', 'capture_time', 'group', 'split', 'reasons')}
                             | {'annotation': annotation, 'revision': self.revision(annotation)})
            return {'items': items, 'reviewed': sum(row['id'] in reviews for row in self.items),
                    'total': len(items), 'excluded': self.excluded, 'include_sealed': self.include_sealed,
                    'manifest_sha256': self.manifest['manifest_sha256'], 'token': self.token}

    def image(self, ident):
        row = self.by_id.get(ident)
        if row is None:
            raise FileNotFoundError('This frame is not in the review queue.')
        path = Path(row['path'])
        if not path.is_absolute():
            path = self.manifest_path.parent / path
        fd = None
        try:
            fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
            info = os.fstat(fd)
            if not stat.S_ISREG(info.st_mode) or not 1 <= info.st_size <= IMAGE_LIMIT:
                raise FileNotFoundError('The frozen image is unavailable.')
            with os.fdopen(fd, 'rb') as handle:
                fd = None
                data = handle.read(IMAGE_LIMIT + 1)
            if len(data) > IMAGE_LIMIT or hashlib.sha256(data).hexdigest() != row['sha256']:
                raise Conflict('The image changed after mining. Create a new frozen manifest before reviewing it.')
            kind = mimetypes.guess_type(path.name)[0]
            if kind not in ('image/jpeg', 'image/png', 'image/webp'):
                raise FileNotFoundError('The frozen image format is unsupported.')
            return data, kind
        except OSError:
            raise FileNotFoundError('The frozen image is unavailable.') from None
        finally:
            if fd is not None:
                os.close(fd)

    def submit(self, data):
        if not isinstance(data, dict):
            raise ValueError('A review must be an object.')
        if set(data) - {'id', 'label', 'boxes', 'tags', 'replace', 'revision'}:
            raise ValueError('Only a frame identity, label, boxes, tags and replacement revision may be submitted.')
        ident = data.get('id')
        row = self.by_id.get(ident) if isinstance(ident, str) else None
        if row is None:
            raise ValueError('This frame is not in the review queue.')
        label, boxes, tags = data.get('label'), data.get('boxes', []), data.get('tags', [])
        if label not in ('visible', 'absent', 'uncertain'):
            raise ValueError('Choose Visible pet, No visible pet, or Uncertain explicitly.')
        if not isinstance(boxes, list) or len(boxes) > 20:
            raise ValueError('Use at most 20 visible-part boxes.')
        for box in boxes:
            if not isinstance(box, list) or len(box) != 4 or not all(
                    isinstance(v, (float, int)) and not isinstance(v, bool) and math.isfinite(v) and 0 <= v <= 1 for v in box):
                raise ValueError('Boxes must contain four finite normalized coordinates.')
            if box[0] >= box[2] or box[1] >= box[3]:
                raise ValueError('Each box needs positive width and height.')
        if label == 'visible' and not boxes:
            raise ValueError('Draw at least one box around the visible pet part before saving Visible pet.')
        if label in ('absent', 'uncertain') and boxes:
            raise ValueError('No visible pet and Uncertain reviews must not retain positive boxes.')
        if not isinstance(tags, list) or not all(isinstance(t, str) and t in TAGS for t in tags) or len(tags) != len(set(tags)):
            raise ValueError('Choose only supported pose and lighting tags.')
        if 'day' in tags and 'ir' in tags:
            raise ValueError('Choose either day or infrared, or leave lighting unmarked.')
        if data.get('replace', False) not in (True, False) or type(data.get('replace', False)) is not bool:
            raise ValueError('Replacement must be explicit.')
        with self.lock:
            previous = self._reviews().get(ident)
            if previous is not None:
                if data.get('replace') is not True:
                    raise Conflict('This frame already has an owner review. Enable replacement to save an edit.')
                if data.get('revision') != self.revision(previous):
                    raise Conflict('This review changed. Reload before replacing it.')
            elif data.get('replace') is True or data.get('revision') is not None:
                raise Conflict('There is no matching review to replace. Reload first.')
            self.image(ident)  # Recheck original bytes before asserting a visual label.
            annotation = {'id': ident, 'manifest_sha256': self.manifest['manifest_sha256'],
                          'sha256': row['sha256'], 'group': row['group'], 'split': row['split'],
                          'label': label, 'boxes': boxes, 'tags': tags,
                          'source': 'owner', 'reviewer': 'local-owner', 'attestation': 'owner-submitted',
                          'ts': time.time(), 'replaces': self.revision(previous)}
            self.annotations_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            fd = os.open(self.annotations_path, os.O_APPEND | os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            with os.fdopen(fd, 'ab') as handle:
                if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                    raise ValueError('Annotation history must be a regular file.')
                # JSONL written by another reviewer may end without a newline.
                size = os.fstat(handle.fileno()).st_size
                last = os.pread(handle.fileno(), 1, size - 1) if size else b''
                separator = b'\n' if last and last != b'\n' else b''
                handle.write(separator + canonical(annotation).replace(b'\n', b'') + b'\n')
                handle.flush()
                os.fsync(handle.fileno())
            return {'annotation': annotation, 'revision': self.revision(annotation)}


def handler(review):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass

        def respond(self, status, value, kind='application/json'):
            body = value if isinstance(value, bytes) else json.dumps(value, allow_nan=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', kind)
            self.send_header('Content-Length', str(len(body)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.send_header('Referrer-Policy', 'no-referrer')
            self.send_header('Content-Security-Policy', "default-src 'self'; img-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'")
            self.end_headers()
            self.wfile.write(body)

        def local_host(self):
            return self.headers.get('Host') in (f'127.0.0.1:{self.server.server_port}', f'localhost:{self.server.server_port}')

        def do_GET(self):
            if not self.local_host():
                return self.respond(403, {'error': 'Use the localhost review address.'})
            url = urlsplit(self.path)
            try:
                if url.path == '/':
                    return self.respond(200, PAGE.read_bytes(), 'text/html; charset=utf-8')
                if url.path == '/api/state':
                    return self.respond(200, review.state())
                if url.path == '/api/frame':
                    ident = parse_qs(url.query).get('id', [''])[0]
                    body, kind = review.image(ident)
                    return self.respond(200, body, kind)
                return self.respond(404, {'error': 'Not found.'})
            except FileNotFoundError as error:
                return self.respond(404, {'error': str(error)})
            except Conflict as error:
                return self.respond(409, {'error': str(error)})
            except ValueError as error:
                return self.respond(400, {'error': str(error)})

        def do_POST(self):
            origin = self.headers.get('Origin')
            expected = {f'http://127.0.0.1:{self.server.server_port}', f'http://localhost:{self.server.server_port}'}
            if not self.local_host() or origin not in expected or not hmac.compare_digest(
                    self.headers.get('X-Hard-Examples-Token', ''), review.token):
                return self.respond(403, {'error': 'This review submission must come from the local review page.'})
            if urlsplit(self.path).path != '/api/annotation':
                return self.respond(404, {'error': 'Not found.'})
            if self.headers.get('Content-Type', '').split(';', 1)[0].strip().lower() != 'application/json':
                return self.respond(415, {'error': 'Send a JSON review.'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= BODY_LIMIT:
                    return self.respond(413, {'error': 'Review body is missing or too large.'})
                data = json.loads(self.rfile.read(length), parse_constant=lambda _s: (_ for _ in ()).throw(ValueError('Non-finite JSON value.')))
                return self.respond(200, review.submit(data))
            except Conflict as error:
                return self.respond(409, {'error': str(error)})
            except (ValueError, OSError) as error:
                return self.respond(400, {'error': str(error)})
    return Handler


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--queue', required=True)
    parser.add_argument('--annotations', required=True, help='Append-only owner review JSONL.')
    parser.add_argument('--port', type=int, default=8788)
    parser.add_argument('--include-sealed', action='store_true', help='Explicitly review test frames for evaluation; never use those labels for training.')
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error('port must be 1..65535')
    review = Review(args.manifest, args.queue, args.annotations, args.include_sealed)
    server = ThreadingHTTPServer(('127.0.0.1', args.port), handler(review))
    print(f'Review {len(review.items)} frames at http://127.0.0.1:{args.port} ({review.excluded} sealed/quarantined frames excluded).', flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == '__main__':
    main()
