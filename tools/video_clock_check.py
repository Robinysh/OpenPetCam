#!/usr/bin/env python3
"""Capture RTP headers and check H.264 picture/configuration clocks without media.

Examples (use a local authenticated relay/tunnel; do not put secrets in arguments):
  python3 tools/video_clock_check.py capture --url rtsp://127.0.0.1:8554/cam --stream-rows \
      --seconds 180 --output eval/video-clock-before.json
  python3 tools/video_clock_check.py analyze eval/video-clock-before.json \
      --require-duration 150 --output eval/video-clock-before-summary.json --strict

Legacy header JSON from the original camera investigation is also supported.
RTP media spans are compared against one another, not their initial offsets.
Arrival-clock ratios are descriptive only: SSH/TCP buffering invalidates short
arrival windows as precise clock measurements. This tool cannot prove SDK capture
timestamps or A/V lip sync; those require independent boundary/physical checks.

Protocol sources: RFC3550 sections5/6; RFC6184 sections5.1/5.7/5.8. The capture
parser handles H.264 single NAL, STAP-A/B and FU-A/B packets. Interleaved MTAP
packetization and H.264 SVC/MVC are explicitly unsupported rather than guessed.
Use --stream-rows for long captures on small hosts; analyze saved headers offline
on the laptop because the analysis step intentionally loads the saved document.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import socket
import struct
import sys
import tempfile
import time
from urllib.parse import urlsplit


VCL = {1, 2, 3, 4, 5}
CONFIG = {6, 7, 8, 9, 10, 11, 12}


def signed_delta(current, previous, bits=32):
    half = 1 << (bits - 1)
    return ((int(current) - int(previous) + half) % (1 << bits)) - half


def first_mb(nal):
    """Read first_mb_in_slice only; this is not a complete H.264 AU decoder."""
    body = bytearray()
    zeros = 0
    for value in nal[1:]:
        if zeros >= 2 and value == 3:
            zeros = 0
            continue
        body.append(value)
        zeros = zeros + 1 if value == 0 else 0
    bits = ''.join(f'{byte:08b}' for byte in body[:8])
    leading = len(bits) - len(bits.lstrip('0'))
    if not bits or leading == len(bits) or 2 * leading + 1 > len(bits):
        return None
    return int(bits[leading:2 * leading + 1], 2) - 1


def parse_rtp(packet, kind='video'):
    if len(packet) < 12:
        raise ValueError('short RTP header')
    flags, pt, sequence, timestamp, ssrc = struct.unpack('!BBHII', packet[:12])
    if flags >> 6 != 2:
        raise ValueError('unsupported RTP version')
    offset = 12 + 4 * (flags & 15)
    if offset > len(packet):
        raise ValueError('truncated RTP CSRC list')
    if flags & 16:
        if offset + 4 > len(packet):
            raise ValueError('truncated RTP extension')
        offset += 4 + 4 * int.from_bytes(packet[offset + 2:offset + 4], 'big')
        if offset > len(packet):
            raise ValueError('truncated RTP extension data')
    end = len(packet)
    if flags & 32:
        padding = packet[-1]
        if not padding or padding > end - offset:
            raise ValueError('invalid RTP padding')
        end -= padding
    payload = packet[offset:end]
    result = {'seq': sequence, 'rtp_ts': timestamp, 'ssrc': ssrc,
              'marker': bool(pt & 128), 'payload_type': pt & 127,
              'payload_bytes': len(payload), 'nal_types': [], 'nal_starts': [],
              'first_slices': 0, 'unsupported_packetization': False}
    if kind != 'video':
        return result
    if not payload:
        raise ValueError('empty H.264 RTP payload')
    nal_type = payload[0] & 31
    nals = []
    if 1 <= nal_type <= 23:
        nals = [payload]
    elif nal_type in (24, 25):
        pos = 1 if nal_type == 24 else 3
        if pos > len(payload):
            raise ValueError('truncated H.264 aggregation header')
        while pos < len(payload):
            if pos + 2 > len(payload):
                raise ValueError('truncated H.264 aggregation size')
            size = int.from_bytes(payload[pos:pos + 2], 'big')
            pos += 2
            if not size or pos + size > len(payload):
                raise ValueError('invalid H.264 aggregate NAL size')
            nals.append(payload[pos:pos + size])
            pos += size
    elif nal_type in (28, 29):
        pos = 2 if nal_type == 28 else 4
        if len(payload) < pos:
            raise ValueError('truncated H.264 fragmentation header')
        reconstructed = payload[1] & 31
        result['nal_types'] = [reconstructed]
        if payload[1] & 128:
            nals = [bytes([(payload[0] & 224) | reconstructed]) + payload[pos:]]
    else:
        result['unsupported_packetization'] = True
        return result
    if nals:
        result['nal_types'] = [nal[0] & 31 for nal in nals]
        result['nal_starts'] = list(result['nal_types'])
        result['first_slices'] = sum(first_mb(nal) == 0 for nal in nals if nal[0] & 31 in (1, 2, 5))
        if any(nal[0] & 128 or (nal[0] & 31) in (14, 15, 19, 20, 21) for nal in nals):
            result['unsupported_packetization'] = True
    return result


def parse_rtcp(packet, arrival_s, channel):
    """Retain only sender-report clock pairs from a compound RTCP packet."""
    reports, pos = [], 0
    while pos < len(packet):
        if len(packet) - pos < 4:
            raise ValueError('truncated RTCP header')
        flags, kind, words = struct.unpack('!BBH', packet[pos:pos + 4])
        size = 4 * (words + 1)
        if flags >> 6 != 2 or size < 4 or pos + size > len(packet):
            raise ValueError('invalid RTCP packet size/version')
        if kind == 200:
            if size < 28:
                raise ValueError('truncated RTCP sender report')
            ssrc, seconds, fraction, timestamp, count, octets = struct.unpack('!IIIIII', packet[pos + 4:pos + 28])
            reports.append({'t': arrival_s, 'channel': channel, 'ssrc': ssrc,
                            'ntp_seconds': seconds + fraction / 2**32,
                            'rtp_ts': timestamp, 'packet_count': count, 'octet_count': octets})
        pos += size
    return reports


def _tracks(document):
    if document.get('tracks'):
        return document['tracks']
    return [{'channel': i * 2, 'kind': control[0],
             'clock_rate': 90000 if control[0] == 'video' else 48000,
             'codec': 'H264' if control[0] == 'video' else 'unknown'}
            for i, control in enumerate(document.get('controls', []))]


def _sequence(rows):
    missing = duplicates = reordered = gaps = 0
    previous = None
    for row in rows:
        value = int(row['seq'])
        if previous is not None:
            delta = (value - previous) % 65536
            if delta == 0:
                duplicates += 1
            elif delta < 32768:
                if delta > 1:
                    missing += delta - 1
                    gaps += 1
                previous = value
            else:
                reordered += 1
        else:
            previous = value
    return {'sequence_gap_events': gaps, 'missing_packets': missing,
            'duplicate_packets': duplicates, 'reordered_packets': reordered}


def _stream(rows, track):
    groups, unwrapped, previous = [], 0, None
    wraps = regressions = 0
    for row in rows:
        raw = int(row['rtp_ts'])
        if previous is None:
            unwrapped = raw
        else:
            delta = signed_delta(raw, previous)
            if delta < 0:
                regressions += 1
            if raw < previous and delta >= 0:
                wraps += 1
            unwrapped += delta
        if previous != raw:
            groups.append({'timestamp': unwrapped, 'raw': raw, 'first_t': float(row['t']),
                           'last_t': float(row['t']), 'nals': set(), 'first_slices': 0})
        group = groups[-1]
        group['last_t'] = float(row['t'])
        nals = row.get('nal_types')
        if nals is None:
            nals = [] if row.get('nal_type') is None else [row['nal_type']]
        group['nals'].update(nals)
        group['first_slices'] += int(row.get('first_slices', 0))
        previous = raw
    rate = int(track['clock_rate'])
    result = {'kind': track['kind'], 'channel': track['channel'], 'codec': track.get('codec'),
              'clock_rate': rate, 'ssrc': rows[0].get('ssrc'), 'packets': len(rows),
              'timestamp_groups': len(groups), 'timestamp_wraps': wraps,
              'timestamp_regressions': regressions, **_sequence(rows),
              'unsupported_packetization_packets': sum(bool(r.get('unsupported_packetization')) for r in rows)}
    result['rtp_span_s'] = (groups[-1]['timestamp'] - groups[0]['timestamp']) / rate if groups else 0
    result['arrival_span_s'] = float(rows[-1]['t']) - float(rows[0]['t'])
    result['group_step_ticks'] = dict(sorted(Counter(b['timestamp'] - a['timestamp'] for a, b in zip(groups, groups[1:])).items()))
    if track['kind'] == 'video':
        pictures = [g for g in groups if g['nals'] & VCL]
        result['picture_groups'] = len(pictures)
        result['idr_groups'] = sum(5 in g['nals'] for g in pictures)
        result['multiple_first_slice_groups'] = sum(g['first_slices'] > 1 for g in pictures)
        result['picture_group_step_ticks'] = dict(sorted(Counter(b['timestamp'] - a['timestamp'] for a, b in zip(pictures, pictures[1:])).items()))
        if pictures:
            start, end = groups.index(pictures[0]), groups.index(pictures[-1])
            config = [g for g in groups[start:end + 1] if g['nals'] and g['nals'] <= CONFIG]
            result['config_only_groups_between_pictures'] = len(config)
            result['config_only_before_idr'] = sum(a['nals'] and a['nals'] <= CONFIG and 5 in b['nals'] for a, b in zip(groups[start:end], groups[start + 1:end + 1]))
            result['picture_rtp_span_s'] = (pictures[-1]['timestamp'] - pictures[0]['timestamp']) / rate
            result['picture_arrival_span_s'] = pictures[-1]['last_t'] - pictures[0]['first_t']
        else:
            result.update(config_only_groups_between_pictures=0, config_only_before_idr=0,
                          picture_rtp_span_s=0, picture_arrival_span_s=0)
    return result


def analyze(document, fps=25.0, require_duration_s=0.0, max_drift_ratio=0.001,
            endpoint_tolerance_s=0.12):
    """Return diagnostics and a fail-closed clock gate, without retiming anything."""
    if fps <= 0 or require_duration_s < 0 or max_drift_ratio < 0 or endpoint_tolerance_s < 0:
        raise ValueError('invalid clock check limits')
    tracks = _tracks(document)
    by_channel = {track['channel']: track for track in tracks}
    rows = document.get('rows', [])
    streams = []
    issues = []
    if document.get('capture_error'):
        issues.append('capture ended with a transport/parser error: ' + str(document['capture_error']))
    for channel, track in by_channel.items():
        if track['kind'] == 'video' and str(track.get('codec', 'H264')).upper() != 'H264':
            issues.append(f'channel {channel}: this validator only supports H.264 video')
        matching = [row for row in rows if row.get('channel') == channel]
        sources = sorted({row.get('ssrc', 0) for row in matching})
        if len(sources) > 1:
            issues.append(f'channel {channel}: multiple SSRC epochs; do not concatenate their clocks')
        for source in sources:
            stream = _stream([row for row in matching if row.get('ssrc', 0) == source], track)
            streams.append(stream)
            for field in ('timestamp_regressions', 'missing_packets', 'reordered_packets',
                          'duplicate_packets', 'unsupported_packetization_packets'):
                if stream[field]:
                    issues.append(f'channel {channel}: {field}={stream[field]}')
            if stream['kind'] == 'video':
                if stream['config_only_groups_between_pictures']:
                    issues.append(f'channel {channel}: configuration-only packets consume distinct internal RTP timestamps')
                if stream['multiple_first_slice_groups']:
                    issues.append(f'channel {channel}: multiple first-slice starts share one RTP timestamp; AU assignment is ambiguous')
                if stream['picture_groups'] < 2:
                    issues.append(f'channel {channel}: fewer than two H.264 picture groups')
                stream['nominal_picture_span_s'] = max(0, stream['picture_groups'] - 1) / fps
                stream['media_excess_over_nominal_picture_count_s'] = stream['picture_rtp_span_s'] - stream['nominal_picture_span_s']
    videos = [s for s in streams if s['kind'] == 'video']
    audios = [s for s in streams if s['kind'] == 'audio']
    if not videos:
        issues.append('no video stream was observed')
    if not audios:
        issues.append('no audio RTP clock reference was observed')
    comparisons = []
    for video in videos:
        for audio in audios:
            video_span, audio_span = video['picture_rtp_span_s'], audio['rtp_span_s']
            excess = video_span - audio_span
            tolerance = endpoint_tolerance_s + max_drift_ratio * audio_span
            comparison = {'video_channel': video['channel'], 'audio_channel': audio['channel'],
                          'video_picture_rtp_span_s': video_span, 'audio_rtp_span_s': audio_span,
                          'video_minus_audio_span_s': excess,
                          'span_ratio_minus_one': video_span / audio_span - 1 if audio_span > 0 else None,
                          'allowed_endpoint_plus_drift_s': tolerance,
                          'within_tolerance': audio_span > 0 and abs(excess) <= tolerance}
            if not comparison['within_tolerance']:
                issues.append(f'video channel {video["channel"]}/audio channel {audio["channel"]}: media elapsed spans diverge')
            comparisons.append(comparison)
            if min(video['picture_arrival_span_s'], audio['arrival_span_s'], audio_span) + endpoint_tolerance_s < require_duration_s:
                issues.append('capture is shorter than the required validation interval')
    if not comparisons and require_duration_s:
        issues.append('duration gate requires both video and audio')
    sender_clocks = []
    for stream in streams:
        reports = [r for r in document.get('rtcp_sender_reports', [])
                   if r['channel'] == stream['channel'] + 1 and r['ssrc'] == stream['ssrc']]
        if len(reports) >= 2:
            elapsed = reports[-1]['ntp_seconds'] - reports[0]['ntp_seconds']
            media = sum(signed_delta(b['rtp_ts'], a['rtp_ts']) for a, b in zip(reports, reports[1:])) / stream['clock_rate']
            sender_clocks.append({'channel': stream['channel'], 'reports': len(reports),
                                  'ntp_elapsed_s': elapsed, 'rtp_elapsed_s': media,
                                  'ratio': media / elapsed if elapsed > 0 else None})
    return {'passed': not issues, 'issues': list(dict.fromkeys(issues)), 'streams': streams,
            'av_elapsed_comparisons': comparisons, 'rtcp_sender_clock_diagnostics': sender_clocks,
            'capture_wall_seconds': document.get('wall_seconds'),
            'limits': {'nominal_fps': fps, 'require_duration_s': require_duration_s,
                       'max_drift_ratio': max_drift_ratio, 'endpoint_tolerance_s': endpoint_tolerance_s},
            'limitations': [
                'Timestamp groups containing VCL are not a complete H.264 access-unit parse; multiple first-slice starts are flagged.',
                'Nominal picture count is diagnostic: real SDK/capture drops must not be filled with fabricated frames.',
                'Whole elapsed spans cancel initial A/V offsets; finite endpoints still contribute uncertainty.',
                'Arrival times and RTCP clock pairs are supplementary; network buffering or relay-generated RTCP can bias them.',
                'No camera SDK capture-time calibration, physical lip sync, or absence of recorder resets is established by RTP headers alone.',
            ]}


class RTSPCapture:
    """Minimal bounded interleaved-TCP RTSP reader for an existing local relay."""
    keepalive_interval_s = 20
    def __init__(self, url, seconds, row_sink=None, rtcp_sink=None):
        parsed = urlsplit(url)
        if parsed.scheme != 'rtsp' or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError('Use a local rtsp:// relay/tunnel without embedded credentials')
        self.url, self.seconds = url, seconds
        self.sock = socket.create_connection((parsed.hostname, parsed.port or 554), timeout=5)
        self.buffer = bytearray()
        self.sequence = 0
        self.session = None
        self.started = None
        self.rows, self.sender_reports, self.tracks = [], [], []
        self.row_sink, self.rtcp_sink = row_sink, rtcp_sink
        self.row_count = self.sender_report_count = 0

    def _message(self, deadline):
        while True:
            if self.buffer[:1] == b'$' and len(self.buffer) >= 4:
                size = int.from_bytes(self.buffer[2:4], 'big')
                if len(self.buffer) >= size + 4:
                    channel, data = self.buffer[1], bytes(self.buffer[4:4 + size])
                    del self.buffer[:4 + size]
                    return 'packet', channel, data
            elif self.buffer and self.buffer[:1] != b'$':
                end = self.buffer.find(b'\r\n\r\n')
                if end >= 0:
                    lines = bytes(self.buffer[:end]).decode('utf-8', errors='replace').split('\r\n')
                    headers = dict((key.lower(), value.strip()) for key, value in (line.split(':', 1) for line in lines[1:] if ':' in line))
                    size = int(headers.get('content-length', '0'))
                    if len(self.buffer) >= end + 4 + size:
                        body = bytes(self.buffer[end + 4:end + 4 + size])
                        del self.buffer[:end + 4 + size]
                        return 'response', (lines[0], headers), body
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            self.sock.settimeout(min(1.0, remaining))
            try:
                data = self.sock.recv(65536)
            except socket.timeout:
                continue
            if not data:
                raise EOFError('RTSP peer closed the connection')
            self.buffer.extend(data)
            if len(self.buffer) > 1 << 20:
                raise ValueError('RTSP framing buffer exceeded 1MiB')

    def _packet(self, channel, data):
        if self.started is None:
            return
        arrival = time.monotonic() - self.started
        if channel % 2:
            for report in parse_rtcp(data, arrival, channel):
                self.sender_report_count += 1
                if self.rtcp_sink is None:
                    self.sender_reports.append(report)
                else:
                    self.rtcp_sink(report)
            return
        track = next((t for t in self.tracks if t['channel'] == channel), None)
        if track is not None:
            row = {'t': arrival, 'channel': channel, **parse_rtp(data, track['kind'])}
            self.row_count += 1
            if self.row_sink is None:
                self.rows.append(row)
            else:
                self.row_sink(row)

    def request(self, method, url, headers=None, deadline=None):
        self.sequence += 1
        fields = {'CSeq': str(self.sequence), **(headers or {})}
        if self.session:
            fields['Session'] = self.session
        self.sock.sendall((method + ' ' + url + ' RTSP/1.0\r\n' + ''.join(k + ': ' + v + '\r\n' for k, v in fields.items()) + '\r\n').encode())
        deadline = deadline or time.monotonic() + 8
        while True:
            message = self._message(deadline)
            if message is None:
                raise TimeoutError('RTSP response deadline exceeded')
            kind, first, data = message
            if kind == 'packet':
                self._packet(first, data)
                continue
            status, response = first
            if status.split()[1] != '200':
                raise RuntimeError('RTSP request failed with status ' + status.split()[1])
            if response.get('cseq') != str(self.sequence):
                raise RuntimeError('RTSP response sequence does not match the request')
            if 'session' in response:
                self.session = response['session'].split(';')[0]
            return response, data

    def collect(self):
        headers, sdp = self.request('DESCRIBE', self.url, {'Accept': 'application/sdp'})
        current, descriptions = None, []
        for line in sdp.decode().splitlines():
            if line.startswith('m='):
                if current is not None:
                    descriptions.append(current)
                current = {'kind': line[2:].split()[0], 'rates': {}}
            elif line.startswith('a=rtpmap:') and current is not None:
                ident, encoding = line[9:].split(None, 1)
                codec, rate, *_ = encoding.split('/')
                current['rates'][ident] = (codec, int(rate))
            elif line.startswith('a=control:') and current is not None:
                control = line[10:]
                if control == '*':
                    continue
                current['control'] = control
        if current is not None:
            descriptions.append(current)
        for description in descriptions:
            if 'control' not in description or description['kind'] not in ('video', 'audio'):
                continue
            codec, rate = next(iter(description['rates'].values()), ('H264', 90000) if description['kind'] == 'video' else ('unknown', 48000))
            self.tracks.append({'kind': description['kind'], 'codec': codec, 'clock_rate': rate,
                                'channel': len(self.tracks) * 2, '_control': description['control']})
        if not self.tracks:
            raise ValueError('RTSP description has no usable media controls')
        for track in self.tracks:
            control = track.pop('_control')
            base = headers.get('content-base', self.url.rstrip('/') + '/')
            url = control if control.startswith('rtsp://') else base.rstrip('/') + '/' + control
            channel = track['channel']
            self.request('SETUP', url, {'Transport': f'RTP/AVP/TCP;unicast;interleaved={channel}-{channel + 1}'})
        self.started = time.monotonic()
        self.request('PLAY', self.url)
        error = None
        deadline = self.started + self.seconds
        keepalive = self.started + self.keepalive_interval_s
        try:
            while True:
                message = self._message(min(deadline, keepalive))
                if message is None:
                    if time.monotonic() >= deadline:
                        break
                    self.request('OPTIONS', self.url, deadline=min(deadline, time.monotonic() + 8))
                    keepalive = time.monotonic() + self.keepalive_interval_s
                    continue
                kind, first, data = message
                if kind == 'packet':
                    self._packet(first, data)
        except (EOFError, OSError, ValueError, RuntimeError) as exc:
            error = type(exc).__name__ + ': ' + str(exc)
        return {'schema': 1, 'wall_seconds': time.monotonic() - self.started,
                'requested_seconds': self.seconds, 'tracks': self.tracks,
                'rows': self.rows, 'rtcp_sender_reports': self.sender_reports,
                'capture_error': error, 'media_payloads_retained': False}

    def close(self):
        self.sock.close()


def _write_new(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as handle:
        json.dump(document, handle, indent=2)
        handle.write('\n')


class StreamingDocument:
    """Write the same JSON schema with bounded memory for RTP and RTCP rows.

    RTCP arrives interleaved with RTP, so its much smaller array is spooled to a
    temporary file and copied in bounded chunks after the RTP array closes.
    Neither packet count contributes to the collector's resident Python lists.
    """
    def __init__(self, handle):
        self.handle = handle
        self.rtcp_file = tempfile.TemporaryFile(mode='w+', encoding='utf-8')
        self.row_count = self.rtcp_count = 0
        self.finished = False
        handle.write('{"rows":[')

    def row(self, value):
        if self.row_count:
            self.handle.write(',')
        json.dump(value, self.handle, separators=(',', ':'))
        self.row_count += 1

    def rtcp(self, value):
        if self.rtcp_count:
            self.rtcp_file.write(',')
        json.dump(value, self.rtcp_file, separators=(',', ':'))
        self.rtcp_count += 1

    def finish(self, document):
        if self.finished:
            raise ValueError('streaming document was already finished')
        self.handle.write('],"rtcp_sender_reports":[')
        self.rtcp_file.seek(0)
        while True:
            chunk = self.rtcp_file.read(65536)
            if not chunk:
                break
            self.handle.write(chunk)
        self.handle.write(']')
        for key, value in document.items():
            if key in ('rows', 'rtcp_sender_reports'):
                continue
            self.handle.write(',')
            json.dump(key, self.handle)
            self.handle.write(':')
            json.dump(value, self.handle, separators=(',', ':'))
        self.handle.write('}\n')
        self.handle.flush()
        self.finished = True

    def close(self):
        self.rtcp_file.close()


def capture_to_file(url, seconds, output, stream_rows=False):
    """Exclusive output; streaming mode retains a valid partial result on errors."""
    output = Path(output)
    if output.exists():
        raise FileExistsError('refusing to overwrite an existing capture')
    reader = None
    if stream_rows:
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x') as handle:
            writer = StreamingDocument(handle)
            try:
                try:
                    reader = RTSPCapture(url, seconds, row_sink=writer.row, rtcp_sink=writer.rtcp)
                    document = reader.collect()
                except (EOFError, OSError, ValueError, RuntimeError) as error:
                    document = {'schema': 1, 'wall_seconds': time.monotonic() - reader.started if reader is not None and reader.started is not None else 0,
                                'requested_seconds': seconds,
                                'tracks': [{key: value for key, value in track.items() if not key.startswith('_')}
                                           for track in reader.tracks] if reader is not None else [],
                                'capture_error': type(error).__name__ + ': ' + str(error), 'media_payloads_retained': False}
                finally:
                    if reader is not None:
                        reader.close()
                writer.finish(document)
                counts = writer.row_count, writer.rtcp_count
            finally:
                writer.close()
    else:
        reader = RTSPCapture(url, seconds)
        try:
            document = reader.collect()
        finally:
            reader.close()
        _write_new(output, document)
        counts = reader.row_count, reader.sender_report_count
    return {'wall_seconds': document['wall_seconds'], 'rtp_packets': counts[0],
            'rtcp_sender_reports': counts[1], 'capture_error': document['capture_error'],
            'media_payloads_retained': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest='command', required=True)
    capture = commands.add_parser('capture', help='Read interleaved RTSP packets; retain metadata only')
    capture.add_argument('--url', required=True, help='Local relay or tunnel URL without credentials')
    capture.add_argument('--seconds', type=float, default=180)
    capture.add_argument('--output', required=True, type=Path)
    capture.add_argument('--stream-rows', action='store_true', help='Write RTP/RTCP rows incrementally with bounded memory; same JSON schema')
    inspect = commands.add_parser('analyze', help='Inspect saved headers without any network access')
    inspect.add_argument('input', type=Path)
    inspect.add_argument('--fps', type=float, default=25)
    inspect.add_argument('--require-duration', type=float, default=0)
    inspect.add_argument('--max-drift-ratio', type=float, default=0.001)
    inspect.add_argument('--endpoint-tolerance', type=float, default=0.12)
    inspect.add_argument('--output', type=Path)
    inspect.add_argument('--strict', action='store_true', help='Exit1 if any clock/transport/duration gate fails')
    args = parser.parse_args(argv)
    if args.command == 'capture':
        if not 1 <= args.seconds <= 3600:
            parser.error('capture duration must be 1..3600 seconds')
        if args.output.exists():
            parser.error('refusing to overwrite an existing capture')
        summary = capture_to_file(args.url, args.seconds, args.output, args.stream_rows)
        print(json.dumps(summary))
        return 1 if summary['capture_error'] else 0
    raw = args.input.read_bytes()
    result = analyze(json.loads(raw), args.fps, args.require_duration,
                     args.max_drift_ratio, args.endpoint_tolerance)
    result['input_sha256'] = hashlib.sha256(raw).hexdigest()
    if args.output:
        _write_new(args.output, result)
    print(json.dumps(result, indent=2))
    return 1 if args.strict and not result['passed'] else 0


if __name__ == '__main__':
    sys.exit(main())
