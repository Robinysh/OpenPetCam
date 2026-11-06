"""Clock checks distinguish config packets from pictures and preserve wraparound."""
import importlib.util
import io
import json
from pathlib import Path
import struct
import socket
import tempfile
import threading
import time
import tracemalloc
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('video_clock_check', ROOT / 'tools/video_clock_check.py')
clock = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(clock)


def packet(payload, timestamp=1234, sequence=1, marker=True, ssrc=7):
    return struct.pack('!BBHII', 0x80, 96 | (128 if marker else 0), sequence, timestamp, ssrc) + payload


def recording(frames=51, separate_config=False, start_video=0, start_audio=0):
    rows, video_sequence, audio_sequence, extras = [], 65520, 65520, 0
    for i in range(frames):
        arrival = i * .04
        stamp = (start_video + (i + extras) * 3600) % 2**32
        nals = [b'\x61\x80', b'\x61\x40']
        if i % 25 == 0:
            if separate_config and i:
                row = clock.parse_rtp(packet(b'\x67\x01', stamp, video_sequence % 65536))
                rows.append({'t': arrival, 'channel': 0, **row})
                video_sequence += 1
                extras += 1
                stamp = (stamp + 3600) % 2**32
                nals = [b'\x68\x01', b'\x65\x80', b'\x65\x40']
            else:
                nals = [b'\x67\x01', b'\x68\x01', b'\x65\x80', b'\x65\x40']
        for j, nal in enumerate(nals):
            row = clock.parse_rtp(packet(nal, stamp, video_sequence % 65536, marker=j == len(nals) - 1))
            rows.append({'t': arrival, 'channel': 0, **row})
            video_sequence += 1
        row = clock.parse_rtp(packet(b'opus', (start_audio + i * 1920) % 2**32,
                                     audio_sequence % 65536, ssrc=8), 'audio')
        rows.append({'t': arrival + .005, 'channel': 2, **row})
        audio_sequence += 1
    return {'wall_seconds': frames * .04, 'tracks': [
        {'channel': 0, 'kind': 'video', 'clock_rate': 90000, 'codec': 'H264'},
        {'channel': 2, 'kind': 'audio', 'clock_rate': 48000, 'codec': 'OPUS'}], 'rows': rows}


class VideoClockTest(unittest.TestCase):
    def test_streaming_document_has_identical_parsed_schema_and_rows(self):
        document = recording(frames=3)
        document['schema'] = 1
        document['rtcp_sender_reports'] = [{'channel': 1, 'ssrc': 7, 'rtp_ts': 1234, 'ntp_seconds': 100.5}]
        document['capture_error'] = None
        handle = io.StringIO()
        writer = clock.StreamingDocument(handle)
        try:
            for row in document['rows']:
                writer.row(row)
            for report in document['rtcp_sender_reports']:
                writer.rtcp(report)
            writer.finish({**document, 'rows': [], 'rtcp_sender_reports': []})
            self.assertEqual(json.loads(handle.getvalue()), document)
            self.assertEqual(writer.row_count, len(document['rows']))
            self.assertEqual(writer.rtcp_count, 1)
        finally:
            writer.close()

    def test_stream_sinks_do_not_accumulate_packet_lists_or_memory(self):
        reader = clock.RTSPCapture.__new__(clock.RTSPCapture)
        reader.started = time.monotonic()
        reader.tracks = [{'channel': 0, 'kind': 'video'}]
        reader.rows, reader.sender_reports = [], []
        reader.row_count = reader.sender_report_count = 0
        counts = {'rtp': 0, 'rtcp': 0}
        def row_sink(_):
            counts['rtp'] += 1
        def rtcp_sink(_):
            counts['rtcp'] += 1
        reader.row_sink, reader.rtcp_sink = row_sink, rtcp_sink
        raw = packet(b'\x65\x80')
        sender = struct.pack('!BBHIIIIII', 0x80, 200, 6, 7, 100, 2**31, 1234, 9, 100)
        tracemalloc.start()
        try:
            for _ in range(10000):
                reader._packet(0, raw)
                reader._packet(1, sender)
            _, peak = tracemalloc.get_traced_memory()
        finally:
            tracemalloc.stop()
        self.assertEqual(reader.rows, [])
        self.assertEqual(reader.sender_reports, [])
        self.assertEqual(counts, {'rtp': 10000, 'rtcp': 10000})
        self.assertEqual((reader.row_count, reader.sender_report_count), (10000, 10000))
        self.assertLess(peak, 2 * 1024 * 1024, 'streaming collector retained packet-count-sized state')

    def test_capture_to_file_streaming_and_memory_modes_preserve_content_and_counts(self):
        fixed = recording(frames=3)
        fixed.update(schema=1, rtcp_sender_reports=[{'channel': 1, 'ssrc': 7, 'rtp_ts': 0}],
                     capture_error=None, media_payloads_retained=False)
        class FakeCapture:
            def __init__(self, url, seconds, row_sink=None, rtcp_sink=None):
                self.rows, self.sender_reports = [], []
                self.row_sink, self.rtcp_sink = row_sink, rtcp_sink
                self.row_count = self.sender_report_count = 0
            def collect(self):
                for row in fixed['rows']:
                    self.row_count += 1
                    if self.row_sink:
                        self.row_sink(row)
                    else:
                        self.rows.append(row)
                for report in fixed['rtcp_sender_reports']:
                    self.sender_report_count += 1
                    if self.rtcp_sink:
                        self.rtcp_sink(report)
                    else:
                        self.sender_reports.append(report)
                return {**fixed, 'rows': self.rows, 'rtcp_sender_reports': self.sender_reports}
            def close(self):
                pass
        with tempfile.TemporaryDirectory() as folder, patch.object(clock, 'RTSPCapture', FakeCapture):
            first, second = Path(folder) / 'memory.json', Path(folder) / 'streamed.json'
            memory = clock.capture_to_file('rtsp://127.0.0.1:8554/cam', 1, first)
            streamed = clock.capture_to_file('rtsp://127.0.0.1:8554/cam', 1, second, stream_rows=True)
            self.assertEqual(json.loads(first.read_text()), json.loads(second.read_text()))
            self.assertEqual(json.loads(second.read_text()), fixed)
            self.assertEqual(streamed, memory)
            self.assertEqual(streamed['rtp_packets'], len(fixed['rows']))
            self.assertEqual(streamed['rtcp_sender_reports'], 1)
            with self.assertRaises(FileExistsError):
                clock.capture_to_file('rtsp://127.0.0.1:8554/cam', 1, second, stream_rows=True)

    def test_stap_a_detects_picture_and_configuration_nals(self):
        nals = [b'\x67\x01', b'\x68\x01', b'\x65\x80']
        payload = b'\x78' + b''.join(struct.pack('!H', len(nal)) + nal for nal in nals)
        self.assertEqual(clock.parse_rtp(packet(payload))['nal_types'], [7, 8, 5],
                         'Treating an aggregate as NAL24 hides its picture from clock grouping')

    def test_shared_sps_pps_multislice_idr_uses_one_picture_timestamp(self):
        result = clock.analyze(recording())
        self.assertTrue(result['passed'], result['issues'])
        video = result['streams'][0]
        self.assertEqual(video['picture_groups'], 51)
        self.assertEqual(video['timestamp_groups'], 51)
        self.assertEqual(video['multiple_first_slice_groups'], 0)
        self.assertEqual(video['config_only_groups_between_pictures'], 0)
        self.assertAlmostEqual(video['picture_rtp_span_s'], 2)

    def test_config_only_rtp_groups_fail_even_without_network_loss(self):
        result = clock.analyze(recording(separate_config=True))
        video = result['streams'][0]
        self.assertFalse(result['passed'])
        self.assertEqual(video['picture_groups'], 51)
        self.assertEqual(video['config_only_groups_between_pictures'], 2)
        self.assertEqual(video['missing_packets'], 0)
        self.assertAlmostEqual(video['media_excess_over_nominal_picture_count_s'], .08)

    def test_nonzero_av_initial_offsets_do_not_create_elapsed_drift(self):
        result = clock.analyze(recording(start_video=1900000000, start_audio=2700000000))
        self.assertTrue(result['passed'], result['issues'])
        self.assertAlmostEqual(result['av_elapsed_comparisons'][0]['video_minus_audio_span_s'], 0)

    def test_timestamp_and_sequence_wrap_are_forward_continuity(self):
        result = clock.analyze(recording(start_video=2**32 - 36000, start_audio=2**32 - 19200))
        self.assertTrue(result['passed'], result['issues'])
        self.assertEqual([s['timestamp_wraps'] for s in result['streams']], [1, 1])
        self.assertEqual([s['missing_packets'] for s in result['streams']], [0, 0])

    def test_rtp_extension_csrc_and_padding_do_not_change_nal_payload(self):
        data = struct.pack('!BBHII', 0xb1, 96, 1, 2, 3) + b'csrc' + struct.pack('!HH', 0x1000, 1) + b'extn' + b'\x65\x80' + b'\0\0\0\x04'
        parsed = clock.parse_rtp(data)
        self.assertEqual(parsed['nal_types'], [5])
        self.assertEqual(parsed['payload_bytes'], 2)
        self.assertEqual(parsed['first_slices'], 1)

    def test_fu_start_and_continuation_are_one_nal_start(self):
        start = clock.parse_rtp(packet(b'\x7c\x85\x80'))
        continuation = clock.parse_rtp(packet(b'\x7c\x05\x80'))
        self.assertEqual(start['nal_types'], [5])
        self.assertEqual(start['nal_starts'], [5])
        self.assertEqual(start['first_slices'], 1)
        self.assertEqual(continuation['nal_types'], [5])
        self.assertEqual(continuation['nal_starts'], [])
        self.assertEqual(continuation['first_slices'], 0)

    def test_truncated_or_unsupported_packetization_is_not_accepted_silently(self):
        for payload in [b'', b'\x78\0\x08\x67', b'\x7c', b'\x79\0']:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                clock.parse_rtp(packet(payload))
        self.assertTrue(clock.parse_rtp(packet(b'\x7a\0\0\0'))['unsupported_packetization'])

    def test_missing_duplicate_and_reordered_packets_are_visible(self):
        for mode in ['missing', 'duplicate', 'reordered']:
            data = recording()
            if mode == 'missing':
                del data['rows'][5]
            elif mode == 'duplicate':
                data['rows'].insert(5, data['rows'][5])
            else:
                data['rows'][5], data['rows'][8] = data['rows'][8], data['rows'][5]
            result = clock.analyze(data)
            with self.subTest(mode=mode):
                self.assertFalse(result['passed'])

    def test_clock_regression_and_multiple_ssrc_are_not_unwrapped_as_long_forward_time(self):
        data = recording()
        video = [row for row in data['rows'] if row['channel'] == 0]
        video[-1]['rtp_ts'] = 0
        result = clock.analyze(data)
        self.assertFalse(result['passed'])
        self.assertGreater(result['streams'][0]['timestamp_regressions'], 0)
        data = recording()
        data['rows'][-3]['ssrc'] = 99
        result = clock.analyze(data)
        self.assertTrue(any('multiple SSRC' in issue for issue in result['issues']))

    def test_multiple_first_slices_with_same_timestamp_flag_ambiguous_pictures(self):
        data = recording()
        row = next(r for r in data['rows'] if r['channel'] == 0 and r['nal_types'] == [1])
        row['first_slices'] = 2
        result = clock.analyze(data)
        self.assertFalse(result['passed'])
        self.assertEqual(result['streams'][0]['multiple_first_slice_groups'], 1)

    def test_missing_audio_incomplete_duration_and_capture_error_fail_closed(self):
        data = recording()
        data['rows'] = [r for r in data['rows'] if r['channel'] != 2]
        self.assertFalse(clock.analyze(data)['passed'])
        self.assertFalse(clock.analyze(recording(), require_duration_s=150)['passed'])
        data = recording()
        data['capture_error'] = 'Connection ended before requested stop'
        self.assertFalse(clock.analyze(data)['passed'])
        data = recording()
        data['tracks'][0]['codec'] = 'H265'
        self.assertFalse(clock.analyze(data)['passed'])

    def test_config_only_capture_boundary_does_not_invent_an_internal_bug(self):
        data = recording()
        # A config packet before the first available picture has no demonstrated
        # relation to an included preceding picture. Report its group, not a bug.
        data['rows'].insert(0, {'t': -.04, 'channel': 0,
                               **clock.parse_rtp(packet(b'\x67\x01', 2**32 - 3600, 65519))})
        result = clock.analyze(data)
        self.assertTrue(result['passed'], result['issues'])
        self.assertEqual(result['streams'][0]['timestamp_groups'], 52)
        self.assertEqual(result['streams'][0]['config_only_groups_between_pictures'], 0)

    def test_rtcp_sender_report_clock_pairs_are_supplementary(self):
        sender = struct.pack('!BBHIIIIII', 0x80, 200, 6, 7, 100, 2**31, 1234, 9, 100)
        rows = clock.parse_rtcp(sender, 1.25, 1)
        self.assertEqual(rows[0]['ssrc'], 7)
        self.assertEqual(rows[0]['ntp_seconds'], 100.5)
        self.assertEqual(rows[0]['rtp_ts'], 1234)
        with self.assertRaises(ValueError):
            clock.parse_rtcp(sender[:-1], 0, 1)

    def test_interleaved_message_reader_keeps_fragmented_headers_and_payload(self):
        first, second = socket.socketpair()
        reader = clock.RTSPCapture.__new__(clock.RTSPCapture)
        reader.sock, reader.buffer = first, bytearray()
        raw = b'$\x00\x00\x03abcRTSP/1.0 200 OK\r\nCSeq: 1\r\nContent-Length: 3\r\n\r\nsdp'
        def send():
            for i in range(0, len(raw), 2):
                second.sendall(raw[i:i + 2])
            second.close()
        worker = threading.Thread(target=send)
        worker.start()
        try:
            self.assertEqual(reader._message(time.monotonic() + 2), ('packet', 0, b'abc'))
            kind, response, body = reader._message(time.monotonic() + 2)
            self.assertEqual(kind, 'response')
            self.assertEqual(response[0], 'RTSP/1.0 200 OK')
            self.assertEqual(body, b'sdp')
        finally:
            first.close()
            worker.join(2)

    def _local_mock_capture(self, stream_rows):
        listener = socket.socket()
        listener.bind(('127.0.0.1', 0))
        listener.listen(1)
        address = listener.getsockname()
        methods, errors = [], []

        def server():
            peer, _ = listener.accept()
            peer.settimeout(2)
            pending = bytearray()
            try:
                while True:
                    while b'\r\n\r\n' not in pending:
                        data = peer.recv(4096)
                        if not data:
                            return
                        pending.extend(data)
                    end = pending.index(b'\r\n\r\n')
                    request = bytes(pending[:end]).decode().split('\r\n')
                    del pending[:end + 4]
                    method = request[0].split()[0]
                    methods.append(method)
                    sequence = next(line.split(':', 1)[1].strip() for line in request if line.lower().startswith('cseq:'))
                    body = b''
                    if method == 'DESCRIBE':
                        body = (b'v=0\r\nm=video 0 RTP/AVP 96\r\na=control:track0\r\na=rtpmap:96 H264/90000\r\n'
                                b'm=audio 0 RTP/AVP 97\r\na=control:track1\r\na=rtpmap:97 L16/44100\r\n')
                    response = f'RTSP/1.0 200 OK\r\nCSeq: {sequence}\r\nSession: fixture\r\nContent-Length: {len(body)}\r\n\r\n'.encode() + body
                    peer.sendall(response)
                    if method == 'PLAY':
                        data = packet(b'\x65\x80')
                        peer.sendall(b'$\0' + len(data).to_bytes(2, 'big') + data)
                        sender = struct.pack('!BBHIIIIII', 0x80, 200, 6, 7, 100, 2**31, 1234, 9, 100)
                        peer.sendall(b'$\x01' + len(sender).to_bytes(2, 'big') + sender)
            except BaseException as error:
                errors.append(error)
            finally:
                peer.close()
                listener.close()

        worker = threading.Thread(target=server)
        worker.start()
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / 'capture.json'
            try:
                with patch.object(clock.RTSPCapture, 'keepalive_interval_s', .05):
                    counts = clock.capture_to_file(f'rtsp://127.0.0.1:{address[1]}/cam', .3, output, stream_rows)
                result = json.loads(output.read_text())
            finally:
                worker.join(2)
        self.assertFalse(errors, errors)
        self.assertIsNone(result['capture_error'])
        self.assertEqual([t['clock_rate'] for t in result['tracks']], [90000, 44100])
        self.assertEqual(result['rows'][0]['nal_types'], [5])
        self.assertGreaterEqual(methods.count('OPTIONS'), 1)
        self.assertFalse(result['media_payloads_retained'])
        self.assertEqual(counts['rtp_packets'], len(result['rows']))
        self.assertEqual(counts['rtcp_sender_reports'], len(result['rtcp_sender_reports']))
        self.assertEqual(counts['rtp_packets'], 1)
        self.assertEqual(counts['rtcp_sender_reports'], 1)
        return result

    def test_local_mock_rtsp_capture_accepts_sdp_order_and_sends_keepalive(self):
        memory = self._local_mock_capture(False)
        streamed = self._local_mock_capture(True)
        # Wall/arrival times naturally differ across two real socket sessions;
        # every protocol field and captured row must otherwise be identical.
        for result in (memory, streamed):
            result.pop('wall_seconds')
            for key in ('rows', 'rtcp_sender_reports'):
                for row in result[key]:
                    row.pop('t')
        self.assertEqual(memory, streamed)


if __name__ == '__main__':
    unittest.main()
