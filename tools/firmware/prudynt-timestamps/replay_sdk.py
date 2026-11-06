#!/usr/bin/env python3
"""Replay an observed bounded SDK metadata trace through the production C++ helper."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--compiler', default='c++')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).resolve().parent
    rows = [json.loads(line) for line in args.trace.read_text().splitlines()]
    abi = [r for r in rows if r.get('kind') == 'abi']
    assert len(abi) == 1 and abi[0]['pack_size'] == 32 and abi[0]['stream_size'] == 16
    packs = [r for r in rows if r.get('kind') == 'pack']
    batches = defaultdict(list)
    for row in packs:
        batches[(row['channel'], row['batch'])].append(row)
    ordered = []
    for key, batch in sorted(batches.items()):
        batch.sort(key=lambda r: r['index'])
        assert [r['index'] for r in batch] == list(range(batch[0]['count'])), key
        ordered.extend(batch)
    payload = ''.join(f"{r['channel']} {r['batch']} {r['index']} {r['count']} {r['timestamp_us']} {int(r['frame_end'])} {r['sdk_type']} {r['prefix']}\n" for r in ordered)
    with tempfile.TemporaryDirectory(prefix='opc-sdk-replay-') as tmp:
        binary = Path(tmp) / 'replay'
        subprocess.run([args.compiler, '-std=c++17', '-Wall', '-Wextra', '-Werror', '-I', str(source),
                        str(source / 'replay.cpp'), '-o', str(binary)], check=True)
        run = subprocess.run([str(binary)], input=payload, capture_output=True, text=True, check=True)
    assignments = [json.loads(line) for line in run.stdout.splitlines()]
    assert len(assignments) == len(packs)
    by_batch = defaultdict(list)
    for row in assignments:
        assert row['valid'] and row['clock_result'] == 0
        by_batch[(row['channel'], row['batch'])].append(row)
    summary = {}
    for channel in sorted({r['channel'] for r in packs}):
        picture_times = []
        for key, batch in sorted(by_batch.items()):
            if key[0] != channel:
                continue
            assert sum(r['frame_start'] for r in batch) == 1
            assert sum(r['frame_end'] for r in batch) == 1
            assert len({r['assigned_us'] for r in batch}) == 1
            actual_vcl = [r for r in batches[key] if r['sdk_type'] in (1, 5)]
            assert len(actual_vcl) == 1
            assert batch[0]['assigned_us'] == actual_vcl[0]['timestamp_us']
            picture_times.append(batch[0]['assigned_us'])
        assert all(b > a for a, b in zip(picture_times, picture_times[1:]))
        summary[str(channel)] = {'pictures': len(picture_times),
                                'capture_span_us': picture_times[-1] - picture_times[0],
                                'preserved_deltas': True, 'config_and_vcl_timestamps_equal': True}
    (args.output / 'assigned.jsonl').write_text(run.stdout)
    receipt = {'status': 'passed', 'trace_sha256': hashlib.sha256(args.trace.read_bytes()).hexdigest(),
               'helper_sha256': hashlib.sha256((source / 'AccessUnitTiming.hpp').read_bytes()).hexdigest(),
               'input_packs': len(packs), 'output_assignments': len(assignments),
               'channels': summary, 'scope': 'Actual T23 SDK pack metadata through exact shared C++ grouping/clock helper; no target binary claim.'}
    (args.output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps(receipt))


if __name__ == '__main__':
    main()
