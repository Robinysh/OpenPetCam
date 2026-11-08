# Device audio observation

This opt-in experiment observes the existing Prudynt audio owner. It does not
open a second microphone consumer and does not modify firmware or configuration.
The preload and reader must be reviewed and smoke-tested on the target before a
bounded, reversible streamer restart. Importing Python modules does nothing to
the camera.

The reference source is Prudynt stable
[`6afc4400c54a4739feeb72ac884587409577a2af`](https://github.com/themactep/prudynt-t/tree/6afc4400c54a4739feeb72ac884587409577a2af/src/audio).
The ABI subset follows the [T23 audio header](https://github.com/gtxaspec/ingenic-headers/blob/master/T23/1.1.2/en/imp/imp_audio.h).
The header's GetFrameAndRef example releases the microphone frame once, through
IMP_AI_ReleaseFrame; it does not release the reference separately.

## Build and test

Run from the repository root:

```
python3 -m sim.acoustics.device.build --target native --output eval/device-build-native
python3 -m sim.acoustics.device.build --target mipsel --output eval/device-build-mipsel
python3 tests/sim/acoustic_device_decode_test.py
python3 tests/sim/acoustic_device_io_test.py
```

The MIPS build uses local Zig, MIPS32r2 little endian/o32 soft float. The reader
is a static musl executable. The preload has no libc dependency and imports the
host uClibc's ordinary dlsym, avoiding musl's time64 dlsym adapter. It uses the
32-bit kernel clock_gettime syscall for call brackets, rather than mixing musl
and uClibc timespec layouts. Build manifests record sources, commands and hashes.
Build success is not proof of the installed SDK ABI or timing semantics.

## Observation

Only when OPC_AUDIO_SOCKET is configured does the preload set up a local
root-only Unix socket. OPC_AUDIO_REFERENCE=1 requests the SDK output-reference
path after the existing application has enabled AO. It intercepts the existing
IMP_AI_GetFrame call; successful GetFrameAndRef supplies the mic/reference pair
to the recorder and returns the microphone frame to Prudynt unchanged. The
application's normal release remains responsible for the pair. A failed paired
SDK read is logged as a fault, and later reads fall back to the original API;
the decoder rejects any fault-containing capture. Unsupported reference enable
leaves reference absent; absence cannot pass reference validation.

The named reader has no SDK calls:

```
opc-audio-stream /tmp/unique-audio.sock 10 > capture.frames
python3 -m sim.acoustics.device.decode capture.frames eval/private-capture
```

The reader uses one fixed 512 KiB byte ring, touched before connecting, to drain
short UNIX-socket bursts while stdout is temporarily blocked. It preserves byte
order and partial-write offsets without changing PCM or allocating a capture in
memory. It sends its stop request independently of stdout after the bounded
duration (at most 600 seconds), then allows at most three seconds to drain the
remaining bytes and receive producer termination. Ring overflow returns exit `12`;
unexpected producer EOF returns `7`; the final output-drain deadline returns `11`.
Any nonzero exit is an incomplete capture, even if its partial bytes are useful
for diagnostics. Successful process exit does not replace the decoder's CRC,
continuity and terminal-frame checks.

The interceptor requests a 256 KiB socket send buffer and records the effective
kernel value, refusing values above 1 MiB. Kernel limits may reduce the request.
It still makes one nonblocking send per record: partial writes or backpressure
disconnect the stream. Such a capture lacks its terminal frame
and is rejected, instead of silently splicing later PCM onto an earlier sample
grid. Historical disconnections are recorded in the initial drop counter;
any counter change within a capture is invalid.

Optional `OPC_AUDIO_DIAGNOSTIC_PATH=/tmp/unique-audio-diagnostic.jsonl` records
socket acceptance and send-failure metadata: effective buffer size, monotonic
time, process/stream/record identifiers, errno and sent/expected byte counts.
It contains no PCM. The path must be a new direct `/tmp` basename; existing files
are preserved. Diagnostics use mode `0600`, a 64 KiB total budget and one best-effort
write per event, with no retry. These fault-path file writes are not a strict
real-time guarantee. Archive the diagnostic and reader stderr alongside partial
captures; do not label missing terminal records as successful acquisition.

## Framing and exports

Each record is little endian: eight-byte magic OPCAUD01, uint32 JSON length,
uint32 PCM length, uint32 CRC32 of JSON+PCM, then UTF-8 JSON and PCM. JSON is
bounded at 8192 bytes; PCM at 256 KiB. A stream has one hello, sequential record
indices, mic/AO records, and one terminal end. Mic payload is microphone bytes
followed by optional reference bytes. AO payload is the exact submitted PCM,
with call brackets and queue **block** counts; it is not a DAC timestamp.

The decoder verifies CRCs, lengths, stream/index/drop continuity, status, format,
mic SDK sequence continuity and fixed-size 48 kHz blocks. SDK block timestamps
are scheduling-jittered; they are diagnostic and are never a phase clock. A
gross discontinuity larger than half a block is rejected. Paired reference must
have matching mic/reference block sizes and timestamps within two microseconds.
All-zero reference sequences are explicitly marked unavailable; pairing then
uses the successful SDK call, mic sequence, matching block timestamps and count.
It preserves every original interval,
SDK timestamp, sequence number, monotonic call bracket and file hash. Leading
and trailing empty SDK references can be explicitly trimmed to a common mic/ref
interval; interior gaps are rejected. `--require-reference-every-frame` rejects
even edge empties. The original complete microphone PCM is retained separately.

Exports include mic.pcm, mic.all.pcm, reference.pcm when present, source.pcm for
AO submissions, and JSON provenance. Pair-contract verification is separate
from the final reference-grid flags. The latter require an active whole-AO to
reference waveform check: one global lag/gain, complete duration within quiet
margins, correlation at least .999, relative error at most .01, and three
energetic subwindows consistent with that one mapping. No local warps or mic
room peaks are used. The active camera pilot established that its SDK reference
is a zero-order hold of AO input samples on the 48 kHz paired grid (16 kHz AO
samples each repeat three times), with a shared attenuation. The validator uses
that fixed integer hold model. The SDK frame has no rate field: current headers
state the paired-AI candidate and AO rate separately; old headers' incorrect AO
rate copy remains visible and is explicitly annotated when decoded.

Reference validation flags describe the observed **digital block grid** only.
They are never set from array shape or timestamps alone. `timestamp_clock_verified`
and `physical_DAC_timing_verified` remain
false: SDK timestamp boundary semantics, analog emission delay and codec group
delay have not been independently established.

## Reversibility

Never replace `/usr/bin/prudynt` or persist LD_PRELOAD. Stage uniquely named
artifacts in camera `/tmp`, verify hashes, test loader behavior in an isolated
process with observation disabled and with an unused socket name, then arm a
separate stock-environment rollback timer before stopping the streamer. Launch
the original executable with observation variables for that one bounded run.
Always restart through the original init script afterwards and verify live
video/audio before cancelling rollback. The caller must handle privacy and
stop its own queued playback. Nothing here authorizes an unattended restart.
