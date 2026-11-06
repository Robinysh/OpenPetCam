# Prudynt picture-clock correction

This patch targets `themactep/prudynt-t` commit
`6afc4400c54a4739feeb72ac884587409577a2af`. It corrects the timestamp handling
in `VideoWorker` and the RTSP video clock; it does not alter audio, encoder FPS,
recorder drift tolerances, or recorded media after capture.

The old worker treats an SPS pack's `frameEnd` as a picture boundary and advances
a nonincreasing timestamp by a nominal frame period. On the observed T23 camera,
a keyframe batch contains SPS/PPS/IDR with identical encoder timestamps but
`frameEnd` values `true/false/true`. That creates one extra video period per GOP.
The RTSP writer also forced equal timestamps forward and clamped real gaps.

The repair classifies NAL prefixes and picture slices before touching the clock.
All NALs in a picture access unit receive its actual VCL capture timestamp. Only
the first NAL opens picture time, and only the final NAL sets the RTP marker.
Config-only events remain clock-neutral. The shared header is compiled by both
the actual worker and the actual RTSP writer; the host tests use that same header.

## Apply to the pinned source

From the OpenPetCam checkout:

```sh
git -C PATH_TO_PRUDYNT rev-parse HEAD
# Must be 6afc4400c54a4739feeb72ac884587409577a2af with untouched affected files.
git -C PATH_TO_PRUDYNT apply --check "$PWD/tools/firmware/prudynt-timestamps/prudynt-6afc4400-access-unit-clock.patch"
git -C PATH_TO_PRUDYNT apply "$PWD/tools/firmware/prudynt-timestamps/prudynt-6afc4400-access-unit-clock.patch"
python3 tests/server/prudynt_timestamp_test.py
python3 tools/firmware/prudynt-timestamps/check_source.py PATH_TO_PRUDYNT --worker
python3 tools/firmware/prudynt-timestamps/check_source.py PATH_TO_PRUDYNT
```

`check_source.py --baseline` extracts and compiles the old timestamp statements
from the pinned Git object. Its assertion is expected to fail: 750 actual pictures
should span 29,960,000 us, but the old code produces 31,119,988 us for the synthetic
GOP-25 input. It is a deliberately RED diagnostic, not an installation command.

The normal native regression checks GOP 10/25/50, duplicate/config-only NALs,
multiple picture units in a batch, multislice pictures, first-slice detection,
real capture gaps, RTP wrap and explicit invalid-time handling. The source checks
also compile the actual modified `VideoWorker` statements and actual modified
`RtspServer::updateVideoTimestamp` method with bounded SDK/session carriers.
Those host adapters do not replace the required full target build.

## Replay observed SDK inputs

```sh
python3 tools/firmware/prudynt-timestamps/replay_sdk.py \
  PRIVATE_SDK_PACK_TRACE.jsonl NEW_PRIVATE_OUTPUT_DIRECTORY
```

The trace contains one `kind=abi` row and `kind=pack` rows with channel, batch,
index, count, timestamp_us, frame_end, sdk_type and the first eight payload bytes
as hexadecimal `prefix`. The replay compiles the exact production helper and
writes per-pack assignments and hashes. Keep traces and outputs under ignored
`eval/`; they are installation evidence, not public fixtures.

For the observed T23 ABI (pack32, stream16, timestamp offset16/frameEnd24), all
782 packs from 684 picture batches were accepted. Both channels preserved every
capture-time difference and assigned identical timestamps to SPS/PPS/VCL in a
batch. The main stream's real mix of roughly 33/67 ms capture gaps is retained.
No zero/repeated/backward picture timestamps occurred in that trace. It contains
no observed multislice picture; multislice coverage is synthetic.

## Discontinuities and limits

A new picture with zero, repeated or backward capture time is rejected explicitly
and requests an IDR. It does not advance the clock or create a nominal timestamp.
A later valid higher timestamp recovers; an explicit encoder/subscriber startup
resets the helper before a new capture epoch. A spontaneous backward clock epoch
will therefore need a stream/encoder restart rather than being silently rebased.
This fail-closed policy has host tests; it was not exercised by a real target
clock reset. The patch preserves real positive gaps instead of filling them.

H264 first-macroblock and H265 first-slice flags define picture boundaries. H264
data-partitioned NALs and malformed slice headers are rejected. The installed
T23 encoder uses ordinary progressive H264 slices. This patch does not certify
arbitrary FMO/ASO streams or unsupported hardware SDK layouts.

A compatible target build, binary/link/load checks, rollback, and a sustained
live raw-RTP/recording trial are separate release requirements. Native test
success alone must not be reported as a repaired camera or recovered old footage.

## Persistent installation without reflashing

`../install_prudynt_clockfix.sh` is an explicit, image-specific installer for the
verified binary SHA256
`45e3e64a41b677ef31c1e164b31150e6863cc90ba5820eb2bb60e4e43c37bf5b`.
It checks the immutable stock binary and SDK hashes before persisting anything.
The exact 967000-byte image stays compressed in `/etc/opc-clockfix`; an owned
`S30opc-clockfix` hook verifies, decompresses into RAM, and bind-mounts it before
the stock `S31prudynt` service starts. ROM and camera configuration are untouched.

Run on the target only after the build, loader, SDK-boundary and live-trial gates:

```sh
sh /tmp/install_prudynt_clockfix.sh dry-run --payload /tmp/prudynt-fixed.gz \
  --sha256 45e3e64a41b677ef31c1e164b31150e6863cc90ba5820eb2bb60e4e43c37bf5b
sh /tmp/install_prudynt_clockfix.sh install --payload /tmp/prudynt-fixed.gz \
  --sha256 45e3e64a41b677ef31c1e164b31150e6863cc90ba5820eb2bb60e4e43c37bf5b
```

Installation only persists the package; it does not remount a running process or
restart the streamer. Space preflight includes a 128 KiB overlay reserve. The
first reboot is a guarded trial: within 180 seconds, independently verify real
video/RTP clock behavior and recordings, then run:

```sh
/etc/opc-clockfix/control accept-boot
```

The command requires this boot's runtime marker, an active bind of the exact
verified image, and a running Prudynt. Those mechanical checks do not substitute
for the operator's video/RTP validation. Accepted, compatible future boots apply
the same image automatically. A changed ROM/SDK or bad payload leaves stock in
place.

An unaccepted first boot disables only its own hook and cold-reboots stock. It
first tries the conservative stop/wait/settle/unmount sequence. If the process
will not exit or the operation lock is stuck, it leaves the live mount alone and
cold-reboots with its hook disabled. This avoids starting a duplicate streamer
against an ISP whose state may not have torn down. A warm stock restart is not
proof of recovered video.

Manual rollback is explicit:

```sh
/etc/opc-clockfix/control rollback                # conservative warm restart
/etc/opc-clockfix/control rollback --cold-reboot  # cold stock recovery
```

Manual warm rollback never kills or reboots a stuck process. It waits for
`pidof prudynt` to be empty, settles two seconds, and unmounts only the known
patched hash. Both forms remove only the owned package/hook/runtime files.
The mock installer checks never call hardware commands:

```sh
python3 tools/firmware/prudynt-timestamps/install_helper_test.py
```

## Reference T23 validation

On the reference T23/SDK 1.3.0 installation, two clean builds of each variant
were byte-identical. The patched image passed the target ABI/dependency audit
and was installed with the guarded overlay method above. A cold reboot loaded
the verified image; a fresh 45-second RTP check passed before boot acceptance.

The sustained check captured 16,499 pictures in 660 seconds, with exactly 16,499
video timestamp groups and no configuration-only groups. Video and audio elapsed
spans differed by 34.5 ms; the matching recorder journal had no drift resets or
late-audio warnings. An unmodified 180-second recorded MP4 decoded throughout
both streams (4,498 video frames and 4,500 audio frames). Camera settings were
unchanged. Detailed installation receipts and private media remain under ignored
`eval/video_timestamp_fix/`, outside the public repository.

These checks establish the source-clock repair on this firmware. They do not
calibrate physical audio/video alignment, validate other SDK layouts, or recover
footage already missing from old recordings.
