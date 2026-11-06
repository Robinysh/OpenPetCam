#!/usr/bin/env bash
# Run inside Linux with the pinned external Thingino cross toolchain.
# SOURCE must already contain the SDK headers and staged target libraries.
set -euo pipefail
if [ "$#" -lt 4 ] || [ "$#" -gt 5 ]; then
  echo "Usage: $0 SOURCE_DIR CROSS_PREFIX OUTPUT_BINARY VERSION_TAG [JOBS]" >&2
  exit 2
fi
source_dir=$(cd "$1" && pwd)
cross_prefix=$2
case "$3" in /*) output_binary=$3;; *) output_binary="$PWD/$3";; esac
version_tag=$4
jobs=${5:-4}
case "$jobs" in ''|*[!0-9]*) echo 'JOBS must be a positive integer' >&2; exit 2;; esac
[ "$jobs" -ge 1 ] && [ "$jobs" -le 16 ] || exit 2
case "$version_tag" in ''|*[!A-Za-z0-9._+-]*) echo 'Invalid version tag' >&2; exit 2;; esac
[ "$(uname -s)" = Linux ] || { echo 'Use the Linux builder container' >&2; exit 2; }
[ -x "${cross_prefix}g++" ] || { echo 'Missing cross compiler' >&2; exit 2; }
"${cross_prefix}g++" -dumpmachine | grep -q '^mipsel-.*uclibc$' || {
  echo 'Expected a MIPS little-endian uClibc compiler' >&2; exit 2;
}
cd "$source_dir"
[ -f src/video/VideoWorker.cpp ] && [ -f include/T23/1.3.0/zh/imp/imp_encoder.h ] || {
  echo 'Missing pinned Prudynt source or T23 SDK 1.3.0 headers' >&2; exit 2;
}
for library in libimp.so libalog.so libsysutils.so libopus.so libfaac.so libhelix-aac.so libjct.so libcurl.so libuclibcshim.so; do
  [ -f "3rdparty/install/lib/$library" ] || { echo "Missing staged target library: $library" >&2; exit 2; }
done
# COMPILE_TIME uses these macros. A fixed epoch and explicit tag remove wall time.
export SOURCE_DATE_EPOCH=1786581030 TZ=UTC LC_ALL=C
make clean
make -j"$jobs" ARCH=mips CROSS_COMPILE="$cross_prefix" SDK_VERSION=1.3.0 \
  USE_WEBSOCKETS=0 USE_PREBUFFER=0 USE_FLAC=0 USE_MP3=0 \
  USE_OPUS=1 USE_AAC=1 USE_OSD_BURNIN=1 USE_EXECINFO=0 \
  commit_tag="$version_tag" DEBUG_STRIP=1 \
  CFLAGS="-Os -DPLATFORM_T23 -DBINARY_DYNAMIC -DLIBC_UCLIBC -fno-stack-protector -march=mips32r2 -mhard-float -mfp32 -mnan=legacy -ffp-contract=off -mno-fused-madd -fPIE -ffile-prefix-map=$source_dir=." \
  LDFLAGS='-pie -L./3rdparty/install/lib -Wl,-rpath-link,./3rdparty/install/lib -Wl,--no-as-needed -luclibcshim -Wl,--as-needed -Wl,-z,now -Wl,--build-id=sha1' \
  LIBS='-limp -lalog -lsysutils -lopus -lfaac -lhelix-aac -ljct -lcurl -latomic -lpthread -lrt' all
cp bin/prudynt "$output_binary"
"${cross_prefix}readelf" -h -A -d "$output_binary"
sha256sum "$output_binary"
