# Reference T23 streamer build

This builds only Prudynt. It does not flash firmware, install a binary, restart a
camera, or modify recorder timing tolerances. The runtime libraries remain those
already installed on the target. Keep all device snapshots and built binaries
under ignored `eval/`.

The reference inputs are:

| Input | Revision or digest |
| --- | --- |
| Thingino package configuration | `2c3eca50a774e1d14eeb99e6eaa63fb4b051b8ae` |
| Prudynt | `6afc4400c54a4739feeb72ac884587409577a2af` |
| Ingenic headers submodule | `4e04a60041e6ffb97c9972e4f27964a887549b1c` |
| T23 SDK | `1.3.0`, headers in `include/T23/1.3.0/zh` |
| Ingenic library reference | `gtxaspec/ingenic-lib@99ed33fd55fdf4fdfaa378a0924c5c6a7a31943b` |
| JCT headers | `a11d22412aed71803a68ebac3137ea59beca2d58` (`v1.2.0`) |
| Opus headers | `22244de5a79bd1d6d623c32e72bf1954b56235be` (`v1.6.1`) |
| ESP8266Audio headers | `74fc1f09bbba5e5c5450b445452ba64ef2d8bbad` (`2.4.2`) |
| FAAC headers | `19504462aef3d444aa09b9cfcad0e2146889fb3b` (`faac-2.0`) |
| curl headers | `8.20.0`, archive SHA256 `4be48e69cf467246cb97d369b85d78a08528f2b37cffef2418ee16e6a4eb596e` |
| Thingino external GCC16/uClibc archive | SHA256 `357c915630a06666f8e2eecac8ff2f766981cb07a592d341d1931b37971b130e` |

The compiler archive is
[`thingino-toolchain-x86_64_xburst1_uclibc_gcc16-linux-mipsel.tar.gz`](https://github.com/themactep/thingino-firmware/releases/download/toolchain-x86_64/thingino-toolchain-x86_64_xburst1_uclibc_gcc16-linux-mipsel.tar.gz).
That release asset can change. Verify the digest; a different digest requires a
new compatibility audit. This archive contains GCC 16.2.0 and uClibc 1.0.59,
whereas the checked target has uClibc 1.0.58 and libstdc++ 6.0.35. No compiler
runtime libraries are installed on the camera. Link against the pulled target
libraries and verify every imported symbol/version against those exact files.

The target's `libimp.so` was byte-identical to the official SDK 1.3.0 library at
the revision above: SHA256
`bd904ec8f4123965ded8c0c57c25ad5c3ae0fdc17aa4a6374278a7c822cb31d9`.
Do not use `build.sh`'s standalone T23 library default, which selects SDK 1.1.0.

## Stage inputs

Use separate clean baseline and patched source directories. Initialize the
pinned Ingenic headers submodule in both. Apply the timestamp patch only to the
patched copy. Save source hashes before building.

Populate `3rdparty/install/include` in each source tree with:

- JCT `src/*.h`;
- Opus `include/*.h`, under an `opus/` subdirectory;
- ESP8266Audio `src/libhelix-aac/*.h`, after applying Prudynt's
  `res/libhelix-aac/0001-not-arduino.patch` and `0002-remove-mangling.patch`;
- FAAC `include/faac.h`;
- curl `include/curl/`, under a `curl/` subdirectory.

Populate `3rdparty/install/lib` with the target's mapped libraries and recreate
their SONAME links. Include the loader/libc aliases discovered on the target,
not guessed replacements. In the reference installation, `libpthread.so.0`,
`libdl.so.0` and `libm.so.0` all resolve to the same `libuClibc-1.0.58.so`.
The live ELF's complete dependency closure must be available to the linker.

## Compile

From the OpenPetCam checkout, prepare a build directory and verify the compiler
archive before extracting it. These host commands use Docker, `curl`, `tar` and
`shasum`:

```sh
build_dir="$PWD/eval/prudynt-build"
mkdir -p "$build_dir/toolchain"
curl -fL \
  https://github.com/themactep/thingino-firmware/releases/download/toolchain-x86_64/thingino-toolchain-x86_64_xburst1_uclibc_gcc16-linux-mipsel.tar.gz \
  -o "$build_dir/toolchain.tar.gz"
printf '%s  %s\n' \
  357c915630a06666f8e2eecac8ff2f766981cb07a592d341d1931b37971b130e \
  "$build_dir/toolchain.tar.gz" | shasum -a 256 -c -
tar -xzf "$build_dir/toolchain.tar.gz" -C "$build_dir/toolchain"

docker build --platform linux/amd64 \
  -f tools/firmware/Dockerfile.prudynt \
  -t openpetcam-prudynt-build:pinned-20261105 .
```

Stage the clean baseline and patched source trees from the previous section at
`$build_dir/prudynt-baseline` and `$build_dir/prudynt-patched`, including their
headers and exact target libraries. The archive extracts a directory named
`mipsel-thingino-linux-uclibc_sdk-buildroot`; retain that name. Relocate the SDK
inside its container mount, then compile in the same mount:

```sh
docker run --rm --platform linux/amd64 \
  --mount "type=bind,source=$build_dir,target=/work/build" \
  --mount "type=bind,source=$PWD/tools/firmware,target=/tools,readonly" \
  openpetcam-prudynt-build:pinned-20261105 sh -eu -c '
    /work/build/toolchain/mipsel-thingino-linux-uclibc_sdk-buildroot/relocate-sdk.sh
    bash /tools/build_prudynt.sh \
      /work/build/prudynt-patched \
      /work/build/toolchain/mipsel-thingino-linux-uclibc_sdk-buildroot/bin/mipsel-linux- \
      /work/build/prudynt-fixed-v1 \
      6afc4400-opc-access-unit-v1 4
  '
```

The result is `$build_dir/prudynt-fixed-v1` on the host. For the unpatched build,
use `/work/build/prudynt-baseline`, output `/work/build/prudynt-baseline-bin`,
and version tag `6afc4400-baseline` in the helper invocation. Do not relocate the
SDK to a host path and then execute it under a different container path.

The helper explicitly selects O32 little-endian MIPS32r2, **hard float / FP32**,
legacy NaNs, dynamic linking, AAC/Opus and burned-in OSD. Websockets, MP3, FLAC,
prebuffer and execinfo are off, matching the reference 8 MB flash profile and
observed dependency list. It fixes `SOURCE_DATE_EPOCH` and the version tag,
cleans generated objects, and prints ELF metadata plus the binary digest.
These flags are a reference device configuration, not a portable camera preset.

## Required checks

Build baseline and patched trees independently. Build each from clean objects
twice and require byte-identical output within each variant. Compare the new
binary's ELF class, byte order, O32 flags, MIPS ABI attributes, interpreter,
DT_NEEDED list, and required symbol versions with the installed binary and exact
target libraries. A successful link alone is insufficient.

The reproducible baseline build is a compatible replacement candidate, not a
claim that the original installed binary has been reproduced byte-for-byte.
The compiler and build metadata differ. Do not run the resulting binary with
`--help` as a loader check: this Prudynt revision initializes the streamer rather
than treating that argument as a harmless help request.

Deployment and bounded hardware validation are separate steps with a verified
stock backup and automatic rollback. The source-clock repair needs observed SDK
pack-boundary checks and post-install RTP/recording evidence; a build receipt
does not establish either.
