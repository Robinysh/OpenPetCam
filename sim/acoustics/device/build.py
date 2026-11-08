"""Build supplemental observation artifacts; never deploy or restart a device."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import shutil
import subprocess


def build(output, target="mipsel"):
    here = Path(__file__).resolve().parent
    out = Path(output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    if target == "mipsel":
        zig = shutil.which("zig")
        if not zig:
            raise ValueError("Zig is required for the MIPS cross-build")
        cc = [zig, "cc", "-target", "mipsel-linux-musleabi", "-mcpu=mips32r2", "-msoft-float"]
        shared = ["-shared", "-nostdlib", "-Wl,--allow-shlib-undefined", "-s"]
        executable = ["-static", "-s"]
    elif target == "native":
        cc = [shutil.which("cc") or "cc"]
        shared = ["-dynamiclib", "-undefined", "dynamic_lookup"] if platform.system() == "Darwin" else ["-shared", "-ldl"]
        executable = []
    else:
        raise ValueError("target must be mipsel or native")
    flags = ["-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-fno-stack-protector"]
    # Zig enables target libc headers with -lc, even for compile-only. The
    # preload link is separate and deliberately has no libc dependency.
    compile_libc = ["-lc"] if target == "mipsel" else []
    commands = [cc + flags + compile_libc + ["-fPIC", "-c", str(here / "tap.c"), "-o", str(out / "tap.o")],
                cc + [str(out / "tap.o"), *shared, "-o", str(out / "opc_audio_tap.so")],
                # -lm: glibc keeps ceil() out of libc (macOS and musl don't)
                cc + flags + compile_libc + [str(here / "reader.c"), *executable, "-o", str(out / "opc-audio-stream"), "-lm"]]
    for cmd in commands:
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode:
            raise subprocess.CalledProcessError(r.returncode, cmd, r.stdout, r.stderr[-2000:] or r.stdout[-2000:])
    paths = [here / n for n in ("tap.c", "reader.c", "imp_abi.h", "build.py", "decode.py")]
    artifacts = [out / "opc_audio_tap.so", out / "opc-audio-stream"]
    manifest = {"target": target, "commands": commands,
                "source_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
                "artifacts": {p.name: {"sha256": hashlib.sha256(p.read_bytes()).hexdigest(), "bytes": p.stat().st_size} for p in artifacts},
                "prudynt_source_pin": "6afc4400c54a4739feeb72ac884587409577a2af",
                "runtime_status": "offline build only; device ABI/reference operation not yet verified"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", choices=("native", "mipsel"), default="mipsel")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    try:
        print(json.dumps(build(a.output, a.target), indent=2))
    except subprocess.CalledProcessError as e:
        raise SystemExit(e.stderr or str(e)) from e
