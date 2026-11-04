"""install.sh keeps a learned Follow policy on a release install (no models shipped), and still retires one that a
checkout with models no longer ships (review 2026-11-04: a normal release retired the camera's learned policy).
  python3 tests/server/install_retire_test.py
"""
import os
import re
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, "..", "..", "server", "install.sh")).read()
m = re.search(r"# A learned Follow policy that is no longer shipped.*?\ndone\n", src, re.S)
assert m, "retirement block not found in install.sh"

with tempfile.TemporaryDirectory() as d:
    block = m.group(0).replace("/opt/petcam/models", f"{d}/opt")
    os.makedirs(f"{d}/opt")
    os.makedirs(f"{d}/release")
    os.makedirs(f"{d}/checkout/models")
    open(f"{d}/opt/follow_policy.json", "w").close()
    open(f"{d}/checkout/models/yolox_nano.onnx", "w").close()

    def run(srcdir):
        subprocess.run(["bash", "-c", f"set -e; SRC={srcdir}\n{block}"], check=True)

    run(f"{d}/release")
    assert os.path.exists(f"{d}/opt/follow_policy.json"), "a release (no models) retired the learned policy"
    run(f"{d}/checkout")
    assert not os.path.exists(f"{d}/opt/follow_policy.json"), "a checkout that no longer ships it must retire it"
    assert any("retired" in f for f in os.listdir(f"{d}/opt")), "retired, not deleted"
print("install retire ok")
