"""Every server module that another server module imports must be installed by install.sh (2026-11-05: roommap.py
was missing from its list and the watcher crash-looped for 2 min after a deploy).
  python3 tests/server/install_files_test.py"""
import os
import re

S = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "server")
mods = {f[:-3] for f in os.listdir(S) if f.endswith(".py")}
inst = open(os.path.join(S, "install.sh")).read()
missing = set()
for f in os.listdir(S):
    if f.endswith(".py"):
        for m in re.findall(r"^\s*(?:from|import)\s+(\w+)", open(os.path.join(S, f)).read(), re.M):
            if m in mods and f"$SRC/{m}.py" not in inst:
                missing.add((f, m))
assert not missing, f"imported but not installed by install.sh: {sorted(missing)}"
print("install files ok")
