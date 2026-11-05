"""Every inline <script> in the app's pages and every .js file the server ships parses (node --check), with the
server-side {{PLACEHOLDERS}} filled the way app.py fills them. A syntax error in index.html blanks the whole app and
no other test would notice. Skips (passes) when node is not installed.
  python3 tests/server/js_syntax_test.py
"""
import glob
import os
import re
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
if not shutil.which("node"):
    print("js syntax: skipped (no node)")
    sys.exit(0)
bad = []


def check(code, name, module=False):
    code = re.sub(r"\{\{[A-Z_]+\}\}", "null", code)            # server-side placeholders (e.g. {{PET_JS}})
    with tempfile.NamedTemporaryFile("w", suffix=".mjs" if module else ".js", delete=False) as t:
        t.write(code)
    r = subprocess.run(["node", "--check", t.name], capture_output=True, text=True)
    os.unlink(t.name)
    if r.returncode:
        bad.append(f"{name}: {[l for l in r.stderr.splitlines() if 'Error' in l][:1]}")


for f in sorted(glob.glob(os.path.join(ROOT, "server", "*.html"))):
    html = open(f).read()
    for i, m in enumerate(re.finditer(r"<script(?![^>]*\bsrc=)([^>]*)>(.*?)</script>", html, re.S)):
        if "ld+json" in m.group(1) or not m.group(2).strip():
            continue
        check(m.group(2), f"{os.path.basename(f)} script {i}", 'type="module"' in m.group(1))
for f in sorted(glob.glob(os.path.join(ROOT, "server", "*.js"))):
    src = open(f).read()
    # an ES module (import/export at the top level) must be parsed AS a module: `node --check file.js` did not report a
    # missing bracket in room.js (2026-11-05), and the browser then refused the whole Room tab
    check(src, os.path.basename(f), module=bool(re.search(r"^\s*(import|export)\s", src, re.M)))
assert not bad, bad
print("js syntax ok")
