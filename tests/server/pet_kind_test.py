"""PET_KIND picks the verifier's zero-shot classes: dog breeds (151-268) or domestic cats (281-285).
Run: python3 tests/server/pet_kind_test.py (no models, camera or server needed)."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CASES = {"": "dog", "dog": "dog", "cat": "cat", "CAT": "cat", "ferret": "dog"}
for given, want in CASES.items():
    env = Path(tempfile.mkdtemp()) / "site.env"
    env.write_text("PET_NAME=T\n" + (f"PET_KIND={given}\n" if given else ""))
    out = subprocess.run([sys.executable, "-I", "-c", f"import sys; sys.path.insert(0, '{ROOT}/server'); import petconf; print(petconf.PET_KIND)"],
                         env={**os.environ, "PETCAM_SITE_ENV": str(env)}, capture_output=True, text=True, check=True).stdout.strip()
    assert out == want, (given, out, want)

dog = json.load(open(ROOT / "server/verifier.default.json"))
cat = json.load(open(ROOT / "server/verifier.cat.json"))
assert dog["classes"] == [151, 268] and cat["classes"] == [281, 285], (dog, cat)
assert cat["mode"] == "zeroshot" and 0 < cat["threshold"] < 1

# fetch_models.sh: an untrained verifier.json follows PET_KIND; a trained one is left alone.
tmp = Path(tempfile.mkdtemp())
(tmp / "models").mkdir(); (tmp / "server").mkdir(); (tmp / "tools").mkdir()
for n in ("verifier.default.json", "verifier.cat.json"):
    (tmp / "server" / n).write_text((ROOT / "server" / n).read_text())
snippet = (ROOT / "tools/fetch_models.sh").read_text().split("# The verifier config:")[1]
(tmp / "tools/pick.sh").write_text("set -e\ncd \"$(dirname \"$0\")/../models\"\n# " + snippet)


def pick(kind, existing=None):
    vj = tmp / "models/verifier.json"
    vj.unlink(missing_ok=True)
    if existing is not None:
        vj.write_text(json.dumps(existing))
    subprocess.run(["bash", str(tmp / "tools/pick.sh")], env={**os.environ, "PET_KIND": kind}, check=True)
    return json.loads(vj.read_text())


assert pick("cat")["classes"] == [281, 285]
assert pick("dog")["classes"] == [151, 268]
assert pick("cat", dog)["classes"] == [281, 285]                         # untrained dog default -> cat
assert pick("dog", cat)["classes"] == [151, 268]                         # and back
trained = {"mode": "logreg", "classes": [151, 268], "threshold": 0.5}
assert pick("cat", trained) == trained                                   # a trained verifier is never replaced
print("ok")
