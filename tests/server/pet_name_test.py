#!/usr/bin/env python3
"""The pet's name is set from the app (pet.json) and wins over site.env; every process follows a rename without a
restart; bad names are refused; with no name anywhere the app asks for one. Runs anywhere."""
import importlib
import json
import os
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "server"))
d = tempfile.mkdtemp()
site = os.path.join(d, "site.env")
open(site, "w").write("PUBLIC_HOST=example.test\n")
os.environ.update(PETCAM_SITE_ENV=site, PETCAM_PET_FILE=os.path.join(d, "pet.json"))
import petconf  # noqa: E402

fails = 0


def check(name, cond):
    global fails
    print(("PASS " if cond else "FAIL ") + name)
    fails += 0 if cond else 1


check("unnamed: asks for a name", not petconf.NAMED and petconf.PET == "Your pet" and not json.loads(petconf.template_vars()["PET_JS"])["named"])
check("app is called OpenPetCam", petconf.APP_NAME == "OpenPetCam" and petconf.template_vars()["APP_NAME"] == "OpenPetCam")
open(site, "w").write("PET_NAME=Biscuit\nPET_PRONOUN=he\n")
importlib.reload(petconf)
check("site.env name still works", petconf.PET == "Biscuit" and petconf.POSS == "his" and petconf.NAMED)
petconf.save_pet("Mochi", "she")
check("app setting wins over site.env", petconf.PET == "Mochi" and petconf.SUBJ == "she")
time.sleep(0.05)
with open(os.environ["PETCAM_PET_FILE"], "w") as f:     # another process renames her
    json.dump({"name": "Nori", "pronoun": "they"}, f)
os.utime(os.environ["PETCAM_PET_FILE"], (time.time() + 5, time.time() + 5))
js = json.loads(petconf.template_vars()["PET_JS"])
check("rename picked up without restart", js["name"] == "Nori" and js["pronoun"] == "they" and js["named"])
check("a newline becomes a space", petconf.valid_name("Rex\nAlert") == "Rex Alert")
for bad in ("", "   ", "x" * 25, "<script>", "a/b", "Rex\x00"):
    try:
        petconf.save_pet(bad, "she")
        check(f"refuses {bad!r}", False)
    except ValueError:
        check(f"refuses {bad!r}", True)
try:
    petconf.save_pet("Rex", "it")
    check("refuses unknown pronoun", False)
except ValueError:
    check("refuses unknown pronoun", True)
check("accepts names in any language", petconf.valid_name("Мурка") == "Мурка" and petconf.valid_name("  O'Malley-Jr  ") == "O'Malley-Jr")
check("escaped in HTML", "&lt;" in (petconf.html_escape("<b>")))
print("\nall passed" if not fails else f"\n{fails} failed")
sys.exit(1 if fails else 0)
