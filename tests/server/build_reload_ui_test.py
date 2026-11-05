"""Verify the already-shipped stale-page fix through its real JavaScript."""
import json
from pathlib import Path
import re
import subprocess
import unittest


SRC = (Path(__file__).resolve().parents[2] / "server/index.html").read_text()
CHECK = re.search(r"^async function checkBuild\(.*?^}\n", SRC, re.M | re.S).group(0)


class BuildReloadTest(unittest.TestCase):
    def check(self, *, build="new", visible=True, holding=False, last=0, ok=True):
        js = "const options=" + json.dumps(locals() | {"self": None}) + ";\n" + r'''
const BUILD="old", rec=null, holding=options.holding, document={visibilityState:options.visible?"visible":"hidden"};
let reloaded=0, fetched=0;const saved={"petcam.reloaded":options.last};
Date.now=()=>100000;
const recall=k=>saved[k]||null, store=(k,v)=>saved[k]=v;
const location={reload:()=>reloaded++};
const fetch=async()=>{fetched++;return {ok:options.ok,json:async()=>({build:options.build})}};
''' + CHECK + r'''
checkBuild().then(()=>console.log(JSON.stringify({reloaded,fetched,saved})));
'''
        result = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_new_build_reloads_and_records_guard(self):
        result = self.check()
        self.assertEqual(result["reloaded"], 1)
        self.assertEqual(result["saved"]["petcam.reloaded"], 100000)

    def test_same_build_does_not_reload(self):
        self.assertEqual(self.check(build="old")["reloaded"], 0)

    def test_background_page_does_not_fetch_or_reload(self):
        self.assertEqual(self.check(visible=False)["fetched"], 0)

    def test_held_talk_does_not_fetch_or_reload(self):
        self.assertEqual(self.check(holding=True)["fetched"], 0)

    def test_recent_reload_does_not_loop(self):
        self.assertEqual(self.check(last=90000)["reloaded"], 0)

    def test_failed_build_check_does_not_reload(self):
        self.assertEqual(self.check(ok=False)["reloaded"], 0)


if __name__ == "__main__":
    unittest.main()
