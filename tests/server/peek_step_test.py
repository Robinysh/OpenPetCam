"""Looking back one minute at a time: each step shows the next saved picture and says which minute it is."""
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("PETCAM_SITE_ENV", str(ROOT / "config/site.env.example"))
sys.path.insert(0, str(ROOT / "server"))
import app

BASE = int(time.mktime((2026, 3, 10, 12, 0, 0, 0, 0, -1)))   # a local noon, far from midnight and clock changes
HTML = ROOT / "server/index.html"


class SnapStepServerTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = temp.name
        mock = patch.object(app, "STATE_DIR", self.root)
        mock.start()
        self.addCleanup(mock.stop)
        for minute in (0, 1, 7, 8):      # pictures at 12:00, 12:01, 12:07, 12:08: a search left 12:02-12:06 empty
            self.save(minute)

    def save(self, minute):
        t = time.localtime(BASE + minute * 60)
        folder = Path(self.root) / "snaps" / time.strftime("%Y-%m-%d", t)
        folder.mkdir(parents=True, exist_ok=True)
        (folder / time.strftime("%H-%M.jpg", t)).write_bytes(b"x")

    def test_step_goes_to_the_neighbouring_minute(self):
        self.assertEqual(app.snap_at(BASE, 1)["ts"], BASE + 60)
        self.assertEqual(app.snap_at(BASE + 60, -1)["ts"], BASE)

    def test_step_crosses_minutes_with_no_picture_instead_of_sticking(self):
        self.assertEqual(app.snap_at(BASE + 60, 1)["ts"], BASE + 7 * 60)
        self.assertEqual(app.snap_at(BASE + 7 * 60, -1)["ts"], BASE + 60)

    def test_step_accepts_the_query_string_form(self):
        self.assertEqual(app.snap_at(str(BASE), "1")["ts"], BASE + 60)
        self.assertEqual(app.snap_at(str(BASE + 60), "-1")["ts"], BASE)

    def test_step_past_the_end_finds_nothing(self):
        self.assertIsNone(app.snap_at(BASE + 8 * 60, 1)["snap"])
        self.assertIsNone(app.snap_at(BASE, -1)["snap"])

    def test_unknown_step_is_a_plain_lookup(self):
        self.assertEqual(app.snap_at(BASE, "sideways")["ts"], BASE)

    def test_plain_lookup_reports_the_minute_it_really_shows(self):
        self.assertEqual(app.snap_at(BASE + 3 * 60)["ts"], BASE + 60)      # nearest within 2 minutes, looking back first
        self.assertTrue(app.snap_at(BASE)["snap"].endswith("/12-00.jpg"))
        self.assertIsNone(app.snap_at(BASE + 4 * 60 + 30)["snap"])         # more than 2 minutes from any picture


class PeekStepUiTest(unittest.TestCase):
    def run_steps(self, answers, clicks):
        """Runs the app's real peek code against a stub page and API. answers: replies in order."""
        src = HTML.read_text()
        start = src.index("function peekShowTime(ts)")
        end = src.index('$("peek-play").onclick')
        code = src[start:end]
        js = r'''
class Cl{constructor(){this.s=new Set();}add(x){this.s.add(x);}remove(x){this.s.delete(x);}
contains(x){return this.s.has(x);}toggle(x,on){on?this.s.add(x):this.s.delete(x);}}
class El{constructor(){this.classList=new Cl();this.style={};this.textContent="";this.src="";this.disabled=false;this.kids=[];this.paused=0;}
pause(){this.paused++;}querySelectorAll(){const k=this.kids;return k.map(x=>({remove(){k.splice(k.indexOf(x),1);}}));}appendChild(x){this.kids.push(x);}}
const els={};const $=(id)=>els[id]||(els[id]=new El());
const document={createElement:()=>new El()};
let peekTs=null,laneRange=[0,144],dayOffset=0;const dayString=()=>"2026-03-10";
const answers=__ANSWERS__;const asked=[];
const api=async(u)=>{asked.push(u);const a=answers.shift();if(a==="fail")throw new Error("net");return a;};
''' + code + r'''
(async()=>{
  const steps=[];
  await peekAt(__TS__);
  steps.push({time:$("peek-time").textContent,peekTs,src:$("peek-img").src,note:$("peek-note").textContent});
  for(const c of __CLICKS__){
    await $(c<0?"peek-earlier":"peek-later").onclick();
    steps.push({peekTs,src:$("peek-img").src,note:$("peek-note").textContent,noteHidden:$("peek-note").classList.contains("hidden"),
      earlierOff:$("peek-earlier").disabled,laterOff:$("peek-later").disabled,cursors:$("timeline").kids.length});
  }
  console.log(JSON.stringify({steps,asked,videoPaused:$("peek-video").paused}));
})();
'''
        js = js.replace("__ANSWERS__", json.dumps(answers)).replace("__TS__", str(BASE)).replace("__CLICKS__", json.dumps(clicks))
        result = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_page_has_step_buttons_in_the_peek_card(self):
        card = re.search(r'id="peek".*?id="peek-note"', HTML.read_text(), re.S)
        self.assertIsNotNone(card)
        self.assertIn('id="peek-earlier"', card.group(0))
        self.assertIn('id="peek-later"', card.group(0))

    def test_step_buttons_do_not_inherit_the_setup_guides_steps_list_layout(self):
        # A class named "steps" picks up the column layout of the numbered list elsewhere on the page.
        card = re.search(r'id="peek".*?id="peek-note"', HTML.read_text(), re.S).group(0)
        self.assertNotRegex(card, r'class="steps"')

    def test_stepping_moves_from_the_picture_shown_and_asks_the_server_for_the_neighbour(self):
        out = self.run_steps([{"snap": "/a.jpg", "ts": BASE}, {"snap": "/b.jpg", "ts": BASE + 60}, {"snap": "/c.jpg", "ts": BASE + 120}], [1, 1])
        self.assertEqual(out["asked"][1], f"/api/snap?ts={BASE}&step=1")
        self.assertEqual(out["asked"][2], f"/api/snap?ts={BASE + 60}&step=1")
        self.assertEqual(out["steps"][2]["src"], "/c.jpg")
        self.assertEqual(out["steps"][2]["peekTs"], BASE + 120)
        self.assertTrue(out["steps"][2]["noteHidden"])

    def test_a_jump_over_empty_minutes_is_named(self):
        out = self.run_steps([{"snap": "/a.jpg", "ts": BASE}, {"snap": "/b.jpg", "ts": BASE + 6 * 60}], [1])
        self.assertIn("5 min", out["steps"][1]["note"])

    def test_end_of_pictures_keeps_the_picture_and_blocks_only_that_direction(self):
        out = self.run_steps([{"snap": "/a.jpg", "ts": BASE}, {"snap": None}], [1])
        last = out["steps"][1]
        self.assertEqual(last["src"], "/a.jpg")
        self.assertEqual(last["peekTs"], BASE)
        self.assertTrue(last["laterOff"])
        self.assertFalse(last["earlierOff"])

    def test_failed_request_keeps_the_picture_and_both_buttons(self):
        out = self.run_steps([{"snap": "/a.jpg", "ts": BASE}, "fail"], [-1])
        last = out["steps"][1]
        self.assertEqual(last["src"], "/a.jpg")
        self.assertFalse(last["earlierOff"])
        self.assertFalse(last["laterOff"])
        self.assertTrue(last["note"])

    def test_only_one_timeline_cursor_follows_the_steps(self):
        out = self.run_steps([{"snap": "/a.jpg", "ts": BASE}, {"snap": "/b.jpg", "ts": BASE + 60}], [1])
        self.assertLessEqual(out["steps"][1]["cursors"], 1)

    def test_video_from_a_previous_peek_is_stopped(self):
        out = self.run_steps([{"snap": "/a.jpg", "ts": BASE}, {"snap": "/b.jpg", "ts": BASE + 60}], [1])
        self.assertGreaterEqual(out["videoPaused"], 2)


if __name__ == "__main__":
    unittest.main()
