"""Saved-view insights render conservative logged measurements without moving the camera."""
import json
from pathlib import Path
import re
import subprocess
import unittest


HTML = Path(__file__).resolve().parents[2] / "server/index.html"


class ViewInsightsUiTest(unittest.TestCase):
    def render(self, data):
        src = HTML.read_text()
        match = re.search(r"^function renderViewInsights\(.*?^}\n",src,re.M|re.S)
        self.assertIsNotNone(match,"the saved-view request has no statistics/suggestions panel")
        code = r'''
class El {constructor(tag="div"){this.tag=tag;this.children=[];this._text="";}
set textContent(v){this._text=String(v);this.children=[];}get textContent(){return this._text+this.children.map(x=>x.textContent).join("");}
appendChild(el){this.children.push(el);return el;}replaceChildren(){this.children=[];this._text="";}}
const body=new El(),$=()=>body,document={createElement:t=>new El(t)};
const fmtDur=m=>Math.round(m)+"m";
''' + match.group(0) + "\nrenderViewInsights(" + json.dumps(data) + ");\nconsole.log(JSON.stringify({text:body.textContent,buttons:body.children.filter(x=>x.tag==='button').length}));"
        result=subprocess.run(["node","-e",code],capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)
        return json.loads(result.stdout)

    def test_observed_time_bouts_and_suggestions_are_visible_with_scope(self):
        result=self.render({"days":7,"observed_seconds":120,"views":[{"name":"Door view","observed_seconds":90,"visible_seconds":60,"visits":2}],
                            "suggestions":[{"x":1200,"y":500,"visible_seconds":30,"visits":1}]})
        for text in ["Follow-logged stationary intervals","Door view","visible bouts","Suggested view","1200","not full-day"]:
            self.assertIn(text,result["text"])
        self.assertEqual(result["buttons"],0)

    def test_no_coverage_is_explicit_and_never_claims_zero_pet_visits(self):
        result=self.render({"days":7,"observed_seconds":0,"views":[],"suggestions":[]})
        self.assertIn("No usable",result["text"])
        self.assertNotIn("NaN",result["text"])
        self.assertNotIn("0 visits",result["text"])


if __name__ == "__main__": unittest.main()
