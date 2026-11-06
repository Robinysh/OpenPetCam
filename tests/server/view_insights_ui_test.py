"""Saved-view insights render conservative logged measurements without moving the camera."""
import json
from pathlib import Path
import re
import subprocess
import unittest


HTML = Path(__file__).resolve().parents[2] / "server/index.html"


class ViewInsightsUiTest(unittest.TestCase):
    def lifecycle(self, scenario):
        src = HTML.read_text()
        start = src.index("function renderViewInsights(")
        end = src.index("function renderPan()", start)
        code = r'''
const assert=require("node:assert/strict");
class El {constructor(tag="div"){this.tag=tag;this.children=[];this._text="";this.listeners={};this.open=false;}
set textContent(v){this._text=String(v);this.children=[];}get textContent(){return this._text+this.children.map(x=>x.textContent).join("");}
appendChild(el){this.children.push(el);return el;}replaceChildren(){this.children=[];this._text="";}
addEventListener(k,fn){this.listeners[k]=fn;}}
const body=new El(),panel=new El(),$=id=>id==="view-insights"?panel:body;
const document={hidden:false,listeners:{},createElement:t=>new El(t),addEventListener(k,fn){this.listeners[k]=fn;}};
const fmtDur=m=>Math.round(m)+"m",location={href:""};
let now=0,nextTimer=0;const timers=new Map(),calls=[];Date.now=()=>now;
const setTimeout=(fn,ms)=>{const id=++nextTimer;timers.set(id,{fn,at:now+ms,ms});return id;};
const clearTimeout=id=>timers.delete(id);
const fetch=(path,opts={})=>new Promise((resolve,reject)=>calls.push({path,opts,resolve,reject}));
const api=async path=>{const r=await fetch(path);return r.json();};
const flush=async()=>{for(let n=0;n<12;n++)await Promise.resolve();};
const reply=async(i,data)=>{calls[i].resolve({ok:true,status:200,json:async()=>data});await flush();};
const toggle=async open=>{panel.open=open;panel.listeners.toggle();await flush();};
const hidden=async value=>{document.hidden=value;document.listeners.visibilitychange();await flush();};
const advance=async ms=>{now+=ms;const due=[...timers].filter(([,t])=>t.at<=now);for(const[id,t]of due){if(timers.delete(id))t.fn();}await flush();};
const ready=(name,extra={})=>({status:"ready",until:1700000000,days:7,observed_seconds:90,views:[{name,observed_seconds:90,visible_seconds:60,visits:2}],...extra});
''' + src[start:end] + "\n(async()=>{\n" + scenario + "\n})().catch(e=>{console.error(e);process.exitCode=1;});"
        result=subprocess.run(["node","-e",code],capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)

    def test_pending_polls_then_ready_stops(self):
        self.lifecycle(r'''
await toggle(true);assert.equal(calls.length,1);
await reply(0,{status:"pending",retry_after_s:5,days:7});
assert.match(body.textContent,/Summarizing recent observations/);
await advance(4999);assert.equal(calls.length,1);await advance(1);assert.equal(calls.length,2);
await reply(1,ready("Door view"));assert.match(body.textContent,/Door view/);
await advance(120000);assert.equal(calls.length,2);assert.equal(timers.size,0);
''')

    def test_stale_ready_keeps_data_and_as_of_time_during_refresh(self):
        self.lifecycle(r'''
await toggle(true);await reply(0,ready("Old view",{refreshing:true}));
assert.match(body.textContent,/Old view/);assert.match(body.textContent,/As of/);assert.match(body.textContent,/Refreshing/);
await advance(5000);assert.equal(calls.length,2);assert.match(body.textContent,/Old view/);
await reply(1,ready("New view",{until:1700000060,refreshing:false}));
assert.match(body.textContent,/New view/);assert.doesNotMatch(body.textContent,/Old view|Refreshing/);
assert.equal(timers.size,0);
''')

    def test_failed_refresh_retains_previous_summary_without_automatic_polling(self):
        self.lifecycle(r'''
await toggle(true);await reply(0,ready("Previous view",{refreshing:false,refresh_error:"Unavailable",retry_after_s:60}));
assert.match(body.textContent,/Previous view/);assert.match(body.textContent,/As of/);
assert.match(body.textContent,/Could not refresh/);assert.match(body.textContent,/close and reopen/);
await advance(120000);assert.equal(calls.length,1);assert.equal(timers.size,0);
''')

    def test_close_aborts_and_late_response_cannot_replace_reopened_panel(self):
        self.lifecycle(r'''
await toggle(true);await toggle(false);assert.equal(calls[0].opts.signal.aborted,true);
await toggle(true);assert.equal(calls.length,2);
await reply(1,ready("New view"));await reply(0,ready("Obsolete view"));
assert.match(body.textContent,/New view/);assert.doesNotMatch(body.textContent,/Obsolete/);
await toggle(false);await advance(120000);assert.equal(calls.length,2);assert.equal(timers.size,0);
''')

    def test_hidden_page_cancels_poll_and_repeated_open_events_do_not_duplicate_requests(self):
        self.lifecycle(r'''
await toggle(true);await toggle(true);assert.equal(calls.length,1);
await reply(0,{status:"pending",retry_after_s:5});await hidden(true);
await advance(120000);assert.equal(calls.length,1);assert.equal(timers.size,0);
await hidden(false);assert.equal(calls.length,2);await hidden(false);assert.equal(calls.length,2);
await reply(1,ready("Visible again"));assert.match(body.textContent,/Visible again/);
''')

    def test_error_is_explicit_and_reopening_respects_retry_delay(self):
        self.lifecycle(r'''
await toggle(true);await reply(0,{status:"error",message:"Unavailable",retry_after_s:60});
assert.match(body.textContent,/couldn't load/);assert.match(body.textContent,/Close and reopen/);
await advance(5000);assert.equal(calls.length,1);assert.equal(timers.size,0);
await toggle(false);await toggle(true);assert.equal(calls.length,1);
await advance(55000);assert.equal(calls.length,2);await reply(1,ready("Recovered"));
assert.match(body.textContent,/Recovered/);
''')

    def test_long_pending_backoff_is_bounded_and_never_rapid(self):
        self.lifecycle(r'''
await toggle(true);
for(let i=0;i<7;i++){
 await reply(i,{status:"pending",retry_after_s:0});
 const jobs=[...timers.values()];assert.equal(jobs.length,1);
 assert.equal(jobs[0].ms,Math.min(60000,5000*2**i));
 await advance(jobs[0].ms);
}
await toggle(false);assert.equal(timers.size,0);
''')

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
