"""Exercise request-history pagination/status through real frontend JS; no server imports."""
import json
from pathlib import Path
import subprocess
import unittest


HTML = Path(__file__).resolve().parents[2] / "server/index.html"


class RequestHistoryFrontendTest(unittest.TestCase):
    def page(self, scenario):
        src = HTML.read_text()
        block = src[src.index("function replyStatus("):src.index('$("report-send").onclick')]
        js = r'''
class Element {
 constructor(){this.children=[];this._text="";this.className="";this.scrollTop=0;this.classes=new Set();
 this.classList={contains:c=>this.classes.has(c),toggle:(c,on)=>on?this.classes.add(c):this.classes.delete(c)};}
 set textContent(v){this._text=String(v);this.children=[];}
 get textContent(){return this._text+this.children.map(c=>typeof c==="string"?c:c.textContent).join("");}
 set innerHTML(v){this.children=[];this._text="";}get scrollHeight(){return this.children.length*10;}
 appendChild(el){this.children.push(el);return el;}append(...els){this.children.push(...els);}setAttribute(){}
}
const els={},$=id=>els[id]||(els[id]=new Element()),document={createElement:()=>new Element()};
const setTimeout=()=>1,clearTimeout=()=>{};
const records=Array.from({length:50},(_,i)=>({id:String(50-i).padStart(4,"0"),ts:1000-i,
 note:"request "+(50-i),reply:{status:"answered",text:"Historical answer",fix:{status:"live"}},
 status:{state:i===0?"in_progress":"needs_review",label:i===0?"In progress":"Needs review",reason:"No verified result yet"},
 resolution:{components:[{id:"part",title:"Remaining work",state:"not_started",evidence:["Hardware limitation still needs review"]}]}}));
const requests=[],api=async path=>{
 requests.push(path);if(!path.includes("page="))return records.slice(0,20);
 const query=new URL(path,"https://camera.invalid").searchParams,before=query.get("before");
 const start=before?records.findIndex(r=>r.id===before)+1:0,page=records.slice(start,start+20);
 return {reports:page,total:records.length,next_before:start+20<records.length?page.at(-1).id:null};
};
''' + block + "\n(async()=>{\n" + scenario + r'''
console.log(JSON.stringify({shown:$("report-list").children.filter(c=>c.className==="msg me").length,
 text:$("report-list").textContent,nextHidden:$("report-older").classList.contains("hidden"),
 count:$("report-history").textContent,requests}));
})().catch(e=>{console.error(e.stack);process.exit(1)});
'''
        result = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_initial_page_shows_twenty_with_older_available(self):
        result = self.page("await loadReports(true);")
        self.assertEqual(result["shown"], 20)
        self.assertFalse(result["nextHidden"])
        self.assertIn("50", result["count"])

    def test_all_fifty_are_reachable_and_survive_polling(self):
        result = self.page("await loadReports(true);await loadReports(false,true);await loadReports(false,true);await loadReports(false);")
        self.assertEqual(result["shown"], 50)
        self.assertTrue(result["nextHidden"])
        self.assertIn("request 1", result["text"])
        self.assertIn("request 50", result["text"])

    def test_canonical_status_overrides_old_live_claim_and_lists_remaining_work(self):
        result = self.page("await loadReports(true);")
        for text in ["Needs review", "No verified result yet", "Remaining work", "Hardware limitation still needs review"]:
            self.assertIn(text, result["text"])
        self.assertNotIn("Fixed", result["text"])

    def test_older_button_uses_cursor(self):
        result = self.page('await loadReports(true);if($("report-older").onclick)await $("report-older").onclick();')
        self.assertEqual(result["shown"], 40)
        self.assertTrue(any("before=0031" in p for p in result["requests"]), result)


if __name__ == "__main__":
    unittest.main()
