"""Actual player error/rejected-play/close behavior; browser media is replaced only at its boundary."""
import json
import os
from pathlib import Path
import re
import subprocess
import unittest


HTML = Path(os.environ.get("PETCAM_UI_SOURCE", Path(__file__).resolve().parents[2] / "server/index.html"))


class ClipPlayerUiTest(unittest.TestCase):
    def run_player(self, scenario):
        src = HTML.read_text()
        player = re.search(r"^function openPlayer\(.*?^}\n", src, re.M | re.S).group(0)
        close = re.search(r'\$\("player-close"\)\.onclick = \(\) => \{.*?^};', src, re.M | re.S).group(0)
        js = r'''
class El {
 constructor(){this.attributes={};this.classes=new Set(["hidden"]);this.textContent="";
 this.classList={add:c=>this.classes.add(c),remove:c=>this.classes.delete(c),contains:c=>this.classes.has(c)};}
 set src(v){this.attributes.src=v;} get src(){return this.attributes.src;}
 set href(v){this.attributes.href=v;} get href(){return this.attributes.href;}
 getAttribute(k){return this.attributes[k]||null;} removeAttribute(k){delete this.attributes[k];}
 play(){return pending||Promise.resolve();} pause(){}
}
let pending=null;const els={},$=id=>els[id]||(els[id]=new El());
let errors=[];const track=()=>{},reportError=(...args)=>errors.push(args);
''' + player + close + "\n(async()=>{\n" + scenario + r'''
await Promise.resolve();await Promise.resolve();
console.log(JSON.stringify({fallback:!$("player-source").classList.contains("hidden"),href:$("player-source").href,
 title:$("player-title").textContent,errors:errors.length}));
})().catch(e=>{console.error(e.stack);process.exit(1)});
'''
        result = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_media_error_offers_the_exact_clip_directly(self):
        result = self.run_player('openPlayer("/media/clips/test.mp4","A moment");$("player-video").error={code:4};$("player-video").onerror();')
        self.assertTrue(result["fallback"])
        self.assertEqual(result["href"], "/media/clips/test.mp4")
        self.assertEqual(result["errors"], 1)

    def test_autoplay_rejection_tells_user_to_press_play(self):
        result = self.run_player('pending=Promise.reject({name:"NotAllowedError",message:"blocked"});openPlayer("/media/clips/test.mp4","A moment");')
        self.assertIn("tap Play", result["title"])
        self.assertTrue(result["fallback"])

    def test_late_rejection_after_close_cannot_restore_fallback_or_report_error(self):
        result = self.run_player('let reject;pending=new Promise((_,r)=>reject=r);openPlayer("/media/clips/test.mp4","A moment");$("player-close").onclick();reject({name:"NotAllowedError",message:"late"});')
        self.assertFalse(result["fallback"])
        self.assertEqual(result["errors"], 0)


if __name__ == "__main__":
    unittest.main()
