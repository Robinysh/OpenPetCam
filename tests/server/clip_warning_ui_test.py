"""Incomplete recordings are explained on the real moment card, while ordinary clips still play."""
import json
from pathlib import Path
import re
import subprocess
import unittest


HTML=Path(__file__).resolve().parents[2]/"server/index.html"


class ClipWarningUiTest(unittest.TestCase):
    def card(self, event):
        src=HTML.read_text()
        helper=re.search(r"^function momentCard\(.*?^}\n",src,re.M|re.S)
        if helper:
            code=helper.group(0)
        else:
            old=re.search(r"  for \(const e of evs\) \{\n(.*?)    feed.appendChild\(card\);",src,re.S)
            self.assertIsNotNone(old)
            code="function momentCard(e) {\n"+old.group(1)+"return card;\n}\n"
        js=r'''
class El{constructor(){this.children=[];this.nodes={};this.attributes={};this.style={};this.innerHTML="";this.textContent="";}
querySelector(s){return this.nodes[s]||(this.nodes[s]=new El());}appendChild(x){this.children.push(x);}
setAttribute(k,v){this.attributes[k]=v;}}
const document={createElement:()=>new El()},KIND={active:["Got up","x"]},SOUND_KINDS=new Set();
const fmtTime=()=>"12:00",visitLabel=()=>"Visit";const actions=[];
const openPlayer=(src)=>actions.push({type:"play",src}),peekAt=ts=>actions.push({type:"still",ts});
''' + code + "\nconst card=momentCard("+json.dumps(event)+r''');card.querySelector(".media").onclick();
console.log(JSON.stringify({text:card.children.map(x=>x.textContent).join(""),playGlyph:card.innerHTML.includes("▶"),actions}));
'''
        result=subprocess.run(["node","-e",js],capture_output=True,text=True,timeout=15)
        self.assertEqual(result.returncode,0,result.stderr)
        return json.loads(result.stdout)

    def test_incomplete_video_explains_missing_playback_and_opens_only_a_still(self):
        warning="This recording contains incomplete video. A playable clip is unavailable."
        result=self.card({"kind":"active","ts":1000,"clip":None,"thumb":"/media/snaps/example.jpg","clip_warning":warning})
        self.assertIn(warning,result["text"])
        self.assertFalse(result["playGlyph"])
        self.assertEqual(result["actions"],[{"type":"still","ts":1000}])

    def test_ordinary_clip_still_has_play_and_plays_its_original_url(self):
        result=self.card({"kind":"active","ts":1000,"clip":"/media/clips/valid.mp4","thumb":None,"clip_warning":None})
        self.assertTrue(result["playGlyph"])
        self.assertEqual(result["actions"],[{"type":"play","src":"/media/clips/valid.mp4"}])

    def test_partial_audio_warning_keeps_its_usable_video_playable(self):
        warning="Video is available, but part of the recording has no audio."
        result=self.card({"kind":"active","ts":1000,"clip":"/media/clips/partial-audio.mp4","thumb":None,"clip_warning":warning})
        self.assertIn(warning,result["text"])
        self.assertTrue(result["playGlyph"])
        self.assertEqual(result["actions"],[{"type":"play","src":"/media/clips/partial-audio.mp4"}])


if __name__=="__main__":unittest.main()
