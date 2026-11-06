"""Execute the actual privacy UI transitions: no retained image or false success."""
import json
from pathlib import Path
import re
import subprocess
import unittest

HTML = Path(__file__).resolve().parents[2] / 'server/index.html'


def function(name):
    match = re.search(r'^(?:async )?function ' + name + r'\(.*?^}\n', HTML.read_text(), re.M | re.S)
    assert match, name
    return match.group(0)


class PrivacyUiTest(unittest.TestCase):
    def run_ui(self, scenario):
        src = HTML.read_text()
        begin = src.index('$("privacy-btn").onclick = async () => {')
        end = src.index('// Where the 16:9 picture', begin)
        code = r'''
let privacyBlocksLive=false, privacyBusy=false, privacyView=null, privacyPending=0, privacyIntentFloor=null, privacyUncertain=false;
let curTab='live', live=null, stopped=0, reconnected=0, tracksStopped=0, loaded=0, polls=0;
const document={hidden:false};
class El {
 constructor(){this.textContent='';this.disabled=false;this.attrs={};this.style={};this.classes=new Set();
 this.classList={toggle:(k,on)=>on?this.classes.add(k):this.classes.delete(k),add:k=>this.classes.add(k),remove:k=>this.classes.delete(k)};}
 setAttribute(k,v){this.attrs[k]=String(v);}getAttribute(k){return this.attrs[k]||null;}
 removeAttribute(k){delete this.attrs[k];} pause(){} load(){loaded++;}
}
const els={};const $=id=>els[id]||(els[id]=new El());
$('live').srcObject={getTracks:()=>[{stop(){tracksStopped++;}}]};
const pauseLive=()=>{stopped++;},startLive=()=>{reconnected++;},cancelTalkForPrivacy=()=>{};
let response={privacy:false,privacy_status:{requested:false,confirmed:true,mode:'capture_running',phase:'running'}};
let fail=false, during=null;
const api=async()=>{during={blocked:privacyBlocksLive,headline:$('conn').textContent,stopped,tracksStopped};if(fail)throw Error('camera unreachable');return response;};
const pollLive=async()=>{polls++;};
''' + ''.join(function(n) for n in ['showPrivacy', 'privacyHeadline', 'privacyNotice', 'renderPrivacy']) + src[begin:end] + '\n(async()=>{\n' + scenario + '\n})().catch(e=>{console.error(e.stack);process.exit(1)});'
        result = subprocess.run(['node', '-e', code], capture_output=True, text=True, timeout=15)
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    def test_confirmed_capture_stop_clears_room_image_and_describes_power_honestly(self):
        r = self.run_ui('''renderPrivacy({privacy:{requested:true,confirmed:true,mode:'capture_stopped',phase:'paused'}});
console.log(JSON.stringify({blocked:privacyBlocksLive,tracksStopped,loaded,src:$('live').srcObject,note:$('privacy-note').textContent,headline:$('conn').textContent,button:$('privacy-btn').attrs,disabled:$('sound-btn').disabled}));''')
        self.assertTrue(r['blocked'])
        self.assertEqual((r['tracksStopped'], r['loaded'], r['src']), (1, 1, None))
        self.assertIn('capture stopped', r['note'])
        self.assertIn('stays powered', r['note'])
        self.assertEqual(r['headline'], 'Camera paused')
        self.assertTrue(r['disabled'])
        self.assertEqual(r['button']['aria-pressed'], 'true')

    def test_failed_confirmation_never_claims_camera_paused_or_reconnects(self):
        r = self.run_ui('''renderPrivacy({privacy:{requested:true,confirmed:false,phase:'error',target:true}});
console.log(JSON.stringify({blocked:privacyBlocksLive,reconnected,note:$('privacy-note').textContent,headline:$('conn').textContent}));''')
        self.assertTrue(r['blocked'])
        self.assertEqual(r['reconnected'], 0)
        self.assertIn('could not be confirmed', r['note'])
        self.assertEqual(r['headline'], 'Privacy needs attention')

    def test_server_unreachable_retains_the_last_privacy_gate(self):
        r = self.run_ui('''renderPrivacy({privacy:{requested:true,confirmed:true,mode:'capture_stopped',phase:'paused'}});
renderPrivacy(null);console.log(JSON.stringify({blocked:privacyBlocksLive,reconnected,loaded}));''')
        self.assertTrue(r['blocked'])
        self.assertEqual(r['reconnected'], 0)
        self.assertEqual(r['loaded'], 1, 'repeated status must not reload the cleared player')

    def test_resume_stays_blocked_until_verified_then_reconnects_once(self):
        r = self.run_ui('''renderPrivacy({privacy:{requested:true,confirmed:true,mode:'capture_stopped',phase:'paused'}});
await $('privacy-btn').onclick();console.log(JSON.stringify({during,blocked:privacyBlocksLive,reconnected,polls,disabled:$('sound-btn').disabled}));''')
        self.assertTrue(r['during']['blocked'])
        self.assertEqual(r['during']['headline'], 'Resuming camera…')
        self.assertFalse(r['blocked'])
        self.assertEqual(r['reconnected'], 1)
        self.assertFalse(r['disabled'])

    def test_local_pause_hides_video_before_request_and_failure_keeps_it_hidden(self):
        r = self.run_ui('''fail=true;await $('privacy-btn').onclick();console.log(JSON.stringify({during,blocked:privacyBlocksLive,reconnected,headline:$('conn').textContent}));''')
        self.assertTrue(r['during']['blocked'])
        self.assertEqual(r['during']['tracksStopped'], 1)
        self.assertEqual(r['during']['headline'], 'Pausing camera…')
        self.assertTrue(r['blocked'])
        self.assertEqual(r['reconnected'], 0)
        self.assertEqual(r['headline'], 'Privacy needs attention')

    def test_old_active_status_cannot_reopen_after_a_newer_pause(self):
        r = self.run_ui("""renderPrivacy({privacy:{requested:true,confirmed:true,mode:'capture_stopped',phase:'paused',intent_changed_at:200}});
renderPrivacy({privacy:{requested:false,confirmed:true,mode:'capture_running',phase:'active',intent_changed_at:100}});
console.log(JSON.stringify({blocked:privacyBlocksLive,reconnected}));""")
        self.assertTrue(r['blocked'])
        self.assertEqual(r['reconnected'], 0)

    def test_newer_explicit_resume_from_another_device_can_reopen(self):
        r = self.run_ui("""renderPrivacy({privacy:{requested:true,confirmed:true,mode:'capture_stopped',phase:'paused',intent_changed_at:200}});
renderPrivacy({privacy:{requested:false,confirmed:true,mode:'capture_running',phase:'active',intent_changed_at:300}});
console.log(JSON.stringify({blocked:privacyBlocksLive,reconnected}));""")
        self.assertFalse(r['blocked'])
        self.assertEqual(r['reconnected'], 1)

    def test_unknown_post_failure_cannot_be_cleared_by_an_old_active_poll(self):
        r = self.run_ui("""fail=true;await $('privacy-btn').onclick();
renderPrivacy({privacy:{requested:false,confirmed:true,mode:'capture_running',phase:'active',intent_changed_at:100}});
console.log(JSON.stringify({blocked:privacyBlocksLive,reconnected}));""")
        self.assertTrue(r['blocked'])
        self.assertEqual(r['reconnected'], 0)

    def test_legacy_blackout_is_not_described_as_stopped_capture(self):
        r = self.run_ui('''renderPrivacy({privacy:{requested:true,confirmed:true,mode:'blackout',phase:'paused'}});
console.log(JSON.stringify({note:$('privacy-note').textContent}));''')
        self.assertIn('Video hidden', r['note'])
        self.assertNotIn('capture stopped', r['note'])


if __name__ == '__main__':
    unittest.main()
