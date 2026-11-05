"""Run the real live-player/status JavaScript with deterministic browser doubles.

Covers startup vs established-stream deadlines, retired asynchronous WHEP
connections, resting detections, and overlay redraws. No camera/network use.
"""
import json
from pathlib import Path
import re
import subprocess
import unittest


HTML = Path(__file__).resolve().parents[2] / "server/index.html"


def function(src, name, optional=False):
    match = re.search(r"^(?:async )?function " + name + r"\(.*?^}\n", src, re.M | re.S)
    if optional and not match:
        return ""
    assert match, name
    return match.group(0)


def node(code):
    result = subprocess.run(["node", "-e", code], capture_output=True, text=True, timeout=20)
    if result.returncode:
        raise AssertionError(result.stderr)
    return json.loads(result.stdout)


class LiveUiTest(unittest.TestCase):
    def player(self, scenario):
        src = HTML.read_text()
        section = src[src.index("let pc = null"):src.index("function showSound(")]
        return node(r'''
let now = 100000, intervals = [], timers = [], events = [], deleted = [];
Date.now = () => now;
const performance = { now: () => now - 100000 };
const video = { currentTime: 0, srcObject: null, play: () => Promise.resolve() };
const connection = { textContent: "", classList: { add() {}, remove() {} } };
const $ = (id) => id === "live" ? video : connection;
const document = { hidden: false };
const location = { href: "https://camera.invalid/" };
const track = (name, data) => events.push(name);
const setInterval = (fn) => { intervals.push(fn); return intervals.length; };
const setTimeout = (fn, ms) => { timers.push({fn, ms, live:true}); return timers.length; };
const clearTimeout = (id) => { if (timers[id-1]) timers[id-1].live = false; };
let offerNumber = 0, heldReply = null, holdPost = false;
const fetch = async (url, options = {}) => {
  if (options.method === "DELETE") { deleted.push(url); return {}; }
  const number = ++offerNumber;
  const reply = {status:201, headers:{get:() => "/cam/whep/" + number}, text:async () => "answer"};
  if (holdPost) return new Promise(resolve => { heldReply = () => resolve(reply); });
  return reply;
};
const peers = [];
class RTCPeerConnection {
  constructor() { this.connectionState = "new"; this.iceGatheringState = "complete"; peers.push(this); }
  addTransceiver() {}
  async createOffer() { return {sdp:"offer"}; }
  async setLocalDescription(desc) { this.localDescription = desc; }
  async setRemoteDescription(desc) {
    if (this.connectionState === "closed") throw Error("closed peer");
    this.connectionState = "connected"; this.onconnectionstatechange();
  }
  close() { this.connectionState = "closed"; this.onconnectionstatechange?.(); }
}
''' + section + r'''
const tick = (ms) => { now += ms; intervals.forEach(fn => fn()); };
const flush = async () => { for(let i=0;i<8;i++) await Promise.resolve(); };
(async () => {
''' + scenario + r'''
})().catch(e => { console.error(e.stack); process.exit(1); });
''')

    def test_connected_startup_gets_fifteen_seconds_for_first_picture(self):
        result = self.player('''await startLive(); tick(2000); tick(9000);
console.log(JSON.stringify({peers:peers.length, events}));''')
        self.assertEqual(result["peers"], 1, "startup was replaced before its first-frame allowance")

    def test_startup_without_any_connection_is_still_bounded(self):
        result = self.player('''holdPost = true; startLive(); await flush(); tick(16000);
console.log(JSON.stringify({peers:peers.length, events}));''')
        self.assertEqual(result["peers"], 2, "a stuck WHEP request never retried")

    def test_playing_stream_still_recovers_after_eight_seconds(self):
        result = self.player('''await startLive(); video.currentTime = 1; video.onplaying();
tick(2000); tick(2000); tick(9000);
console.log(JSON.stringify({peers:peers.length, events}));''')
        self.assertEqual(result["peers"], 2)

    def test_intentionally_closed_peer_cannot_schedule_another_retry(self):
        result = self.player('''await startLive(); await startLive();
console.log(JSON.stringify({scheduled:timers.filter(t => t.live).length}));''')
        self.assertEqual(result["scheduled"], 0, "retired peer scheduled a retry over its replacement")

    def test_late_whep_response_is_deleted_without_owning_new_peer(self):
        result = self.player('''holdPost = true; const old = startLive(); await flush();
holdPost = false; await startLive(); heldReply(); await old;
console.log(JSON.stringify({owned:whepUrl, deleted, peers:peers.length}));''')
        self.assertTrue(result["owned"].endswith("/2"), result)
        self.assertTrue(any(url.endswith("/1") for url in result["deleted"]), result)

    def status(self, record, action="renderLive(input);"):
        src = HTML.read_text()
        definitions = "".join(function(src, name, optional=True) for name in ["visibleBox", "renderLiveBox", "renderLive"])
        resize = re.search(r'window.addEventListener\("resize", (.*?)\);', src).group(1)
        return node(r'''
let now = 1000000, live = null, draws = [], order = [], privacyPending = 0, followPending = 0;
Date.now = () => now;
const el = {classList:{toggle(){}}, querySelector:() => ({textContent:""})};
const $ = () => el;
const document = {activeElement:null};
const petText = () => ["Resting", "here"];
const drawBox = box => {draws.push(box); order.push("draw");};
const frameOn = () => order.push("frame");
const doingSub = () => "", renderModes = () => {}, showPrivacy = () => {},
      renderFind = () => {}, showFollow = () => {}, showTalkVol = () => {};
''' + definitions + "\nconst input = " + json.dumps(record) + ";\n" +
                    "const resize = " + resize + ";\n" + action +
                    "\nconsole.log(JSON.stringify({draws,order}));")

    @staticmethod
    def record(**extra):
        return dict(status="ok", present=True, box=[.2,.2,.5,.5], t=1000,
                    det_t=976, rest_since=900, moving_since=None, frame_motion=0, **extra)

    def test_still_resting_box_survives_detector_interval(self):
        result = self.status(self.record())
        self.assertEqual(result["draws"][-1], [.2,.2,.5,.5])

    def test_resting_full_cycle_status_uses_its_own_quiet_motion(self):
        record = self.record(); del record["frame_motion"]; record["motion"] = .31
        self.assertEqual(self.status(record)["draws"][-1], [.2,.2,.5,.5])

    def test_crop_updates_before_box_coordinates_are_drawn(self):
        record = self.record(); record["det_t"] = 999
        result = self.status(record)
        self.assertEqual(result["order"][:2], ["frame","draw"])

    def test_resize_cannot_resurrect_an_expired_box(self):
        record = self.record(); record["det_t"] = 900
        result = self.status(record, "renderLive(input); resize();")
        self.assertIsNone(result["draws"][-1])

    def test_stale_active_or_unhealthy_boxes_are_hidden(self):
        for extra in [{"moving_since":990}, {"frame_motion":5}, {"status":"privacy"},
                      {"status":"watcher_down"}, {"t":1100}, {"present":False}]:
            record = self.record(); record.update(extra)
            with self.subTest(extra=extra):
                self.assertIsNone(self.status(record)["draws"][-1])


if __name__ == "__main__":
    unittest.main()
