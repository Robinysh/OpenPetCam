"""Hold-to-talk sits next to the joystick, where the thumb already is; there is no Center button.

Report 20261103-111500: the owner never used the Center button left of the joystick, and wanted
hold-to-talk there instead (it lived in the side rail over the video, away from the thumb).

Checks the layout of server/index.html, then runs the real push-to-talk code in node with a tiny
fake DOM: pressing the button by the joystick records, lights that button and labels it.
Usage: python3 tests/server/talk_button_test.py
"""
import json
import os
import re
import shutil
import subprocess
import sys

HTML = os.path.join(os.path.dirname(__file__), "..", "..", "server", "index.html")


def controls_block(src):
    m = re.search(r'<div class="controls">(.*?)\n  </div>', src, re.S)
    assert m, "controls block not found"
    return m.group(1)


def check_layout(src):
    assert 'id="go-start"' not in src, "the Center button is still there"
    assert '"go-start"' not in src, "script still wires the Center button"
    ctl = controls_block(src)
    talk, stick = ctl.find('id="talk-btn"'), ctl.find('id="stick"')
    assert talk != -1, "hold-to-talk is not in the controls row"
    assert talk < stick, "hold-to-talk should be left of the joystick, where Center was"
    tag = re.search(r'<button[^>]*id="talk-btn"[^>]*>', ctl).group(0)
    assert "talk-hold" in tag, tag


def check_behaviour(src):
    m = re.search(r"// =+ push-to-talk =+\n(.*?)\nfunction showTalkVol", src, re.S)
    assert m, "push-to-talk section not found"
    js = r"""
class El {
  constructor(id, cls) { this.id = id; this.cls = new Set(cls); this.h = {}; this.textContent = ""; this.muted = true;
    this.classList = { add: (c) => this.cls.add(c), remove: (c) => this.cls.delete(c), contains: (c) => this.cls.has(c) }; }
  addEventListener(t, f) { (this.h[t] = this.h[t] || []).push(f); }
}
const els = { "talk-btn": new El("talk-btn", ["round", "talk", "talk-hold"]),
              "talk-rail": new El("talk-rail", ["iconbtn", "talk", "talk-hold"]),
              live: new El("live", []), "move-note": new El("move-note", []) };
const labels = [new El("", ["talk-label"]), new El("", ["talk-label"])];
const document = { querySelectorAll: (s) => s === ".talk-hold" ? [els["talk-btn"], els["talk-rail"]]
                                          : s === ".talk-label" ? labels : [] };
const $ = (id) => els[id];
const navigator = { mediaDevices: { getUserMedia: async () => ({ getTracks: () => [{ stop() {} }] }) } };
let deferStops = false;
class MediaRecorder { static isTypeSupported() { return true; } constructor(stream) { this.mimeType = "audio/mp4"; this.stream = stream; }
  start() { this.state = "recording"; } stop() { this.state = "inactive"; if (!deferStops && this.onstop) this.onstop(); } }
const window = { MediaRecorder };
const intervals = []; const setInterval = (f) => { intervals.push(f); return 1; }; const clearInterval = () => {};
const PET = { name: "Biscuit" };
let privacyBlocksLive = false;
let uploads = 0; const fetch = async () => { uploads++; throw Error("unexpected voice upload"); };
""" + m.group(1) + r"""
(async () => {
  const b = els["talk-btn"];
  for (const k of ["pointerdown", "pointerup", "pointercancel"])
    if (!(b.h[k] || []).length) throw new Error("talk-btn has no " + k + " handler");
  await b.h.pointerdown[0]({ preventDefault() {}, currentTarget: b });
  intervals.forEach((f) => f());
  const during = { rec: b.classList.contains("rec"), rail: els["talk-rail"].classList.contains("rec"),
                   label: labels.map((l) => l.textContent) };
  await b.h.pointerup[0]({ preventDefault() {}, currentTarget: b });
  const after = b.classList.contains("rec");
  let tracksStopped = 0;
  const stream = { getTracks: () => [{ stop() { tracksStopped++; } }] };
  navigator.mediaDevices.getUserMedia = async () => stream;
  await talkDown({ preventDefault() {}, currentTarget: b });
  privacyBlocksLive = true; cancelTalkForPrivacy(); await talkUp();
  const cancelled = { recording: !!rec, holding, tracksStopped, uploads, lit: b.classList.contains("rec") };
  privacyBlocksLive = false;
  let allow;
  navigator.mediaDevices.getUserMedia = () => new Promise(resolve => { allow = resolve; });
  const pending = talkDown({ preventDefault() {}, currentTarget: b });
  privacyBlocksLive = true; cancelTalkForPrivacy(); allow(stream); await pending;
  const permissionRace = { recording: !!rec, holding, tracksStopped, uploads };
  privacyBlocksLive = false; navigator.mediaDevices.getUserMedia = async () => stream;
  await talkDown({ preventDefault() {}, currentTarget: b }); recStart = Date.now() - 1000;
  const stopping = talkUp(); privacyBlocksLive = true; cancelTalkForPrivacy(); await stopping;
  const stoppedUpload = uploads;
  const makeStream = () => { let stopped = 0; return { getTracks: () => [{ stop() { stopped++; } }], get stopped() { return stopped; } }; };
  // Browser stop/data events can remain queued across pause and a subsequent resume.
  privacyBlocksLive = false; deferStops = true;
  const retiredStream = makeStream(), freshStream = makeStream();
  navigator.mediaDevices.getUserMedia = async () => retiredStream;
  await talkDown({ preventDefault() {}, currentTarget: b }); recStart = Date.now() - 1000;
  const retiredRecorder = rec, retiredStopping = talkUp();
  privacyBlocksLive = true; cancelTalkForPrivacy(); privacyBlocksLive = false;
  navigator.mediaDevices.getUserMedia = async () => freshStream;
  await talkDown({ preventDefault() {}, currentTarget: b });
  const freshRecorder = rec, freshChunks = recChunks;
  if (retiredRecorder.ondataavailable) retiredRecorder.ondataavailable({ data: new Blob(["voice from before privacy"]) });
  if (retiredRecorder.onstop) retiredRecorder.onstop();
  await retiredStopping;
  const resumedStop = { uploads, sameRecorder: rec === freshRecorder, sameStream: recStream === freshStream,
                        sameChunks: recChunks === freshChunks, chunks: recChunks.length,
                        freshStopped: freshStream.stopped, recording: rec.state === "recording" };
  privacyBlocksLive = true; cancelTalkForPrivacy(); privacyBlocksLive = false;
  // A permission request retired by privacy cannot replace a newer recording after resume.
  let allowRetired;
  navigator.mediaDevices.getUserMedia = () => new Promise(resolve => { allowRetired = resolve; });
  const retiredPermission = talkDown({ preventDefault() {}, currentTarget: b });
  privacyBlocksLive = true; cancelTalkForPrivacy(); privacyBlocksLive = false;
  const currentStream = makeStream(), lateStream = makeStream();
  navigator.mediaDevices.getUserMedia = async () => currentStream;
  await talkDown({ preventDefault() {}, currentTarget: b });
  const currentRecorder = rec;
  allowRetired(lateStream); await retiredPermission;
  const resumedPermission = { uploads, sameRecorder: rec === currentRecorder, sameStream: recStream === currentStream,
                             holding, currentStopped: currentStream.stopped, retiredStopped: lateStream.stopped };
  privacyBlocksLive = true; cancelTalkForPrivacy();
  console.log(JSON.stringify({ during, after, cancelled, permissionRace, stoppedUpload, resumedStop, resumedPermission }));
})().catch((e) => { console.error(e.stack || e); process.exit(1); });
"""
    out = subprocess.run(["node", "-e", js], capture_output=True, text=True, timeout=30)
    assert out.returncode == 0, out.stderr
    r = json.loads(out.stdout)
    assert r["during"]["rec"], "the pressed button should light up while recording"
    assert not r["during"]["rail"], "only the pressed button lights up"
    assert all(l.startswith("Recording") for l in r["during"]["label"]), r
    assert not r["after"], "releasing must clear the recording state"
    assert not any(r["cancelled"][k] for k in ("recording", "holding", "lit", "uploads")), r
    assert r["cancelled"]["tracksStopped"] == 1, r
    assert not any(r["permissionRace"][k] for k in ("recording", "holding", "uploads")), r
    assert r["permissionRace"]["tracksStopped"] == 2, r
    assert r["stoppedUpload"] == 0, "privacy during onstop must discard the voice"
    assert r["resumedStop"]["uploads"] == 0, "retired voice must not upload after privacy is resumed"
    assert all(r["resumedStop"][k] for k in ("sameRecorder", "sameStream", "sameChunks", "recording")), r
    assert r["resumedStop"]["chunks"] == r["resumedStop"]["freshStopped"] == 0, r
    assert r["resumedPermission"]["uploads"] == 0, r
    assert all(r["resumedPermission"][k] for k in ("sameRecorder", "sameStream", "holding")), r
    assert r["resumedPermission"]["currentStopped"] == 0, "late permission must not stop the new recording"
    assert r["resumedPermission"]["retiredStopped"] == 1, "late permission's own stream must close"


def main():
    src = open(HTML).read()
    check_layout(src)
    if shutil.which("node"):
        check_behaviour(src)
    else:
        print("behaviour part skipped: node not installed")
    print("talk button test passed")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print("FAIL:", e)
        sys.exit(1)
