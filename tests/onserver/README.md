# Checks that need a running install

These run on the server (or against real models and recordings), not in CI. The offline gate is
`tools/run_tests.sh` (tests/server, tests/sim).

| Script | Needs | What it checks |
|---|---|---|
| `e2e.py` | the app running on the server | the app's API end to end with a throwaway device |
| `authleak.py` | the app running | auth does not carry over between requests on one keep-alive connection |
| `privacy_test.py` | the camera | privacy mode turns the picture off and back on |
| `highlights_test.py` | the rolling recording | cutting and stitching moment clips |
| `sound_model_test.py` | saved clip audio in /tmp/snd | the bark/sound model on real audio |
| `detector_test.py` | the models and labelled frames: `detector_test.py <frames dir> <images dir>` | detector + verifier on hand-labelled frames |
| `heal_test.py` | nothing (prints a simulated outage timeline) | the streamer self-heal schedule, for reading |
| `profile_latency.py` | the server, as the petcam user | stage-by-stage latency profile |

Run them as root or the petcam user on the server, e.g. `sudo /opt/petcam/venv/bin/python tests/onserver/e2e.py`.
They import from /opt/petcam.

**Do not start a second camera push while the watcher runs** (a probe that calls `FrameSource().push_cmd()`): the
camera keeps one push loop at a time (its pid file), so a second one stops the watcher's, and the watcher falls back
to snapshots until its 10 s stall timer restarts the push (seen 2026-11-04 during testing).
