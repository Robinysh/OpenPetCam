#!/usr/bin/env python3
"""The moment-clip player never fails silently: a clip that can't play is shown in the player and reported
to /api/clientlog with the MediaError code, play() is started from the tap, and closing the player clears
the handlers so a closed player reports nothing. Static checks on index.html. Runs anywhere."""
import os, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRV = os.path.join(HERE, "..", "..", "server")
fails = 0


def check(name, cond):
    global fails
    print(("PASS " if cond else "FAIL ") + name)
    fails += 0 if cond else 1


s = open(f"{SRV}/index.html").read()
m = re.search(r"function openPlayer\(src, title\) \{(.*?)\n\}\n", s, re.S)
body = m.group(1) if m else ""
check("openPlayer exists", bool(m))
check("a playback error is handled", "onerror" in body or "addEventListener(\"error\"" in body)
check("the error is reported to the server with the MediaError code", "reportError(" in body and "v.error" in body and "e.code" in body)
check("the player tells the owner the clip can't play", "can't play" in body)
check("play() is called from the tap (iOS autoplay rules)", ".play()" in body)
check("a cancelled play (closing the player) is not reported", "AbortError" in body)
c = re.search(r'\$\("player-close"\)\.onclick = \(\) => \{(.*?)\n\};', s, re.S)
check("closing clears the error handler before dropping the source", bool(c) and "onerror = null" in c.group(1)
      and c.group(1).index("onerror = null") < c.group(1).index('removeAttribute("src")'))
print("\nall passed" if not fails else f"\n{fails} failed")
sys.exit(1 if fails else 0)
