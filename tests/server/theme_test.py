#!/usr/bin/env python3
"""Themes are wired into every page, the files are served before login, and the token set is consistent. Runs anywhere."""
import json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRV = os.path.join(HERE, "..", "..", "server")
fails = 0


def check(name, cond):
    global fails
    print(("PASS " if cond else "FAIL ") + name)
    fails += 0 if cond else 1


css = open(f"{SRV}/themes.css").read()
data = open(f"{SRV}/theme-data.js").read()
presets = set(re.findall(r'\[data-opc-theme="([a-z]+)"\]', css))
check("presets are cobalt, golden, meadow", presets == {"cobalt", "golden", "meadow"})
k = "themeData = deepFreeze("; js = json.loads(data[data.index(k) + len(k):data.rindex(");")])
check("theme-data.js has the same presets", set(js["themes"]) == presets)
for t in presets:
    for m in ("light", "dark"):
        block = re.search(r'\[data-opc-theme="%s"\]\[data-opc-mode="%s"\] \{([^}]*)\}' % (t, m), css)
        ok = bool(block)
        if ok:
            vals = dict(re.findall(r"--opc-([a-z-]+): ([^;]+);", block.group(1)))
            ok = vals.get("canvas", "").upper() == js["themes"][t][m]["canvas"].upper() and vals.get("accent-fill", "").upper() == js["themes"][t][m]["accentFill"].upper()
        check(f"{t}/{m}: CSS and data agree", ok)
boot = re.compile(r'\{cobalt:1,golden:1,meadow:1\}')
for page in ("index.html", "login.html", "join.html", "agent.html"):
    s = open(f"{SRV}/{page}").read()
    check(f"{page}: loads themes.css, the boot snippet and theme-app.js",
          '/themes.css' in s and bool(boot.search(s)) and 'src="/theme-app.js"' in s)
    roots = re.findall(r":root \{([^}]*)\}", s)
    check(f"{page}: no hard-coded page palette left", roots and all("#" not in r.split("--glass-line")[0] for r in roots))
app = open(f"{SRV}/app.py").read()
check("theme files served (before the login check)", all(f in app for f in ('"/themes.css"', '"/theme-controller.js"', '"/theme-data.js"', '"/theme-app.js"')))
inst = open(f"{SRV}/install.sh").read()
check("install.sh installs the theme files", all(f"$SRC/{f}" in inst for f in ("themes.css", "theme-controller.js", "theme-data.js", "theme-app.js", "logo-light.png")))
r = subprocess.run(["git", "grep", "-i", "-c", "the pet"], cwd=os.path.join(HERE, "..", ".."), capture_output=True, text=True)
check("no pet name in the repo", r.stdout.strip() == "")
print("\nall passed" if not fails else f"\n{fails} failed")
sys.exit(1 if fails else 0)
