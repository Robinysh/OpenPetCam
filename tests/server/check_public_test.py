"""tools/check_public.sh in a linked worktree reads the main checkout's git-ignored config/, so a pet's name or a
private term is refused there too (an agent worktree let a name through on 2026-11-07). Uses a throwaway repo.
  python3 tests/server/check_public_test.py
"""
import os
import shutil
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "..", "..", "tools", "check_public.sh")


def git(cwd, *a):
    subprocess.run(["git", *a], cwd=cwd, check=True, capture_output=True)


def check(cwd, f):
    return subprocess.run(["bash", "tools/check_public.sh", "--files", f], cwd=cwd, capture_output=True, text=True)


d = tempfile.mkdtemp()
try:
    main = os.path.join(d, "main")
    os.makedirs(os.path.join(main, "tools"))
    os.makedirs(os.path.join(main, "config"))
    shutil.copy(SCRIPT, os.path.join(main, "tools", "check_public.sh"))
    with open(os.path.join(main, ".gitignore"), "w") as f:
        f.write("config/\n")
    with open(os.path.join(main, "config", "site.env"), "w") as f:
        f.write("PET_NAME=Biscuitzz\n")
    with open(os.path.join(main, "config", "private_terms.txt"), "w") as f:
        f.write("blue sofa corner\n")
    git(main, "init", "-q")
    git(main, "-c", "user.name=t", "-c", "user.email=t@t", "add", "-A")
    git(main, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "init")
    wt = os.path.join(d, "wt")
    git(main, "worktree", "add", "-q", wt)
    assert not os.path.exists(os.path.join(wt, "config", "site.env"))
    for text, why in (("// Mark Biscuitzz here\n", "pet name"), ("she sleeps in the Blue Sofa Corner\n", "private term")):
        with open(os.path.join(wt, "a.js"), "w") as f:
            f.write(text)
        r = check(wt, "a.js")
        assert r.returncode == 1, (why, r.stdout, r.stderr)
    with open(os.path.join(wt, "a.js"), "w") as f:
        f.write("// Mark the pet\n")
    r = check(wt, "a.js")
    assert r.returncode == 0, (r.stdout, r.stderr)
    print("worktree uses the main checkout's private config: name and term refused, clean file passes")
finally:
    shutil.rmtree(d, ignore_errors=True)
print("check_public ok")
