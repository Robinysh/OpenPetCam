"""Laptop-side settings for the tools: reads ../config/site.env (git-ignored)."""
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, os.path.join(ROOT, "server"))
from petconf import load_env  # noqa: E402

PATH = os.environ.get("PETCAM_CONFIG", os.path.join(ROOT, "config", "site.env"))
if not os.path.exists(PATH):
    sys.exit(f"missing {PATH}: copy config/site.env.example to config/site.env and fill it in")
SITE = load_env(PATH)
PET = SITE.get("PET_NAME", "the pet")
HOST = SITE["SSH_HOST"]
KEY = os.path.expanduser(SITE.get("SSH_KEY", "~/.ssh/id_ed25519"))
# One shared, persistent connection (OpenSSH multiplexing). 2026-11-03: the agent and tools opened ~3,700 SSH logins a
# day, each a key exchange, PAM and a systemd session on a server that lives at its 5% CPU baseline.
_CM = os.path.join(os.path.expanduser("~/.ssh"), "cm-petcam-%C")
SSH = ["ssh", "-i", KEY, "-o", "BatchMode=yes", "-o", "ControlMaster=auto", "-o", f"ControlPath={_CM}",
       "-o", "ControlPersist=30m", "-o", "ServerAliveInterval=30", HOST]
