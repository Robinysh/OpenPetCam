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
SSH = ["ssh", "-i", KEY, "-o", "BatchMode=yes", HOST]
