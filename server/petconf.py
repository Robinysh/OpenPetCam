"""Per-install settings that are not secret: the pet's name, pronouns and network addresses.

install.sh writes them to /etc/petcam/site.env from your config/site.env. Secrets (passwords,
session key, camera login) live separately in /etc/petcam/app.env, readable only by the app.
"""
import json
import os

SITE_ENV = os.environ.get("PETCAM_SITE_ENV", "/etc/petcam/site.env")
STATE = "/var/lib/petcam"     # all runtime data: database, snapshots, clips, settings, reports
DOG_SOUND_KINDS = ("barking", "howling", "whining", "growling", "dog_sound")   # event kinds that are the dog


def load_env(path, unquote=True):
    env = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    v = v.strip()
                    env[k.strip()] = v.strip('"').strip("'") if unquote else v
    except (FileNotFoundError, PermissionError):
        pass
    return env


SITE = load_env(SITE_ENV)

PET = SITE.get("PET_NAME") or "Your pet"
# subject, object, possessive: "she is moving", "keep her in view", "her water bowl"
_PRONOUNS = {"she": ("she", "her", "her"), "he": ("he", "him", "his"), "they": ("they", "them", "their")}
SUBJ, OBJ, POSS = _PRONOUNS.get(SITE.get("PET_PRONOUN", "they").lower(), _PRONOUNS["they"])
APP_NAME = SITE.get("APP_NAME") or f"{PET} cam"

CAM_HOST = SITE.get("CAMERA_WG_IP", "10.77.0.2")      # camera's address inside the WireGuard tunnel
SERVER_WG_IP = SITE.get("SERVER_WG_IP", "10.77.0.1")  # this server's address inside the tunnel
PUBLIC_HOST = SITE.get("PUBLIC_HOST", "localhost")


def html_escape(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def template_vars():
    """Placeholders the web pages use: {{PET}} etc. in HTML, {{PET_JS}} as a JS object."""
    return {
        "PET": html_escape(PET),
        "APP_NAME": html_escape(APP_NAME),
        "PET_POSS_CAP": html_escape(POSS.capitalize()),
        "PET_POSS": html_escape(POSS),
        "PET_JS": json.dumps({"name": PET, "subj": SUBJ, "obj": OBJ, "poss": POSS, "app": APP_NAME})
        .replace("<", "\\u003c"),
    }
