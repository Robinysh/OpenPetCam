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

# subject, object, possessive: "she is moving", "keep her in view", "her water bowl"
_PRONOUNS = {"she": ("she", "her", "her"), "he": ("he", "him", "his"), "they": ("they", "them", "their")}
PET_FILE = os.environ.get("PETCAM_PET_FILE", f"{STATE}/pet.json")   # set from the app (Settings or first open)
PET, PRONOUN, SUBJ, OBJ, POSS, NAMED = "Your pet", "they", "they", "them", "their", False
APP_NAME = SITE.get("APP_NAME") or "OpenPetCam"
_pet_mtime = None


def refresh():
    """The pet's name and pronoun: the app's setting (pet.json) wins over site.env's PET_NAME / PET_PRONOUN.
    Cheap to call often: re-reads only when pet.json changes, so every process follows a rename without a restart."""
    global PET, PRONOUN, SUBJ, OBJ, POSS, NAMED, _pet_mtime
    try:
        m = os.path.getmtime(PET_FILE)
    except OSError:
        m = None
    if m == _pet_mtime and _pet_mtime is not None:
        return
    _pet_mtime = m
    saved = {}
    if m is not None:
        try:
            with open(PET_FILE) as f:
                saved = json.load(f)
        except (OSError, ValueError):
            saved = {}
    name = (saved.get("name") or SITE.get("PET_NAME") or "").strip()
    PET, NAMED = (name, True) if name else ("Your pet", False)
    PRONOUN = (saved.get("pronoun") or SITE.get("PET_PRONOUN") or "they").lower()
    SUBJ, OBJ, POSS = _PRONOUNS.get(PRONOUN, _PRONOUNS["they"])


def valid_name(name):
    """1-24 characters of letters (any language), spaces, apostrophes, hyphens and dots."""
    name = " ".join(str(name or "").split())
    return name if 0 < len(name) <= 24 and all(c.isalpha() or c in " '-.’" for c in name) else None


def save_pet(name, pronoun):
    name = valid_name(name)
    if not name or pronoun not in _PRONOUNS:
        raise ValueError("A name of up to 24 letters, and she, he or they.")
    tmp = PET_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump({"name": name, "pronoun": pronoun}, f)
    os.chmod(tmp, 0o644)
    os.replace(tmp, PET_FILE)
    refresh()
    return {"name": PET, "pronoun": PRONOUN}


refresh()

CAM_HOST = SITE.get("CAMERA_WG_IP", "10.77.0.2")      # camera's address inside the WireGuard tunnel
SERVER_WG_IP = SITE.get("SERVER_WG_IP", "10.77.0.1")  # this server's address inside the tunnel
PUBLIC_HOST = SITE.get("PUBLIC_HOST", "localhost")


def html_escape(s):
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def template_vars():
    """Placeholders the web pages use: {{PET}} etc. in HTML, {{PET_JS}} as a JS object."""
    refresh()
    return {
        "PET": html_escape(PET),
        "PET_UPPER": html_escape(PET.upper()),
        "APP_NAME": html_escape(APP_NAME),
        "PET_POSS_CAP": html_escape(POSS.capitalize()),
        "PET_POSS": html_escape(POSS),
        "PET_JS": json.dumps({"name": PET, "subj": SUBJ, "obj": OBJ, "poss": POSS, "app": APP_NAME,
                              "pronoun": PRONOUN, "named": NAMED})
        .replace("<", "\\u003c"),
    }
