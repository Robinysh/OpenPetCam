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
    """1-24 characters of letters (any language) or digits, spaces, apostrophes, hyphens and dots."""
    name = " ".join(str(name or "").split())
    return name if 0 < len(name) <= 24 and all(c.isalnum() or c in " '-.’" for c in name) else None


def save_pet(name, pronoun):
    name = valid_name(name)
    if not name or pronoun not in _PRONOUNS:
        raise ValueError("A name of up to 24 letters or digits, and she, he or they.")
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

# Picture size the watcher works at (16:9). The defaults are the reference camera's 640x360 substream and a detector
# input 416 wide (256x416 once padded to 16:9). A bigger frame lets it see a small or distant pet: the detector is
# blind to anything under roughly 30 px at its input size. FRAME_WIDTH must be a multiple of 160 (the motion check
# samples every FRAME_WIDTH/160 th pixel to a fixed 160x90); DETECT_WIDTH a multiple of 32.
def _size(name, default, multiple):
    try:
        v = int(SITE.get(name, "") or default)
    except ValueError:
        return default
    return v if v >= multiple and v % multiple == 0 else default


FRAME_W = _size("FRAME_WIDTH", 640, 160)
FRAME_H = FRAME_W * 9 // 16
DETECT_W = _size("DETECT_WIDTH", 416, 32)

# The object detector that proposes boxes: yolox (YOLOX-nano, Apache-2.0, the default) or yolo26 (YOLO26-small via
# tools/make_yolo26.py, AGPL-3.0: check that licence suits you). Without its model file the watcher uses YOLOX.
DETECTOR = SITE.get("DETECTOR", "yolox").strip().lower()
if DETECTOR not in ("yolox", "yolo26"):
    DETECTOR = "yolox"

# What the pet is: dog or cat. Picks the verifier's ImageNet classes (dog breeds 151-268, domestic cats 281-285) in
# models/verifier.json (tools/fetch_models.sh sets it up from verifier.default.json / verifier.cat.json).
PET_KIND = SITE.get("PET_KIND", "dog").strip().lower()
if PET_KIND not in ("dog", "cat"):
    PET_KIND = "dog"

# Which camera this install has, and which of its features exist. Default: the reference Thingino pan/tilt camera.
#   CAMERA=thingino|rpi   rpi = a Raspberry Pi camera on this machine, read by MediaMTX (no tunnel, no camera CGIs)
#   PTZ=1|0               motors. Off by default for rpi: every move, Find and Follow is refused and hidden.
#   AUDIO=1|0             microphone in the stream. Off by default for rpi (the camera module has none).
CAMERA = SITE.get("CAMERA", "thingino").strip().lower()
THINGINO = CAMERA != "rpi"


def _flag(name, default):
    v = SITE.get(name, "").strip().lower()
    return default if v == "" else v in ("1", "yes", "true", "on")


PTZ = _flag("PTZ", THINGINO)
AUDIO = _flag("AUDIO", THINGINO)


def caps():
    """What this install's camera can do; the web app hides what it can't."""
    return {"ptz": PTZ, "audio": AUDIO, "imaging": THINGINO, "talk": THINGINO and AUDIO}


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
                              "pronoun": PRONOUN, "named": NAMED, "caps": caps()})
        .replace("<", "\\u003c"),
    }
