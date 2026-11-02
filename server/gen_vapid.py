"""Generate the web-push (VAPID) key pair once. Run by install.sh with the app's venv python."""
import base64
import os
import sys

from cryptography.hazmat.primitives import serialization
from py_vapid import Vapid01

out = sys.argv[1] if len(sys.argv) > 1 else "/etc/petcam"
priv, pub = os.path.join(out, "vapid_private.pem"), os.path.join(out, "vapid_public.txt")
if not os.path.exists(priv):
    v = Vapid01()
    v.generate_keys()
    with open(priv, "wb") as f:
        f.write(v.private_pem())
    raw = v.public_key.public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    with open(pub, "w") as f:
        f.write(base64.urlsafe_b64encode(raw).decode().rstrip("="))
    print("generated VAPID keys")
