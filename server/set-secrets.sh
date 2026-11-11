#!/usr/bin/env bash
# Run on the petcam server as root, interactively. Asks for:
#   1. the password you want for the web app (stored only as a PBKDF2 hash)
#   2. the camera's root web password (so the app can move the motors)
#   3. optionally, the camera's new RTSP username/password
# Nothing is echoed or logged.
set -euo pipefail
. /etc/petcam/site.env
ENV=/etc/petcam/app.env
CAM=/etc/petcam/camera.env

read -rsp "New web app password: " P1; echo
read -rsp "Repeat it: " P2; echo
[ "$P1" = "$P2" ] || { echo "Passwords differ, nothing changed."; exit 1; }
[ ${#P1} -ge 8 ] || { echo "Use at least 8 characters, nothing changed."; exit 1; }
HASH=$(P="$P1" python3 -c 'import os,hashlib;s=os.urandom(16);i=600000;print("pbkdf2_sha256$%d$%s$%s"%(i,s.hex(),hashlib.pbkdf2_hmac("sha256",os.environ["P"].encode(),s,i).hex()))')

CP=
if [ "${CAMERA:-thingino}" != rpi ]; then
  read -rsp "Camera root password (the one you set in Thingino setup): " CP; echo
  read -rp "Changed the camera's RTSP login from thingino/thingino? [y/N] " YN
else
  YN=n   # a Pi camera on this machine has no password and no RTSP login
fi
if [ "${YN:-n}" = y ] || [ "${YN:-n}" = Y ]; then
  read -rp "  RTSP username: " RU
  read -rsp "  RTSP password: " RP; echo
  ENC=$(U="$RU" P="$RP" python3 -c 'import os,urllib.parse as q;print(q.quote(os.environ["U"],safe="")+":"+q.quote(os.environ["P"],safe=""))')
  printf 'MTX_PATHS_CAM_SOURCE=rtsp://%s@%s:554/ch0\nMTX_PATHS_SUB_SOURCE=rtsp://%s@%s:554/ch1\n' \
    "$ENC" "$CAMERA_WG_IP" "$ENC" "$CAMERA_WG_IP" > "$CAM"
  chown root:mediamtx "$CAM"; chmod 640 "$CAM"
fi

TMP=$(mktemp)
grep -vE '^(APP_PASSWORD_HASH|CAM_WEB_PASS)=' "$ENV" > "$TMP" || true
printf 'APP_PASSWORD_HASH=%s\n' "$HASH" >> "$TMP"
[ -z "$CP" ] || printf 'CAM_WEB_PASS=%s\n' "$CP" >> "$TMP"
install -m 640 -o root -g petcam "$TMP" "$ENV"; rm -f "$TMP"

systemctl restart petcam-app mediamtx
echo "Saved. Web app and video restarted."
