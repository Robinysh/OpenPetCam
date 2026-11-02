#!/usr/bin/env bash
# Deploy OpenPetCam to your server: sync server/, models/ and config/site.env, then run install.sh.
# Idempotent; the same command sets up a new server and updates an existing one.
#   ./deploy.sh                 uses config/site.env
#   PETCAM_CONFIG=x.env ./deploy.sh
set -euo pipefail
cd "$(dirname "$0")"
CONF=${PETCAM_CONFIG:-config/site.env}
[ -f "$CONF" ] || { echo "No $CONF. Copy config/site.env.example to config/site.env and fill it in."; exit 1; }
set -a; . "$CONF"; set +a
KEY=${SSH_KEY/#\~/$HOME}
SSH=(ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=20)
DEST=petcam-src   # in the SSH user's home on the server

tools/fetch_models.sh
"${SSH[@]}" "$SSH_HOST" "mkdir -p -m 700 ~/$DEST/models"
rsync -az --delete --exclude __pycache__ -e "${SSH[*]}" server/ "$SSH_HOST:$DEST/" --exclude models
rsync -az -e "${SSH[*]}" models/ "$SSH_HOST:$DEST/models/"
rsync -az -e "${SSH[*]}" "$CONF" "$SSH_HOST:$DEST/site.env"
"${SSH[@]}" "$SSH_HOST" "sudo bash ~/$DEST/install.sh"
echo "Deployed. Open https://$PUBLIC_HOST"
