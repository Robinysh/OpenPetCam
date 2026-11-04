#!/usr/bin/env bash
# Deploy OpenPetCam to your server: sync server/, models/ and config/site.env, then run install.sh.
# Idempotent; the same command sets up a new server and updates an existing one.
#   ./deploy.sh                 uses config/site.env
#   PETCAM_CONFIG=x.env ./deploy.sh
set -euo pipefail
cd "$(dirname "$0")"
# Agent workers propose; only the harness ships (tools/report_agent.py:ship runs this without the flag).
if [ "${PETCAM_AGENT_WORKER:-}" = 1 ]; then echo "Refusing: agent workers may not deploy; the harness ships after its gates."; exit 1; fi
# One deploy at a time (the owner's session and the agent's harness both deploy).
mkdir -p eval; LOCK=eval/.deploy.lock
if ! mkdir "$LOCK" 2>/dev/null; then
  if [ -n "$(find "$LOCK" -maxdepth 0 -mmin +30 2>/dev/null)" ]; then rmdir "$LOCK"; mkdir "$LOCK"; else echo "Another deploy is running; try again shortly."; exit 75; fi
fi
trap 'rmdir "$LOCK" 2>/dev/null' EXIT
CONF=${PETCAM_CONFIG:-config/site.env}
[ -f "$CONF" ] || { echo "No $CONF. Copy config/site.env.example to config/site.env and fill it in."; exit 1; }
set -a; . "$CONF"; set +a
KEY=${SSH_KEY/#\~/$HOME}
SSH=(ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=20)
DEST=petcam-src   # in the SSH user's home on the server

tools/fetch_models.sh
# Never downgrade a trained verifier to the untrained default (2026-11-02 17:43: a broken models/ folder was
# refilled with the default and shipped, and the toy passed as the pet for 26 min). PETCAM_ALLOW_DEFAULT=1 overrides.
mode() { python3 -c 'import json,sys; print(json.load(sys.stdin).get("mode",""))' 2>/dev/null; }
LOCAL_MODE=$(mode < models/verifier.json || true)
LIVE_MODE=$("${SSH[@]}" "$SSH_HOST" "cat /opt/petcam/models/verifier.json 2>/dev/null" | mode || true)
trained() { [ "$1" = logreg ] || [ "$1" = finetuned ]; }
if trained "$LIVE_MODE" && ! trained "$LOCAL_MODE" && [ "${PETCAM_ALLOW_DEFAULT:-}" != 1 ]; then
  echo "Refusing: the server runs a trained verifier but models/verifier.json is '$LOCAL_MODE'. Restore it (eval/bootstrap/verifier.json) or set PETCAM_ALLOW_DEFAULT=1."
  exit 1
fi
"${SSH[@]}" "$SSH_HOST" "mkdir -p -m 700 ~/$DEST/models"
rsync -az --delete --exclude __pycache__ -e "${SSH[*]}" server/ "$SSH_HOST:$DEST/" --exclude models
rsync -az --delete -e "${SSH[*]}" models/ "$SSH_HOST:$DEST/models/"   # --delete: a model removed here must not come back
rsync -az -e "${SSH[*]}" "$CONF" "$SSH_HOST:$DEST/site.env"
# The version being installed: a release tag exactly (vX.Y.Z), or a git description such as v1.0.0-3-gabc1234
# or -dirty for anything else. Only an exact release is updated automatically from GitHub (server/update.py).
VERSION=$(git describe --tags --always --dirty 2>/dev/null || echo unknown)
"${SSH[@]}" "$SSH_HOST" "echo '$VERSION' > ~/$DEST/VERSION"
"${SSH[@]}" "$SSH_HOST" "sudo bash ~/$DEST/install.sh"
echo "Deployed. Open https://$PUBLIC_HOST"
