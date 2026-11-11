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
DEST=petcam-src   # in the SSH user's home on the server
# SSH_HOST=local: this machine IS the server (everything on one Raspberry Pi). No SSH, copy with rsync, then sudo.
if [ "$SSH_HOST" = local ]; then
  command -v rsync >/dev/null || { echo "Install rsync first: sudo apt install rsync"; exit 1; }
  remote() { bash -c "$1"; }                       # run a command on "the server"
  push() { rsync -a "${@:1:$#-2}" "${@: -2:1}" "$HOME/${@: -1}"; }   # push [rsync options] SRC DEST
else
  KEY=${SSH_KEY/#\~/$HOME}
  SSH=(ssh -i "$KEY" -o BatchMode=yes -o ConnectTimeout=20)
  remote() { "${SSH[@]}" "$SSH_HOST" "$1"; }
  push() { rsync -az -e "${SSH[*]}" "${@:1:$#-2}" "${@: -2:1}" "$SSH_HOST:${@: -1}"; }
fi

tools/fetch_models.sh
# Never downgrade a trained verifier to the untrained default (2026-11-02 17:43: a broken models/ folder was
# refilled with the default and shipped, and the toy passed as the pet for 26 min). PETCAM_ALLOW_DEFAULT=1 overrides.
mode() { python3 -c 'import json,sys; print(json.load(sys.stdin).get("mode",""))' 2>/dev/null; }
LOCAL_MODE=$(mode < models/verifier.json || true)
LIVE_MODE=$(remote "cat /opt/petcam/models/verifier.json 2>/dev/null" | mode || true)
trained() { [ "$1" = logreg ] || [ "$1" = finetuned ]; }
if trained "$LIVE_MODE" && ! trained "$LOCAL_MODE" && [ "${PETCAM_ALLOW_DEFAULT:-}" != 1 ]; then
  echo "Refusing: the server runs a trained verifier but models/verifier.json is '$LOCAL_MODE'. Restore it (eval/bootstrap/verifier.json) or set PETCAM_ALLOW_DEFAULT=1."
  exit 1
fi
remote "mkdir -p -m 700 ~/$DEST/models"
push --delete --exclude __pycache__ --exclude models server/ "$DEST/"
push --delete models/ "$DEST/models/"   # --delete: a model removed here must not come back
push "$CONF" "$DEST/site.env"
# The version being installed: a release tag exactly (vX.Y.Z), or a git description such as v1.0.0-3-gabc1234
# or -dirty for anything else. Only an exact release is updated automatically from GitHub (server/update.py).
VERSION=$(git describe --tags --always --dirty 2>/dev/null || echo unknown)
remote "echo '$VERSION' > ~/$DEST/VERSION"
remote "sudo bash ~/$DEST/install.sh"
echo "Deployed. Open https://$PUBLIC_HOST"
