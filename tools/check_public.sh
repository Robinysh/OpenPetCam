#!/usr/bin/env bash
# Refuse to commit anything that identifies your install or looks like a secret.
#   tools/check_public.sh          check staged files (what the pre-commit hook runs)
#   tools/check_public.sh --all    check every tracked file
#   tools/check_public.sh --files a b   check these files
# Patterns: the values in config/site.env that name your server or home network, the phrases in
# config/private_terms.txt (git-ignored, one per line: things only your home has, e.g. your wall labels),
# plus common secret shapes. Room-model data (eval/, and data or media files under sim/world/) is refused
# outright: the world model of your home is yours, only the machinery is public.
# Install the hook once: git config core.hooksPath .githooks
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
if [ "${1:-}" = --all ]; then files=$(git ls-files)
elif [ "${1:-}" = --files ]; then shift; files="$*"           # explicit paths (tools/contribute.py)
else files=$(git diff --cached --name-only --diff-filter=ACM); fi
[ -n "$files" ] || exit 0
# A linked worktree (an agent's, a release's) has no git-ignored config/: read the main checkout's, or the
# check would pass with no private patterns at all (it let a pet's name through on 2026-11-07).
conf=config
if [ ! -f config/site.env ]; then
  main=$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")
  [ -f "$main/config/site.env" ] && conf="$main/config"
fi
[ -f "$conf/site.env" ] || echo "check_public: no config/site.env here: checking key shapes and paths only" >&2
pats=()
if [ -f "$conf/site.env" ]; then
  while IFS='=' read -r k v; do
    v=${v%\"}; v=${v#\"}
    case "$k" in PUBLIC_IP|PUBLIC_HOST|SSH_HOST|CAMERA_LAN_IP|CAMERA_WG_PUBKEY)
      [ -n "$v" ] && pats+=("$(printf '%s' "${v#*@}" | sed 's/[.[\*^$+?(){}|/]/\\&/g')") ;;
      PET_NAME)   # your pet's name stays out of the public repository too (whole word, any case)
      [ -n "$v" ] && [ ${#v} -gt 2 ] && pet="$v" ;;
    esac
  done < <(grep -E '^[A-Z_]+=' "$conf/site.env")
fi
pats+=('BEGIN [A-Z ]*PRIVATE KEY' 'AKIA[0-9A-Z]{16}' 'pbkdf2_sha256\$[0-9]+\$[0-9a-f]{16}' 'sk-[A-Za-z0-9]{20,}' "$HOME/")
bad=0
for f in $files; do
  case "$f" in
    eval/*) echo "check_public: refusing $f: eval/ holds your home's data"; bad=1 ;;
    sim/world/*.json|sim/world/*.npz|sim/world/*.npy|sim/world/*.f32|sim/world/*.u8|sim/world/*.jpg|sim/world/*.png|sim/world/*.pcm|sim/world/*.wav|sim/world/*.glb|sim/world/*.blend)
      echo "check_public: refusing $f: room-model data belongs in your world folder (eval/<world>/)"; bad=1 ;;
  esac
done
if [ -f "$conf/private_terms.txt" ]; then
  while IFS= read -r term; do
    term=${term%%#*}; term=$(printf '%s' "$term" | sed 's/[[:space:]]*$//')
    [ ${#term} -gt 2 ] || continue
    # shellcheck disable=SC2086
    if hits=$(grep -niIF -- "$term" $files 2>/dev/null); then
      echo "check_public: refusing, a private term ('$term') is in:"; printf '%s\n' "$hits" | cut -c1-160; bad=1
    fi
  done < "$conf/private_terms.txt"
fi
for p in "${pats[@]}"; do
  # shellcheck disable=SC2086
  if hits=$(grep -nIE -- "$p" $files 2>/dev/null); then
    [ "$p" = "$HOME/" ] && hits=$(printf '%s\n' "$hits" | grep -v '^tools/check_public.sh:' || true)
    [ -n "$hits" ] || continue
    echo "check_public: refusing, found something private:"; printf '%s\n' "$hits" | cut -c1-160; bad=1
  fi
done
if [ -n "${pet:-}" ]; then
  # shellcheck disable=SC2086
  if hits=$(grep -niwIF -- "$pet" $files 2>/dev/null); then
    echo "check_public: refusing, your pet's name is in:"; printf '%s\n' "$hits" | cut -c1-160; bad=1
  fi
fi
[ $bad -eq 0 ] || { echo "Move it to config/site.env (git-ignored), your world folder (eval/<world>/) or the server's /etc/petcam/*.env."; exit 1; }
