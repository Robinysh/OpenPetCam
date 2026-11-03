#!/usr/bin/env bash
# Build the server release archive that server/update.py installs:
#   dist/openpetcam-<X.Y.Z>.tar.gz   the contents of server/ plus VERSION and LICENSE (no settings, no models)
#   dist/SHA256SUMS
# Usage: tools/make_release.sh vX.Y.Z   (the GitHub release workflow runs this on every version tag)
set -euo pipefail
cd "$(dirname "$0")/.."
TAG=${1:?usage: tools/make_release.sh vX.Y.Z}
[[ $TAG =~ ^v[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "tag must look like v1.2.3"; exit 1; }
VER=${TAG#v}
grep -q "^## \[$VER\]" CHANGELOG.md || { echo "CHANGELOG.md has no '## [$VER]' section"; exit 1; }
rm -rf dist && mkdir -p dist/stage
git archive HEAD server | tar -x -C dist/stage
mv dist/stage/server dist/stage/openpetcam && cp LICENSE dist/stage/openpetcam/
echo "$TAG" > dist/stage/openpetcam/VERSION
# Reproducible: fixed owner, order and timestamps.
( cd dist/stage/openpetcam && find . -type f | LC_ALL=C sort | tar --owner=0 --group=0 --numeric-owner \
    --mtime='2000-01-01 00:00:00Z' -czf "../../openpetcam-$VER.tar.gz" -T - 2>/dev/null \
  || find . -type f | LC_ALL=C sort | tar -czf "../../openpetcam-$VER.tar.gz" -T - )
rm -rf dist/stage
( cd dist && shasum -a 256 "openpetcam-$VER.tar.gz" > SHA256SUMS )
# The section of the changelog for this version becomes the release notes.
awk -v v="## [$VER]" 'index($0,v)==1{on=1;next} on&&/^## \[/{exit} on' CHANGELOG.md > dist/NOTES.md
cat dist/SHA256SUMS
