#!/usr/bin/env bash
# A quiet, seamless orbit from the already-published weekly observation scene.
# Only the viewpoint changes: this does not run a sighting history backwards.
set -euo pipefail
cd "$(dirname "$0")/../.."
source_film=site/media/blog/film/week-observations.mp4
python3 - "$source_film" <<'PY'
import hashlib
from pathlib import Path
import sys
expected = 'afd2640e186f8d89d40ba5082c1640af3e70a969ce11830a4dc7b8f026677396'
if hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest() != expected:
    raise SystemExit('Not the approved published orbit; refusing to export.')
PY
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
ffmpeg -hide_banner -loglevel error -i "$source_film" \
  -filter_complex '[0:v]setpts=1.5*(PTS-STARTPTS),fps=24,split[f][r];[r]reverse,setpts=PTS-STARTPTS[b];[f][b]concat=n=2:v=1:a=0[v]' \
  -map '[v]' -an -c:v libx264 -crf 22 -preset medium -threads 2 \
  -pix_fmt yuv420p -movflags +faststart -map_metadata -1 "$scratch/room-title.mp4"
mv "$scratch/room-title.mp4" site/media/blog/room-title.mp4
