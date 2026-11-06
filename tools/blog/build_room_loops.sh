#!/usr/bin/env bash
# Recut the already-published film into silent, caption-free article illustrations.
set -euo pipefail
cd "$(dirname "$0")/../.."
scratch=$(mktemp -d)
trap 'rm -rf "$scratch"' EXIT
source_film=site/media/openpetcam-4d-room.mp4
destination=site/media/blog

make_loop() {
  local name=$1 start=$2 duration=$3 speed=$4
  ffmpeg -hide_banner -loglevel error -ss "$start" -t "$duration" -i "$source_film" \
    -filter_complex "[0:v]crop=1080:960:0:280,scale=720:640,setpts=${speed}*(PTS-STARTPTS),fps=24,split[a][b];[b]reverse[r];[a][r]concat=n=2:v=1:a=0[v]" \
    -map '[v]' -an -c:v libx264 -crf 23 -preset medium -threads 2 \
    -pix_fmt yuv420p -movflags +faststart -map_metadata -1 "$scratch/$name.mp4"
  ffmpeg -hide_banner -loglevel error -ss 1 -i "$scratch/$name.mp4" \
    -map 0:v:0 -frames:v 1 -q:v 3 "$scratch/$name.jpg"
}

# Back-and-forth motion closes each loop without a cut. No acoustic overlays survive.
make_loop room-orbit 44.9 1.5 2
make_loop time-layer 47 3.5 1
mkdir -p "$destination"
for name in room-orbit time-layer; do
  mv "$scratch/$name.mp4" "$scratch/$name.jpg" "$destination/"
done
