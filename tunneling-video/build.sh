#!/usr/bin/env bash
# Full pipeline: fonts -> physics cache -> score -> picture -> mux.
# Shares the compositing engine with ../quantum-video/engine.py.
set -euo pipefail
cd "$(dirname "$0")"
./fetch_fonts.sh
python3 physics.py
python3 music.py
python3 render.py --workers "${WORKERS:-4}"
ffmpeg -y -loglevel error -i build/video.mp4 -i build/bgm.wav -map 0:v -map 1:a \
  -vf "hqdn3d=1.5:1.5:4:4,format=yuv420p" -c:v libx264 -preset slow -crf 21 -tune film \
  -x264-params aq-mode=3 -profile:v high -level 4.1 \
  -c:a aac -b:a 256k -ar 48000 -movflags +faststart -shortest quantum-tunneling.mp4
echo "done: quantum-tunneling.mp4"
