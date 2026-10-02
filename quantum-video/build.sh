#!/usr/bin/env bash
# Full pipeline: fonts -> physics cache -> score -> picture -> mux.
set -euo pipefail
cd "$(dirname "$0")"
./fetch_fonts.sh
python3 physics.py
python3 music.py
python3 render.py --workers "${WORKERS:-4}"
ffmpeg -y -loglevel error -i build/video.mp4 -i build/bgm.wav -map 0:v -map 1:a \
  -c:v copy -c:a aac -b:a 256k -movflags +faststart -shortest quantum-mechanics.mp4
echo "done: quantum-mechanics.mp4"
