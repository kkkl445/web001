#!/usr/bin/env bash
# Full pipeline: fonts -> score -> picture -> mux (HD master + transfer-size copy).
# Shares the compositing engine with ../quantum-video/engine.py.
set -euo pipefail
cd "$(dirname "$0")"
[ -n "${SKIP_PREP:-}" ] || ./fetch_fonts.sh
[ -n "${SKIP_PREP:-}" ] || python3 music.py
python3 render.py --workers "${WORKERS:-4}"
# master: the render's own stream, untouched, with the score
ffmpeg -y -loglevel error -i build/video.mp4 -i build/bgm.wav -map 0:v -map 1:a -c:v copy \
  -c:a aac -b:a 320k -ar 48000 -movflags +faststart -shortest attention-master.mp4
# transfer copy for chat
ffmpeg -y -loglevel error -i build/video.mp4 -i build/bgm.wav -map 0:v -map 1:a \
  -vf "hqdn3d=1.5:1.5:4:4,format=yuv420p" -c:v libx264 -preset slow -crf 24 -tune film \
  -x264-params aq-mode=3 -profile:v high -level 4.1 \
  -c:a aac -b:a 192k -ar 48000 -movflags +faststart -shortest attention.mp4
echo "done: attention-master.mp4, attention.mp4"
