"""Render the picture track.

  python3 render.py --preview 5,20,50 --out DIR   # stills (seconds) as PNG
  python3 render.py --workers 4                   # full video -> build/video.mp4
"""

import argparse
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path

import cv2

import scenes as S
from style import ROOT, H, W, finalize, make_background, make_grain, twinkle
from timeline import DURATION, FPS, SCENES, START

BUILD = ROOT / "build"
_state = {}


def _init():
    if not _state:
        bg, vig, tw = make_background()
        _state.update(bg=bg, vig=vig, tw=tw, grain=make_grain())
        import physics
        physics.packet2d()


def frame(fi):
    _init()
    g = fi / FPS
    name, dur = SCENES[-1]
    for n, d in SCENES:
        if START[n] <= g < START[n] + d:
            name, dur = n, d
            break
    F = _state["bg"].copy()
    twinkle(F, _state["tw"], g)
    S.SCENE_FUNCS[name](F, g - START[name], g, fi)
    fade = min(1.0, g / 1.0) * min(1.0, (DURATION - g) / 1.8)
    if fade < 1:
        F *= max(fade, 0.0)
    return finalize(F, _state["vig"], _state["grain"][fi % len(_state["grain"])] * fade)


def _chunk(args):
    f0, f1, path = args
    _init()
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", "17", "-pix_fmt", "yuv420p", "-threads", "2", str(path)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for fi in range(f0, f1):
        p.stdin.write(frame(fi).tobytes())
    p.stdin.close()
    if p.wait() != 0:
        raise RuntimeError(f"ffmpeg failed on {path}")
    return path


def render_all(workers, chunks=16):
    BUILD.mkdir(exist_ok=True)
    total = DURATION * FPS
    edges = [round(total * i / chunks) for i in range(chunks + 1)]
    jobs = [(edges[i], edges[i + 1], BUILD / f"part{i:02d}.mp4") for i in range(chunks)]
    t0 = time.time()
    with Pool(workers) as pool:
        for i, _ in enumerate(pool.imap(_chunk, jobs)):
            print(f"  chunk {i + 1}/{chunks} done ({time.time() - t0:.0f}s)", flush=True)
    lst = BUILD / "parts.txt"
    lst.write_text("".join(f"file '{p.name}'\n" for _, _, p in jobs))
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", str(lst),
                    "-c", "copy", str(BUILD / "video.mp4")], check=True)
    for _, _, p in jobs:
        p.unlink()
    lst.unlink()
    print(f"video track: {BUILD / 'video.mp4'} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preview", help="comma separated times in seconds")
    ap.add_argument("--out", default=str(BUILD / "preview"))
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.preview:
        out = Path(a.out)
        out.mkdir(parents=True, exist_ok=True)
        for s in a.preview.split(","):
            t0 = time.time()
            img = frame(int(round(float(s) * FPS)))
            cv2.imwrite(str(out / f"f_{float(s):07.2f}.png"), img[..., ::-1])
            print(f"{s}s rendered in {time.time() - t0:.2f}s")
    else:
        render_all(a.workers)
