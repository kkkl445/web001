"""Render the picture track.

  python3 render.py --preview 10,45,97 --out DIR   # stills (seconds) as PNG
  python3 render.py --workers 4                    # full video -> build/video.mp4
"""

import argparse
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path

import cv2

import scenes as S
from engine import ROOT, H, W, finalize, make_background, make_grain, window
from timeline import DURATION, FPS, SCENES, START

BUILD = ROOT / "build"
CHAPTER = {f"ch{i}": i for i in range(1, 9)}
_state = {}


def _init():
    if not _state:
        bg, vig = make_background()
        _state.update(bg=bg, vig=vig, grain=make_grain())
        import physics
        physics.orbitals(), physics.double_slit(), physics.tunnelling()


def frame(fi):
    _init()
    g = fi / FPS
    name, dur = SCENES[-1]
    for n, d in SCENES:
        if START[n] <= g < START[n] + d:
            name, dur = n, d
            break
    t = g - START[name]
    F = _state["bg"].copy()
    S.DUST.draw(F, g, 1.0)
    S.SCENE_FUNCS[name](F, t, g, fi)
    num = CHAPTER.get(name, 0)
    S.hud(F, g, window(g, START["ch1"], START["outro"], 1.2, 1.2), num,
          window(t, 0, dur, 0.9, 0.9) if num else 0.0)
    fade = min(1.0, g / 1.2) * min(1.0, (DURATION - g) / 2.0)
    if fade < 1:
        F *= max(fade, 0.0)
    grain = _state["grain"][fi % len(_state["grain"])] * (fade if fade < 1 else 1.0)
    return finalize(F, _state["vig"], grain)


def _chunk(args):
    f0, f1, path = args
    _init()
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
           "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-c:v", "libx264", "-preset", "slow", "-crf", "19", "-tune", "film",
           "-x264-params", "aq-mode=3", "-pix_fmt", "yuv420p", "-threads", "2", str(path)]
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
        for i, p in enumerate(pool.imap(_chunk, jobs)):
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
            fi = int(round(float(s) * FPS))
            t0 = time.time()
            img = frame(fi)
            cv2.imwrite(str(out / f"f_{float(s):07.2f}.png"), img[..., ::-1])
            print(f"{s}s rendered in {time.time() - t0:.2f}s")
    else:
        render_all(a.workers)
