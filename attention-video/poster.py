"""Covers for 注意力 · Attention Is All You Need.

Made for the feed: a big two-line headline people can read at thumbnail size,
the formula glowing in the middle, and the film's star-atlas cat and Ming
table underneath.  9:16 for the video cover, 3:4 for the profile grid.

  python3 poster.py   -> cover-9x16.png (1080x1920), poster-3x4.png (1080x1440)
"""

import cv2
import numpy as np

import plates as P
import scenes as S
import style as ST
from style import GOLD, DIM, add_sprite, caps, zh

W, H = 1080, 1920


def canvas(h=H, seed=4):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:W].astype(np.float32)
    F = np.empty((h, W, 3), np.float32)
    F[:] = ST.rgb("030407")
    d = ((xx - W / 2) / (0.7 * W)) ** 2 + ((yy - 0.45 * h) / (0.55 * h)) ** 2
    F += (ST.rgb("0b1322") - ST.rgb("030407")) * np.exp(-d * 1.4)[..., None]
    stars = np.zeros((h, W), np.float32)
    n = int(900 * h / 1920)
    for x, y, b in zip(rng.uniform(0, W, n), rng.uniform(0, h, n), rng.power(4, n) * 0.22 + 0.02):
        cv2.circle(stars, (int(x * 16), int(y * 16)), int(rng.uniform(0.5, 1.2) * 16), float(b), -1, cv2.LINE_AA, 4)
    F += stars[..., None] * np.array([0.85, 0.9, 1.0], np.float32)
    return F


def finish(F, path):
    h, w = F.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    vig = (1 - 0.55 * np.clip((r - 0.5) / 0.9, 0, 1) ** 1.5).astype(np.float32)
    small = cv2.resize(F, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    br = np.maximum(small - 0.42, 0)
    b = cv2.GaussianBlur(br, (0, 0), 2.5) * 0.55 + cv2.GaussianBlur(br, (0, 0), 11) * 0.75
    F = F + cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR)
    F *= vig[..., None]
    F += np.random.default_rng(8).normal(0, 0.006, (h, w))[..., None].astype(np.float32)
    knee = 0.8
    hi = F > knee
    F[hi] = knee + (1 - knee) * (1 - np.exp(-(F[hi] - knee) / (1 - knee)))
    out = (np.clip(F, 0, 1) * 255 + 0.5).astype(np.uint8)
    cv2.imwrite(str(ST.ROOT / path), out[..., ::-1])
    print("wrote", path, out.shape[1], "x", out.shape[0])


def headline(F, y, size=96):
    """你用的每个 AI / 心脏都是这一行 - bold, high contrast, readable as a thumbnail."""
    ST.punch(F, "你用的每个 AI，", W / 2, y, 10.0, 0.0, size=int(size * 0.86), wght=800)
    ST.punch(F, "心脏都是这一行", W / 2, y + size * 1.22, 10.0, 0.0, size=size, wght=900, gold=(0, 1))


def plate(F, ground, s_cat=300, s_table=250):
    P.baseline(F, ground + 2, 120, 960, 1.0)
    P.table_plate(F, 745, ground, s_table, 1.0, glow=0.0)
    P.cat_plate(F, 320, ground, s_cat, "sit", glow=1.0)


def cover():
    F = canvas()
    add_sprite(F, W / 2, 760, 420, GOLD, 0.10)
    headline(F, 330)
    S.formula_big(F, W / 2, 720, 10.0, 0.0, 1.0, scale=1.0)
    plate(F, 1210)
    zh("7 分钟，彻底看懂 Transformer", 46, 700, 0.06).draw(F, W / 2, 1345, GOLD, 1.0, align="center")
    caps("ATTENTION IS ALL YOU NEED   ·   2017", 20, 0.4).draw(F, W / 2, 1400, DIM, 1.0, align="center")
    finish(F, "cover-9x16.png")


def portrait():
    ST.E.H = 1440                     # the engine clips drawing to its frame size
    F = canvas(1440, seed=5)
    add_sprite(F, W / 2, 610, 380, GOLD, 0.10)
    headline(F, 210, size=92)
    S.formula_big(F, W / 2, 580, 10.0, 0.0, 1.0, scale=0.95)
    plate(F, 1080, s_cat=270, s_table=230)
    zh("7 分钟，彻底看懂 Transformer", 44, 700, 0.06).draw(F, W / 2, 1200, GOLD, 1.0, align="center")
    caps("ATTENTION IS ALL YOU NEED   ·   2017", 19, 0.4).draw(F, W / 2, 1252, DIM, 1.0, align="center")
    finish(F, "poster-3x4.png")
    ST.E.H = H


if __name__ == "__main__":
    cover()
    portrait()
