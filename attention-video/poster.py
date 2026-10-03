"""Covers for 注意力 · Attention Is All You Need.

Made for the feed: one question people can read at thumbnail size - 「它」指的是谁？ -
over the film's opening plate (the star-atlas cat, the Ming table, the sentence
and the gold thread from 「它」 to 小猫).  No formula and no sweeping claims, so
the cover passes the platform's review.  9:16 for the video, 3:4 for the grid.

  python3 poster.py   -> cover-9x16.png (1080x1920), poster-3x4.png (1080x1440)
"""

import cv2
import numpy as np

import plates as P
import style as ST
from style import GOLD, IVORY, DIM, add_sprite, caps, mix, zh

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


def headline(F, y, size=110):
    """「它」指的是谁？ - a question, readable as a thumbnail, with no formula and no big claims."""
    ST.punch(F, "「它」指的是谁？", W / 2, y, 10.0, 0.0, size=size, wght=900, gold=(1,))


def scene(F, ground, y1, y2, s_cat=340, s_table=280, size=76):
    """The film's opening plate: the star-atlas cat glowing because 「它」 points at it, the Ming table,
    the sentence, and the gold thread from 「它」 back to 小猫."""
    P.baseline(F, ground + 2, 110, 970, 1.0)
    P.table_plate(F, 735, ground, s_table, 1.0, glow=0.0)
    P.cat_plate(F, 305, ground, s_cat, "sit", glow=1.0)
    sent = P.Line2(("小猫没有跳上桌子，", "因为它太累了。"),
                   [(0, 0, 2), (0, 2, 4), (0, 4, 6), (0, 6, 8), (1, 0, 2), (1, 2, 3), (1, 3, 4), (1, 4, 5), (1, 5, 6)],
                   size, (y1, y2))
    cols = [IVORY] * 9
    cols[P.IT] = GOLD
    cols[P.CAT] = mix(IVORY, GOLD, 0.65)
    sent.draw(F, 1.0, cols=cols)
    add_sprite(F, *sent.centers[P.IT], 50, GOLD, 0.25)
    P.thread(F, sent.centers[P.IT] + [-size * 0.45, 0], sent.centers[P.CAT] + [0, size * 0.62], 0.9, bow=-50)
    P.dots(F, sent.centers[P.CAT] + [0, -size * 0.7], np.array([305 + 38, ground - s_cat * 0.46]), 1.0)


def cover():
    F = canvas()
    add_sprite(F, W / 2, 820, 440, GOLD, 0.08)
    headline(F, 360)
    zh("AI 是怎么读懂这句话的？", 46, 600, 0.08).draw(F, W / 2, 470, IVORY, 0.9, align="center")
    scene(F, 950, 1120, 1232)
    zh("7 分钟看懂 Transformer", 46, 700, 0.06).draw(F, W / 2, 1395, GOLD, 1.0, align="center")
    caps("ATTENTION IS ALL YOU NEED   ·   2017", 20, 0.4).draw(F, W / 2, 1450, DIM, 1.0, align="center")
    finish(F, "cover-9x16.png")


def portrait():
    ST.E.H = 1440                     # the engine clips drawing to its frame size
    F = canvas(1440, seed=5)
    add_sprite(F, W / 2, 640, 400, GOLD, 0.08)
    headline(F, 220, size=104)
    zh("AI 是怎么读懂这句话的？", 42, 600, 0.08).draw(F, W / 2, 320, IVORY, 0.9, align="center")
    scene(F, 770, 935, 1043, s_cat=320, s_table=265, size=72)
    zh("7 分钟看懂 Transformer", 44, 700, 0.06).draw(F, W / 2, 1200, GOLD, 1.0, align="center")
    caps("ATTENTION IS ALL YOU NEED   ·   2017", 19, 0.4).draw(F, W / 2, 1252, DIM, 1.0, align="center")
    finish(F, "poster-3x4.png")
    ST.E.H = H


if __name__ == "__main__":
    cover()
    portrait()
