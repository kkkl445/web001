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


INK = np.array([0.035, 0.045, 0.075], np.float32)


def badge(F, text, cx, y, size=88, pad_x=46, pad_y=26):
    """Dark text on a solid gold pill - the line that has to be read first."""
    tx = zh(text, size, 900, 0.04)
    w = int(tx.width + 2 * pad_x)
    h = int(size * 1.05 + 2 * pad_y)
    r = h // 2
    ss = 4
    m = np.zeros((h * ss, w * ss), np.uint8)
    cv2.rectangle(m, (r * ss, 0), ((w - r) * ss, h * ss - 1), 255, -1)
    cv2.circle(m, (r * ss, r * ss), r * ss, 255, -1, cv2.LINE_AA)
    cv2.circle(m, ((w - r) * ss, r * ss), r * ss, 255, -1, cv2.LINE_AA)
    m = cv2.resize(m.astype(np.float32) / 255, (w, h), interpolation=cv2.INTER_AREA)
    x0, y0 = cx - w / 2, y - h / 2
    add_sprite(F, cx, y, w * 0.55, GOLD, 0.18)
    ST.blend(F, m, x0, y0, GOLD, 1.0)
    tx.draw(F, cx, y + size * 0.36, INK, 1.0, align="center")
    return h


def headline(F, y, badge_size=88, title_size=132, q_size=48, gap=None):
    """7 分钟看懂 / Transformer / 「它」指的是谁？"""
    h = badge(F, "7 分钟看懂", W / 2, y, badge_size)
    ty = y + h / 2 + title_size * 1.02
    ST.punch(F, "Transformer", W / 2, ty, 10.0, 0.0, size=title_size, wght=800, tracking=0.0)
    ST.add_sprite(F, W / 2, ty - title_size * 0.35, title_size * 2.4, GOLD, 0.05)
    qy = ty + q_size * 1.9
    ST.punch(F, "「它」指的是谁？", W / 2, qy, 10.0, 0.0, size=q_size, wght=700, gold=(1,))
    return qy


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
    add_sprite(F, W / 2, 860, 440, GOLD, 0.07)
    headline(F, 300)
    scene(F, 1050, 1215, 1327)
    caps("ATTENTION IS ALL YOU NEED   ·   2017", 20, 0.4).draw(F, W / 2, 1450, DIM, 1.0, align="center")
    finish(F, "cover-9x16.png")


def portrait():
    ST.E.H = 1440                     # the engine clips drawing to its frame size
    F = canvas(1440, seed=5)
    add_sprite(F, W / 2, 700, 400, GOLD, 0.07)
    headline(F, 160, badge_size=80, title_size=120, q_size=44)
    scene(F, 855, 1015, 1118, s_cat=300, s_table=250, size=70)
    caps("ATTENTION IS ALL YOU NEED   ·   2017", 19, 0.4).draw(F, W / 2, 1245, DIM, 1.0, align="center")
    finish(F, "poster-3x4.png")
    ST.E.H = H


if __name__ == "__main__":
    cover()
    portrait()
