"""Covers for 注意力 · Attention Is All You Need.

The hero image is the film's opening: the sentence, the gold line from 「它」
to 小猫, and the constellation cat lit up because it is the one being looked
at.  9:16 for the video cover, 3:4 for the profile grid.

  python3 poster.py   -> cover-9x16.png (1080x1920), poster-3x4.png (1080x1440)
"""

import math

import cv2
import numpy as np

import figures2d as G
import scenes as S
import style as ST
from style import E, GOLD, IVORY, DIM, T, add_sprite, caps, hairline, serif


def canvas(w, h, seed=4):
    E.W, E.H, ST.W, ST.H = w, h, w, h
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    F = np.empty((h, w, 3), np.float32)
    F[:] = ST.rgb("030407")
    d = ((xx - w / 2) / (0.7 * w)) ** 2 + ((yy - 0.45 * h) / (0.55 * h)) ** 2
    F += (ST.rgb("0b1322") - ST.rgb("030407")) * np.exp(-d * 1.4)[..., None]
    stars = np.zeros((h, w), np.float32)
    n = int(900 * w * h / (1080 * 1920))
    for x, y, b in zip(rng.uniform(0, w, n), rng.uniform(0, h, n), rng.power(4, n) * 0.22 + 0.02):
        cv2.circle(stars, (int(x * 16), int(y * 16)), int(rng.uniform(0.5, 1.2) * 16), float(b), -1, cv2.LINE_AA, 4)
    F += stars[..., None] * np.array([0.85, 0.9, 1.0], np.float32)
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    vig = (1 - 0.55 * np.clip((r - 0.5) / 0.9, 0, 1) ** 1.5).astype(np.float32)
    return F, vig


def finish(F, vig, path):
    h, w = F.shape[:2]
    small = cv2.resize(F, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    br = np.maximum(small - 0.42, 0)
    b = cv2.GaussianBlur(br, (0, 0), 2.5) * 0.55 + cv2.GaussianBlur(br, (0, 0), 11) * 0.75
    F += cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR)
    F *= vig[..., None]
    F += np.random.default_rng(8).normal(0, 0.006, (h, w))[..., None].astype(np.float32)
    knee = 0.8
    hi = F > knee
    F[hi] = knee + (1 - knee) * (1 - np.exp(-(F[hi] - knee) / (1 - knee)))
    out = (np.clip(F, 0, 1) * 255 + 0.5).astype(np.uint8)
    cv2.imwrite(str(ST.ROOT / path), out[..., ::-1])
    print("wrote", path, out.shape[1], "x", out.shape[0])


def scene(F, w, ground, y1, y2, s_cat=140, size=84):
    """Cat, table, the sentence and the gold line from 「它」 to 小猫."""
    sent = S.Sentence(S.SENT_LINES, S.SENT_TOKENS, size, (y1, y2), cx=w / 2 - 20)
    for x in np.linspace(w * 0.14, w * 0.84, 30):
        G.orb(F, x, ground + 4, 1.2, G.STEEL, 0.22)
    tp, te = G.table(w * 0.44, ground, s_cat, 1.5)
    G.draw(F, tp, te, glow=0.0, seed=2)
    cp, ce, ceye = G.cat(w * 0.25, ground, s_cat, down=1.0)
    G.draw(F, cp, ce, glow=1.0, seed=1, extra=ceye)
    S.fan(F, sent, S.IT, S.PWA, 1.0, label=True, a=1.0, hl=1.0)
    tx, ty, _, _ = sent.centers[S.CAT]
    S.link(F, np.array([tx, ty - size - 66]), cp[12] + np.array([0, 18]), 1.0)
    cols = [IVORY] * len(S.SENT_TOKENS)
    cols[S.IT] = GOLD
    sent.draw(F, tok_color=cols)
    sx, sy, _, _ = sent.centers[S.IT]
    add_sprite(F, sx, sy - size * 0.35, 60, GOLD, 0.35)


def title(F, w, y, size):
    for k, line in enumerate(("Attention", "Is All You Need")):
        T(line, "corm", size, 600, tracking=0.02).draw(F, w / 2, y + k * size * 1.08, IVORY, 1.0, align="center",
                                                        glow=0.3, glow_color=GOLD, glow_sigma=16)
    yz = y + size * 1.08 + size * 0.95
    serif("注意力，就是你所需要的一切", int(size * 0.38), 500, 0.2).draw(F, w / 2, yz, GOLD, 1.0, align="center")
    hairline(F, w / 2 - 80, int(yz + size * 0.38), w / 2 + 80, GOLD, 0.6)
    caps("VASWANI ET AL.   ·   2017", int(size * 0.16), 0.45).draw(F, w / 2, yz + size * 0.78, DIM, 1.0,
                                                                   align="center")


def cover():
    w, h = 1080, 1920
    F, vig = canvas(w, h)
    title(F, w, 330, 120)
    scene(F, w, 1030, 1200, 1385)
    serif("「它」指的是谁？你一眼就知道。", 36, 400, 0.12).draw(F, w / 2, 1530, IVORY, 0.9, align="center")
    finish(F, vig, "cover-9x16.png")


def portrait():
    w, h = 1080, 1440
    F, vig = canvas(w, h)
    title(F, w, 200, 104)
    scene(F, w, 830, 985, 1160, s_cat=124, size=78)
    serif("「它」指的是谁？你一眼就知道。", 34, 400, 0.12).draw(F, w / 2, 1330, IVORY, 0.9, align="center")
    finish(F, vig, "poster-3x4.png")


if __name__ == "__main__":
    cover()
    portrait()
