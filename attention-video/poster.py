"""Covers for 注意力 · Attention Is All You Need.

The hero image is the film's opening plate: the cat drawn as a star-atlas
figure, the Ming table, the sentence, and the gold thread from 「它」 to 小猫.
9:16 for the video cover, 3:4 for the profile grid (image above, title below).

  python3 poster.py   -> cover-9x16.png (1080x1920), poster-3x4.png (1080x1440)
"""

import cv2
import numpy as np

import plates as P
import style as ST
from style import GOLD, IVORY, DIM, T, caps, hairline, serif

W, H = 1080, 1920


def canvas(seed=4):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    F = np.empty((H, W, 3), np.float32)
    F[:] = ST.rgb("030407")
    d = ((xx - W / 2) / (0.7 * W)) ** 2 + ((yy - 0.47 * H) / (0.55 * H)) ** 2
    F += (ST.rgb("0b1322") - ST.rgb("030407")) * np.exp(-d * 1.4)[..., None]
    stars = np.zeros((H, W), np.float32)
    for x, y, b in zip(rng.uniform(0, W, 900), rng.uniform(0, H, 900), rng.power(4, 900) * 0.22 + 0.02):
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


def plate(F):
    """The opening plate at the moment 「它」 has found 小猫."""
    P.opening(F, w_cat=1.0, w_table=0.0, sleep=0.0, h=1.0, it_on=1.0, link=1.0, reveal=1.0, swap=0.0, a=1.0)


def title(F, y, size):
    for k, line in enumerate(("Attention", "Is All You Need")):
        T(line, "corm", size, 600, tracking=0.02).draw(F, W / 2, y + k * size * 1.08, IVORY, 1.0, align="center",
                                                        glow=0.3, glow_color=GOLD, glow_sigma=16)
    yz = y + size * 1.08 + size * 0.95
    serif("注意力，就是你所需要的一切", int(size * 0.38), 500, 0.2).draw(F, W / 2, yz, GOLD, 1.0, align="center")
    hairline(F, W / 2 - 80, int(yz + size * 0.38), W / 2 + 80, GOLD, 0.6)
    caps("VASWANI ET AL.   ·   2017", int(size * 0.17), 0.45).draw(F, W / 2, yz + size * 0.78, DIM, 1.0,
                                                                   align="center")


def cover():
    F = canvas()
    title(F, 225, 108)
    plate(F)
    serif("「它」指的是谁？你一眼就知道。", 38, 400, 0.12).draw(F, W / 2, 1420, IVORY, 0.9, align="center")
    finish(F, "cover-9x16.png")


def portrait():
    """3:4: the plate above, the title below; cut from a 9:16 canvas so the plate keeps its proportions."""
    F = canvas(seed=5)
    plate(F)
    title(F, 1380, 100)
    F = F[380:380 + 1440].copy()
    finish(F, "poster-3x4.png")


if __name__ == "__main__":
    cover()
    portrait()
