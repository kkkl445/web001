"""Cover art for 量子隧穿 · 南墙之外.

The hero image is the film's own 2D simulation, caught just after the wave
has hit the wall: the blue part that turned back, the gold part that got
through.  Rendered with the same engine and palette as the film.

  python3 poster.py   -> poster-3x4.png (1080x1440), cover-16x9.png (1920x1080)
"""

import math

import cv2
import numpy as np

import physics as P
import style as S
from style import E

FRAME = 240                      # reflected and transmitted packets both clear of the wall


def canvas(w, h):
    """Point the engine at a w x h canvas and return a starfield background + vignette."""
    E.W, E.H, S.W, S.H = w, h, w, h
    rng = np.random.default_rng(4)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    F = np.empty((h, w, 3), np.float32)
    F[:] = S.rgb("030407")
    d = ((xx - w / 2) / (0.6 * w)) ** 2 + ((yy - 0.45 * h) / (0.6 * h)) ** 2
    F += (S.rgb("0b1322") - S.rgb("030407")) * np.exp(-d * 1.4)[..., None]
    stars = np.zeros((h, w), np.float32)
    n = int(720 * w * h / (1920 * 864))
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
    cv2.imwrite(str(S.ROOT / path), out[..., ::-1])
    print("wrote", path, out.shape[1], "x", out.shape[0])


def palette(v, stops, cols):
    v = np.clip(v, 0, 1)
    return np.stack([np.interp(v, stops, [c[i] for c in cols]) for i in range(3)], -1).astype(np.float32)


COOL = ([0, 0.18, 0.45, 0.75, 1.0],
        [(0, 0, 0), (0.02, 0.05, 0.11), (0.1, 0.3, 0.52), (0.48, 0.76, 0.92), (0.96, 0.97, 0.95)])
WARM = ([0, 0.16, 0.42, 0.72, 1.0],
        [(0, 0, 0), (0.18, 0.07, 0.01), (0.7, 0.4, 0.1), (1.0, 0.8, 0.45), (1.0, 0.97, 0.9)])


def sim_image(x0, x1, y0, y1, boost=1.0):
    """Coloured simulation crop (cells x0..x1 along the motion, y0..y1 across), motion along +x."""
    D = P.packet2d()
    p = D["prob"][FRAME].astype(np.float32)
    r = D["re"][FRAME].astype(np.float32)
    p0 = float(D["prob"][0].astype(np.float32).max())
    v = np.clip((0.55 * p + 0.45 * r * r) / p0, 0, None)
    x = np.arange(P.NX)
    gain = np.where(x >= P.BAR_X0 + P.BAR_W, boost, 1.0)[None, :]          # lift the faint gold side
    v = (v * gain) ** 0.62
    side = np.clip((x - (P.BAR_X0 + P.BAR_W)) / 6.0, 0, 1)[None, :, None].astype(np.float32)
    col = palette(v, *COOL) * (1 - side) + palette(v, *WARM) * side
    return col[y0:y1, x0:x1]


def glass(F, a, b, across0, across1, vertical, scale):
    """The wall: a dark glass slab with glowing edges, fading out at its ends."""
    n = int(across1 - across0)
    prof = (np.clip(np.arange(n) / (0.22 * n), 0, 1) * np.clip((n - np.arange(n)) / (0.22 * n), 0, 1)) ** 1.5
    prof = prof.astype(np.float32)
    th = int(b - a)
    slab = np.repeat(prof[:, None], th, 1)
    edge = np.zeros((n, th + 40), np.float32)
    edge[:, 20] = prof
    edge[:, 20 + th] = prof
    glow = cv2.GaussianBlur(edge, (0, 0), 0.6 * scale)
    core = cv2.GaussianBlur(edge, (0, 0), 0.6)
    if not vertical:                       # wall runs horizontally across the frame
        slab, glow, core = slab.T, glow.T, core.T
        S.blend(F, slab, across0, a, np.array([0.03, 0.05, 0.09], np.float32), 0.9)
        S.add_light(F, glow, across0, a - 20, S.CYAN, 3.2)
        S.add_light(F, core, across0, a - 20, S.mix(S.CYAN, S.WHITE, 0.4), 1.3)
    else:
        S.blend(F, slab, a, across0, np.array([0.03, 0.05, 0.09], np.float32), 0.9)
        S.add_light(F, glow, a - 20, across0, S.CYAN, 3.2)
        S.add_light(F, core, a - 20, across0, S.mix(S.CYAN, S.WHITE, 0.4), 1.3)


def readout(F, x, y, value, zh, en, color, align="left"):
    S.T(value, "stix", 64, None, per_char=False).draw(F, x, y, color, 1.0, align=align)
    zl = S.serif(zh, 18, 500, 0.3)
    el = S.caps(en, 11, 0.4)
    w = zl.width + 12 + el.width
    x0 = x if align == "left" else x - w
    zl.draw(F, x0, y + 34, color, 0.9)
    el.draw(F, x0 + zl.width + 12, y + 33, S.DIM, 0.9)


def title_block(F, cx, y, size=150):
    t = S.serif("量子隧穿", size, 500, 0.32)
    t.draw(F, cx, y, S.IVORY, 1.0, align="center", glow=0.35, glow_color=S.GOLD, glow_sigma=18)
    S.serif("南墙之外", int(size * 0.27), 500, 0.6).draw(F, cx, y + size * 0.62, S.GOLD, 1.0, align="center")
    k = size / 150
    S.hairline(F, cx - 70 * k, int(y + size * 0.78), cx + 70 * k, S.GOLD, 0.6)
    S.caps("QUANTUM TUNNELING   ·   BEYOND THE WALL", int(13 * k + 1), 0.42).draw(
        F, cx, y + size * 0.98, S.DIM, 1.0, align="center")


def portrait():
    w, h = 1080, 1440
    F, vig = canvas(w, h)
    sc = 4.6                                       # px per simulation cell
    wall_y = 840
    x0, x1 = 140, 380                              # cells along the motion (wall at 254..259)
    y0, y1 = 11, 245
    img = sim_image(x0, x1, y0, y1, boost=1.35)
    img = np.rot90(img, 1)                         # motion now points up the page: beyond the wall is above
    ih, iw = int(round(img.shape[0] * sc)), int(round(img.shape[1] * sc))
    img = cv2.resize(np.ascontiguousarray(img), (iw, ih), interpolation=cv2.INTER_CUBIC)
    top = wall_y - int(round((x1 - P.BAR_X0) * sc))
    left = (w - iw) // 2
    S.E.add_rgb(F, img * 0.95, left, top, 1.0)
    glass(F, wall_y - P.BAR_W * sc, wall_y, 0, w, vertical=False, scale=sc)
    D = P.packet2d()
    readout(F, 96, 560, f"{round(100 * float(D['T']))}%", "穿过", "GOT THROUGH", S.GOLD)
    readout(F, w - 96, 1120, f"{round(100 * float(D['R']))}%", "回头", "TURNED BACK", S.CYAN, align="right")
    # fade the top of the frame into darkness so the title sits on black
    yy = np.arange(h, dtype=np.float32)
    shade = np.clip((yy - 250) / 260, 0, 1) ** 1.2 * 0.75 + 0.25
    shade = np.where(yy > 510, 1.0, shade)
    F *= shade[:, None, None]
    title_block(F, w / 2, 230, 150)
    S.serif("撞了南墙，它却不一定回头。", 34, 400, 0.18).draw(F, w / 2, 1318, S.IVORY, 0.95, align="center")
    S.italic("Hit the wall, and it may not turn back.", 22).draw(F, w / 2, 1360, S.DIM, 1.0, align="center")
    finish(F, vig, "poster-3x4.png")


def wide():
    w, h = 1920, 1080
    F, vig = canvas(w, h)
    sc = 5.6
    wall_x = 1290
    x0, x1 = 150, 375
    y0, y1 = 11, 245
    img = sim_image(x0, x1, y0, y1, boost=1.35)
    ih, iw = int(round(img.shape[0] * sc)), int(round(img.shape[1] * sc))
    img = cv2.resize(img, (iw, ih), interpolation=cv2.INTER_CUBIC)
    left = wall_x - int(round((P.BAR_X0 - x0) * sc))
    top = (h - ih) // 2
    xx = np.arange(iw, dtype=np.float32)
    fade = np.clip((xx - 0) / 220, 0, 1)[None, :, None]          # soften the left edge under the title
    S.E.add_rgb(F, img * fade * 0.95, left, top, 1.0)
    glass(F, wall_x, wall_x + P.BAR_W * sc, 0, h, vertical=True, scale=sc)
    D = P.packet2d()
    readout(F, 1680, 300, f"{round(100 * float(D['T']))}%", "穿过", "GOT THROUGH", S.GOLD)
    readout(F, 1190, 300, f"{round(100 * float(D['R']))}%", "回头", "TURNED BACK", S.CYAN, align="right")
    t = S.serif("量子隧穿", 136, 500, 0.3)
    t.draw(F, 120, 540, S.IVORY, 1.0, glow=0.35, glow_color=S.GOLD, glow_sigma=16)
    S.serif("南墙之外", 38, 500, 0.6).draw(F, 124, 614, S.GOLD, 1.0)
    S.hairline(F, 124, 642, 300, S.GOLD, 0.6)
    S.caps("QUANTUM TUNNELING   ·   BEYOND THE WALL", 14, 0.42).draw(F, 124, 678, S.DIM, 1.0)
    S.serif("撞了南墙，它却不一定回头。", 30, 400, 0.18).draw(F, 124, 772, S.IVORY, 0.9)
    finish(F, vig, "cover-16x9.png")


if __name__ == "__main__":
    portrait()
    wide()
