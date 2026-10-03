"""The redesigned pictures, one function per plate.  Each draws a complete,
static composition for a set of progress values; scenes.py animates them.

References: the star-atlas plate (Hevelius, Urania's Mirror) for the cat and
the table; the Tensor2Tensor attention view by Llion Jones, one of the paper's
authors, for "who looks at whom"; 3Blue1Brown's attention grid (dots that grow
with the score) for the table; 18th-century cut-paper profile portraits for
people."""

import math

import numpy as np

import atlas as A
import figures2d as G
from style import (CX, DIM, GOLD, IVORY, E, T, add_sprite, blend, caps, glow_poly, hairline, mix, orb, ramp, serif,
                   smooth)

STEEL = G.STEEL
WARM = mix(GOLD, E.rgb("c98a4a"), 0.5)
RIM = mix(GOLD, IVORY, 0.45)
INK = np.array([0.05, 0.07, 0.11], np.float32)
SIL = np.array([0.012, 0.016, 0.026], np.float32)

TOKENS = ["小猫", "没有", "跳上", "桌子", "因为", "它", "太", "累", "了"]
IT, CAT, TABLE = 5, 0, 3


# ----------------------------------------------------------- the opening ---

def cat_plate(F, x, ground, s, pose="sit", reveal=1.0, glow=0.0, a=1.0, chart=1.0):
    """The cat as a star-atlas figure: stars, then the lines between them, then the engraving."""
    if pose == "sit":
        fig = A.figure(A.CAT_SIT, A.CAT_SIT_TAIL, x, ground, s)
        stars, lines = A.CAT_SIT_STARS, A.CAT_SIT_LINES
    else:
        fig = A.figure(A.CAT_SLEEP, A.CAT_SLEEP_TAIL, x, ground, s, w0=0.065, w1=0.03)
        stars, lines = A.CAT_SLEEP_STARS, A.CAT_SLEEP_LINES
    col = mix(mix(STEEL, IVORY, 0.5), GOLD, glow)
    A.engraved(F, fig["outline"], col, a * (0.75 + 0.25 * glow), reveal=reveal, form_a=0.5 + 0.3 * glow,
               fill=0.03 + 0.05 * glow)
    if reveal > 0.6:
        A.stroke(F, fig["tail_edge"], col, 0.75 * a * smooth(ramp(reveal, 0.6, 0.4)))
        ex, ey = fig["eye"]
        if fig["sleep"]:
            th = np.linspace(0.2, math.pi - 0.2, 12)
            A.stroke(F, np.column_stack([ex + s * 0.022 * np.cos(th), ey + s * 0.008 * np.sin(th)]), col,
                     0.8 * a * smooth(ramp(reveal, 0.7, 0.3)))
        else:
            orb(F, ex, ey, 2.4, col, 0.8 * a * smooth(ramp(reveal, 0.7, 0.3)))
    if chart > 0:
        A.chart(F, A.stars_on(stars, x, ground, s), lines, color=mix(col, IVORY, 0.3), a=0.75 * a * chart,
                glow=glow, size=0.8)
    return fig


def table_plate(F, x, ground, s, h=1.0, reveal=1.0, glow=0.0, a=1.0):
    col = mix(mix(STEEL, IVORY, 0.5), GOLD, glow)
    lines = A.ming_table(x, ground, s, h)
    for k, l in enumerate(lines):
        A.stroke(F, l, col, (0.7 + 0.3 * glow) * a, reveal=smooth(ramp(reveal, 0.5 * k / len(lines), 0.5)),
                 glow=0.3 + 0.6 * glow)
    return lines


def baseline(F, y, x0, x1, a):
    """The plate's ground: a hairline that fades at both ends."""
    if a <= 0.003:
        return
    n = 24
    for k in range(n):
        u0, u1 = k / n, (k + 1) / n
        w = math.sin(math.pi * (u0 + u1) / 2) ** 1.5
        hairline(F, x0 + (x1 - x0) * u0, y, x0 + (x1 - x0) * u1, IVORY, 0.28 * a * w)


class Line2:
    """A two-line sentence, big, with token centres for threads and highlights."""

    def __init__(self, lines, spans, size, ys, cx=CX, tracking=0.14):
        self.size, self.ys = size, ys
        self.txt = [serif(l, size, 400, tracking) for l in lines]
        self.x0 = [cx - tx.width / 2 for tx in self.txt]
        self.spans = spans
        self.char_tok = {}
        self.centers = []
        for k, (ln, a, b) in enumerate(spans):
            for c in range(a, b):
                self.char_tok[(ln, c)] = k
            it = self.txt[ln].items
            xa = self.x0[ln] + it[a][1]
            xb = self.x0[ln] + it[b - 1][1] + it[b - 1][0].shape[1]
            self.centers.append(np.array([(xa + xb) / 2, ys[ln] - size * 0.36]))

    def draw(self, F, a=1.0, cols=None, alphas=None, swap=None):
        for ln, tx in enumerate(self.txt):
            for i, (m, dx, dy) in enumerate(tx.items):
                k = self.char_tok.get((ln, i))
                col = cols[k] if (cols is not None and k is not None) else IVORY
                al = a * (alphas[k] if (alphas is not None and k is not None) else 1.0)
                x, y = self.x0[ln] + dx, self.ys[ln] + dy
                if swap is not None and swap[0] == ln and swap[1] == i and swap[3] > 0:
                    sw = swap[3]
                    blend(F, m, x, y - 18 * sw, col, al * (1 - sw))
                    alt = serif(swap[2], self.size, 400, 0).items[0]
                    blend(F, alt[0], x + (m.shape[1] - alt[0].shape[1]) / 2, self.ys[ln] + alt[2] + 14 * (1 - sw),
                          mix(GOLD, col, sw ** 2), a * sw)
                else:
                    blend(F, m, x, y, col, al)


OPEN = Line2(("小猫没有跳上桌子，", "因为它太累了。"),
             [(0, 0, 2), (0, 2, 4), (0, 4, 6), (0, 6, 8), (1, 0, 2), (1, 2, 3), (1, 3, 4), (1, 4, 5), (1, 5, 6)],
             80, (1095, 1210))


def thread(F, p, q, a, col=GOLD, bow=60, n=60, width=1.6):
    """A fine curved thread between two points, brightest in the middle."""
    if a <= 0.003:
        return
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = (p + q) / 2
    d = q - p
    nrm = np.array([d[1], -d[0]]) / (np.linalg.norm(d) + 1e-9)
    c = m + nrm * bow
    u = np.linspace(0, 1, n)[:, None]
    pts = (1 - u) ** 2 * p + 2 * (1 - u) * u * c + u ** 2 * q
    glow_poly(F, pts, col, a, th=1 if width < 2 else 2, glow=0.8, sigma=4)
    return pts


def opening(F, w_cat=1.0, w_table=0.0, sleep=0.0, h=1.0, it_on=1.0, link=1.0, reveal=1.0, swap=0.0, a=1.0):
    """Plate: the cat and the Ming table above the sentence; 它 joined to whichever it means."""
    ground = 900
    baseline(F, ground + 2, 100, 980, a * reveal)
    table_plate(F, 730, ground, 290, h, reveal=reveal, glow=w_table, a=a)
    cx_cat = 300
    if sleep < 1:
        cat_plate(F, cx_cat, ground, 360, "sit", reveal=reveal, glow=w_cat, a=a * (1 - sleep))
    if sleep > 0:
        cat_plate(F, cx_cat, ground, 360, "sleep", reveal=1.0, glow=w_cat, a=a * sleep)
    cols = [IVORY] * 9
    cols[IT] = mix(IVORY, GOLD, it_on)
    if w_cat > 0.05:
        cols[CAT] = mix(IVORY, GOLD, 0.6 * w_cat * link)
    if w_table > 0.05:
        cols[TABLE] = mix(IVORY, GOLD, 0.6 * w_table * link)
    OPEN.draw(F, a, cols=cols, swap=(1, 4, "高", swap))
    if it_on > 0:
        add_sprite(F, *OPEN.centers[IT], 50, GOLD, 0.22 * it_on * a)
    for k, w in ((CAT, w_cat), (TABLE, w_table)):
        if w * link > 0.02:
            p = OPEN.centers[IT] + [0, -OPEN.size * 0.62]
            q = OPEN.centers[k] + [0, OPEN.size * 0.55]
            thread(F, OPEN.centers[IT] + [-OPEN.size * 0.45, 0], OPEN.centers[k] + [0, OPEN.size * 0.62],
                   0.9 * w * link * a, bow=-50)
            # and from the word up to the figure
            top = OPEN.centers[k] + [0, -OPEN.size * 0.7]
            tgt = np.array([cx_cat + 40, ground - 165]) if k == CAT else np.array([730, ground - 290 * (0.86 * h + 0.14) + 30])
            n = int(np.linalg.norm(tgt - top) / 13)
            for i in range(1, n):
                pp = top + (tgt - top) * i / n
                add_sprite(F, pp[0], pp[1], 1.5, GOLD, 0.5 * w * link * a)


# ------------------------------------------------- who looks at whom (T2T) ---

def head_view(F, w, src=IT, a=1.0, grow=1.0, colors=None, multi=None, title=True):
    """Two columns of the same words; lines from one word on the right to every word on the left,
    as thick and as gold as its attention.  multi: list of (weights, colour) for several heads."""
    n = len(TOKENS)
    y0, dy = 430, 96
    xl, xr = 380, 700
    size = 52
    for i, tok in enumerate(TOKENS):
        y = y0 + dy * i
        lw = serif(tok, size, 400, 0.1)
        rw = serif(tok, size, 400, 0.1)
        if multi is None:
            v = w[i] / w.max()
            if v > 0.08:
                box = np.full((int(size * 1.15), int(lw.width + 28)), 1.0, np.float32)
                E.add_light(F, box, xl - lw.width - 14, y - size * 0.95, GOLD, 0.10 * v ** 1.2 * a * grow)
            lc = mix(IVORY, GOLD, smooth(v)) if v > 0.5 else IVORY
            la = 0.45 + 0.55 * v
        else:
            lc, la = IVORY, 0.85
        lw.draw(F, xl, y, lc, la * a, align="right")
        rc = GOLD if i == src else IVORY
        ra = 1.0 if i == src else 0.32
        rw.draw(F, xr, y, rc, ra * a, align="left")
        if i == src:
            box = np.full((int(size * 1.15), int(rw.width + 28)), 1.0, np.float32)
            E.add_light(F, box, xr - 14, y - size * 0.95, GOLD, 0.12 * a)
    ys = y0 + dy * src - size * 0.36
    heads = multi if multi is not None else [(w, None)]
    for hk, (ww, hc) in enumerate(heads):
        top = ww.max()
        for i in range(n):
            v = ww[i] / top
            if v < 0.04:
                continue
            p = np.array([xr - 16, ys])
            q = np.array([xl + 16, y0 + dy * i - size * 0.36])
            if multi is not None:
                p = p + [0, (hk - (len(heads) - 1) / 2) * 3.0]
            u = np.linspace(0, 1, 40)[:, None]
            c1, c2 = p + [-(xr - xl) * 0.42, 0], q + [(xr - xl) * 0.42, 0]
            pts = (1 - u) ** 3 * p + 3 * (1 - u) ** 2 * u * c1 + 3 * (1 - u) * u ** 2 * c2 + u ** 3 * q
            k = max(2, int(len(pts) * grow))
            col = hc if hc is not None else mix(STEEL, GOLD, smooth(v))
            th = 1 if v < 0.35 else (2 if v < 0.7 else 3)
            glow_poly(F, pts[:k], col, (0.18 + 0.82 * v ** 1.1) * a, th=th, glow=0.5 + 0.9 * v, sigma=3 + 3 * v)
    if title:
        caps("WHO DOES  “IT”  LOOK AT", 22, 0.42).draw(F, CX, 330, DIM, 0.9 * a, align="center")


# ------------------------------------------------- the table (3B1B grid) ---

def grid(F, M, a=1.0, on=1.0, row=None, x0=None, y0=420, cell=86):
    """Rows ask, columns answer; each answer is a dot that grows with its score."""
    n = len(TOKENS)
    x0 = CX - cell * n / 2 + 46 if x0 is None else x0
    # column heads written downwards, two characters stacked
    for j, tok in enumerate(TOKENS):
        for c, ch in enumerate(tok):
            serif(ch, 30, 500, 0).draw(F, x0 + cell * j + cell / 2, y0 - 26 - 34 * (len(tok) - 1 - c), IVORY,
                                       0.75 * a, align="center")
    for i, tok in enumerate(TOKENS):
        col = GOLD if i == row else IVORY
        serif(tok, 32, 500, 0.06).draw(F, x0 - 18, y0 + cell * i + cell * 0.64, col, (1.0 if i == row else 0.75) * a,
                                       align="right")
    st = E.Stroke(x0 - 2, y0 - 2, x0 + cell * n + 3, y0 + cell * n + 3)
    for k in range(n + 1):
        st.line((x0, y0 + cell * k), (x0 + cell * n, y0 + cell * k))
        st.line((x0 + cell * k, y0), (x0 + cell * k, y0 + cell * n))
    st.light(F, STEEL, 0.10 * a)
    vmax = M.max()
    for i in range(n):
        for j in range(n):
            v = M[i, j] / vmax
            cxp, cyp = x0 + cell * (j + 0.5), y0 + cell * (i + 0.5)
            r = cell * 0.44 * math.sqrt(v) * on
            if r < 1.0:
                continue
            hot = row is not None and i == row
            col = GOLD if (hot or v > 0.8) else mix(STEEL, IVORY, 0.5)
            s2 = E.Stroke(cxp - r - 3, cyp - r - 3, cxp + r + 4, cyp + r + 4)
            s2.circle((cxp, cyp), r, th=-1)
            s2.light(F, col, (0.55 + 0.35 * v) * a * (1.0 if (row is None or hot) else 0.45))
            if hot and v > 0.6:
                add_sprite(F, cxp, cyp, r * 1.3, GOLD, 0.25 * a)
    if row is not None:
        s3 = E.Stroke(x0 - 6, y0 + cell * row - 6, x0 + cell * n + 7, y0 + cell * (row + 1) + 7)
        s3.poly([(x0 - 3, y0 + cell * row - 3), (x0 + cell * n + 3, y0 + cell * row - 3),
                 (x0 + cell * n + 3, y0 + cell * (row + 1) + 3), (x0 - 3, y0 + cell * (row + 1) + 3)], closed=True)
        s3.glow(F, GOLD, 0.4 * a, 4)
        s3.light(F, GOLD, 0.8 * a)


# --------------------------------------------------- the telephone game ---

WHISPER_X = [170, 400, 630, 860]
WHISPER_HAIR = ("bun", "short", "long", "curl")


def whisper_row(F, y=560, s=165, a=1.0, lit=(1, 1, 1, 1)):
    busts = [A.bust(x, y, s, hair=h) for x, h in zip(WHISPER_X, WHISPER_HAIR)]
    add_sprite(F, CX, y + 10, 330, WARM, 0.07 * a)
    for b, l in zip(busts, lit):
        A.backlight(F, b["ear"][0] + 4, b["ear"][1] - 4, s * 0.75, WARM, 1.7 * a * l)
        A.silhouette(F, b["outline"], a=a, rim=RIM, rim_a=0.9 * l, fade=(y + s * 0.55, y + s * 1.1),
                     body=SIL)
    return busts


def message(F, busts, k, q, a=1.0):
    """The whisper travelling from bust k's lips to bust k+1's ear; it dims and blurs as it goes."""
    import cv2
    p0 = busts[k]["mouth"] + [10, 0]
    p1 = busts[k + 1]["ear"]
    mid = (p0 + p1) / 2 + [0, -80]
    u = q
    p = (1 - u) ** 2 * p0 + 2 * (1 - u) * u * mid + u ** 2 * p1
    strength = 0.62 ** (k + q)
    col = mix(STEEL, GOLD, strength)
    pts = [(1 - v) ** 2 * p0 + 2 * (1 - v) * v * mid + v ** 2 * p1 for v in np.linspace(0, u, 30)]
    for i, pp in enumerate(pts[:-1]):
        add_sprite(F, pp[0], pp[1], 1.3, col, 0.5 * a * strength * (i / len(pts)))
    orb(F, p[0], p[1], 5 + 5 * (1 - strength), col, (0.5 + 0.8 * strength) * a)
    msg = serif("小猫", 32, 500, 0.15)
    sig = 0.4 + 3.4 * (1 - strength)
    pad = int(sig * 3) + 2
    for m, dx, dy in msg.items:
        mm = np.zeros((m.shape[0] + 2 * pad, m.shape[1] + 2 * pad), np.float32)
        mm[pad:-pad, pad:-pad] = m / 255.0
        mm = cv2.GaussianBlur(mm, (0, 0), sig)
        blend(F, mm, p[0] - msg.width / 2 + dx - pad, p[1] - 44 + dy - pad, col, (0.35 + 0.65 * strength) * a)


# ----------------------------------------------------------- the ending ---

SKY = [(150, 330, .7), (260, 420, 1.0), (390, 300, .6), (520, 380, .9), (640, 290, .7), (770, 360, 1.0),
       (880, 280, .6), (930, 450, .8), (820, 520, .7), (690, 470, .6), (560, 540, 1.0), (430, 500, .7),
       (300, 560, .8), (190, 520, .6), (980, 620, .5), (110, 650, .5)]
SKY_LINES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (5, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12),
             (12, 13), (1, 13), (3, 11), (9, 4), (7, 14), (13, 15)]


def stargazer(F, a=1.0, sky=1.0, tilt=0.42):
    A.milky_way(F, a * sky ** 0.5, p0=(-60, 1480), p1=(1140, 160), width=260)
    b = A.bust(380, 1080, 330, hair="bun", tilt=tilt)
    A.backlight(F, 430, 1010, 260, WARM, 1.2 * a)
    A.silhouette(F, b["outline"], a=a, rim=RIM, rim_a=1.0, fade=(1080 + 190, 1080 + 300), body=SIL)
    A.chart(F, SKY, SKY_LINES, t=sky, t0=0, dur=1, color=mix(GOLD, IVORY, 0.35), a=a, size=0.9)
    return b
