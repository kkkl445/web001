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
from style import (CX, DIM, GOLD, IVORY, E, T, add_sprite, blend, caps, glow_poly, hairline, mix, orb, ramp, zh,
                   smooth)

STEEL = G.STEEL
WARM = mix(GOLD, E.rgb("c98a4a"), 0.5)
RIM = mix(GOLD, IVORY, 0.45)
INK = np.array([0.05, 0.07, 0.11], np.float32)
SIL = np.array([0.012, 0.016, 0.026], np.float32)

TOKENS = ["小猫", "没有", "跳上", "桌子", "因为", "它", "太", "累", "了"]
IT, CAT, TABLE = 5, 0, 3

_CACHE = {}


def cat_figure(pose, x, ground, s):
    key = ("cat", pose, round(x, 1), round(ground, 1), round(s, 1))
    if key not in _CACHE:
        if pose == "sit":
            _CACHE[key] = A.figure(A.CAT_SIT, A.CAT_SIT_TAIL, x, ground, s)
        else:
            _CACHE[key] = A.figure(A.CAT_SLEEP, A.CAT_SLEEP_TAIL, x, ground, s, w0=0.065, w1=0.03)
    return _CACHE[key]


def bust(x, y, s, hair, tilt=0.0):
    key = ("bust", round(x, 1), round(y, 1), round(s, 1), hair, round(tilt, 3))
    if key not in _CACHE:
        if len(_CACHE) > 400:
            _CACHE.clear()
        _CACHE[key] = A.bust(x, y, s, hair=hair, tilt=tilt)
    return _CACHE[key]


# ----------------------------------------------------------- the opening ---

def cat_plate(F, x, ground, s, pose="sit", reveal=1.0, glow=0.0, a=1.0, chart=1.0, t=1e9):
    """The cat as a star-atlas figure: stars, then the lines between them, then the engraving."""
    fig = cat_figure(pose, x, ground, s)
    stars, lines = (A.CAT_SIT_STARS, A.CAT_SIT_LINES) if pose == "sit" else (A.CAT_SLEEP_STARS, A.CAT_SLEEP_LINES)
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
        A.chart(F, A.stars_on(stars, x, ground, s), lines, t=chart, t0=0.0, dur=1.0, color=mix(col, IVORY, 0.3),
                a=0.75 * a, glow=glow, size=0.8)
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
        self.txt = [zh(l, size, 400, tracking) for l in lines]
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

    def draw(self, F, a=1.0, cols=None, alphas=None, swap=None, char_alpha=None):
        n = 0
        for ln, tx in enumerate(self.txt):
            for i, (m, dx, dy) in enumerate(tx.items):
                k = self.char_tok.get((ln, i))
                col = cols[k] if (cols is not None and k is not None) else IVORY
                al = a * (alphas[k] if (alphas is not None and k is not None) else 1.0)
                lift = 0.0
                if char_alpha is not None:
                    ca = char_alpha(n)
                    al *= ca
                    lift = 6 * (1 - ca)
                n += 1
                if al <= 0.003:
                    continue
                x, y = self.x0[ln] + dx, self.ys[ln] + dy + lift
                if swap is not None and swap[0] == ln and swap[1] == i and swap[3] > 0:
                    sw = swap[3]
                    blend(F, m, x, y - 18 * sw, col, al * (1 - sw))
                    alt = zh(swap[2], self.size, 400, 0).items[0]
                    blend(F, alt[0], x + (m.shape[1] - alt[0].shape[1]) / 2, self.ys[ln] + alt[2] + 14 * (1 - sw),
                          mix(GOLD, col, sw ** 2), a * sw)
                else:
                    blend(F, m, x, y, col, al)


OPEN = Line2(("小猫没有跳上桌子，", "因为它太累了。"),
             [(0, 0, 2), (0, 2, 4), (0, 4, 6), (0, 6, 8), (1, 0, 2), (1, 2, 3), (1, 3, 4), (1, 4, 5), (1, 5, 6)],
             80, (1095, 1210))


def thread(F, p, q, a, col=GOLD, bow=60, n=60, width=1.6, grow=1.0):
    """A fine curved thread between two points, drawn on from p."""
    if a <= 0.003 or grow <= 0:
        return
    p, q = np.asarray(p, float), np.asarray(q, float)
    m = (p + q) / 2
    d = q - p
    nrm = np.array([d[1], -d[0]]) / (np.linalg.norm(d) + 1e-9)
    c = m + nrm * bow
    u = np.linspace(0, 1, n)[:, None]
    pts = (1 - u) ** 2 * p + 2 * (1 - u) * u * c + u ** 2 * q
    k = max(2, int(n * min(1.0, grow)))
    glow_poly(F, pts[:k], col, a, th=1 if width < 2 else 2, glow=0.8, sigma=4)
    if grow < 1:
        orb(F, pts[k - 1][0], pts[k - 1][1], 4, col, a)
    return pts


def dots(F, p, q, a, col=GOLD, step=13, grow=1.0):
    """A dotted thread from a word to the thing it names."""
    if a <= 0.003 or grow <= 0:
        return
    p, q = np.asarray(p, float), np.asarray(q, float)
    n = int(np.linalg.norm(q - p) / step)
    for i in range(1, int(n * min(1.0, grow))):
        pp = p + (q - p) * i / n
        add_sprite(F, pp[0], pp[1], 1.5, col, 0.5 * a)


GROUND = 900
CAT_X, TABLE_X = 300, 730


def opening(F, w_cat=1.0, w_table=0.0, sleep=0.0, h=1.0, it_on=1.0, link=1.0, reveal=1.0, swap=0.0, a=1.0,
            hop=0.0, typed=None, chart=1.0, t=1e9, link_table=None):
    """Plate: the cat and the Ming table above the sentence; 它 joined to whichever it means."""
    baseline(F, GROUND + 2, 100, 980, a * reveal)
    table_plate(F, TABLE_X, GROUND, 290, h, reveal=reveal, glow=w_table, a=a)
    # one pose fades out before the other fades in, so the two never show as a double image
    sit_a = 1 - smooth(ramp(sleep, 0.0, 0.5))
    sleep_a = smooth(ramp(sleep, 0.5, 0.5))
    if sit_a > 0:
        cat_plate(F, CAT_X, GROUND - 40 * hop, 360, "sit", reveal=reveal, glow=w_cat, a=a * sit_a, chart=chart)
    if sleep_a > 0:
        cat_plate(F, CAT_X, GROUND, 360, "sleep", reveal=1.0, glow=w_cat, a=a * sleep_a, chart=1.0)
    cols = [IVORY] * 9
    cols[IT] = mix(IVORY, GOLD, it_on)
    lt = link if link_table is None else link_table
    cols[CAT] = mix(IVORY, GOLD, 0.65 * w_cat * min(1.0, link))
    cols[TABLE] = mix(IVORY, GOLD, 0.65 * w_table * min(1.0, lt))
    alphas = None
    if typed is not None:                      # characters appear one by one
        alphas = typed
    OPEN.draw(F, a, cols=cols, swap=(1, 4, "高", swap), char_alpha=alphas)
    if it_on > 0:
        add_sprite(F, *OPEN.centers[IT], 50, GOLD, 0.22 * it_on * a)
    for k, w, lk in ((CAT, w_cat, link), (TABLE, w_table, lt)):
        if w * lk > 0.02:
            thread(F, OPEN.centers[IT] + [-OPEN.size * 0.45, 0], OPEN.centers[k] + [0, OPEN.size * 0.62],
                   0.9 * w * a, bow=-50 if k == CAT else -40, grow=lk)
            top = OPEN.centers[k] + [0, -OPEN.size * 0.7]
            if k == CAT:
                tgt = np.array([CAT_X + 40, GROUND - 165]) if sleep < 0.5 else np.array([CAT_X, GROUND - 95])
            else:
                tgt = np.array([TABLE_X, GROUND - 290 * (0.86 * h + 0.14) + 30])
            dots(F, top, tgt, w * a, grow=lk)


# ------------------------------------------------- who looks at whom (T2T) ---

COL_Y0, COL_DY = 450, 94
COL_XL, COL_XR = 380, 700
COL_SIZE = 52


def col_y(i):
    return COL_Y0 + COL_DY * i - COL_SIZE * 0.36


def columns(F, a=1.0, left_w=None, right_on=None, right_a=0.32, left_a=0.85, xl=COL_XL, xr=COL_XR, appear=1.0,
            tokens=TOKENS, y0=COL_Y0, dy=COL_DY):
    """Two columns of the same words, as in the Tensor2Tensor attention view.  left_w highlights the
    words being looked at (gold boxes as strong as the weight); right_on marks the word that looks."""
    for i, tok in enumerate(tokens):
        y = y0 + dy * i
        ap = a * smooth(ramp(appear, 0.0, 1.0))
        lw = zh(tok, COL_SIZE, 400, 0.1)
        rw = lw
        if left_w is not None:
            v = left_w[i] / max(left_w.max(), 1e-9)
            if v > 0.08:
                box = np.full((int(COL_SIZE * 1.15), int(lw.width + 28)), 1.0, np.float32)
                E.add_light(F, box, xl - lw.width - 14, y - COL_SIZE * 0.95, GOLD, 0.10 * v ** 1.2 * ap)
            lc = mix(IVORY, GOLD, smooth((v - 0.5) / 0.5)) if v > 0.5 else IVORY
            la = 0.45 + 0.55 * v
        else:
            lc, la = IVORY, left_a
        lw.draw(F, xl, y, lc, la * ap, align="right")
        on = 0.0 if right_on is None else right_on[i]
        rw.draw(F, xr, y, mix(IVORY, GOLD, on), (right_a + (1 - right_a) * on) * ap, align="left")
        if on > 0.01:
            box = np.full((int(COL_SIZE * 1.15), int(rw.width + 28)), 1.0, np.float32)
            E.add_light(F, box, xr - 14, y - COL_SIZE * 0.95, GOLD, 0.12 * on * ap)


def fan_lines(F, src, w, a=1.0, grow=1.0, col=None, xl=COL_XL, xr=COL_XR, jitter=0.0, floor=0.04, gold=True):
    """Lines from word src on the right to every word on the left, thick and gold as its weight."""
    if a <= 0.003 or grow <= 0:
        return
    top = w.max()
    p = np.array([xr - 16, col_y(src) + jitter])
    u = np.linspace(0, 1, 40)[:, None]
    for i in range(len(w)):
        v = w[i] / top
        if v < floor:
            continue
        q = np.array([xl + 16, col_y(i)])
        c1, c2 = p + [-(xr - xl) * 0.42, 0], q + [(xr - xl) * 0.42, 0]
        pts = (1 - u) ** 3 * p + 3 * (1 - u) ** 2 * u * c1 + 3 * (1 - u) * u ** 2 * c2 + u ** 3 * q
        k = max(2, int(len(pts) * min(1.0, grow)))
        c = col if col is not None else (mix(STEEL, GOLD, smooth(v)) if gold else STEEL)
        th = 1 if v < 0.35 else (2 if v < 0.7 else 3)
        glow_poly(F, pts[:k], c, (0.18 + 0.82 * v ** 1.1) * a, th=th, glow=0.5 + 0.9 * v, sigma=3 + 3 * v)


def head_view(F, w, src=IT, a=1.0, grow=1.0, title=True):
    columns(F, a, left_w=w, right_on=np.eye(len(TOKENS))[src])
    fan_lines(F, src, w, a, grow)
    if title:
        caps("WHO DOES  “IT”  LOOK AT", 22, 0.42).draw(F, CX, 330, DIM, 0.9 * a, align="center")


# ------------------------------------------------- the table (3B1B grid) ---

def grid(F, M, a=1.0, on=1.0, row=None, x0=None, y0=420, cell=86, row_a=1.0):
    """Rows ask, columns answer; each answer is a dot that grows with its score."""
    n = len(TOKENS)
    x0 = CX - cell * n / 2 + 46 if x0 is None else x0
    # column heads written downwards, two characters stacked
    for j, tok in enumerate(TOKENS):
        for c, ch in enumerate(tok):
            zh(ch, 30, 500, 0).draw(F, x0 + cell * j + cell / 2, y0 - 26 - 34 * (len(tok) - 1 - c), IVORY,
                                       0.75 * a, align="center")
    for i, tok in enumerate(TOKENS):
        col = GOLD if i == row else IVORY
        zh(tok, 32, 500, 0.06).draw(F, x0 - 18, y0 + cell * i + cell * 0.64, col, (1.0 if i == row else 0.75) * a,
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
            dim = 1.0 if row is None else (1.0 if hot else 1 - 0.55 * row_a)
            s2.light(F, col if (hot or row is None) else mix(STEEL, IVORY, 0.5), (0.55 + 0.35 * v) * a * dim)
            if hot and v > 0.6:
                add_sprite(F, cxp, cyp, r * 1.3, GOLD, 0.25 * a * row_a)
    if row is not None and row_a > 0.01:
        a = a * row_a
        s3 = E.Stroke(x0 - 6, y0 + cell * row - 6, x0 + cell * n + 7, y0 + cell * (row + 1) + 7)
        s3.poly([(x0 - 3, y0 + cell * row - 3), (x0 + cell * n + 3, y0 + cell * row - 3),
                 (x0 + cell * n + 3, y0 + cell * (row + 1) + 3), (x0 - 3, y0 + cell * (row + 1) + 3)], closed=True)
        s3.glow(F, GOLD, 0.4 * a, 4)
        s3.light(F, GOLD, 0.8 * a)


# --------------------------------------------------- the telephone game ---

WHISPER_X = [170, 400, 630, 860]
WHISPER_HAIR = ("bun", "short", "long", "curl")


def whisper_row(F, y=560, s=165, a=1.0, lit=(1, 1, 1, 1), each=(1, 1, 1, 1)):
    busts = [bust(x, y, s, h) for x, h in zip(WHISPER_X, WHISPER_HAIR)]
    add_sprite(F, CX, y + 10, 330, WARM, 0.07 * a)
    for b, l, e in zip(busts, lit, each):
        if a * e <= 0.003:
            continue
        A.backlight(F, b["ear"][0] + 4, b["ear"][1] - 4, s * 0.75, WARM, 1.7 * a * e * l)
        A.silhouette(F, b["outline"], a=a * e, rim=RIM, rim_a=0.9 * l, fade=(y + s * 0.55, y + s * 1.1),
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
    msg = zh("小猫", 32, 500, 0.15)
    sig = 0.4 + 3.4 * (1 - strength)
    pad = int(sig * 3) + 2
    for m, dx, dy in msg.items:
        mm = np.zeros((m.shape[0] + 2 * pad, m.shape[1] + 2 * pad), np.float32)
        mm[pad:-pad, pad:-pad] = m / 255.0
        mm = cv2.GaussianBlur(mm, (0, 0), sig)
        blend(F, mm, p[0] - msg.width / 2 + dx - pad, p[1] - 44 + dy - pad, col, (0.35 + 0.65 * strength) * a)


# ----------------------------------------------------------- the ending ---

SKY = [(150, 300, .7), (260, 390, 1.0), (390, 270, .6), (520, 350, .9), (640, 260, .7), (770, 330, 1.0),
       (880, 250, .6), (930, 420, .8), (820, 490, .7), (690, 440, .6), (560, 510, 1.0), (430, 470, .7),
       (300, 530, .8), (190, 490, .6), (980, 590, .5), (110, 620, .5)]
SKY_LINES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (5, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12),
             (12, 13), (1, 13), (3, 11), (9, 4), (7, 14), (13, 15)]


def stargazer(F, a=1.0, sky=1.0, tilt=0.42, chart=1.0, figure=1.0, y=1150):
    A.milky_way(F, a * sky, p0=(-60, 1480), p1=(1140, 160), width=260)
    b = bust(380, y, 330, "bun", tilt)
    A.backlight(F, 430, y - 70, 260, WARM, 1.2 * a * figure)
    A.silhouette(F, b["outline"], a=a * figure, rim=RIM, rim_a=1.0, fade=(y + 190, y + 300), body=SIL)
    if chart > 0:
        A.chart(F, SKY, SKY_LINES, t=chart, t0=0, dur=1, color=mix(GOLD, IVORY, 0.35), a=a, size=0.9)
    return b


# ------------------------------------------------------------ Q, K and V ---
# 它 on the right holds a question; the eight other words on the left each
# hold a label (K) and a content (V).  Lines run from the question to the labels.

QKV_TOK = ["小猫", "没有", "跳上", "桌子", "因为", "太", "累", "了"]
QKV_KEY = ["动物", "否定", "动作", "物件", "原因", "程度", "状态", "语气"]
QKV_MATCH = np.array([0.62, 0.03, 0.05, 0.12, 0.03, 0.04, 0.08, 0.03])
Q_Y0, Q_DY = 430, 102
Q_XW, Q_XK, Q_XV = 230, 262, 520
IT_POS = np.array([830, 830])
Q_CARD = np.array([830, 680])


def q_row_y(i):
    return Q_Y0 + Q_DY * i


def _rrect(F, x0, y0, x1, y1, col, a, r=10, glow=0.3, fill=0.0):
    if a <= 0.003:
        return
    st = E.Stroke(x0 - 3, y0 - 3, x1 + 4, y1 + 4)
    pts = []
    for cx_, cy_, a0 in ((x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)):
        for k in range(7):
            ang = math.radians(a0 + 90 * k / 6)
            pts.append((cx_ + r * math.cos(ang), cy_ + r * math.sin(ang)))
    if fill > 0:
        f = E.Stroke(x0 - 3, y0 - 3, x1 + 4, y1 + 4)
        f.fill(pts)
        f.light(F, col, fill * a)
    st.poly(pts, closed=True)
    if glow > 0:
        st.glow(F, col, glow * a, 4)
    st.light(F, col, a)


def qkv_plate(F, words=1.0, q=0.0, k=None, v=None, match=0.0, flow=0.0, it_a=1.0, a=1.0, pull=0.0):
    """k, v: per-row appear values (0..1).  match: the question meets each label.  flow: contents travel to 它."""
    if a <= 0.003:
        return
    k = np.zeros(8) if k is None else k
    v = np.zeros(8) if v is None else v
    rel = QKV_MATCH / QKV_MATCH.max()
    for i, tok in enumerate(QKV_TOK):
        y = q_row_y(i)
        g = smooth(ramp(match, 0.3, 0.7)) * rel[i] ** 1.5
        zh(tok, 50, 400, 0.08).draw(F, Q_XW, y, mix(IVORY, GOLD, g), (0.85 + 0.15 * g) * a * words, align="right")
        if k[i] > 0:
            ka = a * smooth(k[i])
            x0, x1 = Q_XK, Q_XK + 168
            _rrect(F, x0, y - 44, x1, y + 6, mix(STEEL, GOLD, g), 0.55 * ka, r=8, glow=0.2 + 0.6 * g,
                   fill=0.05 + 0.08 * g)
            T("K", "stix_it", 30, None, per_char=False).draw(F, x0 + 26, y - 7, mix(STEEL, GOLD, g), ka,
                                                             align="center")
            zh(QKV_KEY[i], 28, 500, 0.1).draw(F, x0 + 100, y - 8, mix(IVORY, GOLD, g), 0.9 * ka, align="center")
        if v[i] > 0:
            va = a * smooth(v[i])
            add_sprite(F, Q_XV, y - 19, 18, mix(STEEL, GOLD, 0.5), 0.10 * va)
            orb(F, Q_XV, y - 19, 4.5, mix(STEEL, GOLD, 0.5), 0.9 * va)
            T("V", "stix_it", 22, None, per_char=False).draw(F, Q_XV + 22, y - 10, STEEL, 0.75 * va)
    # 它 and its question
    it_t = zh("它", 76, 500, 0)
    it_t.draw(F, IT_POS[0], IT_POS[1] + 26, GOLD, a * it_a, align="center")
    add_sprite(F, IT_POS[0], IT_POS[1], 60, GOLD, 0.18 * a * it_a)
    if q > 0:
        qa = a * smooth(q)
        cx_, cy_ = Q_CARD
        _rrect(F, cx_ - 100, cy_ - 46, cx_ + 100, cy_ + 46, GOLD, 0.85 * qa, r=12, glow=0.5, fill=0.08)
        T("Q", "stix_it", 40, None, per_char=False).draw(F, cx_ - 62, cy_ + 14, GOLD, qa, align="center")
        zh("我指谁？", 30, 500, 0.1).draw(F, cx_ + 26, cy_ + 11, IVORY, qa, align="center")
        hairline(F, cx_ - 1, int(cy_ + 46), cx_ + 1, GOLD, 0.0)
    # matching beams: question -> labels
    if match > 0:
        p = Q_CARD + [-100, 0]
        u = np.linspace(0, 1, 40)[:, None]
        for i in range(8):
            vv = rel[i]
            q_ = np.array([Q_XK + 168 + 6, q_row_y(i) - 19])
            c1, c2 = p + [-150, 0], q_ + [150, 0]
            pts = (1 - u) ** 3 * p + 3 * (1 - u) ** 2 * u * c1 + 3 * (1 - u) * u ** 2 * c2 + u ** 3 * q_
            gk = max(2, int(len(pts) * min(1.0, ramp(match, 0.03 * i, 0.45))))
            glow_poly(F, pts[:gk], mix(STEEL, GOLD, smooth(vv)), (0.14 + 0.86 * vv ** 1.2) * a,
                      th=1 if vv < 0.4 else 3, glow=0.4 + vv, sigma=3 + 3 * vv)
        pa = smooth(ramp(match, 0.55, 0.3)) * a
        T("62%", "stix", 32, None, per_char=False).draw(F, Q_XK + 260, q_row_y(0) - 40, GOLD, pa, align="center")
    # contents flow to 它, as much as each label matched
    if flow > 0:
        for i in range(8):
            vv = QKV_MATCH[i]
            uu = ease_in_out_(ramp(flow, 0.12 * i, 0.4))
            if uu <= 0 or uu >= 1:
                continue
            p0 = np.array([Q_XV, q_row_y(i) - 19])
            p1 = IT_POS + [-30, 0]
            mid = (p0 + p1) / 2 + [0, 60]
            pp = (1 - uu) ** 2 * p0 + 2 * (1 - uu) * uu * mid + uu ** 2 * p1
            orb(F, pp[0], pp[1], 3 + 10 * vv, mix(STEEL, GOLD, vv / 0.62), (0.3 + 1.2 * vv) * a)
        got = smooth(ramp(flow, 0.6, 0.4)) * a
        if got > 0:
            add_sprite(F, IT_POS[0], IT_POS[1], 90, GOLD, 0.25 * got)
            zh("≈ 小猫", 34, 500, 0.1).draw(F, IT_POS[0], IT_POS[1] + 110, GOLD, got, align="center")


def ease_in_out_(u):
    return u * u * (3 - 2 * u)


# ----------------------------------------------------------- eight heads ---

HEAD_COLS = [GOLD, E.rgb("72d6ff"), E.rgb("ff9a3c"), E.rgb("a99bff"), E.rgb("6fe3c8"), E.rgb("ff8fb1"), IVORY,
             E.rgb("7fa8ff")]


def head_squares(F, on, a=1.0, y=420, labels=None, merge=0.0):
    """The row of coloured squares from the Tensor2Tensor view: one per attention head."""
    n = len(HEAD_COLS)
    sz, gap = 40, 18
    x0 = CX - (n * sz + (n - 1) * gap) / 2
    for k in range(n):
        x = x0 + k * (sz + gap)
        col = mix(HEAD_COLS[k], GOLD, merge)
        box = np.full((sz, sz), 1.0, np.float32)
        E.add_light(F, box, x, y, col, (0.10 + 0.45 * on[k]) * a)
        _rrect(F, x, y, x + sz, y + sz, col, (0.35 + 0.65 * on[k]) * a, r=4, glow=0.6 * on[k])


def links_lines(F, links, col, a=1.0, grow=1.0):
    """Head patterns: lines from word i (right column) to word j (left column)."""
    if a <= 0.003 or grow <= 0:
        return
    u = np.linspace(0, 1, 36)[:, None]
    for i, j, v in links:
        p = np.array([COL_XR - 16, col_y(i)])
        q = np.array([COL_XL + 16, col_y(j)])
        c1, c2 = p + [-(COL_XR - COL_XL) * 0.42, 0], q + [(COL_XR - COL_XL) * 0.42, 0]
        pts = (1 - u) ** 3 * p + 3 * (1 - u) ** 2 * u * c1 + 3 * (1 - u) * u ** 2 * c2 + u ** 3 * q
        k = max(2, int(len(pts) * min(1.0, grow)))
        glow_poly(F, pts[:k], col, a * (0.35 + 0.65 * v), th=2 if v > 0.8 else 1, glow=0.6 + 0.6 * v, sigma=4)


# ------------------------------------------------------- the first page ---

AUTHORS = [("Ashish Vaswani", "Google Brain"), ("Noam Shazeer", "Google Brain"), ("Niki Parmar", "Google Research"),
           ("Jakob Uszkoreit", "Google Research"), ("Llion Jones", "Google Research"),
           ("Aidan N. Gomez", "University of Toronto"), ("Łukasz Kaiser", "Google Brain"),
           ("Illia Polosukhin", "")]
PAGE = (180, 290, 900, 1222)
PAPER = E.rgb("e9e2d0")
INKC = np.array([0.10, 0.10, 0.12], np.float32)


_PAGE_CACHE = {}


def _page_maps():
    if "m" not in _PAGE_CACHE:
        import cv2
        x0, y0, x1, y1 = PAGE
        w, h = x1 - x0, y1 - y0
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        lamp = 0.62 + 0.38 * np.exp(-(((xx - w * 0.35) / (w * 0.9)) ** 2 + ((yy - h * 0.25) / (h * 0.9)) ** 2))
        grain = np.random.default_rng(3).normal(0, 0.012, (h, w)).astype(np.float32)
        pad = 50
        sh = np.zeros((h + 2 * pad, w + 2 * pad), np.float32)
        sh[pad + 14:pad + h + 14, pad + 8:pad + w + 8] = 1
        sh = cv2.GaussianBlur(sh, (0, 0), 16)
        _PAGE_CACHE["m"] = (lamp + grain, sh, pad)
    return _PAGE_CACHE["m"]


def page(F, a=1.0, note=0.0, dim=0.0):
    """The paper's first page, typeset after the original, lit like paper under a lamp."""
    if a <= 0.003:
        return
    x0, y0, x1, y1 = PAGE
    w = x1 - x0
    lamp, sh, pad = _page_maps()
    F[y0 - pad:y1 + pad, x0 - pad:x1 + pad] *= (1 - 0.65 * a * sh)[..., None]
    reg = F[y0:y1, x0:x1]
    reg[:] = reg * (1 - a) + (PAPER * 0.72 * (1 - 0.5 * dim) * lamp[..., None]) * a
    ink = INKC
    ia = a
    T("Attention Is All You Need", "stix", 42, None, per_char=False).draw(F, CX, y0 + 120, ink, ia, align="center")
    for k, (nm, af) in enumerate(AUTHORS):
        row, col = divmod(k, 4)
        x = x0 + w * (0.14 + 0.24 * col)
        y = y0 + 210 + 96 * row
        T(nm + "\u2217", "stix", 17, None, per_char=False).draw(F, x, y, ink, ia, align="center")
        if af:
            T(af, "stix", 14, None, per_char=False).draw(F, x, y + 22, ink, 0.75 * ia, align="center")
        hairline(F, x - 42, int(y + 36), x + 42, ink, 0.25 * ia)
    T("Abstract", "stix", 21, None, per_char=False).draw(F, CX, y0 + 470, ink, ia, align="center")
    rng = np.random.default_rng(5)
    for k in range(12):
        lx0, lx1 = x0 + 110, x1 - 110
        if k == 11:
            lx1 = lx0 + (lx1 - lx0) * 0.55
        xx_ = lx0
        while xx_ < lx1:
            wd = rng.uniform(18, 70)
            hairline(F, xx_, int(y0 + 505 + 25 * k), min(xx_ + wd, lx1), ink, 0.28 * ia, th=4)
            xx_ += wd + 7
    hairline(F, x0 + 60, int(y1 - 168), x0 + 260, ink, 0.6 * ia)
    fn = T("\u2217Equal contribution. Listing order is random.", "stix_it", 25, None, per_char=False)
    if note > 0:
        hw = int((fn.width + 20) * min(1.0, note))
        if hw > 1:
            blend(F, np.full((36, hw), 1.0, np.float32), x0 + 50, y1 - 152, GOLD, 0.5 * a)
    fn.draw(F, x0 + 60, y1 - 124, ink, ia)
    for k in range(3):
        hairline(F, x0 + 60, int(y1 - 94 + 18 * k), x1 - 60 - 160 * (k == 2), ink, 0.2 * ia, th=3)


# ------------------------------------------------------------- the song ---

def record(F, cx, cy, r, t, a=1.0):
    """A vinyl record turning: black disc, fine grooves, an ivory label, the sheen that stays put."""
    if a <= 0.003:
        return
    s = int(r * 2 + 8)
    yy, xx = np.mgrid[0:s, 0:s].astype(np.float32)
    dx, dy = xx - s / 2, yy - s / 2
    rr = np.sqrt(dx * dx + dy * dy)
    ang = np.arctan2(dy, dx)
    disc = np.clip(r - rr + 0.5, 0, 1)
    groove = (0.5 + 0.5 * np.cos(rr * 1.9)) * (rr > r * 0.36) * (rr < r * 0.97)
    sheen = (np.cos(2 * (ang + 0.6)) ** 16) * (rr > r * 0.36) * (rr < r)
    sub = F[int(cy - s / 2):int(cy - s / 2) + s, int(cx - s / 2):int(cx - s / 2) + s]
    base = np.array([0.012, 0.013, 0.016], np.float32)
    m = (disc * a)[..., None]
    sub[:] = sub * (1 - m) + base * m
    E.add_light(F, (groove * 0.025 + sheen * 0.10).astype(np.float32) * disc, cx - s / 2, cy - s / 2, IVORY, a)
    rim = np.exp(-((rr - r) / 1.2) ** 2).astype(np.float32)
    E.add_light(F, rim, cx - s / 2, cy - s / 2, mix(GOLD, IVORY, 0.4), 0.35 * a)
    lab = np.clip(r * 0.33 - rr + 0.5, 0, 1).astype(np.float32)
    E.add_light(F, lab, cx - s / 2, cy - s / 2, mix(GOLD, E.rgb("8a4b2a"), 0.35), 0.55 * a)
    hole = np.clip(4 - rr + 0.5, 0, 1)[..., None] * a
    sub[:] = sub * (1 - hole) + base * hole
    # the label's text turns with the record
    rot = -t * 0.9
    for k, ch in enumerate("ALL YOU NEED IS LOVE"):
        th = rot + k * 0.24
        x = cx + r * 0.24 * math.cos(th)
        y = cy + r * 0.24 * math.sin(th)
        T(ch, "corm", 11, 600, per_char=False).draw(F, x, y + 4, INKC, 0.8 * a, align="center")


# ------------------------------------------------------ the layer stack ---

def layer(F, cx, y, w, a=1.0, detail=1.0, glow=0.0, h=170):
    """One Transformer layer after Figure 1 of the paper, bottom to top: multi-head attention,
    add & norm, feed forward, add & norm."""
    if a <= 0.003:
        return
    col = mix(mix(STEEL, IVORY, 0.4), GOLD, glow)
    _rrect(F, cx - w / 2, y - h, cx + w / 2, y, col, 0.75 * a, r=12, glow=0.25 + 0.6 * glow, fill=0.03 + 0.05 * glow)
    if detail <= 0.01:
        return
    d = a * detail
    bx0, bx1 = cx - w / 2 + 28, cx + w / 2 - 28
    rows = [(y - 14, 34, "Multi-Head Attention", GOLD), (y - 56, 18, "Add & Norm", mix(GOLD, IVORY, 0.6)),
            (y - 82, 34, "Feed Forward", STEEL), (y - 124, 18, "Add & Norm", mix(GOLD, IVORY, 0.6))]
    for yb, hb, name, c in rows:
        _rrect(F, bx0, yb - hb, bx1, yb, c, 0.6 * d, r=5, glow=0.2, fill=0.06)
        T(name, "stix", 15 if hb < 20 else 21, None, per_char=False).draw(F, cx, yb - hb / 2 + (5 if hb < 20 else 7),
                                                                         IVORY, 0.85 * d, align="center")


# ================================================================ v3 ===
# Pictures that carry the explanation: the next-word game, the sky of
# meaning, the dot product on a small chart, softmax as bars, the formula
# taken apart.

def bars(F, items, x0, y0, w, a=1.0, grow=1.0, row=74, size=40, hot=0, note=None):
    """Horizontal probability bars: label, bar, percentage.  items: [(label, p)]."""
    if a <= 0.003:
        return
    pmax = max(p for _, p in items)
    for i, (lab, p) in enumerate(items):
        y = y0 + row * i
        g = smooth(ramp(grow, 0.08 * i, 0.5))
        col = GOLD if i == hot else mix(STEEL, IVORY, 0.35)
        zh(lab, size, 500, 0.0).draw(F, x0, y + size * 0.36, col if i == hot else IVORY, a, align="right")
        bw = (w * p / pmax) * g
        if bw > 1:
            E.add_light(F, np.full((int(size * 0.55), int(bw)), 1.0, np.float32), x0 + 24, y - size * 0.28, col,
                        (0.55 if i == hot else 0.32) * a)
            if i == hot:
                add_sprite(F, x0 + 24 + bw, y, 18, GOLD, 0.3 * a * g)
        T(f"{round(100 * p)}%", "stix", int(size * 0.8), None, per_char=False).draw(
            F, x0 + 40 + bw, y + size * 0.3, col if i == hot else DIM, a * g)
    if note:
        zh(note, 24, 400, 0.1).draw(F, x0 + 24 + w, y0 + row * len(items), DIM, 0.8 * a, align="right")


def guess_line(F, text, x, y, size, a=1.0, typed=None, blank=True, t=0.0, fill=None, fill_a=0.0):
    """A sentence typed out, ending in a blank box with a blinking cursor (or the guessed character)."""
    tx = zh(text, size, 400, 0.12)
    n = len(tx.items)
    x0 = x - (tx.width + size * 1.1) / 2
    for i, (m, dx, dy) in enumerate(tx.items):
        al = a * (1.0 if typed is None else typed(i))
        if al > 0.003:
            blend(F, m, x0 + dx, y + dy, IVORY, al)
    if blank:
        bx = x0 + tx.width + size * 0.12
        on = 1.0 if typed is None else typed(n)
        if on > 0:
            _rrect(F, bx, y - size * 0.92, bx + size * 0.98, y + size * 0.16, GOLD, 0.7 * a * on, r=6, glow=0.4)
            if fill and fill_a > 0:
                zh(fill, size, 500, 0).draw(F, bx + size * 0.49, y, GOLD, a * fill_a, align="center")
            elif (t * 1.6) % 1 < 0.6:
                hairline(F, bx + size * 0.3, int(y + size * 0.02), bx + size * 0.7, GOLD, 0.9 * a * on, th=3)
    return x0


# ------------------------------------------------------- the sky of meaning ---

SKY_WORDS = {
    # animals
    "小猫": (230, 560), "狗": (330, 480), "老虎": (170, 450), "兔子": (150, 640),
    # furniture
    "桌子": (790, 560), "椅子": (890, 490), "床": (860, 660),
    # how one feels
    "累": (290, 1040), "困": (390, 1110), "饿": (190, 1150),
    # fruit
    "香蕉": (760, 1150), "橙子": (880, 1090),
    # phones and companies
    "手机": (730, 860), "公司": (870, 910),
    # people
    "男人": (360, 400), "女人": (580, 350), "国王": (390, 300), "女王": (610, 250),
}
SKY_GROUPS = [["小猫", "狗", "老虎", "兔子"], ["桌子", "椅子", "床"], ["累", "困", "饿"], ["香蕉", "橙子"],
              ["手机", "公司"]]
IT_HOME = np.array([520, 780])


def sky_word(F, name, p, a=1.0, glow=0.0, size=34, mag=1.0, label=True, col=None):
    if a <= 0.003:
        return
    c = col if col is not None else mix(mix(STEEL, IVORY, 0.55), GOLD, glow)
    orb(F, p[0], p[1], (3.0 + 2.0 * mag) * (1 + 0.6 * glow), c, (0.7 + 0.3 * glow) * a)
    if label:
        zh(name, size, 500, 0.05).draw(F, p[0] + 16, p[1] + size * 0.36, mix(IVORY, GOLD, glow), 0.85 * a)


_SCATTER = {w: np.array([np.random.default_rng(len(w) * 31 + i).uniform(120, 960),
                           np.random.default_rng(i * 7 + 3).uniform(300, 1200)])
            for i, w in enumerate(["小猫", "狗", "老虎", "兔子", "桌子", "椅子", "床", "累", "困", "饿", "香蕉", "橙子",
                                   "手机", "公司", "男人", "女人", "国王", "女王"])}


def sky_pos(w, scatter=0.0):
    p = np.array(SKY_WORDS[w], float)
    if scatter > 0:
        p = p + (_SCATTER[w] - p) * scatter
    return p


def sky(F, a=1.0, groups=1.0, words=None, dim=None, glow=None, nebula=1.0, scatter=0.0):
    """The sky of meaning: words as stars, neighbours in meaning close together, each family faintly joined.
    scatter 1 puts every star somewhere random (before the sky has formed)."""
    if a <= 0.003:
        return
    dim = dim or {}
    glow = glow or {}
    form = 1 - scatter
    for gk, grp in enumerate(SKY_GROUPS):
        ga = a * smooth(ramp(groups, 0.12 * gk, 0.5)) * smooth(ramp(form, 0.6, 0.4))
        if ga <= 0.003:
            continue
        pts = np.array([sky_pos(w, scatter) for w in grp], float)
        c = pts.mean(0)
        da = min(dim.get(w, 1.0) for w in grp)
        add_sprite(F, c[0], c[1], 110, mix(STEEL, GOLD, 0.3), 0.05 * ga * nebula * da)
        for i in range(len(pts) - 1):
            glow_poly(F, [pts[i], pts[i + 1]], mix(STEEL, IVORY, 0.4), 0.22 * ga * da, th=1, glow=0.3, sigma=2)
    for w in SKY_WORDS:
        if words is not None and w not in words:
            continue
        gk = next((k for k, grp in enumerate(SKY_GROUPS) if w in grp), 0)
        ga = a * smooth(ramp(groups, 0.12 * gk, 0.5)) if words is None else a
        sky_word(F, w, sky_pos(w, scatter), ga * dim.get(w, 1.0), glow=glow.get(w, 0.0))


def arrow(F, p, q, col, a=1.0, th=2, head=16, glow=0.6):
    p, q = np.asarray(p, float), np.asarray(q, float)
    d = q - p
    L = np.linalg.norm(d)
    if L < 2 or a <= 0.003:
        return
    u = d / L
    n = np.array([-u[1], u[0]])
    glow_poly(F, [p, q - u * head * 0.6], col, a, th=th, glow=glow, sigma=3)
    tip = [q, q - u * head + n * head * 0.42, q - u * head * 0.72, q - u * head - n * head * 0.42]
    st = E.Stroke(min(t[0] for t in tip) - 3, min(t[1] for t in tip) - 3, max(t[0] for t in tip) + 4,
                  max(t[1] for t in tip) + 4)
    st.fill(tip)
    st.light(F, col, a)


# ---------------------------------------------------- the dot product chart ---

DOT_O = np.array([240, 1000])         # origin of the little chart
DOT_S = 210                           # pixels per unit
Q_VEC = np.array([2.0, 0.0])
K_VECS = [("小猫", np.array([1.5, 0.5])), ("累", np.array([1.0, 0.3])), ("跳上", np.array([0.5, 1.0])),
          ("桌子", np.array([0.0, 2.0]))]


def dot_pt(v):
    return DOT_O + np.array([v[0], -v[1]]) * DOT_S


def dot_chart(F, a=1.0, q=1.0, keys=None, focus=None, axes=1.0):
    """Two little axes ('会累的' across, '是个物件' up), the question as a gold arrow, the labels as arrows."""
    if a <= 0.003:
        return
    keys = keys if keys is not None else [1.0] * len(K_VECS)
    aa = a * axes
    x1 = DOT_O[0] + 2.4 * DOT_S
    y1 = DOT_O[1] - 2.3 * DOT_S
    st = E.Stroke(DOT_O[0] - 4, y1 - 4, x1 + 4, DOT_O[1] + 4)
    st.line(DOT_O, (x1, DOT_O[1]))
    st.line(DOT_O, (DOT_O[0], y1))
    for k in range(1, 3):
        st.line((DOT_O[0] + k * DOT_S, DOT_O[1] - 6), (DOT_O[0] + k * DOT_S, DOT_O[1] + 6))
        st.line((DOT_O[0] - 6, DOT_O[1] - k * DOT_S), (DOT_O[0] + 6, DOT_O[1] - k * DOT_S))
    st.light(F, mix(STEEL, IVORY, 0.4), 0.45 * aa)
    zh("会累的", 28, 500, 0.08).draw(F, x1 + 12, DOT_O[1] + 10, DIM, aa)
    zh("是个物件", 28, 500, 0.08).draw(F, DOT_O[0], y1 - 22, DIM, aa, align="center")
    for k in range(1, 3):
        T(str(k), "stix", 22, None, per_char=False).draw(F, DOT_O[0] + k * DOT_S, DOT_O[1] + 34, DIM, aa,
                                                         align="center")
        T(str(k), "stix", 22, None, per_char=False).draw(F, DOT_O[0] - 22, DOT_O[1] - k * DOT_S + 8, DIM, aa,
                                                         align="right")
    for i, (name, v) in enumerate(K_VECS):
        ka = a * keys[i] * (1.0 if focus is None or focus == i else 0.3)
        if ka <= 0.003:
            continue
        tip = dot_pt(v)
        arrow(F, DOT_O, tip, mix(STEEL, IVORY, 0.3), ka, th=2)
        zh(f"{name}", 32, 500, 0.05).draw(F, tip[0] + 16, tip[1] - 8, IVORY, ka)
        T(f"K = ({v[0]:g}, {v[1]:g})", "stix_it", 26, None, per_char=False).draw(F, tip[0] + 16, tip[1] + 26, DIM, ka)
    if q > 0:
        tip = dot_pt(Q_VEC)
        arrow(F, DOT_O, DOT_O + (tip - DOT_O) * smooth(q), GOLD, a * smooth(q), th=3, head=20, glow=1.0)
        T("Q = (2, 0)", "stix_it", 32, None, per_char=False).draw(F, tip[0] + 40, tip[1] + 64, GOLD, a * smooth(q),
                                                                  align="center")
        zh("「它」的问题：前面哪个东西会累？", 30, 500, 0.08).draw(F, CX, DOT_O[1] + 110, GOLD, a * smooth(q),
                                                         align="center")


def dot_sum(F, y, a=1.0, which=0, reveal=1.0):
    """'2 × 1.5 + 0 × 0.5 = 3' written out term by term."""
    name, v = K_VECS[which]
    parts = [("Q · K", "stix_it", GOLD), ("  =  ", "stix", IVORY), ("2", "stix", GOLD), (" × ", "stix", IVORY),
             (f"{v[0]:g}", "stix", IVORY), ("  +  ", "stix", IVORY), ("0", "stix", GOLD), (" × ", "stix", IVORY),
             (f"{v[1]:g}", "stix", IVORY), ("  =  ", "stix", IVORY), (f"{Q_VEC @ v:g}", "stix", GOLD)]
    sz = 46
    ws = [T(p, f, sz, None, per_char=False).width for p, f, _ in parts]
    x = CX - sum(ws) / 2
    for k, ((p, f, c), w) in enumerate(zip(parts, ws)):
        al = a * smooth(ramp(reveal, k / len(parts), 1.5 / len(parts)))
        T(p, f, sz, None, per_char=False).draw(F, x, y, c, al)
        x += w
    zh(f"「它」的问题 · 「{name}」的标签", 28, 500, 0.08).draw(F, CX, y + 54, DIM, a, align="center")


SOFT_ITEMS = [("小猫", 0.571), ("累", 0.210), ("跳上", 0.077), ("桌子", 0.028), ("其余 4 个", 0.114)]
SCORES = [("小猫", 3), ("累", 2), ("跳上", 1), ("桌子", 0), ("其余 4 个", 0)]


def formula_parts(F, cx, y, a=1.0, lit=None, notes=1.0):
    """softmax(QKᵀ/√dₖ)V with each part lit in turn and its meaning written under it."""
    lit = lit or {}
    sz = 66
    pieces = [("Attention(", "stix", None), ("Q", "stix_it", None), (", ", "stix", None), ("K", "stix_it", None),
              (", ", "stix", None), ("V", "stix_it", None), (")", "stix", None)]
    w1 = sum(T(p, f, sz, None, per_char=False).width for p, f, _ in pieces)
    x = cx - w1 / 2
    for p, f, _ in pieces:
        tx = T(p, f, sz, None, per_char=False)
        tx.draw(F, x, y, IVORY, a)
        x += tx.width
    y2 = y + 120
    segs = [("=  softmax(", "stix", "soft"), ("QK", "stix_it", "qk"), ("T", "stix_sup", "qk"), (" / ", "stix", "sq"),
            ("√", "math", "sq"), ("d", "stix_it", "sq"), ("k", "stix_sub", "sq"), (")", "stix", "soft"),
            (" V", "stix_it", "v")]

    def mk(p, f):
        if f == "stix_sup":
            return T(p, "stix", 40, None, per_char=False), -30
        if f == "stix_sub":
            return T(p, "stix_it", 40, None, per_char=False), 14
        return T(p, f, sz, None, per_char=False), 0

    tot = sum(mk(p, f)[0].width for p, f, _ in segs)
    x = cx - tot / 2
    spans = {}
    for p, f, key in segs:
        tx, dy = mk(p, f)
        on = lit.get(key, 0.0)
        tx.draw(F, x, y2 + dy, mix(IVORY, GOLD, on), a)
        if key == "sq" and p == "d":
            hairline(F, x - 2, int(y2 - 60), x + tx.width + 26, mix(IVORY, GOLD, on), a, th=2)
        s0, s1 = spans.get(key, (1e9, -1e9))
        spans[key] = (min(s0, x), max(s1, x + tx.width))
        x += tx.width
    labels = {"soft": "变成百分比", "qk": "比一比", "sq": "别太大", "v": "按比例取内容"}
    order_ = ["soft", "qk", "sq", "v"]
    for r, key in enumerate(order_):
        on = lit.get(key, 0.0) * notes
        if on <= 0.003:
            continue
        s0, s1 = spans[key]
        if key == "soft":
            s0 = spans["soft"][0] + T("=  ", "stix", sz, None, per_char=False).width
            s1 = spans["soft"][0] + T("=  softmax", "stix", sz, None, per_char=False).width
        xm = (s0 + s1) / 2
        yy = y2 + 92 + 66 * r
        hairline(F, s0 + 4, int(y2 + 24), s1 - 4, GOLD, 0.8 * a * on, th=2)
        glow_poly(F, [(xm, y2 + 28), (xm, yy - 34)], GOLD, 0.45 * a * on, th=1, glow=0.3, sigma=2)
        lab = zh(labels[key], 32, 500, 0.08)
        xl = min(max(xm, 80 + lab.width / 2), 950 - lab.width / 2)      # keep clear of the screen's edges
        if abs(xl - xm) > 2:
            glow_poly(F, [(xm, yy - 34), (xl, yy - 34)], GOLD, 0.45 * a * on, th=1, glow=0.3, sigma=2)
        lab.draw(F, xl, yy, GOLD, a * on, align="center")


# ================================================================ v3b ===

def chat(F, question, answer, n, t, a=1.0, y0=360, cand=True, cand_y=None):
    """A chat: the question in a bubble, the answer appearing one character at a time; above the
    newest character, a few candidates flicker - the model is guessing."""
    if a <= 0.003:
        return
    qt = zh(question, 36, 500, 0.06)
    bw = qt.width + 56
    _rrect(F, 960 - bw, y0 - 50, 960, y0 + 22, mix(STEEL, IVORY, 0.4), 0.6 * a, r=20, glow=0.2, fill=0.06)
    qt.draw(F, 960 - 28, y0 - 1, IVORY, a, align="right")
    # wrap the answer to the column
    x0, w, lh, size = 120, 840, 62, 40
    lines, cur = [], ""
    for ch in answer:
        if zh(cur + ch, size, 400, 0.06).width > w:
            lines.append(cur)
            cur = ""
        cur += ch
    lines.append(cur)
    k = 0
    last = None
    yb = y0 + 140
    for li, line in enumerate(lines):
        tx = zh(line, size, 400, 0.06)
        for m, dx, dy in tx.items:
            u = n - k
            k += 1
            if u <= 0:
                break
            al = a * min(1.0, u)
            blend(F, m, x0 + dx, yb + li * lh + dy, IVORY, al)
            last = (x0 + dx + m.shape[1], yb + li * lh)
    if cand and last is not None and n < len(answer):
        # the guess for the next character, re-drawn every few frames
        pool = "的是了在光气蓝红空色线时们中更会被散射眼睛从方向"
        rng = np.random.default_rng(int(t * 6))
        nxt = answer[min(int(n), len(answer) - 1)]
        others = [c for c in rng.choice(list(pool), 4) if c != nxt][:2]
        p0 = rng.uniform(0.45, 0.8)
        ps = sorted(rng.uniform(0.04, (1 - p0) / 2, 2), reverse=True)
        py = cand_y if cand_y is not None else 1010
        zh("下一个字", 28, 500, 0.1).draw(F, 160, py - 56, DIM, 0.9 * a)
        for j, (c, pr) in enumerate(zip([nxt] + others, [p0] + ps)):
            yy = py + 58 * j
            col = GOLD if j == 0 else mix(STEEL, IVORY, 0.4)
            zh(c, 38, 500, 0).draw(F, 180, yy, col if j == 0 else IVORY, 0.95 * a, align="center")
            E.add_light(F, np.full((18, max(1, int(560 * pr))), 1.0, np.float32), 222, yy - 22, col,
                        (0.55 if j == 0 else 0.3) * a)
            T(f"{round(100 * pr)}%", "stix", 30, None, per_char=False).draw(F, 236 + 560 * pr, yy, col if j == 0 else DIM,
                                                                            0.9 * a)
    if last is not None and (t * 1.6) % 1 < 0.6:
        hairline(F, last[0] + 4, int(last[1] + 6), last[0] + 26, GOLD, 0.9 * a, th=3)


CHAT_Q = "为什么天空是蓝色的？"
CHAT_A = ("阳光里有各种颜色的光。穿过大气时，波长短的蓝光，比红光更容易被空气分子散射，"
          "从四面八方进入我们的眼睛，所以天空看起来是蓝色的。")


def numcol(F, x, y, vals, n, a=1.0, size=34, gap=46, gold=False):
    """A column of numbers in brackets, the first n shown."""
    if a <= 0.003:
        return
    shown = vals[:max(0, int(n))]
    for i, v in enumerate(shown):
        txt = f"{v:+.2f}".replace("+", " ").replace("-", "\u2212")
        T(txt, "stix", size, None, per_char=False).draw(F, x, y + gap * i, GOLD if gold else IVORY,
                                                         a * min(1.0, n - i), align="right")
    h = gap * max(1, len(shown))
    st = E.Stroke(x - 130, y - size - 6, x + 30, y + h)
    st.poly([(x - 104, y - size), (x - 116, y - size), (x - 116, y + h - gap + 12), (x - 104, y + h - gap + 12)])
    st.poly([(x + 6, y - size), (x + 18, y - size), (x + 18, y + h - gap + 12), (x + 6, y + h - gap + 12)])
    st.light(F, mix(STEEL, IVORY, 0.5), 0.7 * a)


def matrix(F, x, y, cols, rows, cell, label, a=1.0, seed=0, t=0.0, shimmer=0.0, col=None, sub=None):
    """A weight matrix: a grid of little cells whose brightness is the number stored there."""
    if a <= 0.003:
        return
    rng = np.random.default_rng(seed)
    v = rng.uniform(0, 1, (rows, cols))
    if shimmer > 0:
        v = np.clip(v + shimmer * 0.25 * np.sin(t * 3 + rng.uniform(0, 6.28, (rows, cols))), 0, 1)
    c = col if col is not None else mix(STEEL, IVORY, 0.4)
    img = np.zeros((rows * cell, cols * cell), np.float32)
    for i in range(rows):
        for j in range(cols):
            img[i * cell + 1:(i + 1) * cell - 1, j * cell + 1:(j + 1) * cell - 1] = 0.15 + 0.85 * v[i, j]
    E.add_light(F, img, x, y, c, 0.30 * a)
    _rrect(F, x - 6, y - 6, x + cols * cell + 6, y + rows * cell + 6, c, 0.55 * a, r=6, glow=0.2)
    lab = T(label, "stix_it", 34, None, per_char=False)
    lab.draw(F, x + cols * cell / 2 - (8 if sub else 0), y + rows * cell + 46, c, a, align="center")
    if sub:
        T(sub, "stix_it", 22, None, per_char=False).draw(F, x + cols * cell / 2 + lab.width / 2 - 6,
                                                         y + rows * cell + 54, c, a)


def vec(F, x, y, n, cell, label, col, a=1.0, seed=0, label_zh=None):
    """A short vector drawn as a column of cells, with its letter (and meaning) underneath."""
    if a <= 0.003:
        return
    rng = np.random.default_rng(seed)
    v = rng.uniform(0.2, 1.0, n)
    img = np.zeros((n * cell, cell), np.float32)
    for i in range(n):
        img[i * cell + 2:(i + 1) * cell - 2, 2:cell - 2] = v[i]
    E.add_light(F, img, x, y, col, 0.45 * a)
    _rrect(F, x - 4, y - 4, x + cell + 4, y + n * cell + 4, col, 0.7 * a, r=5, glow=0.4)
    T(label, "stix_it", 44, None, per_char=False).draw(F, x + cell / 2, y + n * cell + 56, col, a, align="center")
    if label_zh:
        zh(label_zh, 28, 500, 0.06).draw(F, x + cell / 2, y + n * cell + 100, IVORY, 0.9 * a, align="center")


# --------------------------------------------------------- the detective ---

BOOK = (110, 330, 970, 1010)
CLUES = [("左撇子", 0, 2), ("十点的雨", 0, 6), ("鞋底的红泥", 1, 1), ("改过的遗嘱", 1, 4), ("没响的狗", 0, 9)]


def _book_maps():
    if "book" not in _PAGE_CACHE:
        x0, y0, x1, y1 = BOOK
        w, h = x1 - x0, y1 - y0
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        spine = np.exp(-((xx - w / 2) / 26) ** 2)
        lamp = 0.68 + 0.32 * np.exp(-(((xx - w * 0.4) / w) ** 2 + ((yy - h * 0.3) / h) ** 2)) - 0.35 * spine
        _PAGE_CACHE["book"] = lamp.astype(np.float32)
    return _PAGE_CACHE["book"]


def book(F, a=1.0, clues=0.0, threads=0.0, blank_a=1.0, t=0.0):
    """An open detective novel lit by a lamp; the last line ends '凶手是＿＿'.  Clues light up and
    threads run from the blank back to each of them."""
    if a <= 0.003:
        return
    x0, y0, x1, y1 = BOOK
    lamp = _book_maps()
    reg = F[y0:y1, x0:x1]
    reg[:] = reg * (1 - a) + (PAPER * 0.55 * lamp[..., None]) * a
    rng = np.random.default_rng(12)
    w = x1 - x0
    rows = 12
    clue_pos = {}
    for side in (0, 1):
        px0 = x0 + 50 + side * (w / 2)
        px1 = px0 + w / 2 - 100
        for r in range(rows):
            yy = y0 + 70 + 46 * r
            if side == 1 and r == rows - 1:
                continue
            xx = px0
            cl = [c for c in CLUES if c[1] == side and c[2] == r]
            while xx < px1:
                wd = rng.uniform(22, 70)
                if cl and xx > px0 + 40 and (side, r) not in clue_pos:
                    tx = zh(cl[0][0], 26, 500, 0.02)
                    clue_pos[(side, r)] = (xx + tx.width / 2, yy - 8, tx, xx)
                    xx += tx.width + 10
                    continue
                hairline(F, xx, int(yy - 6), min(xx + wd, px1), INKC, 0.25 * a, th=5)
                xx += wd + 8
    # the clues
    pts = []
    for (side, r), (cx_, cy_, tx, xs) in clue_pos.items():
        on = smooth(ramp(clues, 0.15 * len(pts), 0.4))
        if on > 0:
            blend(F, np.full((34, int(tx.width + 12)), 1.0, np.float32), xs - 6, cy_ - 22, GOLD, 0.45 * a * on)
        tx.draw(F, xs, cy_ + 6, INKC, a)
        pts.append(np.array([cx_, cy_ - 4]))
    # the last line
    bx = x0 + w / 2 + 50
    by = y0 + 70 + 46 * (rows - 1)
    tl = zh("凶手是", 34, 600, 0.04)
    tl.draw(F, bx, by + 4, INKC, a)
    bl0 = bx + tl.width + 10
    if blank_a > 0:
        blend(F, np.full((3, 130), 1.0, np.float32), bl0, by + 8, GOLD, a * blank_a)
        if (t * 1.6) % 1 < 0.6:
            blend(F, np.full((34, 3), 1.0, np.float32), bl0 + 4, by - 26, GOLD, a * blank_a)
    if threads > 0:
        src = np.array([bl0 + 65, by - 12])
        for k, p in enumerate(pts):
            g = smooth(ramp(threads, 0.12 * k, 0.5))
            if g > 0:
                thread(F, src, p, 0.8 * a, bow=60 if p[0] < src[0] else -40, grow=g)


# ------------------------------------------------------------ the sky laws ---

def kepler(F, cx, cy, a=1.0, dots=1.0, fit=0.0, t=0.0):
    """Tycho's observations as scattered marks; Kepler's ellipse drawn through them, the Sun at a focus."""
    if a <= 0.003:
        return
    A_, e = 330, 0.62
    B_ = A_ * math.sqrt(1 - e * e)
    c = A_ * e
    sun = np.array([cx + c, cy])
    orb(F, sun[0], sun[1], 10, GOLD, a)
    rng = np.random.default_rng(1601)
    th = np.sort(rng.uniform(0, 2 * math.pi, 46))
    for k, tt in enumerate(th):
        g = smooth(ramp(dots, k / len(th) * 0.8, 0.2))
        if g <= 0:
            continue
        p = np.array([cx + A_ * math.cos(tt), cy + B_ * math.sin(tt)]) + rng.normal(0, 5, 2)
        s2 = E.Stroke(p[0] - 10, p[1] - 10, p[0] + 11, p[1] + 11)
        s2.line((p[0] - 7, p[1]), (p[0] + 7, p[1]))
        s2.line((p[0], p[1] - 7), (p[0], p[1] + 7))
        s2.light(F, mix(STEEL, IVORY, 0.5), 0.8 * a * g)
    if fit > 0:
        u = np.linspace(0, 2 * math.pi * min(1.0, fit), max(3, int(160 * fit)))
        pts = np.column_stack([cx + A_ * np.cos(u), cy + B_ * np.sin(u)])
        glow_poly(F, pts, GOLD, 0.9 * a, th=2, glow=0.8, sigma=4)
        if fit >= 1:
            ang = t * 0.9
            orb(F, cx + A_ * math.cos(ang), cy + B_ * math.sin(ang), 5, IVORY, a)


def newton(F, cx, y, a=1.0):
    """F = G m₁m₂ / r², set in STIX."""
    if a <= 0.003:
        return
    sz = 80
    parts = [("F", "stix_it", 0, sz), (" = ", "stix", 0, sz), ("G", "stix_it", 0, sz)]
    w0 = sum(T(p, f, s_, None, per_char=False).width for p, f, _, s_ in parts)
    num = [("m", "stix_it", 0, 64), ("1", "stix", 14, 40), ("m", "stix_it", 0, 64), ("2", "stix", 14, 40)]
    wn = sum(T(p, f, s_, None, per_char=False).width for p, f, _, s_ in num)
    den = [("r", "stix_it", 0, 64), ("2", "stix", -26, 40)]
    wd = sum(T(p, f, s_, None, per_char=False).width for p, f, _, s_ in den)
    wf = max(wn, wd) + 20
    x = cx - (w0 + wf) / 2
    for p, f, dy, s_ in parts:
        tx = T(p, f, s_, None, per_char=False)
        tx.draw(F, x, y + dy, IVORY, a)
        x += tx.width
    xn = x + (wf - wn) / 2
    for p, f, dy, s_ in num:
        tx = T(p, f, s_, None, per_char=False)
        tx.draw(F, xn, y - 48 + dy, IVORY, a)
        xn += tx.width
    hairline(F, x, int(y - 24), x + wf, IVORY, a, th=3)
    xd = x + (wf - wd) / 2
    for p, f, dy, s_ in den:
        tx = T(p, f, s_, None, per_char=False)
        tx.draw(F, xd, y + 46 + dy, IVORY, a)
        xd += tx.width


# ------------------------------------------------------------ many cats ---

CAT_RING = [("sit", 230, 640, 200, False), ("sleep", 820, 600, 230, True), ("sit", 830, 1050, 170, True),
            ("sleep", 250, 1090, 200, False), ("sit", 540, 560, 150, False)]


def many_cats(F, a=1.0, gather=0.0):
    """Five different cats; as they gather they fade into one character."""
    for k, (pose, x, g, s_, flip) in enumerate(CAT_RING):
        ca = a * (1 - smooth(ramp(gather, 0.1 + 0.08 * k, 0.5)))
        if ca <= 0.003:
            continue
        X = x + (CX - x) * 0.6 * smooth(gather)
        G_ = g + (800 - g) * 0.6 * smooth(gather)
        fig = cat_figure(pose, round(X), round(G_), s_)
        out = fig["outline"].copy()
        if flip:
            out[:, 0] = 2 * X - out[:, 0]
        A.engraved(F, out, mix(mix(STEEL, IVORY, 0.5), GOLD, 0.3), ca * 0.85, form_a=0.4, fill=0.03)
    ga = a * smooth(ramp(gather, 0.45, 0.5))
    if ga > 0:
        add_sprite(F, CX, 760, 160, GOLD, 0.18 * ga)
        zh("猫", 240, 500, 0).draw(F, CX, 850, GOLD, ga, align="center")


# --------------------------------------------------------------- starlings ---

_BOIDS = {}


def boids(n=2000, frames=360, seed=7):
    """Starlings: each bird follows only its seven nearest neighbours (plus a gentle pull towards a
    slowly wandering centre).  Positions for every frame, computed once and cached."""
    key = (n, frames, seed)
    if key in _BOIDS:
        return _BOIDS[key]
    import os
    path = os.path.join(os.path.dirname(__file__), "build", f"boids2_{n}_{frames}_{seed}.npy")
    if os.path.exists(path):
        _BOIDS[key] = np.load(path)
        return _BOIDS[key]
    from scipy.spatial import cKDTree
    rng = np.random.default_rng(seed)
    p = rng.normal(0, 1, (n, 2)) * [260, 90] + [540, 760]
    v = rng.normal(0, 1, (n, 2)) * 0.6 + [3.2, 0.4]
    out = np.zeros((frames, n, 2), np.float32)
    warm = 90
    for f in range(frames + warm):
        tr = cKDTree(p)
        d, idx = tr.query(p, k=8)
        nb = idx[:, 1:]
        align = v[nb].mean(1) - v
        coh = p[nb].mean(1) - p
        diff = p[:, None, :] - p[nb]
        close = d[:, 1:] < 16
        sep = (diff / (d[:, 1:, None] ** 2 + 1) * close[..., None]).sum(1) * 16
        ang = f * 0.018
        centre = np.array([540 + 250 * math.sin(ang * 1.3), 780 + 230 * math.sin(ang * 0.8 + 1.0)])
        pull = (centre - p) * 0.0009
        turn = np.array([-v[:, 1], v[:, 0]]).T * (0.012 * math.sin(f * 0.035))
        v = v + align * 0.16 + coh * 0.0015 + sep * 0.08 + pull + turn + rng.normal(0, 0.05, v.shape)
        sp = np.linalg.norm(v, axis=1, keepdims=True) + 1e-6
        v = v / sp * np.clip(sp, 2.6, 4.6)
        p = p + v
        if f >= warm:
            out[f - warm] = p
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, out)
    _BOIDS[key] = out
    return out


_FLOCK = {}


def murmuration(F, t, a=1.0, n=8000):
    """A murmuration seen from the ground: the flock is a thin, bending sheet in three dimensions.
    Where the sheet turns edge-on it crowds into dark-dense ribbons; where it faces us it thins out."""
    if a <= 0.003:
        return
    import cv2
    key = n
    if key not in _FLOCK:
        rng = np.random.default_rng(1967)
        uv = rng.uniform(-1, 1, (n * 3, 2))
        th = np.arctan2(uv[:, 1], uv[:, 0])
        rr = np.hypot(uv[:, 0], uv[:, 1] * 1.6)
        edge = 0.95 + 0.18 * np.sin(3 * th + 0.8) + 0.10 * np.sin(5 * th + 2.1)   # an irregular, living outline
        uv = uv[rr < edge][:n]
        n = len(uv)
        _FLOCK[key] = (uv, rng.normal(0, 1, (n, 3)), rng.uniform(0, 6.28, n))
    uv, jit, ph = _FLOCK[key]
    n = len(uv)
    u, v = uv[:, 0], uv[:, 1]
    # the sheet bends and ripples
    x = u * 430 * (1 + 0.25 * math.sin(t * 0.37))
    y = v * 200 + 90 * np.sin(1.7 * u + t * 0.9) + 40 * np.sin(3.1 * u - t * 1.3)
    z = 160 * np.sin(2.2 * u + t * 0.6) * np.cos(1.5 * v - t * 0.4) + 80 * v * math.sin(t * 0.5)
    P3 = np.column_stack([x, y, z]) + jit * [6, 6, 6] + np.column_stack([np.sin(ph + t * 2.0) * 3,
                                                                         np.cos(ph + t * 1.7) * 3, np.zeros(n)])
    # the whole flock turns
    ax, ay, az = 0.9 * math.sin(t * 0.31), 1.1 * math.sin(t * 0.23 + 0.7), 0.35 * math.sin(t * 0.19)
    Rx = np.array([[1, 0, 0], [0, math.cos(ax), -math.sin(ax)], [0, math.sin(ax), math.cos(ax)]])
    Ry = np.array([[math.cos(ay), 0, math.sin(ay)], [0, 1, 0], [-math.sin(ay), 0, math.cos(ay)]])
    Rz = np.array([[math.cos(az), -math.sin(az), 0], [math.sin(az), math.cos(az), 0], [0, 0, 1]])
    P3 = P3 @ (Rz @ Ry @ Rx).T
    cx = 540 + 120 * math.sin(t * 0.21)
    cy = 760 + 80 * math.sin(t * 0.27 + 1.0)
    px, py = P3[:, 0] + cx, P3[:, 1] + cy
    h, w = F.shape[:2]
    img = np.zeros((h, w), np.float32)
    xi = np.clip(px.astype(int), 0, w - 1)
    yi = np.clip(py.astype(int), 0, h - 1)
    np.add.at(img, (yi, xi), 1.0)
    img = cv2.GaussianBlur(img, (0, 0), 0.9)
    F += (np.clip(img, 0, 1.2) * 1.4 * a)[..., None] * mix(IVORY, GOLD, 0.2)


# ----------------------------------------------------------------- neurons ---

def _branch(out, p, ang, length, width, depth, rng):
    q = p + length * np.array([math.cos(ang), math.sin(ang)])
    mid = (p + q) / 2 + rng.normal(0, length * 0.08, 2)
    u = np.linspace(0, 1, 8)[:, None]
    out.append(((1 - u) ** 2 * p + 2 * (1 - u) * u * mid + u ** 2 * q, width))
    if depth > 0:
        for s_ in (-1, 1):
            _branch(out, q, ang + s_ * rng.uniform(0.25, 0.6), length * rng.uniform(0.6, 0.78), width * 0.7,
                    depth - 1, rng)


def neuron(x, y, s, seed=0):
    """A nerve cell drawn after Ramón y Cajal: a body, branching dendrites, one long axon."""
    rng = np.random.default_rng(seed)
    out = []
    for k in range(6):
        ang = -math.pi / 2 + (k - 2.5) * 0.55 + rng.normal(0, 0.12)
        _branch(out, np.array([x, y]), ang, s * rng.uniform(0.26, 0.34), 1.6, 3, rng)
    ax = [np.array([x, y])]
    ang = math.pi / 2 + rng.normal(0, 0.1)
    for k in range(10):
        ang += rng.normal(0, 0.12)
        ax.append(ax[-1] + s * 0.11 * np.array([math.cos(ang), math.sin(ang)]))
    out.append((np.array(ax), 1.2))
    end = ax[-1]
    for k in range(4):
        _branch(out, end, math.pi / 2 + (k - 1.5) * 0.5, s * 0.12, 0.8, 1, rng)
    return out, np.array([x, y]), ax


def draw_neuron(F, nrn, a=1.0, grow=1.0, col=None, fire=None):
    if a <= 0.003:
        return
    segs, soma, ax = nrn
    c = col if col is not None else mix(mix(STEEL, IVORY, 0.5), GOLD, 0.2)
    n = len(segs)
    for i, (pts, wd) in enumerate(segs):
        g = smooth(ramp(grow, 0.5 * i / n, 0.5))
        if g <= 0:
            continue
        k = max(2, int(len(pts) * g))
        glow_poly(F, pts[:k], c, 0.7 * a, th=1, glow=0.3, sigma=2)
    orb(F, soma[0], soma[1], 7, c, a * smooth(grow))
    if fire is not None and 0 <= fire <= 1:
        i = fire * (len(ax) - 1)
        j = int(i)
        p = ax[j] + (ax[min(j + 1, len(ax) - 1)] - ax[j]) * (i - j)
        orb(F, p[0], p[1], 6, GOLD, a)


# ------------------------------------------------------------ dials, counter ---

def dials(F, x0, y0, cols, rows, gap, a=1.0, t=0.0, turn=0.0, seed=3):
    """Rows of tiny dials - the model's numbers - each turned a little at every wrong guess."""
    if a <= 0.003:
        return
    rng = np.random.default_rng(seed)
    base = rng.uniform(0, 2 * math.pi, (rows, cols))
    rate = rng.normal(0, 1, (rows, cols))
    r = gap * 0.34
    st = E.Stroke(x0 - gap, y0 - gap, x0 + cols * gap + gap, y0 + rows * gap + gap)
    for i in range(rows):
        for j in range(cols):
            cx_, cy_ = x0 + j * gap, y0 + i * gap
            ang = base[i, j] + turn * rate[i, j] * 0.6 + 0.04 * math.sin(t * 3 + base[i, j])
            st.circle((cx_, cy_), r, th=1)
            st.line((cx_, cy_), (cx_ + r * 0.9 * math.cos(ang), cy_ + r * 0.9 * math.sin(ang)))
    st.light(F, mix(STEEL, IVORY, 0.45), 0.55 * a)


def counter(F, value, cx, y, a=1.0, size=96, col=None):
    """A big number with thousands separators."""
    if a <= 0.003:
        return
    txt = f"{int(value):,}"
    T(txt, "stix", size, None, per_char=False).draw(F, cx, y, col if col is not None else GOLD, a, align="center")


def text_stream(F, a=1.0, t=0.0, mask=True, x0=110, x1=970, y0=380, rows=12, gap=62, speed=40.0):
    """Lines of text sliding upwards, each ending in a covered word the model must guess."""
    if a <= 0.003:
        return
    rng_seed = 21
    off = (t * speed) % gap
    first = int(t * speed // gap)
    for r in range(rows + 1):
        idx = first + r
        rng = np.random.default_rng(rng_seed + idx)
        yy = y0 + r * gap - off
        fade = smooth(ramp(yy, y0 - 10, 80)) * (1 - smooth(ramp(yy, y0 + rows * gap - 120, 110)))
        if fade <= 0.003:
            continue
        xx = x0
        end = x1 - rng.uniform(0, 260)
        while xx < end - 60:
            wd = rng.uniform(26, 90)
            hairline(F, xx, int(yy), min(xx + wd, end - 60), mix(STEEL, IVORY, 0.4), 0.35 * a * fade, th=6)
            xx += wd + 10
        if mask:
            _rrect(F, end - 50, yy - 14, end - 6, yy + 18, GOLD, 0.8 * a * fade, r=5, glow=0.4, fill=0.15)


def bleu(F, a=1.0, grow=1.0, y0=820):
    """English-to-German, WMT 2014: the best before, and the Transformer."""
    if a <= 0.003:
        return
    items = [("此前最好的方法", 26.4, mix(STEEL, IVORY, 0.4)), ("Transformer", 28.4, GOLD)]
    for i, (lab, v, c) in enumerate(items):
        y = y0 + 120 * i
        g = smooth(ramp(grow, 0.3 * i, 0.6))
        (zh(lab, 34, 500, 0.06) if i == 0 else T(lab, "inter", 34, 500, per_char=False)).draw(F, 120, y - 30, IVORY, a)
        bw = 640 * (v - 20) / 10 * g
        E.add_light(F, np.full((26, max(1, int(bw))), 1.0, np.float32), 120, y - 8, c, 0.55 * a)
        T(f"{v:.1f}", "stix", 40, None, per_char=False).draw(F, 140 + bw, y + 18, c, a * g)
    zh("英译德 · BLEU 分数（越高越好）", 26, 500, 0.06).draw(F, 120, y0 + 250, DIM, a)
