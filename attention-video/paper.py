"""Paper-and-ink look: warm rice paper, ink-black serif, vermilion annotation.

Attention is drawn the way a careful reader marks up a page: a circle round
the word that is asking, faint pencil lines to every other word, and one
brush stroke to the word that answers.  Compositing is subtractive (ink on
paper) rather than additive light, so nothing glows."""

import math

import cv2
import numpy as np

from style import CX, E, H, SAFE_L, SUB_EN, SUB_ZH, W, T, Stroke, blend, caps, italic, ramp, serif, smooth, window
from style import _wrap

rgb = E.rgb
PAPER = np.array([0.935, 0.905, 0.848], np.float32)
INK = np.array([0.105, 0.095, 0.085], np.float32)
INK_SOFT = np.array([0.42, 0.40, 0.37], np.float32)
PENCIL = np.array([0.38, 0.37, 0.36], np.float32)
RED = np.array([0.70, 0.17, 0.11], np.float32)


# --------------------------------------------------------------- paper ---

def make_paper(seed=11):
    """Rice paper: mottled tone, fibres, fine tooth, a lamp-warm centre and darker edges."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    F = np.empty((H, W, 3), np.float32)
    F[:] = PAPER
    mott = cv2.GaussianBlur(rng.normal(0, 1, (H // 8, W // 8)).astype(np.float32), (0, 0), 4)
    mott = cv2.resize(mott / np.abs(mott).max(), (W, H), interpolation=cv2.INTER_CUBIC)
    F += mott[..., None] * np.array([0.022, 0.020, 0.016], np.float32)
    fib = np.zeros((H, W), np.float32)
    for _ in range(2600):
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        ang = rng.uniform(0, math.pi)
        L = rng.uniform(8, 70)
        bend = rng.normal(0, 0.25)
        pts = []
        for k in range(8):
            u = k / 7
            a = ang + bend * (u - 0.5)
            pts.append((x + math.cos(a) * L * (u - 0.5), y + math.sin(a) * L * (u - 0.5)))
        pts = np.round(np.array(pts) * 16).astype(np.int32)
        cv2.polylines(fib, [pts], False, float(rng.choice([-1.0, 1.0]) * rng.uniform(0.3, 1.0)), 1, cv2.LINE_AA, 4)
    fib = cv2.GaussianBlur(fib, (0, 0), 0.6)
    F += fib[..., None] * np.array([0.030, 0.028, 0.024], np.float32)
    tooth = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 0.7)
    F += tooth[..., None] * 0.012
    r = np.sqrt(((xx - CX) / (0.75 * W)) ** 2 + ((yy - 0.42 * H) / (0.62 * H)) ** 2)
    lamp = 1 + 0.035 * np.exp(-r ** 2 * 2.5)
    edge = 1 - 0.20 * np.clip(r - 0.55, 0, None) ** 1.6
    F *= (lamp * edge)[..., None]
    F *= np.array([1.0, 0.985, 0.955], np.float32) ** np.clip(r - 0.4, 0, None)[..., None]
    return F


def finalize(F, grain):
    F += grain[..., None]
    np.clip(F, 0, 1, out=F)
    return (F * 255 + 0.5).astype(np.uint8)


def make_grain(n=6, amp=0.004, seed=8):
    rng = np.random.default_rng(seed)
    return [cv2.resize(rng.normal(0, amp, (H // 2, W // 2)).astype(np.float32), (W, H),
                       interpolation=cv2.INTER_LINEAR) for _ in range(n)]


# ----------------------------------------------------------- marking ---

def _fill(F, poly, color, a, bleed=0.0):
    poly = np.asarray(poly, np.float64)
    pad = 10
    x0, y0 = int(poly[:, 0].min()) - pad, int(poly[:, 1].min()) - pad
    x1, y1 = int(poly[:, 0].max()) + pad, int(poly[:, 1].max()) + pad
    st = Stroke(x0, y0, x1, y1)
    st.fill(poly)
    if bleed > 0:
        blend(F, cv2.GaussianBlur(st.m.astype(np.float32) / 255, (0, 0), 2.2), x0, y0, color, bleed * a)
    blend(F, st.m, x0, y0, color, a)


def brush(F, pts, width, color=RED, a=1.0, upto=1.0, seed=0):
    """Tapered brush stroke along pts: heavy where the brush lands, lifting off at the end."""
    pts = np.asarray(pts, np.float64)
    if a <= 0.003 or upto <= 0 or len(pts) < 2:
        return
    k = max(2, int(round(len(pts) * min(upto, 1.0))))
    p = pts[:k]
    d = np.gradient(p, axis=0)
    n = np.stack([-d[:, 1], d[:, 0]], 1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    u = np.linspace(0, 1, len(pts))[:k]
    rng = np.random.default_rng(seed)
    wob = 1 + 0.12 * np.interp(u, np.linspace(0, 1, 6), rng.uniform(-1, 1, 6))
    prof = (np.clip(u / 0.08, 0, 1) ** 0.5) * (1 - 0.82 * u ** 2.2) * wob
    w = width * prof / 2
    poly = np.vstack([p + n * w[:, None], (p - n * w[:, None])[::-1]])
    _fill(F, poly, color, 0.88 * a, bleed=0.35)


def pencil(F, pts, a, upto=1.0, color=PENCIL):
    pts = np.asarray(pts, np.float64)
    k = max(2, int(round(len(pts) * min(upto, 1.0))))
    if a <= 0.003 or upto <= 0:
        return
    p = pts[:k]
    st = Stroke(p[:, 0].min() - 4, p[:, 1].min() - 4, p[:, 0].max() + 5, p[:, 1].max() + 5)
    st.poly(p, th=1)
    blend(F, st.m, st.x0, st.y0, color, 0.7 * a)


def ring(cx, cy, rx, ry, seed=0, turns=1.12, n=90):
    """A hand-drawn loop round a word: not quite closed, not quite round."""
    rng = np.random.default_rng(seed)
    th0 = rng.uniform(-2.6, -2.0)
    th = th0 + np.linspace(0, 2 * math.pi * turns, n)
    wob = 1 + 0.05 * np.sin(3 * th + rng.uniform(0, 6.28)) + 0.03 * np.sin(5 * th + rng.uniform(0, 6.28))
    drift = np.linspace(0, 1, n)[:, None] * np.array([rng.uniform(-6, 6), rng.uniform(4, 9)])
    return np.column_stack([cx + rx * wob * np.cos(th), cy + ry * wob * np.sin(th)]) + drift


def seal(F, x, y, chars, size, a=1.0, seed=3):
    """Vermilion seal with the characters left as bare paper (vertical column)."""
    if a <= 0.003:
        return
    rng = np.random.default_rng(seed)
    pad = size * 0.26
    w, h = size * 1.08 + 2 * pad, size * len(chars) * 1.02 + 2 * pad
    m = np.zeros((int(h) + 4, int(w) + 4), np.float32)
    jit = rng.uniform(-1.5, 1.5, (4, 2))
    poly = np.array([(2, 2), (w + 2, 2), (w + 2, h + 2), (2, h + 2)]) + jit
    cv2.fillPoly(m, [np.round(poly * 16).astype(np.int32)], 1.0, cv2.LINE_AA, 4)
    for i, ch in enumerate(chars):
        g = serif(ch, int(size), 700)
        gm, dx, dy = g.items[0]
        gx = int(2 + (w - gm.shape[1]) / 2)
        gy = int(2 + pad + i * size * 1.02 + size * 0.88 + dy)
        hh, ww = gm.shape
        reg = m[gy:gy + hh, gx:gx + ww]
        reg *= 1 - gm[:reg.shape[0], :reg.shape[1]].astype(np.float32) / 255
    tex = cv2.GaussianBlur(rng.random(m.shape).astype(np.float32), (0, 0), 1.2)
    m *= np.clip(0.55 + 1.2 * tex, 0, 1)
    blend(F, m, x, y, RED, 0.9 * a)


def note(F, s, x, y, a, size=40, color=RED, align="center"):
    """Margin note in a hand-ish italic."""
    T(s, "corm_it", size, 600, per_char=False).draw(F, x, y, color, a, align=align)


# ---------------------------------------------------------------- type ---

def subtitle(F, t, t_in, t_out, zh, en, a=1.0):
    al = a * window(t, t_in, t_out, 0.45, 0.45)
    if al <= 0.003:
        return
    zl = _wrap(zh, lambda s_: serif(s_, 40, 400, 0.08), 880, "，。：；？！、—") if zh else []
    el = _wrap(en, lambda s_: italic(s_, 28), 900, " ") if en else []
    y = SUB_ZH - 54 * (len(zl) - 1) - 34 * (len(el) - 1)
    for line in zl:
        serif(line, 40, 400, 0.08).draw(F, CX, y, INK, 0.9 * al, align="center")
        y += 54
    y += SUB_EN - SUB_ZH - 54
    for line in el:
        italic(line, 28).draw(F, CX, y, INK_SOFT, 0.95 * al, align="center")
        y += 34


def subtitles(F, t, items, a=1.0):
    for t_in, t_out, zh, en in items:
        subtitle(F, t, t_in, t_out, zh, en, a)


def chapter_mark(F, a, num, zh, en):
    """A small seal for the chapter, then its name."""
    if a <= 0.003:
        return
    x, y = SAFE_L, 200
    seal(F, x, y, zh, 30, a, seed=len(zh) + int(num))
    T(num, "stix", 22, None, per_char=False).draw(F, x + 62, y + 33, RED, 0.85 * a)
    caps(en, 14, 0.36).draw(F, x + 96, y + 32, INK_SOFT, 0.9 * a)
