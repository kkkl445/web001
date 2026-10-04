"""Cinematic 'deep' visual system, portrait edition for Douyin (9:16):
darkness, starlight, one idea per beat, bilingual subtitles, gold glow.
Text and key action stay inside the phone-UI safe zone (clear of the
status bar, the right-hand button column and the caption area).
Reuses the compositing engine from ../quantum-video by file path."""

import importlib.util
import math
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("engine", ROOT.parent / "quantum-video" / "engine.py")
E = importlib.util.module_from_spec(_spec)
sys.modules["engine"] = E
_spec.loader.exec_module(E)
E.FONT_DIR = ROOT / "fonts"
E.FONT_FILES["inter"] = "Inter.ttf"           # screen-friendly sans for English text and small labels
E.FONT_FILES["inter_it"] = "Inter-Italic.ttf"
E.W, E.H = 1080, 1920          # 9 : 16, full screen in the Douyin feed

W, H = E.W, E.H
CX, CY = W // 2, 820           # optical centre sits above the caption overlay
SUB_ZH, SUB_EN = 1356, 1404    # subtitles end above the caption area (~y 1500)
SAFE_L, SAFE_R = 72, 960       # right-hand button column starts near x 960
T, R, Stroke, blend, add_light, add_sprite, blur = E.T, E.R, E.Stroke, E.blend, E.add_light, E.add_sprite, E.blur
ramp, smooth, ease_out, ease_in_out, ease_in, window, lerp, mix = (E.ramp, E.smooth, E.ease_out, E.ease_in_out,
                                                                  E.ease_in, E.window, E.lerp, E.mix)
rgb = E.rgb

IVORY = rgb("ece6d8")
DIM = rgb("8e9098")
GOLD = rgb("e6b86a")
AMBER = rgb("ff9a3c")
CYAN = rgb("72d6ff")
BLUE = rgb("2a5cff")
WHITE = rgb("ffffff")
CRIMSON = rgb("7a1d2c")


# ---------------------------------------------------------- background ---

def make_background(seed=4):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    F = np.empty((H, W, 3), np.float32)
    F[:] = rgb("070b14")
    d = ((xx - CX) / 760) ** 2 + ((yy - CY) / 1000) ** 2
    F += (rgb("142240") - rgb("070b14")) * np.exp(-d * 1.4)[..., None]
    stars = np.zeros((H, W), np.float32)
    n = 900
    sx, sy = rng.uniform(0, W, n), rng.uniform(0, H, n)
    br = rng.power(4, n) * 0.22 + 0.02
    for x, y, b in zip(sx, sy, br):
        cv2.circle(stars, (int(x * 16), int(y * 16)), int(rng.uniform(0.5, 1.2) * 16), float(b), -1,
                   cv2.LINE_AA, 4)
    F += stars[..., None] * np.array([0.85, 0.9, 1.0], np.float32)
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    vig = (1 - 0.45 * np.clip((r - 0.45) / 0.95, 0, 1) ** 1.5).astype(np.float32)
    tw = dict(x=rng.uniform(0, W, 70), y=rng.uniform(0, H, 70), f=rng.uniform(0.15, 0.6, 70),
              p=rng.uniform(0, 6.28, 70), a=rng.uniform(0.1, 0.35, 70))
    return F, vig, tw


def twinkle(F, tw, g, a=1.0):
    k = 0.5 + 0.5 * np.sin(tw["f"] * g * 6.283 + tw["p"])
    for x, y, b in zip(tw["x"], tw["y"], tw["a"] * k ** 3 * a):
        add_sprite(F, x, y, 1.1, IVORY, b)


def make_grain(n=8, amp=0.009, seed=8):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        g = rng.normal(0, amp, (H // 2, W // 2)).astype(np.float32)
        out.append(cv2.resize(g, (W, H), interpolation=cv2.INTER_LINEAR))
    return out


def finalize(F, vig, grain, bloom=1.0):
    if bloom > 0:
        small = cv2.resize(F, (W // 4, H // 4), interpolation=cv2.INTER_AREA)
        br = np.maximum(small - 0.42, 0)
        b = cv2.GaussianBlur(br, (0, 0), 2.5) * 0.55 + cv2.GaussianBlur(br, (0, 0), 11) * 0.75
        F += cv2.resize(b, (W, H), interpolation=cv2.INTER_LINEAR) * bloom
    F *= vig[..., None]
    F += grain[..., None]
    knee = 0.8
    hi = F > knee
    if hi.any():
        F[hi] = knee + (1 - knee) * (1 - np.exp(-(F[hi] - knee) / (1 - knee)))
    np.clip(F, 0, 1, out=F)
    return (F * 255 + 0.5).astype(np.uint8)


# --------------------------------------------------------------- light ---

def orb(F, x, y, r, color, gain=1.0, core=WHITE):
    if gain <= 0.003:
        return
    add_sprite(F, x, y, r * 0.35, core, 1.2 * gain)
    add_sprite(F, x, y, r, color, 0.55 * gain)
    add_sprite(F, x, y, r * 3.2, color, 0.16 * gain)


def glow_poly(F, pts, color, a=1.0, th=2, glow=0.6, sigma=5, bounds=None):
    pts = np.asarray(pts, float)
    if len(pts) < 2 or a <= 0.003:
        return
    if bounds is None:
        pad = sigma * 3 + 6
        bounds = (pts[:, 0].min() - pad, pts[:, 1].min() - pad, pts[:, 0].max() + pad, pts[:, 1].max() + pad)
    x0, y0, x1, y1 = [int(v) for v in bounds]
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    s = Stroke(x0, y0, x1, y1)
    s.poly(pts, th=th)
    if glow > 0:
        s.glow(F, color, glow * a, sigma)
    s.light(F, mix(color, WHITE, 0.35), a)


def comet(F, hx, hy, direction, length, color, a=1.0, segs=14):
    """Horizontal light streak with a bright head moving in `direction` (+1/-1)."""
    if a <= 0.003:
        return
    for i in range(segs):
        u0, u1 = i / segs, (i + 1) / segs
        xa, xb = hx - direction * length * u0, hx - direction * length * u1
        glow_poly(F, [(xa, hy), (xb, hy)], color, a * (1 - u0) ** 2, th=1, glow=0.5, sigma=3)
    orb(F, hx, hy, 6, color, a)


def hairline(F, x0, y, x1, color, a, th=1):
    if a > 0.003 and x1 > x0:
        blend(F, np.full((th, int(x1 - x0)), 255, np.uint8), x0, y, color, a)


# ---------------------------------------------------------------- type ---

_SUPER = set("⁰¹²³⁴⁵⁶⁷⁸⁹⁻")
_mixed = {}


def zh(s, size, wght=400, tracking=0.0):
    """Chinese text in a sans face that reads well on a phone; superscript digits come from STIX."""
    if not _SUPER.intersection(s):
        return T(s, "sans", size, wght, tracking=tracking)
    key = (s, size, wght, tracking)
    if key not in _mixed:
        runs, cur, sup = [], "", None
        for ch in s:
            k = ch in _SUPER
            if sup is not None and k != sup:
                runs.append((cur, "stix" if sup else "sans", size, None if sup else wght))
                cur = ""
            cur, sup = cur + ch, k
        runs.append((cur, "stix" if sup else "sans", size, None if sup else wght))
        _mixed[key] = E.Text(runs, tracking, per_char=True)
    return _mixed[key]


def sci(mant, exp, size, prefix=""):
    """'3.6 × 10^38' as STIX runs; returns (base Text, exponent Text)."""
    base = T(f"{prefix}{mant} × 10" if mant else f"{prefix}10", "stix", size, None, per_char=False)
    return base, T(str(exp).replace("-", "−"), "stix", int(size * 0.6), None, per_char=False)


def draw_sci(F, mant, exp, x, y, size, color, a=1.0, align="left", prefix=""):
    base, ex = sci(mant, exp, size, prefix)
    w = base.width + 3 + ex.width
    x0 = x - w / 2 if align == "center" else x - w if align == "right" else x
    base.draw(F, x0, y, color, a)
    ex.draw(F, x0 + base.width + 3, y - size * 0.42, color, a)
    return w


def caps(s, size=12, tracking=0.32):
    return T(s, "inter", size, 500, tracking=tracking * 0.75, per_char=False)


def latin(s, size=22, wght=400):
    """English running text (subtitles, notes) in Inter."""
    return T(s, "inter", size, wght, per_char=False)


def _char_blur(m, sigma):
    pad = int(sigma * 3) + 1
    mm = np.zeros((m.shape[0] + 2 * pad, m.shape[1] + 2 * pad), np.float32)
    mm[pad:pad + m.shape[0], pad:pad + m.shape[1]] = m.astype(np.float32) / 255.0
    return cv2.GaussianBlur(mm, (0, 0), sigma), pad


def caret(F, x, y, size, a, t, phase=0.0):
    blink = 0.5 + 0.5 * math.cos((t - phase) * 2 * math.pi * 0.9)
    blend(F, np.full((int(size * 0.95), 2), 255, np.uint8), x, y - size * 0.8, GOLD, a * blink)


def statement(F, s, cx, y, t, t_in, t_out=None, size=60, wght=400, tracking=0.2, cps=9.0, gold=(),
              color=IVORY, a=1.0, cursor=False, align="center", fade_in=0.22):
    """Typewriter line; characters dissolve (blur + drift) on the way out.
    Returns x right after the full line (for a following blank)."""
    tx = zh(s, size, wght, tracking)
    x0 = cx - tx.width / 2 if align == "center" else cx
    n = len(tx.items)
    for i, (m, dx, dy) in enumerate(tx.items):
        u = ease_out(ramp(t, t_in + i / cps, fade_in))
        if u <= 0:
            continue
        col = GOLD if i in gold else color
        al, off, sig = a * u, (1 - u) * 6, 0.0
        if t_out is not None and t > t_out - 0.9:
            v = smooth(ramp(t, t_out - 0.9 + i * 0.025, 0.6))
            al *= 1 - v
            off -= 14 * v
            sig = 3.5 * v
        if al <= 0.003:
            continue
        if sig > 0.3:
            mm, pad = _char_blur(m, sig)
            blend(F, mm, x0 + dx - pad, y + dy + off - pad, col, al)
        else:
            blend(F, m, x0 + dx, y + dy + off, col, al)
    if cursor and n and t >= t_in:
        k = min(n, int((t - t_in) * cps) + 1)
        m, dx, dy = tx.items[k - 1]
        typed_end = t_in + n / cps
        al = a if t < typed_end + 0.3 else a * (0.5 + 0.5 * math.cos((t - typed_end) * 2 * math.pi * 0.9))
        if t_out is not None:
            al *= 1 - smooth(ramp(t, t_out - 0.9, 0.5))
        blend(F, np.full((int(size * 0.95), 2), 255, np.uint8), x0 + dx + m.shape[1] + size * 0.12,
              y - size * 0.8, GOLD, al)
    return x0 + tx.width


def blank(F, x, y, size, a, t, words=None, t_reel=None, speed=7.0, hold=None):
    """Gold fill-in-the-blank: underline + caret, or a slot reel of candidate words.
    hold=(t_hold, index) freezes the reel on words[index] from t_hold on."""
    if a <= 0.003:
        return
    w = size * 2.3
    blend(F, np.full((2, int(w)), 255, np.uint8), x + size * 0.15, y + size * 0.18, GOLD, 0.9 * a)
    if not words or t_reel is None or t < t_reel:
        caret(F, x + size * 0.25, y, size, a, t)
        return
    pos = (t - t_reel) * speed
    if hold is not None and t >= hold[0]:
        pos = float(hold[1])
    k = int(math.floor(pos))
    fr = pos - k
    lh = size * 1.1
    for j in (-1, 0, 1, 2):
        idx = (k + j) % len(words)
        dist = abs(j - fr)
        al = a * max(0.0, 1 - dist * 0.75) ** 1.6
        if al <= 0.01:
            continue
        tw = zh(words[idx], size, 500, 0.1)
        tw.draw(F, x + size * 0.15 + (w - tw.width) / 2, y + (j - fr) * lh, GOLD, al)


def _wrap(text, make, max_w, sep_chars):
    """Greedy two-line wrap, breaking after punctuation (zh) or at spaces (en)."""
    if make(text).width <= max_w:
        return [text]
    best, best_d = None, 1e9
    for i, ch in enumerate(text):
        if ch in sep_chars and 0 < i < len(text) - 1:
            a, b = text[:i + 1].rstrip(), text[i + 1:].lstrip()
            d = abs(make(a).width - make(b).width)
            if d < best_d and make(a).width <= max_w and make(b).width <= max_w:
                best, best_d = (a, b), d
    if best is None:
        i = len(text) // 2
        best = (text[:i], text[i:])
    return list(best)


# Subtitles are drawn after the camera move, so they stay put while the picture zooms.
# render.py turns DEFER on, lets the scene queue its lines, moves the camera, then calls flush_subtitles().
DEFER = [False]
_QUEUE = []
# The scene can ask for a camera: scale about (x, y) plus a shift; render.py applies it.
CAMERA = {}


def camera(scale=1.0, x=None, y=None, dx=0.0, dy=0.0, blur=0.0):
    CAMERA.update(scale=scale, x=CX if x is None else x, y=860 if y is None else y, dx=dx, dy=dy, blur=blur)


def flush_subtitles(F):
    DEFER[0] = False
    for args in _QUEUE:
        subtitle(F, *args)
    _QUEUE.clear()


def subtitle(F, t, t_in, t_out, zh_text, en_text, a=1.0):
    if DEFER[0]:
        _QUEUE.append((t, t_in, t_out, zh_text, en_text, a))
        return
    al = a * window(t, t_in, t_out, 0.45, 0.45)
    if al <= 0.003:
        return
    zl = _wrap(zh_text, lambda s_: zh(s_, 40, 400, 0.08), 880, "，。：；？！、—") if zh_text else []
    el = _wrap(en_text, lambda s_: latin(s_, 25), 900, " ") if en_text else []
    y = SUB_ZH - 54 * (len(zl) - 1) - 34 * (len(el) - 1)
    for line in zl:
        zh(line, 40, 400, 0.08).draw(F, CX, y, IVORY, 0.94 * al, align="center")
        y += 54
    y += SUB_EN - SUB_ZH - 54
    for line in el:
        latin(line, 25).draw(F, CX, y, DIM, 0.95 * al, align="center")
        y += 34


def subtitles(F, t, items, a=1.0):
    for t_in, t_out, zh_text, en_text in items:
        subtitle(F, t, t_in, t_out, zh_text, en_text, a)


def chapter_mark(F, a, num, zh_text, en):
    if a <= 0.003:
        return
    x, y = SAFE_L, 236
    T(num, "stix", 24, None, per_char=False).draw(F, x, y, GOLD, 0.85 * a)
    zh(zh_text, 24, 500).draw(F, x + 46, y, IVORY, 0.85 * a)
    caps(en, 14, 0.36).draw(F, x + 84, y - 1, DIM, 0.9 * a)
    hairline(F, x, y + 20, x + 360, DIM, 0.3 * a)


def gauge(F, cx, y, w, value, color, zh_text, en, value_text, a=1.0, ticks=10):
    """Instrument: label, hairline scale, glowing dot at value (0..1)."""
    if a <= 0.003:
        return
    x0, x1 = cx - w / 2, cx + w / 2
    zl = zh(zh_text, 14, 500, 0.2)
    zl.draw(F, x0, y - 14, DIM, a)
    caps(en, 10, 0.34).draw(F, x0 + zl.width + 12, y - 15, DIM, 0.9 * a)
    latin(value_text, 18).draw(F, x1, y - 12, color, a, align="right")
    hairline(F, x0, y, x1, DIM, 0.35 * a)
    for k in range(ticks + 1):
        blend(F, np.full((4, 1), 255, np.uint8), x0 + w * k / ticks, y - 1, DIM, 0.35 * a)
    xv = x0 + w * value
    if xv > x0 + 1:
        glow_poly(F, [(x0, y), (xv, y)], color, 0.9 * a, th=2, glow=0.7, sigma=4)
    orb(F, xv, y, 7, color, a)


# ------------------------------------------------------------- effects ---

def shock(F, x, y, age, a=1.0, color=GOLD, size=1.0):
    """A reveal's impact: a bright flash and a thin ring of light racing outwards."""
    if age < 0 or age > 1.4 or a <= 0.003:
        return
    u = age / 1.4
    r = (40 + 760 * ease_out(min(1.0, age / 1.1))) * size
    fade = (1 - u) ** 2
    add_sprite(F, x, y, 90 * size * (1 + age), color, 0.9 * a * math.exp(-age * 6))
    add_sprite(F, x, y, 30 * size, WHITE, 1.2 * a * math.exp(-age * 9))
    th = np.linspace(0, 2 * math.pi, 180)
    glow_poly(F, np.column_stack([x + r * np.cos(th), y + r * np.sin(th)]), color, 0.55 * a * fade, th=2,
              glow=1.2, sigma=6)


_PUNCH = {}


def punch(F, s, cx, y, t, t0, t_out=None, size=80, wght=600, color=IVORY, gold=(), a=1.0, tracking=0.08):
    """A headline that lands: it drops in from slightly larger, flashes, then settles."""
    if t < t0 or a <= 0.003:
        return
    key = (s, size, wght, tracking, tuple(gold))
    if key not in _PUNCH:
        tx = zh(s, size, wght, tracking)
        pad = 30
        w, h = int(tx.width + 2 * pad), int(size * 1.6 + 2 * pad)
        base = int(size * 1.15 + pad)
        mw = np.zeros((h, w), np.float32)
        mg = np.zeros((h, w), np.float32)
        for i, (m, dx, dy) in enumerate(tx.items):
            x0, y0 = int(round(pad + dx)), int(round(base + dy))
            tgt = mg if i in gold else mw
            hh, ww = m.shape
            tgt[y0:y0 + hh, x0:x0 + ww] = np.maximum(tgt[y0:y0 + hh, x0:x0 + ww], m / 255.0)
        _PUNCH[key] = (mw, mg, base, w, h)
    mw, mg, base, w, h = _PUNCH[key]
    u = ease_out(ramp(t, t0, 0.32))
    k = 1 + 0.32 * (1 - u)
    al = a * min(1.0, (t - t0) / 0.12)
    if t_out is not None:
        al *= 1 - smooth(ramp(t, t_out - 0.5, 0.5))
    if al <= 0.003:
        return
    ww, hh = max(1, int(w * k)), max(1, int(h * k))
    for m, col in ((mw, color), (mg, GOLD)):
        if m.max() <= 0:
            continue
        ms = cv2.resize(m, (ww, hh), interpolation=cv2.INTER_LINEAR)
        x0 = cx - ww / 2
        y0 = y - base * k
        blend(F, ms, x0, y0, col, al)
        flash = math.exp(-(t - t0) * 5.0)
        if flash > 0.02:
            add_light(F, cv2.GaussianBlur(ms, (0, 0), 10), x0, y0, col, 0.9 * flash * al)


_WARP = {}


def warp(F, age, dur=1.4, a=1.0, cx=CX, cy=860, n=520, seed=11):
    """Stars streaking past - the jump into the sky of meaning."""
    if age < 0 or age > dur or a <= 0.003:
        return
    if n not in _WARP:
        rng = np.random.default_rng(seed)
        _WARP[n] = (rng.uniform(0, 2 * math.pi, n), rng.uniform(0.02, 0.5, n), rng.uniform(0.4, 1.0, n))
    ang, r0, br = _WARP[n]
    u = age / dur
    env = math.sin(math.pi * u) ** 0.8
    speed = 2.6 * (0.3 + u)
    r1 = r0 * math.exp(speed * u * 3.0) * 900
    r2 = r1 * (1 + 0.35 * env)
    img = np.zeros(F.shape[:2], np.float32)
    for a_, p, q, b in zip(ang, r1, r2, br):
        c, s_ = math.cos(a_), math.sin(a_)
        x1, y1, x2, y2 = cx + p * c, cy + p * s_, cx + q * c, cy + q * s_
        cv2.line(img, (int(x1 * 4), int(y1 * 4)), (int(x2 * 4), int(y2 * 4)), float(b), 1, cv2.LINE_AA, 2)
    img = cv2.GaussianBlur(img, (0, 0), 1.2)
    F += (img * 0.9 * env * a)[..., None] * mix(IVORY, GOLD, 0.25)
    add_sprite(F, cx, cy, 140, GOLD, 0.25 * env * a)
