"""Small 2D compositing engine: float RGB frames, antialiased strokes, glows,
typography with per-glyph animation.  numpy + OpenCV + Pillow only."""

import functools
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

W, H = 1920, 1080
ROOT = Path(__file__).resolve().parent
FONT_DIR = ROOT / "fonts"


# ---------------------------------------------------------------- colour ---

def rgb(h):
    h = h.lstrip("#")
    return np.array([int(h[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)


INK = rgb("05070c")
IVORY = rgb("efe9dc")
MUTED = rgb("8a91a2")
GOLD = rgb("dcb574")
CYAN = rgb("7fd0ff")
BLUE = rgb("3d6cff")
VIOLET = rgb("a28fff")
WHITE = rgb("ffffff")


def mix(a, b, u):
    return (a + (b - a) * u).astype(np.float32)


# ---------------------------------------------------------------- easing ---

def clamp01(x):
    return 0.0 if x < 0 else 1.0 if x > 1 else x


def ramp(t, t0, dur):
    return clamp01((t - t0) / dur)


def smooth(x):
    x = clamp01(x)
    return x * x * (3 - 2 * x)


def ease_out(x):
    x = clamp01(x)
    return 1 - (1 - x) ** 3


def ease_in(x):
    x = clamp01(x)
    return x ** 3


def ease_in_out(x):
    x = clamp01(x)
    return 4 * x ** 3 if x < 0.5 else 1 - (-2 * x + 2) ** 3 / 2


def ease_out_expo(x):
    x = clamp01(x)
    return 1.0 if x >= 1 else 1 - 2 ** (-10 * x)


def window(t, t_in, t_out, fin=0.8, fout=0.8):
    """1 between t_in and t_out with eased edges (fade-out ends at t_out)."""
    return ease_out(ramp(t, t_in, fin)) * (1 - smooth(ramp(t, t_out - fout, fout)))


def lerp(a, b, u):
    return a + (b - a) * u


# ----------------------------------------------------------- compositing ---

def _clip(x, y, w, h):
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, W), min(y + h, H)
    return x0, y0, x1, y1


def blend(F, m, x, y, color, a=1.0):
    """Alpha-composite `color` through mask m (uint8 or float 0..1) at (x, y)."""
    if a <= 0.002:
        return
    x, y = int(round(x)), int(round(y))
    h, w = m.shape[:2]
    x0, y0, x1, y1 = _clip(x, y, w, h)
    if x0 >= x1 or y0 >= y1:
        return
    mm = m[y0 - y:y1 - y, x0 - x:x1 - x]
    af = mm.astype(np.float32) * (a / 255.0) if mm.dtype == np.uint8 else mm * a
    reg = F[y0:y1, x0:x1]
    reg += (color[None, None, :] - reg) * af[..., None]


def add_light(F, m, x, y, color, gain=1.0):
    """Additive light through mask m (float or uint8)."""
    if gain <= 0.002:
        return
    x, y = int(round(x)), int(round(y))
    h, w = m.shape[:2]
    x0, y0, x1, y1 = _clip(x, y, w, h)
    if x0 >= x1 or y0 >= y1:
        return
    mm = m[y0 - y:y1 - y, x0 - x:x1 - x]
    af = mm.astype(np.float32) * (gain / 255.0) if mm.dtype == np.uint8 else mm * gain
    F[y0:y1, x0:x1] += af[..., None] * color[None, None, :]


def add_rgb(F, img, x, y, gain=1.0):
    """Add an HxWx3 float image."""
    x, y = int(round(x)), int(round(y))
    h, w = img.shape[:2]
    x0, y0, x1, y1 = _clip(x, y, w, h)
    if x0 >= x1 or y0 >= y1:
        return
    F[y0:y1, x0:x1] += img[y0 - y:y1 - y, x0 - x:x1 - x] * gain


def blur(m, sigma):
    if sigma <= 0:
        return m
    if sigma > 5:
        f = max(2, int(sigma // 2.5))
        hh, ww = m.shape[:2]
        small = cv2.resize(m, (max(1, ww // f), max(1, hh // f)), interpolation=cv2.INTER_AREA)
        small = cv2.GaussianBlur(small, (0, 0), sigma / f)
        return cv2.resize(small, (ww, hh), interpolation=cv2.INTER_LINEAR)
    return cv2.GaussianBlur(m, (0, 0), sigma)


@functools.lru_cache(maxsize=256)
def _sprite(rq):
    r = rq / 4.0
    n = int(math.ceil(r * 3.2)) + 1
    yy, xx = np.mgrid[-n:n + 1, -n:n + 1].astype(np.float32)
    return np.exp(-(xx * xx + yy * yy) / (2 * r * r)).astype(np.float32)


def add_sprite(F, cx, cy, r, color, gain):
    """Additive soft gaussian dot of radius r (sigma) centred at (cx, cy)."""
    if gain <= 0.002 or r <= 0:
        return
    s = _sprite(max(1, int(round(r * 4))))
    n = s.shape[0] // 2
    add_light(F, s, cx - n, cy - n, color, gain)


class Stroke:
    """ROI-bound uint8 canvas for antialiased vector drawing (subpixel, LINE_AA)."""

    def __init__(self, x0, y0, x1, y1):
        self.x0, self.y0 = int(x0), int(y0)
        self.m = np.zeros((int(y1) - self.y0, int(x1) - self.x0), np.uint8)

    def _p(self, p):
        return (int(round((p[0] - self.x0) * 16)), int(round((p[1] - self.y0) * 16)))

    def line(self, p, q, th=1, v=255):
        cv2.line(self.m, self._p(p), self._p(q), v, th, cv2.LINE_AA, 4)

    def dashed(self, p, q, dash=6, gap=6, th=1, v=255):
        p, q = np.asarray(p, float), np.asarray(q, float)
        L = float(np.hypot(*(q - p)))
        if L < 1e-6:
            return
        d = (q - p) / L
        s = 0.0
        while s < L:
            e = min(s + dash, L)
            self.line(p + d * s, p + d * e, th, v)
            s += dash + gap

    def poly(self, pts, th=1, closed=False, v=255):
        a = np.asarray(pts, np.float64)
        if len(a) < 2:
            return
        a = np.round((a - (self.x0, self.y0)) * 16).astype(np.int32)
        cv2.polylines(self.m, [a], closed, v, th, cv2.LINE_AA, 4)

    def fill(self, pts, v=255):
        a = np.round((np.asarray(pts, np.float64) - (self.x0, self.y0)) * 16).astype(np.int32)
        cv2.fillPoly(self.m, [a], v, cv2.LINE_AA, 4)

    def circle(self, c, r, th=1, v=255):
        cv2.circle(self.m, self._p(c), int(round(r * 16)), v, th, cv2.LINE_AA, 4)

    def ellipse(self, c, axes, angle=0, th=1, v=255, a0=0, a1=360):
        cv2.ellipse(self.m, self._p(c), (int(round(axes[0] * 16)), int(round(axes[1] * 16))),
                    angle, a0, a1, v, th, cv2.LINE_AA, 4)

    def blit(self, F, color, a=1.0):
        blend(F, self.m, self.x0, self.y0, color, a)

    def glow(self, F, color, gain, sigma):
        g = blur(self.m.astype(np.float32) * (1 / 255.0), sigma)
        add_light(F, g, self.x0, self.y0, color, gain)

    def light(self, F, color, gain):
        add_light(F, self.m, self.x0, self.y0, color, gain)


# ------------------------------------------------------------ typography ---

FONT_FILES = {
    "serif": "NotoSerifSC.ttf",
    "sans": "NotoSansSC.ttf",
    "corm": "CormorantGaramond.ttf",
    "corm_it": "CormorantGaramond-Italic.ttf",
    "jost": "Jost.ttf",
    "stix": "STIXTwoText.ttf",
    "stix_it": "STIXTwoText-Italic.ttf",
    "math": "NotoSansMath.ttf",
}


@functools.lru_cache(maxsize=None)
def font(name, size, wght=None):
    f = ImageFont.truetype(str(FONT_DIR / FONT_FILES[name]), size)
    if wght is not None:
        try:
            axes = f.get_variation_axes()
            vals = [wght if b"eight" in (a["name"] if isinstance(a["name"], bytes) else a["name"].encode())
                    else a["default"] for a in axes]
            f.set_variation_by_axes(vals)
        except OSError:
            pass
    return f


class Text:
    """A laid-out run of text (possibly mixing fonts), kept as glyph masks.

    runs: list of (string, font_name, size, weight).  per_char=False keeps each
    run whole (kerning preserved) - used for Latin sentences and equations.
    """

    def __init__(self, runs, tracking=0.0, per_char=True):
        self.items = []  # (mask, dx, dy) relative to baseline-left origin
        x = 0.0
        last_track = 0.0
        for s, fname, size, wght in runs:
            f = font(fname, size, wght)
            pieces = list(s) if per_char else [s]
            for ch in pieces:
                adv = f.getlength(ch)
                l, t, r, b = f.getbbox(ch, anchor="ls")
                if r > l and b > t and ch.strip():
                    im = Image.new("L", (r - l + 4, b - t + 4), 0)
                    ImageDraw.Draw(im).text((2 - l, 2 - t), ch, font=f, fill=255, anchor="ls")
                    self.items.append((np.asarray(im), x + l - 2, t - 2))
                last_track = tracking * size
                x += adv + last_track
        self.width = x - last_track

    def draw(self, F, x, y, color, a=1.0, t=None, t0=0.0, mode="chars", stagger=0.035,
             dur=0.8, rise=14.0, align="left", glow=0.0, glow_color=None, glow_sigma=10.0):
        if a <= 0.002:
            return
        if align == "center":
            x -= self.width / 2
        elif align == "right":
            x -= self.width
        n = len(self.items)
        if t is None:
            t = t0 + 1e9
        if mode == "wipe":
            p = ease_in_out(ramp(t, t0, dur))
            if p <= 0:
                return
            feather = 140.0
            front = p * (self.width + feather)
        for i, (m, dx, dy) in enumerate(self.items):
            if mode == "chars":
                u = ease_out(ramp(t, t0 + i * stagger, dur))
                ai, off = a * u, (1 - u) * rise
                if ai <= 0.002:
                    continue
                blend(F, m, x + dx, y + dy + off, color, ai)
                if glow > 0:
                    self._glow(F, m, x + dx, y + dy + off, glow_color if glow_color is not None else color,
                               glow * ai, glow_sigma)
            elif mode == "wipe":
                cols = dx + np.arange(m.shape[1], dtype=np.float32)
                ramp_w = np.clip((front - cols) / feather, 0, 1).astype(np.float32)
                if ramp_w.max() <= 0:
                    continue
                mm = m.astype(np.float32) * (1 / 255.0) * ramp_w[None, :]
                blend(F, mm, x + dx, y + dy, color, a)
                if glow > 0:
                    self._glow(F, (mm * 255).astype(np.uint8), x + dx, y + dy,
                               glow_color if glow_color is not None else color, glow * a, glow_sigma)
            else:  # fade
                u = ease_out(ramp(t, t0, dur))
                blend(F, m, x + dx, y + dy + (1 - u) * rise, color, a * u)
                if glow > 0:
                    self._glow(F, m, x + dx, y + dy + (1 - u) * rise,
                               glow_color if glow_color is not None else color, glow * a * u, glow_sigma)

    @staticmethod
    def _glow(F, m, x, y, color, gain, sigma):
        pad = int(sigma * 3)
        mm = np.zeros((m.shape[0] + 2 * pad, m.shape[1] + 2 * pad), np.float32)
        mm[pad:pad + m.shape[0], pad:pad + m.shape[1]] = m.astype(np.float32) * (1 / 255.0)
        add_light(F, blur(mm, sigma), x - pad, y - pad, color, gain)


_text_cache = {}


def T(s, name="serif", size=32, wght=400, tracking=0.0, per_char=True):
    """Cached single-font Text."""
    key = (s, name, size, wght, tracking, per_char)
    if key not in _text_cache:
        _text_cache[key] = Text([(s, name, size, wght)], tracking, per_char)
    return _text_cache[key]


def R(runs, tracking=0.0, per_char=False):
    """Cached multi-font Text (equations)."""
    key = (tuple(runs), tracking, per_char)
    if key not in _text_cache:
        _text_cache[key] = Text(list(runs), tracking, per_char)
    return _text_cache[key]


# ------------------------------------------------------------ background ---

def make_background():
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    d1 = ((xx - 1280) / 1150) ** 2 + ((yy - 520) / 820) ** 2
    d2 = ((xx - 330) / 900) ** 2 + ((yy - 240) / 700) ** 2
    bg = np.empty((H, W, 3), np.float32)
    bg[:] = INK
    bg += (rgb("0d1628") - INK) * np.exp(-d1 * 1.7)[..., None]
    bg += (rgb("0e0c0b") - INK) * 0.6 * np.exp(-d2 * 2.0)[..., None]
    r = np.sqrt(((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2)
    vig = 1 - 0.5 * np.clip((r - 0.55) / 0.85, 0, 1) ** 1.6
    return bg, vig.astype(np.float32)


def make_grain(n=8, amp=0.006, seed=7):
    rng = np.random.default_rng(seed)
    g = rng.normal(0, amp, (n, H // 2, W // 2)).astype(np.float32)
    # 2x2 grain clumps read as film rather than digital noise
    return [cv2.resize(gi, (W, H), interpolation=cv2.INTER_NEAREST) for gi in g]


def finalize(F, vig, grain):
    F *= vig[..., None]
    F += grain[..., None]
    knee = 0.82
    hi = F > knee
    if hi.any():
        F[hi] = knee + (1 - knee) * (1 - np.exp(-(F[hi] - knee) / (1 - knee)))
    np.clip(F, 0, 1, out=F)
    return (F * 255 + 0.5).astype(np.uint8)
