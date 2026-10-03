"""Visual system for the tunnelling film: a light Swiss-editorial look.

Reuses the compositing engine from ../quantum-video (loaded by file path so
module names there never shadow this folder's own timeline/scenes/...).
"""

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
E.FONT_FILES.update(inter="InterTight.ttf", mono="IBMPlexMono-Regular.ttf", mono_m="IBMPlexMono-Medium.ttf")

W, H = E.W, E.H
T, R, Stroke, blend = E.T, E.R, E.Stroke, E.blend
ramp, smooth, ease_out, ease_in_out, ease_in, window, lerp = (E.ramp, E.smooth, E.ease_out, E.ease_in_out,
                                                             E.ease_in, E.window, E.lerp)
rgb = E.rgb

PAPER = rgb("f2ede3")
INK = rgb("171615")
GRAPHITE = rgb("6f6a62")
RULE = rgb("cbc3b4")
VERMILION = rgb("e2472f")
TINT = rgb("f6d9cf")
BLUE = rgb("2d5aa6")

LX = 160
GRID_R = 1760


# --------------------------------------------------------------- paper ---

def make_paper(seed=5):
    rng = np.random.default_rng(seed)
    F = np.empty((H, W, 3), np.float32)
    F[:] = PAPER
    low = cv2.GaussianBlur(rng.normal(0, 1, (H // 8, W // 8)).astype(np.float32), (0, 0), 6)
    low = cv2.resize(low, (W, H), interpolation=cv2.INTER_CUBIC)
    fib = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), sigmaX=3.0, sigmaY=0.6)
    tex = low / (np.abs(low).max() + 1e-6) * 0.014 + fib / (np.abs(fib).max() + 1e-6) * 0.02
    F += tex[..., None] * np.array([1.0, 0.97, 0.92], np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    r2 = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2
    vig = (1 - 0.075 * np.clip(r2 - 0.25, 0, None)).astype(np.float32)
    return F, vig


def make_grain(n=6, amp=0.006, seed=9):
    rng = np.random.default_rng(seed)
    return [rng.normal(0, amp, (H, W)).astype(np.float32) for _ in range(n)]


def finalize(F, vig, grain):
    F *= vig[..., None]
    F += grain[..., None]
    np.clip(F, 0, 1, out=F)
    return (F * 255 + 0.5).astype(np.uint8)


def _pattern_dots(step=7, r=1.75):
    p = np.zeros((H + step, W + step), np.uint8)
    for y in range(0, H + step, step):
        off = (step // 2) if (y // step) % 2 else 0
        for x in range(off, W + step, step):
            cv2.circle(p, (x * 16, y * 16), int(r * 16), 255, -1, cv2.LINE_AA, 4)
    return p[:H, :W]


def _pattern_hatch(step=9):
    p = np.zeros((H, W), np.uint8)
    for c in range(-H, W + H, step):
        cv2.line(p, (c * 16, 0), ((c + H) * 16, H * 16), 255, 1, cv2.LINE_AA, 4)
    return p


DOTS = _pattern_dots()
HATCH = _pattern_hatch()


def pattern_fill(F, stroke, pattern, color, a):
    """Fill the shape drawn in `stroke` with a print pattern (dots / hatching)."""
    h, w = stroke.m.shape
    x0, y0 = stroke.x0, stroke.y0
    xs0, ys0 = max(x0, 0), max(y0, 0)
    xs1, ys1 = min(x0 + w, W), min(y0 + h, H)
    if xs0 >= xs1 or ys0 >= ys1:
        return
    m = stroke.m[ys0 - y0:ys1 - y0, xs0 - x0:xs1 - x0].astype(np.float32) / 255.0
    pat = pattern[ys0:ys1, xs0:xs1].astype(np.float32) / 255.0
    blend(F, m * pat, xs0, ys0, color, a)


def rect(F, x, y, w, h, color, a=1.0):
    if w >= 1 and h >= 1 and a > 0.002:
        blend(F, np.full((int(h), int(w)), 255, np.uint8), x, y, color, a)


# -------------------------------------------------------------- type -----

def rise(F, text, x, y, color, a, t, t0, dur=0.7, stagger=0.0, align="left", clip_pad=None):
    """Editorial line reveal: glyphs slide up from behind the baseline mask."""
    if a <= 0.002:
        return
    if align == "center":
        x -= text.width / 2
    elif align == "right":
        x -= text.width
    hmax = max((m.shape[0] for m, _, _ in text.items), default=0)
    pad = clip_pad if clip_pad is not None else hmax * 0.28
    clip_y = y + pad
    for i, (m, dx, dy) in enumerate(text.items):
        u = ease_out(ramp(t, t0 + i * stagger, dur))
        if u <= 0:
            continue
        off = (1 - u) * hmax * 1.05
        top = y + dy + off
        vis = int(min(m.shape[0], max(0, math.floor(clip_y - top))))
        if vis <= 0:
            continue
        blend(F, m[:vis], x + dx, top, color, a)


def mono(s, size=16, medium=False, tracking=0.08):
    return T(s, "mono_m" if medium else "mono", size, None, tracking=tracking, per_char=False)


def _is_cjk(ch):
    o = ord(ch)
    return 0x2E80 <= o <= 0x9FFF or 0x3000 <= o <= 0x303F or 0xFF00 <= o <= 0xFFEF


def mixed(s, size=15, tracking=0.08):
    """Mono for Latin, Noto Sans SC for CJK, as one run of text."""
    runs, cur, cj = [], "", None
    for ch in s:
        c = _is_cjk(ch)
        if cj is not None and c != cj:
            runs.append((cur, "sans" if cj else "mono", size, 400 if cj else None))
            cur = ""
        cur += ch
        cj = c
    if cur:
        runs.append((cur, "sans" if cj else "mono", size, 400 if cj else None))
    return R(runs, tracking=tracking, per_char=False)


def sans(s, size=32, wght=400, tracking=0.02):
    return T(s, "sans", size, wght, tracking=tracking)


def serif(s, size=72, wght=900, tracking=0.02):
    return T(s, "serif", size, wght, tracking=tracking)


def sci(F, value, x, y, size, color, a=1.0, align="left", digits=2):
    """Big scientific readout, e.g. 1.4 x 10^-4 (Inter Tight)."""
    if value >= 0.01:
        s = f"{value * 100:.{max(0, digits - 1)}f}%"
        t = T(s, "inter", size, 800, tracking=-0.01, per_char=False)
        t.draw(F, x - (t.width if align == "right" else 0), y, color, a)
        return
    e = math.floor(math.log10(value))
    mant = value / 10 ** e
    base = T(f"{mant:.1f} × 10", "inter", size, 800, tracking=-0.01, per_char=False)
    ex = T(f"−{-e}", "inter", int(size * 0.52), 800, tracking=0.0, per_char=False)
    total = base.width + ex.width + size * 0.06
    x0 = x - total if align == "right" else x
    base.draw(F, x0, y, color, a)
    ex.draw(F, x0 + base.width + size * 0.06, y - size * 0.42, color, a)


# ------------------------------------------------------------- chrome ----

def crop_marks(F, a):
    s = Stroke(0, 0, W, H)
    for cx, cy, dx, dy in ((40, 40, 1, 1), (W - 40, 40, -1, 1), (40, H - 40, 1, -1), (W - 40, H - 40, -1, -1)):
        s.line((cx, cy + dy * 14), (cx, cy + dy * 34))
        s.line((cx + dx * 14, cy), (cx + dx * 34, cy))
    s.blit(F, INK, 0.55 * a)


def header_footer(F, g, a, part, total_time):
    if a <= 0.002:
        return
    sans("量子隧穿", 15, 500, tracking=0.2).draw(F, LX, 72, INK, 0.85 * a)
    mono("QUANTUM TUNNELING", 14, tracking=0.16).draw(F, LX + 98, 72, GRAPHITE, 0.85 * a)
    if part:
        mono(f"PART {part:02d} / 05", 14, medium=True, tracking=0.16).draw(F, GRID_R, 72, INK, 0.85 * a,
                                                                          align="right")
    rect(F, LX, 92, GRID_R - LX, 1, INK, 0.5 * a)
    rect(F, LX, 1004, GRID_R - LX, 1, INK, 0.5 * a)
    mono("A SMALL IDEA", 14, tracking=0.16).draw(F, LX, 1032, GRAPHITE, 0.85 * a)
    sans("一个小概念", 14, 400, tracking=0.2).draw(F, LX + 132, 1032, GRAPHITE, 0.85 * a)
    tc = int(g)
    tt = int(total_time)
    mono(f"{tc // 60:02d}:{tc % 60:02d}  /  {tt // 60:02d}:{tt % 60:02d}", 14, tracking=0.12).draw(
        F, GRID_R, 1032, GRAPHITE, 0.85 * a, align="right")
    prog = min(max(g / total_time, 0), 1)
    rect(F, LX, 1002, (GRID_R - LX) * prog, 3, VERMILION, 0.9 * a)


def wipe(F, g, boundaries, dur=0.42):
    """Solid vermilion panel sweeping across at each section boundary."""
    for b in boundaries:
        if b - dur <= g < b:
            u = ease_in_out((g - (b - dur)) / dur)
            rect(F, 0, 0, W * u, H, VERMILION)
        elif b <= g < b + dur:
            u = ease_in_out((g - b) / dur)
            x0 = W * u
            rect(F, x0, 0, W - x0, H, VERMILION)


# ------------------------------------------------------------ layout -----

def part_header(F, t, a, num, zh, en):
    mono(f"PART {num:02d}", 16, medium=True, tracking=0.2).draw(F, LX, 168, VERMILION, a * smooth(ramp(t, 0.2, 0.5)))
    rise(F, serif(zh, 74, 900, tracking=0.01), LX - 3, 262, INK, a, t, 0.35, dur=0.8, stagger=0.06)
    mono(en.upper(), 15, tracking=0.18).draw(F, LX, 306, GRAPHITE, a * smooth(ramp(t, 0.9, 0.6)))
    rect(F, LX, 334, 48 * ease_in_out(ramp(t, 1.0, 0.6)), 6, VERMILION, a)


def body(F, t, a, lines, y0=420, dy=58):
    """lines: (t_in, text, style, group).  Earlier groups fade to graphite."""
    gstart = {}
    for t_in, _, _, gidx in lines:
        gstart[gidx] = min(gstart.get(gidx, 1e9), t_in)
    y = y0
    for t_in, s, style, gidx in lines:
        later = [v for k, v in gstart.items() if k > gidx]
        d = smooth(ramp(t, min(later), 0.8)) if later else 0.0
        if style == "note":
            rise(F, mixed(s, 18, tracking=0.06), LX, y - 6, GRAPHITE, a, t, t_in, dur=0.6, clip_pad=8)
            y += 44
            continue
        base = VERMILION if style == "accent" else INK
        col = E.mix(base, GRAPHITE, 0.55 * d) if style != "accent" else E.mix(base, GRAPHITE, 0.35 * d)
        rise(F, sans(s, 32, 500 if style == "accent" else 400), LX, y, col, a * (1 - 0.25 * d), t, t_in,
             dur=0.75, stagger=0.012)
        y += dy


def fig_caption(F, t, a, s, t0=1.2):
    rise(F, mixed(s, 15, tracking=0.08), 840, 948, GRAPHITE, a, t, t0, dur=0.6, clip_pad=6)
