"""Figures drawn the way old star atlases and scratchboard engravings draw them.

Two treatments, both after classic references:
- engraved(): a fine contour plus engraved form-lines that follow the shape
  (isolines of the distance to the edge), lit from the upper left, as in
  Hevelius' Firmamentum or the cards of Urania's Mirror.  Stars sit on the
  figure and a few thin lines join them, as on a celestial chart.
- silhouette(): solid cut-paper figures in the manner of Lotte Reiniger,
  black against a low band of light, with a thin rim where the light catches.

Shapes are closed Catmull-Rom curves in figure units (ground at y = 0,
up is negative y, about one unit tall), placed on screen with place()."""

import math

import cv2
import numpy as np

from figures2d import catmull
from style import E, GOLD, IVORY, add_sprite, glow_poly, mix, orb, ramp, smooth

LIGHT = np.array([-0.62, -0.78])       # light comes from the upper left


def place(units, x, ground, s, flip=False):
    p = np.asarray(units, float).copy()
    if flip:
        p[:, 0] = -p[:, 0]
    return p * s + np.array([x, ground])


def curve(ctrl, x, ground, s, flip=False, n=14, closed=True):
    return place(catmull(ctrl, n, closed=closed), x, ground, s, flip)


# ----------------------------------------------------------------- shapes ---
# A cat sitting upright in profile, facing right, its tail wrapped round its paws.
CAT_SIT = [
    (0.31, 0.0), (0.27, -0.06), (0.26, -0.22), (0.27, -0.36), (0.285, -0.47), (0.275, -0.57),
    (0.30, -0.635), (0.355, -0.675), (0.405, -0.70), (0.435, -0.735), (0.43, -0.775), (0.395, -0.82),
    (0.365, -0.865), (0.345, -0.905), (0.315, -1.0), (0.265, -0.905), (0.225, -0.895), (0.185, -0.965),
    (0.155, -0.88), (0.11, -0.82), (0.05, -0.70), (-0.05, -0.59), (-0.15, -0.47), (-0.225, -0.33),
    (-0.265, -0.18), (-0.265, -0.07), (-0.23, 0.0),
]
CAT_SIT_TAIL = [(-0.22, -0.025), (-0.05, -0.012), (0.14, -0.012), (0.30, -0.03), (0.40, -0.065), (0.445, -0.115)]
CAT_SIT_STARS = [(0.315, -1.0, 1.0), (0.39, -0.80, 0.7), (0.285, -0.47, 0.8), (-0.06, -0.58, 0.6),
                 (-0.24, -0.22, 0.85), (0.29, -0.03, 0.55), (0.445, -0.115, 0.5)]
CAT_SIT_LINES = [(0, 1), (1, 2), (2, 3), (3, 4), (2, 5), (5, 6)]
CAT_SIT_EYE = (0.375, -0.80)

# The same cat asleep: lying with its head down on its paws, the tail wrapped round the front.
CAT_SLEEP = [
    (0.40, 0.0), (0.465, -0.05), (0.49, -0.10), (0.48, -0.14), (0.45, -0.20), (0.425, -0.24),
    (0.43, -0.33), (0.37, -0.27), (0.33, -0.27), (0.30, -0.34), (0.26, -0.26), (0.20, -0.30),
    (0.08, -0.37), (-0.08, -0.40), (-0.24, -0.37), (-0.37, -0.28), (-0.44, -0.15), (-0.44, -0.05),
    (-0.38, 0.0),
]
CAT_SLEEP_TAIL = [(-0.40, -0.025), (-0.12, -0.01), (0.16, -0.012), (0.33, -0.04), (0.42, -0.08)]
CAT_SLEEP_STARS = [(0.43, -0.33, 0.9), (0.47, -0.10, 0.65), (0.08, -0.37, 0.75), (-0.37, -0.28, 0.8),
                   (-0.42, -0.04, 0.55), (0.42, -0.08, 0.5)]
CAT_SLEEP_LINES = [(0, 1), (0, 2), (2, 3), (3, 4), (1, 5)]
CAT_SLEEP_EYE = (0.415, -0.15)


def figure(body, tail, x, ground, s, flip=False, w0=0.07, w1=0.028):
    """Body and tail as one closed outline (the tail lies in front of the paws), plus the
    tail's upper edge where it crosses the body, as an engraver would draw it."""
    bp = curve(body, x, ground, s, flip)
    tc = curve(tail, x, ground, s, flip, n=12, closed=False)
    tp = tapered(None, tc, w0 * s, w1 * s, None)
    pad = 20
    x0, y0 = np.floor(np.minimum(bp.min(0), tp.min(0))) - pad
    x1, y1 = np.ceil(np.maximum(bp.max(0), tp.max(0))) + pad
    ss = 4
    m = np.zeros((int((y1 - y0) * ss), int((x1 - x0) * ss)), np.uint8)
    mb = m.copy()
    cv2.fillPoly(mb, [np.round((bp - [x0, y0]) * ss).astype(np.int32)], 255)
    cv2.fillPoly(m, [np.round((tp - [x0, y0]) * ss).astype(np.int32)], 255)
    m = cv2.bitwise_or(m, mb)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=len)[:, 0, :].astype(float) / ss + [x0, y0]
    c = c[::3]
    c = cv2.GaussianBlur(np.vstack([c[-8:], c, c[:8]]).astype(np.float32).reshape(-1, 1, 2), (1, 7), 0)[8:-8, 0]
    # start the outline at the top of the head so it draws on from there
    c = np.roll(c, -int(np.argmin(c[:, 1])), axis=0)
    c = np.vstack([c, c[:1]])
    # the tail's upper edge over the body
    upper = tp[:len(tp) // 2]
    ins = []
    for p in upper:
        xi, yi = int((p[0] - x0) * ss), int((p[1] - y0) * ss - 3 * ss)
        ins.append(0 <= yi < mb.shape[0] and 0 <= xi < mb.shape[1] and mb[yi, xi] > 0)
    edge = upper[np.array(ins, bool)] if any(ins) else upper[:0]
    eye = place([CAT_SIT_EYE if body is CAT_SIT else CAT_SLEEP_EYE], x, ground, s, flip)[0]
    return {"outline": c.astype(float), "tail_edge": edge, "eye": eye, "sleep": body is CAT_SLEEP}


def ming_table(x, ground, s, h=1.0):
    """Side elevation of a Ming-style table: an ice-plate edged top, a waist, an
    apron with curved spandrels, and legs ending in horse-hoof feet.  h scales
    the leg height.  Returns a list of open polylines."""
    w = 1.5 * s
    top = ground - (0.86 * h + 0.14) * s
    x0, x1 = x - w / 2, x + w / 2
    L = []
    t = 0.045 * s
    L.append([(x0 - 0.03 * s, top), (x1 + 0.03 * s, top)])
    L.append([(x1 + 0.03 * s, top), (x1 + 0.03 * s, top + t * 0.55), (x1 + 0.005 * s, top + t)])
    L.append([(x0 - 0.03 * s, top), (x0 - 0.03 * s, top + t * 0.55), (x0 - 0.005 * s, top + t)])
    L.append([(x0 - 0.005 * s, top + t), (x1 + 0.005 * s, top + t)])
    waist = top + t + 0.025 * s
    L.append([(x0 + 0.02 * s, waist), (x1 - 0.02 * s, waist)])
    apron = waist + 0.07 * s
    leg = 0.07 * s
    for side in (-1, 1):
        xo = x0 if side < 0 else x1
        xi = xo - side * leg
        foot = ground
        L.append([(xo, top + t), (xo, foot - 0.06 * s), (xo + side * 0.012 * s, foot - 0.02 * s),
                  (xo - side * 0.01 * s, foot)])
        L.append([(xo - side * 0.01 * s, foot), (xi - side * 0.012 * s, foot)])
        L.append([(xi - side * 0.012 * s, foot), (xi, foot - 0.07 * s), (xi, apron + 0.01 * s)])
    # apron with spandrels curving into the legs
    sp = np.linspace(0, 1, 24)
    for side in (-1, 1):
        xi = (x0 + leg) if side < 0 else (x1 - leg)
        span = 0.32 * s
        xs = xi - side * span * sp
        ys = apron + 0.10 * s * (1 - sp) ** 2.2
        L.append(list(zip(xs, ys)))
    L.append([(x0 + leg + 0.32 * s, apron), (x1 - leg - 0.32 * s, apron)])
    return [np.array(p, float) for p in L]


# --------------------------------------------------------------- engraving ---

def _mask(poly, pad=24, ss=2):
    x0, y0 = int(poly[:, 0].min()) - pad, int(poly[:, 1].min()) - pad
    x1, y1 = int(poly[:, 0].max()) + pad, int(poly[:, 1].max()) + pad
    m = np.zeros(((y1 - y0) * ss, (x1 - x0) * ss), np.uint8)
    cv2.fillPoly(m, [np.round((poly - [x0, y0]) * ss * 16).astype(np.int32)], 255, cv2.LINE_AA, 4)
    return m, x0, y0, ss


def _lightness(m, ss):
    """How much a point inside the shape faces the light: from the gradient of the blurred mask."""
    b = cv2.GaussianBlur(m.astype(np.float32) / 255, (0, 0), 9 * ss)
    gx = cv2.Sobel(b, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(b, cv2.CV_32F, 0, 1, ksize=3)
    n = np.sqrt(gx * gx + gy * gy) + 1e-6
    # outward normal is minus the gradient
    face = (-(gx / n) * LIGHT[0] - (gy / n) * LIGHT[1])
    return np.clip(face, -1, 1)


def engraved(F, poly, color, a=1.0, reveal=1.0, spacing=5.0, depth=7, line_a=0.9, form_a=0.55,
             fill=0.035):
    """Contour + form-lines lit from the upper left.  reveal 0..1 draws the contour on, then the engraving."""
    if a <= 0.003 or reveal <= 0:
        return
    m, x0, y0, ss = _mask(poly)
    inside = m.astype(np.float32) / 255
    if reveal > 0.35 and form_a > 0:
        dt = cv2.distanceTransform(m, cv2.DIST_L2, 5) / ss
        k = dt / spacing
        f = np.abs(k - np.round(k)) * spacing
        lines = np.clip(1.0 - f / (0.55 + 0.1 * ss), 0, 1)
        lines *= ((np.round(k) >= 1) & (np.round(k) <= depth)).astype(np.float32)
        lit = _lightness(m, ss)
        lit = np.clip((lit + 0.15) / 1.0, 0, 1)
        fall = np.clip(1 - dt / (spacing * (depth + 0.5)), 0, 1) ** 0.7
        eng = lines * lit * fall * inside
        eng = cv2.resize(eng, (eng.shape[1] // ss, eng.shape[0] // ss), interpolation=cv2.INTER_AREA)
        E.add_light(F, eng, x0, y0, color, form_a * a * smooth(ramp(reveal, 0.35, 0.65)))
    if fill > 0:
        small = cv2.resize(inside, (inside.shape[1] // ss, inside.shape[0] // ss), interpolation=cv2.INTER_AREA)
        E.add_light(F, cv2.GaussianBlur(small, (0, 0), 6), x0, y0, color, fill * a * smooth(reveal))
    k = max(2, int(len(poly) * min(1.0, reveal / 0.5)))
    glow_poly(F, poly[:k], color, line_a * a, th=1, glow=0.35, sigma=2.5)


def stroke(F, pts, color, a=1.0, reveal=1.0, glow=0.35):
    if a <= 0.003 or reveal <= 0:
        return
    pts = np.asarray(pts, float)
    k = max(2, int(len(pts) * min(1.0, reveal)))
    glow_poly(F, pts[:k], color, a, th=1, glow=glow, sigma=2.5)


def tapered(F, pts, w0, w1, color, a=1.0, reveal=1.0):
    """Outline polygon of a stroke that narrows from w0 to w1 (for tails)."""
    pts = np.asarray(pts, float)
    d = np.gradient(pts, axis=0)
    n = np.column_stack([-d[:, 1], d[:, 0]])
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-9
    w = np.linspace(w0, w1, len(pts))[:, None] / 2
    return np.vstack([pts + n * w, (pts - n * w)[::-1]])


def chart(F, stars, lines, t=1e9, t0=0.0, dur=1.0, color=GOLD, a=1.0, glow=0.0, size=1.0):
    """Stars by magnitude, then the thin constellation lines between them, one by one."""
    if a <= 0.003:
        return
    u = ramp(t, t0, dur)
    n = len(stars)
    for i, (x, y, mag) in enumerate(stars):
        s = smooth(ramp(u, 0.35 * i / n, 0.3))
        if s > 0:
            orb(F, x, y, (2.6 + 3.2 * mag) * size * (1 + 0.5 * glow), mix(color, IVORY, 0.25), s * a * (0.6 + 0.4 * mag))
    prog = ramp(u, 0.35, 0.65) * len(lines)
    for k, (i, j) in enumerate(lines):
        f = min(1.0, max(0.0, prog - k))
        if f <= 0:
            continue
        p, q = np.array(stars[i][:2]), np.array(stars[j][:2])
        d = q - p
        L = np.linalg.norm(d)
        if L < 1:
            continue
        gap = min(8.0, L * 0.2) / L          # lines stop short of the stars, as on printed charts
        glow_poly(F, [p + d * gap, p + d * (gap + (1 - 2 * gap) * f)], color, (0.45 + 0.4 * glow) * a, th=1,
                  glow=0.4 + 0.6 * glow, sigma=3)


def stars_on(units_stars, x, ground, s, flip=False):
    out = []
    for ux, uy, mag in units_stars:
        p = place([(ux, uy)], x, ground, s, flip)[0]
        out.append((p[0], p[1], mag))
    return out


# -------------------------------------------------------------- silhouette ---

def silhouette(F, poly, a=1.0, rim=IVORY, rim_a=0.55, body=None, fade=None):
    """Solid figure darker than the sky behind it, with a thin rim of light on the lit side.
    fade = (y_start, y_end): the figure dissolves into the night between these heights."""
    if a <= 0.003:
        return
    m, x0, y0, ss = _mask(poly, pad=16)
    inside = m.astype(np.float32) / 255
    if fade is not None:
        ys = (np.arange(m.shape[0], dtype=np.float32) / ss + y0)[:, None]
        inside = inside * np.clip((fade[1] - ys) / (fade[1] - fade[0]), 0, 1) ** 1.3
    small = cv2.resize(inside, (inside.shape[1] // ss, inside.shape[0] // ss), interpolation=cv2.INTER_AREA)
    h, w = small.shape
    fx0, fy0 = max(x0, 0), max(y0, 0)
    fx1, fy1 = min(x0 + w, F.shape[1]), min(y0 + h, F.shape[0])
    if fx1 > fx0 and fy1 > fy0:
        sub = F[fy0:fy1, fx0:fx1]
        col = np.array([0.006, 0.008, 0.014], np.float32) if body is None else body
        mm = small[fy0 - y0:fy1 - y0, fx0 - x0:fx1 - x0, None] * a
        sub[:] = sub * (1 - mm) + col * mm
    if rim_a > 0:
        dt = cv2.distanceTransform(m, cv2.DIST_L2, 5) / ss
        lit = np.clip(_lightness(m, ss), 0, 1) ** 1.5
        r = np.exp(-dt / 1.6) * lit * inside
        r = cv2.resize(r, (inside.shape[1] // ss, inside.shape[0] // ss), interpolation=cv2.INTER_AREA)
        E.add_light(F, r, x0, y0, rim, rim_a * a)


def horizon(F, y, color, a=1.0, height=260, x0=0, x1=None):
    """A low band of light along the horizon, brightest at the line and fading upwards."""
    if a <= 0.003:
        return
    x1 = F.shape[1] if x1 is None else x1
    y0 = int(y - height)
    yy = np.arange(height + 40, dtype=np.float32)[:, None]
    prof = np.where(yy < height, np.exp(-(np.abs(height - yy) / (height * 0.38)) ** 1.6),
                    np.exp(-np.abs(yy - height) / 10))
    xs = np.linspace(-1, 1, int(x1 - x0), dtype=np.float32)[None, :]
    band = prof * (1 - 0.55 * xs ** 2)
    E.add_light(F, band.astype(np.float32), int(x0), y0, color, a)


# ------------------------------------------------------------------ people ---
# Profile portraits in the manner of 18th-century cut-paper silhouettes: head and
# shoulders facing right, the face carefully drawn (brow, nose, lips, chin).
# Units: chin to crown = 1, head centre at (0, 0), y down.

BUST = [(0.40, 1.10), (0.36, 0.95), (0.29, 0.85), (0.21, 0.78), (0.170, 0.70), (0.155, 0.61),
        (0.160, 0.535), (0.190, 0.495), (0.285, 0.472), (0.372, 0.452), (0.420, 0.410), (0.418, 0.372),
        (0.396, 0.338), (0.420, 0.312), (0.442, 0.288), (0.428, 0.262), (0.448, 0.237), (0.437, 0.202),
        (0.420, 0.170), (0.470, 0.150), (0.505, 0.120), (0.492, 0.080), (0.445, 0.010), (0.400, -0.050),
        (0.380, -0.075), (0.398, -0.125), (0.395, -0.180), (0.372, -0.285), (0.310, -0.420), (0.180, -0.512),
        (0.000, -0.555), (-0.215, -0.505), (-0.375, -0.365), (-0.460, -0.160), (-0.470, 0.040),
        (-0.420, 0.240), (-0.310, 0.400), (-0.215, 0.470), (-0.200, 0.560), (-0.215, 0.680), (-0.300, 0.800),
        (-0.480, 0.920), (-0.640, 1.010), (-0.700, 1.10)]
BUST_EYE = (0.315, -0.060)
BUST_MOUTH = (0.440, 0.262)
BUST_EAR = (-0.060, 0.080)

BUST_HAIR = {
    "bun": [[(0.330, -0.360), (0.270, -0.470), (0.140, -0.560), (-0.020, -0.595), (-0.250, -0.540),
             (-0.410, -0.380), (-0.495, -0.150), (-0.490, 0.080), (-0.420, 0.270), (-0.300, 0.300),
             (-0.180, 0.100), (-0.040, -0.150), (0.140, -0.330), (0.300, -0.330)],
            ("circle", (-0.420, -0.430), 0.185)],
    "short": [[(0.345, -0.330), (0.300, -0.470), (0.150, -0.580), (-0.030, -0.625), (-0.270, -0.570),
               (-0.440, -0.400), (-0.505, -0.150), (-0.490, 0.120), (-0.430, 0.290), (-0.320, 0.200),
               (-0.170, -0.080), (0.050, -0.300), (0.240, -0.350)]],
    "long": [[(0.335, -0.340), (0.290, -0.480), (0.150, -0.585), (-0.030, -0.630), (-0.280, -0.575),
              (-0.460, -0.400), (-0.530, -0.120), (-0.520, 0.250), (-0.490, 0.620), (-0.430, 0.930),
              (-0.240, 0.900), (-0.180, 0.520), (-0.140, 0.150), (0.000, -0.180), (0.200, -0.350)]],
    "curl": [[(0.345, -0.330), (0.320, -0.500), (0.190, -0.630), (-0.010, -0.690), (-0.290, -0.640),
              (-0.490, -0.450), (-0.560, -0.160), (-0.520, 0.120), (-0.420, 0.320), (-0.290, 0.250),
              (-0.150, -0.030), (0.050, -0.280), (0.250, -0.360)]],
}


def bust(x, y, s, hair="short", tilt=0.0, flip=False):
    """Closed outline of a profile portrait; (x, y) is the head centre, s the chin-to-crown height.
    tilt > 0 lifts the face; the shoulders stay put and the head turns about the neck."""
    piv = np.array([0.0, 0.62])
    ca, sa = math.cos(-tilt), math.sin(-tilt)

    def tf(P, w=None):
        P = np.asarray(P, float)
        if tilt:
            w = np.clip((0.75 - P[:, 1]) / 0.30, 0, 1) if w is None else np.full(len(P), w)
            d = P - piv
            r = np.column_stack([d[:, 0] * ca - d[:, 1] * sa, d[:, 0] * sa + d[:, 1] * ca]) + piv
            P = P + (r - P) * w[:, None]
        return P

    shapes = [tf(catmull(BUST, 10, closed=True))]
    for h in BUST_HAIR.get(hair, []):
        if h[0] == "circle":
            th = np.linspace(0, 2 * math.pi, 40, endpoint=False)
            shapes.append(tf(np.column_stack([h[1][0] + h[2] * np.cos(th), h[1][1] + h[2] * np.sin(th)]), 1.0))
        else:
            shapes.append(tf(catmull(h, 10, closed=True)))
    allp = np.vstack(shapes)
    pad = 0.05
    x0, y0 = allp.min(0) - pad
    x1, y1 = allp.max(0) + pad
    ss = 900
    m = np.zeros((int((y1 - y0) * ss), int((x1 - x0) * ss)), np.uint8)
    for P in shapes:
        cv2.fillPoly(m, [np.round((P - [x0, y0]) * ss).astype(np.int32)], 255, cv2.LINE_8)
    cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cs, key=len)[:, 0, :].astype(float) / ss + [x0, y0]
    c = c[::max(1, len(c) // 900)]
    c = cv2.GaussianBlur(np.vstack([c[-6:], c, c[:6]]).astype(np.float32).reshape(-1, 1, 2), (1, 5), 0)[6:-6, 0]
    c = np.roll(c, -int(np.argmax(c[:, 1] - 0.001 * c[:, 0])), axis=0)    # start at the bottom of the chest

    def scr(P):
        P = np.asarray(P, float).copy()
        if flip:
            P[:, 0] = -P[:, 0]
        return P * s + [x, y]

    out = {"outline": scr(c)}
    for k, p in (("eye", BUST_EYE), ("mouth", BUST_MOUTH), ("ear", BUST_EAR)):
        out[k] = scr(tf([p]))[0]
    return out


def backlight(F, x, y, r, color, a=1.0):
    """Soft warm light behind a figure, so a dark silhouette reads against the night."""
    if a <= 0.003:
        return
    add_sprite(F, x, y, r, color, 0.10 * a)
    add_sprite(F, x, y, r * 0.5, color, 0.08 * a)


def hill(x0, x1, y, rise, n=80):
    """A long low hill: a gentle hump on the horizon line."""
    xs = np.linspace(x0, x1, n)
    u = (xs - x0) / (x1 - x0)
    return np.column_stack([xs, y - rise * np.sin(np.pi * u) ** 1.6])


_MW = {}


def milky_way(F, a=1.0, p0=(-80, 1500), p1=(1180, 120), width=300, seed=11):
    """A diagonal band of the Milky Way: soft light, a dark dust lane, many faint stars."""
    if a <= 0.003:
        return
    key = (p0, p1, width, seed, F.shape)
    if key not in _MW:
        h, w = F.shape[:2]
        rng = np.random.default_rng(seed)
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        d = np.array(p1, np.float32) - np.array(p0, np.float32)
        L = float(np.linalg.norm(d))
        ux, uy = d / L
        along = ((xx - p0[0]) * ux + (yy - p0[1]) * uy) / L
        across = (-(xx - p0[0]) * uy + (yy - p0[1]) * ux)
        wob = 40 * np.sin(along * 7.0 + 1.3) + 25 * np.sin(along * 13.0 + 0.4)
        c = across - wob
        noise = cv2.resize(rng.random((h // 40 + 2, w // 40 + 2)).astype(np.float32), (w, h),
                           interpolation=cv2.INTER_CUBIC)
        noise2 = cv2.resize(rng.random((h // 12 + 2, w // 12 + 2)).astype(np.float32), (w, h),
                            interpolation=cv2.INTER_CUBIC)
        band = np.exp(-(c / width) ** 2 * 2.2) * (0.55 + 0.45 * noise) * (0.8 + 0.2 * noise2)
        lane = 1 - 0.65 * np.exp(-((c + 0.12 * width + 30 * np.sin(along * 9)) / (0.10 * width)) ** 2) * (
            0.6 + 0.4 * noise2)
        glow = (band * lane).astype(np.float32)
        stars = np.zeros((h, w), np.float32)
        n = 5200
        sx, sy = rng.uniform(0, w, n), rng.uniform(0, h, n)
        keep = rng.random(n) < np.exp(-((-(sx - p0[0]) * uy + (sy - p0[1]) * ux) / (width * 0.9)) ** 2)
        for x, y in zip(sx[keep], sy[keep]):
            cv2.circle(stars, (int(x * 16), int(y * 16)), int(rng.uniform(0.35, 0.9) * 16),
                       float(rng.power(3) * 0.35 + 0.05), -1, cv2.LINE_AA, 4)
        _MW[key] = (glow, stars * lane)
    glow, stars = _MW[key]
    F += glow[..., None] * (mix(GOLD, IVORY, 0.55) * 0.075 * a)
    F += stars[..., None] * (np.array([0.9, 0.92, 1.0], np.float32) * a)
