"""量子隧穿 — 南墙之外.  draw(F, t_local, g_global, frame).

One spine: what we call impossible is only improbable, and we live inside
that tiny probability."""

import math

import cv2
import numpy as np

import physics as P
from style import E
from style import (AMBER, CRIMSON, CYAN, DIM, GOLD, IVORY, WHITE, H, W, R, Stroke, T, add_light, add_sprite,
                   blank, blend, caps, chapter_mark, comet, ease_in_out, ease_out, gauge, glow_poly, hairline,
                   italic, lerp, mix, orb, ramp, serif, smooth, statement, subtitle, subtitles, window)
from timeline import BALL_HIT, FILL, LAYERS, NM1, NM2, SHOTS, START, THROUGH_SHOT

_cache = {}


def _once(key, fn):
    if key not in _cache:
        _cache[key] = fn()
    return _cache[key]


def palette(v, stops, cols):
    v = np.clip(v, 0, 1)
    return np.stack([np.interp(v, stops, [c[i] for c in cols]) for i in range(3)], -1).astype(np.float32)


COOL = ([0, 0.18, 0.45, 0.75, 1.0],
        [(0, 0, 0), (0.02, 0.05, 0.11), (0.1, 0.3, 0.52), (0.48, 0.76, 0.92), (0.96, 0.97, 0.95)])
WARM = ([0, 0.16, 0.42, 0.72, 1.0],
        [(0, 0, 0), (0.18, 0.07, 0.01), (0.7, 0.4, 0.1), (1.0, 0.8, 0.45), (1.0, 0.97, 0.9)])
WARM_WHITE = np.array([1.0, 0.93, 0.8], np.float32)
WALL_X = 1180


def underline(F, x, y, size, a, color=GOLD):
    blend(F, np.full((2, int(size * 2.3)), 255, np.uint8), x + size * 0.15, y + size * 0.18, color, 0.9 * a)


# ------------------------------------------------------------ prologue ---

def _ball(F, t, a):
    glow_poly(F, [(WALL_X, 236), (WALL_X, 364)], CYAN, 0.7 * a * smooth(ramp(t, 4.2, 0.8)), th=2, glow=0.8,
              sigma=6)
    if t < 4.6:
        return
    x = lerp(720, WALL_X - 12, ramp(t, 4.6, BALL_HIT - 4.6)) if t < BALL_HIT else \
        WALL_X - 12 - 320 * ease_out(ramp(t, BALL_HIT, 2.4))
    orb(F, x, 300, 8, GOLD, a * smooth(ramp(t, 4.5, 0.3)), core=WARM_WHITE)
    if BALL_HIT <= t < BALL_HIT + 0.7:
        orb(F, WALL_X, 300, 18, GOLD, 0.7 * (1 - (t - BALL_HIT) / 0.7) * a)


def _electron(F, t, t0, through, a):
    tau = t - t0
    if tau < 0 or tau > 3.0:
        return
    if tau < 1.0:
        orb(F, lerp(760, WALL_X - 10, tau), 300, 6, CYAN, a)
        return
    q = tau - 1.0
    fade = 1 - smooth((q - 1.4) / 0.6)
    orb(F, WALL_X, 300, 16, GOLD if through else CYAN, 0.8 * max(0.0, 1 - q / 0.5) * a)
    if through:
        orb(F, WALL_X + 12 + 400 * (q / 2.0), 300, 6, GOLD, a * fade)
    else:
        orb(F, WALL_X - 12 - 300 * ease_out(q / 2.0), 300, 6, CYAN, a * fade)


def _burst(F, q, a):
    P0 = _once("burst", lambda: np.random.default_rng(3).normal(0, 1, (160, 2)))
    if q > 3.0:
        return
    sp = 1 - math.exp(-q * 2.2)
    for vx, vy in P0:
        add_sprite(F, 960 + vx * 420 * sp, 560 + vy * 120 * sp, 1.2, GOLD, 0.6 * a * max(0.0, 1 - q / 3.0))


def prologue(F, t, g, fi):
    chapter_mark(F, window(t, 0.5, 19.6, 1.0, 0.8), "00", "序", "PROLOGUE")
    # beat 1: everybody completes the idiom
    a1 = window(t, 0.8, 9.6, 0.6, 0.8)
    if a1 > 0:
        bw = 64 * 2.45
        xe = statement(F, "不撞南墙不", 960 - bw / 2, 520, t, 1.2, 9.6, size=64, cps=6.5)
        ba = a1 * smooth(ramp(t, 2.0, 0.3)) * (1 - smooth(ramp(t, 8.9, 0.6)))
        if t < FILL:
            blank(F, xe + 6, 520, 64, ba, t)
        else:
            underline(F, xe + 6, 520, 64, ba)
            statement(F, "回头", xe + 6 + 0.42 * 64, 520, t, FILL, 9.6, size=64, cps=5, color=GOLD, align="left",
                      fade_in=0.35)
        _ball(F, t, a1)
        gauge(F, 960, 636, 520, 1.0, CYAN, "回头的概率", "TURNING BACK", "= 100%", a=a1 * smooth(ramp(t, 4.4, 0.6)))
        subtitle(F, t, 4.6, 9.4, "你一定猜到了。在我们的世界里，撞了墙，就只能回头。",
                 "You guessed it. In our world, hit a wall and you turn back.")
    # beat 2: an electron hits the wall
    a2 = window(t, 10.0, 19.9, 0.6, 0.8)
    if a2 > 0:
        glow_poly(F, [(WALL_X, 236), (WALL_X, 364)], CYAN, 0.7 * a2, th=2, glow=0.8, sigma=6)
        statement(F, "可如果撞墙的，是一个电子——", 960, 520, t, 10.0, 16.9, size=56, cps=9)
        statement(F, "撞了南墙，它却不一定回头。", 960, 520, t, 17.2, 19.9, size=56, cps=12, gold={7, 8, 9})
        for i, ts in enumerate(SHOTS):
            _electron(F, t, ts, i == THROUGH_SHOT, a2)
        hit = SHOTS[THROUGH_SHOT] + 1.0
        v = 1.0 - 0.07 * ease_out(ramp(t, hit, 1.0))
        gauge(F, 960, 636, 520, v, CYAN if t < hit else GOLD, "回头的概率", "TURNING BACK",
              "= 100%" if t < hit + 0.3 else "< 100%", a=a2 * smooth(ramp(t, 10.6, 0.6)))
        for i, ts in enumerate(SHOTS):
            q = t - (ts + 1.0)
            if q >= 0:
                x = 960 + (i - 2) * 30
                col = GOLD if i == THROUGH_SHOT else CYAN
                orb(F, x, 694, 4, col, a2 * (0.85 + 0.6 * max(0.0, 1 - q / 0.5)))
    # burst and title
    if 18.7 < t < 26:
        o = smooth(ramp(t, 18.9, 0.9)) * (1 - smooth(ramp(t, 20.0, 0.25)))
        orb(F, 960, 560, 9 + 8 * o, GOLD, 1.3 * o)
        if t >= 20.0:
            q = t - 20.0
            fade = 1 - smooth(ramp(t, 24.6, 1.2))
            hx = 960 + 980 * ease_out(min(q / 2.6, 1.0))
            ca = 0.9 * fade * (1 - smooth(ramp(q, 2.2, 0.5)))
            comet(F, hx, 560, 1, 360, GOLD, ca)
            comet(F, 1920 - hx, 560, -1, 360, GOLD, ca)
            span = 900 * ease_out(min(q / 1.2, 1.0))
            glow_poly(F, [(960 - span, 560), (960 + span, 560)], GOLD, 0.25 * fade, th=1, glow=0.4, sigma=3)
            _burst(F, q, fade)
            statement(F, "量子隧穿", 960 + 0.175 * 112, 520, t, 20.4, 26.0, size=112, wght=500, tracking=0.35,
                      cps=5.5, fade_in=0.9)
            statement(F, "南墙之外", 960 + 0.25 * 28, 616, t, 21.6, 26.0, size=28, wght=500, tracking=0.5,
                      cps=8, color=GOLD, fade_in=0.6)
            caps("QUANTUM TUNNELING   ·   BEYOND THE WALL", 13, 0.42).draw(
                F, 960, 656, DIM, smooth(ramp(t, 22.4, 0.8)) * fade, align="center")


# ----------------------------------------------------------- 01 wall ----

HX = np.linspace(160, 1760, 900)
HC, HW, HG, HH = 1000.0, 150.0, 700.0, 300.0
BALL_E = 0.68


def hill_y(x):
    return HG - HH * np.exp(-((np.asarray(x) - HC) / HW) ** 2)


def _ball_path():
    x_turn = HC - HW * math.sqrt(math.log(1 / BALL_E))
    xs = np.linspace(420, x_turn, 3000)
    v = np.sqrt(np.clip(2 * (BALL_E - np.exp(-((xs - HC) / HW) ** 2)), 1e-5, None))
    tt = np.concatenate([[0], np.cumsum(np.diff(xs) / (0.5 * (v[1:] + v[:-1])))])
    return xs, tt / tt[-1], x_turn


def wall(F, t, g, fi):
    chapter_mark(F, window(t, 0.3, 16, 1.0, 0.8), "01", "墙", "THE WALL")
    A = window(t, 0, 16, 0.6, 0.9) * (1 - 0.85 * smooth(ramp(t, 10.4, 0.8)))
    n = max(2, int(len(HX) * ease_in_out(ramp(t, 0.3, 2.0))))
    pts = np.column_stack([HX[:n], hill_y(HX[:n])])
    glow_poly(F, pts, GOLD, 0.85 * A, th=2, glow=0.8, sigma=5, bounds=(140, 360, 1780, 720))
    if n < len(HX):
        orb(F, pts[-1, 0], pts[-1, 1], 6, GOLD, A)
    yE = HG - BALL_E * HH
    ea = A * smooth(ramp(t, 2.2, 0.8))
    if ea > 0:
        s = Stroke(150, yE - 3, 1770, yE + 4)
        s.dashed((160, yE), (1760, yE), 6, 8)
        s.light(F, IVORY, 0.28 * ea)
        italic("E", 22).draw(F, 168, yE - 10, IVORY, 0.7 * ea)
        italic("V", 22).draw(F, HC + 14, HG - HH - 12, GOLD, 0.8 * ea)
    xs, tt, _ = _once("ball", _ball_path)
    q = (t - 1.6) / 7.6
    u = min(max(q * 2 if q < 0.5 else (1 - q) * 2, 0.0), 1.0)
    bx = float(np.interp(u, tt, xs))
    by = float(hill_y(bx))
    dydx = float(2 * HH * (bx - HC) / HW ** 2 * math.exp(-((bx - HC) / HW) ** 2))
    nrm = math.hypot(1, dydx)
    orb(F, bx + dydx / nrm * 11, by - 11 / nrm, 8, GOLD, A * smooth(ramp(t, 1.2, 0.5)), core=WARM_WHITE)
    fb = A * smooth(ramp(t, 6.0, 1.0))
    if fb > 0:
        xr = HX[hill_y(HX) < yE]
        poly = np.vstack([np.column_stack([xr, hill_y(xr)]), [[xr[-1], yE], [xr[0], yE]]])
        s = Stroke(xr[0] - 4, HG - HH - 4, xr[-1] + 5, yE + 4)
        s.fill(poly)
        s.light(F, CRIMSON, 0.55 * fb)
        s.glow(F, CRIMSON, 0.5 * fb, 8)
        serif("禁区", 16, 500, 0.3).draw(F, HC + 120, HG - HH + 40, mix(CRIMSON, IVORY, 0.5), fb)
        caps("FORBIDDEN", 10, 0.4).draw(F, HC + 120, HG - HH + 60, DIM, fb)
    subtitles(F, t, [(1.0, 5.4, "在经典世界里，墙的含义很简单：",
                      "In the classical world, a wall means one thing:"),
                     (5.8, 10.4, "能量不够，就过不去。不是很难——而是不可能。",
                      "Without enough energy, there is no way through. Not hard — impossible.")])
    statement(F, "不可能", 960 + 0.25 * 96, 520, t, 10.8, 15.9, size=96, wght=400, tracking=0.5, cps=3.2,
              fade_in=0.7)


# ----------------------------------------------------------- 02 wave ----

GX, GY = np.meshgrid((np.arange(480) + 0.5) * 4, (np.arange(270) + 0.5) * 4)


def draw_field(F, I, pal, a=1.0, gamma=0.55):
    if a <= 0.003:
        return
    col = palette(np.clip(I, 0, None) ** gamma, *pal)
    F += cv2.resize(col, (W, H), interpolation=cv2.INTER_CUBIC) * a


def packet_field(cx, cy, sig, phase, k=2 * math.pi / 40):
    env = np.exp(-((GX - cx) ** 2 + (GY - cy) ** 2) / (4 * sig ** 2))
    re = env * np.cos(k * (GX - cx) - phase)
    return 0.5 * env ** 2 + 0.5 * re ** 2


def _detections():
    rng = np.random.default_rng(12)
    return rng.normal(0, 1, (40, 2)), np.sort(rng.uniform(6.0, 11.0, 40))


def wave(F, t, g, fi):
    chapter_mark(F, window(t, 0.3, 18, 1.0, 0.8), "02", "波", "THE WAVE")
    A = window(t, 0, 18, 0.6, 1.0)
    cx = 960 + 140 * ease_in_out(ramp(t, 11.6, 5.6))
    o = smooth(ramp(t, 0.3, 1.0)) * (1 - smooth(ramp(t, 2.4, 1.2)))
    orb(F, 960, 470, 9, CYAN, 1.2 * o * A)
    sig = lerp(10, 78, ease_in_out(ramp(t, 2.0, 2.6)))
    wa = smooth(ramp(t, 2.0, 1.0)) * A
    if wa > 0:
        draw_field(F, packet_field(cx, 470, sig, 5.0 * t), COOL, 0.72 * wa, gamma=0.7)
    pts, tms = _once("detect", _detections)
    for (dx, dy), td in zip(pts, tms):
        q = t - td
        if 0 <= q < 0.9:
            x, y = cx + dx * 78, 470 + dy * 78
            orb(F, x, y, 7, GOLD, 1.3 * A * (1 - q / 0.9) ** 1.5)
            r = 6 + 30 * ease_out(q / 0.9)
            s = Stroke(x - r - 3, y - r - 3, x + r + 4, y + r + 4)
            s.circle((x, y), r, th=1)
            s.light(F, GOLD, 0.8 * A * (1 - q / 0.9))
    wl = A * smooth(ramp(t, 11.6, 1.2))
    if wl > 0:
        glow_poly(F, [(1480, 250), (1480, 690)], CYAN, 0.6 * wl, th=2, glow=0.9, sigma=7)
    subtitles(F, t, [(1.0, 5.6, "但电子不是一颗小球，它是一团「可能」。",
                      "But an electron is not a ball. It is a cloud of possibilities."),
                     (6.0, 10.8, "它会出现在哪里，只有概率可言。", "Where it will show up, only probability can say."),
                     (11.2, 17.2, "而这团「可能」撞上墙时，并不会就此消失。",
                      "And when those possibilities meet a wall, they do not simply vanish.")])


# ------------------------------------------------------------ 03 seep ----

SIM_Y0 = 150
CELL = W / P.NX


def sim_frame(t):
    if t <= 11.5:
        return min(max(t / 13.0 * 360, 0.0), 360.0)
    return min(318.5 + 41.5 * (1 - math.exp(-(t - 11.5) / 1.8)), 360.0)


def _glass():
    yy = np.arange(SIM_Y0, SIM_Y0 + 720)
    prof = ((np.clip((yy - SIM_Y0) / 140, 0, 1) * np.clip((SIM_Y0 + 720 - yy) / 140, 0, 1)) ** 1.5).astype(np.float32)
    return prof, cv2.GaussianBlur(np.pad(prof[:, None], ((0, 0), (12, 12))), (0, 0), 5)


def _sim(F, t, A):
    D = _once("sim", P.packet2d)
    L = np.zeros_like(F)
    x0w, x1w = P.BAR_X0 * CELL, (P.BAR_X0 + P.BAR_W) * CELL
    prof, edge_glow = _once("glass", _glass)
    wa = A * smooth(ramp(t, 0.1, 1.0))
    blend(L, np.repeat(prof[:, None], int(x1w - x0w) + 1, 1), int(x0w), SIM_Y0,
          np.array([0.03, 0.05, 0.09], np.float32), 0.9 * wa)
    for xe in (x0w, x1w):
        add_light(L, prof[:, None], int(round(xe)), SIM_Y0, mix(CYAN, WHITE, 0.4), 0.55 * wa)
        add_light(L, edge_glow, int(round(xe)) - 12, SIM_Y0, CYAN, 0.9 * wa)
    f = sim_frame(t)
    i0 = int(f)
    i1 = min(i0 + 1, 360)
    fr = f - i0
    p = D["prob"][i0].astype(np.float32) * (1 - fr) + D["prob"][i1].astype(np.float32) * fr
    r = D["re"][i0].astype(np.float32) * (1 - fr) + D["re"][i1].astype(np.float32) * fr
    p0 = _once("p0", lambda: float(D["prob"][0].astype(np.float32).max()))
    v = np.clip((0.55 * p + 0.45 * r * r) / p0, 0, None)[32:224] ** 0.62
    side = np.clip((np.arange(P.NX) - (P.BAR_X0 + P.BAR_W)) / 6.0, 0, 1)[None, :, None].astype(np.float32)
    col = palette(v, *COOL) * (1 - side) + palette(v, *WARM) * side
    L[SIM_Y0:SIM_Y0 + 720] += cv2.resize(col, (W, 720), interpolation=cv2.INTER_CUBIC) * 0.85 * wa
    s = 1.0 + 0.06 * ease_in_out(t / 16)
    M = np.float32([[s, 0, 960 * (1 - s)], [0, s, 510 * (1 - s)]])
    F += cv2.warpAffine(L, M, (W, H), flags=cv2.INTER_LINEAR)


SL_L, SL_N, SL_W, SL_Y0, SL_Y1, BEAM_Y = 760.0, 20, 40.0, 340.0, 690.0, 515.0


def layers_done(g):
    n = 0.0
    for k, tk in enumerate(LAYERS):
        if g >= tk:
            n = k + min(1.0, (g - tk) / 0.22)
    return n


def decimal_parts(v):
    if v >= 0.999:
        return "", "1"
    e = math.floor(math.log10(v))
    sig = f"{v / 10 ** e:.1f}".replace(".", "").rstrip("0") or "0"
    return "0." + "0" * (-e - 1), sig


def _beam_level(n):
    return (math.log10(max(P.transmission(n * 0.1) if n > 0 else 1.0, 1e-12)) + 10) / 10


def _slices(F, t, g, a):
    if a <= 0.003:
        return
    n = layers_done(g)
    k = int(n)
    # plates
    for i in range(SL_N):
        x = SL_L + i * SL_W
        lit = smooth(n - i) if n > i else 0.0
        plate = Stroke(x - 1, SL_Y0 - 1, x + SL_W + 2, SL_Y1 + 2)
        plate.line((x, SL_Y0), (x, SL_Y1))
        plate.light(F, mix(CYAN, IVORY, 0.4), (0.16 + 0.22 * lit * _beam_level(i + 1)) * a)
        blend(F, np.full((int(SL_Y1 - SL_Y0), int(SL_W) - 2), 255, np.uint8), x + 1, SL_Y0,
              np.array([0.04, 0.06, 0.1], np.float32), 0.35 * a)
    plate = Stroke(SL_L + SL_N * SL_W - 1, SL_Y0 - 1, SL_L + SL_N * SL_W + 2, SL_Y1 + 2)
    plate.line((SL_L + SL_N * SL_W, SL_Y0), (SL_L + SL_N * SL_W, SL_Y1))
    plate.light(F, mix(CYAN, IVORY, 0.4), 0.16 * a)
    # beam
    ent = smooth(ramp(g, LAYERS[0] - 0.9, 0.8))
    glow_poly(F, [(160, BEAM_Y), (SL_L, BEAM_Y)], GOLD, 0.95 * a * ent, th=3, glow=1.0, sigma=6)
    front = SL_L + n * SL_W
    if n > 0:
        xs = np.arange(SL_L, front, 4.0)
        for x0 in xs:
            lv = _beam_level((x0 - SL_L) / SL_W + 0.5)
            glow_poly(F, [(x0, BEAM_Y), (min(x0 + 4, front), BEAM_Y)], GOLD, a * lv ** 1.6, th=3, glow=0.9,
                      sigma=6)
        if n < SL_N:
            orb(F, front, BEAM_Y, 7, GOLD, a * _beam_level(n) ** 1.2)
    if n >= SL_N - 0.01:
        ex = smooth(ramp(g, LAYERS[-1] + 0.3, 1.2))
        lv = _beam_level(SL_N) ** 1.6
        glow_poly(F, [(SL_L + SL_N * SL_W, BEAM_Y), (SL_L + SL_N * SL_W + 600 * ex, BEAM_Y)], GOLD, a * lv,
                  th=2, glow=0.9, sigma=5)
    # readouts
    serif(f"第 {k} 层", 20, 500, 0.2).draw(F, 170, 252, IVORY, 0.85 * a)
    caps("LAYER", 10, 0.4).draw(F, 170, 274, DIM, a)
    serif(f"厚度 {k / 10:.1f} 纳米", 20, 500, 0.2).draw(F, 1750, 252, IVORY, 0.85 * a, align="right")
    caps("THICKNESS", 10, 0.4).draw(F, 1750, 274, DIM, a, align="right")
    zeros, sig = decimal_parts(P.transmission(k * 0.1) if k > 0 else 1.0)
    tz = T(zeros, "stix", 64, None, per_char=False) if zeros else None
    ts_ = T(sig, "stix", 64, None, per_char=False)
    wz = tz.width if tz else 0
    x0 = 960 - (wz + ts_.width) / 2
    if tz:
        tz.draw(F, x0, 268, IVORY, 0.55 * a)
    ts_.draw(F, x0 + wz, 268, GOLD, a)
    caps("PROBABILITY LEFT", 10, 0.45).draw(F, 960, 296, DIM, 0.9 * a, align="center")
    for tg, word in ((NM1, "一纳米：约七千分之一"), (NM2, "两纳米：约两亿分之一")):
        wa = a * window(g, tg, tg + 2.5 if tg == NM1 else tg + 3.2, 0.5, 0.4)
        serif(word, 26, 500, 0.3).draw(F, 960, 772, GOLD, wa, align="center")
    for i in range(SL_N):
        x = SL_L + i * SL_W + SL_W / 2
        if n > i:
            orb(F, x, 738, 3, GOLD, a * (0.35 + 0.65 * _beam_level(i + 1)))
        else:
            blend(F, np.full((1, 6), 255, np.uint8), x - 3, 738, DIM, 0.5 * a)


def _glyph_points(text, size, tracking, cx, y, n, seed):
    tx = serif(text, size, 400, tracking)
    x0 = cx - tx.width / 2
    pts = []
    for m, dx, dy in tx.items:
        yy, xx = np.nonzero(m > 120)
        pts.append(np.column_stack([xx + x0 + dx, yy + y + dy]))
    pts = np.concatenate(pts).astype(np.float32)
    return pts[np.random.default_rng(seed).integers(0, len(pts), n)]


def _morph(F, t, a=1.0):
    """'不可能' bursts into particles that settle as '概率很小' (t is local morph time)."""
    cx1 = 960 + 0.25 * 100
    if t < 2.4:
        statement(F, "不可能", cx1, 560, t, 0.2, None, size=100, tracking=0.5, cps=3.4, fade_in=0.7,
                  a=a * (1 - smooth(ramp(t, 1.8, 0.5))))
    src = _once("src", lambda: _glyph_points("不可能", 100, 0.5, cx1, 560, 5200, 1))
    dst = _once("dst", lambda: _glyph_points("概率很小", 100, 0.5, cx1, 560, 5200, 2))
    jit = _once("jit", lambda: np.random.default_rng(4).normal(0, 1, (5200, 2)).astype(np.float32))
    if 1.7 < t < 4.7:
        u = ease_in_out(ramp(t, 2.0, 1.9))
        pts = src + (dst - src) * u + jit * 70 * math.sin(math.pi * u)
        pa = a * smooth(ramp(t, 1.7, 0.5)) * (1 - smooth(ramp(t, 4.0, 0.6)))
        xi, yi = pts[:, 0].astype(int), pts[:, 1].astype(int)
        y0, y1 = 300, 820
        ok = (xi > 0) & (xi < W) & (yi > y0) & (yi < y1)
        dens = np.bincount((yi[ok] - y0) * W + xi[ok], minlength=(y1 - y0) * W).reshape(y1 - y0, W)
        dens = dens.astype(np.float32)
        lm = cv2.GaussianBlur(dens, (0, 0), 0.8) * 1.6 + cv2.GaussianBlur(dens, (0, 0), 6) * 1.8
        F[y0:y1] += lm[..., None] * mix(IVORY, GOLD, smooth(u))[None, None] * pa
    if t > 3.9:
        statement(F, "概率很小", cx1, 560, t, 3.9, 7.9, size=100, tracking=0.5, cps=40, fade_in=0.7, color=GOLD,
                  a=a)


def seep(F, t, g, fi):
    chapter_mark(F, window(t, 0.3, 44, 1.0, 0.8), "03", "渗", "THE SEEPING")
    sa = window(t, 0, 16.4, 0.8, 1.2)
    if sa > 0:
        _sim(F, t, sa)
    _slices(F, t, g, window(t, 16.0, 31.8, 1.0, 0.8))
    fa = window(t, 31.4, 36.2, 1.0, 0.6)
    if fa > 0:
        base = R([("T", "stix_it", 88, None), ("  ≈  ", "math", 80, None), ("e", "stix_it", 88, None)])
        sup = R([("−2", "stix", 46, None), ("κa", "stix_it", 46, None)])
        x0 = 960 - (base.width + sup.width) / 2
        base.draw(F, x0, 530, GOLD, fa)
        sup.draw(F, x0 + base.width + 4, 484, GOLD, fa)
        zl = serif("隧穿概率", 15, 500, 0.4)
        cl = caps("TUNNELING PROBABILITY", 10, 0.4)
        lx = 960 - (zl.width + 14 + cl.width) / 2
        zl.draw(F, lx, 594, DIM, fa)
        cl.draw(F, lx + zl.width + 14, 593, DIM, fa)
    if t > 36.0:
        _morph(F, t - 36.0, window(t, 36.0, 44.0, 0.01, 0.6))
    subtitles(F, t, [(1.0, 5.4, "它渗进了墙里——", "It seeps into the wall —"),
                     (5.8, 10.6, "越往里越微弱，却永远不会彻底熄灭。",
                      "fainter the deeper it goes, yet never quite extinguished."),
                     (11.0, 15.6, "墙够薄，另一侧就会留下一缕微光。",
                      "If the wall is thin enough, a glimmer is left on the other side."),
                     (16.8, 21.6, "把墙切成薄片，每片只有 0.1 纳米。",
                      "Slice the wall into layers a tenth of a nanometre thick."),
                     (22.0, 24.9, "越往深处，每一层都只放过约三分之一。",
                      "Deeper in, each layer lets only about a third through."),
                     (25.2, 30.8, "一层接一层，概率塌得飞快——却永远不会变成零。",
                      "Layer after layer the odds collapse — but never reach zero."),
                     (31.6, 35.8, "这就是隧穿的本质：不是一道门槛，而是一场指数级的衰减。",
                      "That is the heart of tunneling: not a threshold, but an exponential fade."),
                     (37.6, 43.4, "所谓「不可能」，其实只是「概率很小」。",
                      "What we call impossible is merely improbable.")])


# ----------------------------------------------------------- 04 zeros ----

Z_X0, Z_X1, Z_Y0, Z_DY = 150.0, 1790.0, 330.0, 46.0


def _zero_row():
    one = T("0", "stix", 34, None, per_char=False)
    n = int((Z_X1 - Z_X0) / (one.width + 6))
    return T("0" * n, "stix", 34, None, tracking=6 / 34, per_char=False), n


def zeros(F, t, g, fi):
    chapter_mark(F, window(t, 0.3, 16, 1.0, 0.8), "04", "零", "THE ZEROS")
    A = window(t, 0, 16, 0.6, 1.2)
    row, per = _once("zrow", _zero_row)
    u = max(0.0, t - 0.8)
    count = 3.0 * u + 26.0 * u ** 2 + 14.0 * u ** 3
    za = A * (1 - smooth(ramp(t, 10.6, 1.4)))
    head = T("0.", "stix", 34, None, per_char=False)
    rows_full = int(count // per)
    scroll = max(0.0, (count / per - 8.5)) * Z_DY
    head.draw(F, Z_X0 - head.width - 4, Z_Y0 - scroll, GOLD, za)
    m = row.items[0][0]
    dx, dy = row.items[0][1], row.items[0][2]
    for r in range(max(0, int(scroll // Z_DY) - 1), rows_full + 1):
        y = Z_Y0 + r * Z_DY - scroll
        if y < 120 or y > 880:
            continue
        frac = min(1.0, count / per - r)
        if frac <= 0:
            continue
        wcut = int(m.shape[1] * frac)
        age = (count / per - r)
        al = za * (0.25 + 0.5 * math.exp(-max(age - 1, 0) / 3.0))
        edge = np.clip((y - 120) / 120, 0, 1) * np.clip((880 - y) / 120, 0, 1)
        blend(F, m[:, :wcut], Z_X0 + dx, y + dy, IVORY, al * edge)
    subtitles(F, t, [(0.6, 5.2, "那么，一个人穿过一堵墙的概率，有多大？",
                      "So what are the odds of a person walking through a wall?"),
                     (5.6, 10.6, "就算从宇宙诞生起，每秒撞一次墙，也几乎不可能撞穿一次。",
                      "Hit it once a second since the Big Bang, and you'd almost surely never get through."),
                     (11.0, 15.6, "所以在我们的世界里，撞了南墙，就只能回头。",
                      "So in our world, hit a wall, and you turn back.")])


# ------------------------------------------------------------- 05 sun ----

def _noise_tex():
    rng = np.random.default_rng(31)
    out = []
    for s in (5, 11):
        n = cv2.GaussianBlur(rng.normal(0, 1, (720, 1200)).astype(np.float32), (0, 0), s / 5)
        out.append(n / np.abs(n).max())
    return out


def sun_disc(F, t, a, cx, cy, R0, heat=1.0):
    if a <= 0.003:
        return
    n1, n2 = _once("noise", _noise_tex)
    h, w = 540, 960
    ys, xs = _once("half_grid", lambda: tuple(np.mgrid[0:540, 0:960].astype(np.float32) * 2))
    r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) / R0
    g1 = np.roll(n1, int(t * 6), 1)[:h, :w]
    g2 = np.roll(n2, -int(t * 3), 0)[:h, :w]
    edge = np.clip((1 - r) * R0 / 3.0, 0, 1).astype(np.float32)
    mu = np.sqrt(np.clip(1 - np.minimum(r, 1) ** 2, 0, 1))
    I = (0.42 + 0.58 * mu ** 0.6) * (0.9 + 0.12 * (0.55 * g1 + 0.45 * g2)) * heat
    col = palette(I, [0, 0.3, 0.62, 0.85, 1.0], [(0, 0, 0), (0.45, 0.1, 0.02), (0.92, 0.42, 0.1),
                                                   (1.0, 0.72, 0.36), (1.0, 0.93, 0.78)])
    cor = np.clip(r - 1, 0, None)
    corona = (np.exp(-cor * R0 / 8) * 0.12 + np.exp(-cor * R0 / 60) * 0.12 + np.exp(-cor * R0 / 220) * 0.07)
    corona = (corona * (1 - edge) * heat ** 1.5)[..., None] * np.array([1.0, 0.6, 0.25], np.float32)
    rim = np.exp(-np.abs(r - 1) * R0 / 2.5) * (1 - heat) * 0.25
    disc = cv2.resize(edge, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
    F *= 1 - disc * a
    F += cv2.resize(col, (W, H), interpolation=cv2.INTER_LINEAR) * disc * 0.72 * a
    F += cv2.resize(corona + rim[..., None] * np.array([0.6, 0.15, 0.05], np.float32), (W, H),
                    interpolation=cv2.INTER_LINEAR) * 0.72 * a


def _earth_sprite(rad=27):
    n = int(rad * 2 + 40)
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) - (n - 1) / 2
    rr = np.sqrt(xx ** 2 + yy ** 2) / rad
    z = np.sqrt(np.clip(1 - rr ** 2, 0, 1))
    lit = np.clip(-(xx / rad) * 0.9 + z * 0.3, 0, 1) * (rr < 1)
    rng = np.random.default_rng(8)
    cloud = cv2.GaussianBlur(rng.random((n, n)).astype(np.float32), (0, 0), 2.0)
    cloud = np.clip((cloud - 0.5) * 6, 0, 1)
    body = np.stack([0.12 + 0.6 * cloud, 0.3 + 0.55 * cloud, 0.62 + 0.35 * cloud], -1) * lit[..., None]
    halo = np.exp(-np.clip(rr - 1, 0, None) * rad / 4) * (rr >= 0.92) * np.clip(-(xx / rad) + 0.6, 0, 1)
    return (body + halo[..., None] * np.array([0.25, 0.55, 0.9], np.float32) * 0.6).astype(np.float32)


def sun(F, t, g, fi):
    chapter_mark(F, window(t, 0.3, 42, 1.0, 0.8), "05", "光", "SUNLIGHT")
    subtitle(F, t, 1.0, 5.0, "可如果这个概率，真的等于零——", "But if that probability were truly zero —")
    heat = smooth(ramp(t, 8.6, 2.2))
    pan = ease_in_out(ramp(t, 30.0, 4.5))
    cx, cy = lerp(960, 470, pan), 470
    R0 = lerp(250 + 12 * ease_out(ramp(t, 8.6, 20)), 170, pan)
    da = smooth(ramp(t, 6.4, 1.6))
    sun_disc(F, t, da, cx, cy, R0, max(heat, 0.02))
    statement(F, "太阳，就是黑的。", 960, 540, t, 5.4, 8.4, size=60, cps=6)
    subtitle(F, t, 5.8, 8.3, "", "— the Sun would be dark.")
    if 8.6 <= t < 10.6:
        q = (t - 8.6) / 2.0
        add_sprite(F, cx, cy, 80 + 260 * ease_out(q), GOLD, 0.6 * (1 - q))
    if pan > 0:
        ea = smooth(ramp(t, 31.0, 2.0))
        sp = _once("earth", _earth_sprite)
        ex, ey = 1470.0, 470.0
        E.add_rgb(F, sp, ex - sp.shape[1] / 2, ey - sp.shape[0] / 2, ea)
        rng = np.random.default_rng(17)
        for i in range(28):
            ph = (t * 0.22 + rng.random()) % 1.0
            x = lerp(cx + R0 + 20, ex - 26, ph)
            y = cy + rng.normal(0, 26) * (1 - ph) + 0 * ph
            add_sprite(F, x, y, 1.4, GOLD, 0.45 * ea * math.sin(math.pi * ph))
        T("地球", "serif", 14, 500, tracking=0.3).draw(F, ex, ey + 62, DIM, ea, align="center")
    subtitles(F, t, [(9.4, 14.0, "太阳中心约一千五百万度，按经典物理，远不足以点燃聚变。",
                      "The Sun's core is about fifteen million degrees — classically, far too cold to fuse."),
                     (14.4, 19.0, "是隧穿，让质子穿过彼此的斥力，聚变发光。",
                      "Tunneling lets protons slip through their mutual repulsion, fuse, and shine."),
                     (19.4, 24.2, "而这种穿越极其罕见：一个质子，平均要等上几十亿年。",
                      "Yet it is so rare that a proton waits, on average, billions of years."),
                     (24.6, 30.0, "正因如此，太阳没有一下子烧完，而是稳稳地燃烧了四十六亿年——",
                      "So the Sun did not burn out in a flash, but has shone steadily for 4.6 billion years —"),
                     (30.4, 34.6, "久到足以让生命出现。", "long enough for life to appear."),
                     (35.0, 41.4, "你此刻感受到的每一缕阳光，都源自无数次「几乎不可能」的穿越。",
                      "Every ray of sunlight on your skin began with countless 'almost impossible' crossings.")])


# ------------------------------------------------------------ epilogue ---

def epilogue(F, t, g, fi):
    chapter_mark(F, window(t, 0.3, 10.0, 1.0, 0.8), "06", "终", "EPILOGUE")
    bw = 64 * 2.45
    xe = statement(F, "不撞南墙不", 960 - bw / 2, 520, t, 1.0, 9.8, size=64, cps=6.5)
    blank(F, xe + 6, 520, 64, smooth(ramp(t, 1.9, 0.3)) * (1 - smooth(ramp(t, 9.2, 0.6))), t)
    subtitle(F, t, 3.4, 9.4, "这一次，答案不止一个。", "This time, there is more than one answer.")
    if t > 10.2:
        statement(F, "量子隧穿", 960 + 0.25 * 44, 530, t, 10.5, None, size=44, wght=500, tracking=0.5, cps=30,
                  fade_in=0.9)
        fa = smooth(ramp(t, 11.1, 0.8))
        hairline(F, 960 - 90 * fa, 562, 960 + 90 * fa, GOLD, 0.6 * fa)
        caps("QUANTUM TUNNELING   ·   BEYOND THE WALL", 12, 0.45).draw(F, 960, 596, DIM, fa, align="center")


SCENE_FUNCS = {"prologue": prologue, "wall": wall, "wave": wave, "seep": seep, "zeros": zeros, "sun": sun,
               "epilogue": epilogue}
