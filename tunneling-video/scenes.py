"""量子隧穿 — cinematic edition.  draw(F, t_local, g_global, frame)."""

import math

import cv2
import numpy as np

import physics as P
from style import (AMBER, CRIMSON, CYAN, DIM, GOLD, IVORY, WHITE, H, W, R, Stroke, T, add_light, add_sprite,
                   blank, blend, caps, chapter_mark, comet, ease_in_out, ease_out, gauge, glow_poly, hairline,
                   italic, lerp, mix, orb, ramp, serif, smooth, statement, subtitle, subtitles, window)
from timeline import START

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


# ------------------------------------------------------------ prologue ---

WALL_X = 1180
REEL_WORDS = ["弹回", "穿过", "弹回", "弹回", "穿过", "弹回", "弹回", "弹回", "穿过", "弹回", "穿过", "弹回",
              "弹回", "弹回", "穿过", "弹回"]


def _shot(F, t, t0, through, a):
    """One electron fired at the little wall: it reflects, or mostly passes."""
    tau = t - t0
    if tau < 0 or tau > 3.2:
        return
    if tau < 1.0:
        orb(F, lerp(760, WALL_X - 10, tau), 300, 6, CYAN, a)
        return
    q = tau - 1.0
    orb(F, WALL_X, 300, 16, CYAN, 0.8 * max(0.0, 1 - q / 0.5) * a)
    main, ghost = (0.22, 1.0) if through else (1.0, 0.2)
    fade = 1 - smooth((q - 1.6) / 0.6)
    orb(F, WALL_X - 10 - 260 * ease_out(q / 2.2), 300, 6, CYAN, a * main * fade)
    orb(F, WALL_X + 10 + 380 * (q / 2.2), 300, 6, GOLD if through else CYAN, a * ghost * fade)


def _burst(F, q, a):
    P0 = _once("burst", lambda: np.random.default_rng(3).normal(0, 1, (160, 2)))
    if q > 3.0:
        return
    sp = 1 - math.exp(-q * 2.2)
    for vx, vy in P0:
        add_sprite(F, 960 + vx * 420 * sp, 560 + vy * 120 * sp, 1.2, GOLD, 0.6 * a * max(0.0, 1 - q / 3.0))


def prologue(F, t, g, fi):
    aA = window(t, 0.6, 8.3, 0.8, 0.7)
    if aA > 0:
        wl = smooth(ramp(t, 0.6, 1.0)) * aA
        glow_poly(F, [(WALL_X, 236), (WALL_X, 364)], CYAN, 0.7 * wl, th=2, glow=0.8, sigma=6)
        x = lerp(720, WALL_X - 12, ramp(t, 1.0, 1.6)) if t < 2.6 else WALL_X - 12 - 320 * ease_out(ramp(t, 2.6, 2.4))
        orb(F, x, 300, 8, GOLD, aA * smooth(ramp(t, 0.9, 0.3)), core=WARM_WHITE)
        if 2.6 <= t < 3.3:
            orb(F, WALL_X, 300, 18, GOLD, 0.7 * (1 - (t - 2.6) / 0.7) * aA)
        statement(F, "球撞向墙，弹了回来。", 960, 500, t, 0.8, 8.2, size=58, cps=8)
        gauge(F, 960, 604, 520, 0.0, CYAN, "穿墙概率", "PROBABILITY", "= 0", a=smooth(ramp(t, 2.0, 0.6)) * aA)
        subtitle(F, t, 2.8, 7.8, "你一定猜到了。在经典世界里，这毫无悬念。",
                 "You guessed it. In the classical world, there is no suspense.")

    aB = window(t, 8.6, 18.9, 0.6, 0.7)
    if aB > 0:
        wl = smooth(ramp(t, 9.0, 0.8)) * aB
        glow_poly(F, [(WALL_X, 236), (WALL_X, 364)], CYAN, 0.7 * wl, th=2, glow=0.8, sigma=6)
        _shot(F, t, 10.6, False, aB)
        _shot(F, t, 14.6, True, aB)
        bw = 58 * 2.45
        xe = statement(F, "电子撞向墙，它会", 960 - bw / 2, 500, t, 8.8, 18.9, size=58, cps=8)
        ba = aB * smooth(ramp(t, 9.8, 0.4)) * (1 - smooth(ramp(t, 18.1, 0.6)))
        blank(F, xe + 6, 500, 58, ba, t, words=REEL_WORDS, t_reel=10.2, speed=5.5)
        v = 0.2 * ease_in_out(ramp(t, 10.2, 1.4)) + 0.015 * math.sin(t * 9.0) * smooth(ramp(t, 11.6, 0.4))
        gauge(F, 960, 604, 520, v, GOLD, "穿墙概率", "PROBABILITY", "= ?", a=aB * smooth(ramp(t, 9.4, 0.6)))
        subtitles(F, t, [(10.8, 14.0, "这一次，谁也说不准。", "This time, no one can say for sure."),
                         (14.3, 18.7, "它多半会弹回——偶尔，却会直接穿过去。",
                          "Most times it bounces back. Now and then, it simply passes through.")])

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
            statement(F, "量子隧穿", 960 + 0.175 * 112, 520, t, 20.6, 26.0, size=112, wght=500, tracking=0.35,
                      cps=5.5, fade_in=0.9)
            statement(F, "不可能之门", 960 + 0.25 * 28, 616, t, 21.7, 26.0, size=28, wght=500, tracking=0.5,
                      cps=9, color=GOLD, fade_in=0.6)
            caps("QUANTUM TUNNELING   ·   THE DOOR OF THE IMPOSSIBLE", 13, 0.42).draw(
                F, 960, 656, DIM, smooth(ramp(t, 22.6, 0.8)) * fade, align="center")


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
    chapter_mark(F, window(t, 0.3, 20, 1.0, 0.8), "01", "墙", "THE WALL")
    dim = 1 - 0.85 * smooth(ramp(t, 16.4, 0.8))
    A = window(t, 0, 20, 0.6, 0.9) * dim
    n = max(2, int(len(HX) * ease_in_out(ramp(t, 0.4, 2.4))))
    pts = np.column_stack([HX[:n], hill_y(HX[:n])])
    glow_poly(F, pts, GOLD, 0.85 * A, th=2, glow=0.8, sigma=5, bounds=(140, 360, 1780, 720))
    if n < len(HX):
        orb(F, pts[-1, 0], pts[-1, 1], 6, GOLD, A)
    yE = HG - BALL_E * HH
    ea = A * smooth(ramp(t, 2.6, 0.8))
    if ea > 0:
        s = Stroke(150, yE - 3, 1770, yE + 4)
        s.dashed((160, yE), (1760, yE), 6, 8)
        s.light(F, IVORY, 0.28 * ea)
        italic("E", 22).draw(F, 168, yE - 10, IVORY, 0.7 * ea)
        italic("V", 22).draw(F, HC + 14, HG - HH - 12, GOLD, 0.8 * ea)
    xs, tt, _ = _once("ball", _ball_path)
    q = (t - 3.0) / 8.0
    u = min(max(q * 2 if q < 0.5 else (1 - q) * 2, 0.0), 1.0)
    bx = float(np.interp(u, tt, xs))
    by = float(hill_y(bx))
    dydx = float(2 * HH * (bx - HC) / HW ** 2 * math.exp(-((bx - HC) / HW) ** 2))
    nrm = math.hypot(1, dydx)
    orb(F, bx + dydx / nrm * 11, by - 11 / nrm, 8, GOLD, A * smooth(ramp(t, 2.6, 0.5)), core=WARM_WHITE)
    fb = A * smooth(ramp(t, 11.6, 1.0))
    if fb > 0:
        xr = HX[hill_y(HX) < yE]
        poly = np.vstack([np.column_stack([xr, hill_y(xr)]), [[xr[-1], yE], [xr[0], yE]]])
        s = Stroke(xr[0] - 4, HG - HH - 4, xr[-1] + 5, yE + 4)
        s.fill(poly)
        s.light(F, CRIMSON, 0.55 * fb)
        s.glow(F, CRIMSON, 0.5 * fb, 8)
        serif("禁区", 16, 500, 0.3).draw(F, HC + 120, HG - HH + 40, mix(CRIMSON, IVORY, 0.5), fb)
        caps("FORBIDDEN", 10, 0.4).draw(F, HC + 120, HG - HH + 60, DIM, fb)
    subtitles(F, t, [(1.0, 6.0, "在经典世界里，想翻过一座山，就得有足够的能量。",
                      "In the classical world, crossing a hill takes enough energy."),
                     (6.3, 11.3, "能量不够？那就只能停在半山腰，再滚回来。",
                      "Not enough? It stops halfway, and rolls back down."),
                     (11.6, 16.2, "比它能量更高的地方，它永远到不了。",
                      "Wherever the hill rises above its energy, it can never go."),
                     (16.8, 19.7, "这，就是经典世界的铁律。", "That is the iron law of the classical world.")])
    statement(F, "不可能", 960 + 0.25 * 96, 520, t, 16.8, 19.9, size=96, wght=400, tracking=0.5, cps=3.2,
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
    n = 46
    return rng.normal(0, 1, (n, 2)), np.sort(rng.uniform(10.0, 16.0, n))


def wave(F, t, g, fi):
    chapter_mark(F, window(t, 0.3, 22, 1.0, 0.8), "02", "波", "THE WAVE")
    A = window(t, 0, 22, 0.6, 1.0)
    cx = 960 + 140 * ease_in_out(ramp(t, 16.4, 5.0))
    o = smooth(ramp(t, 0.4, 1.4)) * (1 - smooth(ramp(t, 3.4, 1.4)))
    orb(F, 960, 470, 9, CYAN, 1.2 * o * A)
    sig = lerp(10, 78, ease_in_out(ramp(t, 3.0, 3.0)))
    wa = smooth(ramp(t, 3.0, 1.0)) * A
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
    wl = A * smooth(ramp(t, 16.4, 1.2))
    if wl > 0:
        glow_poly(F, [(1480, 250), (1480, 690)], CYAN, 0.6 * wl, th=2, glow=0.9, sigma=7)
    subtitles(F, t, [(1.0, 5.0, "但电子，不是一颗小球。", "But an electron is not a ball."),
                     (5.4, 10.2, "量子力学说，它是一团关于「可能性」的波。",
                      "Quantum mechanics says it is a wave of possibilities."),
                     (10.6, 15.8, "波越亮的地方，越可能在那里找到它。",
                      "The brighter the wave, the likelier you are to find it there."),
                     (16.4, 21.4, "而波，从来不会在墙面上戛然而止。", "And a wave never simply stops at a wall.")])


# -------------------------------------------------------- 03 through ----

SIM_Y0 = 150
CELL = W / P.NX


def sim_frame(t):
    if t <= 20.5:
        return min(max((t - 0.5) / 22.0 * 360, 0.0), 360.0)
    return min(327.3 + 32.7 * (1 - math.exp(-(t - 20.5) / 2.2)), 360.0)


def _glass():
    yy = np.arange(SIM_Y0, SIM_Y0 + 720)
    prof = ((np.clip((yy - SIM_Y0) / 140, 0, 1) * np.clip((SIM_Y0 + 720 - yy) / 140, 0, 1)) ** 1.5).astype(np.float32)
    edge_glow = cv2.GaussianBlur(np.pad(prof[:, None], ((0, 0), (12, 12))), (0, 0), 5)
    return prof, edge_glow


def through(F, t, g, fi):
    chapter_mark(F, window(t, 0.3, 30, 1.0, 0.8), "03", "穿", "THROUGH")
    A = window(t, 0, 30, 0.8, 1.0)
    D = _once("sim", P.packet2d)
    L = np.zeros_like(F)
    x0w, x1w = P.BAR_X0 * CELL, (P.BAR_X0 + P.BAR_W) * CELL
    prof, edge_glow = _once("glass", _glass)
    wa = A * smooth(ramp(t, 0.2, 1.2))
    glass = np.repeat(prof[:, None], int(x1w - x0w) + 1, 1)
    blend(L, glass, int(x0w), SIM_Y0, np.array([0.03, 0.05, 0.09], np.float32), 0.9 * wa)
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
    I = (0.55 * p + 0.45 * r * r) / p0
    v = np.clip(I[32:224], 0, None) ** 0.62
    side = np.clip((np.arange(P.NX) - (P.BAR_X0 + P.BAR_W)) / 6.0, 0, 1)[None, :, None].astype(np.float32)
    col = palette(v, *COOL) * (1 - side) + palette(v, *WARM) * side
    L[SIM_Y0:SIM_Y0 + 720] += cv2.resize(col, (W, 720), interpolation=cv2.INTER_CUBIC) * 0.85 * A * smooth(ramp(t, 0.5, 1.0))
    s = 1.0 + 0.07 * ease_in_out(t / 30)
    M = np.float32([[s, 0, 960 * (1 - s)], [0, s, 510 * (1 - s)]])
    F += cv2.warpAffine(L, M, (W, H), flags=cv2.INTER_LINEAR)
    ra = A * smooth(ramp(t, 22.4, 1.0))
    if ra > 0:
        serif("弹回", 15, 500, 0.3).draw(F, 520, 800, DIM, ra, align="center")
        caps("REFLECTED", 10, 0.4).draw(F, 520, 818, DIM, ra, align="center")
        italic(f"{round(float(D['R']) * 100)}%", 54).draw(F, 520, 876, CYAN, ra, align="center")
        serif("穿过", 15, 500, 0.3).draw(F, 1400, 800, GOLD, ra, align="center")
        caps("TRANSMITTED", 10, 0.4).draw(F, 1400, 818, DIM, ra, align="center")
        italic(f"{round(float(D['T']) * 100)}%", 54).draw(F, 1400, 876, GOLD, ra, align="center")
    subtitles(F, t, [(1.0, 5.6, "看，一团波正向墙飞去。", "Watch: a wave flies toward the wall."),
                     (6.0, 11.0, "撞上墙的一刻，它在墙前激起层层涟漪——",
                      "As it strikes, ripples rise in front of the wall —"),
                     (11.4, 16.4, "而一部分，悄悄渗进了墙里。", "while part of it quietly seeps into the wall."),
                     (16.8, 21.8, "墙够薄，它便在另一侧重新出现。",
                      "If the wall is thin enough, it reappears on the other side."),
                     (22.4, 28.8, "电子穿墙而过——这，就是量子隧穿。",
                      "The electron passes through. This is quantum tunneling.")])


# ----------------------------------------------------------- 04 thin ----

NL_X0, NL_X1, NL_Y, NL_MAX = 300.0, 1620.0, 660.0, 2.2
STEP_T = [3.0, 7.5, 11.0, 14.5]
STEP_A = [0.5, 1.0, 1.5, 2.0]
WORDS = ["约四十二分之一", "约七千分之一", "约一百二十万分之一", "约两亿分之一"]


def nl_x(a):
    return NL_X0 + a / NL_MAX * (NL_X1 - NL_X0)


def thickness(t):
    a = 0.0
    for ts, av in zip(STEP_T, STEP_A):
        a = lerp(a, av, ease_in_out(ramp(t, ts, 0.8)))
    return a


def decimal_parts(T_):
    e = math.floor(math.log10(T_))
    sig = f"{T_ / 10 ** e:.1f}".replace(".", "").rstrip("0") or "0"
    return "0." + "0" * (-e - 1), sig


def thin(F, t, g, fi):
    chapter_mark(F, window(t, 0.3, 26, 1.0, 0.8), "04", "薄", "THIN")
    A = window(t, 0, 26, 0.6, 1.0)
    serif("电子 1 eV · 势垒 2 eV", 15, 400, 0.12).draw(F, 1824, 92, DIM, 0.9 * A * smooth(ramp(t, 0.8, 0.8)),
                                                       align="right")
    la = A * (1 - 0.75 * smooth(ramp(t, 17.6, 0.8)))
    d = ease_in_out(ramp(t, 0.4, 1.4))
    glow_poly(F, [(NL_X0, NL_Y), (NL_X0 + (NL_X1 - NL_X0) * d, NL_Y)], GOLD, 0.8 * la, th=2, glow=0.8, sigma=5)
    tk = Stroke(NL_X0 - 4, NL_Y - 10, NL_X1 + 5, NL_Y + 14)
    for k in range(23):
        x = nl_x(k / 10)
        tk.line((x, NL_Y + 4), (x, NL_Y + (12 if k % 5 == 0 else 7)))
    tk.light(F, IVORY, 0.35 * la * d)
    for av in (0.5, 1.0, 1.5, 2.0):
        T(f"{av:.1f} nm", "stix", 16, None, per_char=False).draw(F, nl_x(av), NL_Y + 36, DIM, la * d,
                                                               align="center")
    serif("墙厚", 14, 500, 0.3).draw(F, NL_X0, NL_Y - 22, DIM, la * d)
    caps("THICKNESS", 10, 0.4).draw(F, NL_X0 + 40, NL_Y - 23, DIM, la * d)
    a = thickness(t)
    if t > STEP_T[0]:
        Tr = P.transmission(max(a, 0.05))
        b = (math.log10(Tr) + 10) / 10
        xm = nl_x(a)
        blend(F, np.full((110, 1), 255, np.uint8), xm, NL_Y - 116, GOLD, 0.4 * la)
        orb(F, xm, NL_Y, 6 + 10 * b, GOLD, la * (0.25 + 0.95 * b))
        k = max(i for i, ts in enumerate(STEP_T) if t >= ts)
        ra = la * (1 - smooth(ramp(t, 17.2, 0.6)))
        zeros, sig = decimal_parts(P.transmission(STEP_A[k]))
        if t - STEP_T[k] < 0.55:
            rng = np.random.default_rng(int(t * 30))
            zeros = "0." + "".join(str(v) for v in rng.integers(0, 10, len(zeros) - 2))
        tz = T(zeros, "stix", 96, None, per_char=False)
        ts_ = T(sig, "stix", 96, None, per_char=False)
        x0 = 960 - (tz.width + ts_.width) / 2
        tz.draw(F, x0, 440, IVORY, 0.55 * ra)
        ts_.draw(F, x0 + tz.width, 440, GOLD, ra)
        serif(WORDS[k], 30, 500, 0.3).draw(F, 960 + 0.15 * 30, 516, GOLD,
                                           ra * smooth(ramp(t, STEP_T[k] + 0.5, 0.5)), align="center")
    fa = A * smooth(ramp(t, 18.0, 1.0))
    if fa > 0:
        base = R([("T", "stix_it", 88, None), ("  ≈  ", "math", 80, None), ("e", "stix_it", 88, None)])
        sup = R([("−2", "stix", 46, None), ("κa", "stix_it", 46, None)])
        x0 = 960 - (base.width + sup.width) / 2
        base.draw(F, x0, 450, GOLD, fa)
        sup.draw(F, x0 + base.width + 4, 404, GOLD, fa)
        zl = serif("隧穿概率", 15, 500, 0.4)
        cl = caps("TUNNELING PROBABILITY", 10, 0.4)
        lx = 960 - (zl.width + 14 + cl.width) / 2
        zl.draw(F, lx, 514, DIM, fa)
        cl.draw(F, lx + zl.width + 14, 513, DIM, fa)
    subtitles(F, t, [(1.0, 5.6, "穿过去的概率，取决于墙有多厚。",
                      "Whether it gets through depends on how thick the wall is."),
                     (6.0, 13.8, "墙每厚一点，概率就塌下一大截。", "With every bit of thickness, the odds collapse."),
                     (14.6, 17.8, "两纳米：大约两亿次，才能穿过去一次。",
                      "Two nanometres: about once in two hundred million tries."),
                     (18.2, 21.8, "厚度翻倍，概率缩小了近三万倍。",
                      "Double the thickness, and the odds shrink nearly thirty-thousand-fold."),
                     (22.2, 25.6, "所以，你我永远穿不过一堵墙。", "That is why you and I never walk through walls.")])


# ---------------------------------------------------------- 05 light ----

def _noise_tex():
    rng = np.random.default_rng(31)
    out = []
    for s in (5, 11):
        n = cv2.GaussianBlur(rng.normal(0, 1, (720, 1200)).astype(np.float32), (0, 0), s / 5)
        out.append(n / np.abs(n).max())
    return out


def sun(F, t, a):
    if a <= 0.003:
        return
    n1, n2 = _once("noise", _noise_tex)
    R0 = 270.0
    cx, cy = 960.0, 450.0 + 24 * (1 - ease_out(t / 9))
    h, w = 540, 960
    ys, xs = _once("half_grid", lambda: tuple(np.mgrid[0:540, 0:960].astype(np.float32) * 2))
    r = np.sqrt((xs - cx) ** 2 + (ys - cy) ** 2) / R0
    mu = np.sqrt(np.clip(1 - r ** 2, 0, 1))
    g1 = np.roll(n1, int(t * 6), 1)[:h, :w]
    g2 = np.roll(n2, -int(t * 3), 0)[:h, :w]
    edge = np.clip((1 - r) * R0 / 3.0, 0, 1).astype(np.float32)
    mu = np.sqrt(np.clip(1 - np.minimum(r, 1) ** 2, 0, 1))
    I = (0.42 + 0.58 * mu ** 0.6) * (0.9 + 0.12 * (0.55 * g1 + 0.45 * g2))
    col = palette(I, [0, 0.3, 0.62, 0.85, 1.0], [(0, 0, 0), (0.45, 0.1, 0.02), (0.92, 0.42, 0.1),
                                                   (1.0, 0.72, 0.36), (1.0, 0.93, 0.78)])
    cor = np.clip(r - 1, 0, None)
    corona = (np.exp(-cor * R0 / 8) * 0.12 + np.exp(-cor * R0 / 60) * 0.12 + np.exp(-cor * R0 / 220) * 0.07)
    corona = (corona * (1 - edge))[..., None] * np.array([1.0, 0.6, 0.25], np.float32)
    disc = cv2.resize(edge, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
    F *= 1 - disc * a
    F += cv2.resize(col, (W, H), interpolation=cv2.INTER_LINEAR) * disc * 0.72 * a
    F += cv2.resize(corona, (W, H), interpolation=cv2.INTER_LINEAR) * 0.72 * a


def _stm_field():
    """Half-res luminous surface of a hexagonal atom lattice seen at a slant (STM-like)."""
    hh, ww = 540, 960
    img = np.zeros((hh, ww), np.float32)
    hl = np.zeros((hh, ww), np.float32)
    r = 0
    y = 82.0
    while y < 430:
        persp = 0.62 + 0.05 * r
        dx = 40 * persp
        sig = 9.0 * persp
        depth = 0.45 + 0.55 * min(1.0, r / 7)
        for c in range(-4, 40):
            x = 480 + (c - 18) * dx + (r % 2) * dx / 2
            if not (-40 < x < ww + 40):
                continue
            x0, x1 = int(max(x - 4 * sig, 0)), int(min(x + 4 * sig, ww))
            y0, y1 = int(max(y - 3 * sig, 0)), int(min(y + 3 * sig, hh))
            if y0 >= y1 or x0 >= x1:
                continue
            yy, xx = np.mgrid[y0:y1, x0:x1].astype(np.float32)
            img[y0:y1, x0:x1] += depth * np.exp(-((xx - x) ** 2 + ((yy - y) / 0.72) ** 2) / (2 * sig ** 2))
            hl[y0:y1, x0:x1] += depth * np.exp(-((xx - x + sig * 0.35) ** 2 + (yy - y + sig * 0.3) ** 2)
                                               / (2 * (sig * 0.32) ** 2))
        r += 1
        y += 26 + 3.0 * r
    fade = (np.clip(1 - np.abs(np.arange(ww) - 480) / 500, 0, 1) ** 0.7)[None, :]
    I = np.clip(img, 0, 1.2) ** 1.4 * fade
    S = np.clip(hl, 0, 1) * fade
    col = np.stack([np.interp(I, [0, 0.2, 0.55, 1.0], [0, 0.28, 0.82, 1.0]),
                    np.interp(I, [0, 0.2, 0.55, 1.0], [0, 0.1, 0.42, 0.74]),
                    np.interp(I, [0, 0.2, 0.55, 1.0], [0, 0.02, 0.1, 0.32])], -1).astype(np.float32)
    col += S[..., None] * np.array([0.55, 0.5, 0.42], np.float32)
    return cv2.resize(col, (W, H), interpolation=cv2.INTER_CUBIC)


def atoms(F, t, a):
    if a <= 0.003:
        return
    field = _once("stm", _stm_field)
    scan = lerp(170, 800, ease_in_out(ramp(t, 0.4, 6.2)))
    ys = np.arange(H, dtype=np.float32)
    reveal = np.clip((scan - ys) / 60, 0, 1)[:, None, None]
    F += field * reveal * 0.78 * a
    if t < 7.0:
        sa = a * (1 - smooth(ramp(t, 6.4, 0.6)))
        glow_poly(F, [(140, scan), (1780, scan)], CYAN, 0.45 * sa, th=1, glow=0.6, sigma=4)
        orb(F, 960 + 800 * math.sin(t * 6.0), scan, 5, CYAN, sa)


def _traces():
    rng = np.random.default_rng(77)
    step = 36
    tr = []
    for _ in range(56):
        x = int(rng.integers(2, W // step - 2)) * step
        y = int(rng.integers(3, (H - 160) // step)) * step
        d = [(1, 0), (0, 1), (-1, 0), (0, -1)][int(rng.integers(0, 4))]
        pts = [(x, y)]
        for _ in range(int(rng.integers(6, 22))):
            if rng.random() < 0.3:
                d = (d[1], d[0]) if rng.random() < 0.5 else (-d[1], -d[0])
            x, y = x + d[0] * step, y + d[1] * step
            if not (step <= x < W - step and step <= y < H - 150):
                break
            pts.append((x, y))
        if len(pts) > 3:
            tr.append(np.array(pts, float))
    layer = np.zeros((H, W), np.uint8)
    for p in tr:
        cv2.polylines(layer, [np.round(p * 16).astype(np.int32)], False, 255, 1, cv2.LINE_AA, 4)
        for q in (p[0], p[-1]):
            cv2.circle(layer, (int(q[0] * 16), int(q[1] * 16)), 3 * 16, 255, 1, cv2.LINE_AA, 4)
    cv2.rectangle(layer, (840 * 16, 330 * 16), (1080 * 16, 570 * 16), 255, 2, cv2.LINE_AA, 4)
    lens = [np.concatenate([[0], np.cumsum(np.hypot(*np.diff(p, axis=0).T))]) for p in tr]
    return tr, lens, layer.astype(np.float32) / 255.0


def chip(F, t, a):
    if a <= 0.003:
        return
    tr, lens, layer = _once("traces", _traces)
    F += layer[..., None] * mix(CYAN, IVORY, 0.3)[None, None] * 0.1 * a
    rng = np.random.default_rng(5)
    for _ in range(46):
        j = int(rng.integers(0, len(tr)))
        p, L = tr[j], lens[j]
        sp = rng.uniform(220, 380)
        u = ((t * sp + rng.uniform(0, 2000)) % (L[-1] + 300)) - 150
        for k in range(6):
            uu = u - k * 9
            if 0 <= uu <= L[-1]:
                add_sprite(F, np.interp(uu, L, p[:, 0]), np.interp(uu, L, p[:, 1]), 2.4 if k == 0 else 1.6, GOLD,
                           a * (0.9 if k == 0 else 0.35 * (1 - k / 6)))
    add_sprite(F, 960, 450, 90, GOLD, 0.12 * a * (0.85 + 0.15 * math.sin(t * 2.0)))


def _spark_set():
    rng = np.random.default_rng(99)
    n = 2600
    return (rng.uniform(0, W, n), rng.uniform(60, H - 170, n),
            rng.uniform(START["light"] + 23.0, START["epilogue"] + 6.0, n), rng.random(n) < 0.6)


def sparks(F, g, a):
    if a <= 0.003:
        return
    xs, ys, t0, warm = _once("sparks", _spark_set)
    age = g - t0
    on = (age >= 0) & (age < 0.6)
    if not on.any():
        return
    w = (1 - age[on] / 0.6) ** 2
    xi = (xs[on] / 2).astype(int)
    yi = (ys[on] / 2).astype(int)
    for mask, col in ((warm[on], GOLD), (~warm[on], CYAN)):
        dens = np.bincount(yi[mask] * 960 + xi[mask], weights=w[mask], minlength=960 * 540)
        dens = dens.reshape(540, 960).astype(np.float32)
        dens = cv2.GaussianBlur(dens, (0, 0), 0.9) * 7.0 + cv2.GaussianBlur(dens, (0, 0), 5) * 9.0
        F += cv2.resize(dens, (W, H), interpolation=cv2.INTER_LINEAR)[..., None] * col[None, None] * a


def light(F, t, g, fi):
    chapter_mark(F, window(t, 0.3, 26, 1.0, 0.8), "05", "光", "THE LIGHT")
    sun(F, t, window(t, 0, 9.6, 1.4, 1.2))
    atoms(F, t - 8.8, window(t, 8.8, 17.6, 1.0, 1.0))
    chip(F, t - 16.8, window(t, 16.8, 24.8, 1.0, 1.2))
    sparks(F, g, smooth(ramp(t, 23.4, 1.6)))
    subtitles(F, t, [(0.8, 4.8, "可正是这「几乎不可能」，", "And yet, this 'almost impossible' —"),
                     (5.0, 9.0, "让太阳里的质子越过斥力，聚变发光。",
                      "lets protons in the Sun overcome repulsion, fuse, and shine."),
                     (9.6, 16.6, "扫描隧道显微镜，靠隧穿电流「看见」了单个原子。",
                      "The scanning tunneling microscope sees single atoms by tunneling current."),
                     (17.6, 24.0, "你手机里的每一张照片，也靠电子穿过绝缘层来保存。",
                      "Every photo on your phone is kept by electrons tunneling through an insulator.")])


# ------------------------------------------------------------ epilogue ---

def _glyph_points(text, size, tracking, cx, y, n, seed):
    tx = serif(text, size, 400, tracking)
    x0 = cx - tx.width / 2
    pts = []
    for m, dx, dy in tx.items:
        yy, xx = np.nonzero(m > 120)
        pts.append(np.column_stack([xx + x0 + dx, yy + y + dy]))
    pts = np.concatenate(pts).astype(np.float32)
    return pts[np.random.default_rng(seed).integers(0, len(pts), n)]


def epilogue(F, t, g, fi):
    chapter_mark(F, window(t, 0.3, 13.0, 1.0, 0.8), "06", "终", "EPILOGUE")
    sparks(F, g, 1 - smooth(ramp(t, 4.0, 2.0)))
    subtitle(F, t, 0.5, 4.6, "最不可能的事，每一秒都在宇宙中发生着。",
             "The most improbable things are happening every second, everywhere.")
    cx1 = 960 + 0.25 * 100
    if t < 8.0:
        statement(F, "不可能", cx1, 560, t, 4.8, None, size=100, tracking=0.5, cps=3.4, fade_in=0.7,
                  a=1 - smooth(ramp(t, 7.5, 0.5)))
    src = _once("src", lambda: _glyph_points("不可能", 100, 0.5, cx1, 560, 5200, 1))
    dst = _once("dst", lambda: _glyph_points("概率很小", 100, 0.5, cx1, 560, 5200, 2))
    jit = _once("jit", lambda: np.random.default_rng(4).normal(0, 1, (5200, 2)).astype(np.float32))
    if 7.4 < t < 10.4:
        u = ease_in_out(ramp(t, 8.0, 1.6))
        pts = src + (dst - src) * u + jit * 70 * math.sin(math.pi * u)
        pa = smooth(ramp(t, 7.4, 0.5)) * (1 - smooth(ramp(t, 9.7, 0.6)))
        col = mix(IVORY, GOLD, smooth(u))
        xi, yi = pts[:, 0].astype(int), pts[:, 1].astype(int)
        y0, y1 = 300, 820
        ok = (xi > 0) & (xi < W) & (yi > y0) & (yi < y1)
        dens = np.bincount((yi[ok] - y0) * W + xi[ok], minlength=(y1 - y0) * W).reshape(y1 - y0, W)
        dens = dens.astype(np.float32)
        lm = cv2.GaussianBlur(dens, (0, 0), 0.8) * 1.6 + cv2.GaussianBlur(dens, (0, 0), 6) * 1.8
        F[y0:y1] += lm[..., None] * col[None, None] * pa
    if t > 9.6:
        statement(F, "概率很小", cx1, 560, t, 9.6, 12.9, size=100, tracking=0.5, cps=40, fade_in=0.7, color=GOLD)
    subtitle(F, t, 7.8, 12.6, "在量子世界里，没有绝对的不可能——只有概率很小。",
             "In the quantum world, nothing is truly impossible — only improbable.")
    if t > 12.9:
        statement(F, "量子隧穿", 960 + 0.25 * 44, 530, t, 13.2, None, size=44, wght=500, tracking=0.5, cps=30,
                  fade_in=0.9)
        fa = smooth(ramp(t, 13.8, 0.8))
        hairline(F, 960 - 90 * fa, 562, 960 + 90 * fa, GOLD, 0.6 * fa)
        caps("QUANTUM TUNNELING", 12, 0.5).draw(F, 960, 596, DIM, fa, align="center")


SCENE_FUNCS = {"prologue": prologue, "wall": wall, "wave": wave, "through": through, "thin": thin,
               "light": light, "epilogue": epilogue}
