"""量子隧穿 — 南墙之外.  draw(F, t_local, g_global, frame).

One spine: what we call impossible is only improbable - and an improbable
thing, tried enough times, happens.  Each chapter's picture *is* its idea:
a beam that dims layer by layer, zeros that will not end, a counter that
knocks 10^38 times, sparks that add up to sunlight."""

import math

import cv2
import numpy as np

import physics as P
import timeline as TL
from style import E
from style import (CRIMSON, CY, CYAN, DIM, GOLD, IVORY, WHITE, H, W, R, Stroke, T, add_light, add_sprite, blank,
                   blend, caps, chapter_mark, comet, draw_sci, ease_in, ease_in_out, ease_out, gauge, glow_poly,
                   hairline, italic, lerp, mix, orb, ramp, serif, smooth, statement, subtitle, subtitles, window)
from timeline import BALL_HIT, FILL, SHOTS, START, THROUGH_SHOT

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
STEEL = mix(CYAN, IVORY, 0.35)
WALL_X = 1180


def underline(F, x, y, size, a, color=GOLD):
    blend(F, np.full((2, int(size * 2.3)), 255, np.uint8), x + size * 0.15, y + size * 0.18, color, 0.9 * a)


def label(F, zh, en, x, y, a, color=DIM, align="left", size=15):
    """Small bilingual tag: serif zh + spaced caps en on one line."""
    if a <= 0.003:
        return
    zl = serif(zh, size, 500, 0.2)
    el = caps(en, 10, 0.4)
    w = zl.width + 12 + el.width
    x0 = x - w / 2 if align == "center" else x - w if align == "right" else x
    zl.draw(F, x0, y, color, a)
    el.draw(F, x0 + zl.width + 12, y - 1, DIM, 0.9 * a)


def streak(F, x, y, ang, length, color, a, segs=10):
    """Light streak from (x, y) along `ang`, brightest at the head."""
    if a <= 0.003 or length < 2:
        return
    dx, dy = math.cos(ang), math.sin(ang)
    for i in range(segs):
        u0, u1 = i / segs, (i + 1) / segs
        glow_poly(F, [(x + dx * length * u0, y + dy * length * u0), (x + dx * length * u1, y + dy * length * u1)],
                  color, a * u1 ** 2, th=1, glow=0.5, sigma=3)
    orb(F, x + dx * length, y + dy * length, 4, color, a)


# ------------------------------------------------------------ prologue ---

BY = 192          # ball / electron line


def _ball(F, t, a):
    glow_poly(F, [(WALL_X, BY - 64), (WALL_X, BY + 64)], CYAN, 0.7 * a * smooth(ramp(t, 4.2, 0.8)), th=2,
              glow=0.8, sigma=6)
    if t < 4.6:
        return
    x = lerp(720, WALL_X - 12, ramp(t, 4.6, BALL_HIT - 4.6)) if t < BALL_HIT else \
        WALL_X - 12 - 320 * ease_out(ramp(t, BALL_HIT, 2.4))
    orb(F, x, BY, 8, GOLD, a * smooth(ramp(t, 4.5, 0.3)), core=WARM_WHITE)
    if BALL_HIT <= t < BALL_HIT + 0.7:
        orb(F, WALL_X, BY, 18, GOLD, 0.7 * (1 - (t - BALL_HIT) / 0.7) * a)


def _electron(F, t, t0, through, a):
    tau = t - t0
    if tau < 0 or tau > 3.0:
        return
    if tau < 1.0:
        orb(F, lerp(760, WALL_X - 10, tau), BY, 6, CYAN, a)
        return
    q = tau - 1.0
    fade = 1 - smooth((q - 1.4) / 0.6)
    orb(F, WALL_X, BY, 16, GOLD if through else CYAN, 0.8 * max(0.0, 1 - q / 0.5) * a)
    if through:
        orb(F, WALL_X + 12 + 400 * (q / 2.0), BY, 6, GOLD, a * fade)
    else:
        orb(F, WALL_X - 12 - 300 * ease_out(q / 2.0), BY, 6, CYAN, a * fade)


def _burst(F, q, a, cy):
    P0 = _once("burst", lambda: np.random.default_rng(3).normal(0, 1, (160, 2)))
    if q > 3.0:
        return
    sp = 1 - math.exp(-q * 2.2)
    for vx, vy in P0:
        add_sprite(F, 960 + vx * 420 * sp, cy + vy * 110 * sp, 1.2, GOLD, 0.6 * a * max(0.0, 1 - q / 3.0))


def prologue(F, t, g, fi):
    chapter_mark(F, window(t, 0.5, 19.6, 1.0, 0.8), "00", "序", "PROLOGUE")
    # beat 1: everybody completes the idiom - certainty
    a1 = window(t, 0.8, 9.6, 0.6, 0.8)
    if a1 > 0:
        bw = 64 * 2.45
        xe = statement(F, "不撞南墙不", 960 - bw / 2, 412, t, 1.2, 9.6, size=64, cps=6.5)
        ba = a1 * smooth(ramp(t, 2.0, 0.3)) * (1 - smooth(ramp(t, 8.9, 0.6)))
        if t < FILL:
            blank(F, xe + 6, 412, 64, ba, t)
        else:
            underline(F, xe + 6, 412, 64, ba)
            statement(F, "回头", xe + 6 + 0.42 * 64, 412, t, FILL, 9.6, size=64, cps=5, color=GOLD, align="left",
                      fade_in=0.35)
        _ball(F, t, a1)
        gauge(F, 960, 528, 520, 1.0, CYAN, "回头的概率", "TURNING BACK", "= 100%", a=a1 * smooth(ramp(t, 4.4, 0.6)))
        subtitle(F, t, 4.6, 9.4, "你一定猜到了。在我们的世界里，撞了墙，就只能回头。",
                 "You guessed it. In our world, hit a wall and you turn back.")
    # beat 2: an electron hits the wall
    a2 = window(t, 10.0, 19.9, 0.6, 0.8)
    if a2 > 0:
        glow_poly(F, [(WALL_X, BY - 64), (WALL_X, BY + 64)], CYAN, 0.7 * a2, th=2, glow=0.8, sigma=6)
        statement(F, "可如果撞墙的，是一个电子——", 960, 412, t, 10.0, 16.9, size=56, cps=9)
        statement(F, "撞了南墙，它却不一定回头。", 960, 412, t, 17.2, 19.9, size=56, cps=12, gold={7, 8, 9})
        for i, ts in enumerate(SHOTS):
            _electron(F, t, ts, i == THROUGH_SHOT, a2)
        hit = SHOTS[THROUGH_SHOT] + 1.0
        v = 1.0 - 0.07 * ease_out(ramp(t, hit, 1.0))
        gauge(F, 960, 528, 520, v, CYAN if t < hit else GOLD, "回头的概率", "TURNING BACK",
              "= 100%" if t < hit + 0.3 else "< 100%", a=a2 * smooth(ramp(t, 10.6, 0.6)))
        for i, ts in enumerate(SHOTS):
            q = t - (ts + 1.0)
            if q >= 0:
                col = GOLD if i == THROUGH_SHOT else CYAN
                orb(F, 960 + (i - 2) * 30, 586, 4, col, a2 * (0.85 + 0.6 * max(0.0, 1 - q / 0.5)))
    # burst and title
    if 18.7 < t < 26:
        cy = 452
        o = smooth(ramp(t, 18.9, 0.9)) * (1 - smooth(ramp(t, 20.0, 0.25)))
        orb(F, 960, cy, 9 + 8 * o, GOLD, 1.3 * o)
        if t >= 20.0:
            q = t - 20.0
            fade = 1 - smooth(ramp(t, 24.6, 1.2))
            hx = 960 + 980 * ease_out(min(q / 2.6, 1.0))
            ca = 0.9 * fade * (1 - smooth(ramp(q, 2.2, 0.5)))
            comet(F, hx, cy, 1, 360, GOLD, ca)
            comet(F, 1920 - hx, cy, -1, 360, GOLD, ca)
            span = 900 * ease_out(min(q / 1.2, 1.0))
            glow_poly(F, [(960 - span, cy), (960 + span, cy)], GOLD, 0.25 * fade, th=1, glow=0.4, sigma=3)
            _burst(F, q, fade, cy)
            statement(F, "量子隧穿", 960 + 0.175 * 112, 412, t, 20.4, 26.0, size=112, wght=500, tracking=0.35,
                      cps=5.5, fade_in=0.9)
            statement(F, "南墙之外", 960 + 0.25 * 28, 508, t, 21.6, 26.0, size=28, wght=500, tracking=0.5,
                      cps=8, color=GOLD, fade_in=0.6)
            caps("QUANTUM TUNNELING   ·   BEYOND THE WALL", 13, 0.42).draw(
                F, 960, 548, DIM, smooth(ramp(t, 22.4, 0.8)) * fade, align="center")


# ----------------------------------------------------------- 01 wall ----

HX = np.linspace(160, 1760, 900)
HC, HW, HG, HH = 1000.0, 150.0, 592.0, 300.0
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
    chapter_mark(F, window(t, 0.3, 15, 1.0, 0.8), "01", "墙", "THE WALL")
    A = window(t, 0, 15, 0.6, 0.9) * (1 - 0.85 * smooth(ramp(t, 9.8, 0.8)))
    n = max(2, int(len(HX) * ease_in_out(ramp(t, 0.3, 2.0))))
    pts = np.column_stack([HX[:n], hill_y(HX[:n])])
    glow_poly(F, pts, GOLD, 0.85 * A, th=2, glow=0.8, sigma=5, bounds=(140, HG - HH - 40, 1780, HG + 20))
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
    subtitles(F, t, [(1.0, 5.2, "在经典世界里，墙的含义很简单：",
                      "In the classical world, a wall means one thing:"),
                     (5.6, 9.8, "能量不够，就过不去。不是很难——而是不可能。",
                      "Without enough energy, there is no way through. Not hard — impossible.")])
    statement(F, "不可能", 960 + 0.25 * 96, 412, t, 10.2, 14.9, size=96, wght=400, tracking=0.5, cps=3.2,
              fade_in=0.7)


# ------------------------------------------------------------ 02 seep ----

GX, GY = np.meshgrid((np.arange(W // 4) + 0.5) * 4, (np.arange(H // 4) + 0.5) * 4)
CELL = W / P.NX
SIM_R0, SIM_R1 = 32, 224
SIM_Y0 = 40
CLOUD_Y = SIM_Y0 + (P.PY0 - SIM_R0) * CELL
SIM_X0 = P.PX0 * CELL
SIM_LOCAL = TL.SIM0 - START["seep"]


def draw_field(F, I, pal, a=1.0, gamma=0.55):
    if a <= 0.003:
        return
    col = palette(np.clip(I, 0, None) ** gamma, *pal)
    F += cv2.resize(col, (W, H), interpolation=cv2.INTER_CUBIC) * a


def packet_field(cx, cy, sig, phase, k=P.K0 / CELL):
    env = np.exp(-((GX - cx) ** 2 + (GY - cy) ** 2) / (4 * sig ** 2))
    re = env * np.cos(k * (GX - cx) - phase)
    return 0.5 * env ** 2 + 0.5 * re ** 2


def sim_frame(ts):
    if ts <= 11.5:
        return min(max(ts / 13.0 * 360, 0.0), 360.0)
    return min(318.5 + 41.5 * (1 - math.exp(-(ts - 11.5) / 1.8)), 360.0)


def _glass():
    hh = int((SIM_R1 - SIM_R0) * CELL)
    yy = np.arange(hh)
    prof = ((np.clip(yy / 140, 0, 1) * np.clip((hh - yy) / 140, 0, 1)) ** 1.5).astype(np.float32)
    return prof, cv2.GaussianBlur(np.pad(prof[:, None], ((0, 0), (12, 12))), (0, 0), 5)


def _glass_wall(Lr, a):
    if a <= 0.003:
        return
    x0w, x1w = P.BAR_X0 * CELL, (P.BAR_X0 + P.BAR_W) * CELL
    prof, edge_glow = _once("glass", _glass)
    blend(Lr, np.repeat(prof[:, None], int(x1w - x0w) + 1, 1), int(x0w), SIM_Y0,
          np.array([0.03, 0.05, 0.09], np.float32), 0.9 * a)
    for xe in (x0w, x1w):
        add_light(Lr, prof[:, None], int(round(xe)), SIM_Y0, mix(CYAN, WHITE, 0.4), 0.55 * a)
        add_light(Lr, edge_glow, int(round(xe)) - 12, SIM_Y0, CYAN, 0.9 * a)


def _sim_field(Lr, ts, a):
    if a <= 0.003:
        return
    D = _once("sim", P.packet2d)
    f = sim_frame(ts)
    i0 = int(f)
    i1 = min(i0 + 1, 360)
    fr = f - i0
    p = D["prob"][i0].astype(np.float32) * (1 - fr) + D["prob"][i1].astype(np.float32) * fr
    r = D["re"][i0].astype(np.float32) * (1 - fr) + D["re"][i1].astype(np.float32) * fr
    p0 = _once("p0", lambda: float(D["prob"][0].astype(np.float32).max()))
    v = np.clip((0.55 * p + 0.45 * r * r) / p0, 0, None)[SIM_R0:SIM_R1] ** 0.62
    side = np.clip((np.arange(P.NX) - (P.BAR_X0 + P.BAR_W)) / 6.0, 0, 1)[None, :, None].astype(np.float32)
    col = palette(v, *COOL) * (1 - side) + palette(v, *WARM) * side
    hh = int((SIM_R1 - SIM_R0) * CELL)
    Lr[SIM_Y0:SIM_Y0 + hh] += cv2.resize(col, (W, hh), interpolation=cv2.INTER_CUBIC) * 0.85 * a


def _readout(F, x, y, value, zh, en, color, a):
    if a <= 0.003:
        return
    T(value, "stix", 54, None, per_char=False).draw(F, x, y, color, a, align="center")
    label(F, zh, en, x, y + 36, 0.9 * a, align="center")


def seep(F, t, g, fi):
    chapter_mark(F, window(t, 0.3, 31.2, 1.0, 0.8), "02", "渗", "THE SEEPING")
    out = 1 - smooth(ramp(t, 30.4, 1.4))
    # the electron becomes a cloud of possibilities
    o = smooth(ramp(t, 0.3, 1.0)) * (1 - smooth(ramp(t, 2.0, 1.2)))
    orb(F, 720, CLOUD_Y, 9, CYAN, 1.2 * o)
    cx = lerp(720, SIM_X0, ease_in_out(ramp(t, 8.6, 2.0)))
    sig = lerp(10, P.SIG * CELL, ease_in_out(ramp(t, 1.6, 2.4)))
    wa = smooth(ramp(t, 1.6, 1.0)) * (1 - smooth(ramp(t, 10.2, 0.9)))
    if wa > 0:
        draw_field(F, packet_field(cx, CLOUD_Y, sig, 5.0 * t), COOL, 0.72 * wa, gamma=0.7)
    pts, tms = _once("detect", TL.detections)
    for (dx, dy), td in zip(pts, tms):
        q = g - td
        if 0 <= q < 0.9:
            x, y = cx + dx * sig, CLOUD_Y + dy * sig
            orb(F, x, y, 7, GOLD, 1.3 * (1 - q / 0.9) ** 1.5)
            r = 6 + 30 * ease_out(q / 0.9)
            s = Stroke(x - r - 3, y - r - 3, x + r + 4, y + r + 4)
            s.circle((x, y), r, th=1)
            s.light(F, GOLD, 0.8 * (1 - q / 0.9))
    # the honest 2D simulation, under a slow camera push
    ga = smooth(ramp(t, 8.0, 1.5)) * out
    sa = smooth(ramp(t, 10.0, 1.2)) * out
    if ga > 0:
        Lr = np.zeros_like(F)
        _glass_wall(Lr, ga)
        ts = max(0.0, t - SIM_LOCAL)
        _sim_field(Lr, ts, sa)
        s = 1.0 + 0.06 * ease_in_out(ts / 16)
        M = np.float32([[s, 0, 960 * (1 - s)], [0, s, CLOUD_Y * (1 - s)]])
        F += cv2.warpAffine(Lr, M, (W, H), flags=cv2.INTER_LINEAR)
    D = _once("sim", P.packet2d)
    split = TL.SPLIT - START["seep"]
    pr, pt = round(100 * float(D["R"])), round(100 * float(D["T"]))
    _readout(F, 470, 186, f"{pr}%", "回头", "TURNED BACK", CYAN, smooth(ramp(t, split, 0.8)) * out)
    _readout(F, 1450, 186, f"{pt}%", "穿过", "GOT THROUGH", GOLD, smooth(ramp(t, split + 0.5, 0.8)) * out)
    gauge(F, 960, 716, 520, float(D["R"]), CYAN, "回头的概率", "TURNING BACK", f"= {pr}%",
          a=smooth(ramp(t, split + 1.2, 0.8)) * out)
    subtitles(F, t, [(0.8, 4.4, "但电子不是一颗小球，它是一团「可能」。",
                      "But an electron is not a ball. It is a cloud of possibilities."),
                     (4.8, 8.8, "它会出现在哪里，只有概率可言。", "Where it will show up, only probability can say."),
                     (11.0, 15.4, "当这团「可能」撞上墙，它并不会就此消失——",
                      "When those possibilities meet a wall, they do not simply vanish —"),
                     (15.8, 20.4, "它渗进墙里，越往里越微弱，", "they seep into it, fainter the deeper they go,"),
                     (20.8, 25.0, "墙够薄，另一侧就会留下一缕微光。",
                      "and if the wall is thin enough, a glimmer is left on the other side."),
                     (25.4, 30.8, "这堵墙前，电子八成回头，两成穿了过去。",
                      "At this wall, eight in ten electrons turn back. Two in ten get through.")])


# ---------------------------------------------------------- 03 layers ----

SL_L, SL_N, SL_W, SL_Y0, SL_Y1, BEAM_Y = 560.0, 20, 40.0, 236.0, 536.0, 386.0


def layers_done(g):
    n = 0.0
    for k, tk in enumerate(TL.LAYERS):
        if g >= tk:
            n = k + min(1.0, (g - tk) / 0.22)
    return n


def decimal_parts(v):
    if v >= 0.999:
        return "", "1"
    e = math.floor(math.log10(v))
    sig = f"{v / 10 ** e:.1f}".replace(".", "").rstrip("0") or "0"
    return "0." + "0" * (-e - 1), sig


def back_text(T_):
    """'99.9999995%': just enough digits to show it is not 100."""
    if T_ >= 0.999:
        return "0%"
    d = max(1, math.ceil(-math.log10(100 * T_)) + 1)
    return f"{100 * (1 - T_):.{d}f}".rstrip("0").rstrip(".") + "%"


def _beam_level(n):
    return (math.log10(max(P.transmission(n * 0.1) if n > 0 else 1.0, 1e-12)) + 10) / 10


def _slices(F, t, g, a):
    if a <= 0.003:
        return
    n = layers_done(g)
    k = int(n)
    for i in range(SL_N):
        x = SL_L + i * SL_W
        lit = smooth(n - i) if n > i else 0.0
        plate = Stroke(x - 1, SL_Y0 - 1, x + 3, SL_Y1 + 2)
        plate.line((x, SL_Y0), (x, SL_Y1))
        plate.light(F, mix(CYAN, IVORY, 0.4), (0.16 + 0.22 * lit * _beam_level(i + 1)) * a)
        blend(F, np.full((int(SL_Y1 - SL_Y0), int(SL_W) - 2), 255, np.uint8), x + 1, SL_Y0,
              np.array([0.04, 0.06, 0.1], np.float32), 0.35 * a)
    xe = SL_L + SL_N * SL_W
    plate = Stroke(xe - 1, SL_Y0 - 1, xe + 3, SL_Y1 + 2)
    plate.line((xe, SL_Y0), (xe, SL_Y1))
    plate.light(F, mix(CYAN, IVORY, 0.4), 0.16 * a)
    ent = smooth(ramp(g, TL.LAYERS[0] - 0.9, 0.8))
    glow_poly(F, [(140, BEAM_Y), (SL_L, BEAM_Y)], GOLD, 0.95 * a * ent, th=3, glow=1.0, sigma=6)
    front = SL_L + n * SL_W
    if n > 0:
        for x0 in np.arange(SL_L, front, 4.0):
            lv = _beam_level((x0 - SL_L) / SL_W + 0.5)
            glow_poly(F, [(x0, BEAM_Y), (min(x0 + 4, front), BEAM_Y)], GOLD, a * lv ** 1.6, th=3, glow=0.9,
                      sigma=6)
        if n < SL_N:
            orb(F, front, BEAM_Y, 7, GOLD, a * _beam_level(n) ** 1.2)
    if n >= SL_N - 0.01:
        ex = smooth(ramp(g, TL.LAYERS[-1] + 0.3, 1.2))
        glow_poly(F, [(xe, BEAM_Y), (xe + 440 * ex, BEAM_Y)], GOLD, a * _beam_level(SL_N) ** 1.6, th=2,
                  glow=0.9, sigma=5)
    serif(f"第 {k} 层", 20, 500, 0.2).draw(F, 140, 150, IVORY, 0.85 * a)
    caps("LAYER", 10, 0.4).draw(F, 140, 172, DIM, a)
    serif(f"厚度 {k / 10:.1f} 纳米", 20, 500, 0.2).draw(F, 1780, 150, IVORY, 0.85 * a, align="right")
    caps("THICKNESS", 10, 0.4).draw(F, 1780, 172, DIM, a, align="right")
    Tk = P.transmission(k * 0.1) if k > 0 else 1.0
    zeros, sig = decimal_parts(Tk)
    tz = T(zeros, "stix", 56, None, per_char=False) if zeros else None
    ts_ = T(sig, "stix", 56, None, per_char=False)
    wz = tz.width if tz else 0
    x0 = 960 - (wz + ts_.width) / 2
    if tz:
        tz.draw(F, x0, 176, IVORY, 0.55 * a)
    ts_.draw(F, x0 + wz, 176, GOLD, a)
    label(F, "穿过的概率", "GETTING THROUGH", 960, 206, 0.9 * a, align="center", size=13)
    for tg, word, dur in ((TL.NM1, "一纳米：约七千分之一", 2.5), (TL.NM2, "两纳米：约两亿分之一", 3.2)):
        serif(word, 26, 500, 0.3).draw(F, 960, 622, GOLD, a * window(g, tg, tg + dur, 0.5, 0.4), align="center")
    for i in range(SL_N):
        x = SL_L + i * SL_W + SL_W / 2
        if n > i:
            orb(F, x, 566, 3, GOLD, a * (0.35 + 0.65 * _beam_level(i + 1)))
        else:
            blend(F, np.full((1, 6), 255, np.uint8), x - 3, 566, DIM, 0.5 * a)
    gauge(F, 960, 700, 560, 1 - Tk, CYAN, "回头的概率", "TURNING BACK", back_text(Tk), a=a)


def _glyph_points(text, size, tracking, cx, y, n, seed):
    tx = serif(text, size, 400, tracking)
    x0 = cx - tx.width / 2
    pts = []
    for m, dx, dy in tx.items:
        yy, xx = np.nonzero(m > 120)
        pts.append(np.column_stack([xx + x0 + dx, yy + y + dy]))
    pts = np.concatenate(pts).astype(np.float32)
    return pts[np.random.default_rng(seed).integers(0, len(pts), n)]


MORPH_Y = 450


def _morph(F, t, a=1.0):
    """'不可能' bursts into particles that settle as '概率很小' (t is local morph time)."""
    cx1 = 960 + 0.25 * 100
    if t < 2.4:
        statement(F, "不可能", cx1, MORPH_Y, t, 0.2, None, size=100, tracking=0.5, cps=3.4, fade_in=0.7,
                  a=a * (1 - smooth(ramp(t, 1.8, 0.5))))
    src = _once("src", lambda: _glyph_points("不可能", 100, 0.5, cx1, MORPH_Y, 5200, 1))
    dst = _once("dst", lambda: _glyph_points("概率很小", 100, 0.5, cx1, MORPH_Y, 5200, 2))
    jit = _once("jit", lambda: np.random.default_rng(4).normal(0, 1, (5200, 2)).astype(np.float32))
    if 1.7 < t < 4.7:
        u = ease_in_out(ramp(t, 2.0, 1.9))
        pts = src + (dst - src) * u + jit * 70 * math.sin(math.pi * u)
        pa = a * smooth(ramp(t, 1.7, 0.5)) * (1 - smooth(ramp(t, 4.0, 0.6)))
        xi, yi = pts[:, 0].astype(int), pts[:, 1].astype(int)
        y0, y1 = 180, 700
        ok = (xi > 0) & (xi < W) & (yi > y0) & (yi < y1)
        dens = np.bincount((yi[ok] - y0) * W + xi[ok], minlength=(y1 - y0) * W).reshape(y1 - y0, W)
        dens = dens.astype(np.float32)
        lm = cv2.GaussianBlur(dens, (0, 0), 0.8) * 1.6 + cv2.GaussianBlur(dens, (0, 0), 6) * 1.8
        F[y0:y1] += lm[..., None] * mix(IVORY, GOLD, smooth(u))[None, None] * pa
    if t > 3.9:
        statement(F, "概率很小", cx1, MORPH_Y, t, 3.9, 7.9, size=100, tracking=0.5, cps=40, fade_in=0.7,
                  color=GOLD, a=a)


Z_X0, Z_X1, Z_Y0, Z_DY, Z_TOP, Z_BOT = 150.0, 1790.0, 250.0, 46.0, 110, 690


def _zero_row():
    one = T("0", "stix", 34, None, per_char=False)
    n = int((Z_X1 - Z_X0) / (one.width + 6))
    return T("0" * n, "stix", 34, None, tracking=6 / 34, per_char=False), n


def _zeros(F, t, a):
    """Rows of zeros after the decimal point, arriving faster than you can read."""
    if a <= 0.003:
        return
    row, per = _once("zrow", _zero_row)
    u = max(0.0, t)
    count = 3.0 * u + 26.0 * u ** 2 + 14.0 * u ** 3
    head = T("0.", "stix", 34, None, per_char=False)
    rows_full = int(count // per)
    scroll = max(0.0, (count / per - 8.5)) * Z_DY
    if Z_Y0 - scroll > Z_TOP:
        head.draw(F, Z_X0 - head.width - 4, Z_Y0 - scroll, GOLD, a)
    m, dx, dy = row.items[0]
    for r in range(max(0, int(scroll // Z_DY) - 1), rows_full + 1):
        y = Z_Y0 + r * Z_DY - scroll
        if y < Z_TOP or y > Z_BOT:
            continue
        frac = min(1.0, count / per - r)
        if frac <= 0:
            continue
        age = count / per - r
        al = a * (0.25 + 0.5 * math.exp(-max(age - 1, 0) / 3.0))
        edge = np.clip((y - Z_TOP) / 110, 0, 1) * np.clip((Z_BOT - y) / 110, 0, 1)
        blend(F, m[:, :int(m.shape[1] * frac)], Z_X0 + dx, y + dy, IVORY, al * edge)


def layers(F, t, g, fi):
    chapter_mark(F, window(t, 0.3, 44.4, 1.0, 0.8), "03", "层", "THE LAYERS")
    _slices(F, t, g, window(t, 0.0, 17.6, 1.0, 0.8))
    f0 = TL.FORMULA - START["layers"]
    fa = window(t, f0, f0 + 4.6, 1.0, 0.6)
    if fa > 0:
        base = R([("T", "stix_it", 88, None), ("  ≈  ", "math", 80, None), ("e", "stix_it", 88, None)])
        sup = R([("−2", "stix", 46, None), ("κa", "stix_it", 46, None)])
        x0 = 960 - (base.width + sup.width) / 2
        base.draw(F, x0, 440, GOLD, fa)
        sup.draw(F, x0 + base.width + 4, 394, GOLD, fa)
        label(F, "隧穿概率", "TUNNELING PROBABILITY", 960, 504, fa, align="center")
        label(F, "κ 由墙高决定，a 是墙厚", "HEIGHT · THICKNESS", 960, 534, 0.8 * fa, align="center", size=14)
    m0 = TL.MORPH0 - START["layers"]
    if t > m0:
        _morph(F, t - m0, window(t, m0, m0 + 8.0, 0.01, 0.6))
    z0, z1 = (z - START["layers"] for z in TL.ZEROS)
    _zeros(F, t - z0, window(t, z0 - 0.2, z1 + 1.4, 0.4, 1.4))
    p0 = TL.PIVOT - START["layers"]
    statement(F, "可如果，撞得足够多次呢？", 960, 430, t, p0, 49.9, size=52, cps=7, gold={6, 7, 8, 9})
    subtitles(F, t, [(0.6, 5.6, "把墙切成薄片，每片只有 0.1 纳米。",
                      "Slice the wall into layers a tenth of a nanometre thick."),
                     (6.0, 9.0, "越往深处，每一层都只放过约三分之一。",
                      "Deeper in, each layer lets only about a third through."),
                     (9.4, 12.9, "一层接一层，概率塌得飞快——", "Layer after layer, the odds collapse —"),
                     (13.2, 17.4, "刻度上，它已经和 100% 分不出来了。",
                      "On the dial, it can no longer be told apart from 100%."),
                     (18.0, 22.2, "这就是隧穿的本质：不是一道门槛，而是一场指数级的衰减。",
                      "That is the heart of tunneling: not a threshold, but an exponential fade."),
                     (24.0, 29.6, "所谓「不可能」，其实只是「概率很小」。",
                      "What we call impossible is merely improbable."),
                     (31.0, 35.0, "那么，一个人穿墙而过的概率呢？",
                      "So what are the odds of a person walking through a wall?"),
                     (35.4, 41.0, "小数点后面的 0，就算写满整个可观测宇宙，也写不完。",
                      "Write out the zeros after the decimal point, and the observable universe runs out of room."),
                     (41.4, 45.0, "对我们来说，这和「不可能」没有区别。", "For us, that is as good as impossible."),
                     (p0 + 0.4, 49.6, "", "But what if you could hit the wall enough times?")])


# ---------------------------------------------------------- 04 prison ----

NUC = 240


def _nuclei():
    rng = np.random.default_rng(77)
    x, y, d = rng.uniform(70, W - 70, NUC), rng.uniform(70, 700, NUC), rng.power(2, NUC)
    img = np.zeros((H, W, 3), np.float32)
    for xi, yi, di in zip(x, y, d):
        add_sprite(img, xi, yi, 1.0 + 1.6 * di, mix(IVORY, GOLD, 0.3), 0.1 + 0.28 * di)
        add_sprite(img, xi, yi, 5 + 6 * di, mix(CYAN, IVORY, 0.4), 0.03 + 0.04 * di)
    return x, y, d, img


def _geiger(F, g, a, t_lo, t_hi):
    """A lump of uranium: dim nuclei, and every click of the counter is one of them letting go."""
    if a <= 0.003:
        return
    x, y, d, img = _once("nuclei", _nuclei)
    F += img * a
    for te, i, ang in _once("geiger", TL.geiger):
        q = g - te
        if te < t_lo or te > t_hi or q < 0 or q > 0.6:
            continue
        orb(F, x[i], y[i], 5 + 4 * d[i], GOLD, 1.3 * a * (1 - q / 0.6) ** 2, core=WARM_WHITE)
        streak(F, x[i], y[i], ang, 30 + 260 * ease_out(q / 0.6), GOLD, 0.9 * a * (1 - q / 0.6))


VR, VK, VH = 86.0, 0.36, 210.0          # px per nuclear radius, ellipse flattening, px per barrier height
VCX, VCY = 960.0, 440.0
FLOOR, E_A, RMAX = 0.0, 0.15, 8.5       # well floor, alpha energy (4.2 / 28 MeV), outer edge (radii)
R_E = 1 / E_A                           # where the Coulomb slope comes back down to E
RINGS = [1.12, 1.3, 1.55, 1.9, 2.4, 3.0, 3.8, 4.8, 6.0, 7.2, 8.5]
MERID = 36
F_ASSAULT, N_HALF = 1e21, 1.41e38       # hits per second; hits in one half-life of U-238
P_HIT = math.log(2) / N_HALF


def phi(g):
    return 0.35 + 0.045 * (g - TL.VOLCANO)


def vproj(r, th, z, ph, s=1.0, cx=VCX, cy=VCY):
    a = np.asarray(th) + ph
    return np.stack([cx + r * np.cos(a) * VR * s, cy + r * np.sin(a) * VR * VK * s - np.asarray(z) * VH * s], -1)


def _arcs(r, z, ph, s=1.0, cx=VCX, cy=VCY, n=96):
    """Ring split into its far (back) and near (front) halves."""
    th_b = np.linspace(math.pi, 2 * math.pi, n // 2) - ph
    th_f = np.linspace(0, math.pi, n // 2) - ph
    return vproj(r, th_b, z, ph, s, cx, cy), vproj(r, th_f, z, ph, s, cx, cy)


def _volcano(F, g, a, build=1.0, s=1.0, cx=VCX, cy=VCY, wall_glow=0.0):
    """Gamow's picture of a nucleus: a crater (the well) inside a 1/r Coulomb mountain."""
    if a <= 0.003:
        return
    ph = phi(g)
    pad = 12
    box = (int(cx - RMAX * VR * s - pad), int(cy - (VH + VR * VK) * s - pad),
           int(cx + RMAX * VR * s + pad), int(cy + RMAX * VR * VK * s - FLOOR * VH * s + pad))
    back, front = Stroke(*box), Stroke(*box)
    r_end = 1 + (RMAX - 1) * ease_in_out(build)
    for r in (0.45, 0.8, 1.0):                       # crater floor
        b, f = _arcs(r, FLOOR, ph, s, cx, cy)
        back.poly(b)
        front.poly(f)
    b, f = _arcs(1.0, 1.0, ph, s, cx, cy)           # rim
    back.poly(b)
    front.poly(f)
    for r in RINGS:
        if r <= r_end:
            b, f = _arcs(r, 1 / r, ph, s, cx, cy, n=160 if r > 4 else 96)
            back.poly(b)
            front.poly(f)
    rr = 1 + (np.linspace(0, 1, 36) ** 1.8) * (r_end - 1)
    for k in range(MERID):
        th = 2 * math.pi * k / MERID
        st = front if math.sin(th + ph) > 0 else back
        st.poly(vproj(rr, th, 1 / rr, ph, s, cx, cy))
        if k % 3 == 0:                               # inner wall and floor spokes
            st.poly(vproj(np.array([1.0, 1.0]), th, np.array([1.0, FLOOR]), ph, s, cx, cy))
            st.poly(vproj(np.array([1.0, 0.0]), th, np.array([FLOOR, FLOOR]), ph, s, cx, cy))
    back.glow(F, STEEL, 0.18 * a, 4)
    back.light(F, STEEL, 0.22 * a)
    front.glow(F, STEEL, 0.45 * a, 4)
    front.light(F, mix(STEEL, WHITE, 0.2), 0.6 * a)
    if wall_glow > 0:                                # the inner wall lit by hits
        w = Stroke(*box)
        b, f = _arcs(1.0, E_A, ph, s, cx, cy)
        w.poly(b)
        w.poly(f)
        w.glow(F, CYAN, 0.9 * wall_glow * a, 6)
        w.light(F, CYAN, 0.6 * wall_glow * a)
    en = Stroke(*box)                                # the particle's energy, inside and outside
    for r in (1.0, R_E):
        if r <= r_end + 0.01:
            pts = np.concatenate(_arcs(r, E_A, ph, s, cx, cy, n=200 if r > 2 else 80))
            for i in range(0, len(pts) - 1, 2):
                en.line(pts[i], pts[i + 1])
    en.light(F, IVORY, 0.35 * a)


def _tunnel(F, g, a, th, prog=1.0):
    """Dashed gold path through the wall at the particle's energy."""
    if a <= 0.003 or prog <= 0:
        return
    rr = np.linspace(1.0, 1.0 + (R_E - 1.0) * prog, 60)
    pts = vproj(rr, th, np.full_like(rr, E_A), phi(g))
    s = Stroke(pts[:, 0].min() - 20, pts[:, 1].min() - 20, pts[:, 0].max() + 20, pts[:, 1].max() + 20)
    for i in range(0, len(pts) - 1, 2):
        s.line(pts[i], pts[i + 1], th=2)
    s.glow(F, GOLD, 0.8 * a, 5)
    s.light(F, GOLD, 0.9 * a)


def _hit_angles():
    hits = TL.prison_hits()
    rng = np.random.default_rng(5)
    ang = [0.4]
    for _ in hits[1:]:
        ang.append(ang[-1] + math.pi + rng.uniform(-0.75, 0.75))
    return np.array(hits), np.array(ang)


def _escape_theta():
    return 0.5 - phi(TL.ESCAPE)


def _alpha(F, g, a):
    """The alpha particle at its energy level: visible bounces, then a blur, then the escape.
    Returns how brightly the inner wall should flash."""
    if a <= 0.003:
        return 0.0
    hits, ang = _once("hitang", _hit_angles)
    ph = phi(g)
    rw = 0.93
    if g < hits[0]:
        u = ramp(g, hits[0] - 1.0, 1.0)
        p = vproj(rw * u, ang[0], E_A, ph)
        orb(F, p[0], p[1], 7, GOLD, a * smooth(ramp(g, hits[0] - 1.4, 0.5)), core=WARM_WHITE)
        return 0.0
    if g < TL.BLUR:
        k = int(np.searchsorted(hits, g, side="right") - 1)
        if k + 1 < len(hits):
            u = (g - hits[k]) / (hits[k + 1] - hits[k])
            pa, pb = vproj(rw, ang[k], E_A, ph), vproj(rw, ang[k + 1], E_A, ph)
            p = pa + (pb - pa) * u
            orb(F, p[0], p[1], 7, GOLD, a, core=WARM_WHITE)
        q = g - hits[k]
        if q < 0.45:
            pw = vproj(1.0, ang[k], E_A, ph)
            orb(F, pw[0], pw[1], 14, CYAN, 0.9 * a * (1 - q / 0.45) ** 2)
        return 0.25 * max(0.0, 1 - q / 0.3)
    th_e = _escape_theta()
    if g < TL.COUNT[1] + 0.2:                        # too fast to see: a web of chords
        rng = np.random.default_rng(int(g * 30))
        th = rng.uniform(0, 2 * math.pi, 22)
        th2 = th + math.pi + rng.uniform(-0.8, 0.8, 22)
        yc = VCY - E_A * VH
        st = Stroke(VCX - VR - 10, yc - VR * VK - 10, VCX + VR + 10, yc + VR * VK + 10)
        for a1, a2 in zip(th, th2):
            st.line(vproj(rw, a1, E_A, ph), vproj(rw, a2, E_A, ph))
        bl = smooth(ramp(g, TL.BLUR - 0.3, 0.6))
        st.light(F, GOLD, 0.16 * a * bl)
        st.glow(F, GOLD, 0.35 * a * bl, 6)
        add_sprite(F, VCX, yc, 40, GOLD, 0.18 * a * bl)
        return 0.45 + 0.25 * rng.random()
    if g < TL.ESCAPE:                                # one last run at the wall
        u = ramp(g, TL.COUNT[1] + 0.2, TL.ESCAPE - TL.COUNT[1] - 0.2)
        p = vproj(rw * u, th_e, E_A, ph)
        orb(F, p[0], p[1], 7, GOLD, a, core=WARM_WHITE)
        return 0.0
    q = g - TL.ESCAPE
    if q < 0.45:                                     # inside the wall: a ghost on the tunnel
        p = vproj(1.0 + (R_E - 1.0) * ease_in_out(q / 0.45), th_e, E_A, ph)
        orb(F, p[0], p[1], 6, GOLD, 0.45 * a)
        return 0.0
    q -= 0.45                                        # out: it rolls down the Coulomb slope, speeding up
    r = R_E + 0.5 * 9.0 * q * q
    if r < RMAX:
        p = vproj(r, th_e, 1 / r, ph)
    else:
        p0 = vproj(RMAX, th_e, 1 / RMAX, ph)
        vx, vy = math.cos(th_e + ph), math.sin(th_e + ph) * VK
        nrm = math.hypot(vx, vy)
        d = (r - RMAX) * VR
        p = p0 + np.array([vx / nrm * d, vy / nrm * d])
    orb(F, p[0], p[1], 8, GOLD, a * (1 - smooth(ramp(q, 1.6, 0.5))), core=WARM_WHITE)
    if q < 0.6:
        e = vproj(R_E, th_e, E_A, ph)
        orb(F, e[0], e[1], 22, GOLD, 1.1 * a * (1 - q / 0.6) ** 2)
    return 0.0


def hits_at(g):
    t0, t1 = TL.COUNT
    if g < t0:
        return 0.0
    hits, _ = _once("hitang", _hit_angles)
    if g < TL.BLUR:
        return float(((hits >= t0) & (hits <= g)).sum())
    n0 = max(1.0, float(((hits >= t0) & (hits <= TL.BLUR)).sum()))
    u = ramp(g, TL.BLUR, t1 - TL.BLUR)
    return 10 ** lerp(math.log10(n0), math.log10(N_HALF), ease_in_out(u))


def fmt_time(s):
    yr = s / 3.156e7
    if s < 1:
        return "不到 1 秒"
    if s < 60:
        return f"{int(s)} 秒"
    if s < 3600:
        return f"{int(s / 60)} 分钟"
    if s < 86400:
        return f"{int(s / 3600)} 小时"
    if yr < 1:
        return f"{int(s / 86400)} 天"
    if yr < 1e4:
        return f"{int(yr):,} 年"
    if yr < 1e8:
        return f"{int(yr / 1e4):,} 万年"
    return f"{yr / 1e8:.0f} 亿年"


def _counters(F, g, a):
    if a <= 0.003:
        return
    N = hits_at(min(g, TL.COUNT[1]))
    label(F, "撞墙次数", "HITS", 150, 140, a)
    if N < 1e6:
        T(f"{int(N):,}", "stix", 46, None, per_char=False).draw(F, 150, 200, GOLD, a)
    else:
        e = int(math.floor(math.log10(N)))
        draw_sci(F, f"{N / 10 ** e:.1f}", e, 150, 200, 46, GOLD, a)
    label(F, "经过时间", "ELAPSED", 1770, 140, a, align="right")
    serif(fmt_time(N / F_ASSAULT), 38, 500, 0.06).draw(F, 1770, 200, IVORY, a, align="right")
    v = math.exp(-N * P_HIT)
    pct = 100 * v
    txt = "100%" if pct > 99.95 else f"{pct:.1f}%" if pct > 99 else f"{pct:.0f}%"
    gauge(F, 960, 716, 560, v, CYAN, "仍被困住的概率", "STILL TRAPPED", txt, a=a)


def prison(F, t, g, fi):
    P0 = START["prison"]
    chapter_mark(F, window(t, 8.0, 47.4, 1.0, 0.8), "04", "狱", "THE PRISON")
    # a Geiger counter in the dark: the sound of tunnelling, before we know it is
    _geiger(F, g, window(t, 0.2, 10.4, 1.2, 1.4), P0, TL.YEAR28 + 1.2)
    y0 = TL.YEAR28 - P0
    statement(F, "1928", 960, 420, t, y0, y0 + 4.6, size=110, wght=400, tracking=0.3, cps=6, fade_in=0.8)
    caps("GÖTTINGEN", 13, 0.6).draw(F, 960, 470, GOLD, window(t, y0 + 0.8, y0 + 4.6, 0.8, 0.8), align="center")
    # the nucleus: a prison
    v0, f0 = TL.VOLCANO - P0, TL.FIELD - P0
    shrink = ease_in(ramp(t, f0 - 0.4, 1.6))
    va = smooth(ramp(t, v0, 1.2)) * (1 - smooth(ramp(t, f0 - 0.2, 1.4)))
    if va > 0:
        flash = _alpha(F, g, va) if shrink <= 0 else 0.0
        _volcano(F, g, va, build=ramp(t, v0, 3.4), s=1 - 0.92 * shrink, cy=VCY - 60 * shrink, wall_glow=flash)
        th_e = _escape_theta()
        gm = TL.GAMOW - P0
        _tunnel(F, g, 0.8 * va * window(t, gm, TL.COUNT[0] - P0 - 0.2, 0.6, 1.0), th_e,
                prog=ease_in_out(ramp(t, gm, 1.6)))
        es = TL.ESCAPE - P0
        _tunnel(F, g, va * window(t, es, es + 2.6, 0.12, 1.4), th_e, prog=ease_out(ramp(t, es, 0.45)))
        la = va * window(t, v0 + 2.0, TL.COUNT[0] - P0, 1.0, 0.8)
        ph = phi(g)
        rim = vproj(1.0, -ph, 1.0, ph)
        label(F, "墙", "THE WALL", rim[0] + 22, rim[1] - 8, la, color=STEEL)
        en = vproj(R_E, -ph, E_A, ph)
        label(F, "α 粒子的能量", "ITS ENERGY", en[0] + 18, en[1] - 6, la)
        label(F, "铀-238 原子核", "URANIUM-238 NUCLEUS", 960, 690, la, align="center")
        _counters(F, g, va * window(t, TL.COUNT[0] - P0, f0, 0.6, 1.0))
    _geiger(F, g, smooth(ramp(t, f0 - 0.2, 1.4)) * (1 - smooth(ramp(t, 47.0, 1.0))), TL.FIELD, 1e9)
    subtitles(F, t, [(0.8, 3.8, "这是盖革计数器的声音。", "This is the sound of a Geiger counter."),
                     (4.2, 7.6, "每一声「咔嗒」，都是一个粒子，从原子核里逃了出来。",
                      "Every click is a particle escaping from an atomic nucleus."),
                     (8.4, 12.4, "可直到 1928 年，没人说得清：它是怎么逃出来的。",
                      "Yet until 1928, no one could explain how it got out."),
                     (13.0, 17.6, "原子核像一座监狱：四周的墙，比 α 粒子的能量高出好几倍。",
                      "A nucleus is a prison: its walls stand several times higher than the alpha particle's energy."),
                     (18.0, 22.0, "按经典物理，它永远翻不出去。", "By classical physics, it could never get out."),
                     (22.4, 27.0, "24 岁的伽莫夫说：它不是翻过去的，是穿过去的。",
                      "George Gamow, aged 24, said: it does not climb over. It tunnels through."),
                     (27.4, 31.6, "它每秒撞墙约 10²¹ 次，几乎每一次，都被弹了回来。",
                      "It hits the wall some 10²¹ times a second, and almost every time it bounces back."),
                     (32.0, 35.6, "平均要撞上约 10³⁸ 次，才穿得过去一次。",
                      "On average it takes about 10³⁸ hits to get through once."),
                     (36.0, 41.0, "一块铀，要等 45 亿年，才有一半的原子核逃出来——差不多正是地球的年龄。",
                      "Half the nuclei in a lump of uranium take 4.5 billion years to get out — about the age of the Earth."),
                     (43.4, 47.6, "所以，每一声「咔嗒」背后，都是 10³⁸ 次撞墙。",
                      "So behind every single click lie 10³⁸ hits on the wall.")])


# ----------------------------------------------------------- 05 light ----

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
    h, w = H // 2, W // 2
    ys, xs = _once("half_grid", lambda: tuple(np.mgrid[0:h, 0:w].astype(np.float32) * 2))
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
    disc = cv2.resize(edge, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
    F *= 1 - disc * a
    F += cv2.resize(col, (W, H), interpolation=cv2.INTER_LINEAR) * disc * 0.72 * a
    F += cv2.resize(corona, (W, H), interpolation=cv2.INTER_LINEAR) * 0.72 * a


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


NPRO = 46


def _protons():
    rng = np.random.default_rng(23)
    return (rng.uniform(80, W - 80, NPRO), rng.uniform(90, 700, NPRO), rng.uniform(0, 6.28, (NPRO, 2)),
            rng.uniform(0.15, 0.35, NPRO))


def _proton_xy(g, s, ox=0.0, oy=0.0):
    px, py, ph, sp = _once("protons", _protons)
    x = 960 + (px + ox + 18 * np.sin(sp * g + ph[:, 0]) - 960) * s
    y = CY + (py + oy + 18 * np.sin(sp * g * 1.3 + ph[:, 1]) - CY) * s
    return x, y


PAIRS = [((700, 300), 0.3), ((1250, 520), 2.2), ((480, 560), -0.9), ((1420, 250), 1.2)]


def _proton_pairs(F, g, a, s):
    """Scripted near-misses: two protons close in, meet the electric wall, and fly apart."""
    for (c, ang), tc in zip(PAIRS, TL.BOUNCES):
        q = g - tc
        if abs(q) > 1.6:
            continue
        pa = a * (1 - smooth(ramp(abs(q), 1.1, 0.5)))
        d = 30 + 170 * abs(q)
        out_ang = ang + (0.65 if q > 0 else 0.0)
        for sgn in (-1, 1):
            x = 960 + (c[0] + sgn * d * math.cos(out_ang) - 960) * s
            y = CY + (c[1] + sgn * d * math.sin(out_ang) - CY) * s
            orb(F, x, y, 5, CYAN, pa, core=mix(CYAN, WHITE, 0.6))
            st = Stroke(x - 30, y - 30, x + 31, y + 31)
            st.circle((x, y), 24 * s, th=1)
            st.light(F, CYAN, 0.22 * pa)
        if 0 <= q < 0.5:
            orb(F, 960 + (c[0] - 960) * s, CY + (c[1] - CY) * s, 20, CYAN, 0.9 * a * (1 - q / 0.5) ** 2)


def zoom_level(g):
    """log10 of the number of protons in view."""
    u = ramp(g, TL.ZOOM[0], TL.ZOOM[1] - TL.ZOOM[0])
    return 1.7 + 55.3 * u ** 2.2


def _zoom_scale(g):
    return 10 ** (-(zoom_level(g) - 1.7) / 2)


def _sparks_frame(fb):
    """Fusions born on frame fb (each one a proton that finally got through)."""
    f_first = int(round(TL.FIRST_SPARK * 30))
    if fb < f_first:
        return np.zeros((0, 2))
    if fb == f_first:
        x, y = _proton_xy(fb / 30, _zoom_scale(fb / 30))
        return np.array([[x[0], y[0]]])
    v = ramp(fb / 30, TL.FIRST_SPARK + 0.8, TL.SUNFORM - TL.FIRST_SPARK - 0.8)
    if v <= 0:
        return np.zeros((0, 2))
    n = int(10 ** (3.6 * v ** 1.3)) - 1
    rng = np.random.default_rng(fb)
    return np.column_stack([rng.uniform(0, W, n), rng.uniform(0, H, n)])


def _text_mask():
    y, x = np.mgrid[0:H, 0:W].astype(np.float32)
    bottom = 1 - 0.55 * np.clip((y - 700) / 90, 0, 1)
    counter = 0.6 * np.exp(-((x - 960) / 240) ** 2 - ((y - 150) / 70) ** 2)
    return bottom[..., None].astype(np.float32), counter[..., None].astype(np.float32)


def _field(F, g, fi, a, rd, za=1.0):
    """The core seen from ever farther away; every spark is one fusion."""
    z = zoom_level(g)
    s = _zoom_scale(g)
    Lr = np.zeros_like(F)
    da = a * (1 - smooth((z - 2.6) / 1.2))
    if da > 0.003:                                  # discrete protons, tiled as we pull back
        tiles = int(math.ceil(0.5 / s))
        for ix in range(-tiles, tiles + 1):
            for iy in range(-tiles, tiles + 1):
                x, y = _proton_xy(g, s, ix * W, iy * H)
                ok = (x > -20) & (x < W + 20) & (y > -20) & (y < H + 20)
                for xi, yi in zip(x[ok], y[ok]):
                    add_sprite(Lr, xi, yi, max(1.2, 5 * s), mix(CYAN, WHITE, 0.4), 0.55 * da)
                    if s > 0.3:
                        add_sprite(Lr, xi, yi, 22 * s, CYAN, 0.05 * da)
        _proton_pairs(Lr, g, da, s)
    pa = a * smooth((z - 2.2) / 2.0)
    if pa > 0.003:                                  # beyond resolution: a shimmering plasma
        n1, n2 = _once("noise", _noise_tex)
        tex = 0.5 + 0.5 * (0.6 * np.roll(n1, fi, 1)[:H // 2, :W // 2] + 0.4 * np.roll(n2, -fi, 0)[:H // 2, :W // 2])
        tex = cv2.resize(tex.astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR)
        Lr += tex[..., None] * mix(CYAN, IVORY, 0.3) * 0.07 * pa
    acc = np.zeros((H // 4, W // 4), np.float32)    # sparks live five frames
    for k in range(5):
        pts = _sparks_frame(fi - k)
        if len(pts) == 0:
            continue
        w = (1 - k / 5) ** 1.5
        if len(pts) < 50:
            one = len(pts) == 1
            for x, y in pts:
                orb(Lr, x, y, 6 if one else 4, GOLD, (1.8 if one else 1.0) * w * a, core=WARM_WHITE)
        else:
            xi = np.clip((pts[:, 0] / 4).astype(int), 0, W // 4 - 1)
            yi = np.clip((pts[:, 1] / 4).astype(int), 0, H // 4 - 1)
            np.add.at(acc, (yi, xi), w)
    if acc.any():
        acc = cv2.GaussianBlur(acc, (0, 0), 0.7) * 1.5
        warm = cv2.GaussianBlur(acc, (0, 0), 14) * 1.4
        Lr += cv2.resize(np.minimum(acc, 2.0), (W, H), interpolation=cv2.INTER_LINEAR)[..., None] * GOLD * a
        Lr += cv2.resize(warm, (W, H), interpolation=cv2.INTER_LINEAR)[..., None] * mix(GOLD, WARM_WHITE, 0.3) * a
    bottom, counter = _once("text_mask", _text_mask)  # keep the counter and subtitles readable
    Lr *= bottom * (1 - za * counter)
    if rd < 2000:                                   # the field gathers into a disc: the Sun
        ys, xs = _once("q_grid", lambda: np.mgrid[0:H // 4, 0:W // 4].astype(np.float32) * 4)
        m = np.clip((rd - np.sqrt((xs - 960) ** 2 + (ys - CY) ** 2)) / 18, 0, 1)
        Lr *= cv2.resize(m, (W, H), interpolation=cv2.INTER_LINEAR)[..., None]
    F += Lr


def _phone_rect(sc):
    w, h = W * sc, H * sc
    return 960 - w / 2, CY - h / 2, w, h


def _rounded_mask(w, h, r):
    m = np.zeros((h, w), np.uint8)
    r = int(r)
    cv2.rectangle(m, (r, 0), (w - r, h), 255, -1)
    cv2.rectangle(m, (0, r), (w, h - r), 255, -1)
    for cx, cy in ((r, r), (w - r, r), (r, h - r), (w - r, h - r)):
        cv2.circle(m, (cx * 16, cy * 16), r * 16, 255, -1, cv2.LINE_AA, 4)
    return m.astype(np.float32) / 255


def _round_rect(x, y, w, h, r, n=10):
    pts = []
    for cx, cy, a0 in ((x + w - r, y + r, -90), (x + w - r, y + h - r, 0), (x + r, y + h - r, 90),
                       (x + r, y + r, 180)):
        for k in range(n + 1):
            a = math.radians(a0 + 90 * k / n)
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    return pts


def _flash_cell(F, g, a, x0, y0, w, h):
    """A flash memory cell: electrons tunnel up through a few-nanometre wall into the storage layer."""
    if a <= 0.003:
        return
    cx = x0 + w / 2
    ych, ywall, ytop, ybox = y0 + h * 0.76, y0 + h * 0.62, y0 + h * 0.32, y0 + h * 0.54
    bx0, bx1 = cx - w * 0.30, cx + w * 0.30
    st = Stroke(bx0 - 4, ytop - 4, bx1 + 5, ybox + 5)
    st.poly([(bx0, ytop), (bx1, ytop), (bx1, ybox), (bx0, ybox)], closed=True)
    st.glow(F, GOLD, 0.25 * a, 4)
    st.light(F, GOLD, 0.6 * a)
    glow_poly(F, [(bx0 - 20, ywall), (bx1 + 20, ywall)], CYAN, 0.9 * a, th=2, glow=1.0, sigma=5)
    label(F, "存储层", "STORAGE", bx0 + 14, ytop + 26, a, color=GOLD)
    label(F, "几纳米厚的墙", "A FEW NANOMETRES", bx1, ywall + 26, a, color=CYAN, align="right")
    label(F, "电子", "ELECTRONS", x0 + w * 0.08, ych + 40, a)
    rng = np.random.default_rng(9)
    lanes = rng.uniform(0, 1, 28)
    for i, ph0 in enumerate(lanes):                  # electrons drifting in the channel
        x = x0 + w * 0.08 + ((g * 0.12 + ph0) % 1.0) * w * 0.84
        add_sprite(F, x, ych + 6 * math.sin(i * 1.7 + g * 2), 2.2, CYAN, 0.8 * a)
    t0 = TL.FLASH + 0.6
    n_jump = int(max(0.0, (g - t0) / 0.32))
    for j in range(min(n_jump + 1, 40)):             # each jump: through the wall, into the store
        rx, ry = rng.uniform(0.08, 0.92), rng.uniform(0.2, 0.8)
        q = g - (t0 + 0.32 * j)
        if q < 0:
            continue
        x = bx0 + (bx1 - bx0) * rx
        ys = ytop + (ybox - ytop) * ry
        if q < 0.5:
            y = lerp(ych, ys, ease_in_out(q / 0.5))
            add_sprite(F, x, y, 2.4, GOLD if y < ywall else CYAN, a)
            if abs(y - ywall) < 8:
                orb(F, x, ywall, 7, GOLD, 0.9 * a)
        else:
            add_sprite(F, x, ys, 2.4, GOLD, 0.85 * a)
    serif("闪存", 20, 500, 0.4).draw(F, cx, y0 + h * 0.15, IVORY, a, align="center")
    caps("FLASH MEMORY", 11, 0.5).draw(F, cx, y0 + h * 0.15 + 22, DIM, a, align="center")


def _phone(F, base, t, g, shrink):
    """Pull back from the picture: it was playing on a phone all along."""
    screen = F.copy()
    F[:] = base
    sc = lerp(1.0, 0.47, shrink)
    x0, y0, w, h = _phone_rect(sc)
    iw, ih = int(round(w)), int(round(h))
    ix, iy = int(round(x0)), int(round(y0))
    img = cv2.resize(screen, (iw, ih), interpolation=cv2.INTER_AREA)
    fl = TL.FLASH - START["light"]
    ca = smooth(ramp(t, fl - 0.4, 1.0))
    img *= 1 - 0.97 * ca
    m = _rounded_mask(iw, ih, 26 * sc / 0.47)[..., None]
    reg = F[iy:iy + ih, ix:ix + iw]
    reg[:] = reg * (1 - m) + img[:reg.shape[0], :reg.shape[1]] * m
    pad = 16 * sc / 0.47
    st = Stroke(x0 - pad - 6, y0 - pad - 6, x0 + w + pad + 7, y0 + h + pad + 7)
    st.poly(_round_rect(x0 - pad, y0 - pad, w + 2 * pad, h + 2 * pad, 40 * sc / 0.47), closed=True)
    st.glow(F, IVORY, 0.25 * shrink, 5)
    st.light(F, IVORY, 0.55 * shrink)
    orb(F, x0 - pad / 2, CY, 2.5, IVORY, 0.5 * shrink)
    _flash_cell(F, g, ca * (1 - smooth(ramp(t, 57.0, 0.9))), x0, y0, w, h)


def light(F, t, g, fi):
    P0 = START["light"]
    chapter_mark(F, window(t, 0.3, 48.2, 1.0, 0.8), "05", "光", "THE LIGHT")
    base = F.copy()
    # 1926: Eddington and the critics
    add_sprite(F, 960, CY, 260, mix(GOLD, CRIMSON, 0.5), 0.05 * window(t, 0, 13.8, 2.0, 1.4))
    q0 = TL.QUOTE - P0
    statement(F, "我们不和说恒星不够热的人争论，", 960, 340, t, q0, q0 + 8.8, size=40, cps=10)
    statement(F, "我们只请他，去找一个更热的地方。", 960, 410, t, q0 + 1.9, q0 + 8.8, size=40, cps=10,
              gold={10, 11, 12, 13, 14})
    qa = window(t, q0 + 4.0, q0 + 8.8, 0.8, 0.9)
    serif("—— 亚瑟·爱丁顿，1926", 18, 500, 0.2).draw(F, 960, 474, DIM, qa, align="center")
    italic("“We tell him to go and find a hotter place.”", 22).draw(F, 960, 512, DIM, 0.85 * qa, align="center")
    # 1929
    y0 = TL.YEAR29 - P0
    statement(F, "1929", 960, 420, t, y0, y0 + 4.4, size=110, wght=400, tracking=0.3, cps=6, fade_in=0.8)
    caps("ATKINSON  &  HOUTERMANS", 13, 0.5).draw(F, 960, 470, GOLD, window(t, y0 + 0.8, y0 + 4.4, 0.8, 0.8),
                                                   align="center")
    # the core, pulled back from 10^2 protons to 10^57 - until the sparks are sunlight
    z0, sf = TL.ZOOM[0] - P0, TL.SUNFORM - P0
    fa = window(t, y0 + 4.4, sf + 3.4, 1.2, 1.6)
    za = window(t, z0, sf + 1.2, 0.6, 1.0)
    if fa > 0:
        _field(F, g, fi, fa, lerp(2200, 230, ease_out(ramp(t, sf, 2.6))), za)
    if za > 0:
        label(F, "视野中的质子", "PROTONS IN VIEW", 960, 120, za, align="center")
        draw_sci(F, None, int(zoom_level(g)), 960, 176, 44, IVORY, za, align="center")
    heat = smooth(ramp(t, sf + 1.4, 2.0))
    pan = ease_in_out(ramp(t, TL.EARTH - P0, 4.5))
    cx = lerp(960, 560, pan)
    R0 = lerp(230 + 8 * ease_out(ramp(t, sf + 2.6, 10)), 160, pan)
    if heat > 0:
        sun_disc(F, t, heat, cx, CY, R0, 1.0)
    if pan > 0:
        ea = smooth(ramp(t, TL.EARTH - P0 + 1.0, 2.0))
        sp = _once("earth", _earth_sprite)
        ex = 1450.0
        E.add_rgb(F, sp, ex - sp.shape[1] / 2, CY - sp.shape[0] / 2, ea)
        rng = np.random.default_rng(17)
        for i in range(28):
            ph = (t * 0.22 + rng.random()) % 1.0
            x = lerp(cx + R0 + 20, ex - 26, ph)
            y = CY + rng.normal(0, 26) * (1 - ph)
            add_sprite(F, x, y, 1.4, GOLD, 0.45 * ea * math.sin(math.pi * ph))
        serif("地球", 14, 500, 0.3).draw(F, ex, CY + 62, DIM, ea, align="center")
    shrink = ease_in_out(ramp(t, TL.PHONE - P0, 1.8))
    if shrink > 0:
        _phone(F, base, t, g, shrink)
    subtitles(F, t, [(0.6, 4.6, "同一时期，天文学家也被困住了：太阳中心，似乎不够热。",
                      "Astronomers were stuck as well: the Sun's core seemed too cold for fusion."),
                     (q0 + 4.8, q0 + 8.6, "（他说的「更热的地方」，是地狱。）",
                      "(The hotter place he had in mind was hell.)"),
                     (y0 + 0.4, y0 + 4.6, "1929 年，阿特金森和豪特曼斯找到了答案：隧穿。",
                      "In 1929, Atkinson and Houtermans found the answer: tunnelling."),
                     (19.0, 23.0, "质子之间隔着一堵电的墙。它们不必翻过去，可以穿过去。",
                      "An electric wall stands between protons. They need not climb it — they can tunnel through."),
                     (23.4, 27.4, "可这种穿越难得惊人：一个质子，平均要等上几十亿年。",
                      "Yet it is astonishingly rare: a proton waits, on average, billions of years."),
                     (27.8, 31.8, "可太阳里，有约 10⁵⁷ 个质子。", "But the Sun holds some 10⁵⁷ protons."),
                     (32.2, 36.2, "每一秒，都有约 3.6×10³⁸ 个，穿过那堵墙。",
                      "Every second, about 3.6×10³⁸ of them tunnel through."),
                     (36.6, 40.4, "无数个「几乎不可能」，汇成了阳光。",
                      "Countless near-impossibilities add up to sunlight."),
                     (40.8, 45.4, "正因为每一次都这么难，太阳才没有一口气烧完，而是稳稳地亮了四十六亿年——",
                      "Because each crossing is so hard, the Sun did not burn out at once: it has shone for 4.6 billion years —"),
                     (45.8, 48.4, "久到足以让生命出现。", "long enough for life to appear."),
                     (49.0, 52.4, "而你手机里的每一张照片，", "And every photo on your phone"),
                     (52.8, 57.4, "都是电子穿过几纳米厚的墙，一个个写进去的。",
                      "was written by electrons tunnelling, one by one, through a wall a few nanometres thick.")])


# ------------------------------------------------------------ epilogue ---

def _ridge(seed, base, amp, n=480):
    rng = np.random.default_rng(seed)
    x = np.linspace(-20, W + 20, n)
    y = np.full(n, float(base))
    for f, a in ((1.2, 1.0), (2.7, 0.5), (6.1, 0.22), (13.0, 0.08)):
        y -= amp * a * np.sin(2 * math.pi * f * x / W + rng.uniform(0, 6.28))
    return np.column_stack([x, y])


def _great_wall(ridge):
    """Crenellated wall riding the ridge, with three watchtowers."""
    x, y = ridge[:, 0], ridge[:, 1] - 16
    xs = np.arange(0, W, 4.0)
    ys = np.interp(xs, x, y)
    top = np.column_stack([xs, ys - np.where((np.arange(len(xs)) // 3) % 2 == 0, 6, 0)])
    towers = []
    for tx in (430, 1110, 1630):
        ty = float(np.interp(tx, x, y))
        towers.append([(tx - 20, ty), (tx - 20, ty - 34), (tx - 12, ty - 34), (tx - 12, ty - 40), (tx - 4, ty - 40),
                       (tx - 4, ty - 34), (tx + 4, ty - 34), (tx + 4, ty - 40), (tx + 12, ty - 40),
                       (tx + 12, ty - 34), (tx + 20, ty - 34), (tx + 20, ty)])
    return top, towers


def _world(F, t, a):
    """Mountains at dawn and an old wall along the ridge - light leaking over and through it."""
    if a <= 0.003:
        return
    far = _once("rf", lambda: _ridge(3, 520, 46))
    mid = _once("rm", lambda: _ridge(7, 570, 60))
    near = _once("rn", lambda: _ridge(11, 640, 70))
    top, towers = _once("gw", lambda: _great_wall(near))
    rise = ease_out(ramp(t, 0.0, 14.0))
    sx, sy = 1240, lerp(640, 520, rise)
    add_sprite(F, sx, sy, 300, mix(GOLD, CRIMSON, 0.25), 0.16 * a)
    add_sprite(F, sx, sy, 110, GOLD, 0.35 * a)
    add_sprite(F, sx, sy, 34, WARM_WHITE, 0.9 * a)
    dark = np.array([0.012, 0.016, 0.026], np.float32)
    for rid, col, gain in ((far, CYAN, 0.18), (mid, STEEL, 0.25)):
        s = Stroke(0, rid[:, 1].min() - 12, W, H)
        s.fill(np.vstack([rid, [[W + 20, H], [-20, H]]]))
        s.blit(F, dark, 0.85 * a)
        glow_poly(F, rid, col, gain * a, th=1, glow=0.5, sigma=4,
                  bounds=(0, rid[:, 1].min() - 20, W, rid[:, 1].max() + 20))
    s = Stroke(0, top[:, 1].min() - 50, W, H)
    s.fill(np.vstack([top, [[W, H], [0, H]]]))
    for tw in towers:
        s.fill(np.array(tw))
    s.blit(F, dark, 0.95 * a)
    bounds = (0, top[:, 1].min() - 60, W, near[:, 1].max() + 30)
    glow_poly(F, top, GOLD, 0.55 * a, th=1, glow=0.6, sigma=4, bounds=bounds)
    glow_poly(F, near, mix(GOLD, IVORY, 0.4), 0.3 * a, th=1, glow=0.4, sigma=4, bounds=bounds)
    for tw in towers:
        glow_poly(F, tw, GOLD, 0.6 * a, th=1, glow=0.6, sigma=4)
    rng = np.random.default_rng(44)                  # motes drifting up through the wall
    for i in range(26):
        ph = (t * 0.035 + rng.random()) % 1.0
        x = rng.uniform(150, W - 150) + 60 * math.sin(t * 0.3 + i)
        y = lerp(720, 330, ph)
        add_sprite(F, x, y, 1.5, GOLD, 0.7 * a * math.sin(math.pi * ph) ** 2)


def epilogue(F, t, g, fi):
    wa = (window(t, 0.0, 29.0, 2.0, 3.0) * (1 - 0.7 * smooth(ramp(t, 9.0, 1.4)))
          * (1 + 0.4 * smooth(ramp(t, 20.6, 1.2))))
    _world(F, t, wa)
    n0 = TL.NOWALL - START["epilogue"]
    statement(F, "宇宙里，没有绝对的南墙。", 960, 250, t, n0, 8.8, size=54, cps=6, gold={9, 10})
    subtitle(F, t, 4.6, 8.8, "只有很小的概率，和足够多的尝试。", "Only small probabilities — and enough tries.")
    c0 = TL.CALLBACK - START["epilogue"]
    bw = 64 * 2.45
    xe = statement(F, "不撞南墙不", 960 - bw / 2, 412, t, c0, 20.4, size=64, cps=6.5)
    blank(F, xe + 6, 412, 64, smooth(ramp(t, c0 + 0.9, 0.3)) * (1 - smooth(ramp(t, 19.8, 0.6))), t)
    gauge(F, 960, 528, 520, 0.93, GOLD, "回头的概率", "TURNING BACK", "< 100%", a=window(t, c0 + 1.6, 20.4, 0.8, 0.8))
    subtitle(F, t, 12.2, 19.8, "下一次撞上南墙，也许不必急着回头。",
             "Next time you hit a wall, maybe don't be so quick to turn back.")
    f0 = TL.FINAL - START["epilogue"]
    if t > f0 - 0.3:
        statement(F, "量子隧穿", 960 + 0.25 * 60, 400, t, f0, None, size=60, wght=500, tracking=0.5, cps=30,
                  fade_in=0.9)
        fa = smooth(ramp(t, f0 + 0.6, 0.8))
        hairline(F, 960 - 90 * fa, 440, 960 + 90 * fa, GOLD, 0.6 * fa)
        serif("南墙之外", 20, 500, 0.6).draw(F, 960 + 6, 480, GOLD, fa, align="center")
        caps("QUANTUM TUNNELING   ·   BEYOND THE WALL", 12, 0.45).draw(F, 960, 512, DIM, fa, align="center")


SCENE_FUNCS = {"prologue": prologue, "wall": wall, "seep": seep, "layers": layers, "prison": prison,
               "light": light, "epilogue": epilogue}
