"""Scenes for 量子隧穿.  draw(F, t_local, g_global, frame_index)."""

import math

import numpy as np

import physics as P
from style import (DOTS, GRAPHITE, HATCH, INK, LX, PAPER, R, RULE, T, TINT, VERMILION, Stroke, body,
                   ease_in_out, ease_out, fig_caption, lerp, mixed, mono, part_header, pattern_fill, ramp, rect,
                   rise, sans, sci, serif, smooth, window)
from timeline import RELEASE, START

CX = 1300.0          # figure centre
GROUND = 740.0       # energy-diagram ground line
S1 = 380.0           # px per unit hill height
Y_E = GROUND - P.BALL_E * S1
PXU = 880.0 / 120.0  # px per simulation length unit
X_START = P.X0 * PXU


def ink_dot(F, x, y, r, color, a=1.0):
    s = Stroke(x - r - 3, y - r - 3, x + r + 4, y + r + 4)
    s.circle((x, y), r, th=-1)
    s.blit(F, color, a)


def leader(F, p, q, a, color=INK):
    s = Stroke(min(p[0], q[0]) - 4, min(p[1], q[1]) - 4, max(p[0], q[0]) + 5, max(p[1], q[1]) + 5)
    s.line(p, q)
    s.blit(F, color, 0.8 * a)
    ink_dot(F, p[0], p[1], 2.5, color, a)


# --------------------------------------------------------------- opening ---

WALL = (1236, 1276, 222, 518)
_vol = {}


def _volley():
    if not _vol:
        rng = np.random.default_rng(21)
        n = 220
        _vol.update(t0=np.sort(rng.uniform(0.9, 6.6, n)), y=rng.uniform(236, 504, n),
                    v=rng.uniform(470, 600, n), through=rng.random(n) < 0.19)
    return _vol


def opening(F, t, g, fi):
    V = _volley()
    x0w, x1w, y0w, y1w = WALL
    wh = ease_in_out(ramp(t, 0.4, 0.8))
    cy = (y0w + y1w) / 2
    rect(F, x0w, cy - (cy - y0w) * wh, x1w - x0w, (y1w - y0w) * wh, INK)
    rise(F, mono("BARRIER", 13, medium=True, tracking=0.2), (x0w + x1w) / 2, 204, INK, 1.0, t, 0.9,
         align="center", clip_pad=5)
    rect(F, LX, 370, (x0w - LX - 12) * ease_in_out(ramp(t, 0.3, 1.0)), 1, RULE, 1.0)
    rect(F, x1w + 12, 370, (1760 - x1w - 12) * ease_in_out(ramp(t, 0.5, 1.0)), 1, RULE, 1.0)

    ver = Stroke(LX - 10, 220, 1780, 522)
    gry = Stroke(LX - 10, 220, 1780, 522)
    passed = arrived = 0
    for t0, y, v, th in zip(V["t0"], V["y"], V["v"], V["through"]):
        tau = t - t0
        if tau < 0:
            continue
        hit = (x0w - 5 - LX) / v
        if tau < hit:
            x = LX + v * tau
            ver.line((max(LX, x - 34), y), (x, y), th=1, v=90)
            ver.circle((x, y), 3.4, th=-1)
            continue
        arrived += 1
        if th:
            x = LX + v * tau
            if x > x1w + 4:
                passed += 1
                if x < 1760:
                    fade = 1 - smooth((x - 1660) / 100)
                    ver.line((max(x1w + 4, x - 34), y), (x, y), th=1, v=int(90 * fade))
                    ver.circle((x, y), 3.4, th=-1, v=int(255 * fade))
        else:
            q = tau - hit
            fade = math.exp(-q / 0.7)
            if fade > 0.03:
                x = x0w - 5 - 0.7 * v * q
                gry.circle((x, y), 3.0, th=-1, v=int(255 * fade))
    gry.blit(F, GRAPHITE, 0.8)
    ver.blit(F, VERMILION, 1.0)

    ca = smooth(ramp(t, 1.4, 0.6))
    mono("PASSED THROUGH", 13, tracking=0.2).draw(F, 1760, 580, GRAPHITE, ca, align="right")
    T(f"{passed}", "inter", 84, 800, tracking=-0.02, per_char=False).draw(F, 1700, 670, VERMILION, ca,
                                                                         align="right")
    T(f"/{max(arrived, 0)}", "inter", 28, 600, per_char=False).draw(F, 1706, 670, GRAPHITE, ca)
    ra = smooth(ramp(t, 7.6, 0.8))
    sans("约五分之一的电子，穿过了一堵「实心」的墙", 18, 400, tracking=0.06).draw(F, 1760, 712, GRAPHITE, ra,
                                                                         align="right")
    rise(F, mixed("FIG. 0 — 220 个电子射向同一堵墙", 14, tracking=0.08), LX, 560, GRAPHITE, 1.0, t, 1.2,
         clip_pad=6)

    rise(F, mono("QUANTUM TUNNELING", 18, medium=True, tracking=0.32), LX + 4, 668, VERMILION, 1.0, t, 2.6,
         clip_pad=6)
    rise(F, serif("量子隧穿", 196, 900, tracking=0.0), LX - 8, 858, INK, 1.0, t, 2.0, dur=0.9, stagger=0.09)
    rise(F, sans("穿过一堵，本不该穿过的墙。", 34, 400, tracking=0.08), LX + 4, 934, INK, 1.0, t, 3.4, dur=0.8,
         stagger=0.03)


# ------------------------------------------------------ energy diagram ---

def profile(xr, morph):
    """Barrier height (fraction of S1) at offset xr px: Gaussian hill -> rectangle."""
    hw = P.A / 2 * PXU
    rect_ = 0.5 * (np.tanh((xr + hw) / 1.2) - np.tanh((xr - hw) / 1.2))
    top = (GROUND - 399) / S1
    return (1 - morph) * P.hill(xr) + morph * rect_ * top


def energy_figure(F, t, a, morph, forbidden=0.0, draw_in=1.0, e_label="能量 E"):
    xs = np.linspace(-440, 440, 900)
    hgt = profile(xs, morph)
    ys = GROUND - hgt * S1
    pts = np.column_stack([CX + xs, ys])
    shape = Stroke(840, 160, 1780, 760)
    shape.fill(np.vstack([pts, [[CX + 440, GROUND], [CX - 440, GROUND]]]))
    pattern_fill(F, shape, HATCH, INK, 0.55 * a * draw_in)
    if forbidden > 0:
        fz = Stroke(840, 160, 1780, 760)
        mask = ys < Y_E
        if mask.any():
            fp = np.column_stack([CX + xs[mask], ys[mask]])
            fz.fill(np.vstack([fp, [[fp[-1, 0], Y_E], [fp[0, 0], Y_E]]]))
            pattern_fill(F, fz, DOTS, VERMILION, 0.9 * a * forbidden)
    ln = Stroke(840, 160, 1780, 760)
    n = max(2, int(len(pts) * draw_in))
    ln.poly(pts[:n], th=2)
    ln.line((CX - 440, GROUND), (CX - 440 + 880 * draw_in, GROUND), th=2)
    ln.blit(F, INK, a)
    el = Stroke(840, Y_E - 3, 1780, Y_E + 3)
    el.dashed((CX - 440, Y_E), (CX - 440 + 880 * draw_in, Y_E), 7, 6, th=1)
    el.blit(F, INK, 0.7 * a)
    sans(e_label, 17, 500, tracking=0.12).draw(F, CX + 440, Y_E + 48, INK, a * draw_in, align="right")
    mono("GROUND", 12, tracking=0.2).draw(F, CX + 440, GROUND + 26, GRAPHITE, a * draw_in, align="right")


# ------------------------------------------------------------ part 01 ---

P1 = [
    (2.0, "想象一颗小球，滚向一座小山。", "body", 0),
    (5.0, "它的能量不够翻过山顶——", "body", 1),
    (7.0, "爬到半山腰，停下，又滚了回来。", "body", 1),
    (10.5, "比它能量更高的那片区域，", "body", 2),
    (12.5, "对它来说，是绝对的禁区。", "accent", 2),
]
_ball = {}


def p1(F, t, g, fi):
    A = window(t, 0, 16, 0.01, 0.6)
    part_header(F, t, A, 1, "经典的墙", "The classical wall")
    body(F, t, A, P1)
    di = ease_in_out(ramp(t, 0.3, 1.4))
    fb = smooth(ramp(t, 10.5, 1.0))
    energy_figure(F, t, A, 0.0, forbidden=fb, draw_in=di, e_label="小球的能量 E")
    rise(F, sans("山顶", 17, 500, tracking=0.12), CX + 26, GROUND - S1 - 14, INK, A, t, 1.2, clip_pad=5)
    if fb > 0:
        leader(F, (CX + 40, Y_E - 40), (CX + 170, Y_E - 140), A * fb, VERMILION)
        sans("禁区", 22, 700, tracking=0.2).draw(F, CX + 178, Y_E - 140, VERMILION, A * fb)
        mono("CLASSICALLY FORBIDDEN", 12, tracking=0.16).draw(F, CX + 178, Y_E - 118, VERMILION, A * fb)

    if "f" not in _ball:
        _ball["f"], _ball["turn"] = P.ball_path(X_START, 8.0)
    xo = _ball["f"](t - 2.4)
    br = 16
    hx = float(P.hill(xo))
    slope = -2 * xo / P.HILL_W ** 2 * hx * S1
    nx, ny = -slope / math.hypot(1, slope), -1 / math.hypot(1, slope)
    bx, by = CX + xo + nx * br, GROUND - hx * S1 + ny * br
    ba = A * smooth(ramp(t, 1.0, 0.6))
    ink_dot(F, bx, by, br, INK, ba)
    ang = (xo - X_START) / br
    sp = Stroke(bx - br, by - br, bx + br + 1, by + br + 1)
    sp.line((bx, by), (bx + 0.7 * br * math.cos(ang), by + 0.7 * br * math.sin(ang)), th=2)
    sp.blit(F, PAPER, ba)
    if t > 6.4:
        tx = CX + _ball["turn"]
        ty = GROUND - float(P.hill(_ball["turn"])) * S1
        k = smooth(ramp(t, 6.4, 0.6)) * A
        leader(F, (tx, ty - 34), (tx - 90, ty - 120), k, VERMILION)
        sans("折返点", 17, 500, tracking=0.12).draw(F, tx - 96, ty - 130, VERMILION, k, align="right")
    fig_caption(F, t, A, "FIG. 1 — 经典粒子：能量不够，就过不去")


# ------------------------------------------------------- wave packet ---

def sim_s(g):
    if g < RELEASE:
        return 0.0
    u = g - RELEASE
    if u <= 13.6:
        return 5.0 * u
    return 68.0 + 12.0 * (1 - math.exp(-(u - 13.6) * 5.0 / 12.0))


def draw_packet(F, g, a, glow_barrier=0.0):
    D = P.packet()
    s = sim_s(g)
    f = min(s / P.S_END * (P.FRAMES - 1), P.FRAMES - 1.0001)
    i0 = int(f)
    fr = f - i0
    re = D["re"][i0] * (1 - fr) + D["re"][i0 + 1] * fr
    im = D["im"][i0] * (1 - fr) + D["im"][i0 + 1] * fr
    hold = max(0.0, (RELEASE - g)) + max(0.0, g - (RELEASE + 16))
    ph = -2 * math.pi * 0.55 * hold
    if ph:
        c, sn = math.cos(ph), math.sin(ph)
        re, im = re * c - im * sn, re * sn + im * c
    prob = re ** 2 + im ** 2
    X = CX + P.XS * PXU
    p0 = 1 / (math.sqrt(2 * math.pi) * P.SIGMA)
    hgt = np.minimum(prob / p0 * 165, 250)
    pts = np.column_stack([X, Y_E - hgt])
    fill = Stroke(840, 160, 1780, 760)
    fill.fill(np.vstack([pts, [[X[-1], Y_E], [X[0], Y_E]]]))
    pattern_fill(F, fill, DOTS, VERMILION, 0.85 * a)
    r0 = (2 * math.pi * P.SIGMA ** 2) ** -0.25
    rl = Stroke(840, 160, 1780, 760)
    rl.poly(np.column_stack([X, Y_E - np.clip(re / r0 * 80, -200, 200)]), th=1)
    rl.blit(F, INK, 0.45 * a)
    ln = Stroke(840, 160, 1780, 760)
    ln.poly(pts, th=3)
    ln.blit(F, VERMILION, a)
    return X, hgt


P2 = [
    (2.0, "可在微观世界，电子不是小球。", "body", 0),
    (5.0, "量子力学用「波函数」描述它——", "body", 1),
    (8.0, "波在哪里高，", "body", 2),
    (9.8, "哪里就更可能找到电子。", "body", 2),
    (13.0, "现在，让这团波撞向墙。", "accent", 3),
]


def p2(F, t, g, fi):
    A = window(t, 0, 16, 0.5, 0.6)
    part_header(F, t, A, 2, "电子是一团波", "The electron is a wave")
    body(F, t, A, P2)
    m = ease_in_out(ramp(t, 0.6, 1.8))
    energy_figure(F, t, A, m, forbidden=(1 - smooth(ramp(t, 0.2, 0.8))), e_label="电子的能量 E")
    rise(F, sans("势垒", 17, 500, tracking=0.12), CX + 20, 386, INK, A * smooth(ramp(t, 1.6, 0.5)), t, 1.6,
         clip_pad=5)
    ba = A * (1 - smooth(ramp(t, 1.2, 0.9)))
    bx = CX + X_START
    ink_dot(F, bx, Y_E - 16, 16, INK, ba)
    if 1.2 < t < 3.2:
        q = (t - 1.2) / 2.0
        ring = Stroke(bx - 130, Y_E - 150, bx + 130, Y_E + 110)
        ring.circle((bx, Y_E - 16), 16 + 110 * ease_out(q), th=1)
        ring.blit(F, INK, 0.7 * (1 - q) * A)
    pa = A * smooth(ramp(t, 1.8, 1.2))
    X, hgt = draw_packet(F, g, pa)
    k = A * smooth(ramp(t, 8.0, 0.6))
    if k > 0:
        px = CX + X_START
        leader(F, (px + 30, Y_E - 150), (px + 120, Y_E - 230), k, VERMILION)
        R([("|", "stix", 26, None), ("ψ", "stix_it", 26, None), ("|", "stix", 26, None),
           ("²", "stix", 26, None)]).draw(F, px + 128, Y_E - 236, VERMILION, k)
        sans("找到电子的概率", 16, 400, tracking=0.12).draw(F, px + 128, Y_E - 208, VERMILION, k)
        leader(F, (px - 50, Y_E + 30), (px - 120, Y_E + 120), k * 0.9, INK)
        R([("Re ", "stix", 20, None), ("ψ", "stix_it", 20, None)]).draw(F, px - 126, Y_E + 140, INK, k * 0.9,
                                                                       align="right")
    fig_caption(F, t, A, "FIG. 2 — 小球化作波包；势垒理想化为一堵薄墙")


# ------------------------------------------------------------ part 03 ---

P3 = [
    (1.5, "撞上墙的一瞬，波并没有消失：", "body", 0),
    (4.5, "它在墙里急剧衰减，", "body", 1),
    (6.5, "却不会立刻降到零。", "body", 1),
    (10.0, "墙够薄，另一侧就还剩一点波——", "body", 2),
    (13.5, "这一点，就是电子「穿墙而过」的概率。", "accent", 3),
]


def p3(F, t, g, fi):
    A = window(t, 0, 20, 0.01, 0.01)
    part_header(F, t, A, 3, "渗进墙里", "Leaking through")
    body(F, t, A, P3)
    energy_figure(F, t, A, 1.0, e_label="电子的能量 E")
    sans("势垒", 17, 500, tracking=0.12).draw(F, CX + 20, 386, INK, A)
    draw_packet(F, g, A)
    k = A * window(t, 3.8, 9.6, 0.6, 0.8)
    if k > 0:
        leader(F, (CX + 2, Y_E - 6), (CX + 90, 250), k, INK)
        sans("墙内：指数衰减", 18, 500, tracking=0.1).draw(F, CX + 98, 246, INK, k)
        R([("ψ", "stix_it", 22, None), (" ∝ ", "math", 20, None), ("e", "stix_it", 22, None)]).draw(
            F, CX + 98, 280, GRAPHITE, k)
        R([("−κx", "stix_it", 15, None)]).draw(F, CX + 98 + 62, 268, GRAPHITE, k)
    D = P.packet()
    pa = A * smooth(ramp(t, 8.6, 0.8))
    if pa > 0:
        xr = CX - 36 * PXU
        xt = CX + 36 * PXU
        sans("反射", 18, 500, tracking=0.2).draw(F, xr, 214, GRAPHITE, pa, align="center")
        T(f"{round(float(D['R']) * 100)}%", "inter", 60, 800, per_char=False).draw(F, xr, 276, GRAPHITE, pa,
                                                                                    align="center")
        sans("穿过", 18, 500, tracking=0.2).draw(F, xt, 214, VERMILION, pa, align="center")
        T(f"{round(float(D['T']) * 100)}%", "inter", 60, 800, per_char=False).draw(F, xt, 276, VERMILION, pa,
                                                                                    align="center")
    fig_caption(F, t, A, "FIG. 3 — 薛定谔方程数值模拟：一个波包撞上势垒", t0=0.4)


# ------------------------------------------------------------ part 04 ---

P4 = [
    (2.0, "隧穿的概率，对墙的厚度极度敏感。", "body", 0),
    (3.8, "电子能量 1 eV · 势垒高 2 eV", "note", 0),
    (7.0, "墙厚 1 纳米：约万分之一能穿过；", "body", 1),
    (12.0, "墙厚 2 纳米：只剩约两亿分之一。", "body", 1),
    (14.6, "厚度只翻了一倍，概率却缩小近三万倍。", "body", 2),
    (17.0, "所以，别指望自己能穿过墙。", "accent", 3),
]
NM0, NM1 = -2.6, 4.6
PXN = 880.0 / (NM1 - NM0)
YC = 420.0


def a_of(t):
    a = 0.5
    a = lerp(a, 1.0, ease_in_out(ramp(t, 4.5, 2.5)))
    a = lerp(a, 2.0, ease_in_out(ramp(t, 9.5, 2.5)))
    return a


def nm_x(x):
    return 860 + (x - NM0) * PXN


def p4(F, t, g, fi):
    A = window(t, 0, 20, 0.01, 0.01)
    part_header(F, t, A, 4, "一纳米的差距", "One nanometre")
    body(F, t, A, P4)
    a = a_of(t)
    xa0, xa1 = nm_x(0.0), nm_x(a)
    da = ease_in_out(ramp(t, 0.3, 1.0))
    bar = Stroke(xa0 - 2, 196, xa1 + 3, 646)
    bar.fill([(xa0, 200), (xa1, 200), (xa1, 640), (xa0, 640)])
    pattern_fill(F, bar, HATCH, INK, 0.5 * A * da)
    ed = Stroke(xa0 - 2, 196, xa1 + 3, 646)
    ed.poly([(xa0, 640), (xa0, 200), (xa1, 200), (xa1, 640)], th=2)
    ed.blit(F, INK, A * da)
    rect(F, 860, YC, 880 * da, 1, GRAPHITE, 0.6 * A)
    dim = Stroke(xa0 - 6, 168, xa1 + 7, 186)
    dim.line((xa0, 178), (xa1, 178))
    dim.line((xa0, 171), (xa0, 185))
    dim.line((xa1, 171), (xa1, 185))
    dim.blit(F, INK, A * da)
    mono(f"a = {a:.2f} nm", 15, medium=True, tracking=0.06).draw(F, (xa0 + xa1) / 2, 160, INK, A * da,
                                                                align="center")

    xs = np.linspace(NM0, NM1, 1400)
    psi, Tr = P.stationary(a, xs)
    w = 2 * math.pi * 0.5
    wave = (psi * np.exp(-1j * w * t)).real
    amp_t = math.sqrt(Tr)
    M = max(1.0, 0.85 / amp_t)
    right = xs > a
    yv = wave * 80.0
    yv[right] *= M
    X = nm_x(xs)
    wa = A * smooth(ramp(t, 0.8, 1.0))
    if M > 1.5:
        pa = wa * smooth(ramp(M, 1.5, 3))
        rect(F, xa1 + 2, 200, 1740 - xa1 - 2, 440, TINT, 0.55 * pa)
        sans("放大", 16, 500, tracking=0.2).draw(F, 1734, 232, VERMILION, pa, align="right")
        mr = float(f"{M:.2g}")
        T(f"×{mr:,.0f}", "inter", 30, 800, per_char=False).draw(F, 1734, 270, VERMILION, pa, align="right")
    ln = Stroke(850, 160, 1750, 680)
    left = ~right
    ln.poly(np.column_stack([X[left], YC - yv[left]]), th=3)
    ln.poly(np.column_stack([X[right], YC - yv[right]]), th=3)
    ln.blit(F, VERMILION, wa)
    inside = (xs >= 0) & (xs <= a)
    env = np.abs(psi[inside]) * 80.0
    ev = Stroke(850, 160, 1750, 680)
    ev.poly(np.column_stack([X[inside], YC - env]), th=1)
    ev.blit(F, INK, 0.7 * wa)
    sans("入射", 16, 500, tracking=0.2).draw(F, 868, 232, INK, wa)

    ra = A * smooth(ramp(t, 1.2, 0.8))
    sans("透射概率 T", 18, 500, tracking=0.16).draw(F, 860, 712, INK, ra)
    sci(F, Tr, 860, 806, 76, VERMILION, ra)
    mono("ELECTRON · E = 1 eV · V = 2 eV", 13, tracking=0.12).draw(F, 862, 846, GRAPHITE, ra)

    cx0, cx1, cy0, cy1 = 1340.0, 1740.0, 700.0, 880.0
    a_lo, a_hi = 0.3, 2.1

    def cxy(aa, TT):
        return (cx0 + (aa - a_lo) / (a_hi - a_lo) * (cx1 - cx0), cy0 + (-math.log10(TT)) / 9.0 * (cy1 - cy0))
    ch = Stroke(cx0 - 6, cy0 - 6, cx1 + 8, cy1 + 8)
    ch.line((cx0, cy0), (cx0, cy1))
    ch.line((cx0, cy1), (cx1, cy1))
    for lv in (0, 3, 6, 9):
        y = cy0 + lv / 9 * (cy1 - cy0)
        ch.line((cx0 - 5, y), (cx0, y))
    ch.blit(F, INK, 0.6 * ra)
    for lv, lab in ((0, "1"), (3, "10⁻³"), (6, "10⁻⁶"), (9, "10⁻⁹")):
        y = cy0 + lv / 9 * (cy1 - cy0)
        R([(lab, "stix", 15, None)]).draw(F, cx0 - 10, y + 5, GRAPHITE, ra, align="right")
    for av in (0.5, 1.0, 1.5, 2.0):
        mono(f"{av:.1f}", 12).draw(F, cxy(av, 1)[0], cy1 + 20, GRAPHITE, ra, align="center")
    mono("a / nm", 12, tracking=0.1).draw(F, cx1, cy1 + 38, GRAPHITE, ra, align="right")
    mono("log T", 12, tracking=0.1).draw(F, cx0 + 8, cy0 + 4, GRAPHITE, ra)
    aa = np.linspace(a_lo, a_hi, 200)
    cv = Stroke(cx0 - 6, cy0 - 6, cx1 + 8, cy1 + 8)
    cv.poly([cxy(v, P.transmission(v)) for v in aa], th=2)
    cv.blit(F, INK, ra)
    px, py = cxy(a, Tr)
    ink_dot(F, px, py, 6, VERMILION, ra)
    fig_caption(F, t, A, "FIG. 4 — 定态解：波在墙内指数衰减，墙右侧（已放大）即透射部分", t0=0.6)


# ------------------------------------------------------------ part 05 ---

P5 = [
    (1.0, "隧穿并不只活在课本里。", "body", 0),
    (4.0, "它让太阳得以燃烧，", "body", 1),
    (8.0, "让原子核发生衰变，", "body", 2),
    (12.0, "让我们看见单个原子，", "body", 3),
    (16.0, "也藏在你的手机里。", "body", 4),
    (20.5, "——它一直都在。", "accent", 5),
]
TILES = [
    ("太阳核聚变", "核心约 1500 万度，质子仍难越过电斥力，", "靠隧穿才得以聚变、发光。"),
    ("α 衰变", "α 粒子从原子核的势阱中「漏」了出来，", "1928 年由伽莫夫用隧穿解释。"),
    ("扫描隧道显微镜", "隧穿电流随距离剧变，借此「看见」原子，", "获 1986 年诺贝尔物理学奖。"),
    ("闪存", "电子隧穿过极薄的氧化层，", "在浮栅中写入、擦除数据。"),
]
TILE_T = [4.0, 8.0, 12.0, 16.0]


def _tile_sun(F, x0, y0, t, a):
    """Two protons meet a Coulomb barrier (1/r) around a nuclear well and tunnel in."""
    cx, base = x0 + 190, y0 + 226
    rn, e_lvl = 0.18, 0.4
    u = np.linspace(-1, 1, 500)
    v = np.where(np.abs(u) < rn, -0.35, rn / np.maximum(np.abs(u), 1e-3))
    s = Stroke(x0, y0 + 60, x0 + 421, y0 + 270)
    s.poly(np.column_stack([cx + u * 180, base - v * 120]), th=2)
    s.line((x0, base), (x0 + 380, base))
    s.blit(F, INK, a)
    ey = base - e_lvl * 120
    el = Stroke(x0, ey - 4, x0 + 381, ey + 4)
    el.dashed((x0, ey), (x0 + 380, ey), 5, 5)
    el.blit(F, GRAPHITE, 0.7 * a)
    sun = Stroke(x0 + 352, y0 + 70, x0 + 421, y0 + 140)
    sun.circle((x0 + 386, y0 + 104), 30, th=-1)
    pattern_fill(F, sun, DOTS, VERMILION, a)
    ph = (t % 4.0) / 4.0
    q = ease_in_out(min(ph / 0.6, 1.0))
    d = lerp(180, 10, q)
    for x in (cx - d, cx + d):
        in_wall = rn * 180 < abs(x - cx) < rn / e_lvl * 180
        ink_dot(F, x, ey, 9, VERMILION, a * (0.4 if in_wall else 1.0))
        sans("+", 14, 700).draw(F, x, ey + 5, PAPER, a * (0.4 if in_wall else 1.0), align="center")
    if ph > 0.6:
        k = (ph - 0.6) / 0.4
        ring = Stroke(cx - 90, ey - 90, cx + 90, ey + 90)
        ring.circle((cx, ey), 14 + 66 * ease_out(k), th=2)
        ring.blit(F, VERMILION, a * (1 - k))


def _tile_alpha(F, x0, y0, t, a):
    cx, base = x0 + 170, y0 + 214
    u = np.linspace(-1.5, 2.6, 500)
    rw = 0.35
    y = np.where(np.abs(u) < rw, base + 22, base - np.minimum(120, 40 / np.maximum(np.abs(u) - rw + 0.33, 1e-3)))
    pts = np.column_stack([cx + u * 95, y])
    s = Stroke(x0, y0 + 60, x0 + 421, y0 + 262)
    s.poly(pts[(pts[:, 0] >= x0) & (pts[:, 0] <= x0 + 420)], th=2)
    s.blit(F, INK, a)
    el = Stroke(x0, base - 34, x0 + 421, base - 26)
    el.dashed((x0, base - 30), (x0 + 420, base - 30), 5, 5)
    el.blit(F, GRAPHITE, 0.7 * a)
    ph = t % 5.0
    if ph < 3.0:
        x = cx + rw * 95 * 0.82 * math.sin(2 * math.pi * ph * 1.3)
        aa = 1.0
    else:
        q = (ph - 3.0) / 2.0
        x = cx + rw * 95 + ease_out(q) * 230
        aa = 0.4 if x < cx + (rw + 1.0) * 95 else 1.0 - smooth((q - 0.8) / 0.2)
    ink_dot(F, x, base - 30, 10, VERMILION, a * aa)
    R([("α", "stix_it", 15, None)]).draw(F, x, base - 25, PAPER, a * aa, align="center")
    sans("原子核势阱", 13, 400, tracking=0.12).draw(F, cx, base + 48, GRAPHITE, a, align="center")


def _tile_stm(F, x0, y0, t, a):
    base = y0 + 238
    xs_at = [x0 + 40 + i * 49 for i in range(8)]
    s = Stroke(x0, y0 + 60, x0 + 421, y0 + 262)
    for xa in xs_at:
        s.circle((xa, base - 18), 18, th=2)
    s.line((x0, base + 1), (x0 + 420, base + 1), th=2)
    s.blit(F, INK, a)
    ph = (t % 5.0) / 5.0
    xt = lerp(x0 + 20, x0 + 400, ph)
    tip_y = base - 70
    tip = Stroke(xt - 18, tip_y - 50, xt + 19, tip_y + 2)
    tip.fill([(xt - 14, tip_y - 46), (xt + 14, tip_y - 46), (xt, tip_y)])
    tip.blit(F, INK, a)

    def surf(x):
        d = np.min([np.abs(x - xa) for xa in xs_at], axis=0)
        return base - 18 - np.sqrt(np.clip(18 ** 2 - d ** 2, 0, None))
    xx = np.linspace(x0 + 20, xt, max(2, int((xt - x0 - 20) / 2)))
    cur = np.exp(-(surf(xx) - tip_y - 34) / 6.0)
    tr = Stroke(x0, y0 + 60, x0 + 421, y0 + 120)
    tr.line((x0 + 20, y0 + 114), (x0 + 400, y0 + 114), th=1, v=110)
    tr.poly(np.column_stack([xx, y0 + 112 - cur * 44]), th=2)
    tr.blit(F, VERMILION, a)
    gap = float(surf(np.array([xt]))[0] - tip_y)
    rate = math.exp(-(gap - 34) / 6.0)
    for k in range(4):
        q = (t * 2.2 + k / 4) % 1.0
        if (k / 4) < rate + 0.15:
            ink_dot(F, xt + (k - 1.5) * 3, lerp(tip_y + gap - 4, tip_y + 2, q), 2.6, VERMILION, a)


def _tile_flash(F, x0, y0, t, a):
    """Floating-gate cell: electrons tunnel through a thin oxide to write, back to erase."""
    lx0, lx1 = x0 + 10, x0 + 280
    cg, fg, ox, ch = (y0 + 78, y0 + 102), (y0 + 122, y0 + 156), (y0 + 156, y0 + 176), (y0 + 176, y0 + 214)
    st = Stroke(lx0 - 2, cg[0] - 2, lx1 + 3, cg[1] + 3)
    st.fill([(lx0, cg[0]), (lx1, cg[0]), (lx1, cg[1]), (lx0, cg[1])])
    st.blit(F, INK, a)
    for (ya, yb), pat in ((fg, HATCH), (ch, DOTS)):
        st = Stroke(lx0 - 2, ya - 2, lx1 + 3, yb + 3)
        st.fill([(lx0, ya), (lx1, ya), (lx1, yb), (lx0, yb)])
        pattern_fill(F, st, pat, INK, 0.55 * a)
        st.m[:] = 0
        st.poly([(lx0, ya), (lx1, ya), (lx1, yb), (lx0, yb)], th=1, closed=True)
        st.blit(F, INK, a)
    rect(F, lx0, ox[0], lx1 - lx0, ox[1] - ox[0], TINT, 0.9 * a)
    for name, (ya, yb), col in (("CONTROL GATE", cg, GRAPHITE), ("FLOATING GATE", fg, GRAPHITE),
                                ("TUNNEL OXIDE", ox, VERMILION), ("CHANNEL", ch, GRAPHITE)):
        mono(name, 11, tracking=0.12).draw(F, lx1 + 12, (ya + yb) / 2 + 4, col, a)
    ph = t % 6.0
    program = ph < 3.0
    q = ph / 3.0 if program else (ph - 3.0) / 3.0
    n_stored = int(round((q if program else 1 - q) * 8))
    for i in range(n_stored):
        ink_dot(F, lx0 + 20 + i * 32, (fg[0] + fg[1]) / 2, 4.2, VERMILION, a)
    for k in range(5):
        u = (t * 0.35 + k / 5) % 1.0
        ink_dot(F, lerp(lx0 + 8, lx1 - 8, u), (ch[0] + ch[1]) / 2, 3.4, VERMILION, 0.85 * a)
    kk = (ph * 3) % 1.0
    xj = lx0 + 30 + ((int(ph * 3) * 53) % 220)
    yj = lerp(ch[0] + 14, fg[0] + 18, kk) if program else lerp(fg[0] + 18, ch[0] + 14, kk)
    ink_dot(F, xj, yj, 3.8, VERMILION, a)
    sans("写入" if program else "擦除", 16, 700, tracking=0.2).draw(F, x0 + 420, y0 + 70, VERMILION, a,
                                                                 align="right")


TILE_DRAW = [_tile_sun, _tile_alpha, _tile_stm, _tile_flash]


def p5(F, t, g, fi):
    A = window(t, 0, 28, 0.01, 0.01)
    part_header(F, t, A, 5, "它一直都在", "Hiding in plain sight")
    body(F, t, A, P5)
    for i, (title, d1, d2) in enumerate(TILES):
        x0 = 840 + (i % 2) * 480
        y0 = 170 + (i // 2) * 390
        ti = TILE_T[i]
        a = A * smooth(ramp(t, ti, 0.6))
        if a <= 0:
            continue
        active = (ti <= t < (TILE_T[i + 1] if i + 1 < 4 else 20.5))
        ru = ease_in_out(ramp(t, ti, 0.8))
        rect(F, x0, y0, 420 * ru, 2, VERMILION if active else INK, A)
        T(f"{i + 1:02d}", "inter", 26, 800, per_char=False).draw(F, x0, y0 + 42, VERMILION, a)
        rise(F, serif(title, 30, 900, tracking=0.04), x0 + 46, y0 + 42, INK, A, t, ti + 0.1, dur=0.6,
             stagger=0.04)
        TILE_DRAW[i](F, x0, y0, t - ti, a)
        rise(F, sans(d1, 18, 400, tracking=0.04), x0, y0 + 296, GRAPHITE, A, t, ti + 0.4, dur=0.6, clip_pad=7)
        rise(F, sans(d2, 18, 400, tracking=0.04), x0, y0 + 324, GRAPHITE, A, t, ti + 0.5, dur=0.6, clip_pad=7)


# -------------------------------------------------------------- outro ---

def outro(F, t, g, fi):
    a = 1 - smooth(ramp(t, 8.2, 0.7))
    rise(F, sans("在量子世界里，", 42, 400, tracking=0.08), LX, 330, INK, a, t, 0.4, dur=0.8, stagger=0.03)
    big = serif("不可能", 190, 900, tracking=0.02)
    rise(F, big, LX - 8, 580, INK, a, t, 1.3, dur=0.9, stagger=0.1)
    sw = ease_in_out(ramp(t, 4.0, 0.45))
    rect(F, LX - 20, 580 - 78, (big.width + 40) * sw, 16, VERMILION, a)
    rise(F, serif("只是「概率很小」。", 96, 900, tracking=0.02), LX - 4, 760, VERMILION, a, t, 4.6, dur=0.8,
         stagger=0.05)
    rise(F, mono("IMPOSSIBLE → IMPROBABLE", 16, medium=True, tracking=0.24), LX, 830, GRAPHITE, a, t, 5.6,
         clip_pad=6)

    e = smooth(ramp(t, 8.9, 0.8))
    if e > 0:
        rise(F, serif("量子隧穿", 72, 900, tracking=0.12), 960 + 4, 540, INK, e, t, 8.9, dur=0.8, stagger=0.07,
             align="center")
        rect(F, 960 - 24, 572, 48 * ease_in_out(ramp(t, 9.4, 0.5)), 5, VERMILION, e)
        mono("QUANTUM TUNNELING — FIN", 15, medium=True, tracking=0.3).draw(F, 960, 620, GRAPHITE,
                                                                          e * smooth(ramp(t, 9.6, 0.6)),
                                                                          align="center")


SCENE_FUNCS = {"opening": opening, "p1": p1, "p2": p2, "p3": p3, "p4": p4, "p5": p5, "outro": outro}
PART_NUM = {"p1": 1, "p2": 2, "p3": 3, "p4": 4, "p5": 5}
