"""The ten scenes.  Each draw function receives the float frame F, the local
scene time t and the global time g."""

import math

import numpy as np

import physics as P
from engine import (BLUE, CYAN, GOLD, IVORY, MUTED, VIOLET, WHITE, W, H, R, T, Stroke, add_light,
                    add_rgb, add_sprite, blend, blur, ease_in, ease_in_out, ease_out, ease_out_expo,
                    lerp, mix, ramp, smooth, window)
from timeline import START

LX = 160          # left text column
BODY_Y0 = 436
BODY_DY = 62


# ---------------------------------------------------------------- shared ---

def hline(F, x, y, w, color, a, h=1):
    if w >= 1 and a > 0.002:
        blend(F, np.full((h, int(w)), 255, np.uint8), x, y, color, a)


def header(F, t, a, num, zh, en):
    T(f"{num:02d}", "jost", 220, 200).draw(F, 792, 300, IVORY, a * 0.05, t=t, t0=0.2, mode="fade",
                                           dur=1.8, rise=0, align="right")
    T(f"CHAPTER  {num:02d}", "jost", 18, 500, tracking=0.42).draw(F, LX, 150, GOLD, a, t=t, t0=0.25,
                                                                  stagger=0.02, dur=0.7, rise=8)
    T(zh, "serif", 62, 600, tracking=0.08).draw(F, LX - 2, 234, IVORY, a, t=t, t0=0.45, stagger=0.08,
                                                dur=0.9, rise=18)
    T(en, "corm_it", 34, 500, per_char=False).draw(F, LX, 288, MUTED, a, t=t, t0=0.95, mode="wipe", dur=1.3)
    hline(F, LX, 326, 96 * ease_in_out(ramp(t, 1.1, 1.0)), GOLD, 0.85 * a)


def body(F, t, a, lines):
    """lines: (t_in, text, style, group).  Earlier groups dim as new ones arrive."""
    gstart = {}
    for t_in, _, _, gidx in lines:
        gstart[gidx] = min(gstart.get(gidx, 1e9), t_in)
    y = BODY_Y0
    for t_in, s, style, gidx in lines:
        later = [v for k, v in gstart.items() if k > gidx]
        d = 1 - 0.6 * smooth(ramp(t, min(later), 0.9)) if later else 1.0
        if style == "note":
            y += 14
            T(s, "sans", 22, 300, tracking=0.06).draw(F, LX, y, MUTED, a * d, t=t, t0=t_in, stagger=0.02,
                                                     dur=0.7, rise=8)
            y += 44
            continue
        color = GOLD if style == "gold" else IVORY
        T(s, "serif", 33, 500 if style == "gold" else 400, tracking=0.04).draw(
            F, LX, y, color, a * d, t=t, t0=t_in, stagger=0.03, dur=0.75, rise=10)
        y += BODY_DY


def label(F, s, x, y, a, color=MUTED, size=18, align="center", t=None, t0=0.0):
    T(s, "sans", size, 300, tracking=0.16).draw(F, x, y, color, a, t=t, t0=t0, mode="fade", dur=0.8,
                                                rise=6, align=align)


class Dust:
    def __init__(self, n=150, seed=3):
        rng = np.random.default_rng(seed)
        self.x0 = rng.uniform(0, W, n)
        self.y0 = rng.uniform(0, H, n)
        self.vx = rng.normal(0, 4, n)
        self.vy = rng.normal(-3, 3, n)
        self.r = rng.uniform(0.6, 1.7, n)
        self.a = rng.uniform(0.03, 0.2, n) * (self.r / 1.7) ** 0.5
        self.f = rng.uniform(0.1, 0.5, n)
        self.p = rng.uniform(0, 6.28, n)

    def draw(self, F, g, a):
        x = (self.x0 + self.vx * g) % (W + 40) - 20
        y = (self.y0 + self.vy * g) % (H + 40) - 20
        tw = 0.55 + 0.45 * np.sin(self.f * g * 6.28 + self.p)
        for xi, yi, ri, ai in zip(x, y, self.r, self.a * tw * a):
            add_sprite(F, xi, yi, ri, IVORY, ai)


DUST = Dust()


def hud(F, g, a, num, a_num):
    if a <= 0.002:
        return
    T("QUANTUM MECHANICS", "jost", 14, 400, tracking=0.42).draw(F, LX, 994, MUTED, 0.8 * a)
    T("量子力学", "sans", 14, 300, tracking=0.3).draw(F, LX + 272, 995, MUTED, 0.8 * a)
    hline(F, LX, 1014, 1600, IVORY, 0.09 * a)
    prog = (g - START["ch1"]) / (START["outro"] - START["ch1"])
    hline(F, LX, 1014, 1600 * min(max(prog, 0), 1), GOLD, 0.55 * a)
    if num:
        T(f"{num:02d}  /  08", "jost", 14, 400, tracking=0.3).draw(F, 1760, 994, MUTED, 0.8 * a_num,
                                                                  align="right")


# ------------------------------------------------------- orbital clouds ---

def project(pts, cx, cy, scale, yaw, tilt):
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    ca, sa = math.cos(yaw), math.sin(yaw)
    xs = x * ca - y * sa
    depth = x * sa + y * ca
    ct, st = math.cos(tilt), math.sin(tilt)
    ys = -(z * ct - depth * st)
    return cx + xs * scale, cy + ys * scale


def splat_cloud(F, sx, sy, gain, core=CYAN, halo=VIOLET, weights=None, bounds=None, halo_gain=1.0):
    if gain <= 0.002:
        return
    if bounds is None:
        x0, x1 = int(max(np.min(sx) - 60, 0)), int(min(np.max(sx) + 60, W))
        y0, y1 = int(max(np.min(sy) - 60, 0)), int(min(np.max(sy) + 60, H))
    else:
        x0, y0, x1, y1 = bounds
    w, h = x1 - x0, y1 - y0
    if w <= 2 or h <= 2:
        return
    ix = (sx - x0).astype(np.int32)
    iy = (sy - y0).astype(np.int32)
    ok = (ix >= 0) & (ix < w) & (iy >= 0) & (iy < h)
    idx = iy[ok] * w + ix[ok]
    wts = None if weights is None else weights[ok]
    dens = np.bincount(idx, weights=wts, minlength=w * h).astype(np.float32).reshape(h, w)
    c = blur(dens, 0.75)
    c = 1 - np.exp(-c * 2.4)
    hl = blur(dens, 16.0)
    hl2 = blur(dens, 4.0)
    img = (c[..., None] * mix(core, WHITE, 0.35)[None, None] * 0.75
           + np.minimum(hl2 * 0.6, 1.0)[..., None] * core[None, None] * 0.28
           + np.minimum(hl * 5.0, 1.3)[..., None] * halo[None, None] * 0.2 * halo_gain)
    add_rgb(F, img, x0, y0, gain)


def cloud_subset(n, fi, frac=0.7):
    rng = np.random.default_rng(1000 + fi)
    return rng.random(n) < frac


# --------------------------------------------------------------- opening ---

_lat = None


def _lattice():
    global _lat
    if _lat is None:
        a = 58.0
        pts = []
        for j in range(-26, 27):
            for i in range(-30, 31):
                pts.append(((i + 0.5 * (j % 2)) * a, j * a * math.sqrt(3) / 2))
        _lat = np.array(pts, np.float32)
    return _lat


def opening(F, t, g, fi):
    cx, cy = 960.0, 470.0
    s = math.exp(3.65 * ease_in_out(t / 8.8))
    rot = 0.08 + 0.025 * t
    la = window(t, 0.2, 10.2, 1.8, 2.6)
    if la > 0:
        L = _lattice()
        c, sn = math.cos(rot), math.sin(rot)
        px = cx + (L[:, 0] * c - L[:, 1] * sn) * s
        py = cy + (L[:, 0] * sn + L[:, 1] * c) * s
        rc = 1.0 * s ** 0.78
        rg = 4.2 * s ** 0.9
        m = (px > -3 * rg) & (px < W + 3 * rg) & (py > -3 * rg) & (py < H + 3 * rg)
        d2 = ((px[m] - cx) / 900) ** 2 + ((py[m] - cy) / 620) ** 2
        fall = np.exp(-d2 * 1.6)
        center_fade = 1 - smooth(ramp(t, 6.2, 2.6))
        for x, y, f in zip(px[m], py[m], fall):
            add_sprite(F, x, y, rc, IVORY, 0.5 * f * la)
            add_sprite(F, x, y, rg, CYAN, 0.16 * f * la * (0.4 + 0.6 * center_fade))
    # the atom at the centre opens into an electron cloud
    ca = smooth(ramp(t, 5.4, 3.0)) * window(t, 0, 15.9, 0.1, 1.8)
    if ca > 0:
        pts = P.orbitals()["f_z3"]
        sub = cloud_subset(len(pts), fi, 0.75)
        sc = lerp(2.0, 8.6, ease_out(ramp(t, 5.4, 4.5)))
        sx, sy = project(pts[sub], cx, cy, sc, 0.5 + 0.16 * t, 0.42)
        splat_cloud(F, sx, sy, 0.42 * ca, core=CYAN, halo=VIOLET)

    ta = window(t, 0.8, 4.4, 0.1, 0.9)
    T("把世界不断放大——", "serif", 38, 400, tracking=0.16).draw(F, 960, 905, IVORY, ta, t=t, t0=0.8,
                                                           stagger=0.07, dur=1.0, rise=10, align="center")
    tb = window(t, 4.7, 7.7, 0.1, 0.8)
    T("在原子之下，规则开始变得陌生。", "serif", 38, 400, tracking=0.16).draw(
        F, 960, 905, IVORY, tb, t=t, t0=4.7, stagger=0.05, dur=1.0, rise=10, align="center")

    ha = window(t, 7.9, 15.8, 0.1, 1.5)
    if ha > 0:
        T("量子力学", "serif", 132, 500, tracking=0.34).draw(
            F, 960 + 0.17 * 132, 560, IVORY, ha, t=t, t0=8.0, stagger=0.13, dur=1.5, rise=22, align="center",
            glow=0.22, glow_color=GOLD, glow_sigma=22)
        T("QUANTUM  MECHANICS", "jost", 22, 400, tracking=0.62).draw(
            F, 960 + 0.31 * 22, 638, GOLD, ha, t=t, t0=8.9, stagger=0.025, dur=0.9, rise=8, align="center")
        w = 280 * ease_in_out(ramp(t, 9.3, 1.3))
        hline(F, 960 - w / 2, 678, w, GOLD, 0.6 * ha)
        T("从一份能量，到整个现代世界", "sans", 26, 300, tracking=0.3).draw(
            F, 960 + 0.15 * 26, 730, MUTED, ha, t=t, t0=9.9, stagger=0.04, dur=1.0, rise=8, align="center")


# ----------------------------------------------------- 01 quantisation ---

CH1 = [
    (2.0, "1900 年，普朗克提出一个大胆的假设：", "body", 0),
    (5.2, "能量并不是连续流淌的，", "body", 1),
    (7.2, "而是一份一份地被吸收与释放。", "body", 1),
    (11.0, "这不可再分的最小一份，", "body", 2),
    (12.8, "被称作「量子」。", "gold", 2),
]


def ch1(F, t, g, fi):
    A = window(t, 0, 20, 0.9, 0.9)
    header(F, t, A, 1, "量子化", "Quantization")
    body(F, t, A, CH1)

    ox, oy, ux, vy = 900.0, 800.0, 150.0, 95.0
    pa = ease_in_out(ramp(t, 0.8, 1.4))
    ax = Stroke(860, 230, 1790, 840)
    ax.line((ox, oy), (ox + 860 * pa, oy))
    ax.line((ox, oy), (ox, oy - 556 * pa))
    if pa > 0.98:
        ax.poly([(ox + 852, oy - 5), (ox + 860, oy), (ox + 852, oy + 5)])
        ax.poly([(ox - 5, oy - 548), (ox, oy - 556), (ox + 5, oy - 548)])
    ax.blit(F, IVORY, 0.38 * A)
    label(F, "能量  E", ox + 18, oy - 538, A * pa, align="left", t=t, t0=1.8)
    label(F, "时间 / 过程", ox + 860, oy + 36, A * pa, align="right", t=t, t0=1.8)

    rp = ease_in_out(ramp(t, 2.0, 2.4))
    m = ease_in_out(ramp(t, 5.4, 2.2))
    if rp > 0:
        u = np.linspace(0, 5.6 * rp, 900)
        fr = u - np.floor(u)
        stair = np.floor(u) + np.clip((fr - 0.95) / 0.05, 0, 1) ** 2 * (3 - 2 * np.clip((fr - 0.95) / 0.05, 0, 1))
        v = (1 - m) * u + m * stair
        cs = Stroke(860, 230, 1790, 840)
        cs.poly(np.column_stack([ox + u * ux, oy - v * vy]), th=2)
        col = mix(CYAN, GOLD, m)
        cs.glow(F, col, 0.9 * A, 8)
        cs.blit(F, mix(IVORY, GOLD, m * 0.8), A)
        label(F, "经典图景：能量连续变化", 1510, 610, A * (1 - smooth(ramp(t, 5.4, 0.6))), align="left",
              t=t, t0=3.6)
        label(F, "量子图景：能量一份一份", 1510, 610, A * smooth(ramp(t, 7.0, 0.8)), color=GOLD, align="left")
    if m > 0.9:
        for n in range(1, 6):
            ta = A * smooth(ramp(t, 7.2 + 0.12 * n, 0.6))
            yy = oy - n * vy
            dl = Stroke(ox - 6, yy - 2, ox + n * ux + 2, yy + 3)
            dl.line((ox - 5, yy), (ox, yy))
            dl.dashed((ox + n * ux - 60, yy), (ox + n * ux - 4, yy), 3, 5)
            dl.blit(F, IVORY, 0.3 * ta)
            lab = R([(f"{n}" if n > 1 else "", "stix", 22, None), ("hν", "stix_it", 22, None)])
            lab.draw(F, ox - 14, yy + 7, MUTED, ta, align="right")

    if t > 8.6:
        tt = (t - 8.6) % 5.0
        hop, rest = 0.78, 0.42
        if tt < 5 * hop:
            n = int(tt / hop)
            f = (tt - n * hop) / hop
            if f < rest:
                uu, vv, flash = n + 0.5, n, max(0.0, 1 - f / 0.25)
            else:
                q = ease_in_out((f - rest) / (1 - rest))
                uu, vv, flash = n + 0.5 + q, n + q + 0.75 * math.sin(math.pi * q), 0.0
            pa2 = 1.0 if n > 0 or tt > 0.3 else tt / 0.3
        else:
            uu, vv, flash = 5.5, 5, 0.0
            pa2 = 1 - (tt - 5 * hop) / (5.0 - 5 * hop)
        px, py = ox + uu * ux, oy - vv * vy - 9
        add_sprite(F, px, py, 2.6, WHITE, 0.95 * A * pa2)
        add_sprite(F, px, py, 11, GOLD, 0.5 * A * pa2)
        if flash > 0 and (t - 8.6) > 0.5:
            add_sprite(F, px, py + 6, 26, GOLD, 0.35 * flash * A * pa2)

    eq = R([("E", "stix_it", 88, None), (" = ", "stix", 88, None), ("h", "stix_it", 88, None),
            ("ν", "stix_it", 88, None)])
    eq.draw(F, 968, 470, IVORY, A, t=t, t0=13.6, mode="wipe", dur=1.4, glow=0.3, glow_color=GOLD,
            glow_sigma=16)
    cap = R([("h", "stix_it", 24, None), (" ≈ ", "math", 22, None), ("6.626 × 10", "stix", 24, None),
             ("⁻³⁴", "stix", 24, None), (" J·s", "stix", 24, None)])
    cap.draw(F, 972, 518, MUTED, A, t=t, t0=14.4, mode="wipe", dur=1.2)
    label(F, "普朗克常数", 972, 554, A, align="left", t=t, t0=15.0)


# ------------------------------------------- 02 / 03 double slit ---------

CH2 = [
    (2.0, "让电子一个接一个地，射向两条狭缝。", "body", 0),
    (6.0, "每个电子只在屏上留下一个点——", "body", 1),
    (8.2, "像一颗粒子。", "body", 1),
    (11.5, "可成千上万个点累积起来，", "body", 2),
    (13.5, "却排出明暗相间的干涉条纹——", "body", 2),
    (16.5, "像一列波。", "body", 2),
    (19.5, "它既是粒子，也是波。", "gold", 3),
]

CH3 = [
    (1.5, "若在缝边放一台探测器，", "body", 0),
    (3.5, "确认电子究竟穿过了哪条缝——", "body", 0),
    (9.0, "干涉条纹，消失了。", "body", 1),
    (13.0, "在量子世界，测量不是旁观，", "gold", 2),
    (15.0, "而是参与。", "gold", 2),
]

_ds = {}


def _ds_fields():
    if not _ds:
        bx, sx, cy = P.DS_BARRIER_X, P.DS_SCREEN_X, P.DS_CY
        yy, xx = np.mgrid[252:868:2, int(bx) + 4:int(sx) - 2:2].astype(np.float32)
        r1 = np.hypot(xx - bx, yy - (cy - P.DS_HALF_SEP))
        r2 = np.hypot(xx - bx, yy - (cy + P.DS_HALF_SEP))
        ang = np.arctan2(yy - cy, xx - bx)
        env = np.exp(-(ang / 0.75) ** 2) * np.clip((xx - bx) / 30, 0, 1)
        _ds["right"] = (r1, r2, env)
        yl, xl = np.mgrid[252:868:2, int(P.DS_SOURCE[0]):int(bx) - 2:2].astype(np.float32)
        r0 = np.hypot(xl - P.DS_SOURCE[0], yl - cy)
        a0 = np.arctan2(yl - cy, xl - P.DS_SOURCE[0])
        _ds["left"] = (r0, np.exp(-(a0 / 0.5) ** 2) * np.clip(r0 / 25, 0, 1))
    return _ds


def _ds_screen_xy(xu, y):
    return P.DS_SCREEN_X + 16 + xu * 46, P.DS_CY + y


def double_slit(F, g, fi):
    a = window(g, START["ch2"], START["ch4"], 1.0, 0.9)
    if a <= 0:
        return
    t2 = g - START["ch2"]
    t3 = g - START["ch3"]
    D = P.double_slit()
    S = P.DS_SOURCE
    bx, sxl, cy, hs = P.DS_BARRIER_X, P.DS_SCREEN_X, P.DS_CY, P.DS_HALF_SEP

    pa = ease_in_out(ramp(t2, 0.5, 1.5))
    st = Stroke(840, 220, 1790, 900)
    gap = 7
    for y0, y1 in ((250, cy - hs - gap), (cy - hs + gap, cy + hs - gap), (cy + hs + gap, 870)):
        st.line((bx, y0), (bx, y1), th=2)
    st.blit(F, IVORY, 0.6 * a * pa)
    sc = Stroke(sxl - 3, 240, sxl + 3, 880)
    sc.line((sxl, 250), (sxl, 870))
    sc.blit(F, IVORY, 0.4 * a * pa)
    add_sprite(F, S[0], S[1], 2.6, WHITE, 0.9 * a * pa)
    add_sprite(F, S[0], S[1], 13, GOLD, 0.35 * a * pa)
    label(F, "电子源", S[0], 912, a * pa)
    label(F, "双缝", bx, 912, a * pa)
    label(F, "探测屏", sxl + 40, 912, a * pa)

    # wave picture
    wa = smooth(ramp(t2, 16.0, 2.2)) * (1 - smooth(ramp(t3, 4.5, 1.2))) * a
    if wa > 0.003:
        f = _ds_fields()
        k = 2 * np.pi / P.DS_LAMBDA
        ph = 14.0 * g
        r1, r2, env = f["right"]
        psi = (np.exp(1j * (k * r1 - ph)) / np.sqrt(r1 / 60 + 1) + np.exp(1j * (k * r2 - ph)) / np.sqrt(r2 / 60 + 1))
        img = (psi.real ** 2 * env).astype(np.float32)
        img = cv2_resize(img, 2)
        add_light(F, img, int(bx) + 4, 252, CYAN, 0.16 * wa)
        r0, e0 = f["left"]
        img0 = (np.cos(k * r0 - ph) ** 2 * e0 / np.sqrt(r0 / 80 + 1)).astype(np.float32)
        add_light(F, cv2_resize(img0, 2), int(S[0]), 252, CYAN, 0.14 * wa)

    # detector (chapter 3)
    det = (1205.0, 430.0)
    da = smooth(ramp(t3, 1.4, 1.2)) * a if t3 > 0 else 0.0
    on = smooth(ramp(t3, 4.5, 0.6))
    if da > 0:
        ds = Stroke(1120, 395, 1240, 470)
        ds.circle(det, 15, th=1)
        ds.dashed((det[0] - 12, det[1] + 10), (bx + 6, cy - hs), 3, 5)
        ds.dashed((det[0] - 4, det[1] + 15), (bx + 6, cy + hs - 30), 3, 5)
        ds.blit(F, mix(IVORY, GOLD, on), (0.5 + 0.4 * on) * da)
        add_sprite(F, det[0], det[1], 3.0, mix(IVORY, GOLD, on), (0.5 + 0.5 * on) * da)
        if on > 0:
            flick = 0.5 + 0.5 * math.sin(g * 23.0) * math.sin(g * 7.3)
            add_sprite(F, det[0], det[1], 22, GOLD, 0.25 * on * da * (0.6 + 0.4 * flick))
        label(F, "探测器", det[0] + 30, det[1] + 6, da, align="left")

    # hits on the screen
    xs_l, ws_l, ys_l = [], [], []
    n1 = int(np.searchsorted(D["t1"] + 0.75, t2))
    ghost = 1 - 0.9 * smooth(ramp(t3, 4.5, 1.0))
    if n1 > 0:
        x, y = _ds_screen_xy(D["x1"][:n1], D["y1"][:n1])
        xs_l.append(x), ys_l.append(y), ws_l.append(np.full(n1, ghost, np.float32))
    n2 = int(np.searchsorted(D["t2"] + 0.75, t3)) if t3 > 0 else 0
    if n2 > 0:
        x, y = _ds_screen_xy(D["x2"][:n2], D["y2"][:n2])
        xs_l.append(x), ys_l.append(y), ws_l.append(np.ones(n2, np.float32))
    if xs_l:
        splat_cloud(F, np.concatenate(xs_l), np.concatenate(ys_l), 0.95 * a, core=CYAN, halo=BLUE,
                    weights=np.concatenate(ws_l), bounds=(int(sxl) + 2, 236, int(sxl) + 80, 884), halo_gain=0.6)

    # intensity profile
    def profile(ys, wscale, alpha, dashed=False):
        if len(ys) < 3:
            return
        hist, _ = np.histogram(ys - cy, bins=150, range=(-300, 300))
        hist = blur(hist.astype(np.float32)[None, :], 2.6)[0]
        hist = hist / max(hist.max(), 7.0) * wscale
        yy = np.linspace(cy - 298, cy + 298, 150)
        pts = np.column_stack([sxl + 92 + hist * 70, yy])
        pr = Stroke(sxl + 80, 240, 1800, 880)
        if dashed:
            for i in range(0, len(pts) - 1, 3):
                pr.line(pts[i], pts[i + 1])
        else:
            pr.poly(pts, th=1)
        pr.glow(F, GOLD, 0.5 * alpha, 5)
        pr.blit(F, GOLD, alpha)

    if n1 > 0:
        profile(D["y1"][:n1] + cy, 1.0, 0.8 * a * (1 - 0.7 * smooth(ramp(t3, 4.5, 1.0))), dashed=t3 > 5.5)
    if n2 > 0:
        profile(D["y2"][:n2] + cy, 1.0, 0.8 * a)

    # arrival flashes
    for tarr, xu, yv, tloc in ((D["t1"] + 0.75, D["x1"], D["y1"], t2), (D["t2"] + 0.75, D["x2"], D["y2"], t3)):
        lo = int(np.searchsorted(tarr, tloc - 0.4))
        hi = int(np.searchsorted(tarr, tloc))
        if 0 < hi - lo <= 25:
            for i in range(lo, hi):
                q = (tloc - tarr[i]) / 0.4
                x, y = _ds_screen_xy(xu[i], yv[i])
                add_sprite(F, x, y, 7, GOLD, 0.6 * (1 - q) * a)

    # lone electrons in flight
    if 0 < t2 < 10:
        for te in D["t1"][:11]:
            tau = t2 - te
            if 0 <= tau < 0.38:
                q = tau / 0.38
                x, y = lerp(S[0], bx, q), S[1]
                add_sprite(F, x, y, 2.4, WHITE, 0.95 * a)
                add_sprite(F, x, y, 9, CYAN, 0.45 * a)
            elif 0.38 <= tau < 0.78:
                q = (tau - 0.38) / 0.4
                arc = Stroke(bx, 240, sxl, 880)
                rr = q * (sxl - bx)
                for yc in (cy - hs, cy + hs):
                    arc.ellipse((bx, yc), (rr, rr), 0, th=1, a0=-60, a1=60)
                arc.glow(F, CYAN, 0.6 * (1 - q) * a, 3)
                arc.blit(F, CYAN, 0.5 * (1 - q) * a)
    if 0 < t3 < 10:
        for i, te in enumerate(D["t2"][:6]):
            tau = t3 - te
            if 0 <= tau < 0.78:
                slit_y = cy + hs * (1 if D["slit2"][i] > 0 else -1)
                hx, hy = _ds_screen_xy(D["x2"][i], D["y2"][i])
                if tau < 0.38:
                    q = tau / 0.38
                    x, y = lerp(S[0], bx, q), lerp(S[1], slit_y, q)
                else:
                    q = (tau - 0.38) / 0.4
                    x, y = lerp(bx, hx, q), lerp(slit_y, hy, q)
                    add_sprite(F, bx, slit_y, 16, GOLD, 0.6 * (1 - q) * a)
                    add_sprite(F, det[0], det[1], 26, GOLD, 0.5 * (1 - q) * a)
                add_sprite(F, x, y, 2.4, WHITE, 0.95 * a)
                add_sprite(F, x, y, 9, CYAN, 0.45 * a)


def cv2_resize(img, f):
    import cv2
    return cv2.resize(img, (img.shape[1] * f, img.shape[0] * f), interpolation=cv2.INTER_CUBIC)


def ch2(F, t, g, fi):
    A = window(t, 0, 24, 0.9, 0.9)
    header(F, t, A, 2, "波粒二象性", "Wave–Particle Duality")
    body(F, t, A, CH2)
    double_slit(F, g, fi)


def ch3(F, t, g, fi):
    A = window(t, 0, 20, 0.9, 0.9)
    header(F, t, A, 3, "测量", "The Act of Measurement")
    body(F, t, A, CH3)
    double_slit(F, g, fi)


# ----------------------------------------- 04 superposition & collapse ---

CH4 = [
    (2.0, "量子力学用「波函数」描述粒子。", "body", 0),
    (5.0, "它不说粒子在哪里，", "body", 1),
    (6.8, "只给出在各处找到它的概率。", "body", 1),
    (10.0, "测量之前，多种可能同时存在——", "body", 2),
    (12.2, "这就是「叠加态」。", "gold", 2),
    (16.2, "测量的一瞬，可能性坍缩为一个结果。", "body", 3),
]

_c4 = {}


def ch4(F, t, g, fi):
    A = window(t, 0, 24, 0.9, 0.9)
    header(F, t, A, 4, "叠加与坍缩", "Superposition & Collapse")
    body(F, t, A, CH4)

    cx, cy, sc, tilt = 1300.0, 478.0, 12.5, 0.34
    pts = P.orbitals()["d_z2"]
    yaw = 0.6 + 0.2 * t
    tc0, tc1 = 15.0, 16.0
    if "target" not in _c4:
        sxa, sya = project(pts, cx, cy, sc, 0.6 + 0.2 * tc0, tilt)
        cand = np.where(sya < cy - 150)[0]
        i0 = cand[np.argmin(np.abs(sxa[cand] - cx - 18))]
        tx, ty = float(sxa[i0]), float(sya[i0])
        d = np.hypot(sxa - tx, sya - ty)
        _c4.update(target=(tx, ty), delay=(d / d.max()).astype(np.float32))
    tx, ty = _c4["target"]

    gain = 0.78 * smooth(ramp(t, 0.6, 2.6)) * A
    if t < tc1 + 0.05:
        sub = cloud_subset(len(pts), fi, 0.72)
        sx, sy = project(pts[sub], cx, cy, sc, yaw, tilt)
        if t > tc0:
            u = np.clip((t - tc0 - _c4["delay"][sub] * 0.35) / 0.65, 0, 1)
            e = u ** 3
            sx = sx + (tx - sx) * e
            sy = sy + (ty - sy) * e
        splat_cloud(F, sx, sy, gain, core=CYAN, halo=VIOLET)
    if t > tc1 - 0.1:
        q = t - tc1
        flash = max(0.0, 1 - q / 0.9) if q > 0 else 0.0
        add_sprite(F, tx, ty, 70, CYAN, 0.55 * flash * A)
        add_sprite(F, tx, ty, 26, WHITE, 0.6 * flash * A)
        if q > 0:
            ring = Stroke(tx - 260, ty - 260, tx + 260, ty + 260)
            ring.circle((tx, ty), 20 + 220 * ease_out(q / 1.4), th=1)
            ring.blit(F, CYAN, 0.6 * max(0.0, 1 - q / 1.4) * A)
        pulse = 0.85 + 0.15 * math.sin(t * 3.0)
        add_sprite(F, tx, ty, 2.8, WHITE, A * pulse)
        add_sprite(F, tx, ty, 12, GOLD, 0.55 * A * pulse)
        label(F, "测量结果", tx + 24, ty + 6, A, color=GOLD, align="left", t=t, t0=16.6)

    la = A * (1 - smooth(ramp(t, tc0, 0.6)))
    label(F, "概率云", 1580, 236, la, align="left", t=t, t0=7.0)
    R([("|", "stix", 26, None), ("ψ", "stix_it", 26, None), ("|²", "stix", 26, None)]).draw(
        F, 1582, 272, MUTED, la, t=t, t0=7.3, mode="fade")

    eq = R([("i", "stix_it", 56, None), ("ħ", "stix_it", 56, None), ("  ∂", "math", 52, None),
            ("ψ", "stix_it", 56, None), ("/", "stix", 56, None), ("∂", "math", 52, None),
            ("t", "stix_it", 56, None), ("  =  ", "stix", 56, None), ("Ĥ", "stix_it", 56, None),
            ("ψ", "stix_it", 56, None)])
    eq.draw(F, 1300, 878, IVORY, A, t=t, t0=3.0, mode="wipe", dur=1.6, align="center", glow=0.25,
            glow_color=GOLD, glow_sigma=14)
    label(F, "薛定谔方程  ·  1926", 1300, 922, A, t=t, t0=3.8)


# ------------------------------------------------- 05 uncertainty ------

CH5 = [
    (2.0, "海森堡发现：位置与动量，", "body", 0),
    (4.0, "无法同时被精确地确定。", "body", 0),
    (7.5, "位置越清晰，动量就越模糊；", "body", 1),
    (9.5, "动量越清晰，位置就越模糊。", "body", 1),
    (13.5, "这并非仪器不够精密，", "body", 2),
    (15.5, "而是自然本身的规则。", "gold", 2),
]


def ch5(F, t, g, fi):
    A = window(t, 0, 20, 0.9, 0.9)
    header(F, t, A, 5, "不确定性原理", "The Uncertainty Principle")
    body(F, t, A, CH5)

    cx, x0, x1 = 1300.0, 880.0, 1720.0
    yb1, yb2 = 470.0, 790.0
    pa = ease_in_out(ramp(t, 0.6, 1.4))
    ax = Stroke(860, 200, 1740, 860)
    for yb in (yb1, yb2):
        ax.line((cx - 420 * pa, yb), (cx + 420 * pa, yb))
        ax.line((cx, yb - 4), (cx, yb + 4))
    ax.blit(F, IVORY, 0.3 * A)
    label(F, "位置", x0, yb1 + 34, A * pa, align="left")
    label(F, "动量", x0, yb2 + 34, A * pa, align="left")
    R([("x", "stix_it", 22, None)]).draw(F, x1, yb1 + 32, MUTED, A * pa, align="right")
    R([("p", "stix_it", 22, None)]).draw(F, x1, yb2 + 32, MUTED, A * pa, align="right")

    s0 = 56.0
    sx = s0 * math.exp(-0.85 * math.sin(2 * math.pi * (t - 5.5) / 8.0) * smooth(ramp(t, 3.0, 2.5)))
    sp = s0 * s0 / sx
    X = np.linspace(-420, 420, 700)
    ca = A * smooth(ramp(t, 1.2, 1.2))

    def curve(yb, sig, color, amp, re=None):
        hgt = amp * (s0 / sig) ** 0.5
        y = yb - hgt * np.exp(-X ** 2 / (2 * sig ** 2))
        pts = np.column_stack([cx + X, y])
        fl = Stroke(860, 200, 1740, 860)
        fl.fill(np.vstack([pts, [[cx + 420, yb], [cx - 420, yb]]]))
        fl.blit(F, color, 0.1 * ca)
        ln = Stroke(860, 200, 1740, 860)
        ln.poly(pts, th=2)
        ln.glow(F, color, 0.8 * ca, 7)
        ln.blit(F, mix(color, WHITE, 0.25), ca)
        if re is not None:
            rl = Stroke(860, 200, 1740, 860)
            rl.poly(np.column_stack([cx + X, yb - hgt * 0.62 * re]), th=1)
            rl.blit(F, color, 0.35 * ca)
        return hgt

    env = np.exp(-X ** 2 / (4 * sx ** 2))
    curve(yb1, sx, CYAN, 150, re=env * np.cos(2 * np.pi * X / 34 - 4.0 * t))
    curve(yb2, sp, GOLD, 150)

    for yb, sig, sym, col in ((yb1, sx, "x", CYAN), (yb2, sp, "p", GOLD)):
        bk = Stroke(860, 200, 1740, 880)
        yy = yb + 16
        bk.line((cx - sig, yy), (cx + sig, yy))
        bk.line((cx - sig, yy - 5), (cx - sig, yy + 5))
        bk.line((cx + sig, yy - 5), (cx + sig, yy + 5))
        bk.blit(F, col, 0.8 * ca)
        R([("Δ", "stix", 24, None), (sym, "stix_it", 24, None)]).draw(F, cx, yy + 30, col, ca, align="center")

    eq = R([("Δ", "stix", 54, None), ("x", "stix_it", 54, None), (" · ", "stix", 54, None),
            ("Δ", "stix", 54, None), ("p", "stix_it", 54, None), ("  ≥  ", "math", 48, None),
            ("ħ", "stix_it", 54, None), (" / 2", "stix", 54, None)])
    eq.draw(F, cx, 930, IVORY, A, t=t, t0=11.5, mode="wipe", dur=1.4, align="center", glow=0.3,
            glow_color=GOLD, glow_sigma=14)


# ---------------------------------------------------- 06 tunnelling ----

CH6 = [
    (2.0, "经典世界里，能量不足的小球，", "body", 0),
    (4.0, "翻不过山坡，只能折返。", "body", 0),
    (7.5, "量子世界里，粒子却有一定概率", "body", 1),
    (10.2, "直接「穿墙而过」。", "gold", 1),
    (14.0, "太阳内部的核聚变，", "body", 2),
    (16.0, "正是依靠隧穿效应才得以持续。", "body", 2),
]


def ch6(F, t, g, fi):
    A = window(t, 0, 20, 0.9, 0.9)
    header(F, t, A, 6, "量子隧穿", "Quantum Tunneling")
    body(F, t, A, CH6)

    D = P.tunnelling()
    cx, pxu, y0, S = 1300.0, 7.0, 770.0, 470.0
    yV = y0 - P.TUN_V0 * S
    yE = y0 - 0.5 * S
    pa = ease_in_out(ramp(t, 0.6, 1.4))
    hw = P.TUN_A / 2 * pxu
    bar = Stroke(1280, 440, 1320, 780)
    bar.fill([(cx - hw, y0), (cx - hw, yV), (cx + hw, yV), (cx + hw, y0)])
    bar.blit(F, GOLD, 0.16 * A * pa)
    ed = Stroke(1280, 440, 1320, 780)
    ed.poly([(cx - hw, y0), (cx - hw, yV), (cx + hw, yV), (cx + hw, y0)], th=1)
    ed.glow(F, GOLD, 0.4 * A * pa, 6)
    ed.blit(F, GOLD, 0.8 * A * pa)
    label(F, "势垒", cx, y0 + 34, A * pa, color=GOLD)
    base = Stroke(860, y0 - 2, 1740, y0 + 3)
    base.line((cx - 434 * pa, y0), (cx + 434 * pa, y0))
    base.blit(F, IVORY, 0.25 * A)
    el = Stroke(860, yE - 2, 1740, yE + 3)
    el.dashed((cx - 434, yE), (cx - 434 + 868 * pa, yE), 5, 7)
    el.blit(F, IVORY, 0.28 * A)
    label(F, "粒子能量", 866, yE - 14, A * pa, align="left")

    # classical ball: rolls in, bounces back
    ba = window(t, 1.2, 6.2, 0.4, 0.8)
    if ba > 0:
        xl = cx + (-42 * pxu)
        xr = cx - hw - 9
        u = (t - 1.2) / 2.6
        bx_ = lerp(xl, xr, ease_in_out(u)) if u <= 1 else lerp(xr, xl - 40, ease_out((t - 3.8) / 2.4))
        add_sprite(F, bx_, yE - 9, 3.2, WHITE, 0.9 * A * ba)
        add_sprite(F, bx_, yE - 9, 12, GOLD, 0.5 * A * ba)
        label(F, "经典粒子", bx_, yE - 34, A * ba * 0.9, color=GOLD)
        if 3.75 < t < 4.4:
            add_sprite(F, xr + 4, yE - 9, 22, GOLD, 0.4 * (1 - (t - 3.75) / 0.65) * A)

    # quantum wave packet (split-step simulation)
    qa = smooth(ramp(t, 5.4, 1.0)) * A
    if qa > 0:
        sim_t = max(0.0, t - 6.0) * 7.0
        fpos = min(sim_t / P.TUN_T_END * (P.TUN_FRAMES - 1), P.TUN_FRAMES - 1.001)
        i0 = int(fpos)
        fr = fpos - i0
        prob = D["prob"][i0] * (1 - fr) + D["prob"][i0 + 1] * fr
        re = D["re"][i0] * (1 - fr) + D["re"][i0 + 1] * fr
        X = cx + P.TUN_XS * pxu
        p0 = 1 / (math.sqrt(2 * math.pi) * P.TUN_SIGMA)
        hgt = np.minimum(prob / p0 * 140, 300)
        pts = np.column_stack([X, yE - hgt])
        fl = Stroke(860, 200, 1740, 790)
        fl.fill(np.vstack([pts, [[X[-1], yE], [X[0], yE]]]))
        fl.blit(F, CYAN, 0.13 * qa)
        ln = Stroke(860, 200, 1740, 790)
        ln.poly(pts, th=2)
        ln.glow(F, CYAN, 0.85 * qa, 7)
        ln.blit(F, mix(CYAN, WHITE, 0.3), qa)
        r0 = (2 * math.pi * P.TUN_SIGMA ** 2) ** -0.25
        rl = Stroke(860, 200, 1740, 790)
        rl.poly(np.column_stack([X, yE - np.clip(re / r0 * 70, -160, 160)]), th=1)
        rl.blit(F, VIOLET, 0.4 * qa)
        label(F, "量子波包", 1300 + P.TUN_X0 * pxu, yE - 170, qa * (1 - smooth(ramp(t, 8.5, 0.8))),
              color=CYAN)
        ra = A * smooth(ramp(t, 15.2, 1.0))
        T(f"反射 ≈ {round(float(D['R']) * 100)}%", "sans", 20, 300, tracking=0.12).draw(
            F, 1040, 650, MUTED, ra, align="center")
        T(f"透射 ≈ {round(float(D['T']) * 100)}%", "sans", 20, 400, tracking=0.12).draw(
            F, 1560, 650, CYAN, ra, align="center")


# --------------------------------------------------- 07 entanglement ---

CH7 = [
    (2.0, "两个处于纠缠态的粒子，", "body", 0),
    (4.0, "无论相隔多远——", "body", 0),
    (9.8, "测量其中一个，", "body", 1),
    (12.3, "另一个的结果便立刻与之关联。", "body", 1),
    (15.5, "爱因斯坦称之为「鬼魅般的超距作用」。", "body", 2),
    (18.3, "但它无法被用来超光速传递信息。", "gold", 3),
    (20.2, "2022 年诺贝尔物理学奖授予了纠缠光子实验。", "note", 3),
]


def ch7(F, t, g, fi):
    A = window(t, 0, 24, 0.9, 0.9)
    header(F, t, A, 7, "量子纠缠", "Quantum Entanglement")
    body(F, t, A, CH7)

    cx, cy = 1300.0, 500.0
    tm = 12.0
    sep = ease_out_expo(ramp(t, 1.3, 3.6))
    ax_, bx_ = cx - 340 * sep, cx + 340 * sep
    born = smooth(ramp(t, 1.0, 0.4))
    if 0.9 < t < 2.6:
        q = (t - 1.0) / 1.6
        add_sprite(F, cx, cy, 60, VIOLET, 0.5 * max(0.0, 1 - q) * A)
        ring = Stroke(cx - 200, cy - 200, cx + 200, cy + 200)
        ring.circle((cx, cy), 10 + 170 * ease_out(q), th=1)
        ring.blit(F, VIOLET, 0.5 * max(0.0, 1 - q) * A)

    # entanglement threads
    thr = A * born * (1 - smooth(ramp(t, tm, 1.6)))
    if thr > 0 and sep > 0.02:
        s = np.linspace(0, 1, 400)
        xs = lerp(ax_, bx_, s)
        th = Stroke(840, 380, 1760, 620)
        for k, ph in enumerate((0.0, 2.1, 4.2)):
            amp = 16 + 6 * k
            ys = cy + amp * np.sin(2 * np.pi * 3.5 * s - 3.0 * t + ph) * np.sin(np.pi * s) ** 1.2
            th.poly(np.column_stack([xs, ys]), th=1)
        th.glow(F, VIOLET, 0.55 * thr, 6)
        th.blit(F, mix(VIOLET, CYAN, 0.4), 0.45 * thr)
        if t > tm - 0.05:
            add_sprite(F, cx, cy, 120, VIOLET, 0.25 * max(0.0, 1 - (t - tm) / 1.0) * A)

    measured = smooth(ramp(t, tm, 0.25))
    for side, px in ((-1, ax_), (1, bx_)):
        col = mix(VIOLET, GOLD if side < 0 else CYAN, measured)
        add_sprite(F, px, cy, 3.4, WHITE, 0.95 * A * born)
        add_sprite(F, px, cy, 18, col, 0.5 * A * born)
        arr = Stroke(px - 70, cy - 70, px + 70, cy + 70)
        if measured < 1:
            base = 2 * np.pi * 0.85 * t + (0 if side < 0 else np.pi)
            for j in range(6):
                ang = base - j * 0.16
                tip = (px + 50 * math.sin(ang), cy - 50 * math.cos(ang))
                arr.line((px, cy), tip, th=1, v=int(255 * (1 - j / 6) ** 1.5))
            arr.blit(F, col, 0.7 * A * born * (1 - measured))
            arr.m[:] = 0
        if measured > 0:
            d = -1 if side < 0 else 1
            tip = (px, cy + d * 52)
            arr.line((px, cy), tip, th=2)
            arr.line(tip, (px - 8, tip[1] - d * 10), th=2)
            arr.line(tip, (px + 8, tip[1] - d * 10), th=2)
            arr.glow(F, col, 0.7 * A * measured, 5)
            arr.blit(F, col, A * measured)
        R([("A" if side < 0 else "B", "stix_it", 30, None)]).draw(F, px, cy + 100, MUTED, A * born,
                                                                     align="center")
        if measured > 0:
            label(F, "自旋向上" if side < 0 else "自旋向下", px, cy - 78 if side < 0 else cy - 78, A,
                  color=GOLD if side < 0 else CYAN, t=t, t0=tm + 0.3)
    if t > tm:
        q = (t - tm) / 1.2
        if q < 1:
            ring = Stroke(ax_ - 120, cy - 120, ax_ + 120, cy + 120)
            ring.circle((ax_, cy), 18 + 90 * ease_out(q), th=1)
            ring.blit(F, GOLD, 0.7 * (1 - q) * A)
            add_sprite(F, ax_, cy, 40, GOLD, 0.5 * (1 - q) * A)
            add_sprite(F, bx_, cy, 40, CYAN, 0.5 * (1 - q) * A)

    dm = smooth(ramp(t, 4.6, 1.0)) * A
    if dm > 0:
        y = cy + 150
        dk = Stroke(840, y - 8, 1760, y + 8)
        dk.line((ax_, y), (bx_, y))
        dk.line((ax_, y - 5), (ax_, y + 5))
        dk.line((bx_, y - 5), (bx_, y + 5))
        dk.blit(F, IVORY, 0.3 * dm)
        label(F, "相隔：任意距离", cx, y + 36, dm)


# ------------------------------------------------------ 08 everywhere ---

CH8 = [
    (2.0, "听起来遥远，它却早已在你身边。", "body", 0),
    (5.0, "手机芯片、激光、核磁共振、", "body", 1),
    (7.0, "卫星导航里的原子钟……", "body", 1),
    (9.0, "都建立在量子力学之上。", "body", 1),
    (13.0, "而量子计算与量子通信，", "gold", 2),
    (15.0, "正在推开下一扇门。", "gold", 2),
]

TILES = [
    ("芯片", "SEMICONDUCTORS", "chip"),
    ("激光", "LASERS", "laser"),
    ("核磁共振", "MRI", "mri"),
    ("原子钟", "ATOMIC CLOCKS", "atom"),
    ("量子计算", "QUANTUM COMPUTING", "bloch"),
    ("量子通信", "QUANTUM NETWORKS", "link"),
]


def _icon(F, kind, c, t, color, a):
    x, y = c
    s = Stroke(x - 70, y - 60, x + 110, y + 60)
    glow = 0.0
    if kind == "chip":
        s.poly([(x - 28, y - 28), (x + 28, y - 28), (x + 28, y + 28), (x - 28, y + 28)], th=1, closed=True)
        s.poly([(x - 12, y - 12), (x + 12, y - 12), (x + 12, y + 12), (x - 12, y + 12)], th=1, closed=True)
        for k in (-16, 0, 16):
            s.line((x + k, y - 28), (x + k, y - 40))
            s.line((x + k, y + 28), (x + k, y + 40))
            s.line((x - 28, y + k), (x - 40, y + k))
            s.line((x + 28, y + k), (x + 40, y + k))
    elif kind == "laser":
        s.poly([(x - 40, y - 12), (x - 12, y - 12), (x - 12, y + 12), (x - 40, y + 12)], th=1, closed=True)
        bs = Stroke(x - 14, y - 6, x + 110, y + 6)
        bs.line((x - 10, y), (x + 100, y), th=2)
        p = 0.65 + 0.35 * math.sin(t * 5)
        bs.glow(F, color, 0.9 * a * p, 4)
        bs.blit(F, mix(color, WHITE, 0.4), a)
    elif kind == "mri":
        s.circle((x, y), 36, th=2)
        s.circle((x, y), 22, th=1)
        s.line((x - 56, y + 8), (x + 56, y + 8))
    elif kind == "atom":
        for ang in (0, 60, 120):
            s.ellipse((x, y), (40, 13), ang, th=1)
        add_sprite(F, x, y, 3, WHITE, a)
        for k, ang in enumerate((0, 60, 120)):
            ph = t * 2.2 + k * 2.1
            ex, ey = 40 * math.cos(ph), 13 * math.sin(ph)
            ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
            add_sprite(F, x + ex * ca - ey * sa, y + ex * sa + ey * ca, 2.4, color, a)
    elif kind == "bloch":
        s.circle((x, y), 38, th=1)
        s.ellipse((x, y), (38, 11), 0, th=1)
        s.line((x, y - 46), (x, y + 46))
        ph = t * 1.3
        tip = (x + 30 * math.cos(ph) * 0.9, y - 24 + 6 * math.sin(ph))
        s.line((x, y), tip, th=2)
        add_sprite(F, tip[0], tip[1], 3, WHITE, a)
        glow = 1.0
    elif kind == "link":
        add_sprite(F, x - 40, y, 3.2, WHITE, a)
        add_sprite(F, x + 60, y, 3.2, WHITE, a)
        s.circle((x - 40, y), 9, th=1)
        s.circle((x + 60, y), 9, th=1)
        xs = np.linspace(x - 30, x + 50, 120)
        ys = y + 9 * np.sin((xs - x) / 6.0 - t * 4) * np.sin(np.pi * (xs - x + 30) / 80)
        s.poly(np.column_stack([xs, ys]), th=1)
        glow = 1.0
    if glow:
        s.glow(F, color, 0.5 * a, 5)
    s.blit(F, color, a)


def ch8(F, t, g, fi):
    A = window(t, 0, 20, 0.9, 0.9)
    header(F, t, A, 8, "无处不在", "All Around Us")
    body(F, t, A, CH8)
    hi = smooth(ramp(t, 13.0, 1.0))
    for i, (zh, en, kind) in enumerate(TILES):
        col, row = i % 3, i // 3
        tx, ty = 850 + col * 320, 236 + row * 350
        t0 = 2.6 + 0.32 * i
        a = A * smooth(ramp(t, t0, 0.8))
        future = i >= 4
        accent = mix(IVORY, GOLD, hi) if future else IVORY
        dim = (1 - 0.45 * hi) if not future else 1.0
        hline(F, tx, ty, 290 * ease_in_out(ramp(t, t0, 1.0)), GOLD if future else IVORY,
              (0.55 if future else 0.3) * A)
        T(f"{i + 1:02d}", "jost", 15, 400, tracking=0.3).draw(F, tx, ty + 34, MUTED, a * dim)
        if future:
            T("NEXT", "jost", 13, 500, tracking=0.4).draw(F, tx + 290, ty + 34, GOLD, a * hi, align="right")
        _icon(F, kind, (tx + 60, ty + 126), t, accent, a * dim * 0.9)
        T(zh, "serif", 34, 500, tracking=0.06).draw(F, tx, ty + 228, accent, a * dim, t=t, t0=t0 + 0.2,
                                                     stagger=0.05, dur=0.7, rise=8)
        T(en, "jost", 13, 400, tracking=0.3).draw(F, tx, ty + 262, MUTED, a * dim)


# ------------------------------------------------------------- outro ---

def outro(F, t, g, fi):
    ca = window(t, 0.0, 19.6, 2.0, 3.0)
    pts = P.orbitals()["f_z3"]
    sub = cloud_subset(len(pts), fi, 0.6)
    sx, sy = project(pts[sub], 960, 520, 12.0, 0.3 + 0.08 * t, 0.5)
    splat_cloud(F, sx, sy, 0.16 * ca, core=CYAN, halo=VIOLET)

    qa = window(t, 0.6, 11.0, 0.1, 1.4)
    T("「我想我可以有把握地说，", "serif", 46, 400, tracking=0.1).draw(
        F, 960 - 23, 470, IVORY, qa, t=t, t0=0.8, stagger=0.07, dur=1.0, rise=12, align="center")
    T("没有人真正理解量子力学。」", "serif", 46, 400, tracking=0.1).draw(
        F, 960 + 23, 548, IVORY, qa, t=t, t0=2.6, stagger=0.07, dur=1.0, rise=12, align="center")
    T("“I think I can safely say that nobody understands quantum mechanics.”", "corm_it", 30, 500,
      per_char=False).draw(F, 960, 626, MUTED, qa, t=t, t0=5.0, mode="wipe", dur=2.0, align="center")
    T("—— 理查德 · 费曼", "serif", 26, 500, tracking=0.12).draw(F, 960, 700, GOLD, qa, t=t, t0=6.6,
                                                              stagger=0.05, dur=0.8, align="center")

    fa = window(t, 11.9, 19.8, 0.1, 2.4)
    T("世界的底层，远比我们想象的更奇妙。", "serif", 44, 400, tracking=0.14).draw(
        F, 960, 520, IVORY, fa, t=t, t0=12.0, stagger=0.07, dur=1.2, rise=12, align="center",
        glow=0.12, glow_color=GOLD, glow_sigma=16)
    w = 200 * ease_in_out(ramp(t, 13.6, 1.2))
    hline(F, 960 - w / 2, 580, w, GOLD, 0.6 * fa)
    T("量子力学", "serif", 26, 500, tracking=0.5).draw(F, 960 + 6, 640, IVORY, fa, t=t, t0=14.2,
                                                      stagger=0.1, dur=1.0, align="center")
    T("QUANTUM  MECHANICS", "jost", 15, 400, tracking=0.6).draw(F, 960 + 4, 676, GOLD, fa, t=t, t0=14.8,
                                                                stagger=0.02, dur=0.8, align="center")


SCENE_FUNCS = {
    "opening": opening, "ch1": ch1, "ch2": ch2, "ch3": ch3, "ch4": ch4, "ch5": ch5,
    "ch6": ch6, "ch7": ch7, "ch8": ch8, "outro": outro,
}
