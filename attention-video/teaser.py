"""10-second teaser for 注意力 · Attention Is All You Need (9:16, Douyin).

The viewer performs attention before being told what it is: in
"小猫没有跳上桌子，因为它太累了。" everybody knows instantly who 「它」 is.
Above the sentence a cat made of starlight (a 3D net of stars) acts it out - tries to jump onto a table,
falls short, lies down tired.  The fan of lines out of
「它」 is what a Transformer computes, and whichever word wins, its drawing
lights up.  Change one word (累 -> 高): the table grows, the cat looks up,
and the attention moves to the table.  (Weights illustrative.)

  python3 teaser.py   -> attention-teaser.mp4
"""

import math
import subprocess
from multiprocessing import Pool

import numpy as np

import synth as A
import figures as FG
import render3d as R3
import wire3d as W3
from style import (CX, GOLD, IVORY, H, W, E, ROOT, add_sprite, caps, ease_in_out, ease_out,
                   finalize, glow_poly, make_background, make_grain, mix, orb, ramp, serif, smooth, statement,
                   subtitles, twinkle, window, T)

FPS, DUR = 30, 10.0
STEEL = mix(np.array([0.45, 0.84, 1.0], np.float32), IVORY, 0.45)

SIZE, TRACK = 84, 0.12
L1, L2 = "小猫没有跳上桌子，", "因为它太累了。"
Y1, Y2 = 930, 1150
X0 = 520                                   # sentence centre (a touch left of the button column)
TOKENS = [(0, 0, 2, "小猫"), (0, 2, 4, "没有"), (0, 4, 6, "跳上"), (0, 6, 8, "桌子"),
          (1, 0, 2, "因为"), (1, 2, 3, "它"), (1, 3, 4, "太"), (1, 4, 5, "累"), (1, 5, 6, "了")]
IT, CAT, TABLE = 5, 0, 3
WA = np.array([0.58, 0.04, 0.05, 0.09, 0.04, 0.06, 0.04, 0.07, 0.03])   # ...因为它太累了
WB = np.array([0.09, 0.03, 0.07, 0.56, 0.03, 0.06, 0.05, 0.08, 0.03])   # ...因为它太高了
T1, T2 = 0.25, 1.35
T_HOP, T_LIE = 1.9, 2.5
T_IT, T_FAN, T_SWAP, T_RE, T_NAME = 2.6, 3.2, 5.0, 5.6, 7.8

# -------------------------------------------------------- the 3D scene ---

T_TABLE_IN, T_CAT_IN = 0.3, 0.6
TABLE_H0, TABLE_H1 = 1.0, 2.2
CAT_ORIGIN = (-0.72, 0.0, 0.05)



def camera(t):
    yaw = 0.12 + 0.36 * ease_in_out(t / DUR)                  # a slow drift round the scene shows its depth
    return R3.Camera(target=(0.45, 0.9, 0.0), dist=6.0, yaw=yaw, pitch=0.17, focal=1500, cx=520, cy=560)


def cat_state(t):
    down = ease_in_out(ramp(t, T_LIE, 0.8)) * (1 - ease_in_out(ramp(t, T_SWAP + 0.1, 0.8)))
    hop = math.sin(math.pi * ramp(t, T_HOP, 0.55)) if T_HOP <= t <= T_HOP + 0.55 else 0.0
    look = ease_in_out(ramp(t, T_SWAP + 0.5, 0.7))
    return round(down, 3), round(hop, 3), round(look, 3)


def table_h(t):
    return round(TABLE_H0 + (TABLE_H1 - TABLE_H0) * ease_in_out(ramp(t, T_SWAP, 1.0)), 3)


_wcat = []


def scene(F, t, g_cat, g_table):
    """Draw the table and the cat as starlight in 3D; returns screen anchors for the word links."""
    if not _wcat:                                    # one star net modelled sitting, one lying down
        _wcat.extend([W3.WireCat(), W3.WireCat(n=200, seed=5, rest_down=1.0)])
    sit_net, lie_net = _wcat
    cam = camera(t)
    down, hop, look = cat_state(t)
    h = table_h(t)
    r_tab = ramp(t, T_TABLE_IN, 1.0)
    r_cat = ramp(t, T_CAT_IN, 1.2)
    W3.floor_stars(F, cam, smooth(ramp(t, 0.1, 1.0)))
    tp, te = W3.table_wire(h)
    W3.draw_wire(F, cam, tp, None, te, g_table, 1.0, reveal=r_tab, seed=2)
    for net, a, sd in ((sit_net, 1 - smooth(down), 1), (lie_net, smooth(down), 4)):
        if a > 0.01:
            P, N, eyes = net.pose(down, hop, look, CAT_ORIGIN)
            W3.draw_wire(F, cam, P, N, net.edges, g_cat, a, reveal=r_cat, seed=sd, rest=net.rest)
    a_eye = smooth(ramp(t, T_CAT_IN + 1.0, 0.4))
    head = np.mean(eyes, axis=0)
    for e in eyes:                                   # open and bright while it sits; shut while it lies tired
        p = cam.project(e)
        near = 0.55 + 0.45 * float(np.clip((cam.project(head)[2] - p[2]) / 0.08 + 0.5, 0, 1))
        if down < 0.5:
            orb(F, p[0], p[1], 3.0, GOLD, a_eye * near * (1 - 2 * down) * (0.9 + 0.5 * g_cat), core=WARM)
        else:
            glow_poly(F, [(p[0] - 6, p[1] - 1), (p[0], p[1] + 2), (p[0] + 6, p[1] - 1)], GOLD,
                      a_eye * near * (2 * down - 1) * 0.8, th=1, glow=0.4, sigma=2)
    chest = cam.project(np.asarray(CAT_ORIGIN) + np.array([0.15 + 0.4 * down + 0.28 * hop,
                                                           0.45 - 0.2 * down + 0.55 * hop, 0]))
    top = cam.project(np.array([0.6, h, 0.5]))
    return chest[:2], top[:2]


WARM = np.array([1.0, 0.93, 0.8], np.float32)


# ------------------------------------------------------------- sentence ---

def layout():
    out = []
    for txt, y in ((L1, Y1), (L2, Y2)):
        tx = serif(txt, SIZE, 400, TRACK)
        x0 = X0 - tx.width / 2
        out.append(([(x0 + dx, x0 + dx + m.shape[1]) for m, dx, dy in tx.items], y))
    tok = []
    for line, a, b, _ in TOKENS:
        boxes, y = out[line]
        tok.append(((boxes[a][0] + boxes[b - 1][1]) / 2, y, line))
    return tok


TOK = layout()


def bezier(p0, p1, p2, n=48):
    u = np.linspace(0, 1, n)[:, None]
    return (1 - u) ** 2 * p0 + 2 * (1 - u) * u * p1 + u ** 2 * p2


def arc(i):
    """Curve from 「它」 to token i: up into the gap for line 1, a loop under line 2 for its neighbours."""
    sx, sy, _ = TOK[IT]
    tx, ty, line = TOK[i]
    if line == 0:
        p0 = np.array([sx, sy - SIZE * 0.92])
        p2 = np.array([tx, ty + SIZE * 0.22])
        p1 = np.array([(sx + tx) / 2, (p0[1] + p2[1]) / 2 + 40])
    else:
        p0 = np.array([sx, sy + SIZE * 0.24])
        p2 = np.array([tx, ty + SIZE * 0.24])
        p1 = np.array([(sx + tx) / 2, sy + SIZE * 0.24 + 34 + 0.16 * abs(tx - sx)])
    return bezier(p0, p1, p2)


ARCS = [arc(i) for i in range(len(TOKENS))]


def weights(t):
    uni = np.full(len(TOKENS), 1 / len(TOKENS))
    w = uni + (WA - uni) * ease_in_out(ramp(t, T_FAN + 0.3, 0.9))
    return w + (WB - w) * ease_in_out(ramp(t, T_RE, 1.0))


def line2(F, t):
    """Second line, typed like statement(), with one character that can change its mind."""
    a_txt = serif(L2, SIZE, 400, TRACK)
    b_txt = serif(L2.replace("累", "高"), SIZE, 400, TRACK)
    x0 = X0 - a_txt.width / 2
    sw = smooth(ramp(t, T_SWAP, 0.55))
    for i, (m, dx, dy) in enumerate(a_txt.items):
        u = ease_out(ramp(t, T2 + i / 7.5, 0.25))
        if u <= 0:
            continue
        col = GOLD if (i == 2 and t >= T_IT) else IVORY
        off = (1 - u) * 6
        if i == 4 and sw > 0:
            E.blend(F, m, x0 + dx, Y2 + dy + off - 18 * sw, IVORY, u * (1 - sw))
            mb, dxb, dyb = b_txt.items[i]
            E.blend(F, mb, x0 + dxb, Y2 + dyb + 14 * (1 - sw), mix(GOLD, IVORY, sw ** 2), sw)
        else:
            E.blend(F, m, x0 + dx, Y2 + dy + off, col, u)


def link(F, a, b, alpha, col):
    """Dotted thread from a word up to the thing it names in the picture."""
    if alpha <= 0.01:
        return
    n = int(np.hypot(*(b - a)) / 14)
    for k in range(1, n):
        p = a + (b - a) * k / n
        add_sprite(F, p[0], p[1], 1.6, col, 0.55 * alpha)


def frame(fi, bg, vig, tw, grain):
    t = fi / FPS
    F = bg.copy()
    twinkle(F, tw, t)
    w = weights(t)
    hl = smooth(ramp(t, T_FAN + 0.8, 0.5))
    rel = w / w.max()
    # the picture: a 3D cat and a table, lit gold when 「它」 is looking at them
    g_cat, g_tab = (hl * smooth((rel[i] - 0.3) / 0.7) for i in (CAT, TABLE))
    a_cat, a_tab = scene(F, t, g_cat, g_tab)
    for i, target in ((CAT, a_cat), (TABLE, a_tab)):
        tx, ty, _ = TOK[i]
        link(F, np.array([tx, ty - SIZE - 66]), np.asarray(target) + np.array([0, 14]),
             hl * smooth((rel[i] - 0.5) / 0.5), GOLD)
    # the fan of attention out of 「它」
    grow = ease_out(ramp(t, T_FAN, 0.9))
    if grow > 0:
        top = int(np.argmax(w))
        for i, pts in enumerate(ARCS):
            if i == IT:
                continue
            k = max(2, int(len(pts) * grow))
            col = mix(STEEL, GOLD, smooth(rel[i]))
            glow_poly(F, pts[:k], col, (0.26 + 0.74 * rel[i] ** 1.2) * grow,
                      th=1 if rel[i] < 0.4 else 2 if rel[i] < 0.8 else 3, glow=0.4 + 1.2 * rel[i],
                      sigma=4 + 4 * rel[i])
            if k < len(pts):
                orb(F, pts[k - 1][0], pts[k - 1][1], 4, col, 0.8 * grow)
        tx, ty, _ = TOK[top]
        add_sprite(F, tx, ty - SIZE * 0.35, 70, GOLD, 0.2 * hl)
        T(f"{round(100 * w[top])}%", "stix", 36, None, per_char=False).draw(
            F, tx, ty - SIZE - 16, GOLD, hl, align="center")
    statement(F, L1, X0, Y1, t, T1, None, size=SIZE, tracking=TRACK, cps=7.5, fade_in=0.25)
    line2(F, t)
    if t >= T_IT:                                    # 「它」 glow pulse
        sx, sy, _ = TOK[IT]
        p = math.exp(-(t - T_IT) * 2.2)
        add_sprite(F, sx, sy - SIZE * 0.35, 46 + 30 * p, GOLD, 0.25 + 0.5 * p)
    if T_SWAP <= t < T_SWAP + 0.9:                   # the swapped word flashes
        sx, sy, _ = TOK[7]
        q = (t - T_SWAP) / 0.9
        add_sprite(F, sx, sy - SIZE * 0.35, 40 + 60 * q, GOLD, 0.7 * math.sin(math.pi * q))
    subtitles(F, t, [(T_IT, T_SWAP - 0.2, "「它」指的是谁？你一眼就知道。", "Who is 'it'? You know at a glance."),
                     (T_SWAP, T_NAME - 0.2, "换一个字，「它」就换了对象。", "Change one word, and 'it' points elsewhere."),
                     (T_NAME, 10.4, "你刚才做的这件事，就叫「注意力」。", "What you just did is called attention.")])
    caps("ATTENTION", 26, 0.9).draw(F, CX, 1296, GOLD, 0.9 * smooth(ramp(t, T_NAME + 0.3, 0.8)), align="center")
    fade = min(1.0, t / 0.5) * min(1.0, (DUR - t) / 0.7)
    F *= max(fade, 0.0)
    return finalize(F, vig, grain[fi % len(grain)] * fade)


_state = {}


def _render(fi):
    if not _state:
        bg, vig, tw = make_background()
        _state.update(bg=bg, vig=vig, tw=tw, grain=make_grain())
    s = _state
    return frame(fi, s["bg"], s["vig"], s["tw"], s["grain"]).tobytes()


def score(path):
    n = int(DUR * A.SR)
    pad, pno, bel, fx, tick = (A.stereo(n) for _ in range(5))
    for t0, hold, notes in ((0.0, 5.2, [53, 57, 60, 64, 67]), (5.0, 3.0, [50, 57, 60, 64, 65]),
                            (T_NAME, 2.4, [46, 53, 57, 60, 62])):
        for m in notes:
            A.place(pad, A.pad_note(m, hold, attack=1.2, release=2.0), t0)
    A.place(bel, A.bell(74, 0.25, ratio=2.0, dur=3.0), T_TABLE_IN + 0.2, pan=0.3)
    A.place(bel, A.bell(79, 0.25, ratio=2.0, dur=3.0), T_CAT_IN + 0.3, pan=-0.4)
    A.place(fx, A.thud(0.35), T_HOP + 0.5)
    A.place(pno, A.piano(65, 0.28), T_LIE + 0.1)
    A.place(pno, A.piano(60, 0.24), T_LIE + 0.5)
    A.place(bel, A.bell(81, 0.5), T_IT, pan=0.1)
    for k, m in enumerate((69, 72, 76, 79, 81, 84, 88)):
        A.place(bel, A.bell(m, 0.22, ratio=2.0, dur=3.0), T_FAN + 0.11 * k, pan=-0.6 + 0.2 * k)
    A.place(pno, A.piano(41, 0.4), T_FAN + 0.9)
    A.place(pno, A.piano(65, 0.35), T_FAN + 0.95)
    A.place(fx, A.riser(0.6, 0.35), T_SWAP - 0.55)
    A.place(fx, A.riser(1.0, 0.25), T_SWAP)
    A.place(bel, A.bell(76, 0.45, ratio=2.0), T_SWAP + 0.05, pan=-0.2)
    for k, m in enumerate((86, 84, 81, 79, 76)):
        A.place(bel, A.bell(m, 0.18, ratio=2.0, dur=2.5), T_RE + 0.12 * k, pan=0.5 - 0.2 * k)
    A.place(pno, A.piano(38, 0.4), T_RE + 1.0)
    A.place(pno, A.piano(62, 0.32), T_RE + 1.05)
    A.place(fx, A.boom(0.6), T_NAME)
    for k, m in enumerate((46, 58, 62, 65, 69, 74)):
        A.place(pno, A.piano(m, 0.38), T_NAME + 0.08 * k)
    A.place(bel, A.bell(86, 0.4), T_NAME + 0.6, pan=0.3)
    ir = A.build_ir(3.5)
    pad = A.hp(A.lp(pad / (np.abs(pad).max() + 1e-9) * 0.35, 3000), 90)
    pno = pno / (np.abs(pno).max() + 1e-9) * 0.5
    bel = bel / (np.abs(bel).max() + 1e-9) * 0.4
    fx = fx / (np.abs(fx).max() + 1e-9) * 0.45
    tick = tick / (np.abs(tick).max() + 1e-9) * 0.12
    dry = pad + pno + bel + fx + tick
    mix_ = A.hp(dry + A.reverb(pad * 0.3 + pno * 0.5 + bel * 0.8 + tick * 0.6, ir) * 0.5, 28)
    env = np.ones(n)
    fi_, fo = int(0.3 * A.SR), int(0.9 * A.SR)
    env[:fi_] = A.rc(fi_)
    env[-fo:] = 1 - A.rc(fo)
    return A.master(mix_ * env[:, None], path)


def main():
    build = ROOT / "build"
    build.mkdir(exist_ok=True)
    lufs, peak = score(build / "teaser.wav")
    print(f"audio {lufs:.1f} LUFS, peak {peak:.1f} dBFS")
    cmd = ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
           "-r", str(FPS), "-i", "-", "-i", str(build / "teaser.wav"), "-map", "0:v", "-map", "1:a",
           "-c:v", "libx264", "-preset", "slow", "-crf", "16", "-pix_fmt", "yuv420p", "-profile:v", "high",
           "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", "-shortest", str(ROOT / "attention-teaser.mp4")]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    with Pool(4) as pool:
        for buf in pool.imap(_render, range(int(DUR * FPS)), chunksize=8):
            p.stdin.write(buf)
    p.stdin.close()
    p.wait()
    print("wrote attention-teaser.mp4")


if __name__ == "__main__":
    main()
