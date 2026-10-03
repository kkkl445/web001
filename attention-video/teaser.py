"""10-second teaser for 注意力 · Attention Is All You Need (9:16, Douyin).

The viewer performs attention before being told what it is: in
"小猫没有跳上桌子，因为它太累了。" everybody knows instantly who 「它」 is.
Change one word (累 -> 高) and 「它」 jumps to the table.  The fan of lines
drawn out of 「它」 is exactly what a Transformer computes: how much each
word should look at every other word (weights illustrative).

  python3 teaser.py   -> attention-teaser.mp4
"""

import math
import subprocess
from multiprocessing import Pool

import numpy as np

import synth as A
from style import (CX, DIM, GOLD, IVORY, H, W, E, add_sprite, caps, chapter_mark, ease_in_out, ease_out,
                   finalize, glow_poly, make_background, make_grain, mix, orb, ramp, serif, smooth, statement,
                   subtitles, twinkle, window, ROOT, T)

FPS, DUR = 30, 10.0
STEEL = mix(np.array([0.45, 0.84, 1.0], np.float32), IVORY, 0.45)

SIZE, TRACK = 84, 0.12
L1, L2 = "小猫没有跳上桌子，", "因为它太累了。"
Y1, Y2 = 700, 980
X0 = 520                                   # sentence centre (a touch left of the button column)
TOKENS = [(0, 0, 2, "小猫"), (0, 2, 4, "没有"), (0, 4, 6, "跳上"), (0, 6, 8, "桌子"),
          (1, 0, 2, "因为"), (1, 2, 3, "它"), (1, 3, 4, "太"), (1, 4, 5, "累"), (1, 5, 6, "了")]
IT = 5
WA = np.array([0.58, 0.04, 0.05, 0.09, 0.04, 0.06, 0.04, 0.07, 0.03])   # ...因为它太累了
WB = np.array([0.09, 0.03, 0.07, 0.56, 0.03, 0.06, 0.05, 0.08, 0.03])   # ...因为它太高了
T1, T2 = 0.25, 1.35                         # typing starts
T_IT, T_FAN, T_SWAP, T_RE, T_NAME = 2.6, 3.2, 5.0, 5.6, 7.8


def layout():
    """Glyph boxes for every token: (x_left, x_right) on its line."""
    out = []
    for line, txt, y in ((0, L1, Y1), (1, L2, Y2)):
        tx = serif(txt, SIZE, 400, TRACK)
        x0 = X0 - tx.width / 2
        boxes = [(x0 + dx, x0 + dx + m.shape[1]) for m, dx, dy in tx.items]
        out.append((boxes, y))
    tok = []
    for line, a, b, _ in TOKENS:
        boxes, y = out[line]
        tok.append(((boxes[a][0] + boxes[b - 1][1]) / 2, y, line))
    return tok, out


TOK, LINES = layout()


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
        p1 = np.array([(sx + tx) / 2, sy + SIZE * 0.24 + 70 + 0.35 * abs(tx - sx)])
    return bezier(p0, p1, p2)


ARCS = [arc(i) for i in range(len(TOKENS))]


def weights(t):
    uni = np.full(len(TOKENS), 1 / len(TOKENS))
    w = uni + (WA - uni) * ease_in_out(ramp(t, T_FAN + 0.3, 0.9))
    return w + (WB - w) * ease_in_out(ramp(t, T_RE, 1.0))


def frame(fi, bg, vig, tw, grain):
    t = fi / FPS
    F = bg.copy()
    twinkle(F, tw, t)
    chapter_mark(F, window(t, 0.3, 10, 1.0, 0.6), "00", "序", "PROLOGUE")
    w = weights(t)
    grow = ease_out(ramp(t, T_FAN, 0.9))
    # the fan of attention out of 「它」
    if grow > 0:
        top = int(np.argmax(w))
        for i, pts in enumerate(ARCS):
            if i == IT:
                continue
            k = max(2, int(len(pts) * grow))
            rel = w[i] / w.max()
            col = mix(STEEL, GOLD, smooth(rel))
            glow_poly(F, pts[:k], col, (0.26 + 0.74 * rel ** 1.2) * grow, th=1 if rel < 0.4 else 2 if rel < 0.8 else 3,
                      glow=0.4 + 1.2 * rel, sigma=4 + 4 * rel)
            if k < len(pts):
                orb(F, pts[k - 1][0], pts[k - 1][1], 4, col, 0.8 * grow)
        # the winner glows, with its share of attention
        tx, ty, line = TOK[top]
        hl = smooth(ramp(t, T_FAN + 0.8, 0.5))
        add_sprite(F, tx, ty - SIZE * 0.35, 70, GOLD, 0.22 * hl)
        pct = f"{round(100 * w[top])}%"
        T(pct, "stix", 42, None, per_char=False).draw(F, tx, ty - SIZE - 24, GOLD, hl, align="center")
    # the sentence, typed; 「它」 lights up; 累 turns into 高
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
    na = smooth(ramp(t, T_NAME + 0.3, 0.8))
    caps("ATTENTION", 26, 0.9).draw(F, CX, 1236, GOLD, 0.9 * na, align="center")
    fade = min(1.0, t / 0.5) * min(1.0, (DUR - t) / 0.7)
    F *= max(fade, 0.0)
    return finalize(F, vig, grain[fi % len(grain)] * fade)


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
    for line, (txt, t0) in enumerate(((L1, T1), (L2, T2))):
        for k in range(len(txt)):
            A.place(tick, A.blip(int(A.rng.choice([84, 86, 88, 91])), 0.12), t0 + k / 7.5, pan=-0.3 + 0.08 * k)
    A.place(bel, A.bell(81, 0.5), T_IT, pan=0.1)
    for k, m in enumerate((69, 72, 76, 79, 81, 84, 88)):
        A.place(bel, A.bell(m, 0.22, ratio=2.0, dur=3.0), T_FAN + 0.11 * k, pan=-0.6 + 0.2 * k)
    A.place(pno, A.piano(41, 0.4), T_FAN + 0.9)
    A.place(pno, A.piano(65, 0.35), T_FAN + 0.95)
    A.place(fx, A.riser(0.6, 0.35), T_SWAP - 0.55)
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
