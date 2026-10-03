"""10-second teaser for 注意力 · Attention Is All You Need (9:16, Douyin), paper-and-ink edition.

The viewer performs attention before being told what it is: in
"小猫没有跳上桌子，因为它太累了。" everybody knows instantly who 「它」 is.
A reader's pencil links 「它」 to every word, and one vermilion brush stroke
lands on 小猫.  Change one word (累 -> 高) and the stroke moves to 桌子.
That is what a Transformer computes (the percentages here are illustrative).

  python3 teaser.py   -> attention-teaser.mp4
"""

import subprocess
from multiprocessing import Pool

import numpy as np

import paper as P
import synth as A
from style import E, H, W, ROOT, ease_in_out, ease_out, ramp, serif, smooth, statement, window

FPS, DUR = 30, 10.0

SIZE, TRACK = 84, 0.12
L1, L2 = "小猫没有跳上桌子，", "因为它太累了。"
Y1, Y2 = 700, 980
X0 = 520                                   # sentence centre (a touch left of the button column)
TOKENS = [(0, 0, 2, "小猫"), (0, 2, 4, "没有"), (0, 4, 6, "跳上"), (0, 6, 8, "桌子"),
          (1, 0, 2, "因为"), (1, 2, 3, "它"), (1, 3, 4, "太"), (1, 4, 5, "累"), (1, 5, 6, "了")]
IT, CAT, TABLE = 5, 0, 3
WA = np.array([0.58, 0.04, 0.05, 0.09, 0.04, 0.06, 0.04, 0.07, 0.03])   # ...因为它太累了
WB = np.array([0.09, 0.03, 0.07, 0.56, 0.03, 0.06, 0.05, 0.08, 0.03])   # ...因为它太高了
T1, T2 = 0.25, 1.35
T_IT, T_FAN, T_MARK_A, T_SWAP, T_RE, T_MARK_B, T_NAME = 2.6, 3.2, 4.0, 5.0, 5.6, 6.2, 7.8


def layout():
    out = []
    for txt, y in ((L1, Y1), (L2, Y2)):
        tx = serif(txt, SIZE, 400, TRACK)
        x0 = X0 - tx.width / 2
        out.append(([(x0 + dx, x0 + dx + m.shape[1]) for m, dx, dy in tx.items], y))
    tok = []
    for line, a, b, _ in TOKENS:
        boxes, y = out[line]
        tok.append(((boxes[a][0] + boxes[b - 1][1]) / 2, y, line, boxes[b - 1][1] - boxes[a][0]))
    return tok


TOK = layout()


def bezier(p0, p1, p2, n=60):
    u = np.linspace(0, 1, n)[:, None]
    return (1 - u) ** 2 * p0 + 2 * (1 - u) * u * p1 + u ** 2 * p2


def arc(i):
    """From 「它」 to token i: up across the gap for line 1, a loop under line 2 for its neighbours."""
    sx, sy, _, _ = TOK[IT]
    tx, ty, line, _ = TOK[i]
    if line == 0:
        p0 = np.array([sx, sy - SIZE * 0.98])
        p2 = np.array([tx, ty + SIZE * 0.26])
        p1 = np.array([(sx + tx) / 2, (p0[1] + p2[1]) / 2 + 40])
    else:
        p0 = np.array([sx, sy + SIZE * 0.3])
        p2 = np.array([tx, ty + SIZE * 0.3])
        p1 = np.array([(sx + tx) / 2, sy + SIZE * 0.3 + 70 + 0.35 * abs(tx - sx)])
    return bezier(p0, p1, p2)


ARCS = [arc(i) for i in range(len(TOKENS))]


def weights(t):
    uni = np.full(len(TOKENS), 1 / len(TOKENS))
    w = uni + (WA - uni) * ease_in_out(ramp(t, T_FAN + 0.3, 0.9))
    return w + (WB - w) * ease_in_out(ramp(t, T_RE, 1.0))


def ring_around(i, seed):
    tx, ty, _, wd = TOK[i]
    return P.ring(tx, ty - SIZE * 0.36, wd / 2 + SIZE * 0.1, SIZE * 0.66, seed=seed)


RING_IT = ring_around(IT, 1)
RING_CAT = ring_around(CAT, 2)
RING_TABLE = ring_around(TABLE, 3)


def mark(F, t, i, t0, ring_pts, pct, t_off=None):
    """Brush stroke from 「它」 to the answer, a loop round it, and a margin note."""
    a = 1.0 if t_off is None else 1 - smooth(ramp(t, t_off, 0.5))
    if t < t0 or a <= 0.003:
        return
    P.brush(F, ARCS[i], 9, a=a, upto=ease_in_out(ramp(t, t0, 0.5)), seed=i)
    P.brush(F, ring_pts, 6, a=a, upto=ease_out(ramp(t, t0 + 0.4, 0.45)), seed=i + 10)
    tx, ty, _, wd = TOK[i]
    P.note(F, pct, tx + wd / 2 + 36, ty - SIZE - 6, a * smooth(ramp(t, t0 + 0.7, 0.4)))


def line2(F, t):
    """Second line, typed in ink, with one character that changes its mind."""
    a_txt = serif(L2, SIZE, 400, TRACK)
    b_txt = serif(L2.replace("累", "高"), SIZE, 400, TRACK)
    x0 = X0 - a_txt.width / 2
    sw = smooth(ramp(t, T_SWAP, 0.55))
    for i, (m, dx, dy) in enumerate(a_txt.items):
        u = ease_out(ramp(t, T2 + i / 7.5, 0.25))
        if u <= 0:
            continue
        off = (1 - u) * 6
        if i == 4 and sw > 0:
            E.blend(F, m, x0 + dx, Y2 + dy + off - 16 * sw, P.INK, u * (1 - sw))
            mb, dxb, dyb = b_txt.items[i]
            E.blend(F, mb, x0 + dxb, Y2 + dyb + 12 * (1 - sw), P.INK, sw)
        else:
            E.blend(F, m, x0 + dx, Y2 + dy + off, P.INK, u)


def frame(fi, paper, grain):
    t = fi / FPS
    F = paper.copy()
    P.chapter_mark(F, window(t, 0.3, 10, 1.0, 0.6), "00", "序", "PROLOGUE")
    w = weights(t)
    grow = ease_out(ramp(t, T_FAN, 0.8))
    if grow > 0:                                     # a reader's pencil, from 「它」 to every word
        for i, pts in enumerate(ARCS):
            if i != IT:
                P.pencil(F, pts, 0.45 + 0.55 * w[i] / w.max(), upto=grow)
    statement(F, L1, X0, Y1, t, T1, None, size=SIZE, tracking=TRACK, cps=7.5, fade_in=0.25, color=P.INK)
    line2(F, t)
    P.brush(F, RING_IT, 6, upto=ease_out(ramp(t, T_IT, 0.45)), seed=7)
    mark(F, t, CAT, T_MARK_A, RING_CAT, "58%", t_off=T_SWAP)
    mark(F, t, TABLE, T_MARK_B, RING_TABLE, "56%")
    P.seal(F, 846, 1040, "注意力", 40, smooth(ramp(t, T_NAME + 0.2, 0.15)))
    P.subtitles(F, t, [(T_IT, T_SWAP - 0.2, "「它」指的是谁？你一眼就知道。", "Who is 'it'? You know at a glance."),
                       (T_SWAP, T_NAME - 0.2, "换一个字，「它」就换了对象。", "Change one word, and 'it' points elsewhere."),
                       (T_NAME, 10.4, "你刚才做的这件事，就叫「注意力」。", "What you just did is called attention.")])
    fade = min(1.0, (DUR - t) / 0.7)
    if fade < 1:                                     # the marks dissolve back into blank paper
        F[:] = paper + (F - paper) * max(fade, 0.0)
    F *= 0.2 + 0.8 * smooth(min(1.0, t / 0.6))      # open from dark onto the page
    return P.finalize(F, grain[fi % len(grain)])


_state = {}


def _render(fi):
    if not _state:
        _state.update(paper=P.make_paper(), grain=P.make_grain())
    return frame(fi, _state["paper"], _state["grain"]).tobytes()


def score(path):
    n = int(DUR * A.SR)
    pad, pno, plk, fx = (A.stereo(n) for _ in range(4))
    for t0, hold, notes in ((0.0, 5.3, [50, 57, 62, 64, 69]), (5.0, 3.1, [47, 54, 57, 62, 64]),
                            (T_NAME, 2.4, [43, 50, 54, 57, 62])):
        for m in notes:
            A.place(pad, A.pad_note(m, hold, attack=1.4, release=2.0), t0)

    def pl(m, t, vel=0.5, pan=0.0):
        A.place(plk, A.pluck(m, vel), t, pan=pan)

    pl(69, 0.30, 0.35, -0.2)
    pl(74, 1.40, 0.35, 0.2)
    A.place(fx, A.brush_swish(0.45, 0.35), T_IT, pan=-0.1)
    pl(76, T_IT + 0.05, 0.45)
    for k in range(len(TOKENS) - 1):
        A.place(fx, A.pencil_scratch(0.12, 0.08), T_FAN + 0.06 * k, pan=-0.5 + 0.12 * k)
    for k, m in enumerate((62, 66, 69)):
        pl(m, T_FAN + 0.18 * k, 0.3, -0.3 + 0.3 * k)
    A.place(fx, A.brush_swish(0.5, 0.45), T_MARK_A, pan=-0.3)
    pl(81, T_MARK_A + 0.45, 0.5, -0.2)
    A.place(pno, A.piano(50, 0.4), T_MARK_A + 0.45)
    A.place(fx, A.brush_swish(0.4, 0.3), T_MARK_A + 0.4, pan=-0.4)
    for k, m in enumerate((78, 76, 74)):
        pl(m, T_SWAP + 0.15 * k, 0.32, 0.2)
    A.place(fx, A.brush_swish(0.5, 0.45), T_MARK_B, pan=0.3)
    pl(78, T_MARK_B + 0.45, 0.5, 0.3)
    A.place(pno, A.piano(47, 0.4), T_MARK_B + 0.45)
    A.place(fx, A.brush_swish(0.4, 0.3), T_MARK_B + 0.4, pan=0.4)
    A.place(fx, A.stamp(0.8), T_NAME + 0.2)
    for k, m in enumerate((43, 55, 62, 67, 71, 74)):
        pl(m, T_NAME + 0.3 + 0.09 * k, 0.42, -0.4 + 0.16 * k)
    A.place(pno, A.piano(43, 0.38), T_NAME + 0.3)
    ir = A.build_ir(3.5)
    pad = A.hp(A.lp(pad / (np.abs(pad).max() + 1e-9) * 0.28, 1800), 90)
    pno = pno / (np.abs(pno).max() + 1e-9) * 0.45
    plk = plk / (np.abs(plk).max() + 1e-9) * 0.55
    fx = fx / (np.abs(fx).max() + 1e-9) * 0.4
    dry = pad + pno + plk + fx
    mix_ = A.hp(dry + A.reverb(pad * 0.3 + pno * 0.5 + plk * 0.6 + fx * 0.2, ir) * 0.45, 28)
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
