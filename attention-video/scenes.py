"""注意力 · Attention Is All You Need.  draw(F, t_local, g_global, frame).

One spine: you already do attention, instantly.  Machines used to read one
word at a time and forget the beginning; the Transformer lets every word look
at every other word at once.  Pictures appear only where they help: the cat
and the table, a whispered message passing along a row of people, a reader at
a bookshelf, a cat chasing a dog, someone looking up at the stars."""

import math

import cv2
import numpy as np

import figures2d as G
import timeline as TL
from style import (AMBER, CX, CY, DIM, GOLD, IVORY, H, W, E, R, T, add_sprite, blend, caps, comet, ease_in_out,
                   ease_out, glow_poly, hairline, italic, lerp, mix, orb, ramp, serif, smooth, statement, subtitles,
                   window)

STEEL = G.STEEL
WARM = np.array([1.0, 0.93, 0.8], np.float32)
VIOLET = E.rgb("a99bff")
TEAL = E.rgb("6fe3c8")
ROSE = E.rgb("ff8fb1")
SKY = E.rgb("7fa8ff")
HEAD_COLS = [GOLD, E.rgb("72d6ff"), AMBER, VIOLET, TEAL, ROSE, IVORY, SKY]


# -------------------------------------------------------------- sentence ---

class Sentence:
    """A sentence laid out on lines; tokens are spans of characters; arcs connect tokens."""

    def __init__(self, lines, tokens, size, ys, cx=520, tracking=0.12):
        self.lines, self.tokens, self.size, self.ys = lines, tokens, size, ys
        self.txt = [serif(l, size, 400, tracking) for l in lines]
        self.x0 = [cx - tx.width / 2 for tx in self.txt]
        self.tracking = tracking
        self.char_tok = {}
        self.centers = []
        for k, (ln, a, b, _) in enumerate(tokens):
            for c in range(a, b):
                self.char_tok[(ln, c)] = k
            items = self.txt[ln].items
            xa = self.x0[ln] + items[a][1]
            xb = self.x0[ln] + items[b - 1][1] + items[b - 1][0].shape[1]
            self.centers.append(((xa + xb) / 2, ys[ln], ln, xb - xa))

    def draw(self, F, t=1e9, t0=0.0, cps=8.0, a=1.0, tok_alpha=None, tok_color=None, swap=None):
        """swap = (line, char, new_char, u) crossfades one character."""
        n = 0
        for ln, tx in enumerate(self.txt):
            for i, (m, dx, dy) in enumerate(tx.items):
                u = ease_out(ramp(t, t0 + n / cps, 0.25))
                n += 1
                if u <= 0:
                    continue
                k = self.char_tok.get((ln, i))
                al = a * u * (tok_alpha[k] if (tok_alpha is not None and k is not None) else 1.0)
                col = tok_color[k] if (tok_color is not None and k is not None) else IVORY
                x, y = self.x0[ln] + dx, self.ys[ln] + dy + (1 - u) * 6
                if swap is not None and swap[0] == ln and swap[1] == i and swap[3] > 0:
                    sw = swap[3]
                    blend(F, m, x, y - 18 * sw, col, al * (1 - sw))
                    alt = serif(self.lines[ln][:i] + swap[2], self.size, 400, self.tracking).items[i]
                    blend(F, alt[0], self.x0[ln] + alt[1], self.ys[ln] + alt[2] + 14 * (1 - sw),
                          mix(GOLD, col, sw ** 2), a * sw)
                else:
                    blend(F, m, x, y, col, al)

    def top(self, k, pad=0.92):
        x, y, _, _ = self.centers[k]
        return np.array([x, y - self.size * pad])

    def bottom(self, k, pad=0.24):
        x, y, _, _ = self.centers[k]
        return np.array([x, y + self.size * pad])

    def arc(self, i, j, n=48, lift=1.0):
        xi, yi, li, _ = self.centers[i]
        xj, yj, lj, _ = self.centers[j]
        if li == lj:
            dx = abs(xj - xi)
            if li == 0 and len(self.lines) > 1:
                p0, p2 = self.top(i), self.top(j)
                p1 = np.array([(xi + xj) / 2, p0[1] - (26 + 0.22 * dx) * lift])
            else:
                p0, p2 = self.bottom(i, 0.3), self.bottom(j, 0.3)
                p1 = np.array([(xi + xj) / 2, p0[1] + (30 + 0.16 * dx) * lift])
        else:
            up, lo = (i, j) if li < lj else (j, i)
            a_, b_ = self.bottom(up, 0.26), self.top(lo, 0.98)
            p0, p2 = (b_, a_) if li > lj else (a_, b_)
            p1 = (p0 + p2) / 2 + np.array([0, 40])
        u = np.linspace(0, 1, n)[:, None]
        return (1 - u) ** 2 * p0 + 2 * (1 - u) * u * p1 + u ** 2 * p2


def fan(F, S, src, w, grow, label=True, a=1.0, hl=1.0, colors=None):
    """Attention out of one token: thicker and golder where the weight is higher."""
    if grow <= 0 or a <= 0.003:
        return
    rel = w / w.max()
    for i in range(len(S.tokens)):
        if i == src:
            continue
        pts = S.arc(src, i)
        k = max(2, int(len(pts) * grow))
        col = colors if colors is not None else mix(STEEL, GOLD, smooth(rel[i]))
        glow_poly(F, pts[:k], col, (0.26 + 0.74 * rel[i] ** 1.2) * grow * a,
                  th=1 if rel[i] < 0.4 else 2 if rel[i] < 0.8 else 3, glow=0.4 + 1.2 * rel[i], sigma=4 + 4 * rel[i])
        if k < len(pts):
            orb(F, pts[k - 1][0], pts[k - 1][1], 4, col, 0.8 * grow * a)
    if label:
        top = int(np.argmax(w))
        tx, ty, _, _ = S.centers[top]
        add_sprite(F, tx, ty - S.size * 0.35, 70, GOLD, 0.2 * hl * a)
        T(f"{round(100 * w[top])}%", "stix", 36, None, per_char=False).draw(F, tx, ty - S.size - 16, GOLD, hl * a,
                                                                          align="center")


def link(F, a, b, alpha, col=GOLD):
    """Dotted thread from a word to the thing it names in the picture."""
    if alpha <= 0.01:
        return
    n = int(np.hypot(*(b - a)) / 14)
    for k in range(1, n):
        p = a + (b - a) * k / n
        add_sprite(F, p[0], p[1], 1.6, col, 0.55 * alpha)


def floor_dots(F, y, a, x0=150, x1=900, n=32):
    if a > 0.003:
        for x in np.linspace(x0, x1, n):
            orb(F, x, y + 4, 1.2, STEEL, 0.22 * a)


SENT_LINES = ("小猫没有跳上桌子，", "因为它太累了。")
SENT_TOKENS = [(0, 0, 2, "小猫"), (0, 2, 4, "没有"), (0, 4, 6, "跳上"), (0, 6, 8, "桌子"),
               (1, 0, 2, "因为"), (1, 2, 3, "它"), (1, 3, 4, "太"), (1, 4, 5, "累"), (1, 5, 6, "了")]
IT, CAT, TABLE = 5, 0, 3


# ------------------------------------------------------------ prologue ---

P_S = Sentence(SENT_LINES, SENT_TOKENS, 84, (930, 1150))
PWA = np.array([0.58, 0.04, 0.05, 0.09, 0.04, 0.06, 0.04, 0.07, 0.03])   # ...因为它太累了
PWB = np.array([0.09, 0.03, 0.07, 0.56, 0.03, 0.06, 0.05, 0.08, 0.03])   # ...因为它太高了
GROUND = 760


def _p_weights(t):
    uni = np.full(len(SENT_TOKENS), 1 / len(SENT_TOKENS))
    w = uni + (PWA - uni) * ease_in_out(ramp(t, TL.P_FAN + 0.3, 0.9))
    return w + (PWB - w) * ease_in_out(ramp(t, TL.P_RE, 1.0))


def _burst(F, q, a, cy):
    P0 = np.random.default_rng(3).normal(0, 1, (140, 2))
    if q > 3.0:
        return
    sp = 1 - math.exp(-q * 2.2)
    for vx, vy in P0:
        add_sprite(F, CX + vx * 300 * sp, cy + vy * 160 * sp, 1.2, GOLD, 0.6 * a * max(0.0, 1 - q / 3.0))


def prologue(F, t, g, fi):
    A = 1 - smooth(ramp(t, 9.9, 1.0))
    if A > 0.003:
        w = _p_weights(t)
        hl = smooth(ramp(t, TL.P_FAN + 0.8, 0.5))
        rel = w / w.max()
        floor_dots(F, GROUND, smooth(ramp(t, 0.2, 1.0)) * A)
        h = 1.5 + 1.6 * ease_in_out(ramp(t, TL.P_SWAP, 1.0))
        tp, te = G.table(470, GROUND, 140, h)
        down = ease_in_out(ramp(t, TL.P_LIE, 0.8)) * (1 - ease_in_out(ramp(t, TL.P_SWAP + 0.1, 0.8)))
        hop = math.sin(math.pi * ramp(t, TL.P_HOP, 0.55)) if TL.P_HOP <= t <= TL.P_HOP + 0.55 else 0.0
        cp, ce, ceye = G.cat(262, GROUND, 140, down=down, hop=hop, look_up=ease_in_out(ramp(t, TL.P_SWAP + 0.5, 0.6)))
        G.draw(F, tp, te, t, 0.35, 1.1, hl * smooth((rel[TABLE] - 0.3) / 0.7), seed=2, a=A)
        G.draw(F, cp, ce, t, 0.6, 1.3, hl * smooth((rel[CAT] - 0.3) / 0.7), seed=1, extra=ceye, a=A)
        for i, target in ((CAT, cp[12]), (TABLE, (tp[0] + tp[1]) / 2)):
            tx, ty, _, _ = P_S.centers[i]
            link(F, np.array([tx, ty - 84 - 66]), np.asarray(target) + np.array([0, 18]),
                 hl * smooth((rel[i] - 0.5) / 0.5) * A)
        fan(F, P_S, IT, w, ease_out(ramp(t, TL.P_FAN, 0.9)), a=A, hl=hl)
        cols = [IVORY] * len(SENT_TOKENS)
        if t >= TL.P_IT:
            cols[IT] = GOLD
        P_S.draw(F, t, TL.P_T1, cps=7.5, a=A, tok_color=cols, swap=(1, 4, "高", smooth(ramp(t, TL.P_SWAP, 0.55))))
        if t >= TL.P_IT:
            sx, sy, _, _ = P_S.centers[IT]
            p = math.exp(-(t - TL.P_IT) * 2.2)
            add_sprite(F, sx, sy - 84 * 0.35, 46 + 30 * p, GOLD, (0.25 + 0.5 * p) * A)
        if TL.P_SWAP <= t < TL.P_SWAP + 0.9:
            sx, sy, _, _ = P_S.centers[7]
            q = (t - TL.P_SWAP) / 0.9
            add_sprite(F, sx, sy - 84 * 0.35, 40 + 60 * q, GOLD, 0.7 * math.sin(math.pi * q))
        caps("ATTENTION", 26, 0.9).draw(F, CX, 1296, GOLD, 0.9 * smooth(ramp(t, TL.P_NAME + 0.3, 0.8)) * A,
                                        align="center")
    subtitles(F, t, [(TL.P_IT, TL.P_SWAP - 0.2, "「它」指的是谁？你一眼就知道。", "Who is 'it'? You know at a glance."),
                     (TL.P_SWAP, TL.P_NAME - 0.2, "换一个字，「它」就换了对象。",
                      "Change one word, and 'it' points elsewhere."),
                     (TL.P_NAME, 10.6, "你刚才做的这件事，就叫「注意力」。", "What you just did is called attention.")])
    # the turn: someone taught machines to do it
    statement(F, "2017 年，八位研究者，", CX, 760, t, 10.9, 15.1, size=54, cps=10)
    statement(F, "让机器只靠这件事，读懂语言。", CX, 850, t, 11.7, 15.1, size=54, cps=10, gold={3, 4, 5, 6, 7})
    # title
    if t > 14.7:
        cy = 760
        o = smooth(ramp(t, 14.7, 0.7)) * (1 - smooth(ramp(t, 15.4, 0.25)))
        orb(F, CX, cy, 9 + 8 * o, GOLD, 1.3 * o)
        if t >= 15.4:
            q = t - 15.4
            fade = 1 - smooth(ramp(t, 22.7, 1.2))
            hx = CX + 640 * ease_out(min(q / 2.6, 1.0))
            ca = 0.85 * fade * (1 - smooth(ramp(q, 2.2, 0.5)))
            comet(F, hx, cy + 150, 1, 260, GOLD, ca)
            comet(F, 2 * CX - hx, cy + 150, -1, 260, GOLD, ca)
            _burst(F, q, fade, cy)
            for k, (line, y) in enumerate((("Attention", 690), ("Is All You Need", 806))):
                tx = T(line, "corm", 104, 600, tracking=0.02)
                al = fade * ease_out(ramp(t, 15.5 + 0.5 * k, 1.0))
                tx.draw(F, CX, y + (1 - al) * 10, IVORY, al, align="center", glow=0.25, glow_color=GOLD,
                        glow_sigma=14)
            fa = fade * smooth(ramp(t, 17.0, 0.9))
            serif("注意力，就是你所需要的一切", 40, 500, 0.2).draw(F, CX, 900, GOLD, fa, align="center")
            hairline(F, CX - 80 * fa, 940, CX + 80 * fa, GOLD, 0.6 * fa)
            caps("VASWANI ET AL.   ·   2017", 18, 0.42).draw(F, CX, 980, DIM, fa, align="center")


# ------------------------------------------------------------ sequential ---

SQ_S = Sentence(("小猫没有跳上桌子，", "因为它太累了。", "它整整睡了一个下午。"),
                SENT_TOKENS + [(2, 0, 1, "它"), (2, 1, 3, "整整"), (2, 3, 4, "睡"), (2, 4, 5, "了"), (2, 5, 8, "一个下"),
                               (2, 8, 9, "午")], 66, (1000, 1110, 1220))
WHISPER_X = [170, 400, 630, 860]


def _read_pos(t):
    """Index of the token being read (fractional), one word at a time."""
    if t < TL.SEQ_READ0:
        return -1.0
    n = (t - TL.SEQ_READ0) / TL.SEQ_READ_DT
    if t > 22.0:                                              # the sentence grows; reading crawls on
        n = (22.0 - TL.SEQ_READ0) / TL.SEQ_READ_DT + (t - 22.0) / 1.3
    return min(n, len(SQ_S.tokens) - 1 + 0.999)


def sequential(F, t, g, fi):
    A = window(t, 0.0, 33.6, 0.8, 0.9)
    # the telephone game: profiles in a row, the message whispered from mouth to ear
    ys = 620
    pa = A * (1 - 0.55 * smooth(ramp(t, 13.5, 1.5)))
    profs = [G.profile(x, ys, 195, hair=h) for x, h in zip(WHISPER_X, ("bun", "short", "long", "short"))]
    for k, pr in enumerate(profs):
        G.draw_profile(F, pr, t, 0.3 + 0.25 * k, 1.4, glow=0.0, a=pa, fill=0.05)
    caps("A WHISPER, PASSED ALONG", 14, 0.5).draw(F, CX, 420, DIM, 0.8 * window(t, 2.0, 13.0, 1.0, 1.0) * A,
                                                   align="center")
    msg = serif("小猫", 34, 500, 0.15)
    for k in range(len(profs) - 1):
        t0 = TL.SEQ_PASS0 + k * TL.SEQ_PASS_DT
        q = ramp(t, t0, 1.1)
        held = t0 - TL.SEQ_PASS_DT * 0.6 <= t < t0 if k == 0 else False
        if 0 < q < 1 or held or (k == len(profs) - 2 and t >= t0 + 1.1):
            a_pt, b_pt = profs[k]["mouth"] + [8, 0], profs[k + 1]["ear_pt"]
            if held:
                p = a_pt
            elif q >= 1:
                p = b_pt
            else:
                mid = (a_pt + b_pt) / 2 + np.array([0, -70])
                u = ease_in_out(q)
                p = (1 - u) ** 2 * a_pt + 2 * (1 - u) * u * mid + u ** 2 * b_pt
            strength = 0.62 ** (k + (q if q < 1 else 1))
            col = mix(STEEL, GOLD, strength)
            fade_out = 1 - smooth(ramp(t, 13.0, 1.0))
            orb(F, p[0], p[1], 6 + 6 * (1 - strength), col, (0.4 + 0.9 * strength) * A * fade_out)
            # the word it carries blurs a little more with every retelling
            sig = 0.5 + 3.2 * (1 - strength)
            pad = int(sig * 3) + 2
            for m, dx, dy in msg.items:
                mm = np.zeros((m.shape[0] + 2 * pad, m.shape[1] + 2 * pad), np.float32)
                mm[pad:-pad, pad:-pad] = m / 255.0
                mm = cv2.GaussianBlur(mm, (0, 0), sig)
                blend(F, mm, p[0] - msg.width / 2 + dx - pad, p[1] - 30 + dy - pad, col,
                      (0.35 + 0.65 * strength) * A * fade_out)
    # the sentence read one word at a time, the start fading from memory
    pos = _read_pos(t)
    n = len(SQ_S.tokens)
    alphas = np.full(n, 0.22)
    cols = [IVORY] * n
    for k in range(n):
        if pos >= k:
            age = pos - k
            alphas[k] = 0.16 + 0.84 * math.exp(-age / 2.6)
    cur = int(pos) if pos >= 0 else -1
    if cur >= 0:
        cols[cur] = GOLD
        alphas[cur] = 1.0
    third = smooth(ramp(t, 21.4, 0.8))
    sa = A * smooth(ramp(t, 7.0, 1.0))
    _draw_lines(F, SQ_S, sa, third, alphas, cols)
    if cur >= 0:
        x, y, _, wd = SQ_S.centers[cur]
        hairline(F, x - wd / 2, int(y + 16), x + wd / 2, GOLD, 0.9 * sa)
        add_sprite(F, x, y - 24, 40, GOLD, 0.12 * sa)
    if pos >= IT:                                             # reaching back for 小猫: the thread is thin
        rb = smooth(ramp(t, TL.SEQ_READ0 + IT * TL.SEQ_READ_DT, 0.8)) * (1 - smooth(ramp(t, 21.0, 0.8)))
        if rb > 0:
            pts = SQ_S.arc(IT, CAT)
            for k in range(0, len(pts) - 1, 3):
                add_sprite(F, pts[k][0], pts[k][1], 1.4, STEEL, 0.5 * rb * sa)
            T("?", "corm_it", 40, 600, per_char=False).draw(F, SQ_S.centers[CAT][0], SQ_S.centers[CAT][1] - 80,
                                                           STEEL, 0.7 * rb * sa, align="center")
    sp = smooth(ramp(t, 22.4, 0.6)) * A
    if sp > 0 and cur >= 0:
        label = serif(f"第 {cur + 1} 步", 26, 500, 0.15)
        label.draw(F, 900, 900, IVORY, 0.8 * sp, align="right")
        caps("ONE WORD PER STEP", 12, 0.4).draw(F, 900, 926, DIM, 0.9 * sp, align="right")
    subtitles(F, t, [(1.0, 5.6, "在很长一段时间里，机器读句子，主要靠一个字一个字地读。",
                      "For a long time, machines read a sentence mostly one word at a time."),
                     (6.0, 11.0, "每读一个字，就把记住的东西，传给下一个字。",
                      "Each step passes what it remembers on to the next."),
                     (11.4, 16.6, "就像传话游戏：传得越远，开头就越模糊。",
                      "Like a game of telephone: the further it travels, the fainter the start."),
                     (17.0, 22.0, "等它读到「它」，「小猫」已经快想不起来了。",
                      "By the time it reaches 'it', 'the cat' is nearly forgotten."),
                     (22.4, 27.6, "而且只能排队，一个接一个，没法同时进行。",
                      "And it has to queue, one after another, never all at once."),
                     (28.0, 33.4, "句子越长，就越慢，也越健忘。", "The longer the sentence, the slower — and the more forgetful.")])


def _draw_lines(F, S, a, third, alphas, cols):
    n = 0
    for ln, tx in enumerate(S.txt):
        la = a if ln < 2 else a * third
        for i, (m, dx, dy) in enumerate(tx.items):
            k = S.char_tok.get((ln, i))
            al = la * (alphas[k] if k is not None else 0.22)
            col = cols[k] if k is not None else IVORY
            if al > 0.003:
                blend(F, m, S.x0[ln] + dx, S.ys[ln] + dy, col, al)


# --------------------------------------------------------------- at once ---

AO_S = Sentence(SENT_LINES, SENT_TOKENS, 72, (470, 620))
_LOGITS = np.array([
    # 小猫 没有 跳上 桌子 因为  它   太   累   了
    [2.6, 0.6, 1.4, 0.8, 0.2, 0.9, 0.1, 0.6, 0.1],   # 小猫
    [1.0, 2.2, 1.9, 0.4, 0.2, 0.1, 0.1, 0.2, 0.3],   # 没有
    [1.5, 1.2, 2.2, 1.9, 0.1, 0.2, 0.1, 0.1, 0.2],   # 跳上
    [0.6, 0.2, 1.8, 2.5, 0.1, 0.5, 0.2, 0.1, 0.1],   # 桌子
    [0.3, 0.6, 0.4, 0.2, 2.1, 0.7, 0.5, 1.6, 0.2],   # 因为
    [2.9, 0.2, 0.4, 0.9, 0.3, 1.0, 0.2, 0.8, 0.1],   # 它
    [0.2, 0.1, 0.1, 0.2, 0.3, 0.8, 2.0, 2.1, 0.4],   # 太
    [1.6, 0.2, 0.3, 0.1, 1.2, 1.5, 1.4, 2.1, 0.9],   # 累
    [0.4, 0.5, 0.3, 0.1, 0.2, 0.4, 0.6, 1.5, 2.0],   # 了
])
ATT = np.exp(_LOGITS * 1.4)
ATT /= ATT.sum(1, keepdims=True)


def _matrix(F, t, a, x0, y0, cell, M, labels, t_all, highlight=None):
    """The attention table: rows ask, columns answer; every cell lights at the same instant."""
    if a <= 0.003:
        return
    n = len(labels)
    for i, lab in enumerate(labels):
        serif(lab, 22, 500, 0.05).draw(F, x0 - 12, y0 + cell * i + cell * 0.66, IVORY, 0.75 * a, align="right")
        serif(lab, 20, 500, 0.0).draw(F, x0 + cell * i + cell / 2, y0 - 12, IVORY, 0.7 * a, align="center")
    on = smooth(ramp(t, t_all, 0.35))
    flash = math.exp(-max(0.0, t - t_all) * 3.0) if t >= t_all else 0.0
    vmax = M.max()
    for i in range(n):
        for j in range(n):
            v = M[i, j] / vmax
            x, y = x0 + cell * j, y0 + cell * i
            blend(F, np.full((cell - 4, cell - 4), 255, np.uint8), x + 2, y + 2, np.array([0.06, 0.08, 0.13],
                  np.float32), 0.8 * a)
            if on > 0:
                col = mix(STEEL, GOLD, smooth(v))
                E.add_light(F, np.full((cell - 6, cell - 6), 1.0, np.float32), x + 3, y + 3, col,
                            (0.05 + 0.6 * v ** 1.3) * on * a)
                if v > 0.55:
                    add_sprite(F, x + cell / 2, y + cell / 2, cell * 0.35, col, 0.25 * v * on * a)
    if flash > 0:
        add_sprite(F, x0 + cell * n / 2, y0 + cell * n / 2, cell * n * 0.45, GOLD, 0.35 * flash * a)
    if highlight is not None:
        i = highlight
        st = E.Stroke(x0 - 4, y0 + cell * i - 4, x0 + cell * n + 5, y0 + cell * (i + 1) + 5)
        st.poly([(x0 - 2, y0 + cell * i - 2), (x0 + cell * n + 2, y0 + cell * i - 2),
                 (x0 + cell * n + 2, y0 + cell * (i + 1) + 2), (x0 - 2, y0 + cell * (i + 1) + 2)], closed=True)
        st.glow(F, GOLD, 0.6 * a, 4)
        st.light(F, GOLD, 0.9 * a)


def atonce(F, t, g, fi):
    A = window(t, 0.0, 31.6, 0.6, 0.9)
    appear = ease_out(ramp(t, 1.0, 0.5))
    flash = math.exp(-max(0.0, t - 1.0) * 2.5) if t >= 1.0 else 0.0
    sym = (ATT + ATT.T) / 2
    web = ease_out(ramp(t, TL.ATO_WEB, 1.2)) * (1 - 0.6 * smooth(ramp(t, 10.4, 0.8)))
    if web > 0:
        for i in range(len(SENT_TOKENS)):
            for j in range(i + 1, len(SENT_TOKENS)):
                pts = AO_S.arc(i, j, lift=0.9)
                v = sym[i, j] / sym.max()
                glow_poly(F, pts, mix(STEEL, GOLD, v), (0.12 + 0.5 * v) * web * A, th=1, glow=0.4, sigma=3)
    fa = smooth(ramp(t, 10.6, 0.8)) * (1 - smooth(ramp(t, 16.0, 0.8)))
    fan(F, AO_S, IT, ATT[IT], fa, label=False, a=A)
    cols = [IVORY] * len(SENT_TOKENS)
    if 10.6 <= t < 16.4:
        cols[IT] = GOLD
    AO_S.draw(F, a=A * appear, tok_color=cols)
    if flash > 0:
        for k in range(len(SENT_TOKENS)):
            x, y, _, _ = AO_S.centers[k]
            add_sprite(F, x, y - 26, 34, GOLD, 0.5 * flash * A)
    ta = smooth(ramp(t, TL.ATO_TABLE - 0.4, 0.8)) * A
    _matrix(F, t, ta, 290, 730, 52, ATT, [lab for _, _, _, lab in SENT_TOKENS], TL.ATO_TABLE + 1.2,
            highlight=IT if t > TL.ATO_TABLE + 2.2 else None)
    serif("（数值为示意）", 18, 400, 0.1).draw(F, 758, 1240, DIM, 0.7 * ta, align="right")
    subtitles(F, t, [(0.6, 5.0, "2017 年的这篇论文，换了一种读法。", "The 2017 paper read in a different way."),
                     (5.4, 10.4, "所有的字同时出现，每个字都同时看向其他所有的字。",
                      "Every word appears at once, and each looks at all the others."),
                     (10.8, 15.8, "每个字都在问：在这句话里，谁和我最有关系？",
                      "Each word asks: in this sentence, who matters most to me?"),
                     (16.2, 22.2, "答案是一张表：每一格，是一个字对另一个字的注意力。",
                      "The answer is a table: each cell is how much one word attends to another."),
                     (22.6, 31.4, "不用排队，一次算完。句子再长，开头也不会被忘掉。",
                      "No queue — it is all computed at once, and the start is never lost.")])


# ------------------------------------------------------------------- qkv ---

BOOKS = [("小猫", 0.62), ("没有", 0.03), ("跳上", 0.05), ("桌子", 0.12), ("因为", 0.03), ("太", 0.04), ("累", 0.08),
         ("了", 0.03)]
SHELF = (470, 430, 470, 480)


def _shelf():
    x0, y0, w, h = SHELF
    rng = np.random.default_rng(6)
    books = [(44, int(rng.uniform(370, 440))) for _ in BOOKS]
    return G.shelf(x0, y0, w, h, books)


SH_PTS, SH_EDGES, SH_BOXES = _shelf()
READER = G.profile(205, 730, 230, hair="bun")


def qkv(F, t, g, fi):
    A = window(t, 0.0, 41.6, 0.6, 0.9)
    sc = 1 - 0.93 * smooth(ramp(t, TL.QKV_FORMULA - 0.4, 1.0))       # the scene steps back for the formula
    a_sc = A * sc
    G.draw_profile(F, READER, t, 0.4, 1.6, glow=0.25 * smooth(ramp(t, TL.QKV_MATCH + 3, 1.0)), a=a_sc, fill=0.05)
    G.draw(F, SH_PTS, SH_EDGES, t, 0.8, 2.0, a=a_sc, star=0.7, seed=7)
    match = np.array([b[1] for b in BOOKS])
    mrel = match / match.max()
    pull = ease_in_out(ramp(t, TL.QKV_MATCH + 2.4, 0.9))
    for k, ((bx, by, bw, bh), (lab, _)) in enumerate(zip(SH_BOXES, BOOKS)):
        lift = 46 * pull * (1 if k == 0 else 0)
        ka = smooth(ramp(t, TL.QKV_K + 0.12 * k, 0.6)) * a_sc
        glowk = smooth(ramp(t, TL.QKV_MATCH + 1.0, 0.8)) * mrel[k] ** 1.5
        col = mix(IVORY, GOLD, glowk)
        for c_i, ch in enumerate(lab):                        # titles run down the spines
            serif(ch, 26, 500).draw(F, bx + bw / 2, by + 48 + c_i * 33 - lift, col, ka, align="center")
        if ka > 0:
            T("K", "stix_it", 18, None, per_char=False).draw(F, bx + bw / 2, by - 10 - lift, STEEL, 0.8 * ka,
                                                             align="center")
        va = smooth(ramp(t, TL.QKV_V + 0.1 * k, 0.6)) * a_sc
        if va > 0:
            orb(F, bx + bw / 2, by + bh - 34 - lift, 5, mix(STEEL, GOLD, 0.4), 0.8 * va)
            T("V", "stix_it", 16, None, per_char=False).draw(F, bx + bw / 2, by + bh - 10 - lift, STEEL, 0.7 * va,
                                                             align="center")
        if lift > 0:
            hairline(F, bx, int(by - lift), bx + bw, GOLD, 0.5 * a_sc)
    # the question card, held up in front of the reader
    qa = smooth(ramp(t, TL.QKV_Q, 0.6)) * a_sc
    card = np.array([400, 650])
    if qa > 0:
        st = E.Stroke(card[0] - 46, card[1] - 34, card[0] + 47, card[1] + 35)
        st.poly([(card[0] - 44, card[1] - 32), (card[0] + 44, card[1] - 32), (card[0] + 44, card[1] + 32),
                 (card[0] - 44, card[1] + 32)], closed=True)
        st.glow(F, GOLD, 0.4 * qa, 4)
        st.light(F, GOLD, 0.8 * qa)
        T("Q", "stix_it", 40, None, per_char=False).draw(F, card[0], card[1] + 14, GOLD, qa, align="center")
        serif("「它」指谁？", 20, 500, 0.1).draw(F, card[0], card[1] + 62, IVORY, 0.8 * qa, align="center")
    # matching: a beam from the question to every label, strongest where they agree
    ma = smooth(ramp(t, TL.QKV_MATCH, 0.8)) * a_sc
    if ma > 0:
        for k, (bx, by, bw, bh) in enumerate(SH_BOXES):
            p0 = card + np.array([44, -10])
            p1 = np.array([bx + bw / 2, by - 24 - (46 * pull if k == 0 else 0)])
            v = mrel[k]
            glow_poly(F, [p0, p0 + (p1 - p0) * ease_out(ramp(t, TL.QKV_MATCH + 0.05 * k, 0.6))],
                      mix(STEEL, GOLD, v), (0.15 + 0.85 * v ** 1.2) * ma, th=1 if v < 0.5 else 2, glow=0.4 + v,
                      sigma=3 + 3 * v)
        pa = smooth(ramp(t, TL.QKV_MATCH + 1.2, 0.6)) * a_sc
        bx, by, bw, bh = SH_BOXES[0]
        T(f"{round(100 * BOOKS[0][1])}%", "stix", 30, None, per_char=False).draw(F, bx + bw / 2, by - 50 - 46 * pull,
                                                                                GOLD, pa, align="center")
    # the value flows back to the one who asked
    fl = ramp(t, TL.QKV_MATCH + 3.4, 1.4)
    if 0 < fl < 1:
        bx, by, bw, bh = SH_BOXES[0]
        a_pt = np.array([bx + bw / 2, by + bh - 34 - 46])
        b_pt = READER["eye"] + np.array([40, 10])
        u = ease_in_out(fl)
        mid = (a_pt + b_pt) / 2 + np.array([0, -90])
        p = (1 - u) ** 2 * a_pt + 2 * (1 - u) * u * mid + u ** 2 * b_pt
        orb(F, p[0], p[1], 7, GOLD, a_sc * math.sin(math.pi * fl) + 0.2 * a_sc)
    # the one line
    fa = smooth(ramp(t, TL.QKV_FORMULA, 1.0)) * A
    if fa > 0:
        _formula(F, CX, 760, fa)
        gl = smooth(ramp(t, TL.QKV_FORMULA + 1.2, 0.8)) * A
        for k, (sym, zh) in enumerate((("Q", "提问"), ("K", "标签"), ("V", "内容"))):
            x = CX - 220 + 220 * k
            T(sym, "stix_it", 34, None, per_char=False).draw(F, x - 22, 990, GOLD, gl, align="center")
            serif(zh, 24, 500, 0.1).draw(F, x + 20, 988, IVORY, 0.85 * gl, align="center")
    subtitles(F, t, [(1.0, 4.8, "它是怎么算出来的？每个字都带着三样东西。",
                      "How is it computed? Every word carries three things."),
                     (5.0, 9.6, "一个问题 Q：我在找什么？", "A query, Q: what am I looking for?"),
                     (10.0, 14.6, "一张标签 K：我是谁？", "A key, K: what am I?"),
                     (15.0, 19.8, "一份内容 V：我能提供什么？", "A value, V: what do I have to offer?"),
                     (20.2, 27.0, "问题和标签越对得上，就从那里取走越多的内容。",
                      "The better a query matches a key, the more of that value it takes."),
                     (27.6, 33.6, "整篇论文的核心，就是这一行。", "The heart of the whole paper is this one line."),
                     (34.0, 41.4, "它为每个字都这样算一遍——而且是同时算。",
                      "It does this for every word — and for all of them at once.")])


def _formula(F, cx, y, a):
    """Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V, set by hand from STIX pieces."""
    sz = 50

    def run(parts, x, yy, measure=False):
        for txt, font, size, dy in parts:
            tx = T(txt, font, size, None, per_char=False)
            if not measure:
                tx.draw(F, x, yy + dy, IVORY if font != "stix_it" else GOLD, a)
            x += tx.width + 2
        return x

    l1 = [("Attention", "stix", sz, 0), ("(", "stix", sz, 0), ("Q", "stix_it", sz, 0), (", ", "stix", sz, 0),
          ("K", "stix_it", sz, 0), (", ", "stix", sz, 0), ("V", "stix_it", sz, 0), (")", "stix", sz, 0)]
    w1 = run(l1, 0, 0, True)
    run(l1, cx - w1 / 2, y, False)
    l2a = [("=  softmax", "stix", sz, 0), ("(", "stix", sz, 0), ("QK", "stix_it", sz, 0), ("T", "stix", 30, -22),
           (" / ", "stix", sz, 0)]
    l2b = [("d", "stix_it", sz, 0), ("k", "stix_it", 30, 10), (")", "stix", sz, 0), (" V", "stix_it", sz, 0)]
    sq = T("√", "math", sz, None, per_char=False)
    w2 = run(l2a, 0, 0, True) + sq.width + run(l2b, 0, 0, True)
    x = cx - w2 / 2
    x = run(l2a, x, y + 90)
    sq.draw(F, x, y + 90, IVORY, a)
    xs = x + sq.width - 2
    x = run(l2b, xs, y + 90)
    hairline(F, xs, y + 90 - 46, xs + T("d", "stix_it", sz, None, per_char=False).width +
             T("k", "stix_it", 30, None, per_char=False).width + 4, IVORY, a, th=2)


# ----------------------------------------------------------------- heads ---

HD_S = Sentence(SENT_LINES, SENT_TOKENS, 76, (840, 1030))
HEADS = [
    ("指代", "它 → 小猫", [(5, 0, 1.0), (7, 0, 0.6)]),
    ("动作", "跳上 → 桌子", [(2, 3, 1.0), (1, 2, 0.5)]),
    ("主语", "小猫 → 跳上", [(0, 2, 1.0), (0, 1, 0.6)]),
    ("原因", "累 → 因为", [(7, 4, 1.0), (6, 7, 0.6)]),
    ("左邻", "每个字 → 前一个字", [(k, k - 1, 0.7) for k in range(1, 9)]),
    ("右邻", "每个字 → 后一个字", [(k, k + 1, 0.7) for k in range(0, 8)]),
    ("否定", "没有 → 跳上", [(1, 2, 1.0), (1, 3, 0.5)]),
    ("全局", "看整句", [(i, j, 0.25) for i in range(9) for j in range(9) if i < j and (i + j) % 3 == 0]),
]


def heads(F, t, g, fi):
    A = window(t, 0.0, 25.6, 0.6, 0.9)
    HD_S.draw(F, a=A * ease_out(ramp(t, 0.4, 0.6)))
    cur = int((t - TL.HEADS_T0) / TL.HEADS_DT) if t >= TL.HEADS_T0 else -1
    allin = smooth(ramp(t, TL.HEADS_ALL, 1.2))
    merge = smooth(ramp(t, 19.6, 1.6))
    for k, (name, desc, links) in enumerate(HEADS):
        start = TL.HEADS_T0 + k * TL.HEADS_DT
        solo = smooth(ramp(t, start, 0.35)) * (1 - smooth(ramp(t, start + TL.HEADS_DT, 0.35)))
        a = A * max(solo, 0.55 * allin)
        if a <= 0.003:
            continue
        col = mix(HEAD_COLS[k], GOLD, merge)
        for i, j, v in links:
            pts = HD_S.arc(i, j, lift=0.8 + 0.08 * k)
            gp = ease_out(ramp(t, start, 0.5))
            kk = max(2, int(len(pts) * gp))
            glow_poly(F, pts[:kk], col, a * (0.35 + 0.65 * v), th=2 if v > 0.8 else 1, glow=0.6 + 0.6 * v, sigma=4)
            if solo > 0.5:
                orb(F, pts[kk - 1][0], pts[kk - 1][1], 3.5, col, a)
    # eight dots: which pair of eyes is looking
    for k in range(8):
        x = CX - 7 * 34 / 2 + 34 * k
        on = 1.0 if k == cur else 0.0
        add_sprite(F, x, 400, 3.0 + 2.5 * on, HEAD_COLS[k], (0.35 + 0.65 * max(on, allin)) * A)
    if 0 <= cur < 8 and t < TL.HEADS_ALL:
        name, desc, _ = HEADS[cur]
        la = A * smooth(ramp(t, TL.HEADS_T0 + cur * TL.HEADS_DT, 0.3))
        serif(f"第 {cur + 1} 双眼睛 · {name}", 34, 500, 0.1).draw(F, CX, 480, HEAD_COLS[cur], la, align="center")
        serif(desc, 24, 400, 0.08).draw(F, CX, 522, IVORY, 0.75 * la, align="center")
    if allin > 0:
        serif("多头注意力", 40, 500, 0.3).draw(F, CX, 490, GOLD, A * allin, align="center")
        caps("MULTI-HEAD ATTENTION", 14, 0.5).draw(F, CX, 530, DIM, A * allin, align="center")
    subtitles(F, t, [(0.6, 4.8, "一种看法不够，那就同时用八种。", "One way of looking isn't enough, so it uses eight at once."),
                     (5.2, 11.8, "有的盯着「它」指谁，有的盯着谁做了什么，有的只看身边的字。",
                      "One tracks who 'it' is, one tracks who did what, one only watches its neighbours."),
                     (12.2, 18.4, "这叫「多头注意力」：八双眼睛，看同一句话。",
                      "That is multi-head attention: eight pairs of eyes on one sentence."),
                     (18.8, 25.4, "最后，把八种看法合在一起。", "Then the eight views are put together.")])


# ----------------------------------------------------------------- order ---

def order(F, t, g, fi):
    A = window(t, 0.0, 25.6, 0.6, 0.9)
    sw = ease_in_out(ramp(t, TL.ORD_SWAP, 1.2))
    gr = 610
    run = t * 9.0
    # cat chases dog -> dog chases cat: the two swap places
    cat_x = lerp(300, 690, sw)
    dog_x = lerp(700, 290, sw)
    ca = A * (1 - 0.5 * smooth(ramp(t, TL.ORD_WAVES, 1.0)))
    cp, ce, ceye = G.cat(cat_x, gr, 125, run=run)
    dp, de, deye = G.dog(dog_x, gr, 118, run=run + 1.3)
    chaser = 1 - sw
    G.draw(F, cp, ce, glow=chaser, extra=ceye, a=ca, seed=1)
    G.draw(F, dp, de, glow=1 - chaser, extra=deye, a=ca, seed=2)
    for k in range(24):                                     # the ground streams past
        x = (150 + (k * 34 - t * 260) % 816)
        orb(F, x, gr + 6, 1.2, STEEL, 0.22 * ca)
    # the three characters trade places
    y = 830
    sz = 130
    xs = [CX - 150, CX, CX + 150]
    chars = ["猫", "追", "狗"]
    for k, ch in enumerate(chars):
        x = xs[k]
        if k == 0:
            x = lerp(xs[0], xs[2], sw)
            yy = y - 70 * math.sin(math.pi * sw)
        elif k == 2:
            x = lerp(xs[2], xs[0], sw)
            yy = y + 50 * math.sin(math.pi * sw)
        else:
            yy = y
        col = GOLD if ch in ("猫", "狗") else IVORY
        serif(ch, sz, 500).draw(F, x, yy, col, A * smooth(ramp(t, 0.6 + 0.2 * k, 0.6)), align="center")
    # position waves: every place gets its own fingerprint
    wa = smooth(ramp(t, TL.ORD_WAVES, 1.2)) * A
    if wa > 0:
        freqs = [1.0, 2.1, 4.3, 8.5]
        for i, f in enumerate(freqs):
            yc = 950 + i * 66
            xx = np.linspace(170, 870, 300)
            ph = (xx - xs[0]) / 150.0
            yy = yc - 20 * np.sin(ph * f * 0.9)
            glow_poly(F, np.column_stack([xx, yy]), mix(STEEL, GOLD, i / 3), 0.55 * wa, th=1, glow=0.5, sigma=3)
            for k in range(3):
                v = -20 * math.sin(k * f * 0.9)
                orb(F, xs[k], yc + v, 4.5, GOLD, 0.9 * wa * smooth(ramp(t, TL.ORD_WAVES + 1.0 + 0.4 * k, 0.5)))
        for k in range(3):
            for yy in np.arange(925, 1180, 9):
                add_sprite(F, xs[k], yy, 1.0, IVORY, 0.25 * wa)
            T(str(k + 1), "stix", 22, None, per_char=False).draw(F, xs[k], 1215, DIM, wa, align="center")
    subtitles(F, t, [(0.6, 5.6, "可是，同时看所有的字，就分不清谁先谁后了。",
                      "But seeing every word at once means losing their order."),
                     (6.0, 11.2, "「猫追狗」和「狗追猫」，用的是同样三个字。",
                      "'Cat chases dog' and 'dog chases cat' use the same three characters."),
                     (11.6, 18.6, "于是，每个位置都被印上一组快慢不同的波，像一个独一无二的指纹。",
                      "So every position is stamped with waves of different speeds — a fingerprint all its own."),
                     (19.0, 25.4, "有了位置，它才分得清：谁在追谁。", "With positions, it can tell who is chasing whom.")])


# ----------------------------------------------------------------- after ---

AUTHORS = ["ASHISH VASWANI", "NOAM SHAZEER", "NIKI PARMAR", "JAKOB USZKOREIT",
           "LLION JONES", "AIDAN N. GOMEZ", "ŁUKASZ KAISER", "ILLIA POLOSUKHIN"]


def after(F, t, g, fi):
    A = window(t, 0.0, 29.6, 0.6, 0.9)
    # the first page
    pa = A * (1 - smooth(ramp(t, TL.AFT_BEATLES - 0.4, 0.8)))
    if pa > 0.003:
        pp, pe = G.page(190, 330, 700, 640)
        G.draw(F, pp, pe, t, 0.3, 1.0, a=pa, star=0.6, seed=3)
        tx = T("Attention Is All You Need", "corm", 50, 600, tracking=0.01)
        tx.draw(F, 540, 460, IVORY, pa * smooth(ramp(t, 0.8, 0.8)), align="center")
        for k, nm in enumerate(AUTHORS):
            row, col = divmod(k, 2)
            x = 380 + 320 * col
            y = 530 + 30 * row
            caps(nm + " ∗", 12, 0.3).draw(F, x, y, IVORY, 0.75 * pa * smooth(ramp(t, 1.2 + 0.08 * k, 0.5)),
                                                align="center")
        for k in range(7):                                  # text, suggested
            w_ = 560 if k % 3 else 470
            hairline(F, 260, 690 + 28 * k, 260 + w_, DIM, 0.25 * pa)
        fa = smooth(ramp(t, TL.AFT_BYLINE, 0.8)) * pa
        italic("∗ Equal contribution. Listing order is random.", 26).draw(F, 260, 920, GOLD, max(0.35 * pa, fa))
        if fa > 0:
            hairline(F, 260, 932, 260 + 480 * ease_out(ramp(t, TL.AFT_BYLINE + 0.4, 0.8)), GOLD, 0.7 * fa)
    # the Beatles
    ba = A * window(t, TL.AFT_BEATLES, TL.AFT_GPT + 0.2, 0.8, 0.8)
    if ba > 0.003:
        G.record(F, CX, 700, 260, t, ba, glow=0.3)
        italic("All You Need Is Love", 46).draw(F, CX, 930, IVORY, ba, align="center")
        caps("THE BEATLES   ·   1967", 15, 0.45).draw(F, CX, 972, DIM, ba, align="center")
    # GPT: the T is the Transformer, and the tower of layers
    ga = A * smooth(ramp(t, TL.AFT_GPT, 0.8))
    if ga > 0.003:
        lift = 360 * ease_in_out(ramp(t, TL.AFT_TOWER, 1.2))
        for k, ch in enumerate("GPT"):
            T(ch, "corm", 150, 600, per_char=False).draw(F, CX - 110 + 110 * k, 760 - lift,
                                                        GOLD if ch == "T" else IVORY, ga, align="center")
        caps("GENERATIVE PRE-TRAINED TRANSFORMER", 16, 0.4).draw(F, CX, 820 - lift, DIM, ga, align="center")
        ta = A * smooth(ramp(t, TL.AFT_TOWER + 0.6, 0.6))
        n_blocks = int(1 + 8 * ease_out(ramp(t, TL.AFT_TOWER + 0.6, 4.0)))
        for k in range(n_blocks):
            bp, be = G.iso_box(CX - 60, 1150 - 52 * k, 320, 34, d=0.5)
            G.draw(F, bp, be, a=ta * (0.55 + 0.45 * (k == n_blocks - 1)), star=0.5,
                   glow=0.6 if k == n_blocks - 1 else 0.0, seed=k)
        for k, (yr, nm) in enumerate((("2018", "GPT"), ("2018", "BERT"), ("2022", "ChatGPT"))):
            la = A * smooth(ramp(t, TL.AFT_TOWER + 1.2 + 1.0 * k, 0.6))
            y = 1080 - 160 * k
            T(yr, "stix", 22, None, per_char=False).draw(F, 930, y, DIM, la, align="right")
            T(nm, "corm", 38, 600, per_char=False).draw(F, 930, y + 38, IVORY, la, align="right")
        if n_blocks >= 9:
            pa2 = A * smooth(ramp(t, TL.AFT_TOWER + 4.4, 0.8))
            php, phe = G.phone(CX - 40, 592, 110, 190)
            # the phone stands on the top slab: hide the slab's back edge behind its screen
            m = np.zeros(F.shape[:2], np.float32)
            cv2.fillPoly(m, [np.round(php * 16).astype(np.int32)], 1.0, cv2.LINE_AA, 4)
            bg = np.median(F[560:640, 380:420].reshape(-1, 3), 0)
            m = (pa2 * m)[..., None]
            F[:] = F * (1 - m) + bg * m
            G.draw(F, php, phe, a=pa2, star=0.4, glow=0.4)
            for k, (dx, w_) in enumerate(((-10, 56), (14, 54), (-12, 46))):
                hairline(F, CX - 40 + dx - w_ / 2, 550 + 30 * k, CX - 40 + dx + w_ / 2, GOLD if k % 2 else IVORY,
                         0.6 * pa2, th=8)
    subtitles(F, t, [(0.6, 5.0, "2017 年 6 月，这篇论文发表，作者一共八位。",
                      "June 2017: the paper comes out, with eight authors."),
                     (5.4, 10.8, "署名旁边写着：贡献相同，排名随机。",
                      "A note by the names: equal contribution, order random."),
                     (11.2, 16.6, "而它的标题，是在致敬披头士的一首歌。", "And its title tips its hat to a Beatles song."),
                     (17.0, 22.2, "后来，GPT 里的「T」，就是 Transformer。", "Later, the 'T' in GPT stood for Transformer."),
                     (22.6, 29.4, "你和 AI 的每一次对话，背后都是这一行公式，一层一层地叠起来。",
                      "Every conversation you have with AI runs on that one line, stacked layer upon layer.")])


# ---------------------------------------------------------------- ending ---

_SKY = np.random.default_rng(21)
SKY_STARS = np.column_stack([_SKY.uniform(140, 940, 34), _SKY.uniform(330, 860, 34)])


def ending(F, t, g, fi):
    A = window(t, 0.0, 24.0, 0.8, 0.01)
    # someone looking up; the sky's stars find each other
    pr = G.profile(250, 1110, 190, tilt=0.42 * ease_in_out(ramp(t, 1.0, 3.0)), hair="short")
    G.draw_profile(F, pr, t, 0.3, 1.8, glow=0.2, a=A * (1 - 0.4 * smooth(ramp(t, TL.END_TITLE, 1.5))), fill=0.06)
    sa = smooth(ramp(t, 1.5, 2.0)) * A
    rng = np.random.default_rng(9)
    for k, (x, y) in enumerate(SKY_STARS):
        tw_ = 0.75 + 0.25 * math.sin(t * 1.7 + k)
        orb(F, x, y, 2.6 + 1.6 * rng.random(), mix(STEEL, GOLD, 0.3), 0.7 * sa * tw_)
    web = ease_out(ramp(t, 6.0, 5.0)) * A * (1 - 0.5 * smooth(ramp(t, TL.END_LINE, 1.0)))
    if web > 0:
        D = np.linalg.norm(SKY_STARS[:, None] - SKY_STARS[None], axis=2)
        for i in range(len(SKY_STARS)):
            for j in np.argsort(D[i])[1:3]:
                if i < j:
                    p, q = SKY_STARS[i], SKY_STARS[j]
                    glow_poly(F, [p, p + (q - p) * web], mix(STEEL, GOLD, 0.4), 0.35 * web, th=1, glow=0.4, sigma=3)
        eye = pr["eye"]
        for k in np.argsort(np.linalg.norm(SKY_STARS - eye, axis=1))[:3]:
            p = SKY_STARS[k]
            glow_poly(F, [eye, eye + (p - eye) * web], GOLD, 0.25 * web, th=1, glow=0.6, sigma=4)
    statement(F, "注意力，", CX, 560, t, TL.END_LINE, None, size=64, cps=5, gold={0, 1, 2})
    statement(F, "也许真的就是你所需要的一切。", CX, 660, t, TL.END_LINE + 1.0, None, size=56, cps=7)
    ta = smooth(ramp(t, TL.END_TITLE, 1.2))
    T("Attention Is All You Need", "corm_it", 44, 500, per_char=False).draw(F, CX, 820, IVORY, ta, align="center")
    caps("VASWANI ET AL.   ·   2017", 16, 0.45).draw(F, CX, 862, DIM, ta, align="center")
    subtitles(F, t, [(0.8, 5.6, "机器学会了注意力，于是开始读懂语言。", "Machines learned attention, and began to read language."),
                     (6.0, 11.8, "而在一个什么都在争夺你注意力的时代——",
                      "And in an age when everything is competing for your attention —"),
                     (TL.END_LINE + 0.4, 18.6, "", "perhaps attention really is all you need.")])


SCENE_FUNCS = {"prologue": prologue, "sequential": sequential, "atonce": atonce, "qkv": qkv, "heads": heads,
               "order": order, "after": after, "ending": ending}
