"""注意力 · Attention Is All You Need.  draw(F, t_local, g_global, frame).

One spine: you already do attention, instantly.  Machines used to read one
word at a time and forget the beginning; the Transformer lets every word look
at every other word at once.  The pictures follow classic references (see
plates.py): a star-atlas plate of the cat and the table, cut-paper portraits
for the telephone game and the stargazer, the authors' own two-column
attention view, a 3Blue1Brown-style grid, and the paper's first page."""

import math

import numpy as np

import plates as P
import timeline as TL
from style import (CX, DIM, GOLD, IVORY, T, add_sprite, blend, caps, comet, ease_in_out, ease_out, glow_poly,
                   hairline, lerp, mix, orb, ramp, serif, smooth, statement, subtitles, window)

STEEL = P.STEEL


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



SENT_LINES = ("小猫没有跳上桌子，", "因为它太累了。")
SENT_TOKENS = [(0, 0, 2, "小猫"), (0, 2, 4, "没有"), (0, 4, 6, "跳上"), (0, 6, 8, "桌子"),
               (1, 0, 2, "因为"), (1, 2, 3, "它"), (1, 3, 4, "太"), (1, 4, 5, "累"), (1, 5, 6, "了")]
IT, CAT, TABLE = 5, 0, 3


# ------------------------------------------------------------ prologue ---

def _burst(F, q, a, cy):
    P0 = np.random.default_rng(3).normal(0, 1, (140, 2))
    if q > 3.0:
        return
    sp = 1 - math.exp(-q * 2.2)
    for vx, vy in P0:
        add_sprite(F, CX + vx * 300 * sp, cy + vy * 160 * sp, 1.2, GOLD, 0.6 * a * max(0.0, 1 - q / 3.0))


def plate_title(F, text, y, a):
    """An atlas plate's title: spaced capitals between two short rules."""
    if a <= 0.003:
        return
    tx = caps(text, 34, 0.9)
    tx.draw(F, CX, y, GOLD, a, align="center")
    half = tx.width / 2 + 26
    hairline(F, CX - half - 70, y - 9, CX - half, GOLD, 0.6 * a)
    hairline(F, CX + half, y - 9, CX + half + 70, GOLD, 0.6 * a)


def prologue(F, t, g, fi):
    A = 1 - smooth(ramp(t, 9.9, 1.0))
    if A > 0.003:
        reveal = smooth(ramp(t, 0.2, 1.8))
        hop = math.sin(math.pi * ramp(t, TL.P_HOP, 0.5)) if TL.P_HOP <= t <= TL.P_HOP + 0.5 else 0.0
        sleep = ease_in_out(ramp(t, TL.P_LIE, 0.6)) * (1 - ease_in_out(ramp(t, TL.P_SWAP + 0.1, 0.8)))
        h = 1.0 + 0.75 * ease_in_out(ramp(t, TL.P_SWAP, 1.0))
        P.opening(F, w_cat=smooth(ramp(t, TL.P_FAN + 0.3, 0.8)) * (1 - smooth(ramp(t, TL.P_RE, 1.0))),
                  w_table=smooth(ramp(t, TL.P_RE, 1.0)), sleep=sleep, h=h,
                  it_on=smooth(ramp(t, TL.P_IT, 0.3)), link=ease_out(ramp(t, TL.P_FAN, 0.9)),
                  link_table=ease_out(ramp(t, TL.P_RE, 0.9)), reveal=reveal,
                  swap=smooth(ramp(t, TL.P_SWAP, 0.55)), a=A, hop=hop, chart=ramp(t, 0.1, 1.3),
                  typed=lambda n: ease_out(ramp(t, TL.P_T1 + n / 7.5, 0.25)))
        if t >= TL.P_IT:
            sx, sy = P.OPEN.centers[IT]
            p = math.exp(-(t - TL.P_IT) * 2.2)
            add_sprite(F, sx, sy, 46 + 30 * p, GOLD, 0.5 * p * A)
        if TL.P_SWAP <= t < TL.P_SWAP + 0.9:
            sx, sy = P.OPEN.centers[7]
            q = (t - TL.P_SWAP) / 0.9
            add_sprite(F, sx, sy, 40 + 60 * q, GOLD, 0.7 * math.sin(math.pi * q) * A)
        plate_title(F, "ATTENTION", 330, smooth(ramp(t, TL.P_NAME + 0.2, 0.8)) * A)
    subtitles(F, t, [(TL.P_IT, TL.P_SWAP - 0.2, "「它」指的是谁？你一眼就知道。", "Who is 'it'? You know at a glance."),
                     (TL.P_SWAP, TL.P_NAME - 0.2, "换一个字，「它」就换了对象。",
                      "Change one word, and 'it' points elsewhere."),
                     (TL.P_NAME, 10.6, "你刚才做的这件事，就叫「注意力」。", "What you just did is called attention.")])
    # the turn: someone taught machines to do it
    statement(F, "2017 年，八位研究者，", CX, 760, t, 10.9, 15.1, size=58, cps=10)
    statement(F, "让机器只靠这件事，读懂语言。", CX, 860, t, 11.7, 15.1, size=58, cps=10, gold={3, 4, 5, 6, 7})
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
            serif("注意力，就是你所需要的一切", 42, 500, 0.2).draw(F, CX, 905, GOLD, fa, align="center")
            hairline(F, CX - 80 * fa, 945, CX + 80 * fa, GOLD, 0.6 * fa)
            caps("VASWANI ET AL.   ·   2017", 20, 0.42).draw(F, CX, 990, DIM, fa, align="center")


# ------------------------------------------------------------ sequential ---

SQ_S = Sentence(("小猫没有跳上桌子，", "因为它太累了。", "它整整睡了一个下午。"),
                SENT_TOKENS + [(2, 0, 1, "它"), (2, 1, 3, "整整"), (2, 3, 4, "睡"), (2, 4, 5, "了"), (2, 5, 8, "一个下"),
                               (2, 8, 9, "午")], 66, (1000, 1110, 1220))


def _read_pos(t):
    """Index of the token being read (fractional), one word at a time."""
    if t < TL.SEQ_READ0:
        return -1.0
    n = (t - TL.SEQ_READ0) / TL.SEQ_READ_DT
    if t > 22.0:                                              # the sentence grows; reading crawls on
        n = (22.0 - TL.SEQ_READ0) / TL.SEQ_READ_DT + (t - 22.0) / 1.3
    return min(n, len(SQ_S.tokens) - 1 + 0.999)



def _draw_lines(F, S, a, third, alphas, cols):
    for ln, tx in enumerate(S.txt):
        la = a if ln < 2 else a * third
        for i, (m, dx, dy) in enumerate(tx.items):
            k = S.char_tok.get((ln, i))
            al = la * (alphas[k] if k is not None else 0.22)
            col = cols[k] if k is not None else IVORY
            if al > 0.003:
                blend(F, m, S.x0[ln] + dx, S.ys[ln] + dy, col, al)


def sequential(F, t, g, fi):
    A = window(t, 0.0, 33.6, 0.8, 0.9)
    # the telephone game: cut-paper portraits in a row, a whisper passed from lips to ear
    pa = A * (1 - 0.55 * smooth(ramp(t, 13.5, 1.5)))
    each = [smooth(ramp(t, 0.4 + 0.35 * k, 1.2)) for k in range(4)]
    busts = P.whisper_row(F, y=560, s=165, a=pa, each=each)
    fade_out = 1 - smooth(ramp(t, 13.0, 1.0))
    for k in range(3):
        t0 = TL.SEQ_PASS0 + k * TL.SEQ_PASS_DT
        q = ramp(t, t0, 1.1)
        held = k == 0 and t0 - 1.4 <= t < t0
        last = k == 2 and t >= t0 + 1.1
        if 0 < q < 1 or held or last:
            aa = A * fade_out * (smooth(ramp(t, t0 - 1.4, 0.5)) if held else 1.0)
            P.message(F, busts, k, ease_in_out(q), aa)
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
            T("?", "corm_it", 44, 600, per_char=False).draw(F, SQ_S.centers[CAT][0], SQ_S.centers[CAT][1] - 84,
                                                           STEEL, 0.7 * rb * sa, align="center")
    sp = smooth(ramp(t, 22.4, 0.6)) * A
    if sp > 0 and cur >= 0:
        serif(f"第 {cur + 1} 步", 32, 500, 0.15).draw(F, 960, 880, IVORY, 0.85 * sp, align="right")
        caps("ONE WORD PER STEP", 15, 0.4).draw(F, 960, 910, DIM, 0.9 * sp, align="right")
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


# --------------------------------------------------------------- at once ---


def _overshoot(u):
    """0 -> 1 with a small bounce past 1: the dots pop."""
    if u <= 0:
        return 0.0
    if u >= 1:
        return 1.0
    c = 1.7
    v = u - 1
    return 1 + (c + 1) * v ** 3 + c * v ** 2


def atonce(F, t, g, fi):
    A = window(t, 0.0, 31.6, 0.6, 0.9)
    appear = ease_out(ramp(t, 1.0, 0.5))
    flash = math.exp(-max(0.0, t - 1.0) * 2.5) if t >= 1.0 else 0.0
    ca = A * (1 - smooth(ramp(t, TL.ATO_TABLE - 0.6, 0.8)))       # the columns give way to the table
    if ca > 0.003:
        focus = smooth(ramp(t, 10.6, 0.8))
        web = ease_out(ramp(t, TL.ATO_WEB, 1.2)) * (1 - focus)
        # every word looks at every word: each word on the right sends its lines, one after another
        if web > 0:
            for k in range(9):
                gk = ease_out(ramp(t, TL.ATO_WEB + 0.07 * k, 0.8))
                P.fan_lines(F, k, ATT[k], 0.4 * web * ca, gk, gold=False, floor=0.06)
        right_on = np.zeros(9)
        if web > 0:
            for k in range(9):
                right_on[k] = 0.5 * math.exp(-max(0.0, t - TL.ATO_WEB - 0.07 * k) * 3.0) * (t >= TL.ATO_WEB + 0.07 * k)
        right_on[IT] = max(right_on[IT], focus)
        if focus < 1:
            P.columns(F, ca * appear * (1 - focus), right_on=right_on, right_a=0.85)
        if focus > 0:
            P.columns(F, ca * appear * focus, left_w=ATT[IT], right_on=right_on, right_a=0.32)
            P.fan_lines(F, IT, ATT[IT], ca * focus, ease_out(ramp(t, 10.6, 0.9)))
        if flash > 0:
            for k in range(9):
                add_sprite(F, P.COL_XL - 40, P.col_y(k), 34, GOLD, 0.45 * flash * A)
                add_sprite(F, P.COL_XR + 40, P.col_y(k), 34, GOLD, 0.45 * flash * A)
    ta = smooth(ramp(t, TL.ATO_TABLE - 0.4, 0.8)) * A
    if ta > 0.003:
        pop = _overshoot(ramp(t, TL.ATO_TABLE + 1.2, 0.45))
        P.grid(F, ATT, a=ta, on=pop, row=IT, row_a=smooth(ramp(t, TL.ATO_TABLE + 2.4, 0.7)))
        fl = math.exp(-max(0.0, t - TL.ATO_TABLE - 1.2) * 3.0) if t >= TL.ATO_TABLE + 1.2 else 0.0
        if fl > 0:
            add_sprite(F, CX + 46, 420 + 86 * 4.5, 340, GOLD, 0.25 * fl * ta)
        serif("（数值为示意）", 24, 400, 0.1).draw(F, 960, 1262, DIM, 0.8 * ta, align="right")
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

def qkv(F, t, g, fi):
    A = window(t, 0.0, 41.6, 0.6, 0.9)
    sc = 1 - 0.93 * smooth(ramp(t, TL.QKV_FORMULA - 0.4, 1.0))       # the scene steps back for the formula
    k = np.array([ramp(t, TL.QKV_K + 0.12 * i, 0.5) for i in range(8)])
    v = np.array([ramp(t, TL.QKV_V + 0.1 * i, 0.5) for i in range(8)])
    P.qkv_plate(F, words=smooth(ramp(t, 0.4, 1.0)), q=ramp(t, TL.QKV_Q, 0.6), k=k, v=v,
                match=ramp(t, TL.QKV_MATCH, 1.6), flow=ramp(t, TL.QKV_MATCH + 3.4, 2.0),
                it_a=smooth(ramp(t, 0.8, 1.0)), a=A * sc)
    if TL.QKV_Q <= t < TL.QKV_Q + 1.0:
        add_sprite(F, P.Q_CARD[0], P.Q_CARD[1], 80, GOLD, 0.4 * math.exp(-(t - TL.QKV_Q) * 3) * A)
    fa = smooth(ramp(t, TL.QKV_FORMULA, 1.0)) * A
    if fa > 0:
        _formula(F, CX, 740, fa)
        gl = smooth(ramp(t, TL.QKV_FORMULA + 1.2, 0.8)) * A
        for kk, (sym, zh) in enumerate((("Q", "提问"), ("K", "标签"), ("V", "内容"))):
            x = CX - 230 + 230 * kk
            T(sym, "stix_it", 40, None, per_char=False).draw(F, x - 26, 990, GOLD, gl, align="center")
            serif(zh, 30, 500, 0.1).draw(F, x + 24, 988, IVORY, 0.85 * gl, align="center")
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



def _formula(F, cx, y, a):
    """Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V, set by hand from STIX pieces."""
    sz = 62

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
    l2a = [("=  softmax", "stix", sz, 0), ("(", "stix", sz, 0), ("QK", "stix_it", sz, 0), ("T", "stix", 37, -27),
           (" / ", "stix", sz, 0)]
    l2b = [("d", "stix_it", sz, 0), ("k", "stix_it", 37, 12), (")", "stix", sz, 0), (" V", "stix_it", sz, 0)]
    sq = T("√", "math", sz, None, per_char=False)
    w2 = run(l2a, 0, 0, True) + sq.width + run(l2b, 0, 0, True)
    x = cx - w2 / 2
    x = run(l2a, x, y + 110)
    sq.draw(F, x, y + 110, IVORY, a)
    xs = x + sq.width - 2
    x = run(l2b, xs, y + 110)
    hairline(F, xs, y + 110 - 57, xs + T("d", "stix_it", sz, None, per_char=False).width +
             T("k", "stix_it", 37, None, per_char=False).width + 4, IVORY, a, th=2)


# ----------------------------------------------------------------- heads ---

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
    P.columns(F, A * ease_out(ramp(t, 0.4, 0.6)), left_a=0.85, right_a=0.85)
    cur = int((t - TL.HEADS_T0) / TL.HEADS_DT) if t >= TL.HEADS_T0 else -1
    allin = smooth(ramp(t, TL.HEADS_ALL, 1.2))
    merge = smooth(ramp(t, 19.6, 1.6))
    on = np.zeros(8)
    for k, (name, desc, links) in enumerate(HEADS):
        start = TL.HEADS_T0 + k * TL.HEADS_DT
        solo = smooth(ramp(t, start, 0.35)) * (1 - smooth(ramp(t, start + TL.HEADS_DT, 0.35)))
        on[k] = max(solo, allin)
        a = A * max(solo, 0.55 * allin)
        if a <= 0.003:
            continue
        P.links_lines(F, links, mix(P.HEAD_COLS[k], GOLD, merge), a, grow=ease_out(ramp(t, start, 0.5)))
    P.head_squares(F, on, A * smooth(ramp(t, 1.0, 0.8)), y=330, merge=merge)
    if 0 <= cur < 8 and t < TL.HEADS_ALL:
        name, desc, _ = HEADS[cur]
        la = A * smooth(ramp(t, TL.HEADS_T0 + cur * TL.HEADS_DT, 0.3))
        serif(f"第 {cur + 1} 双眼睛 · {name}", 34, 500, 0.1).draw(F, CX, 296, P.HEAD_COLS[cur], la, align="center")
    if allin > 0:
        serif("多头注意力", 36, 500, 0.3).draw(F, CX, 296, GOLD, A * allin, align="center")
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
    y = 760
    sz = 150
    xs = [CX - 170, CX, CX + 170]
    chars = ["猫", "追", "狗"]
    run = 1 - smooth(ramp(t, TL.ORD_WAVES - 0.5, 0.8))               # a light trot until the waves arrive
    for k, ch in enumerate(chars):
        x = xs[k]
        yy = y
        if k == 0:
            x = lerp(xs[0], xs[2], sw)
            yy = y - 175 * math.sin(math.pi * sw)
        elif k == 2:
            x = lerp(xs[2], xs[0], sw)
            yy = y + 175 * math.sin(math.pi * sw)
        if k != 1:
            yy -= 9 * run * abs(math.sin(math.pi * (t - 0.6) / 0.333 + (0 if k == 0 else 1.2)))
        col = GOLD if ch in ("猫", "狗") else IVORY
        al = A * smooth(ramp(t, 0.6 + 0.2 * k, 0.6))
        if k == 1:
            al *= 1 - 0.6 * math.sin(math.pi * sw)
        serif(ch, sz, 500).draw(F, x, yy, col, al, align="center")
    caps("WHO CHASES WHOM", 20, 0.5).draw(F, CX, 385, DIM, 0.8 * A * window(t, 1.0, 11.0, 0.8, 0.8),
                                          align="center")
    # position waves: every place gets its own fingerprint
    wa = smooth(ramp(t, TL.ORD_WAVES, 1.2)) * A
    if wa > 0:
        freqs = [1.0, 2.1, 4.3, 8.5]
        for i, f in enumerate(freqs):
            yc = 930 + i * 72
            xx = np.linspace(150, 930, 320)
            ph = (xx - xs[0]) / 170.0
            yw = yc - 22 * np.sin(ph * f * 0.9)
            glow_poly(F, np.column_stack([xx, yw]), mix(STEEL, GOLD, i / 3), 0.6 * wa, th=1, glow=0.5, sigma=3)
            for k in range(3):
                v = -22 * math.sin(k * f * 0.9)
                orb(F, xs[k], yc + v, 5, GOLD, 0.95 * wa * smooth(ramp(t, TL.ORD_WAVES + 1.0 + 0.4 * k, 0.5)))
        for k in range(3):
            for yy in np.arange(900, 1170, 9):
                add_sprite(F, xs[k], yy, 1.0, IVORY, 0.25 * wa)
            T(str(k + 1), "stix", 28, None, per_char=False).draw(F, xs[k], 1215, DIM, wa, align="center")
        caps("POSITION", 15, 0.5).draw(F, CX, 1252, DIM, 0.8 * wa, align="center")
    subtitles(F, t, [(0.6, 5.6, "可是，同时看所有的字，就分不清谁先谁后了。",
                      "But seeing every word at once means losing their order."),
                     (6.0, 11.2, "「猫追狗」和「狗追猫」，用的是同样三个字。",
                      "'Cat chases dog' and 'dog chases cat' use the same three characters."),
                     (11.6, 18.6, "于是，每个位置都被印上一组快慢不同的波，像一个独一无二的指纹。",
                      "So every position is stamped with waves of different speeds — a fingerprint all its own."),
                     (19.0, 25.4, "有了位置，它才分得清：谁在追谁。", "With positions, it can tell who is chasing whom.")])


# ----------------------------------------------------------------- after ---

def _title_echo(F, y, a, gold_from, size=40):
    """'Attention Is All You Need' with 'All You Need' picked out in gold."""
    full = "Attention Is All You Need"
    tx = T(full, "corm_it", size, 500, per_char=False)
    x0 = CX - tx.width / 2
    tx.draw(F, x0, y, IVORY, 0.7 * a)
    if gold_from > 0:
        pre = T("Attention Is ", "corm_it", size, 500, per_char=False).width
        T("All You Need", "corm_it", size, 500, per_char=False).draw(F, x0 + pre, y, GOLD, a * gold_from)


def _song(F, y, a, gold_from, size=58):
    full = "All You Need Is Love"
    tx = T(full, "corm_it", size, 500, per_char=False)
    x0 = CX - tx.width / 2
    tx.draw(F, x0, y, IVORY, a)
    if gold_from > 0:
        T("All You Need", "corm_it", size, 500, per_char=False).draw(F, x0, y, GOLD, a * gold_from)


def after(F, t, g, fi):
    A = window(t, 0.0, 29.6, 0.6, 0.9)
    # the first page
    pa = A * smooth(ramp(t, 0.3, 0.9)) * (1 - smooth(ramp(t, TL.AFT_BEATLES - 0.6, 0.8)))
    if pa > 0.003:
        P.page(F, pa, note=ease_out(ramp(t, TL.AFT_BYLINE + 0.3, 0.9)))
    # the Beatles
    ba = A * window(t, TL.AFT_BEATLES, TL.AFT_GPT + 0.2, 0.8, 0.8)
    if ba > 0.003:
        P.record(F, CX, 720, 250, t, ba)
        echo = smooth(ramp(t, TL.AFT_BEATLES + 1.6, 0.8))
        _title_echo(F, 400, ba, echo)
        _song(F, 1070, ba, echo)
        caps("THE BEATLES   ·   1967", 20, 0.45).draw(F, CX, 1125, DIM, ba, align="center")
    # GPT: the T is the Transformer, and the stack of layers
    ga = A * smooth(ramp(t, TL.AFT_GPT, 0.8))
    if ga > 0.003:
        lift = 320 * ease_in_out(ramp(t, TL.AFT_TOWER, 1.2))
        for k, ch in enumerate("GPT"):
            T(ch, "corm", 160, 600, per_char=False).draw(F, CX - 118 + 118 * k, 780 - lift,
                                                        GOLD if ch == "T" else IVORY, ga, align="center")
        caps("GENERATIVE PRE-TRAINED TRANSFORMER", 20, 0.4).draw(F, CX, 845 - lift, DIM, ga, align="center")
        ta = A * smooth(ramp(t, TL.AFT_TOWER + 0.6, 0.6))
        if ta > 0.003:
            n_blocks = int(1 + 8 * ease_out(ramp(t, TL.AFT_TOWER + 0.6, 4.0)))
            n_blocks = min(n_blocks, 8)
            sx = CX - 70
            for k in range(n_blocks):
                yb = 1210 - 30 * k
                top = k == n_blocks - 1
                # each new layer covers the ones beneath it, like a card laid on a deck
                cover = np.full((150, 400), 1.0, np.float32)
                blend(F, cover, sx - 200, yb - 150, np.array([0.02, 0.028, 0.045], np.float32), 0.92 * ta)
                P.layer(F, sx, yb, 400, ta * (0.6 + 0.4 * top), detail=1.0 if top else 0.0,
                        glow=0.5 * smooth(ramp(t, TL.AFT_TOWER + 4.6, 0.6)) if top else 0.0)
            na = A * smooth(ramp(t, TL.AFT_TOWER + 4.6, 0.6))
            T("N×", "stix", 40, None, per_char=False).draw(F, sx - 230, 1120, GOLD, na, align="right")
        for k, (yr, nm) in enumerate((("2018", "GPT"), ("2018", "BERT"), ("2022", "ChatGPT"))):
            la = A * smooth(ramp(t, TL.AFT_TOWER + 1.2 + 1.0 * k, 0.6))
            y = 1150 - 150 * k
            T(yr, "stix", 24, None, per_char=False).draw(F, 965, y, DIM, la, align="right")
            T(nm, "corm", 40, 600, per_char=False).draw(F, 965, y + 40, IVORY, la, align="right")
    subtitles(F, t, [(0.6, 5.0, "2017 年 6 月，这篇论文发表，作者一共八位。",
                      "June 2017: the paper comes out, with eight authors."),
                     (5.4, 10.8, "署名旁边写着：贡献相同，排名随机。",
                      "A note by the names: equal contribution, order random."),
                     (11.2, 16.6, "而它的标题，是在致敬披头士的一首歌。", "And its title tips its hat to a Beatles song."),
                     (17.0, 22.2, "后来，GPT 里的「T」，就是 Transformer。", "Later, the 'T' in GPT stood for Transformer."),
                     (22.6, 29.4, "你和 AI 的每一次对话，背后都是这一行公式，一层一层地叠起来。",
                      "Every conversation you have with AI runs on that one line, stacked layer upon layer.")])


# ---------------------------------------------------------------- ending ---

def ending(F, t, g, fi):
    A = window(t, 0.0, 24.0, 0.8, 0.01)
    P.stargazer(F, a=A, sky=smooth(ramp(t, 0.5, 3.5)), tilt=0.12 + 0.30 * ease_in_out(ramp(t, 1.0, 3.0)),
                chart=ramp(t, 5.0, 6.0) * (1 - 0.45 * smooth(ramp(t, TL.END_LINE, 1.0))),
                figure=smooth(ramp(t, 0.3, 2.0)) * (1 - 0.35 * smooth(ramp(t, TL.END_TITLE, 1.5))))
    statement(F, "注意力，", CX + 40, 670, t, TL.END_LINE, None, size=64, cps=5, gold={0, 1, 2})
    statement(F, "也许真的就是你所需要的一切。", CX + 20, 765, t, TL.END_LINE + 1.0, None, size=50, cps=7)
    ta = smooth(ramp(t, TL.END_TITLE, 1.2))
    T("Attention Is All You Need", "corm_it", 46, 500, per_char=False).draw(F, CX + 80, 890, IVORY, ta,
                                                                            align="center")
    caps("VASWANI ET AL.   ·   2017", 18, 0.45).draw(F, CX + 80, 934, DIM, ta, align="center")
    subtitles(F, t, [(0.8, 5.6, "机器学会了注意力，于是开始读懂语言。", "Machines learned attention, and began to read language."),
                     (6.0, 11.8, "而在一个什么都在争夺你注意力的时代——",
                      "And in an age when everything is competing for your attention —"),
                     (TL.END_LINE + 0.4, 18.6, "", "perhaps attention really is all you need.")])


SCENE_FUNCS = {"prologue": prologue, "sequential": sequential, "atonce": atonce, "qkv": qkv, "heads": heads,
               "order": order, "after": after, "ending": ending}
