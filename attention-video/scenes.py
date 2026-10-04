"""注意力 · Attention Is All You Need.  draw(F, t_local, g_global, frame).  Third cut.

An AI chatbot only ever guesses the next word; to guess well it must know who 「它」
is.  Words are numbers - stars in a sky of meaning - and attention is how each
star looks around and moves to where it belongs: a question, a label, a
content; multiply and add; softmax; mix.  Then layers, training, and the
bigger question: why guessing the next word looks like intelligence.

The pictures follow classic references (see plates.py): a star-atlas plate,
cut-paper portraits, the authors' own two-column view, a 3Blue1Brown-style
grid, Kepler's ellipse, a murmuration, a Cajal neuron, the paper's first page."""

import math

import numpy as np

import plates as P
import timeline as TL
import style as ST
from style import (CX, DIM, GOLD, IVORY, WHITE, T, add_sprite, blend, caps, comet, ease_in, ease_in_out, ease_out,
                   glow_poly, hairline, lerp, mix, orb, punch, ramp, shock, smooth, statement, subtitles, window, zh)

STEEL = P.STEEL


# -------------------------------------------------------------- sentence ---

class Sentence:
    """A sentence laid out on lines; tokens are spans of characters; arcs connect tokens."""

    def __init__(self, lines, tokens, size, ys, cx=520, tracking=0.12):
        self.lines, self.tokens, self.size, self.ys = lines, tokens, size, ys
        self.txt = [zh(l, size, 400, tracking) for l in lines]
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
                    alt = zh(self.lines[ln][:i] + swap[2], self.size, 400, self.tracking).items[i]
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


def _burst(F, q, a, cy):
    P0 = np.random.default_rng(3).normal(0, 1, (140, 2))
    if q > 3.0:
        return
    sp = 1 - math.exp(-q * 2.2)
    for vx, vy in P0:
        add_sprite(F, CX + vx * 300 * sp, cy + vy * 160 * sp, 1.2, GOLD, 0.6 * a * max(0.0, 1 - q / 3.0))


def title_card(F, t, t0, t_fade):
    """The paper's title arriving: a point of light, two comets, the words."""
    cy = 760
    o = smooth(ramp(t, t0 - 0.7, 0.7)) * (1 - smooth(ramp(t, t0, 0.25)))
    orb(F, CX, cy, 9 + 8 * o, GOLD, 1.3 * o)
    if t < t0:
        return
    q = t - t0
    fade = 1 - smooth(ramp(t, t_fade, 1.1))
    hx = CX + 640 * ease_out(min(q / 2.6, 1.0))
    ca = 0.85 * fade * (1 - smooth(ramp(q, 2.2, 0.5)))
    comet(F, hx, cy + 150, 1, 260, GOLD, ca)
    comet(F, 2 * CX - hx, cy + 150, -1, 260, GOLD, ca)
    _burst(F, q, fade, cy)
    for k, (line, y) in enumerate((("Attention", 690), ("Is All You Need", 806))):
        tx = T(line, "corm", 104, 600, tracking=0.02)
        al = fade * ease_out(ramp(t, t0 + 0.1 + 0.5 * k, 1.0))
        tx.draw(F, CX, y + (1 - al) * 10, IVORY, al, align="center", glow=0.25, glow_color=GOLD, glow_sigma=14)
    fa = fade * smooth(ramp(t, t0 + 1.6, 0.9))
    zh("注意力，就是你所需要的一切", 42, 500, 0.2).draw(F, CX, 905, GOLD, fa, align="center")
    hairline(F, CX - 80 * fa, 945, CX + 80 * fa, GOLD, 0.6 * fa)
    caps("VASWANI ET AL.   ·   2017", 20, 0.42).draw(F, CX, 990, DIM, fa, align="center")


# ---------------------------------------------------------------- camera ---

def kick(age):
    """A small jolt of the camera when something lands."""
    return 0.05 * math.exp(-age * 5.0) if age >= 0 else 0.0


def cam(t, dur, extra=0.0, x=None, y=None, out_at=None, out_len=0.8):
    """The default slow push-in, plus jolts, plus an optional zoom-through at the end of the shot."""
    sc = 1.0 + 0.035 * t / dur + extra
    blur = 0.0
    if out_at is not None and t > out_at:
        u = ease_in(ramp(t, out_at, out_len))
        sc *= 1 + 2.2 * u
        blur = 6 * u
    ST.camera(scale=sc, x=x, y=y, blur=blur)


_FORM = {}


def _formula_mask():
    """softmax(QKᵀ/√dₖ)V drawn once into a mask, so it can be scaled freely."""
    if "m" not in _FORM:
        tmp = np.zeros((260, 1060, 3), np.float32)
        sz = 84
        x = 30
        y = 170
        for p_, f, dy, s_ in (("softmax(", "stix", 0, sz), ("QK", "stix_it", 0, sz), ("T", "stix", -40, 50),
                               (" / ", "stix", 0, sz), ("√", "math", 0, sz), ("d", "stix_it", 0, sz),
                               ("k", "stix_it", 16, 50), (")", "stix", 0, sz), ("V", "stix_it", 0, sz)):
            tx = T(p_, f, s_, None, per_char=False)
            tx.draw(tmp, x, y + dy, WHITE, 1.0)
            if p_ == "d":
                xs = x
            if p_ == "k":
                hairline(tmp, xs - 2, y - 76, x + tx.width + 6, WHITE, 1.0, th=3)
            x += tx.width + 2
        m = tmp[..., 0]
        ys, xs_ = np.nonzero(m > 0.01)
        m = m[max(0, ys.min() - 20):ys.max() + 20, max(0, xs_.min() - 20):xs_.max() + 20]
        _FORM["m"] = m
    return _FORM["m"]


def formula_big(F, cx, cy, t, t0, a=1.0, scale=1.0):
    if t < t0 or a <= 0.003:
        return
    import cv2
    m = _formula_mask()
    u = ease_out(ramp(t, t0, 0.35))
    k = scale * (1 + 0.4 * (1 - u))
    al = a * min(1.0, (t - t0) / 0.1)
    h, w = m.shape
    ms = cv2.resize(m, (max(1, int(w * k)), max(1, int(h * k))), interpolation=cv2.INTER_LINEAR)
    x0, y0 = cx - ms.shape[1] / 2, cy - ms.shape[0] / 2
    blend(F, ms, x0, y0, IVORY, al)
    ST.add_light(F, cv2.GaussianBlur(ms, (0, 0), 14), x0, y0, GOLD, (0.35 + 1.2 * math.exp(-(t - t0) * 4)) * al)


def ai_label(F, a=1.0, y=250):
    """The disclosure required for AI-generated video: a small, clear tag at the top of the frame."""
    if a <= 0.003:
        return
    tx = zh("本视频由 AI 辅助生成", 28, 500, 0.08)
    w = tx.width + 44
    st = ST.Stroke(CX - w / 2 - 3, y - 26, CX + w / 2 + 4, y + 27)
    st.poly([(CX - w / 2, y - 23), (CX + w / 2, y - 23), (CX + w / 2, y + 23), (CX - w / 2, y + 23)], closed=True)
    blend(F, np.full((46, int(w)), 1.0, np.float32), CX - w / 2, y - 23, np.array([0.0, 0.0, 0.0], np.float32),
          0.45 * a)
    st.light(F, IVORY, 0.55 * a)
    tx.draw(F, CX, y + 10, IVORY, 0.92 * a, align="center")


def cold(F, t, g, fi):
    """The puzzle is on screen from the very first frame: no logo, no build-up."""
    a = 1 - smooth(ramp(t, TL.CO_L3 - 0.05, 0.3))
    punch(F, "下一个字是什么？", CX, 450, t, TL.CO_FORM - 0.12, size=86, wght=800, a=a, gold=(2, 3))
    zh("小猫没有跳上桌子，", 70, 500, 0.12).draw(F, CX, 640, IVORY, a, align="center")
    fill = smooth(ramp(t, TL.CO_L2 - 0.5, 0.25))
    P.guess_line(F, "因为它太", CX, 750, 70, a=a, t=t, fill="累", fill_a=fill)
    if TL.CO_L2 - 0.5 <= t < TL.CO_L2 + 0.5:
        add_sprite(F, CX + 175, 725, 70, GOLD, 0.6 * math.exp(-(t - TL.CO_L2 + 0.5) * 4) * a)
    ba = a * smooth(ramp(t, TL.CO_L1, 0.25))
    P.bars(F, GUESS[:3], 330, 905, 520, a=ba, grow=ramp(t, TL.CO_L1 + 0.15, 2.0), row=92, size=54, hot=0,
           note="（示意）")
    punch(F, "AI 每说一句话，都在玩这个游戏。", CX, 1268, t, TL.CO_L2, size=56, wght=700, a=a,
          gold=(11, 12, 13, 14))
    shock(F, CX, 1245, t - TL.CO_L2, 0.5 * a, size=0.6)
    # the promise, then straight through it into the film
    b = 1 - smooth(ramp(t, TL.CO_GO + 0.6, 0.4))
    punch(F, "7 分钟，", CX, 700, t, TL.CO_L3, size=110, wght=800, a=b, gold=(0,))
    punch(F, "看懂它怎么猜。", CX, 860, t, TL.CO_L3 + 0.25, size=96, wght=800, a=b, gold=(2, 3, 4, 5))
    shock(F, CX, 760, t - TL.CO_L3, b, size=1.1)
    cam(t, 6, extra=kick(t - TL.CO_L1) * 0.5 + kick(t - TL.CO_L2) + kick(t - TL.CO_L3), y=820,
        out_at=TL.CO_GO, out_len=1.0)
    subtitles(F, t, [(0.0, TL.CO_L2 - 0.1, "", "What's the next word?"),
                     (TL.CO_L2, TL.CO_L3 - 0.1, "", "Every sentence an AI writes is this game."),
                     (TL.CO_L3, 5.8, "", "In 7 minutes: how it guesses.")])


# ------------------------------------------------------------------ hook ---

GUESS = [("累", .41), ("高", .23), ("小", .09), ("胖", .06), ("困", .05)]


def hook(F, t, g, fi):
    # the chat: a whole answer, one guessed character at a time
    ca = window(t, TL.H_CHAT, TL.H_PLATE - 0.1, 0.4, 0.6)
    if ca > 0.003:
        u = ramp(t, TL.H_CHAT + 0.6, 8.4)
        n = len(P.CHAT_A) * (0.75 * u + 0.25 * u ** 2)
        P.chat(F, P.CHAT_Q, P.CHAT_A, n, t, a=ca, y0=420, cand_y=1090)
    # who is 「它」? the star-atlas plate
    pa = window(t, TL.H_PLATE, TL.H_TURN - 0.1, 0.6, 0.8)
    if pa > 0.003:
        hop = math.sin(math.pi * ramp(t, TL.H_HOP, 0.5)) if TL.H_HOP <= t <= TL.H_HOP + 0.5 else 0.0
        sleep = ease_in_out(ramp(t, TL.H_LIE, 0.6)) * (1 - ease_in_out(ramp(t, TL.H_SWAP + 0.1, 0.8)))
        P.opening(F, w_cat=smooth(ramp(t, TL.H_LINK + 0.3, 0.8)) * (1 - smooth(ramp(t, TL.H_RE, 1.0))),
                  w_table=smooth(ramp(t, TL.H_RE, 1.0)), sleep=sleep,
                  h=1.0 + 0.75 * ease_in_out(ramp(t, TL.H_SWAP, 1.0)), it_on=smooth(ramp(t, TL.H_IT, 0.3)),
                  link=ease_out(ramp(t, TL.H_LINK, 0.9)), link_table=ease_out(ramp(t, TL.H_RE, 0.9)),
                  reveal=smooth(ramp(t, TL.H_PLATE, 1.4)), swap=smooth(ramp(t, TL.H_SWAP, 0.55)), a=pa, hop=hop,
                  chart=ramp(t, TL.H_PLATE, 1.2))
        if TL.H_IT <= t < TL.H_IT + 2:
            sx, sy = P.OPEN.centers[IT]
            add_sprite(F, sx, sy, 60, GOLD, 0.5 * math.exp(-(t - TL.H_IT) * 2.2) * pa)
        if TL.H_SWAP <= t < TL.H_SWAP + 0.9:
            sx, sy = P.OPEN.centers[7]
            q = (t - TL.H_SWAP) / 0.9
            add_sprite(F, sx, sy, 40 + 60 * q, GOLD, 0.7 * math.sin(math.pi * q) * pa)
    subtitles(F, t, [(0.3, 4.6, "看它回答问题：每个字出现之前，它都在猜。",
                      "Watch it answer: before every character appears, it is guessing."),
                     (5.0, 9.2, "猜一个，接上，再猜下一个——一整段回答，就是这样写出来的。",
                      "Guess one, add it, guess again - that is how a whole answer is written."),
                     (9.6, 14.0, "可要猜得准，它得先弄明白：这里的「它」，到底是谁？",
                      "But to guess well, it must first work out who 'it' is."),
                     (14.4, 18.8, "换一个字，「它」就换了对象。你一眼就知道。",
                      "Change one word and 'it' changes too. You see it at a glance.")])
    statement(F, "而让机器也学会这件事的，", CX, 720, t, TL.H_TURN, TL.H_TITLE - 0.5, size=58, cps=11)
    statement(F, "是 2017 年的一篇论文。", CX, 820, t, TL.H_TURN + 1.1, TL.H_TITLE - 0.5, size=58, cps=11,
              gold={2, 3, 4, 5})
    title_card(F, t, TL.H_TITLE, TL.H_TITLE + 4.4)
    shock(F, CX, 760, t - TL.H_TITLE, 0.8)
    # the open loop: the question the last third answers
    punch(F, "看到最后：只会猜字的 AI，", CX, 1110, t, TL.H_TITLE + 2.4, size=46, wght=600,
          a=1 - smooth(ramp(t, TL.H_TITLE + 4.6, 0.5)))
    punch(F, "为什么会变聪明？", CX, 1190, t, TL.H_TITLE + 2.9, size=62, wght=800, gold=(0, 1, 2, 3, 4, 5, 6),
          a=1 - smooth(ramp(t, TL.H_TITLE + 4.6, 0.5)))
    # close in on the chat (phone-sized text), then pull back for the plate
    zc = 1 - smooth(ramp(t, TL.H_PLATE - 0.7, 1.0))
    cam(t, 29.6, extra=0.18 * zc + kick(t - TL.H_IT) * 0.6 + kick(t - TL.H_SWAP) * 0.6 + kick(t - TL.H_TITLE)
        + kick(t - TL.H_TITLE - 2.9) * 0.5, y=860 - 80 * zc, out_at=28.6, out_len=1.0)


# ------------------------------------------------------------------- sky ---

NUMS = [0.21, -1.30, 0.84, 0.05, -0.62, 1.17, -0.40, 0.33, 0.91, -0.08, 0.57, -1.02, 0.14, 0.76]
APPLE_HOME = np.array([800, 990])
APPLE_FRUIT = np.array([815, 1130])
APPLE_PHONE = np.array([795, 880])


def sky_scene(F, t, g, fi):
    A = window(t, 0.0, 45.6, 0.3, 0.6)
    ST.warp(F, t, 1.5)
    cam(t, 46, extra=kick(t - TL.S_STAR - 1.2) * 0.6, out_at=45.2, out_len=0.8)
    # one word becomes a list of numbers
    na = A * (1 - smooth(ramp(t, TL.S_STAR, 0.8)))
    if na > 0.003:
        zh("猫", 170, 500, 0).draw(F, 290, 760, IVORY, na * smooth(ramp(t, 1.0, 0.6)), align="center")
        n = 1 + 3.5 * max(0.0, t - TL.S_NUM - 0.8)
        P.numcol(F, 720, 400, NUMS, min(n, len(NUMS)), na * smooth(ramp(t, TL.S_NUM + 0.6, 0.5)), size=36, gap=50)
        if n > len(NUMS):
            zh("⋮", 44, 500, 0).draw(F, 690, 400 + 50 * len(NUMS) + 10, IVORY, na, align="center")
        ca = na * smooth(ramp(t, TL.S_COUNT, 0.6))
        if ca > 0:
            v = 512 if t < TL.S_COUNT + 2.6 else 512 + (12288 - 512) * ease_in_out(ramp(t, TL.S_COUNT + 2.6, 1.4))
            P.counter(F, v, 290, 1000, ca, size=72, col=GOLD)
            zh("个数" if t < TL.S_COUNT + 2.6 else "个数 · GPT-3", 30, 500, 0.1).draw(F, 290, 1060, DIM, ca,
                                                                                    align="center")
            if t < TL.S_COUNT + 2.6:
                zh("论文里", 30, 500, 0.1).draw(F, 290, 900, DIM, ca, align="center")
    # the numbers become a star and fly to their place
    fly = ramp(t, TL.S_STAR, 1.2)
    if 0 < fly < 1:
        p0, p1 = np.array([660, 700]), np.array(P.SKY_WORDS["小猫"], float)
        u = ease_in_out(fly)
        p = p0 + (p1 - p0) * u + np.array([0, -120]) * math.sin(math.pi * u)
        orb(F, p[0], p[1], 8, GOLD, A)
    sa = A * smooth(ramp(t, TL.S_STAR + 0.8, 0.6))
    if sa > 0.003:
        king = smooth(ramp(t, TL.S_KING, 0.6)) * (1 - smooth(ramp(t, 25.8, 0.6)))
        others = 1 - 0.65 * king
        apple = smooth(ramp(t, 26.2, 0.6))
        it_on = smooth(ramp(t, TL.S_IT, 0.6))
        dim = {w: others for w in P.SKY_WORDS}
        for w in ("男人", "女人", "国王", "女王"):
            dim[w] = 1.0 if king > 0.01 else 0.75
        if apple > 0:
            for w in P.SKY_WORDS:
                if w not in ("香蕉", "橙子", "手机", "公司"):
                    dim[w] = min(dim[w], 1 - 0.5 * apple * (1 - it_on))
        P.sky(F, sa, groups=ramp(t, TL.S_GROUPS, 2.0), dim=dim,
              glow={"男人": king * 0.6, "女人": king * 0.6, "国王": king * 0.6, "女王": king * 0.6})
        if king > 0:
            for k, (p_, q_) in enumerate((("男人", "女人"), ("国王", "女王"))):
                gk = ease_out(ramp(t, TL.S_KING + 0.6 + 1.0 * k, 0.8))
                pp, qq = np.array(P.SKY_WORDS[p_]) + [8, -2], np.array(P.SKY_WORDS[q_]) - [12, -3]
                P.arrow(F, pp, pp + (qq - pp) * gk, GOLD, 0.9 * king * sa)
        if apple > 0:
            m1 = ease_in_out(ramp(t, TL.S_APPLE + 0.4, 1.0)) * (1 - ease_in_out(ramp(t, TL.S_APPLE + 2.6, 0.8)))
            m2 = ease_in_out(ramp(t, TL.S_APPLE + 2.6, 1.0))
            pa_ = APPLE_HOME + (APPLE_FRUIT - APPLE_HOME) * m1 + (APPLE_PHONE - APPLE_HOME) * m2
            aa = apple * (1 - 0.5 * it_on)
            P.sky_word(F, "苹果", pa_, sa * aa, glow=0.8, size=36, mag=1.3)
            for k, (s_, tt) in enumerate((("我吃了一个苹果", TL.S_APPLE), ("苹果发布了新手机", TL.S_APPLE + 2.6))):
                la = window(t, tt, tt + 2.4 if k == 0 else TL.S_IT, 0.4, 0.4) * sa
                zh(s_, 40, 500, 0.1).draw(F, CX, 1260, GOLD, la, align="center")
        if it_on > 0:
            pull = ease_in_out(ramp(t, TL.S_PULL + 1.2, 2.4))
            tgt = np.array(P.SKY_WORDS["小猫"], float) + [70, 80]
            p = P.IT_HOME + (tgt - P.IT_HOME) * 0.75 * pull
            tw = 0.6 + 0.4 * math.sin(t * 3.0)
            P.sky_word(F, "它", p, sa * it_on * (0.6 + 0.4 * tw * (1 - pull) + 0.4 * pull), glow=0.3 + 0.7 * pull,
                       size=42, mag=0.6 + 0.8 * pull)
            th = ease_out(ramp(t, TL.S_PULL, 1.0))
            if th > 0:
                P.thread(F, p, np.array(P.SKY_WORDS["小猫"]) + [10, 12], 0.9 * sa, bow=30, grow=th)
    subtitles(F, t, [(0.4, 4.0, "机器不认识字，只认识数字。", "Machines don't read words, only numbers."),
                     (4.2, 9.4, "所以第一步，把每个字变成一串数字：论文里是 512 个，GPT-3 里是 12288 个。",
                      "So first, each word becomes a list of numbers: 512 in the paper, 12,288 in GPT-3."),
                     (9.8, 14.6, "把这串数字想成一个位置——每个字，都是「意义星空」里的一颗星。",
                      "Think of the numbers as a place: every word is a star in a sky of meaning."),
                     (15.0, 19.0, "意思相近的字，挨得也近。", "Words with similar meanings sit close together."),
                     (19.4, 25.6, "连方向都有意思：从「男人」到「女人」，和从「国王」到「女王」，几乎是同一步。",
                      "Even directions mean something: man to woman is almost the same step as king to queen."),
                     (26.0, 29.6, "可同一个字，在不同的句子里，意思并不一样。",
                      "But the same word can mean different things in different sentences."),
                     (30.0, 35.0, "同一个「苹果」，放进两句话里，就成了两个意思。",
                      "The same 'apple', put in two sentences, means two different things."),
                     (35.4, 39.4, "而「它」自己几乎没有意思，孤零零地悬在中间。",
                      "And 'it', on its own, means almost nothing - it hangs alone in the middle."),
                     (39.8, 45.6, "注意力要做的，就是让每个字看看身边的字，再把自己挪到该去的地方。",
                      "Attention lets each word look at the others, then move to where it belongs.")])


# ------------------------------------------------------------ sequential ---

SQ_S = Sentence(("小猫没有跳上桌子，", "因为它太累了。", "它整整睡了一个下午。"),
                SENT_TOKENS + [(2, 0, 1, "它"), (2, 1, 3, "整整"), (2, 3, 4, "睡"), (2, 4, 5, "了"), (2, 5, 8, "一个下"),
                               (2, 8, 9, "午")], 66, (1000, 1110, 1220))


def _read_pos(t):
    """Index of the token being read (fractional), one word at a time."""
    if t < TL.SEQ_READ0:
        return -1.0
    n = (t - TL.SEQ_READ0) / TL.SEQ_READ_DT
    if t > 16.4:                                              # the sentence grows; reading crawls on
        n = (16.4 - TL.SEQ_READ0) / TL.SEQ_READ_DT + (t - 16.4) / 1.0
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
    A = window(t, 0.0, 29.6, 0.8, 0.9)
    pa = A * (1 - 0.55 * smooth(ramp(t, 11.0, 1.5)))
    each = [smooth(ramp(t, 0.4 + 0.3 * k, 1.0)) for k in range(4)]
    busts = P.whisper_row(F, y=560, s=165, a=pa, each=each)
    fade_out = 1 - smooth(ramp(t, 10.6, 1.0))
    for k in range(3):
        t0 = TL.SEQ_PASS0 + k * TL.SEQ_PASS_DT
        q = ramp(t, t0, 1.1)
        held = k == 0 and t0 - 1.4 <= t < t0
        last = k == 2 and t >= t0 + 1.1
        if 0 < q < 1 or held or last:
            aa = A * fade_out * (smooth(ramp(t, t0 - 1.4, 0.5)) if held else 1.0)
            P.message(F, busts, k, ease_in_out(q), aa)
    pos = _read_pos(t)
    n = len(SQ_S.tokens)
    alphas = np.full(n, 0.22)
    cols = [IVORY] * n
    for k in range(n):
        if pos >= k:
            alphas[k] = 0.16 + 0.84 * math.exp(-(pos - k) / 2.6)
    cur = int(pos) if pos >= 0 else -1
    if cur >= 0:
        cols[cur] = GOLD
        alphas[cur] = 1.0
    sa = A * smooth(ramp(t, 5.0, 1.0))
    _draw_lines(F, SQ_S, sa, smooth(ramp(t, 15.6, 0.8)), alphas, cols)
    if cur >= 0:
        x, y, _, wd = SQ_S.centers[cur]
        hairline(F, x - wd / 2, int(y + 16), x + wd / 2, GOLD, 0.9 * sa)
        add_sprite(F, x, y - 24, 40, GOLD, 0.12 * sa)
    if pos >= IT:
        rb = smooth(ramp(t, TL.SEQ_READ0 + IT * TL.SEQ_READ_DT, 0.8)) * (1 - smooth(ramp(t, 16.0, 0.8)))
        if rb > 0:
            pts = SQ_S.arc(IT, CAT)
            for k in range(0, len(pts) - 1, 3):
                add_sprite(F, pts[k][0], pts[k][1], 1.4, STEEL, 0.5 * rb * sa)
            zh("？", 44, 500, 0).draw(F, SQ_S.centers[CAT][0], SQ_S.centers[CAT][1] - 84, STEEL, 0.8 * rb * sa,
                                     align="center")
    sp = smooth(ramp(t, 22.0, 0.6)) * A
    if sp > 0 and cur >= 0:
        zh(f"第 {cur + 1} 步", 32, 500, 0.15).draw(F, 960, 880, IVORY, 0.85 * sp, align="right")
        caps("ONE WORD PER STEP", 15, 0.4).draw(F, 960, 912, DIM, 0.9 * sp, align="right")
    subtitles(F, t, [(0.8, 5.4, "在很长一段时间里，机器读句子，主要靠一个字一个字地读。",
                      "For a long time, machines read a sentence mostly one word at a time."),
                     (5.8, 10.6, "每读一个字，就把记住的东西，传给下一个字。",
                      "Each step passes what it remembers on to the next."),
                     (11.0, 16.0, "就像传话游戏：传得越远，开头就越模糊。",
                      "Like a game of telephone: the further it travels, the fainter the start."),
                     (16.4, 21.6, "等它读到「它」，「小猫」已经快想不起来了。",
                      "By the time it reaches 'it', 'the cat' is nearly forgotten."),
                     (22.0, 29.4, "而且只能排队，没法同时进行——句子越长，越慢，也越健忘。",
                      "And it must queue, never all at once: the longer the sentence, the slower and more forgetful.")])


# ------------------------------------------------------------------ core ---

def _overshoot(u):
    """0 -> 1 with a small bounce past 1: the dots pop."""
    if u <= 0:
        return 0.0
    if u >= 1:
        return 1.0
    c = 1.7
    v = u - 1
    return 1 + (c + 1) * v ** 3 + c * v ** 2


QKV_ROWS = [("Q", GOLD, "问题", "我在找什么？", TL.C_Q), ("K", mix(STEEL, IVORY, 0.2), "标签", "我是什么？", TL.C_K),
            ("V", IVORY, "内容", "选中我，就给你这个", TL.C_V)]
MIX_FINAL = (0.571 * np.array(P.SKY_WORDS["小猫"], float) + 0.210 * np.array(P.SKY_WORDS["累"], float)
             + 0.219 * P.IT_HOME)


def core(F, t, g, fi):
    A = window(t, 0.0, 97.6, 0.6, 0.8)
    cam(t, 98, extra=kick(t - TL.C_KNOW) + kick(t - TL.C_FORMULA) + kick(t - TL.C_GRID - 0.8) +
        0.6 * kick(t - TL.C_SUM - 2.3))
    # A. every word looks at every word - but whom should 它 look at?
    a1 = A * (1 - smooth(ramp(t, TL.C_MAT - 0.4, 0.6)))
    if a1 > 0.003:
        appear = ease_out(ramp(t, 0.4, 0.6))
        ask = smooth(ramp(t, TL.C_ASK, 0.8))
        web = ease_out(ramp(t, TL.C_WEB, 1.2)) * (1 - ask)
        if web > 0:
            for k in range(9):
                P.fan_lines(F, k, ATT[k], 0.35 * web * a1, ease_out(ramp(t, TL.C_WEB + 0.07 * k, 0.8)), gold=False,
                            floor=0.06)
        right_on = np.zeros(9)
        right_on[IT] = ask
        P.columns(F, a1 * appear, right_on=right_on, right_a=0.85 - 0.5 * ask)
        if ask > 0:
            P.links_lines(F, [(IT, j, 0.15) for j in range(9) if j != IT], STEEL, 0.6 * ask * a1,
                          grow=ease_out(ramp(t, TL.C_ASK, 0.9)))
            zh("？", 70, 500, 0).draw(F, P.COL_XR + 130, P.col_y(IT) + 26, GOLD, ask * a1, align="center")
    # B. three tables turn a word's numbers into a question, a label and a content
    a2 = A * window(t, TL.C_MAT, 25.6, 0.6, 0.6)
    if a2 > 0.003:
        zh("它", 64, 500, 0).draw(F, 150, 400, GOLD, a2, align="center")
        P.numcol(F, 205, 500, NUMS[:8], 1 + 7 * max(0.0, t - TL.C_MAT), a2, size=30, gap=44)
        for k, (letter, col, name, meaning, tk) in enumerate(QKV_ROWS):
            ak = a2 * smooth(ramp(t, TL.C_MAT + 0.8 + 0.4 * k, 0.6))
            nxt = QKV_ROWS[k + 1][4] if k < 2 else 99
            hl = smooth(ramp(t, tk, 0.5)) * (1 - 0.55 * smooth(ramp(t, nxt, 0.5)))
            dim_ = 0.55 + 0.45 * max(hl, 1.0 if t < TL.C_Q else 0.0)
            y = 420 + 280 * k
            zh("×", 40, 400, 0).draw(F, 258, y + 88, IVORY, ak * dim_, align="center")
            P.matrix(F, 290, y, 8, 6, 24, "W", a=ak * dim_, seed=k, col=mix(col, IVORY, 0.3), sub=letter)
            gk = ease_out(ramp(t, TL.C_MAT + 1.6 + 0.4 * k, 0.6))
            if gk > 0:
                P.arrow(F, (500, y + 72), (500 + 76 * gk, y + 72), IVORY, ak * dim_ * 0.8, th=2, head=12)
            P.vec(F, 596, y, 6, 24, letter, col, a=ak * dim_ * gk, seed=k + 5)
            ta = ak * (0.35 + 0.65 * hl)
            zh(name, 40, 600, 0.1).draw(F, 680, y + 62, mix(IVORY, col, hl), ta)
            zh(meaning, 28, 400, 0.04).draw(F, 680, y + 112, IVORY, ta * 0.9)
            if hl > 0.05:
                add_sprite(F, 608, y + 72, 70, col, 0.12 * hl * a2)
    # C. the question meets the labels on a little chart
    a3 = A * window(t, 26.0, 51.6, 0.6, 0.6)
    if a3 > 0.003:
        keys = [smooth(ramp(t, TL.C_KARROWS + 0.6 * i, 0.5)) for i in range(4)]
        which = 0 if t < TL.C_ALIGN + 2.2 else 3
        focus = which if TL.C_SUM <= t < TL.C_SCORES else None
        P.dot_chart(F, a3, q=ramp(t, TL.C_QARROW, 0.8), keys=keys, focus=focus)
        if TL.C_SUM <= t < TL.C_SCORES:
            t0 = TL.C_SUM if which == 0 else TL.C_ALIGN + 2.2
            t1 = TL.C_ALIGN + 2.0 if which == 0 else TL.C_SCORES - 0.1
            P.dot_sum(F, 1225, a3 * window(t, t0, t1, 0.3, 0.3), which, reveal=ramp(t, t0, 2.4))
            shock(F, 760, 1210, t - t0 - 2.3, 0.5 * a3, size=0.35)
        sc = smooth(ramp(t, TL.C_SCORES, 0.6)) * a3
        if sc > 0:
            for i, (name, v) in enumerate(P.K_VECS):
                s_ = int(round(float(P.Q_VEC @ v)))
                gi = smooth(ramp(t, TL.C_SCORES + 0.5 * i, 0.4))
                zh(f"{name}  {s_} 分", 36, 500, 0.06).draw(F, 960, 470 + 60 * i, GOLD if i == 0 else IVORY,
                                                          sc * gi, align="right")
    # D. softmax: scores become shares; the contents flow to 它 in those shares
    a4 = A * window(t, TL.C_SOFT, 63.6, 0.6, 0.6)
    if a4 > 0.003:
        zh("分数", 30, 500, 0.1).draw(F, 230, 440, DIM, a4, align="center")
        zh("softmax 之后", 30, 500, 0.1).draw(F, 640, 440, DIM, a4 * smooth(ramp(t, TL.C_SOFT + 0.8, 0.5)),
                                            align="center")
        for i, (lab, scv) in enumerate(P.SCORES):
            T(f"{scv}", "stix", 44, None, per_char=False).draw(F, 250, 520 + 92 * i + 16, GOLD if i == 0 else IVORY,
                                                              0.9 * a4, align="right")
        P.bars(F, P.SOFT_ITEMS, 470, 520, 340, a=a4, grow=ramp(t, TL.C_SOFT + 1.0, 1.8), row=92, size=40, hot=0)
        zh("加起来 = 100%", 30, 500, 0.1).draw(F, 880, 1010, DIM, 0.9 * a4 * smooth(ramp(t, TL.C_SOFT + 3, 0.6)),
                                             align="right")
        ma = smooth(ramp(t, TL.C_MIX, 0.6)) * a4
        if ma > 0:
            it_p = np.array([860, 1150])
            zh("它", 64, 500, 0).draw(F, it_p[0], it_p[1] + 22, GOLD, ma, align="center")
            pmax = P.SOFT_ITEMS[0][1]
            for i, (lab, p) in enumerate(P.SOFT_ITEMS):
                u = ease_in_out(ramp(t, TL.C_MIX + 0.6 + 0.25 * i, 1.1))
                if u <= 0 or u >= 1:
                    continue
                p0 = np.array([470 + 24 + 340 * p / pmax + 90, 520 + 92 * i])
                mid = (p0 + it_p) / 2 + [60, 0]
                pp = (1 - u) ** 2 * p0 + 2 * (1 - u) * u * mid + u ** 2 * it_p
                orb(F, pp[0], pp[1], 3 + 16 * p, mix(STEEL, GOLD, p / pmax), (0.4 + 1.0 * p / pmax) * ma)
            got = smooth(ramp(t, TL.C_MIX + 2.4, 0.6))
            add_sprite(F, it_p[0], it_p[1], 70 + 30 * got, GOLD, 0.25 * got * ma)
    # E. back in the sky: 它 moves to where its new numbers put it
    a5 = A * window(t, TL.C_MOVE, 71.6, 0.6, 0.6)
    if a5 > 0.003:
        dim = {w: 0.5 for w in P.SKY_WORDS}
        dim.update({"小猫": 1.0, "狗": 0.8, "老虎": 0.8, "兔子": 0.8, "累": 1.0})
        P.sky(F, a5, dim=dim)
        u = ease_in_out(ramp(t, TL.C_MOVE + 0.8, 2.6))
        p = P.IT_HOME + (MIX_FINAL - P.IT_HOME) * u
        for name, w in (("小猫", 0.571), ("累", 0.210)):
            q = np.array(P.SKY_WORDS[name], float) + [10, 10]
            glow_poly(F, [p, q], GOLD, (0.25 + 1.0 * w) * a5, th=1 if w < 0.4 else 3, glow=0.4 + w, sigma=3 + 3 * w)
        P.sky_word(F, "它", p, a5, glow=0.4 + 0.6 * u, size=44, mag=1.0 + 0.6 * u)
        la = smooth(ramp(t, TL.C_MOVE + 1.0, 0.6)) * a5
        zh("57% 小猫  +  21% 累  +  …", 32, 500, 0.06).draw(F, CX, 1250, DIM, la, align="center")
        shock(F, p[0], p[1], t - TL.C_KNOW, a5, size=0.8)
    # F. the one line, taken apart
    a6 = A * window(t, TL.C_FORMULA, 89.6, 0.8, 0.6)
    if a6 > 0.003:
        shock(F, CX, 680, t - TL.C_FORMULA, a6, size=0.9)
        P.formula_parts(F, CX, 620, a6, lit={"qk": smooth(ramp(t, TL.C_QK, 0.5)), "soft": smooth(ramp(t, TL.C_SM, 0.5)),
                                             "v": smooth(ramp(t, TL.C_TV, 0.5)), "sq": smooth(ramp(t, TL.C_SQ, 0.5))})
    # G. every word at once: the whole table
    a7 = A * smooth(ramp(t, TL.C_GRID, 0.6))
    if a7 > 0.003:
        P.grid(F, ATT, a=a7, on=_overshoot(ramp(t, TL.C_GRID + 0.8, 0.45)), row=IT,
               row_a=smooth(ramp(t, TL.C_GRID + 2.2, 0.7)))
        shock(F, CX + 46, 420 + 86 * 4.5, t - TL.C_GRID - 0.8, a7, size=1.0)
        zh("（数值为示意）", 24, 400, 0.1).draw(F, 960, 1262, DIM, 0.8 * a7, align="right")
    subtitles(F, t, [(0.6, 5.0, "这篇论文换了个思路：不排队，让每个字直接看向所有的字。",
                      "The paper took another road: no queue - every word looks straight at every other."),
                     (5.4, 9.0, "可它怎么知道，该看谁？", "But how does it know whom to look at?"),
                     (9.4, 15.0, "每个字，用自己的那串数字，分别乘上三张「表」，得到三样东西：",
                      "Each word multiplies its numbers by three tables, and gets three things:"),
                     (15.4, 18.4, "Q，问题：我在找什么？", "Q, the query: what am I looking for?"),
                     (18.8, 21.8, "K，标签：我是什么？", "K, the key: what am I?"),
                     (22.2, 25.6, "V，内容：要是你选中我，我给你什么。", "V, the value: what I hand over if you pick me."),
                     (26.0, 31.0, "比如「它」的问题，大概是：“前面哪个东西会累？”",
                      "For 'it', the question is roughly: which thing before me can get tired?"),
                     (31.4, 36.0, "每个字也亮出自己的标签。问题和标签，比一比有多对得上。",
                      "Every word shows its key. Compare: how well do question and key match?"),
                     (36.4, 41.0, "怎么比？两串数字对应相乘，再加起来。",
                      "How? Multiply the numbers pair by pair, then add them up."),
                     (41.4, 46.0, "两支箭头方向越一致，得分越高；互相垂直，就是 0 分。",
                      "The more two arrows agree, the higher the score; at right angles, zero."),
                     (46.4, 51.6, "小猫 3 分，累 2 分，跳上 1 分，桌子 0 分……", "Cat 3, tired 2, jump 1, table 0..."),
                     (52.0, 58.0, "再把分数变成百分比，加起来正好 100%。这一步叫 softmax：分数越高，分到的越多。",
                      "Then scores become shares that add up to 100% - softmax: the higher the score, the bigger the share."),
                     (58.4, 63.6, "最后按这个比例，把每个字的内容 V 混在一起，交给「它」。",
                      "Finally, every word's value V is mixed in those shares and handed to 'it'."),
                     (64.0, 67.6, "于是，「它」那串数字里，一大半来自小猫。",
                      "So more than half of what 'it' now holds comes from the cat."),
                     (68.0, 71.6, "「它」终于知道，自己是谁了。", "'It' finally knows who it is."),
                     (72.4, 76.6, "刚才这几步，写成一行，就是整篇论文的核心。",
                      "Those steps, written as one line, are the heart of the paper."),
                     (77.0, 83.4, "Q 乘 K 是“比一比”，softmax 是“变成百分比”，最后乘 V 是“按比例取内容”。",
                      "Q times K compares; softmax makes shares; times V mixes the contents."),
                     (83.8, 89.6, "分母上的根号呢？论文里每支箭头有 64 个数，除以根号 64，也就是 8，免得分数大得失控。",
                      "And the square root? Each arrow has 64 numbers; dividing by 8 keeps the scores in check."),
                     (90.0, 94.0, "而且，所有字是同时这样算的：一整张表，一次算完。",
                      "And every word does this at the same time: one whole table, all at once."),
                     (94.4, 97.6, "这正是显卡最擅长的事——所以它算得快，也能做得很大。",
                      "Exactly what graphics chips do best - so it is fast, and it can grow huge.")])



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
    A = window(t, 0.0, 21.6, 0.6, 0.8)
    cam(t, 22, extra=kick(t - TL.HEADS_ALL) + kick(t - TL.HEADS_MERGE - 1.0) * 0.6)
    P.columns(F, A * ease_out(ramp(t, 0.4, 0.6)), left_a=0.85, right_a=0.85)
    cur = int((t - TL.HEADS_T0) / TL.HEADS_DT) if t >= TL.HEADS_T0 else -1
    allin = smooth(ramp(t, TL.HEADS_ALL, 1.2))
    merge = smooth(ramp(t, TL.HEADS_MERGE, 1.6))
    on = np.zeros(8)
    for k, (name, desc, links) in enumerate(HEADS):
        start = TL.HEADS_T0 + k * TL.HEADS_DT
        solo = smooth(ramp(t, start, 0.3)) * (1 - smooth(ramp(t, start + TL.HEADS_DT, 0.3)))
        on[k] = max(solo, allin)
        a = A * max(solo, 0.55 * allin)
        if a > 0.003:
            P.links_lines(F, links, mix(P.HEAD_COLS[k], GOLD, merge), a, grow=ease_out(ramp(t, start, 0.45)))
    P.head_squares(F, on, A * smooth(ramp(t, 0.8, 0.8)), y=330, merge=merge)
    if 0 <= cur < 8 and t < TL.HEADS_ALL:
        name, desc, _ = HEADS[cur]
        la = A * smooth(ramp(t, TL.HEADS_T0 + cur * TL.HEADS_DT, 0.3))
        zh(f"第 {cur + 1} 组 · {name}", 34, 500, 0.1).draw(F, CX, 296, P.HEAD_COLS[cur], la, align="center")
    if allin > 0:
        zh("多头注意力", 36, 500, 0.3).draw(F, CX, 296, GOLD, A * allin, align="center")
    subtitles(F, t, [(0.6, 5.4, "一组 Q、K、V，只能看出一种关系。那就同时用八组。",
                      "One set of Q, K and V sees one kind of relation. So use eight at once."),
                     (5.8, 11.2, "有的找「它」指谁，有的找谁做了什么，有的只看身边的字。",
                      "One finds who 'it' is, one who did what, one only watches its neighbours."),
                     (11.6, 17.0, "这叫「多头注意力」。", "This is multi-head attention."),
                     (17.4, 21.4, "最后，把八种看法拼在一起。", "Then the eight views are put together.")])


# ----------------------------------------------------------------- order ---

def order(F, t, g, fi):
    A = window(t, 0.0, 19.6, 0.6, 0.8)
    cam(t, 20, extra=kick(t - TL.ORD_SWAP - 1.2) * 0.6)
    sw = ease_in_out(ramp(t, TL.ORD_SWAP, 1.2))
    y = 760
    xs = [CX - 170, CX, CX + 170]
    run = 1 - smooth(ramp(t, TL.ORD_WAVES - 0.5, 0.8))
    for k, ch in enumerate(["猫", "追", "狗"]):
        x, yy = xs[k], y
        if k == 0:
            x = lerp(xs[0], xs[2], sw)
            yy = y - 175 * math.sin(math.pi * sw)
        elif k == 2:
            x = lerp(xs[2], xs[0], sw)
            yy = y + 175 * math.sin(math.pi * sw)
        if k != 1:
            yy -= 9 * run * abs(math.sin(math.pi * (t - 0.6) / 0.333 + (0 if k == 0 else 1.2)))
        al = A * smooth(ramp(t, 0.6 + 0.2 * k, 0.6)) * ((1 - 0.6 * math.sin(math.pi * sw)) if k == 1 else 1.0)
        zh(ch, 150, 500).draw(F, x, yy, GOLD if k != 1 else IVORY, al, align="center")
    wa = smooth(ramp(t, TL.ORD_WAVES, 1.2)) * A
    if wa > 0:
        for i, f in enumerate([1.0, 2.1, 4.3, 8.5]):
            yc = 930 + i * 72
            xx = np.linspace(150, 930, 320)
            yw = yc - 22 * np.sin((xx - xs[0]) / 170.0 * f * 0.9)
            glow_poly(F, np.column_stack([xx, yw]), mix(STEEL, GOLD, i / 3), 0.6 * wa, th=1, glow=0.5, sigma=3)
            for k in range(3):
                orb(F, xs[k], yc - 22 * math.sin(k * f * 0.9), 5, GOLD,
                    0.95 * wa * smooth(ramp(t, TL.ORD_WAVES + 1.0 + 0.4 * k, 0.5)))
        for k in range(3):
            for yy in np.arange(900, 1170, 9):
                add_sprite(F, xs[k], yy, 1.0, IVORY, 0.25 * wa)
            T(str(k + 1), "stix", 28, None, per_char=False).draw(F, xs[k], 1215, DIM, wa, align="center")
        zh("位置", 26, 500, 0.2).draw(F, CX, 1260, DIM, 0.8 * wa, align="center")
    subtitles(F, t, [(0.6, 4.4, "同时看所有的字，会把顺序丢掉：", "Looking at every word at once loses the order:"),
                     (4.6, 9.6, "「猫追狗」和「狗追猫」，是同样三个字。",
                      "'cat chases dog' and 'dog chases cat' are the same three characters."),
                     (10.0, 15.6, "所以每个位置，都被加上一组快慢不同的波，像指纹一样。",
                      "So every position gets a set of waves of different speeds added, like a fingerprint."),
                     (16.0, 19.4, "顺序，就藏进了数字里。", "The order hides inside the numbers.")])


# ----------------------------------------------------------------- learn ---

def learn(F, t, g, fi):
    A = window(t, 0.0, 45.6, 0.6, 0.5)
    cam(t, 46, extra=kick(t - TL.L_COUNT - 2.9) + kick(t - TL.L_96 - 0.6) * 0.5, out_at=45.1, out_len=0.9)
    # one layer; six; ninety-six
    la = A * (1 - smooth(ramp(t, TL.L_MOVE - 0.4, 0.6)))
    if la > 0.003:
        stack = ramp(t, TL.L_STACK, 2.0)
        n6 = 1 + int(5 * stack + 1e-6) if t >= TL.L_STACK else 1
        squash = ease_in_out(ramp(t, TL.L_96, 1.2))
        for k in range(n6):
            yb = 1060 - 40 * k * (1 - squash)
            top = k == n6 - 1
            cover = np.full((170, 520), 1.0, np.float32)
            blend(F, cover, CX - 260, yb - 170, np.array([0.02, 0.028, 0.045], np.float32), 0.92 * la * (1 - squash))
            P.layer(F, CX, yb, 520, la * (0.6 + 0.4 * top) * (1 - squash), detail=(1.0 if top else 0.0),
                    glow=0.4 if top else 0.0)
        if squash > 0:
            n96 = int(6 + 90 * ease_in(ramp(t, TL.L_96 + 0.6, 1.4)))
            for k in range(n96):
                y = 1080 - 7 * k
                hairline(F, CX - 260, int(y), CX + 260, mix(STEEL, GOLD, k / 96), 0.55 * la * squash)
        lab_a = la * smooth(ramp(t, TL.L_STACK + 1.0, 0.5))
        txt = "论文：6 层" if t < TL.L_96 + 0.8 else "GPT-3：96 层"
        zh(txt, 44, 600, 0.1).draw(F, CX, 400, GOLD, lab_a, align="center")
    # in the sky, each layer moves 它 a little further
    ma = A * window(t, TL.L_MOVE, TL.L_WHO - 0.2, 0.6, 0.6)
    if ma > 0.003:
        dim = {w: 0.55 for w in P.SKY_WORDS}
        dim.update({"小猫": 1.0, "狗": 0.85, "老虎": 0.85, "兔子": 0.85})
        P.sky(F, ma, dim=dim)
        steps = 6
        u = ramp(t, TL.L_MOVE + 0.8, 4.0) * steps
        k = int(u)
        fr = ease_in_out(u - k)
        path = [P.IT_HOME + (MIX_FINAL - P.IT_HOME) * min(1.0, i / 2) if i <= 2 else
                MIX_FINAL + (np.array(P.SKY_WORDS["小猫"], float) + [55, 45] - MIX_FINAL) * ((i - 2) / (steps - 2))
                for i in range(steps + 1)]
        k = min(k, steps - 1)
        p = path[k] + (path[k + 1] - path[k]) * (fr if u < steps else 1.0)
        for i in range(min(k + 1, steps)):
            add_sprite(F, path[i][0], path[i][1], 2.0, GOLD, 0.6 * ma)
        P.sky_word(F, "它", p, ma, glow=0.8, size=44, mag=1.5)
        zh(f"第 {min(k + 1, steps)} 层", 34, 500, 0.1).draw(F, CX, 1250, DIM, ma, align="center")
    # who fills in the numbers?
    wa = A * window(t, TL.L_WHO, TL.L_MASK - 0.2, 0.6, 0.6)
    if wa > 0.003:
        for k, (letter, col, _, _, _) in enumerate(QKV_ROWS):
            P.matrix(F, 170 + 270 * k, 640, 8, 8, 24, "W", a=wa, seed=k, col=mix(col, IVORY, 0.3), sub=letter)
            zh("？", 90, 600, 0).draw(F, 170 + 270 * k + 96, 800, GOLD, wa * smooth(ramp(t, TL.L_WHO + 0.8, 0.5)),
                                     align="center")
    sa = A * window(t, TL.L_MASK, TL.L_NUDGE + 3.0, 0.6, 0.8)
    if sa > 0.003:
        P.text_stream(F, sa * (1 - 0.6 * smooth(ramp(t, TL.L_NUDGE, 0.8))), t=t - TL.L_MASK, y0=380, rows=13)
    da = A * window(t, TL.L_NUDGE, TL.L_PAPER - 0.2, 0.8, 0.6)
    if da > 0.003:
        P.dials(F, 180, 470, 12, 9, 65, a=da * (1 - 0.5 * smooth(ramp(t, TL.L_COUNT, 0.6))), t=t,
                turn=ramp(t, TL.L_NUDGE, 6.0) * 2 + 0.6 * math.sin(t * 1.3))
        ca = da * smooth(ramp(t, TL.L_COUNT, 0.6))
        if ca > 0:
            v = 175_000_000_000 * ease_out(ramp(t, TL.L_COUNT + 0.3, 2.6))
            P.counter(F, v, CX, 1150, ca, size=84)
            shock(F, CX, 1120, t - TL.L_COUNT - 2.9, ca, size=0.7)
            zh("个数字 · GPT-3", 32, 500, 0.1).draw(F, CX, 1220, DIM, ca, align="center")
    pa = A * smooth(ramp(t, TL.L_PAPER, 0.8))
    if pa > 0.003:
        zh("8 块显卡 · 3.5 天", 56, 600, 0.06).draw(F, CX, 560, GOLD, pa, align="center")
        P.bleu(F, pa, grow=ramp(t, TL.L_PAPER + 1.0, 1.6), y0=780)
    subtitles(F, t, [(0.6, 4.6, "一次注意力，加一步简单的计算，叫做一「层」。",
                      "One round of attention plus one simple step makes a layer."),
                     (5.0, 9.4, "论文叠了 6 层；GPT-3 叠了 96 层。", "The paper stacked 6 layers; GPT-3 stacks 96."),
                     (9.8, 15.0, "每过一层，每颗星都再挪一挪，意思就更准一点。",
                      "With every layer, every star moves again, and the meaning sharpens."),
                     (15.4, 19.4, "可那三张「表」里的数字，是谁填的？", "But who fills in the numbers in those three tables?"),
                     (19.8, 24.6, "没有人填。给它海量的文字，遮住下一个字，让它猜。",
                      "Nobody. Give it oceans of text, hide the next word, and let it guess."),
                     (25.0, 30.4, "猜错了，就把所有数字往“猜对”的方向，轻轻拧一点。",
                      "When it guesses wrong, turn every number a little towards the right answer."),
                     (30.8, 36.0, "重复几千亿次。GPT-3 一共有 1750 亿个这样的数字。",
                      "Repeat hundreds of billions of times. GPT-3 has 175 billion such numbers."),
                     (36.4, 45.4, "当年，论文里的模型只用 8 块显卡训练了 3.5 天，就在英译德上超过了此前所有的方法。",
                      "Back then, the paper's model trained on 8 GPUs for 3.5 days - and beat every earlier method at English-to-German.")])


# ------------------------------------------------------------------ mind ---
# Why would guessing the next word look like intelligence?

NEURONS = None
SKILLS = ["翻译", "写诗", "解题", "写代码", "讲笑话", "做总结"]


def _neurons():
    global NEURONS
    if NEURONS is None:
        rng = np.random.default_rng(86)
        pts = np.column_stack([rng.uniform(170, 910, 16), rng.uniform(470, 960, 16)])
        NEURONS = [P.neuron(x, y, 150, seed=int(k)) for k, (x, y) in enumerate(pts)]
    return NEURONS


def mind(F, t, g, fi):
    A = window(t, 0.0, 97.6, 0.5, 0.8)
    cam(t, 98, extra=kick(t - TL.M_ASK - 1.2) + kick(t - TL.M_GEN) + kick(t - TL.M_EMERGE) +
        kick(t - TL.M_NEWTON) * 0.6 + kick(t - TL.M_DEBATE) * 0.6, out_at=97.2, out_len=0.8)
    punch(F, "只是“猜下一个字”，", CX, 720, t, TL.M_ASK, TL.M_BOOK - 0.3, size=64, wght=600)
    punch(F, "为什么会变聪明？", CX, 840, t, TL.M_ASK + 1.2, TL.M_BOOK - 0.3, size=72, wght=700, gold=(4, 5, 6))
    # the detective novel
    ba = A * window(t, TL.M_BOOK, TL.M_CANT - 0.2, 0.8, 0.6)
    if ba > 0.003:
        P.book(F, ba, clues=ramp(t, TL.M_CLUES, 2.0), threads=ramp(t, TL.M_CLUES + 1.2, 3.0), t=t)
    # it cannot memorise, so it must find the rule
    ca = A * window(t, TL.M_CANT, TL.M_GEN - 0.2, 0.6, 0.6)
    if ca > 0.003:
        P.text_stream(F, ca * 0.7, t=t, mask=False, y0=380, rows=13, speed=90.0)
        box = (CX - 150, 640, CX + 150, 860)
        cover = np.full((box[3] - box[1], box[2] - box[0]), 1.0, np.float32)
        blend(F, cover, box[0], box[1], np.array([0.02, 0.028, 0.045], np.float32), 0.95 * ca)
        P._rrect(F, *box, GOLD, 0.8 * ca, r=10, glow=0.5)
        zh("它的数字", 28, 500, 0.1).draw(F, CX, box[1] - 18, GOLD, ca, align="center")
        rule = ease_out(ramp(t, TL.M_CANT + 2.6, 1.6))
        for k in range(3):
            y = 700 + 55 * k
            w = (180, 140, 200)[k] * smooth(ramp(rule, 0.25 * k, 0.5))
            if w > 1:
                glow_poly(F, [(CX - w / 2, y), (CX + w / 2, y)], GOLD, 0.9 * ca, th=2, glow=0.7, sigma=4)
        zh("规律", 34, 600, 0.2).draw(F, CX, 840, GOLD, ca * smooth(ramp(t, TL.M_CANT + 4.0, 0.6)), align="center")
    punch(F, "找规律，就是概括。", CX, 800, t, TL.M_GEN, TL.M_HUMAN - 0.1, size=80, wght=700, gold=(6, 7))
    shock(F, CX, 770, t - TL.M_GEN, 0.7 * A, size=0.8)
    statement(F, "人类也一直是这么做的。", CX, 780, t, TL.M_HUMAN + 0.2, TL.M_KEPLER - 0.2, size=58, cps=10)
    # Tycho's marks, Kepler's ellipse, Newton's line
    ka = A * window(t, TL.M_KEPLER, TL.M_CAT - 0.2, 0.6, 0.6)
    if ka > 0.003:
        kd = 1 - 0.55 * smooth(ramp(t, TL.M_NEWTON, 0.6))
        P.kepler(F, CX, 720, ka * kd, dots=ramp(t, TL.M_KEPLER + 0.2, 2.6), fit=ramp(t, TL.M_KEPLER + 3.4, 2.2), t=t)
        zh("第谷 · 二十多年的观测", 30, 500, 0.08).draw(F, CX, 360, DIM, ka * kd * smooth(ramp(t, TL.M_KEPLER, 0.6)),
                                                align="center")
        zh("开普勒 · 椭圆", 30, 500, 0.08).draw(F, CX, 410, GOLD,
                                           ka * kd * smooth(ramp(t, TL.M_KEPLER + 4.8, 0.6)), align="center")
        zh("（示意）", 22, 400, 0.1).draw(F, 950, 1010, DIM, 0.7 * ka * kd, align="right")
        na = ka * smooth(ramp(t, TL.M_NEWTON, 0.8))
        P.newton(F, CX, 1160, na)
        zh("天上 · 地上 · 同一个公式", 28, 500, 0.1).draw(F, CX, 1260, GOLD, na * smooth(ramp(t, TL.M_NEWTON + 1, 0.6)),
                                                  align="center")
    # many cats, one word
    ma = A * window(t, TL.M_CAT, TL.M_SKY - 0.1, 0.6, 0.6)
    if ma > 0.003:
        P.many_cats(F, ma, gather=ramp(t, TL.M_CAT + 1.8, 3.4))
    # the sky that nobody drew
    sa = A * window(t, TL.M_SKY, TL.M_EMERGE - 0.2, 0.8, 0.6)
    if sa > 0.003:
        P.sky(F, sa, scatter=1 - ease_in_out(ramp(t, TL.M_SKY + 0.6, 3.6)))
    punch(F, "这就是「涌现」。", CX, 800, t, TL.M_EMERGE, TL.M_BIRDS - 0.1, size=86, wght=700, gold=(4, 5))
    shock(F, CX, 770, t - TL.M_EMERGE, A, size=1.1)
    # starlings
    fa = A * window(t, TL.M_BIRDS, TL.M_NEURON - 0.2, 1.0, 0.8)
    if fa > 0.003:
        P.murmuration(F, (t - TL.M_BIRDS) * 1.1 + 2.0, fa)
    # neurons
    na = A * window(t, TL.M_NEURON, TL.M_SCALE - 0.2, 0.6, 0.6)
    if na > 0.003:
        one = 1 - smooth(ramp(t, TL.M_NEURON + 2.0, 0.8))
        if one > 0:
            P.draw_neuron(F, P.neuron(CX, 520, 640, seed=2), na * one, grow=ramp(t, TL.M_NEURON, 1.6),
                          fire=ramp(t, TL.M_NEURON + 1.2, 0.8))
        many = smooth(ramp(t, TL.M_NEURON + 2.0, 0.8))
        if many > 0:
            for k, nr in enumerate(_neurons()):
                ph = (t * 0.7 + k * 0.37) % 1.6
                P.draw_neuron(F, nr, na * many * 0.5, grow=1.0, fire=ph if ph <= 1 else None)
            P.counter(F, 86_000_000_000 * ease_out(ramp(t, TL.M_NEURON + 2.4, 1.6)), CX, 1200, na * many, size=60)
    # sums and products, piled up - and abilities nobody taught
    xa = A * window(t, TL.M_SCALE, TL.M_DEBATE - 0.2, 0.6, 0.6)
    if xa > 0.003:
        rng = np.random.default_rng(5)
        for k in range(160):
            x, y = rng.uniform(100, 980), rng.uniform(360, 1220)
            ch = "×" if k % 2 else "+"
            tw = 0.5 + 0.5 * math.sin(t * 2 + k)
            zh(ch, 26, 400, 0).draw(F, x, y, STEEL, 0.25 * xa * tw, align="center")
        for k, s_ in enumerate(SKILLS):
            gk = smooth(ramp(t, TL.M_SCALE + 2.2 + 0.5 * k, 0.5))
            x = CX + (-230 if k % 2 == 0 else 230)
            y = 560 + 140 * (k // 2)
            if gk > 0:
                add_sprite(F, x, y - 20, 70, GOLD, 0.15 * gk * xa)
                zh(s_, 52, 600, 0.1).draw(F, x, y, GOLD, gk * xa, align="center")
    punch(F, "这算不算真正的智能？", CX, 790, t, TL.M_DEBATE, TL.M_GUESS - 0.1, size=68, wght=700)
    # perhaps intelligence is generalising
    ga = A * window(t, TL.M_GUESS, 97.6, 0.8, 0.8)
    if ga > 0.003:
        P.sky(F, ga * 0.35)
    statement(F, "也许智能，本来就是一种概括：", CX, 720, t, TL.M_GUESS, TL.M_QUESTION - 0.2, size=54, cps=10,
              gold={9, 10})
    statement(F, "从看过的一切里找出规律，", CX, 810, t, TL.M_GUESS + 1.4, TL.M_QUESTION - 0.2, size=54, cps=10)
    statement(F, "去猜还没看到的。", CX, 900, t, TL.M_GUESS + 2.8, TL.M_QUESTION - 0.2, size=54, cps=10)
    statement(F, "如果是这样，它离“懂”还有多远？", CX, 740, t, TL.M_QUESTION, 97.4, size=54, cps=10)
    statement(F, "如果不止于此——", CX, 850, t, TL.M_QUESTION + 2.4, 97.4, size=54, cps=10)
    statement(F, "缺的，又是什么？", CX, 940, t, TL.M_QUESTION + 3.6, 97.4, size=54, cps=8, gold={0, 1})
    subtitles(F, t, [(5.4, 10.0, "想象一本侦探小说，最后一页写着：“谜底是——”",
                      "Picture a detective novel whose last page reads: 'And the answer is -'"),
                     (10.4, 16.0, "要猜出谜底，只认得字是不够的：得记住每一条线索，弄懂每一个人的动机。",
                      "To guess the answer, knowing the words is not enough: you must hold every clue and grasp every motive."),
                     (16.4, 20.0, "为了猜得准，它被逼着去理解。", "To guess well, it is forced to understand."),
                     (20.4, 26.6, "而它的数字，装不下它读过的全部文字——背不下来，就只能找规律。",
                      "And its numbers cannot hold all the text it has read - unable to memorise, it must find the rules."),
                     (34.0, 40.6, "第谷记了二十多年的星星位置，开普勒把它们概括成了三条定律；",
                      "Tycho logged the planets for over twenty years; Kepler distilled them into three laws;"),
                     (41.0, 45.6, "牛顿又把天上和地上，概括成了一个公式。",
                      "Newton distilled the heavens and the earth into one formula."),
                     (46.0, 52.0, "就连一个字，也是概括：世上没有两只一样的猫，我们却只用一个「猫」字。",
                      "Even a word is a generalisation: no two cats are alike, yet we use one word, 'cat'."),
                     (52.4, 58.6, "那片意义星空，也没有人画过——是它为了猜对下一个字，自己长出来的。",
                      "Nobody drew that sky of meaning - it grew by itself, in order to guess the next word."),
                     (62.4, 69.0, "一只椋鸟，只管跟着身边的七只鸟飞；成千上万只聚在一起，天上就出现了谁也没设计过的形状。",
                      "A starling only follows its seven nearest neighbours; thousands together draw shapes nobody designed."),
                     (69.4, 74.6, "一个神经元，只会放电或不放电；八百六十亿个连在一起，就有了你。",
                      "A neuron only fires or stays quiet; eighty-six billion of them together make you."),
                     (75.0, 81.0, "模型里的每一步，都只是乘法和加法；可堆到上千亿个数字，它开始翻译、写诗、解题——没人专门教过它这些。",
                      "Every step in the model is just multiplying and adding; stacked to hundreds of billions, it translates, writes poems, solves problems - nobody taught it those."),
                     (81.4, 85.0, "", "Is this real intelligence? Scientists are still arguing."),
                     (85.4, 91.4, "", "Perhaps intelligence is generalising: finding the rules in all you have seen, to guess what you haven't."),
                     (91.8, 97.4, "", "If so, how far is it from understanding? And if not - what is missing?")])
    if TL.M_DEBATE + 0.3 <= t < TL.M_GUESS - 0.2:
        zh("科学家们还在争论。", 34, 500, 0.1).draw(F, CX, 880, DIM, A * window(t, TL.M_DEBATE + 1.4, TL.M_GUESS - 0.3,
                                                                             0.6, 0.5), align="center")


# ----------------------------------------------------------------- after ---

def _title_echo(F, y, a, gold_from, size=40):
    full = "Attention Is All You Need"
    tx = T(full, "corm_it", size, 500, per_char=False)
    x0 = CX - tx.width / 2
    tx.draw(F, x0, y, IVORY, 0.7 * a)
    if gold_from > 0:
        pre = T("Attention Is ", "corm_it", size, 500, per_char=False).width
        T("All You Need", "corm_it", size, 500, per_char=False).draw(F, x0 + pre, y, GOLD, a * gold_from)


def _song(F, y, a, gold_from, size=58):
    tx = T("All You Need Is Love", "corm_it", size, 500, per_char=False)
    x0 = CX - tx.width / 2
    tx.draw(F, x0, y, IVORY, a)
    if gold_from > 0:
        T("All You Need", "corm_it", size, 500, per_char=False).draw(F, x0, y, GOLD, a * gold_from)


def after(F, t, g, fi):
    A = window(t, 0.0, 29.6, 0.6, 0.8)
    cam(t, 30, extra=kick(t - TL.A_GPT))
    pa = A * smooth(ramp(t, TL.A_PAGE, 0.9)) * (1 - smooth(ramp(t, TL.A_SONG - 0.6, 0.8)))
    if pa > 0.003:
        P.page(F, pa, note=ease_out(ramp(t, TL.A_NOTE, 0.9)))
    ba = A * window(t, TL.A_SONG, TL.A_GPT + 0.2, 0.8, 0.8)
    if ba > 0.003:
        P.record(F, CX, 720, 250, t, ba)
        echo = smooth(ramp(t, TL.A_SONG + 1.4, 0.8))
        _title_echo(F, 400, ba, echo)
        _song(F, 1070, ba, echo)
        caps("THE BEATLES   ·   1967", 20, 0.45).draw(F, CX, 1125, DIM, ba, align="center")
    ga = A * window(t, TL.A_GPT, TL.A_CHAT, 0.8, 0.6)
    if ga > 0.003:
        for k, ch in enumerate("GPT"):
            T(ch, "corm", 180, 600, per_char=False).draw(F, CX - 130 + 130 * k, 800, GOLD if ch == "T" else IVORY, ga,
                                                        align="center")
        caps("GENERATIVE PRE-TRAINED TRANSFORMER", 22, 0.4).draw(F, CX, 880, DIM, ga, align="center")
        shock(F, CX, 740, t - TL.A_GPT, ga, size=0.9)
    ca = A * smooth(ramp(t, TL.A_CHAT, 0.6))
    if ca > 0.003:
        u = ramp(t, TL.A_CHAT + 0.6, 10.5)
        P.chat(F, P.CHAT_Q, P.CHAT_A, len(P.CHAT_A) * (0.5 * u + 0.5 * u ** 1.8), t, a=ca, y0=420, cand_y=1090)
    subtitles(F, t, [(0.6, 6.4, "2017 年 6 月，论文发表，作者八位，署名旁边写着：贡献相同，排名随机。",
                      "June 2017: the paper comes out with eight authors, and a note: equal contribution, order random."),
                     (7.0, 12.0, "标题，是在致敬披头士的《All You Need Is Love》。",
                      "Its title tips its hat to the Beatles' 'All You Need Is Love'."),
                     (12.6, 17.0, "后来，GPT 里的「T」，就是 Transformer。", "Later, the 'T' in GPT stood for Transformer."),
                     (17.4, 23.2, "你和 AI 的每一次对话：它读完你写的每个字，让所有字互相“注意”几十遍，",
                      "Every time you talk to an AI, it reads all you wrote, lets every word attend to every other dozens of times,"),
                     (23.6, 29.4, "猜出下一个字——再接上，从头再来。", "guesses the next word - then adds it, and starts again.")])


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
    ai_label(F, smooth(ramp(t, TL.END_TITLE + 1.0, 0.8)))
    subtitles(F, t, [(0.8, 5.6, "机器学会了注意力，于是开始读懂语言。", "Machines learned attention, and began to read language."),
                     (6.0, 11.8, "而在一个什么都在争夺你注意力的时代——",
                      "And in an age when everything is competing for your attention -"),
                     (TL.END_LINE + 0.4, 18.6, "", "perhaps attention really is all you need.")])


SCENE_FUNCS = {"cold": cold, "hook": hook, "sky": sky_scene, "sequential": sequential, "core": core, "heads": heads,
               "order": order, "learn": learn, "mind": mind, "after": after, "ending": ending}
