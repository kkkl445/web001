"""Original score for 注意力 · Attention Is All You Need, synthesised from scratch.

Pads, felt piano, glass bells and plucked strings, with a soft pulse (kick,
hat, bass) under the busy stretches and a hit on every reveal.  Sound follows
the ideas: a tick for every number and every word read, a whisper that gets
quieter with every retelling, a run of notes as the arithmetic is written
out, a drop when 「它」 lands beside the cat, everything ringing at once when
the table appears, eight voices for eight heads, the position waves as four
pure tones, and a flutter of tiny ticks under the starlings.

  python3 music.py   -> build/bgm.wav
"""

import numpy as np

import synth as A
import timeline as TL
from style import ROOT
from timeline import DURATION, START

SR = A.SR
N = int(DURATION * SR)
rng = np.random.default_rng(2017)

CHORDS = {
    "Fmaj9": (41, [53, 57, 60, 64, 67]), "Dm9": (38, [50, 57, 60, 64, 65]), "Bbmaj9": (34, [46, 53, 57, 60, 62]),
    "Bbmaj7#11": (34, [46, 50, 57, 64, 65]), "Gm9": (43, [50, 55, 58, 65, 69]), "Am7": (45, [52, 57, 60, 64, 67]),
    "C69": (36, [48, 55, 62, 64, 69]), "A7sus4": (45, [52, 57, 62, 67, 71]), "Dsus2": (38, [50, 57, 62, 64, 69]),
}
CD, HK, SK, SQ, CO, HD, OR, LN, MD, AF, EN = (START[k] for k in ("cold", "hook", "sky", "sequential", "core", "heads",
                                                                  "order", "learn", "mind", "after", "ending"))
PLAN = [(0.0, "Dm9"), (CD + TL.CO_L2, "Bbmaj9"), (CD + TL.CO_L3, "A7sus4"),
        (HK, "Bbmaj9"), (HK + 5.0, "C69"),
        (HK + TL.H_PLATE, "Fmaj9"), (HK + TL.H_SWAP, "Dm9"), (HK + TL.H_TURN, "Gm9"), (HK + TL.H_TITLE, "Fmaj9"),
        (HK + TL.H_TITLE + 3.2, "C69"),
        (SK, "Dm9"), (SK + TL.S_STAR, "Bbmaj9"), (SK + 15.0, "Fmaj9"), (SK + TL.S_KING, "Am7"), (SK + 26.0, "Gm9"),
        (SK + TL.S_APPLE, "Bbmaj7#11"), (SK + TL.S_IT, "A7sus4"), (SK + TL.S_PULL, "Dm9"),
        (SQ, "Dm9"), (SQ + 5.8, "Bbmaj7#11"), (SQ + 11.0, "Gm9"), (SQ + 16.4, "A7sus4"), (SQ + 22.0, "Dm9"),
        (CO, "Fmaj9"), (CO + TL.C_ASK, "Am7"), (CO + TL.C_MAT, "Dm9"), (CO + TL.C_Q, "Bbmaj9"), (CO + TL.C_K, "Gm9"),
        (CO + TL.C_V, "C69"), (CO + 26.0, "Fmaj9"), (CO + TL.C_KARROWS, "Dm9"), (CO + TL.C_SUM, "Bbmaj9"),
        (CO + TL.C_ALIGN, "Gm9"), (CO + TL.C_SCORES, "Am7"), (CO + TL.C_SOFT, "Bbmaj7#11"), (CO + TL.C_MIX, "C69"),
        (CO + TL.C_MOVE, "Dm9"), (CO + TL.C_KNOW, "Fmaj9"), (CO + TL.C_FORMULA, "Bbmaj9"), (CO + TL.C_QK, "Gm9"),
        (CO + TL.C_SQ, "A7sus4"), (CO + TL.C_GRID, "Fmaj9"), (CO + TL.C_GPU, "C69"),
        (HD, "Am7"), (HD + 5.8, "Dm9"), (HD + TL.HEADS_ALL, "Bbmaj9"), (HD + TL.HEADS_MERGE, "Fmaj9"),
        (OR, "Dm9"), (OR + TL.ORD_SWAP, "Gm9"), (OR + TL.ORD_WAVES, "C69"), (OR + 16.0, "Fmaj9"),
        (LN, "Bbmaj9"), (LN + TL.L_STACK, "Gm9"), (LN + TL.L_MOVE, "Dm9"), (LN + TL.L_WHO, "A7sus4"),
        (LN + TL.L_MASK, "Dm9"), (LN + TL.L_NUDGE, "Bbmaj7#11"), (LN + TL.L_COUNT, "Gm9"), (LN + TL.L_PAPER, "Fmaj9"),
        (LN + 41.0, "C69"),
        (MD, "Dm9"), (MD + TL.M_BOOK, "Gm9"), (MD + TL.M_CLUES, "Bbmaj7#11"), (MD + TL.M_FORCED, "A7sus4"),
        (MD + TL.M_CANT, "Dm9"), (MD + TL.M_GEN, "Fmaj9"), (MD + TL.M_HUMAN, "C69"), (MD + TL.M_KEPLER, "Am7"),
        (MD + TL.M_NEWTON, "Fmaj9"), (MD + TL.M_CAT, "Bbmaj9"), (MD + TL.M_SKY, "Gm9"), (MD + TL.M_EMERGE, "Fmaj9"),
        (MD + TL.M_BIRDS, "Dm9"), (MD + TL.M_NEURON, "Bbmaj7#11"), (MD + TL.M_SCALE, "C69"), (MD + TL.M_DEBATE, "Gm9"),
        (MD + TL.M_GUESS, "Bbmaj9"), (MD + TL.M_QUESTION, "A7sus4"),
        (AF, "Bbmaj9"), (AF + TL.A_NOTE, "Gm9"), (AF + TL.A_SONG, "Fmaj9"), (AF + TL.A_GPT, "Dm9"),
        (AF + TL.A_CHAT, "Bbmaj9"), (AF + 23.6, "C69"),
        (EN, "Gm9"), (EN + 6.0, "Bbmaj7#11"), (EN + TL.END_LINE, "Fmaj9"), (EN + TL.END_TITLE, "Dsus2")]


def chord_at(t):
    name = PLAN[0][1]
    for ts, nm in PLAN:
        if t >= ts:
            name = nm
    return name


def tones(name, lo, hi):
    pcs = {p % 12 for p in CHORDS[name][1]}
    return [m for m in range(lo, hi + 1) if m % 12 in pcs]


def whisper(dur, vel):
    """A breathy hush: soft band noise with a vowel-ish formant."""
    n = int(dur * SR)
    t = np.arange(n) / SR
    z = A.bp(rng.standard_normal(n), 1800, 5500) + 0.5 * A.bp(rng.standard_normal(n), 700, 1200)
    env = np.sin(np.pi * np.clip(t / dur, 0, 1)) ** 1.5
    v = z * env
    return v / (np.abs(v).max() + 1e-9) * vel


def hat(vel, open_=False):
    """A soft hi-hat: a short burst of bright noise."""
    n = int((0.25 if open_ else 0.07) * SR)
    t = np.arange(n) / SR
    v = A.hp(rng.standard_normal(n), 7000) * np.exp(-t / (0.08 if open_ else 0.018))
    return v / (np.abs(v).max() + 1e-9) * vel


def sine_tone(f, dur, vel, attack=0.6):
    t = np.arange(int(dur * SR)) / SR
    env = np.clip(t / attack, 0, 1) * np.clip((dur - t) / 1.2, 0, 1)
    return np.sin(2 * np.pi * f * t) * env * vel


def score():
    pad, sub, pno, bel, fx, tick, plk, shim, drm = (A.stereo(N) for _ in range(9))
    for i, (t0, name) in enumerate(PLAN):
        t1 = PLAN[i + 1][0] if i + 1 < len(PLAN) else DURATION
        hold = max(1.0, t1 - t0 + 0.3)
        bass, voicing = CHORDS[name]
        for m in voicing:
            A.place(pad, A.pad_note(m, hold), t0 + rng.uniform(0, 0.15))
        A.place(sub, A.sub_note(bass, hold), t0)
        for m in voicing[2:5]:
            n = int((hold + 2) * SR)
            tt = np.arange(n) / SR
            sw = np.sin(np.pi * np.clip(tt / (hold + 2), 0, 1)) ** 2
            lfo = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.07, 0.2) * tt + rng.uniform(0, 6.28))
            A.place(shim, np.sin(2 * np.pi * A.mtof(m + 24) * tt) * sw * lfo * 0.08, t0, pan=rng.uniform(-0.8, 0.8))

    def note(m, t, vel, pan=None):
        if 0 <= t < DURATION - 0.5:
            A.place(pno, A.piano(m, round(vel, 2)), t + rng.normal(0, 0.006),
                    pan=np.clip((m - 68) / 20, -0.6, 0.6) if pan is None else pan)

    def phrase(t0, t1, every, vel, lo=64, hi=81, prob=0.8):
        prev = 69
        for tt in np.arange(t0, t1, every):
            if rng.random() > prob:
                continue
            cand = tones(chord_at(tt), lo, hi)
            target = prev + rng.choice([-5, -3, -2, 2, 3, 4])
            m = min(cand, key=lambda c: abs(c - target) + (3 if c == prev else 0))
            prev = m
            note(m, tt, vel * rng.uniform(0.85, 1.05))

    def groove(t0, t1, vel=1.0, bpm=100, hats=True, build=False, bass=True):
        """A pulse under the busy stretches: kick on the beat, hat off the beat, a bass note each bar."""
        beat = 60.0 / bpm
        n = int((t1 - t0) / beat)
        for k in range(n):
            tt = t0 + k * beat
            u = k / max(1, n - 1)
            v = vel * ((0.45 + 0.55 * u) if build else 1.0)
            A.place(drm, A.soft_kick(0.55 * v), tt)
            if hats:
                A.place(drm, hat(0.10 * v), tt + beat / 2, pan=0.25)
                if k % 4 == 3:
                    A.place(drm, hat(0.07 * v, open_=True), tt + beat * 0.75, pan=-0.25)
            if bass and k % 4 == 0:
                A.place(sub, A.sub_note(CHORDS[chord_at(tt + 0.01)][0] + 12, beat * 3.6) * 0.6 * v, tt)
        if build:
            for k in range(8):
                A.place(drm, hat(0.08 * vel * (k + 1) / 8), t1 - beat * (8 - k) / 4, pan=0.3)

    # --- cold open: the question lands, the candidates pop, the claim, the promise, a rush
    c0 = CD
    A.place(fx, A.boom(1.1), c0 + TL.CO_FORM)
    for k, m in enumerate((38, 50, 57, 62, 65, 69, 74)):
        note(m, c0 + TL.CO_FORM + 0.04 * k, 0.46)
    A.place(fx, A.thud(0.6), c0 + TL.CO_L1)
    for k, m in enumerate((72, 69, 65)):
        A.place(plk, A.pluck(m, 0.34 - 0.05 * k), c0 + TL.CO_L1 + 0.15 + 0.16 * k, pan=-0.3 + 0.2 * k)
    A.place(fx, A.boom(0.8), c0 + TL.CO_L2)
    A.place(bel, A.bell(81, 0.5), c0 + TL.CO_L2 + 0.05, pan=0.2)
    A.place(fx, A.boom(0.9), c0 + TL.CO_L3)
    for k, m in enumerate((45, 57, 64, 69, 72)):
        note(m, c0 + TL.CO_L3 + 0.05 * k, 0.42)
    groove(c0 + TL.CO_L1, c0 + TL.CO_GO, 0.8, bpm=100, hats=True, bass=False)
    A.place(fx, A.riser(1.0, 0.7), c0 + TL.CO_GO)

    # --- hook: the chat, 它, the turn, the title
    h = HK
    groove(h + TL.H_CHAT + 0.6, h + TL.H_PLATE - 0.2, 0.7, bpm=100)
    for tt in np.arange(h + TL.H_CHAT + 0.9, h + TL.H_PLATE - 0.6, 0.22):
        if rng.random() < 0.6:
            A.place(tick, A.blip(int(rng.choice([88, 91, 93])), 0.05), tt, pan=rng.uniform(-0.5, 0.5))
    A.place(bel, A.bell(79, 0.35, ratio=2.0, dur=3.0), h + TL.H_PLATE + 0.3, pan=-0.4)
    A.place(fx, A.thud(0.35), h + TL.H_HOP + 0.5)
    note(65, h + TL.H_LIE + 0.1, 0.28)
    note(60, h + TL.H_LIE + 0.5, 0.24)
    A.place(bel, A.bell(81, 0.5), h + TL.H_IT, pan=0.1)
    for k, m in enumerate((69, 72, 76, 79, 81, 84, 88)):
        A.place(bel, A.bell(m, 0.22, ratio=2.0, dur=3.0), h + TL.H_LINK + 0.11 * k, pan=-0.6 + 0.2 * k)
    note(41, h + TL.H_LINK + 0.9, 0.4)
    A.place(fx, A.riser(0.6, 0.35), h + TL.H_SWAP - 0.55)
    A.place(bel, A.bell(76, 0.45, ratio=2.0), h + TL.H_SWAP + 0.05, pan=-0.2)
    for k, m in enumerate((86, 84, 81, 79, 76)):
        A.place(bel, A.bell(m, 0.18, ratio=2.0, dur=2.5), h + TL.H_RE + 0.12 * k, pan=0.5 - 0.2 * k)
    note(38, h + TL.H_RE + 1.0, 0.4)
    phrase(h + TL.H_TURN, h + TL.H_TITLE - 0.4, 0.9, 0.3, lo=62, hi=76, prob=1.0)
    A.place(fx, A.riser(1.6, 0.6), h + TL.H_TITLE - 1.6)
    A.place(fx, A.boom(1.0), h + TL.H_TITLE)
    for k, m in enumerate((41, 53, 57, 60, 64, 67, 72)):
        note(m, h + TL.H_TITLE + 0.07 * k, 0.44)
    A.place(bel, A.bell(81, 0.6), h + TL.H_TITLE + 0.5, pan=-0.3)
    A.place(fx, A.riser(1.2, 0.5), h + TL.H_TITLE + 5.1)

    # --- sky: the warp, numbers, a star, the sky of meaning
    k0 = SK
    A.place(fx, A.boom(0.6), k0)
    for i in range(14):
        A.place(tick, A.blip(int([84, 86, 88, 91][i % 4]), 0.14), k0 + TL.S_NUM + 0.8 + i / 3.5, pan=0.2)
    A.place(fx, A.riser(1.4, 0.3), k0 + TL.S_COUNT + 2.6)
    A.place(bel, A.bell(84, 0.4, ratio=2.0), k0 + TL.S_COUNT + 4.0, pan=-0.3)
    A.place(fx, A.riser(1.0, 0.3), k0 + TL.S_STAR - 0.2)
    A.place(bel, A.bell(88, 0.45), k0 + TL.S_STAR + 1.2, pan=-0.4)
    for g_ in range(5):
        A.place(bel, A.bell([72, 76, 79, 81, 84][g_], 0.22, ratio=2.0, dur=3.0), k0 + TL.S_GROUPS + 0.24 * g_,
                pan=-0.6 + 0.3 * g_)
    groove(k0 + TL.S_GROUPS, k0 + TL.S_IT - 0.3, 0.45, bpm=100, hats=True)
    for i in range(2):
        A.place(plk, A.pluck([69, 76][i], 0.32), k0 + TL.S_KING + 0.6 + 1.0 * i, pan=-0.2 + 0.4 * i)
    A.place(plk, A.pluck(72, 0.3), k0 + TL.S_APPLE + 0.4, pan=0.3)
    A.place(plk, A.pluck(67, 0.3), k0 + TL.S_APPLE + 2.6, pan=0.4)
    note(50, k0 + TL.S_IT, 0.3)
    A.place(bel, A.bell(81, 0.4), k0 + TL.S_PULL + 0.2, pan=-0.3)
    note(38, k0 + TL.S_PULL + 1.2, 0.36)
    phrase(k0 + 15.0, k0 + 45.0, 2.0, 0.24, lo=62, hi=79, prob=0.7)

    # --- sequential: one tick per word; a whisper that fades with every retelling
    s = SQ
    for k in range(3):
        t0 = s + TL.SEQ_PASS0 + k * TL.SEQ_PASS_DT
        A.place(fx, whisper(1.1, 0.5 * 0.62 ** k), t0, pan=-0.5 + 0.33 * k)
        A.place(plk, A.pluck(int([81, 76, 72][k]), 0.42 * 0.7 ** k), t0 + 1.0, pan=-0.4 + 0.3 * k)
    steps = [s + TL.SEQ_READ0 + k * TL.SEQ_READ_DT for k in range(int((16.4 - TL.SEQ_READ0) / TL.SEQ_READ_DT) + 1)]
    steps += list(np.arange(s + 16.4 + 1.0, s + 29.4, 1.0))
    for k, ts in enumerate(steps):
        A.place(tick, A.blip(int([79, 81, 84, 86][k % 4]), 0.32), ts, pan=-0.4 + 0.05 * (k % 16))
        A.place(fx, A.soft_kick(0.18), ts)
    phrase(s + 1.0, s + 29.0, 2.6, 0.26, lo=57, hi=74, prob=0.6)

    # --- core: every word at once; Q, K, V; multiply and add; softmax; the mix; the line; the table
    c = CO
    for k, m in enumerate((65, 69, 72, 76, 77, 81, 84, 88, 89)):
        A.place(bel, A.bell(m, 0.2, ratio=2.0, dur=4.0), c + 0.5, pan=-0.8 + 0.2 * k)
    for k in range(9):
        A.place(tick, A.blip(int(rng.choice([88, 91, 93, 96])), 0.12), c + TL.C_WEB + 0.07 * k,
                pan=rng.uniform(-0.8, 0.8))
    A.place(bel, A.bell(77, 0.4, ratio=3.5), c + TL.C_ASK + 0.3, pan=0.3)
    groove(c + TL.C_MAT, c + 25.8, 0.55, bpm=100)
    for k in range(3):
        A.place(fx, A.soft_kick(0.18), c + TL.C_MAT + 0.8 + 0.4 * k)
        A.place(plk, A.pluck([57, 60, 64][k], 0.3), c + TL.C_MAT + 1.6 + 0.4 * k, pan=-0.3 + 0.3 * k)
    for tk, m in ((TL.C_Q, 84), (TL.C_K, 79), (TL.C_V, 76)):
        A.place(bel, A.bell(m, 0.42), c + tk, pan=-0.2)
    A.place(bel, A.bell(81, 0.45), c + TL.C_QARROW, pan=-0.3)
    groove(c + TL.C_KARROWS, c + TL.C_SOFT - 0.2, 0.5, bpm=100)
    for i in range(4):
        A.place(plk, A.pluck([69, 72, 76, 79][i], 0.3), c + TL.C_KARROWS + 0.6 * i, pan=-0.4 + 0.25 * i)
    for t0, res in ((TL.C_SUM, 84), (TL.C_ALIGN + 2.2, 62)):
        for k in range(10):
            A.place(tick, A.blip(int([84, 86, 88, 91, 93][k % 5]), 0.16), c + t0 + 0.22 * k, pan=-0.3 + 0.06 * k)
        A.place(bel, A.bell(res, 0.45, ratio=2.0), c + t0 + 2.3, pan=0.2)
    for i in range(4):
        A.place(plk, A.pluck([79, 76, 72, 67][i], 0.3), c + TL.C_SCORES + 0.5 * i, pan=0.4)
    groove(c + TL.C_SOFT, c + TL.C_KNOW, 0.7, bpm=100, build=True)
    for i in range(5):
        A.place(bel, A.bell([72, 76, 79, 81, 84][i], 0.2, ratio=2.0, dur=3.0), c + TL.C_SOFT + 1.0 + 0.14 * i,
                pan=-0.3 + 0.15 * i)
    for k, m in enumerate((53, 60, 65, 69)):
        note(m, c + TL.C_SOFT + 3.0 + 0.06 * k, 0.32)
    for i in range(5):
        A.place(bel, A.bell([88, 84, 81, 79, 76][i], 0.25 - 0.03 * i, ratio=2.0, dur=3.0),
                c + TL.C_MIX + 0.6 + 0.25 * i, pan=0.3)
    note(41, c + TL.C_MIX + 2.4, 0.38)
    A.place(fx, A.riser(2.4, 0.5), c + TL.C_KNOW - 2.4)
    A.place(fx, A.boom(1.0), c + TL.C_KNOW)
    for k, m in enumerate((41, 53, 57, 60, 64, 69, 72)):
        note(m, c + TL.C_KNOW + 0.06 * k, 0.46)
    A.place(fx, A.riser(1.4, 0.5), c + TL.C_FORMULA - 1.4)
    A.place(fx, A.boom(0.9), c + TL.C_FORMULA)
    for k, m in enumerate((46, 58, 62, 65, 69, 74)):
        note(m, c + TL.C_FORMULA + 0.08 * k, 0.42)
    for tk, m in ((TL.C_QK, 81), (TL.C_SM, 84), (TL.C_TV, 88), (TL.C_SQ, 79)):
        A.place(bel, A.bell(m, 0.4), c + tk, pan=0.2)
    A.place(fx, A.riser(0.8, 0.45), c + TL.C_GRID)
    A.place(fx, A.boom(1.0), c + TL.C_GRID + 0.8)
    for k, m in enumerate((41, 53, 60, 65, 69, 72, 77, 81)):
        note(m, c + TL.C_GRID + 0.8, 0.38)
    groove(c + TL.C_GRID + 0.8, c + 97.4, 0.8, bpm=100)
    phrase(c + 1.0, c + TL.C_FORMULA - 1.0, 2.4, 0.22, lo=62, hi=79, prob=0.6)
    phrase(c + TL.C_GRID + 2.0, c + 97.4, 1.2, 0.3, lo=65, hi=84)

    # --- heads: eight voices, then all of them
    hd = HD
    scale = [69, 72, 74, 76, 79, 81, 84, 86]
    groove(hd, hd + TL.HEADS_ALL, 0.6, bpm=100, build=True)
    for k in range(8):
        t0 = hd + TL.HEADS_T0 + k * TL.HEADS_DT
        A.place(bel, A.bell(scale[k], 0.38, ratio=[2.0, 3.5][k % 2], dur=3.0), t0, pan=-0.7 + 0.2 * k)
        A.place(plk, A.pluck(scale[k] - 12, 0.25), t0 + 0.05, pan=-0.7 + 0.2 * k)
    for k, m in enumerate(scale):
        A.place(bel, A.bell(m, 0.2, ratio=2.0, dur=4.0), hd + TL.HEADS_ALL + 0.03 * k, pan=-0.7 + 0.2 * k)
    A.place(fx, A.boom(0.8), hd + TL.HEADS_ALL)
    groove(hd + TL.HEADS_ALL, hd + 21.4, 0.85, bpm=100)
    A.place(fx, A.riser(1.4, 0.35), hd + TL.HEADS_MERGE - 1.4)
    for k, m in enumerate((41, 53, 60, 64, 69, 72)):
        note(m, hd + TL.HEADS_MERGE + 0.1 * k, 0.4)

    # --- order: a light trot, a swap, and the waves as four pure tones
    o = OR
    for tt in np.arange(o + 0.6, o + TL.ORD_WAVES - 0.4, 0.333):
        A.place(fx, A.soft_kick(0.16), tt)
        A.place(tick, A.blip(int(rng.choice([86, 88])), 0.08), tt + 0.166, pan=rng.uniform(-0.3, 0.3))
    A.place(fx, A.riser(0.8, 0.35), o + TL.ORD_SWAP - 0.6)
    A.place(bel, A.bell(81, 0.45, ratio=2.0), o + TL.ORD_SWAP + 0.6, pan=0.4)
    A.place(bel, A.bell(76, 0.45, ratio=2.0), o + TL.ORD_SWAP + 0.7, pan=-0.4)
    for k, f in enumerate((220.0, 330.0, 495.0, 742.5)):
        A.place(shim, sine_tone(f, 9.0 - 0.4 * k, 0.07), o + TL.ORD_WAVES + 0.4 * k, pan=-0.5 + 0.33 * k)
    for k in range(3):
        A.place(bel, A.bell(79 + 3 * k, 0.3), o + TL.ORD_WAVES + 1.0 + 0.4 * k, pan=-0.4 + 0.4 * k)

    # --- learn: layers stacking, the star moving, the dials turning, the count, the result
    ln = LN
    A.place(bel, A.bell(76, 0.4), ln + TL.L_LAYER, pan=-0.2)
    for k in range(1, 6):
        tb = ln + TL.L_STACK + 2.0 * k / 5
        A.place(fx, A.soft_kick(0.25 + 0.03 * k), tb)
        A.place(tick, A.blip(int(72 + 2 * k), 0.2), tb, pan=-0.2 + 0.08 * k)
    A.place(fx, A.riser(1.4, 0.45), ln + TL.L_96)
    for k in range(24):
        tb = ln + TL.L_96 + 0.6 + 1.4 * (k / 24) ** 0.5
        A.place(tick, A.blip(int(84 + (k % 8)), 0.07), tb, pan=rng.uniform(-0.5, 0.5))
    for k in range(6):
        A.place(plk, A.pluck([60, 64, 67, 69, 72, 76][k], 0.28), ln + TL.L_MOVE + 0.8 + k * 4.0 / 6, pan=-0.4)
    A.place(bel, A.bell(77, 0.4, ratio=3.5), ln + TL.L_WHO + 0.8, pan=0.2)
    groove(ln + TL.L_MASK, ln + TL.L_COUNT + 2.9, 0.75, bpm=100, build=True)
    for tt in np.arange(ln + TL.L_MASK + 0.6, ln + TL.L_NUDGE, 1.55):
        A.place(tick, A.blip(91, 0.1), tt, pan=0.5)
    for tt in np.arange(ln + TL.L_NUDGE + 0.4, ln + TL.L_COUNT, 0.11):
        if rng.random() < 0.55:
            A.place(tick, A.blip(int(rng.choice([93, 96, 98, 100])), 0.04), tt, pan=rng.uniform(-0.8, 0.8))
    A.place(fx, A.riser(2.6, 0.55), ln + TL.L_COUNT + 0.3)
    A.place(fx, A.boom(1.0), ln + TL.L_COUNT + 2.9)
    for k, m in enumerate((38, 50, 57, 62, 65, 69)):
        note(m, ln + TL.L_COUNT + 2.9 + 0.06 * k, 0.44)
    A.place(bel, A.bell(84, 0.45), ln + TL.L_PAPER, pan=-0.3)
    A.place(plk, A.pluck(64, 0.3), ln + TL.L_PAPER + 1.0)
    A.place(plk, A.pluck(72, 0.38), ln + TL.L_PAPER + 1.5)
    phrase(ln + TL.L_PAPER + 2.0, ln + 45.0, 1.6, 0.28, lo=64, hi=81)
    A.place(fx, A.riser(1.0, 0.45), ln + 45.0)

    # --- mind: the novel, the rule, Kepler and Newton, one word, the sky, starlings, neurons, the question
    md = MD
    A.place(fx, A.boom(0.7), md + TL.M_ASK)
    A.place(fx, A.boom(0.8), md + TL.M_ASK + 1.2)
    for k, m in enumerate((45, 57, 64, 69)):
        note(m, md + TL.M_ASK + 1.2 + 0.06 * k, 0.4)
    A.place(fx, whisper(1.4, 0.18), md + TL.M_BOOK, pan=0.0)
    for k in range(5):
        A.place(plk, A.pluck([67, 69, 72, 74, 76][k], 0.26), md + TL.M_CLUES + 0.3 * k, pan=-0.4 + 0.2 * k)
        A.place(bel, A.bell([79, 81, 84, 86, 88][k], 0.2, ratio=2.0, dur=3.0), md + TL.M_CLUES + 1.2 + 0.36 * k,
                pan=0.4 - 0.2 * k)
    for k, m in enumerate((45, 57, 64, 67, 71)):
        note(m, md + TL.M_FORCED + 0.1 * k, 0.34)
    for tt in np.arange(md + TL.M_CANT + 0.2, md + TL.M_CANT + 2.6, 0.12):
        A.place(tick, A.blip(int(rng.choice([91, 93, 96])), 0.04), tt, pan=rng.uniform(-0.7, 0.7))
    for k in range(3):
        A.place(bel, A.bell([72, 76, 79][k], 0.35), md + TL.M_CANT + 2.6 + 0.4 * k, pan=-0.2 + 0.2 * k)
    A.place(fx, A.riser(1.0, 0.45), md + TL.M_GEN - 1.0)
    A.place(fx, A.boom(0.9), md + TL.M_GEN)
    for k, m in enumerate((41, 53, 57, 60, 64, 67, 72)):
        note(m, md + TL.M_GEN + 0.06 * k, 0.42)
    for k in range(0, 46, 3):
        A.place(tick, A.blip(int(rng.choice([86, 88, 91])), 0.08), md + TL.M_KEPLER + 0.2 + 2.6 * 0.8 * k / 46,
                pan=rng.uniform(-0.6, 0.6))
    A.place(fx, A.riser(2.2, 0.35), md + TL.M_KEPLER + 3.4)
    A.place(bel, A.bell(84, 0.45), md + TL.M_KEPLER + 5.6, pan=0.3)
    A.place(fx, A.boom(0.7), md + TL.M_NEWTON)
    for k, m in enumerate((41, 53, 60, 65, 69)):
        note(m, md + TL.M_NEWTON + 0.08 * k, 0.4)
    A.place(bel, A.bell(81, 0.45), md + TL.M_CAT + 3.3, pan=-0.2)
    for k in range(12):
        A.place(bel, A.bell(int(67 + 2 * k), 0.12, ratio=2.0, dur=3.0), md + TL.M_SKY + 0.6 + 0.3 * k,
                pan=rng.uniform(-0.8, 0.8))
    A.place(fx, A.riser(1.6, 0.6), md + TL.M_EMERGE - 1.6)
    A.place(fx, A.boom(1.2), md + TL.M_EMERGE)
    for k, m in enumerate((41, 53, 57, 60, 64, 67, 72, 76)):
        note(m, md + TL.M_EMERGE + 0.05 * k, 0.46)
    groove(md + TL.M_BIRDS, md + TL.M_DEBATE - 0.3, 0.8, bpm=100)
    for tt in np.arange(md + TL.M_BIRDS + 0.5, md + TL.M_NEURON - 0.4, 0.07):
        if rng.random() < 0.6:
            A.place(tick, A.blip(int(rng.choice([96, 98, 100, 103])), 0.03), tt, pan=rng.uniform(-0.9, 0.9))
    A.place(plk, A.pluck(64, 0.36), md + TL.M_NEURON + 1.2)
    for tt in np.arange(md + TL.M_NEURON + 2.2, md + TL.M_SCALE - 0.4, 0.18):
        if rng.random() < 0.5:
            A.place(plk, A.pluck(int(rng.choice([72, 76, 79, 84])), 0.1), tt, pan=rng.uniform(-0.8, 0.8))
    for k in range(6):
        A.place(bel, A.bell([72, 74, 76, 79, 81, 84][k], 0.32), md + TL.M_SCALE + 2.2 + 0.5 * k, pan=-0.5 + 0.2 * k)
    A.place(fx, A.boom(0.7), md + TL.M_DEBATE)
    for k, m in enumerate((46, 58, 62, 65, 69)):
        note(m, md + TL.M_GUESS + 0.1 * k, 0.36)
    for k, m in enumerate((45, 57, 62, 64, 69)):
        note(m, md + TL.M_QUESTION + 0.12 * k, 0.34)
    A.place(bel, A.bell(86, 0.35), md + TL.M_QUESTION + 3.6, pan=0.3)
    phrase(md + TL.M_FORCED + 1.0, md + TL.M_GEN - 0.5, 1.8, 0.24, lo=62, hi=77, prob=0.6)
    phrase(md + TL.M_KEPLER, md + TL.M_EMERGE - 1.0, 2.0, 0.24, lo=64, hi=79, prob=0.6)
    phrase(md + TL.M_DEBATE, md + 97.0, 2.2, 0.24, lo=62, hi=77, prob=0.6)

    # --- after: the page, the note, the song, the T, every conversation
    f = AF
    phrase(f + 0.6, f + TL.A_NOTE, 1.5, 0.3, lo=64, hi=79)
    A.place(bel, A.bell(81, 0.45), f + TL.A_NOTE + 0.3, pan=0.3)
    for k, m in enumerate((53, 60, 64, 69, 72)):
        note(m, f + TL.A_SONG + 0.12 * k, 0.36)
    phrase(f + TL.A_SONG + 1.0, f + TL.A_GPT, 0.75, 0.3, lo=65, hi=84, prob=0.9)
    A.place(fx, A.boom(0.9), f + TL.A_GPT)
    for k, m in enumerate((38, 50, 57, 62, 65, 69)):
        note(m, f + TL.A_GPT + 0.08 * k, 0.4)
    groove(f + TL.A_CHAT, f + 29.2, 0.55, bpm=100)
    for tt in np.arange(f + TL.A_CHAT + 0.7, f + 29.0, 0.24):
        if rng.random() < 0.6:
            A.place(tick, A.blip(int(rng.choice([88, 91, 93])), 0.05), tt, pan=rng.uniform(-0.5, 0.5))

    # --- ending: quiet, a held breath, the line, the last chord
    e = EN
    phrase(e + 0.8, e + TL.END_LINE - 0.4, 2.2, 0.26, lo=62, hi=77, prob=0.7)
    for k in range(12):
        A.place(tick, A.blip(int(rng.choice([88, 91, 93, 96, 98])), 0.1), e + 6.0 + 0.4 * k,
                pan=rng.uniform(-0.7, 0.7))
    A.place(fx, A.riser(1.2, 0.35), e + TL.END_LINE - 1.2)
    A.place(fx, A.boom(0.6), e + TL.END_LINE)
    for k, m in enumerate((41, 53, 57, 60, 64, 67)):
        note(m, e + TL.END_LINE + 0.1 * k, 0.4)
    A.place(bel, A.bell(81, 0.45), e + TL.END_LINE + 1.0, pan=-0.3)
    for k, m in enumerate((38, 50, 57, 62, 64, 69)):
        note(m, e + TL.END_TITLE + 0.1 * k, 0.38)
    A.place(bel, A.bell(86, 0.4), e + TL.END_TITLE + 0.8, pan=0.3)
    return dict(pad=pad, sub=sub, pno=pno, bel=bel, fx=fx, tick=tick, plk=plk, shim=shim, drm=drm)


def main():
    out_dir = ROOT / "build"
    out_dir.mkdir(exist_ok=True)
    ir = A.build_ir()
    S = score()

    def norm(x, rms):
        return x * (rms / (np.sqrt((x ** 2).mean()) + 1e-12))

    pad = A.hp(A.lp(norm(S["pad"], 0.06), 3000), 90)
    sub = norm(S["sub"], 0.03)
    pno = norm(S["pno"], 0.045)
    bel = S["bel"] / (np.abs(S["bel"]).max() + 1e-9) * 0.28
    fx = S["fx"] / (np.abs(S["fx"]).max() + 1e-9) * 0.45
    tick = S["tick"] / (np.abs(S["tick"]).max() + 1e-9) * 0.13
    plk = S["plk"] / (np.abs(S["plk"]).max() + 1e-9) * 0.3
    shim = norm(S["shim"], 0.014)
    drm = A.lp(S["drm"], 9000) / (np.abs(S["drm"]).max() + 1e-9) * 0.42
    dry = pad + sub + pno + A.pingpong(pno) * 0.18 + bel + fx + tick + plk + shim + drm
    send = pad * 0.3 + pno * 0.5 + bel * 0.8 + tick * 0.6 + plk * 0.6 + shim * 0.6 + fx * 0.25 + drm * 0.08
    mix = A.hp(dry + A.reverb(send, ir) * 0.55, 28)
    arc = A.automation([(0, -1), (CD + 5.6, 0), (HK, -4), (HK + TL.H_PLATE, -5),
                        (HK + TL.H_TITLE, 0), (HK + TL.H_TITLE + 4.2, -2), (SK, -2), (SK + 3, -5), (SK + 45, -4), (SQ, -5),
                        (SQ + 22, -3), (CO, -4), (CO + 26, -4), (CO + TL.C_KNOW, 0), (CO + TL.C_FORMULA, 0),
                        (CO + 80, -3), (CO + TL.C_GRID, 0), (HD, -3), (HD + TL.HEADS_ALL, 0), (OR, -3), (OR + 12, -4),
                        (LN, -4), (LN + TL.L_COUNT, -2), (LN + TL.L_COUNT + 3, 0), (LN + 40, -3),
                        (MD, -2), (MD + 5, -5), (MD + TL.M_GEN, -1), (MD + TL.M_KEPLER, -4), (MD + TL.M_EMERGE, 0),
                        (MD + TL.M_NEURON, -1), (MD + TL.M_DEBATE, -4), (MD + TL.M_QUESTION, -5),
                        (AF, -4), (AF + TL.A_GPT, 0), (AF + TL.A_CHAT, -2), (EN, -6), (EN + TL.END_LINE, -1),
                        (DURATION, -3)], N)
    mix *= 10 ** (arc / 20)
    env = np.sqrt(A.lp(np.mean(mix ** 2, 1), 3.0, 1).clip(1e-12))
    thr = np.percentile(env, 85)
    mix *= np.where(env > thr, (env / thr) ** (1 / 2.0 - 1), 1.0)[:, None]
    fi = int(0.05 * SR)
    mix[:fi] *= A.rc(fi)[:, None]
    fo = int(3.0 * SR)
    mix[-fo:] *= (1 - A.rc(fo))[:, None] ** 1.5
    lufs, peak = A.master(mix, out_dir / "bgm.wav", -14.5)
    print(f"bgm: {out_dir / 'bgm.wav'}  {DURATION}s  {lufs:.1f} LUFS  peak {peak:.1f} dBFS")


if __name__ == "__main__":
    main()
