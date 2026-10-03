"""Original score for 注意力 · Attention Is All You Need, synthesised from scratch.

Warm and unhurried: pads, felt piano, glass bells, plucked strings.  Sound
follows the ideas: one soft tick per word while the old way reads one word at
a time, a whisper that gets quieter with every retelling, everything ringing
at the same instant when the attention table appears, eight voices for eight
heads, and the position waves heard as four pure tones.

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
P_, SQ, AO, QK, HD, OR, AF, EN = (START[k] for k in ("prologue", "sequential", "atonce", "qkv", "heads", "order",
                                                     "after", "ending"))
PLAN = [(0.0, "Fmaj9"), (P_ + 5.0, "Dm9"), (P_ + 7.8, "Bbmaj9"), (P_ + 10.6, "Gm9"), (P_ + 15.4, "Fmaj9"),
        (P_ + 19.0, "C69"),
        (SQ, "Dm9"), (SQ + 6.0, "Bbmaj7#11"), (SQ + 11.4, "Gm9"), (SQ + 17.0, "A7sus4"), (SQ + 22.4, "Dm9"),
        (SQ + 28.0, "Bbmaj9"),
        (AO, "Fmaj9"), (AO + 5.4, "C69"), (AO + 10.8, "Am7"), (AO + 16.2, "Bbmaj9"), (AO + 22.6, "Fmaj9"),
        (QK, "Dm9"), (QK + 5.0, "Bbmaj9"), (QK + 10.0, "Gm9"), (QK + 15.0, "Am7"), (QK + 20.2, "Bbmaj7#11"),
        (QK + TL.QKV_FORMULA, "Fmaj9"), (QK + 34.0, "C69"),
        (HD, "Am7"), (HD + 5.2, "Dm9"), (HD + 12.2, "Bbmaj9"), (HD + 18.8, "Fmaj9"),
        (OR, "Dm9"), (OR + 6.0, "Gm9"), (OR + 11.6, "C69"), (OR + 19.0, "Fmaj9"),
        (AF, "Bbmaj9"), (AF + 5.4, "Gm9"), (AF + 11.2, "Fmaj9"), (AF + 17.0, "Dm9"), (AF + 22.6, "Bbmaj9"),
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


def sine_tone(f, dur, vel, attack=0.6):
    t = np.arange(int(dur * SR)) / SR
    env = np.clip(t / attack, 0, 1) * np.clip((dur - t) / 1.2, 0, 1)
    return np.sin(2 * np.pi * f * t) * env * vel


def score():
    pad, sub, pno, bel, fx, tick, plk, shim = (A.stereo(N) for _ in range(8))
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

    # --- prologue: the teaser's beats, then the turn and the title
    p = P_
    for k in range(9 + 7):
        A.place(tick, A.blip(int(rng.choice([84, 86, 88, 91])), 0.1), p + TL.P_T1 + k / 7.5, pan=-0.3 + 0.04 * k)
    A.place(bel, A.bell(74, 0.25, ratio=2.0, dur=3.0), p + 0.5, pan=0.3)
    A.place(bel, A.bell(79, 0.25, ratio=2.0, dur=3.0), p + 0.9, pan=-0.4)
    A.place(fx, A.thud(0.35), p + TL.P_HOP + 0.5)
    note(65, p + TL.P_LIE + 0.1, 0.28)
    note(60, p + TL.P_LIE + 0.5, 0.24)
    A.place(bel, A.bell(81, 0.5), p + TL.P_IT, pan=0.1)
    for k, m in enumerate((69, 72, 76, 79, 81, 84, 88)):
        A.place(bel, A.bell(m, 0.22, ratio=2.0, dur=3.0), p + TL.P_FAN + 0.11 * k, pan=-0.6 + 0.2 * k)
    note(41, p + TL.P_FAN + 0.9, 0.4)
    note(65, p + TL.P_FAN + 0.95, 0.35)
    A.place(fx, A.riser(0.6, 0.35), p + TL.P_SWAP - 0.55)
    A.place(bel, A.bell(76, 0.45, ratio=2.0), p + TL.P_SWAP + 0.05, pan=-0.2)
    for k, m in enumerate((86, 84, 81, 79, 76)):
        A.place(bel, A.bell(m, 0.18, ratio=2.0, dur=2.5), p + TL.P_RE + 0.12 * k, pan=0.5 - 0.2 * k)
    note(38, p + TL.P_RE + 1.0, 0.4)
    note(62, p + TL.P_RE + 1.05, 0.32)
    A.place(fx, A.boom(0.5), p + TL.P_NAME)
    for k, m in enumerate((46, 58, 62, 65, 69, 74)):
        note(m, p + TL.P_NAME + 0.08 * k, 0.36)
    phrase(p + 11.0, p + 14.4, 0.9, 0.3, lo=62, hi=76, prob=1.0)
    A.place(fx, A.riser(1.6, 0.6), p + 13.8)
    A.place(fx, A.boom(1.0), p + 15.4)
    for k, m in enumerate((41, 53, 57, 60, 64, 67, 72)):
        note(m, p + 15.4 + 0.07 * k, 0.44)
    A.place(bel, A.bell(81, 0.6), p + 15.9, pan=-0.3)
    phrase(p + 17.3, p + 23.5, 1.6, 0.32, lo=64, hi=81)

    # --- sequential: one tick per word; a whisper that fades with every retelling
    s = SQ
    for k in range(3):
        t0 = s + TL.SEQ_PASS0 + k * TL.SEQ_PASS_DT
        A.place(fx, whisper(1.1, 0.5 * 0.62 ** k), t0, pan=-0.5 + 0.33 * k)
        A.place(plk, A.pluck(int([81, 76, 72, 69][k]), 0.42 * 0.7 ** k), t0 + 1.0, pan=-0.4 + 0.3 * k)
    steps = [s + TL.SEQ_READ0 + k * TL.SEQ_READ_DT for k in range(int((22.0 - TL.SEQ_READ0) / TL.SEQ_READ_DT) + 1)]
    steps += list(np.arange(s + 22.0 + 1.3, s + 33.4, 1.3))
    for k, ts in enumerate(steps):
        A.place(tick, A.blip(int([79, 81, 84, 86][k % 4]), 0.32), ts, pan=-0.4 + 0.05 * (k % 16))
        A.place(fx, A.soft_kick(0.18), ts)
    phrase(s + 1.0, s + 33.0, 2.6, 0.26, lo=57, hi=74, prob=0.6)

    # --- at once: everything sounds together
    a = AO
    for k, m in enumerate((65, 69, 72, 76, 77, 81, 84, 88, 89)):
        A.place(bel, A.bell(m, 0.22, ratio=2.0, dur=4.0), a + 1.0, pan=-0.8 + 0.2 * k)
    A.place(fx, A.boom(0.45), a + 1.0)
    for k in range(18):
        A.place(tick, A.blip(int(rng.choice([88, 91, 93, 96])), 0.12), a + TL.ATO_WEB + 0.07 * k,
                pan=rng.uniform(-0.8, 0.8))
    A.place(bel, A.bell(81, 0.45), a + 10.8, pan=0.2)
    A.place(fx, A.riser(1.2, 0.35), a + TL.ATO_TABLE)
    A.place(fx, A.boom(0.7), a + TL.ATO_TABLE + 1.2)
    for k, m in enumerate((46, 58, 65, 69, 72, 77, 81)):
        note(m, a + TL.ATO_TABLE + 1.2, 0.38)
    phrase(a + 18.0, a + 31.0, 1.8, 0.3, lo=65, hi=84)

    # --- QKV: a question, eight labels, eight contents, the match, the line
    q = QK
    A.place(bel, A.bell(79, 0.45), q + TL.QKV_Q, pan=-0.3)
    for k in range(8):
        A.place(tick, A.blip(int(84 + [0, 2, 4, 7, 9, 12, 14, 16][k]), 0.22), q + TL.QKV_K + 0.12 * k,
                pan=-0.1 + 0.11 * k)
        A.place(plk, A.pluck(int([57, 60, 62, 64, 67, 69, 72, 74][k]), 0.22), q + TL.QKV_V + 0.1 * k,
                pan=-0.1 + 0.11 * k)
    A.place(fx, A.riser(1.0, 0.4), q + TL.QKV_MATCH - 0.2)
    A.place(bel, A.bell(84, 0.5), q + TL.QKV_MATCH + 1.2, pan=0.3)
    A.place(bel, A.bell(76, 0.35, ratio=2.0), q + TL.QKV_MATCH + 2.4, pan=0.4)
    for k, m in enumerate((72, 76, 79, 84)):
        A.place(bel, A.bell(m, 0.25, ratio=2.0, dur=3.0), q + TL.QKV_MATCH + 3.4 + 0.25 * k, pan=0.4 - 0.25 * k)
    phrase(q + 1.0, q + TL.QKV_MATCH, 2.2, 0.26, lo=62, hi=77, prob=0.65)
    A.place(fx, A.riser(1.4, 0.5), q + TL.QKV_FORMULA - 1.4)
    A.place(fx, A.boom(0.95), q + TL.QKV_FORMULA)
    for k, m in enumerate((41, 53, 57, 60, 64, 67, 72)):
        note(m, q + TL.QKV_FORMULA + 0.08 * k, 0.44)
    A.place(bel, A.bell(84, 0.55), q + TL.QKV_FORMULA + 0.5, pan=-0.3)
    phrase(q + TL.QKV_FORMULA + 2.0, q + 41.0, 1.7, 0.32, lo=64, hi=84)

    # --- heads: eight voices, then all of them
    h = HD
    scale = [69, 72, 74, 76, 79, 81, 84, 86]
    for k in range(8):
        t0 = h + TL.HEADS_T0 + k * TL.HEADS_DT
        A.place(bel, A.bell(scale[k], 0.38, ratio=[2.0, 3.5][k % 2], dur=3.0), t0, pan=-0.7 + 0.2 * k)
        A.place(plk, A.pluck(scale[k] - 12, 0.25), t0 + 0.05, pan=-0.7 + 0.2 * k)
    for k, m in enumerate(scale):
        A.place(bel, A.bell(m, 0.2, ratio=2.0, dur=4.0), h + TL.HEADS_ALL + 0.03 * k, pan=-0.7 + 0.2 * k)
    A.place(fx, A.boom(0.55), h + TL.HEADS_ALL)
    A.place(fx, A.riser(1.6, 0.35), h + 18.0)
    for k, m in enumerate((41, 53, 60, 64, 69, 72)):
        note(m, h + 19.6 + 0.1 * k, 0.4)
    phrase(h + 20.5, h + 25.5, 1.4, 0.3, lo=65, hi=81)

    # --- order: a playful chase, a swap, and the waves as tones
    o = OR
    for tt in np.arange(o + 0.6, o + TL.ORD_WAVES, 0.333):
        A.place(fx, A.soft_kick(0.14), tt)
        A.place(tick, A.blip(int(rng.choice([86, 88])), 0.08), tt + 0.166, pan=rng.uniform(-0.3, 0.3))
    A.place(fx, A.riser(0.8, 0.35), o + TL.ORD_SWAP - 0.6)
    A.place(bel, A.bell(81, 0.45, ratio=2.0), o + TL.ORD_SWAP + 0.6, pan=0.4)
    A.place(bel, A.bell(76, 0.45, ratio=2.0), o + TL.ORD_SWAP + 0.7, pan=-0.4)
    for k, f in enumerate((220.0, 330.0, 495.0, 742.5)):
        A.place(shim, sine_tone(f, 13.5 - 0.4 * k, 0.07), o + TL.ORD_WAVES + 0.4 * k, pan=-0.5 + 0.33 * k)
    for k in range(3):
        A.place(bel, A.bell(79 + 3 * k, 0.3), o + TL.ORD_WAVES + 1.0 + 0.4 * k, pan=-0.4 + 0.4 * k)
    phrase(o + 12.0, o + 25.5, 1.8, 0.28, lo=62, hi=79)

    # --- after: the page, the note, the song, the T, the tower
    f = AF
    phrase(f + 0.6, f + TL.AFT_BYLINE, 1.5, 0.3, lo=64, hi=79)
    A.place(bel, A.bell(81, 0.45), f + TL.AFT_BYLINE + 0.4, pan=0.3)
    for k, m in enumerate((53, 60, 64, 69, 72)):
        note(m, f + TL.AFT_BEATLES + 0.12 * k, 0.36)
    phrase(f + TL.AFT_BEATLES + 1.0, f + TL.AFT_GPT, 0.75, 0.3, lo=65, hi=84, prob=0.9)
    A.place(fx, A.boom(0.8), f + TL.AFT_GPT)
    for k, m in enumerate((38, 50, 57, 62, 65, 69)):
        note(m, f + TL.AFT_GPT + 0.08 * k, 0.4)
    for k in range(8):
        tb = f + TL.AFT_TOWER + 0.6 + 4.0 * (1 - (1 - (k + 1) / 9) ** (1 / 3))
        A.place(fx, A.soft_kick(0.2 + 0.02 * k), tb)
        A.place(tick, A.blip(int(72 + 2 * k), 0.2), tb, pan=-0.2 + 0.04 * k)
    A.place(bel, A.bell(86, 0.5), f + TL.AFT_TOWER + 4.6, pan=0.3)

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
    return dict(pad=pad, sub=sub, pno=pno, bel=bel, fx=fx, tick=tick, plk=plk, shim=shim)


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
    dry = pad + sub + pno + A.pingpong(pno) * 0.18 + bel + fx + tick + plk + shim
    send = pad * 0.3 + pno * 0.5 + bel * 0.8 + tick * 0.6 + plk * 0.6 + shim * 0.6 + fx * 0.25
    mix = A.hp(dry + A.reverb(send, ir) * 0.55, 28)
    arc = A.automation([(0, -10), (P_ + 9.6, -6), (P_ + 13.8, -5), (P_ + 15.4, 0), (P_ + 20.5, -2), (SQ, -5),
                        (SQ + 12, -4), (SQ + 22, -3), (AO, -4), (AO + 1.0, -1), (AO + 10, -4), (AO + 18, -1),
                        (QK, -4), (QK + TL.QKV_MATCH, -2), (QK + TL.QKV_FORMULA, 0.5), (HD, -4), (HD + 15, -1),
                        (OR, -3), (OR + 12, -4), (AF, -4), (AF + TL.AFT_GPT, 0), (EN, -6), (EN + TL.END_LINE, -1),
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
