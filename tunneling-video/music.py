"""Original score for 量子隧穿, synthesised from scratch (no samples).

F major, 120 BPM (bar = 2 s), chord every 2 bars.  A minimalist marimba
ostinato whose 12-step figure rotates against the 8-step bar, plucked bass,
shaker, soft pad and felt piano; paper-swish risers on the wipes, bell
accents on the examples, a hit on the "impossible" strike-through.

  python3 music.py   -> build/bgm.wav
"""

import functools
import re
import subprocess

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import maximum_filter1d

from style import ROOT
from timeline import DURATION, START, STRIKE_HIT, TILE_HITS, TITLE_HIT, TUNNEL_HIT, WIPES

SR = 48000
N = int(DURATION * SR)
SLOT = 4.0
BEAT = 0.5
rng = np.random.default_rng(1928)

CHORDS = {  # bass, voicing
    "Fmaj9": (41, [53, 57, 60, 64, 67]),
    "Am7": (45, [52, 57, 60, 64, 67]),
    "Bbmaj9": (46, [50, 53, 57, 60, 62]),
    "C69": (36, [52, 55, 57, 62, 64]),
    "Dm9": (38, [50, 53, 57, 60, 64]),
    "Gm9": (43, [50, 53, 58, 62, 69]),
    "A7sus4": (45, [50, 55, 57, 62, 64]),
}
PROG = (["Fmaj9", "Am7", "Bbmaj9"] +                        # opening 0-12
        ["C69", "Fmaj9", "Am7", "Bbmaj9"] +                 # p1 12-28
        ["C69", "Dm9", "Bbmaj9", "C69"] +                   # p2 28-44
        ["Fmaj9", "Am7", "Bbmaj9", "C69", "Fmaj9"] +        # p3 44-64
        ["Dm9", "Bbmaj9", "Gm9", "A7sus4", "Dm9"] +         # p4 64-84
        ["Bbmaj9", "C69", "Am7", "Dm9", "Bbmaj9", "C69", "Fmaj9"] +  # p5 84-112
        ["Dm9", "Bbmaj9", "Fmaj9"])                         # outro 112-124
assert len(PROG) * SLOT == DURATION


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def stereo(n):
    return np.zeros((n, 2))


def place(bus, x, t0, gain=1.0, pan=0.0):
    i0 = int(round(t0 * SR))
    if i0 >= len(bus):
        return
    if x.ndim == 1:
        x = np.stack([x * np.sqrt((1 - pan) / 2), x * np.sqrt((1 + pan) / 2)], 1)
    j0 = max(0, -i0)
    i0 = max(0, i0)
    n = min(len(x) - j0, len(bus) - i0)
    bus[i0:i0 + n] += x[j0:j0 + n] * gain


def lp(x, fc, order=2):
    b, a = signal.butter(order, fc / (SR / 2), "low")
    return signal.lfilter(b, a, x, axis=0)


def hp(x, fc, order=2):
    b, a = signal.butter(order, fc / (SR / 2), "high")
    return signal.lfilter(b, a, x, axis=0)


def bp(x, f1, f2, order=2):
    b, a = signal.butter(order, [f1 / (SR / 2), f2 / (SR / 2)], "band")
    return signal.lfilter(b, a, x, axis=0)


def rc(n):
    return 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))


def automation(points):
    ts, gs = zip(*points)
    return np.interp(np.arange(N) / SR, ts, gs)[:, None]


def chord_at(t):
    return PROG[min(int(t // SLOT), len(PROG) - 1)]


def tones(name, lo, hi):
    pcs = {p % 12 for p in CHORDS[name][1]} | {CHORDS[name][0] % 12}
    return [m for m in range(lo, hi + 1) if m % 12 in pcs]


# ------------------------------------------------------------- voices ---

@functools.lru_cache(maxsize=None)
def marimba(m, vel):
    f = mtof(m)
    dur = 1.8
    t = np.arange(int(dur * SR)) / SR
    tau = 0.55 * (440 / f) ** 0.35
    v = (np.sin(2 * np.pi * f * t) * np.exp(-t / tau)
         + 0.32 * np.sin(2 * np.pi * 3.93 * f * t + 0.4) * np.exp(-t / (tau * 0.22))
         + 0.1 * np.sin(2 * np.pi * 9.2 * f * t + 1.1) * np.exp(-t / (tau * 0.07)))
    click = bp(np.random.default_rng(m).standard_normal(int(0.006 * SR)), 1800, 5200) * 0.25
    v[:len(click)] += click * np.linspace(1, 0, len(click))
    na = int(0.0015 * SR)
    v[:na] *= rc(na)
    v[-int(0.1 * SR):] *= np.linspace(1, 0, int(0.1 * SR))
    return v * vel


@functools.lru_cache(maxsize=None)
def pluck_bass(m, vel):
    f = mtof(m)
    t = np.arange(int(1.4 * SR)) / SR
    v = np.zeros_like(t)
    for h in range(1, 7):
        v += (1 / h) * np.sin(2 * np.pi * f * h * t) * np.exp(-t / (0.9 / h ** 1.3))
    na = int(0.003 * SR)
    v[:na] *= rc(na)
    v[-int(0.15 * SR):] *= np.linspace(1, 0, int(0.15 * SR))
    return lp(v, 1400) * vel


@functools.lru_cache(maxsize=None)
def piano(m, vel):
    f = mtof(m)
    t = np.arange(int(5.0 * SR)) / SR
    bright = 0.35 + 0.65 * vel
    v = np.zeros_like(t)
    r = np.random.default_rng(m * 7)
    for k in range(1, 14):
        fk = f * k * np.sqrt(1 + 0.00032 * k * k)
        if fk > 11000:
            break
        amp = k ** -1.05 * np.exp(-(k - 1) * (1.2 - 0.65 * bright))
        tau = 3.0 * (220 / f) ** 0.4 / (1 + 0.5 * (k - 1))
        env = 0.6 * np.exp(-t / (tau * 0.22)) + 0.4 * np.exp(-t / tau)
        ph = r.uniform(0, 6.28)
        v += amp * env * (np.sin(2 * np.pi * fk * t + ph) + 0.55 * np.sin(2 * np.pi * fk * 1.0006 * t + ph + 1.3))
    na = int(0.004 * SR)
    v[:na] *= rc(na)
    v[-int(0.4 * SR):] *= np.linspace(1, 0, int(0.4 * SR))
    return v * vel * 0.3


def bell(m, vel, ratio=3.5, dur=5.0):
    f = mtof(m)
    t = np.arange(int(dur * SR)) / SR
    idx = 1.5 * np.exp(-t / 0.6) + 0.2
    v = np.sin(2 * np.pi * f * t + idx * np.sin(2 * np.pi * f * ratio * t))
    env = np.exp(-t / 1.8)
    na = int(0.006 * SR)
    env[:na] *= rc(na)
    env[-int(0.3 * SR):] *= np.linspace(1, 0, int(0.3 * SR))
    return v * env * vel * 0.2


def pad_note(m, hold, attack=1.4, release=2.2):
    f = mtof(m)
    n = int((hold + release) * SR)
    t = np.arange(n) / SR
    out = stereo(n)
    H = max(1, min(7, int(5000 / f)))
    for d, pan in zip((-6.0, 6.0), (-0.6, 0.6)):
        fd = f * 2 ** (d / 1200)
        v = np.zeros(n)
        for h in range(1, H + 1):
            v += (1.0 / h) * np.exp(-(h - 1) / 1.6) * np.sin(2 * np.pi * fd * h * t + rng.uniform(0, 6.28))
        out[:, 0] += v * np.sqrt((1 - pan) / 2)
        out[:, 1] += v * np.sqrt((1 + pan) / 2)
    env = np.ones(n)
    na = int(attack * SR)
    env[:na] = rc(na)
    nh = int(hold * SR)
    env[nh:] = np.exp(-(t[nh:] - hold) / (release / 4.5))
    env[-int(0.05 * SR):] *= np.linspace(1, 0, int(0.05 * SR))
    return out * env[:, None] / 2


@functools.lru_cache(maxsize=None)
def _shaker_base():
    n = int(0.05 * SR)
    t = np.arange(n) / SR
    v = bp(np.random.default_rng(3).standard_normal(n), 5000, 11000)
    return v * (1 - np.exp(-t / 0.003)) * np.exp(-t / 0.018)


def soft_kick(vel):
    t = np.arange(int(0.5 * SR)) / SR
    fr = 48 + 55 * np.exp(-t / 0.03)
    return np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-t / 0.18) * (1 - np.exp(-t / 0.002)) * vel


def tick(vel):
    t = np.arange(int(0.03 * SR)) / SR
    return np.sin(2 * np.pi * 3100 * t) * np.exp(-t / 0.004) * vel


def boom(gain=1.0):
    t = np.arange(int(4.0 * SR)) / SR
    fr = 32 + 50 * np.exp(-t / 0.14)
    v = np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-t / 1.2) * (1 - np.exp(-t / 0.004))
    v += lp(rng.standard_normal(len(t)), 300) * np.exp(-t / 0.25) * 0.4
    v[-int(0.4 * SR):] *= np.linspace(1, 0, int(0.4 * SR))
    return v * gain


def swish(length=0.5):
    """Paper-swipe riser: noise sweeping up through four bands into the cut."""
    n = int(length * SR)
    t = np.arange(n) / SR
    z = rng.standard_normal((n, 2))
    out = np.zeros((n, 2))
    bands = [(300, 700), (700, 1500), (1500, 3200), (3200, 7000)]
    for i, (f1, f2) in enumerate(bands):
        c = (i + 0.5) / len(bands)
        env = np.exp(-((t / length - c) / 0.22) ** 2)
        out += bp(z, f1, f2) * env[:, None]
    out *= (t / length)[:, None] ** 1.5
    out[-int(0.004 * SR):] *= np.linspace(1, 0, int(0.004 * SR))[:, None]
    return out / (np.abs(out).max() + 1e-9)


# ------------------------------------------------------------- reverb ---

def build_ir(sec=3.0):
    n = int(sec * SR)
    t = np.arange(n) / SR
    ir = stereo(n)
    for ch in range(2):
        z = rng.standard_normal(n)
        ir[:, ch] = (lp(z, 600) * np.exp(-6.9 * t / 2.6) + bp(z, 600, 4000) * np.exp(-6.9 * t / 2.0)
                     + hp(z, 4000) * np.exp(-6.9 * t / 1.0))
    ir[:int(0.01 * SR)] *= rc(int(0.01 * SR))[:, None]
    ir = np.vstack([np.zeros((int(0.018 * SR), 2)), ir])
    return ir / np.sqrt((ir ** 2).sum(0, keepdims=True))


def reverb(x, ir):
    return np.stack([signal.oaconvolve(x[:, c], ir[:, c])[:N] for c in range(2)], 1)


def pingpong(x, d, fb=0.35, n=5):
    out = np.zeros_like(x)
    D = int(d * SR)
    y = x.mean(1)
    for k in range(1, n + 1):
        y = lp(y, 3500, 1)
        if k * D >= len(x):
            break
        out[k * D:, k % 2] += fb ** k * y[:len(y) - k * D]
    return out


# -------------------------------------------------------------- score ---

def section(t):
    for name in ("outro", "p5", "p4", "p3", "p2", "p1", "opening"):
        if t >= START[name]:
            return name
    return "opening"


def score():
    pad, mar, mar2, bass, pno, bel, shk, kick, tk, fx = (stereo(N) for _ in range(10))

    for i, name in enumerate(PROG):
        t0 = i * SLOT
        hold = SLOT + 0.3 if i < len(PROG) - 1 else DURATION - t0
        for m in CHORDS[name][1]:
            place(pad, pad_note(m, hold), t0 + rng.uniform(0, 0.12))

    # marimba ostinato: 12-step figure over an 8-step bar -> the accent pattern rotates
    figure = [0, 2, 4, 1, 3, 5, 2, 4, 1, 3, 0, 5]
    step = 0
    t = 0.0
    while t < DURATION - 0.3:
        sec = section(t)
        on = sec in ("p1", "p2", "p3", "p5") or (sec == "opening" and t >= TITLE_HIT) or (sec == "p4" and t < 76)
        if sec == "outro" and t < STRIKE_HIT:
            on = True
        if on:
            tn = tones(chord_at(t), 62, 81)[:6]
            m = tn[figure[step % 12] % len(tn)]
            vel = 0.55 if step % 3 == 0 else 0.36
            if sec in ("opening", "p1"):
                vel *= 0.75
            place(mar, marimba(m, round(vel, 2)), t + rng.normal(0, 0.004), pan=((step % 4) - 1.5) * 0.18)
            if sec in ("p3", "p5") and step % 8 in (3, 6):
                place(mar2, marimba(m + 12, 0.3), t, pan=0.5 if step % 2 else -0.5)
        step += 1
        t += 0.25

    # plucked bass: beat 1 and the "and" of 2
    for bar in np.arange(START["p1"], START["outro"], 2.0):
        sec = section(bar)
        if sec == "p4" and bar >= 76:
            continue
        b = CHORDS[chord_at(bar)][0]
        b = b + 12 if b < 40 else b
        place(bass, pluck_bass(b, 0.8), bar)
        if sec in ("p2", "p3", "p5"):
            place(bass, pluck_bass(b + 7 if (bar // 2) % 2 else b, 0.55), bar + 0.75)

    # shaker 16ths, soft kick on 1 & 3, ticking clock under part 4
    for s16 in np.arange(START["p2"], START["outro"], 0.125):
        sec = section(s16)
        if sec in ("p2", "p3", "p5"):
            acc = 1.0 if (s16 * 8) % 2 == 1 else 0.55
            place(shk, _shaker_base() * acc * rng.uniform(0.8, 1.1), s16 + rng.normal(0, 0.003),
                  pan=rng.uniform(-0.3, 0.3))
    for b in np.arange(START["p3"], START["outro"], 1.0):
        sec = section(b)
        if sec in ("p3", "p5"):
            place(kick, soft_kick(0.9 if b % 2 == 0 else 0.55), b)
    for b in np.arange(START["p4"], START["p5"], 0.5):
        place(tk, tick(0.6 if b % 1 == 0 else 0.35), b, pan=0.3 if b % 1 else -0.3)

    # felt piano: long sparse notes
    def phrase(t0, t1, every, vel):
        prev = 72
        for tt in np.arange(t0, t1, every):
            cand = tones(chord_at(tt), 65, 84)
            target = prev + rng.choice([-4, -2, 2, 3, 5])
            m = min(cand, key=lambda c: abs(c - target) + (3 if c == prev else 0))
            prev = m
            place(pno, piano(m, vel), tt, pan=np.clip((m - 72) / 18, -0.5, 0.5))
    phrase(START["p1"] + 0.0, START["p2"], 2.0, 0.45)
    phrase(START["p4"] + 0.0, START["p5"], 4.0, 0.5)
    phrase(START["p5"] + 0.0, START["outro"], 2.0, 0.42)
    phrase(START["outro"], STRIKE_HIT, 2.0, 0.4)

    # accents
    place(fx, boom(0.7), TITLE_HIT)
    for k, m in enumerate((53, 60, 64, 69, 72, 76)):
        place(pno, piano(m, 0.45), TITLE_HIT + 0.05 * k)
    place(bel, bell(84, 0.7), TUNNEL_HIT, pan=0.4)
    place(bel, bell(79, 0.5), TUNNEL_HIT + 0.25, pan=-0.4)
    for i, th in enumerate(TILE_HITS):
        top = tones(chord_at(th), 79, 91)
        place(bel, bell(top[(i * 2) % len(top)], 0.6), th, pan=(-0.5, 0.5, -0.3, 0.3)[i])
    for w in WIPES:
        place(fx, swish(0.45) * 0.5, w - 0.45)
        place(kick, soft_kick(0.7), w)
    place(fx, boom(0.9), STRIKE_HIT)
    for k, m in enumerate((46, 53, 57, 62, 65, 69)):
        place(pno, piano(m, 0.5), STRIKE_HIT + 0.04 * k)
    place(bel, bell(81, 0.6, ratio=2.0), STRIKE_HIT + 0.05, pan=0.3)
    end = START["outro"] + 8.0
    for k, m in enumerate((41, 53, 60, 64, 67, 72, 76)):
        place(pno, piano(m, 0.42), end + 0.06 * k)
    place(bel, bell(84, 0.45), end + 0.5, pan=-0.3)

    return dict(pad=pad, mar=mar, mar2=mar2, bass=bass, pno=pno, bel=bel, shk=shk, kick=kick, tk=tk, fx=fx)


def loudness(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af", "ebur128=peak=true",
                        "-f", "null", "-"], capture_output=True, text=True)
    return (float(re.findall(r"I:\s+(-?[\d.]+) LUFS", r.stderr)[-1]),
            float(re.findall(r"Peak:\s+(-?[\d.]+) dBFS", r.stderr)[-1]))


def main():
    out_dir = ROOT / "build"
    out_dir.mkdir(exist_ok=True)
    ir = build_ir()
    S = score()

    def norm(x, rms):
        return x * (rms / (np.sqrt((x ** 2).mean()) + 1e-12))

    pad = hp(lp(norm(S["pad"], 0.035), 2600), 120) * automation(
        [(0, 0), (3, 1), (60, 1), (64, 1.2), (84, 1.1), (112, 1.0), (DURATION, 1.1)])
    mar = norm(S["mar"], 0.05)
    mar2 = norm(S["mar2"], 0.012)
    bass = norm(S["bass"], 0.035)
    pno = norm(S["pno"], 0.03)
    bel = S["bel"] / (np.abs(S["bel"]).max() + 1e-9) * 0.25
    shk = norm(S["shk"], 0.006)
    kick = S["kick"] / (np.abs(S["kick"]).max() + 1e-9) * 0.25
    tk = norm(S["tk"], 0.004)
    fx = S["fx"] / (np.abs(S["fx"]).max() + 1e-9) * 0.35

    dry = (pad + mar + pingpong(mar, 0.375) * 0.18 + mar2 + pingpong(mar2, 0.375, 0.45) * 0.3 + bass + pno
           + bel + shk + kick + tk + fx)
    send = pad * 0.25 + mar * 0.25 + mar2 * 0.4 + pno * 0.45 + bel * 0.7 + fx * 0.2
    mix = hp(dry + reverb(send, ir) * 0.5, 30)

    arc = automation([(0, -8), (1.8, -5), (2.2, -1), (11, -2), (12.2, -3.5), (27, -3), (28.2, -2), (43, -1.5),
                      (44.2, 0), (63, 0), (64.2, -3), (83, -2.5), (84.2, 0.5), (111, 0.5), (112.2, -4),
                      (115.8, -4), (116.2, 0), (DURATION, 0)])
    mix *= 10 ** (arc / 20)
    env = np.sqrt(lp(np.mean(mix ** 2, 1), 3.0, 1).clip(1e-12))
    thr = np.percentile(env, 90)
    mix *= np.where(env > thr, (env / thr) ** (1 / 1.8 - 1), 1.0)[:, None]
    fi = int(0.03 * SR)
    mix[:fi] *= rc(fi)[:, None]
    fo = int(3.0 * SR)
    mix[-fo:] *= (1 - rc(fo))[:, None] ** 1.5

    tmp = out_dir / "bgm_pre.wav"
    sf.write(tmp, (mix / np.abs(mix).max() * 0.5).astype(np.float32), SR, subtype="FLOAT")
    lufs, _ = loudness(tmp)
    mix = mix / np.abs(mix).max() * 0.5 * 10 ** ((-16.0 - lufs) / 20)
    ceiling = 10 ** (-1.5 / 20)
    pk = maximum_filter1d(np.maximum(np.abs(mix[:, 0]), np.abs(mix[:, 1])), int(0.01 * SR))
    gain = np.minimum(1.0, ceiling / np.maximum(pk, 1e-9))
    gain = lp(-maximum_filter1d(-gain, int(0.02 * SR)), 30, 1)
    mix = np.clip(mix * gain[:, None], -0.98, 0.98)
    out = out_dir / "bgm.wav"
    sf.write(out, mix.astype(np.float32), SR, subtype="PCM_24")
    tmp.unlink()
    lufs, peak = loudness(out)
    print(f"bgm: {out}  {DURATION}s  {lufs:.1f} LUFS  peak {peak:.1f} dBFS")


if __name__ == "__main__":
    main()
