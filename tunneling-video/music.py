"""Original score for 量子隧穿 (cinematic edition), synthesised from scratch.

A dark ambient bed (warm pads, sub drone, sparse felt piano, glass bells)
with sound design locked to the picture: reel ticks, impacts, risers into
the title and the morph, detection chimes, a falling pulse per thickness
step, swells under the Sun and the sparks.  Sync points: timeline.py.

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
from timeline import (BALL_HIT, BURST, CALLBACK, COLLIDE, DETECT, DURATION, EARTH, EMERGE, FILL, FINAL, FORMULA,
                      IGNITE, IMPOSSIBLE, LAYERS, MORPH, NM1, NM2, PEAK, SHOTS, START, THROUGH_SHOT, TITLE, ZEROS)

SR = 48000
N = int(DURATION * SR)
rng = np.random.default_rng(1928)

CHORDS = {  # bass, voicing
    "Dadd9": (38, [50, 57, 62, 64, 69]),
    "Dm9": (38, [50, 57, 60, 64, 65]),
    "Bbmaj9": (34, [46, 53, 57, 60, 62]),
    "Bbmaj7#11": (34, [46, 50, 57, 64, 65]),
    "Gm9": (43, [50, 55, 58, 65, 69]),
    "Fmaj9": (41, [53, 57, 60, 64, 67]),
    "Am7": (45, [52, 57, 60, 64, 67]),
    "C69": (36, [48, 55, 62, 64, 69]),
    "A7sus4": (45, [52, 57, 62, 67, 71]),
    "Dsus2": (38, [50, 57, 62, 64, 69]),
}
Wl, Wv, Sp, Zr, Su, Ep = (START[k] for k in ("wall", "wave", "seep", "zeros", "sun", "epilogue"))
PLAN = [(0.0, "Dadd9"), (10.0, "Bbmaj7#11"), (TITLE, "Dm9"),
        (Wl, "Dm9"), (Wl + 6, "Gm9"), (IMPOSSIBLE, "Dm9"),
        (Wv, "Fmaj9"), (Wv + 6, "Am7"), (Wv + 12, "Bbmaj9"),
        (Sp, "Dm9"), (Sp + 6, "Bbmaj9"), (Sp + 12, "Gm9"), (Sp + 16, "Dm9"), (NM1, "Bbmaj7#11"),
        (NM2, "A7sus4"), (FORMULA, "Gm9"), (MORPH[0] - 2, "A7sus4"), (MORPH[1], "Fmaj9"),
        (Zr, "Dm9"), (Zr + 6, "Bbmaj7#11"), (Zr + 11, "Gm9"),
        (Su, "Dadd9"),
        (IGNITE, "Fmaj9"), (Su + 14.4, "C69"), (Su + 19.4, "Am7"), (Su + 24.6, "Bbmaj9"), (EARTH, "Fmaj9"),
        (PEAK, "C69"), (Su + 38.5, "Bbmaj9"),
        (Ep, "Dm9"), (Ep + 5.5, "Bbmaj9"), (FINAL, "Dsus2")]


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
    name = PLAN[0][1]
    for ts, nm in PLAN:
        if t >= ts:
            name = nm
    return name


def tones(name, lo, hi):
    pcs = {p % 12 for p in CHORDS[name][1]}
    return [m for m in range(lo, hi + 1) if m % 12 in pcs]


# ------------------------------------------------------------- voices ---

def pad_note(m, hold, attack=2.4, release=3.2):
    f = mtof(m)
    n = int((hold + release) * SR)
    t = np.arange(n) / SR
    out = stereo(n)
    Hh = max(1, min(8, int(6000 / f)))
    for d, pan in zip((-7.0, -2.0, 2.0, 7.0), (-0.75, -0.25, 0.25, 0.75)):
        fd = f * 2 ** (d / 1200)
        v = np.zeros(n)
        for h in range(1, Hh + 1):
            v += (1.0 / h) * np.exp(-(h - 1) / 1.7) * np.sin(2 * np.pi * fd * h * t + rng.uniform(0, 6.28))
        v *= 1 + 0.15 * np.sin(2 * np.pi * rng.uniform(0.05, 0.15) * t + rng.uniform(0, 6.28))
        out[:, 0] += v * np.sqrt((1 - pan) / 2)
        out[:, 1] += v * np.sqrt((1 + pan) / 2)
    env = np.ones(n)
    na = min(int(attack * SR), n)
    env[:na] = rc(na)
    nh = int(hold * SR)
    env[nh:] = np.exp(-(t[nh:] - hold) / (release / 4.5))
    env[-int(0.05 * SR):] *= np.linspace(1, 0, int(0.05 * SR))
    return out * env[:, None] / 4


def sub_note(m, hold, release=2.6):
    f = mtof(m)
    n = int((hold + release) * SR)
    t = np.arange(n) / SR
    v = np.sin(2 * np.pi * f * t) + 0.2 * np.sin(4 * np.pi * f * t)
    env = np.ones(n)
    na = min(int(1.6 * SR), n)
    env[:na] = rc(na)
    nh = int(hold * SR)
    env[nh:] = np.exp(-(t[nh:] - hold) / (release / 4))
    return v * env


@functools.lru_cache(maxsize=None)
def piano(m, vel):
    f = mtof(m)
    t = np.arange(int(6.0 * SR)) / SR
    bright = 0.35 + 0.65 * vel
    v = np.zeros_like(t)
    r = np.random.default_rng(m * 13)
    for k in range(1, 14):
        fk = f * k * np.sqrt(1 + 0.00032 * k * k)
        if fk > 11000:
            break
        amp = k ** -1.05 * np.exp(-(k - 1) * (1.2 - 0.65 * bright))
        tau = 3.4 * (220 / f) ** 0.4 / (1 + 0.5 * (k - 1))
        env = 0.6 * np.exp(-t / (tau * 0.22)) + 0.4 * np.exp(-t / tau)
        ph = r.uniform(0, 6.28)
        v += amp * env * (np.sin(2 * np.pi * fk * t + ph) + 0.55 * np.sin(2 * np.pi * fk * 1.0006 * t + ph + 1.3))
    na = int(0.004 * SR)
    v[:na] *= rc(na)
    v[-int(0.4 * SR):] *= np.linspace(1, 0, int(0.4 * SR))
    return v * vel * 0.3


@functools.lru_cache(maxsize=None)
def bell(m, vel, ratio=3.5, dur=5.0):
    f = mtof(m)
    t = np.arange(int(dur * SR)) / SR
    idx = 1.5 * np.exp(-t / 0.6) + 0.2
    v = np.sin(2 * np.pi * f * t + idx * np.sin(2 * np.pi * f * ratio * t))
    env = np.exp(-t / 1.8)
    na = int(0.005 * SR)
    env[:na] *= rc(na)
    env[-int(0.3 * SR):] *= np.linspace(1, 0, int(0.3 * SR))
    return v * env * vel * 0.2


@functools.lru_cache(maxsize=None)
def blip(m, vel):
    """Tiny glassy tick for reels and detections."""
    f = mtof(m)
    t = np.arange(int(0.35 * SR)) / SR
    v = np.sin(2 * np.pi * f * t + 0.8 * np.sin(2 * np.pi * f * 2.0 * t) * np.exp(-t / 0.05))
    v *= np.exp(-t / 0.07) * (1 - np.exp(-t / 0.001))
    return v * vel


def boom(gain=1.0, dur=5.0):
    t = np.arange(int(dur * SR)) / SR
    fr = 30 + 55 * np.exp(-t / 0.16)
    v = np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-t / 1.6) * (1 - np.exp(-t / 0.004))
    v = v + lp(rng.standard_normal(len(t)), 260) * np.exp(-t / 0.3) * 0.45
    v[-int(0.5 * SR):] *= np.linspace(1, 0, int(0.5 * SR))
    return v * gain


def thud(gain=1.0):
    t = np.arange(int(0.6 * SR)) / SR
    fr = 60 + 90 * np.exp(-t / 0.02)
    v = np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-t / 0.12) * (1 - np.exp(-t / 0.002))
    return v * gain


def riser(length, gain=1.0):
    n = int(length * SR)
    t = np.arange(n) / SR
    z = rng.standard_normal((n, 2))
    out = np.zeros((n, 2))
    bands = [(200, 500), (500, 1200), (1200, 3000), (3000, 7500)]
    for i, (f1, f2) in enumerate(bands):
        c = (i + 0.6) / len(bands)
        out += bp(z, f1, f2) * np.exp(-((t / length - c) / 0.28) ** 2)[:, None]
    out *= ((t / length) ** 2)[:, None]
    out[-int(0.01 * SR):] *= np.linspace(1, 0, int(0.01 * SR))[:, None]
    return out / (np.abs(out).max() + 1e-9) * gain


def rumble(length, gain=1.0):
    n = int(length * SR)
    t = np.arange(n) / SR
    v = lp(rng.standard_normal(n), 120, 4)
    v = v / (np.abs(v).max() + 1e-9)
    env = np.sin(np.pi * np.clip(t / length, 0, 1)) ** 2
    return v * env * gain


def soft_kick(vel):
    t = np.arange(int(0.6 * SR)) / SR
    fr = 44 + 50 * np.exp(-t / 0.035)
    return np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-t / 0.22) * (1 - np.exp(-t / 0.002)) * vel


# -------------------------------------------------------------- reverb ---

def build_ir(sec=5.0):
    n = int(sec * SR)
    t = np.arange(n) / SR
    ir = stereo(n)
    for ch in range(2):
        z = rng.standard_normal(n)
        ir[:, ch] = (lp(z, 500) * np.exp(-6.9 * t / 4.2) + bp(z, 500, 4000) * np.exp(-6.9 * t / 3.3)
                     + hp(z, 4000) * np.exp(-6.9 * t / 1.6))
    ir[:int(0.012 * SR)] *= rc(int(0.012 * SR))[:, None]
    ir = np.vstack([np.zeros((int(0.024 * SR), 2)), ir])
    return ir / np.sqrt((ir ** 2).sum(0, keepdims=True))


def reverb(x, ir):
    return np.stack([signal.oaconvolve(x[:, c], ir[:, c])[:N] for c in range(2)], 1)


def pingpong(x, d=0.75, fb=0.35, n=6):
    out = np.zeros_like(x)
    D = int(d * SR)
    y = x.mean(1)
    for k in range(1, n + 1):
        y = lp(y, 3200, 1)
        if k * D >= len(x):
            break
        out[k * D:, k % 2] += fb ** k * y[:len(y) - k * D]
    return out


# --------------------------------------------------------------- score ---

def score():
    pad, sub, pno, bel, fx, tick, shim, kick = (stereo(N) for _ in range(8))

    for i, (t0, name) in enumerate(PLAN):
        t1 = PLAN[i + 1][0] if i + 1 < len(PLAN) else DURATION
        hold = max(1.0, t1 - t0 + 0.3)
        bass, voicing = CHORDS[name]
        for m in voicing:
            place(pad, pad_note(m, hold), t0 + rng.uniform(0, 0.15))
        place(sub, sub_note(bass, hold), t0)
        for m in voicing[2:5]:
            n = int((hold + 2) * SR)
            tt = np.arange(n) / SR
            sw = np.sin(np.pi * np.clip(tt / (hold + 2), 0, 1)) ** 2
            lfo = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.07, 0.2) * tt + rng.uniform(0, 6.28))
            place(shim, np.sin(2 * np.pi * mtof(m + 24) * tt) * sw * lfo * 0.08, t0, pan=rng.uniform(-0.8, 0.8))

    def note(m, t, vel, pan=None):
        if 0 <= t < DURATION - 0.5:
            place(pno, piano(m, round(vel, 2)), t + rng.normal(0, 0.006),
                  pan=np.clip((m - 68) / 20, -0.6, 0.6) if pan is None else pan)

    def phrase(t0, t1, every, vel, lo=64, hi=81, prob=0.85):
        prev = 69
        for tt in np.arange(t0, t1, every):
            if rng.random() > prob:
                continue
            cand = tones(chord_at(tt), lo, hi)
            target = prev + rng.choice([-5, -3, -2, 2, 3, 4])
            m = min(cand, key=lambda c: abs(c - target) + (3 if c == prev else 0))
            prev = m
            note(m, tt, vel * rng.uniform(0.85, 1.05))

    # --- prologue: the idiom, the ball, the electrons, the title
    note(69, 1.2, 0.3)
    note(74, FILL, 0.45)
    place(bel, bell(81, 0.3), FILL + 0.05, pan=0.2)
    place(fx, thud(0.8), BALL_HIT)
    place(bel, bell(76, 0.3, ratio=2.0), BALL_HIT + 0.02, pan=0.3)
    note(62, 10.0, 0.35)
    for i, ts in enumerate(SHOTS):
        hit = ts + 1.0
        if i == THROUGH_SHOT:
            for kk, m in enumerate((81, 86, 88, 93)):
                place(bel, bell(m, 0.5), hit + 0.07 * kk, pan=0.5)
        else:
            place(bel, bell(74, 0.4, ratio=2.0), hit, pan=0.35)
            place(fx, thud(0.35), hit)
    for kk, m in enumerate((62, 69, 74)):
        note(m, 17.2 + 0.4 * kk, 0.38)
    place(fx, riser(BURST - 18.6, 0.6), 18.6)
    place(fx, boom(1.0), BURST)
    for kk, m in enumerate((50, 57, 62, 65, 69, 74)):
        note(m, TITLE + 0.07 * kk, 0.42)
    place(bel, bell(81, 0.6), TITLE + 0.5, pan=-0.3)
    for c in (Wl, Wv, Sp, Zr, Ep):
        place(fx, boom(0.35), c)

    # --- wall
    phrase(Wl + 1.0, IMPOSSIBLE - 1, 2.0, 0.42)
    place(fx, boom(0.8), IMPOSSIBLE)
    for kk, m in enumerate((38, 50, 57, 60, 65)):
        note(m, IMPOSSIBLE + 0.06 * kk, 0.45)

    # --- wave: slow arpeggio of wonder, chimes on detections
    for tt in np.arange(Wv + 2.0, Wv + 17.5, 0.5):
        tn = tones(chord_at(tt), 62, 79)[:6]
        idx = [0, 2, 4, 1, 3, 5, 2, 4][int(round((tt - Wv) * 2)) % 8]
        note(tn[idx % len(tn)], tt, 0.22 if int(tt * 2) % 2 else 0.28)
    r2 = np.random.default_rng(12)
    r2.normal(0, 1, (40, 2))
    for td in np.sort(r2.uniform(6.0, 11.0, 40)):
        place(tick, blip(int(rng.choice([86, 88, 91, 93, 95])), 0.45), Wv + td, pan=rng.uniform(-0.6, 0.6))

    # --- seep: collision, emergence, layers (each fainter, lower), formula, morph
    place(fx, rumble(4.0, 0.9), COLLIDE - 1.5)
    place(fx, thud(0.6), COLLIDE)
    for kk, m in enumerate((76, 81, 83, 88)):
        place(bel, bell(m, 0.4), EMERGE + 0.18 * kk, pan=0.6)
    phrase(Sp + 8.5, Sp + 16, 2.0, 0.34)
    for k, tk in enumerate(LAYERS):
        place(tick, blip(96 - k, 0.75 * (1 - k / 24)), tk, pan=-0.5 + k / 20)
        place(kick, soft_kick(0.25 * (1 - k / 30)), tk)
    place(bel, bell(79, 0.5), NM1, pan=0.2)
    place(bel, bell(67, 0.5), NM2, pan=-0.2)
    place(fx, boom(0.85), FORMULA)
    for kk, m in enumerate((43, 55, 62, 65, 70, 74)):
        note(m, FORMULA + 0.07 * kk, 0.42)
    place(fx, riser(MORPH[1] - MORPH[0] + 0.4, 0.6), MORPH[0] - 0.4)
    place(fx, boom(0.85), MORPH[1])
    for kk, m in enumerate((41, 53, 60, 64, 67, 72, 76)):
        note(m, MORPH[1] + 0.07 * kk, 0.45)
    place(bel, bell(84, 0.55), MORPH[1] + 0.4, pan=0.3)

    # --- zeros: accelerating clicks that blur into a whirr
    z0 = ZEROS[0]
    u, last = 0.0, -1
    while u < 10.2:
        count = 3.0 * u + 26.0 * u ** 2 + 14.0 * u ** 3
        if int(count) // 3 != last:
            last = int(count) // 3
            place(tick, blip(int(rng.choice([91, 93, 96])), 0.22), z0 + u, pan=rng.uniform(-0.5, 0.5))
            u += 1 / 30
        else:
            u += 1 / 600
    n = int(10.4 * SR)
    tt = np.arange(n) / SR
    whirr = bp(rng.standard_normal((n, 2)), 2500, 7000) * ((tt / 10.4) ** 3 * (1 - smooth_np(tt, 9.6, 10.4)))[:, None]
    place(fx, whirr * 0.25, z0)
    phrase(Zr + 1.0, Zr + 15, 2.0, 0.3, lo=57, hi=72, prob=0.6)

    # --- sun: silence, a single low note, ignition, warmth, the Earth
    note(38, Su + 5.4, 0.4)
    note(50, Su + 5.45, 0.25)
    place(fx, riser(1.6, 0.35), IGNITE - 1.6)
    place(fx, boom(1.0), IGNITE)
    for kk, m in enumerate((41, 53, 57, 60, 64, 67, 72)):
        note(m, IGNITE + 0.09 * kk, 0.45)
    phrase(IGNITE + 2.0, EARTH, 2.0, 0.4, lo=65, hi=84)
    for kk, m in enumerate((53, 60, 64, 69, 72, 76)):
        note(m, EARTH + 0.12 * kk, 0.42)
    place(bel, bell(84, 0.5), EARTH + 0.6, pan=0.4)
    phrase(EARTH + 3.0, Ep - 0.5, 1.5, 0.4, lo=67, hi=86)
    place(bel, bell(88, 0.45), PEAK, pan=-0.4)

    # --- epilogue: the open blank, the last chord
    note(62, CALLBACK, 0.35)
    note(69, CALLBACK + 1.6, 0.3)
    for kk, m in enumerate((38, 50, 57, 62, 64, 69)):
        note(m, FINAL + 0.1 * kk, 0.38)
    place(bel, bell(81, 0.4), FINAL + 0.8, pan=-0.3)

    return dict(pad=pad, sub=sub, pno=pno, bel=bel, fx=fx, tick=tick, shim=shim, kick=kick)


def smooth_np(t, a, b):
    x = np.clip((t - a) / (b - a), 0, 1)
    return x * x * (3 - 2 * x)


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

    pad = hp(lp(norm(S["pad"], 0.06), 3000), 90)
    sub = norm(S["sub"], 0.03)
    pno = norm(S["pno"], 0.045)
    bel = S["bel"] / (np.abs(S["bel"]).max() + 1e-9) * 0.28
    fx = S["fx"] / (np.abs(S["fx"]).max() + 1e-9) * 0.45
    tick = S["tick"] / (np.abs(S["tick"]).max() + 1e-9) * 0.12
    shim = norm(S["shim"], 0.014)
    kick = S["kick"] / (np.abs(S["kick"]).max() + 1e-9) * 0.2

    dry = pad + sub + pno + pingpong(pno) * 0.18 + bel + fx + tick + shim + kick
    send = pad * 0.3 + pno * 0.5 + bel * 0.8 + tick * 0.6 + shim * 0.6 + fx * 0.25
    mix = hp(dry + reverb(send, ir) * 0.55, 28)

    arc = automation([(0, -12), (9.5, -9), (10.0, -7), (18.6, -5), (20.1, 0), (25, -2), (Wl, -4),
                      (IMPOSSIBLE, -1), (Wv, -4), (Wv + 17, -2.5), (Sp, -3), (COLLIDE, -1), (Sp + 16, -3),
                      (NM2, -0.5), (FORMULA, 0), (MORPH[1], 0.5), (Zr, -3), (Zr + 10.5, 0), (Zr + 15.5, -6),
                      (Su + 0.6, -24), (Su + 5.0, -24), (Su + 5.4, -12), (IGNITE - 0.2, -12), (IGNITE, 0.5),
                      (EARTH, 0), (PEAK, 1), (Ep - 1, -1), (Ep, -5), (FINAL, -2), (DURATION, -2)])
    mix *= 10 ** (arc / 20)
    env = np.sqrt(lp(np.mean(mix ** 2, 1), 3.0, 1).clip(1e-12))
    thr = np.percentile(env, 85)
    mix *= np.where(env > thr, (env / thr) ** (1 / 2.0 - 1), 1.0)[:, None]
    fi = int(0.05 * SR)
    mix[:fi] *= rc(fi)[:, None]
    fo = int(3.0 * SR)
    mix[-fo:] *= (1 - rc(fo))[:, None] ** 1.5

    tmp = out_dir / "bgm_pre.wav"
    sf.write(tmp, (mix / np.abs(mix).max() * 0.5).astype(np.float32), SR, subtype="FLOAT")
    lufs, _ = loudness(tmp)
    mix = mix / np.abs(mix).max() * 0.5 * 10 ** ((-14.5 - lufs) / 20)
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
