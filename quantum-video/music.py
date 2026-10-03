"""Original ambient score, synthesised from scratch (no samples).

D minor, 60 BPM, chord every 2 bars (8 s).  Layers: warm detuned pad,
sub, felt piano, FM glass bells, shimmer, air, soft pulse, cinematic hits,
reverse-reverb swells, through a synthetic stereo hall.
Sync points come from timeline.py.

  python3 music.py   -> build/bgm.wav
"""

import functools
import re
import subprocess

import numpy as np
import soundfile as sf
from scipy import signal

from engine import ROOT
from timeline import COLLAPSE_HIT, DURATION, FINAL_HIT, LIFT, MEASURE_HIT, TITLE_HIT

SR = 48000
N = int(DURATION * SR)
SLOT = 8.0
rng = np.random.default_rng(2026)

CHORDS = {  # bass, pad voicing (MIDI)
    "Dm9": (38, [50, 57, 60, 64, 65]),
    "Bbmaj9": (34, [46, 53, 57, 60, 62]),
    "Fmaj9": (41, [53, 57, 60, 64, 67]),
    "C69": (36, [48, 55, 62, 64, 69]),
    "Gm9": (43, [50, 55, 58, 65, 69]),
    "A9sus4": (45, [52, 57, 62, 67, 71]),
    "Dsus2": (38, [50, 57, 62, 64, 69]),
}
PROG = ["Dm9", "Bbmaj9", "Fmaj9", "C69"] * 2 + ["Dm9", "Bbmaj9", "Gm9", "A9sus4"] + \
       ["Dm9", "Bbmaj9", "Fmaj9", "C69"] + ["Dm9", "Bbmaj9", "Gm9", "A9sus4"] + \
       ["Bbmaj9", "Fmaj9", "C69", "Bbmaj9", "A9sus4", "Dsus2"]
assert len(PROG) * SLOT == DURATION


def mtof(m):
    return 440.0 * 2 ** ((m - 69) / 12)


def stereo(n):
    return np.zeros((n, 2), np.float64)


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


def automation(points):
    ts, gs = zip(*points)
    return np.interp(np.arange(N) / SR, ts, gs)[:, None]


def rc_ramp(n):
    return 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))


# ------------------------------------------------------------ voices ---

def pad_note(m, hold, attack=2.6, release=3.6):
    f = mtof(m)
    dur = hold + release
    n = int(dur * SR)
    t = np.arange(n) / SR
    out = stereo(n)
    H = max(1, min(9, int(6500 / f)))
    for d, pan in zip((-7.0, -2.5, 2.5, 7.0), (-0.75, -0.25, 0.25, 0.75)):
        fd = f * 2 ** (d / 1200)
        v = np.zeros(n)
        for h in range(1, H + 1):
            v += (1.0 / h) * np.exp(-(h - 1) / 1.9) * np.sin(2 * np.pi * fd * h * t + rng.uniform(0, 6.283))
        v *= 1 + 0.15 * np.sin(2 * np.pi * rng.uniform(0.05, 0.16) * t + rng.uniform(0, 6.283))
        out[:, 0] += v * np.sqrt((1 - pan) / 2)
        out[:, 1] += v * np.sqrt((1 + pan) / 2)
    env = np.ones(n)
    na = int(attack * SR)
    env[:na] = rc_ramp(na)
    nh = int(hold * SR)
    env[nh:] = np.exp(-(t[nh:] - hold) / (release / 4.5))
    env[-int(0.05 * SR):] *= np.linspace(1, 0, int(0.05 * SR))
    return out * env[:, None] / 4


def sub_note(m, hold, release=2.5):
    f = mtof(m)
    n = int((hold + release) * SR)
    t = np.arange(n) / SR
    v = np.sin(2 * np.pi * f * t) + 0.22 * np.sin(4 * np.pi * f * t)
    env = np.ones(n)
    na = int(1.4 * SR)
    env[:na] = rc_ramp(na)
    nh = int(hold * SR)
    env[nh:] = np.exp(-(t[nh:] - hold) / (release / 4))
    return v * env


def piano(m, vel, dur=6.0):
    return _piano(m, round(vel * 20) / 20, dur)


@functools.lru_cache(maxsize=None)
def _piano(m, vel, dur):
    f = mtof(m)
    n = int(dur * SR)
    t = np.arange(n) / SR
    B = 0.00032
    bright = 0.35 + 0.65 * vel
    v = np.zeros(n)
    for k in range(1, 16):
        fk = f * k * np.sqrt(1 + B * k * k)
        if fk > 11000:
            break
        amp = k ** -1.05 * np.exp(-(k - 1) * (1.2 - 0.65 * bright))
        tau = 3.4 * (220 / f) ** 0.4 / (1 + 0.5 * (k - 1))
        env = 0.6 * np.exp(-t / (tau * 0.22)) + 0.4 * np.exp(-t / tau)
        ph = rng.uniform(0, 6.283)
        v += amp * env * (np.sin(2 * np.pi * fk * t + ph) + 0.55 * np.sin(2 * np.pi * fk * 1.0006 * t + ph + 1.3))
    na = int(0.004 * SR)
    v[:na] *= rc_ramp(na)
    thump = lp(rng.standard_normal(int(0.05 * SR)), 900) * np.exp(-np.arange(int(0.05 * SR)) / (0.01 * SR))
    v[:len(thump)] += thump * 0.04 * vel
    v[-int(0.4 * SR):] *= np.linspace(1, 0, int(0.4 * SR))
    return v * vel * 0.32


def bell(m, vel, dur=7.0, ratio=3.5):
    f = mtof(m)
    n = int(dur * SR)
    t = np.arange(n) / SR
    idx = 1.6 * np.exp(-t / 0.7) + 0.2
    v = np.sin(2 * np.pi * f * t + idx * np.sin(2 * np.pi * f * ratio * t))
    v += 0.25 * np.sin(2 * np.pi * f * 2.0 * t) * np.exp(-t / 1.2)
    env = np.exp(-t / 2.4)
    na = int(0.006 * SR)
    env[:na] *= rc_ramp(na)
    env[-int(0.3 * SR):] *= np.linspace(1, 0, int(0.3 * SR))
    return v * env * vel * 0.18


def boom(gain=1.0, dur=6.0):
    n = int(dur * SR)
    t = np.arange(n) / SR
    fr = 29 + 58 * np.exp(-t / 0.16)
    v = np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-t / 1.7)
    v *= 1 - np.exp(-t / 0.004)
    nz = lp(rng.standard_normal(n), 260, 2) * np.exp(-t / 0.3) * 0.5
    v = v + nz
    v[-int(0.5 * SR):] *= np.linspace(1, 0, int(0.5 * SR))
    return v * gain


def soft_kick(vel):
    n = int(0.6 * SR)
    t = np.arange(n) / SR
    fr = 46 + 50 * np.exp(-t / 0.035)
    v = np.sin(2 * np.pi * np.cumsum(fr) / SR) * np.exp(-t / 0.22)
    v *= 1 - np.exp(-t / 0.002)
    return v * vel


def chord_tones(name, lo, hi):
    pcs = sorted({p % 12 for p in CHORDS[name][1]})
    return [m for m in range(lo, hi + 1) if m % 12 in pcs]


def chord_at(t):
    return PROG[min(int(t // SLOT), len(PROG) - 1)]


# ----------------------------------------------------------- the score ---

def build_reverb(sec=5.2):
    n = int(sec * SR)
    t = np.arange(n) / SR
    ir = stereo(n)
    for ch in range(2):
        z = rng.standard_normal(n)
        ir[:, ch] = (lp(z, 500) * np.exp(-6.9 * t / 4.4) + bp(z, 500, 4000) * np.exp(-6.9 * t / 3.4)
                     + hp(z, 4000) * np.exp(-6.9 * t / 1.6))
    ir[: int(0.012 * SR)] *= rc_ramp(int(0.012 * SR))[:, None]
    ir = np.vstack([np.zeros((int(0.024 * SR), 2)), ir])
    return ir / np.sqrt((ir ** 2).sum(0, keepdims=True))


def reverb(x, ir):
    return np.stack([signal.oaconvolve(x[:, c], ir[:, c])[:N] for c in range(2)], 1)


def pingpong(x, d=0.75, fb=0.38, n=6):
    out = np.zeros_like(x)
    D = int(d * SR)
    y = x.mean(1)
    for k in range(1, n + 1):
        y = lp(y, 3200, 1)
        if k * D >= len(x):
            break
        out[k * D:, k % 2] += fb ** k * y[:len(y) - k * D]
    return out


def score():
    pad, sub, pno, bel, shim, kick, hits = (stereo(N) for _ in range(7))

    # pad + sub, chord by chord
    for i, name in enumerate(PROG):
        t0 = i * SLOT
        bass, voicing = CHORDS[name]
        last = i == len(PROG) - 1
        hold = SLOT + 0.4 if not last else DURATION - t0
        for m in voicing:
            place(pad, pad_note(m, hold), t0 + rng.uniform(0, 0.25))
        place(sub, sub_note(bass, hold), t0)
        # shimmer: chord tones two octaves up, slow independent swells
        for m in voicing[1:4]:
            n = int((hold + 3) * SR)
            tt = np.arange(n) / SR
            sw = np.sin(np.pi * np.clip(tt / (hold + 3), 0, 1)) ** 2
            lfo = 0.5 + 0.5 * np.sin(2 * np.pi * rng.uniform(0.07, 0.2) * tt + rng.uniform(0, 6.28))
            v = np.sin(2 * np.pi * mtof(m + 24) * tt) * sw * lfo * 0.08
            place(shim, v, t0, pan=rng.uniform(-0.8, 0.8))

    def note(m, t, vel, pan=None):
        if t >= DURATION - 0.5:
            return
        pan = np.clip((m - 66) / 20, -0.6, 0.6) if pan is None else pan
        place(pno, piano(m, vel), t + rng.normal(0, 0.008), pan=pan)

    # piano: sparse melodic phrases / eighth-note arpeggios by section
    prev = 69
    beat = 0.0
    while beat < DURATION:
        t = beat
        name = chord_at(t)
        bar_pos = (t % SLOT)
        sparse = (16 <= t < 60) or (188 <= t < 199)
        arp = (60 <= t < 144) or (160 <= t < 188)
        if arp:
            tones = chord_tones(name, 57, 77)[:6]
            idx = [0, 2, 1, 3, 2, 4, 3, 1][int(round(bar_pos * 2)) % 8]
            vel = 0.24 if 104 <= t < 144 else 0.28
            if (t * 2) % 2 == 0:
                vel += 0.05
            note(tones[idx % len(tones)], t, vel * rng.uniform(0.85, 1.1))
        if sparse or arp:
            step = 1.5 if sparse else 2.0
            if abs((bar_pos / step) - round(bar_pos / step)) < 1e-6 and rng.random() < (0.8 if sparse else 0.55):
                cand = chord_tones(name, 64, 81)
                target = prev + rng.choice([-5, -3, -2, 2, 3, 4])
                m = min(cand, key=lambda c: abs(c - target) + (3 if c == prev else 0))
                prev = m
                v = 0.5 if not (188 <= t) else 0.4
                note(m, t, v * rng.uniform(0.85, 1.05))
                if 160 <= t < 188:
                    place(bel, bell(m + 12, 0.35), t, pan=rng.uniform(-0.5, 0.5))
        beat += 0.5

    # chapter 7: glass bells, mysterious
    for k, t in enumerate(np.arange(144.5, 160.0, 1.5)):
        cand = chord_tones(chord_at(t), 76, 88)
        place(bel, bell(cand[(k * 3) % len(cand)], 0.5), t, pan=(-0.6 if k % 2 else 0.6))
    for t in (144, 152):
        note(CHORDS[chord_at(t)][0] + 12, t, 0.45)

    # soft pulse under the tense chapters and the lift
    for t in np.arange(104, 144, 1.0):
        place(kick, soft_kick(0.5 if t % 4 == 0 else 0.32), t)
    for t in np.arange(168, 188, 1.0):
        place(kick, soft_kick(0.55 if t % 4 == 0 else 0.36), t)

    # hits
    place(hits, boom(1.0), TITLE_HIT)
    for m, dt in ((50, 0.0), (57, 0.06), (64, 0.12), (69, 0.18), (74, 0.26)):
        note(m, TITLE_HIT + dt, 0.42)
    place(bel, bell(81, 0.6), TITLE_HIT + 0.3, pan=0.3)
    place(hits, boom(0.9), COLLAPSE_HIT)
    place(bel, bell(81, 0.7), COLLAPSE_HIT, pan=-0.2)
    place(bel, bell(88, 0.5), COLLAPSE_HIT + 0.02, pan=0.2)
    place(hits, boom(0.55), MEASURE_HIT)
    place(bel, bell(88, 0.8, ratio=2.0), MEASURE_HIT, pan=-0.5)
    place(bel, bell(83, 0.6, ratio=2.0), MEASURE_HIT + 0.01, pan=0.5)
    place(hits, boom(0.7), LIFT)
    place(hits, boom(0.85), FINAL_HIT)
    for m, dt in ((38, 0.0), (50, 0.05), (57, 0.12), (64, 0.2), (69, 0.3), (74, 0.42), (76, 0.56)):
        note(m, FINAL_HIT + dt, 0.45)
    place(bel, bell(88, 0.55), FINAL_HIT + 0.6, pan=0.4)

    return dict(pad=pad, sub=sub, pno=pno, bel=bel, shim=shim, kick=kick, hits=hits)


def swells(ir):
    """Reverse-reverb risers into the main hits."""
    out = stereo(N)
    for hit, chord, length in ((TITLE_HIT, "Dm9", 4.0), (COLLAPSE_HIT, "Dm9", 3.5), (LIFT, "Fmaj9", 4.0),
                               (FINAL_HIT, "Dsus2", 4.0)):
        src = stereo(int(1.5 * SR))
        for m in CHORDS[chord][1]:
            place(src, bell(m + 12, 0.5, dur=1.5), 0.0, pan=rng.uniform(-0.7, 0.7))
        wet = np.stack([signal.oaconvolve(src[:, c], ir[:, c]) for c in range(2)], 1)
        seg = wet[: int(length * SR)][::-1].copy()
        seg *= np.linspace(0, 1, len(seg))[:, None] ** 2
        place(out, seg / (np.abs(seg).max() + 1e-9), hit - length, gain=0.35)
    return out


def loudness(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af", "ebur128=peak=true",
                        "-f", "null", "-"], capture_output=True, text=True)
    i = float(re.findall(r"I:\s+(-?[\d.]+) LUFS", r.stderr)[-1])
    p = float(re.findall(r"Peak:\s+(-?[\d.]+) dBFS", r.stderr)[-1])
    return i, p


def main():
    out_dir = ROOT / "build"
    out_dir.mkdir(exist_ok=True)
    ir = build_reverb()
    S = score()

    def norm(x, target_rms):
        r = np.sqrt((x ** 2).mean()) + 1e-12
        return x * (target_rms / r)

    pad = hp(lp(norm(S["pad"], 0.055), 3200), 110)
    pad *= automation([(0, 0), (5, 0.9), (16, 1), (186, 1), (192, 0.75), (200, 0.95), (DURATION, 0.9)])
    sub = norm(S["sub"], 0.026) * automation([(0, 0), (14, 0), (20, 1), (144, 1), (148, 0.45), (160, 0.45),
                                             (166, 1.1), (188, 1.1), (193, 0.4), (200, 0.9), (DURATION, 0.9)])
    pno = norm(S["pno"], 0.05)
    bel = S["bel"] / (np.abs(S["bel"]).max() + 1e-9) * 0.3
    shim = norm(S["shim"], 0.016) * automation([(0, 0), (4, 1), (16, 0.7), (26, 0.25), (144, 0.25),
                                                (150, 1), (168, 1), (188, 0.7), (200, 1.1), (DURATION, 1)])
    kick = S["kick"] / (np.abs(S["kick"]).max() + 1e-9) * 0.16
    hits = S["hits"] / (np.abs(S["hits"]).max() + 1e-9) * 0.42
    air_t = np.arange(N) / SR
    air = hp(rng.standard_normal((N, 2)), 5500, 2) * 0.004
    air *= (0.6 + 0.4 * np.sin(2 * np.pi * 0.05 * air_t))[:, None]
    air *= automation([(0, 0.0), (3, 1), (16, 0.5), (188, 0.6), (DURATION, 0.8)])
    sw = swells(ir)

    dry = pad + sub + pno + pingpong(pno) * 0.22 + bel * 0.8 + shim + kick + hits + air + sw * 0.6
    send = pad * 0.3 + pno * 0.55 + bel * 0.9 + shim * 0.6 + hits * 0.25 + sw * 0.4
    mix = dry + reverb(send, ir) * 0.55

    mix = hp(mix, 28, 2)
    # gentle glue compression on the loudest moments only (RMS detector, 1.6:1)
    env = np.sqrt(lp(np.mean(mix ** 2, 1), 3.0, 1).clip(1e-12))
    thr = np.percentile(env, 92)
    g = np.where(env > thr, (env / thr) ** (1 / 1.6 - 1), 1.0)
    mix *= g[:, None]
    # the arc: quiet open, breathe with the chapters, peak on the lift, settle for the outro
    arc_db = automation([(0, -9), (7.5, -5), (8.2, 0), (14, -1), (17, -4.5), (58, -4), (62, -2.5),
                         (100, -2), (104, -1.5), (142, -1.5), (146, -6), (158, -5), (160, -3.5),
                         (167.5, -2), (168.2, 0.5), (186, 0.5), (190, -6.5), (199.5, -6), (200.2, -1),
                         (DURATION, -1)])
    mix *= 10 ** (arc_db / 20)
    # fades
    fi = int(0.05 * SR)
    mix[:fi] *= rc_ramp(fi)[:, None]
    fo = int(5.0 * SR)
    mix[-fo:] *= (1 - rc_ramp(fo))[:, None] ** 1.5

    tmp = out_dir / "bgm_pre.wav"
    sf.write(tmp, (mix / np.abs(mix).max() * 0.5).astype(np.float32), SR, subtype="FLOAT")
    lufs, _ = loudness(tmp)
    target = -17.0
    mix = mix / np.abs(mix).max() * 0.5 * 10 ** ((target - lufs) / 20)
    # transparent peak control: smooth gain riding above -1.5 dBFS
    ceiling = 10 ** (-1.5 / 20)
    pk = np.maximum.reduce([np.abs(mix[:, 0]), np.abs(mix[:, 1])])
    from scipy.ndimage import maximum_filter1d
    pk = maximum_filter1d(pk, int(0.01 * SR))
    gain = np.minimum(1.0, ceiling / np.maximum(pk, 1e-9))
    gain = -maximum_filter1d(-gain, int(0.02 * SR))
    gain = lp(gain, 30, 1)
    mix *= gain[:, None]
    mix = np.clip(mix, -0.98, 0.98)
    out = out_dir / "bgm.wav"
    sf.write(out, mix.astype(np.float32), SR, subtype="PCM_24")
    tmp.unlink()
    lufs, peak = loudness(out)
    print(f"bgm: {out}  {DURATION}s  {lufs:.1f} LUFS  peak {peak:.1f} dBFS")


if __name__ == "__main__":
    main()
