"""Synthesis toolkit shared by the score: voices, effects, mastering.

Everything is generated from code (additive pads, felt piano, FM bells,
glassy blips, booms, risers, synthetic reverb) - no samples."""

import functools
import re
import subprocess

import numpy as np
import soundfile as sf
from scipy import signal
from scipy.ndimage import maximum_filter1d

SR = 48000
rng = np.random.default_rng(2017)


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


def geiger_click(vel):
    """Sharp tick of a Geiger-Mueller tube: broadband snap, a little body."""
    t = np.arange(int(0.04 * SR)) / SR
    snap = bp(rng.standard_normal(len(t)) * np.exp(-t / 0.0009), 1500, 11000)
    body = np.sin(2 * np.pi * rng.uniform(2600, 3400) * t) * np.exp(-t / 0.003) * 0.35
    thump = np.sin(2 * np.pi * 170 * t) * np.exp(-t / 0.007) * 0.5
    v = snap / (np.abs(snap).max() + 1e-9) + body + thump
    return v * vel


def pulse_train(t0, t1, r0, r1, gain=1.0):
    """Hits too fast to count: a click train whose rate climbs until it becomes a tone."""
    n = int((t1 - t0) * SR)
    u = np.arange(n) / n
    rate = r0 * (r1 / r0) ** (u ** 1.3)
    ph = np.cumsum(rate) / SR
    imp = np.zeros(n)
    idx = np.nonzero(np.diff(np.floor(ph)) > 0)[0]
    imp[idx] = 1.0
    k = np.exp(-np.arange(int(0.004 * SR)) / (0.0008 * SR))
    v = signal.fftconvolve(imp, k)[:n]
    v = bp(v, 300, 6000) + 0.5 * bp(v, 60, 300)
    v = v / (np.abs(v).max() + 1e-9) * (0.35 + 0.65 * u ** 1.5)
    v[-int(0.06 * SR):] *= np.linspace(1, 0, int(0.06 * SR))
    v[:int(0.3 * SR)] *= rc(int(0.3 * SR))
    return v * gain


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
    return np.stack([signal.oaconvolve(x[:, c], ir[:, c])[:len(x)] for c in range(2)], 1)


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


def smooth_np(t, a, b):
    x = np.clip((t - a) / (b - a), 0, 1)
    return x * x * (3 - 2 * x)


def loudness(path):
    r = subprocess.run(["ffmpeg", "-hide_banner", "-nostats", "-i", str(path), "-af", "ebur128=peak=true",
                        "-f", "null", "-"], capture_output=True, text=True)
    return (float(re.findall(r"I:\s+(-?[\d.]+) LUFS", r.stderr)[-1]),
            float(re.findall(r"Peak:\s+(-?[\d.]+) dBFS", r.stderr)[-1]))




def automation(points, n):
    ts, gs = zip(*points)
    return np.interp(np.arange(n) / SR, ts, gs)[:, None]


def master(mix, out, lufs_target=-14.5, ceiling_db=-1.5):
    """Loudness-normalise and peak-limit a stereo mix, write 24-bit WAV."""
    tmp = out.with_name(out.stem + "_pre.wav")
    sf.write(tmp, (mix / np.abs(mix).max() * 0.5).astype(np.float32), SR, subtype="FLOAT")
    lufs, _ = loudness(tmp)
    mix = mix / np.abs(mix).max() * 0.5 * 10 ** ((lufs_target - lufs) / 20)
    ceiling = 10 ** (ceiling_db / 20)
    pk = maximum_filter1d(np.maximum(np.abs(mix[:, 0]), np.abs(mix[:, 1])), int(0.01 * SR))
    gain = np.minimum(1.0, ceiling / np.maximum(pk, 1e-9))
    gain = lp(-maximum_filter1d(-gain, int(0.02 * SR)), 30, 1)
    mix = np.clip(mix * gain[:, None], -0.98, 0.98)
    sf.write(out, mix.astype(np.float32), SR, subtype="PCM_24")
    tmp.unlink()
    return loudness(out)
