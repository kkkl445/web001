"""Precomputed physics used by the visuals.  Everything is deterministic
(fixed seeds) and cached to .cache/ so render workers share one copy."""

import numpy as np

from engine import ROOT

CACHE = ROOT / ".cache"
CACHE.mkdir(exist_ok=True)


def _cached(name, fn):
    p = CACHE / f"{name}.npz"
    if p.exists():
        return dict(np.load(p))
    d = fn()
    np.savez_compressed(p, **d)
    return d


# ------------------------------------------------- hydrogen orbitals -----

def _sample_orbital(density, box, n, seed):
    """Rejection-sample n points from an (unnormalised) 3D density."""
    rng = np.random.default_rng(seed)
    probe = rng.uniform(-box, box, (400_000, 3))
    dmax = density(probe).max() * 1.05
    out = []
    got = 0
    while got < n:
        p = rng.uniform(-box, box, (2_000_000, 3))
        keep = rng.uniform(0, dmax, len(p)) < density(p)
        out.append(p[keep])
        got += keep.sum()
    return np.concatenate(out)[:n].astype(np.float32)


def _d_z2(p):
    r = np.linalg.norm(p, axis=1) + 1e-9
    c = p[:, 2] / r
    return r ** 4 * np.exp(-2 * r / 3) * (3 * c * c - 1) ** 2


def _f_z3(p):
    r = np.linalg.norm(p, axis=1) + 1e-9
    c = p[:, 2] / r
    return r ** 6 * np.exp(-r / 2) * (5 * c ** 3 - 3 * c) ** 2


def orbitals():
    return _cached("orbitals", lambda: {
        "d_z2": _sample_orbital(_d_z2, 24.0, 110_000, 1),
        "f_z3": _sample_orbital(_f_z3, 42.0, 110_000, 2),
    })


# -------------------------------------------------------- double slit ----

# Geometry in screen pixels (shared with the scene code).
DS_SOURCE = (880.0, 560.0)
DS_BARRIER_X = 1140.0
DS_SCREEN_X = 1620.0
DS_CY = 560.0
DS_HALF_SEP = 46.0
DS_LAMBDA = 16.0
DS_HALF_SPAN = 300.0


def ds_interference(Y):
    L = DS_SCREEN_X - DS_BARRIER_X
    k = 2 * np.pi / DS_LAMBDA
    r1 = np.hypot(L, Y - DS_HALF_SEP)
    r2 = np.hypot(L, Y + DS_HALF_SEP)
    amp = np.exp(1j * k * r1) / np.sqrt(r1) + np.exp(1j * k * r2) / np.sqrt(r2)
    return np.abs(amp) ** 2 * np.exp(-(Y / 235.0) ** 2)


def ds_which_path(Y):
    return np.exp(-((Y - 34) / 150.0) ** 2) + np.exp(-((Y + 34) / 150.0) ** 2)


def _sample_1d(pdf_fn, n, rng):
    Y = np.linspace(-DS_HALF_SPAN, DS_HALF_SPAN, 6001)
    pdf = pdf_fn(Y)
    cdf = np.cumsum(pdf)
    cdf /= cdf[-1]
    return np.interp(rng.uniform(0, 1, n), cdf, Y).astype(np.float32)


def _emission_times(t_single, singles, t_fast0, t_fast1, total, rng, k=5.5):
    """A few lone electrons at a steady pace, then an accelerating flood."""
    ts = [t_single + i * singles[1] for i in range(singles[0])]
    n_fast = total - len(ts)
    # rate grows exponentially: cumulative count ~ exp(u*k)-1
    u = np.log1p(np.sort(rng.uniform(0, 1, n_fast)) * (np.exp(k) - 1)) / k
    fast = t_fast0 + u * (t_fast1 - t_fast0)
    return np.concatenate([np.array(ts), fast]).astype(np.float32)


def double_slit():
    def build():
        rng = np.random.default_rng(11)
        # pass 1: interference (chapter 2), times relative to chapter start
        t1 = _emission_times(1.6, (11, 0.62), 8.4, 21.5, 4200, rng)
        y1 = _sample_1d(ds_interference, len(t1), rng)
        x1 = rng.uniform(0, 1, len(t1)).astype(np.float32)
        # pass 2: which-path detector on (chapter 3)
        t2 = _emission_times(5.0, (6, 0.55), 8.3, 13.5, 3600, rng, k=3.2)
        y2 = _sample_1d(ds_which_path, len(t2), rng)
        x2 = rng.uniform(0, 1, len(t2)).astype(np.float32)
        slit2 = np.where(y2 + rng.normal(0, 60, len(y2)) > 0, 1, -1).astype(np.float32)
        return dict(t1=t1, y1=y1, x1=x1, t2=t2, y2=y2, x2=x2, slit2=slit2)
    return _cached("double_slit", build)


# ---------------------------------------------------------- tunnelling ---

TUN_V0 = 0.62
TUN_A = 2.6
TUN_K0 = 1.0
TUN_X0 = -42.0
TUN_SIGMA = 5.0
TUN_T_END = 78.0
TUN_FRAMES = 481
TUN_XS = np.linspace(-62, 62, 1000)


def tunnelling():
    def build():
        N, Lbox = 8192, 640.0
        x = (np.arange(N) - N / 2) * (Lbox / N)
        dx = x[1] - x[0]
        k = 2 * np.pi * np.fft.fftfreq(N, dx)
        V = TUN_V0 * 0.5 * (np.tanh((x + TUN_A / 2) / 0.12) - np.tanh((x - TUN_A / 2) / 0.12))
        absorb = np.exp(-np.clip((np.abs(x) - 280) / 20, 0, None) ** 2)
        psi = (2 * np.pi * TUN_SIGMA ** 2) ** -0.25 * np.exp(
            -(x - TUN_X0) ** 2 / (4 * TUN_SIGMA ** 2) + 1j * TUN_K0 * x)
        dt = 0.01
        half_v = np.exp(-0.5j * V * dt)
        kin = np.exp(-0.5j * k ** 2 * dt)
        steps_per = int(round(TUN_T_END / (TUN_FRAMES - 1) / dt))
        sel = np.searchsorted(x, TUN_XS)
        prob, re = [], []
        for f in range(TUN_FRAMES):
            prob.append(np.abs(psi[sel]) ** 2)
            re.append(psi[sel].real)
            if f == TUN_FRAMES - 1:
                break
            for _ in range(steps_per):
                psi = half_v * psi
                psi = np.fft.ifft(kin * np.fft.fft(psi))
                psi = half_v * psi * absorb
        p = np.abs(psi) ** 2
        T = p[x > TUN_A / 2].sum() * dx
        R = p[x < -TUN_A / 2].sum() * dx
        return dict(prob=np.array(prob, np.float32), re=np.array(re, np.float32),
                    T=np.float32(T), R=np.float32(R))
    return _cached("tunnelling", build)


if __name__ == "__main__":
    o = orbitals()
    print({k: v.shape for k, v in o.items()})
    d = double_slit()
    print({k: v.shape for k, v in d.items()})
    tn = tunnelling()
    print("T=%.3f R=%.3f" % (tn["T"], tn["R"]), tn["prob"].shape, tn["prob"].max())
