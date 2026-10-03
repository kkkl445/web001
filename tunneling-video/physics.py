"""Physics behind the pictures (all deterministic, cached in .cache/).

* classical ball on a Gaussian hill (energy conservation)
* time-dependent Schroedinger equation, split-step Fourier (hbar = m = 1)
* exact stationary scattering off a rectangular barrier for a real electron
"""

import math

import numpy as np

from style import ROOT

CACHE = ROOT / ".cache"
CACHE.mkdir(exist_ok=True)


def _cached(name, fn):
    p = CACHE / f"{name}.npz"
    if p.exists():
        return dict(np.load(p))
    d = fn()
    np.savez_compressed(p, **d)
    return d


# ------------------------------------------------- classical ball (P1) ---

HILL_W = 120.0      # px, Gaussian width of the hill
BALL_E = 0.68       # ball energy as a fraction of the hill height


def hill(xr):
    """Hill height (fraction of peak) at offset xr (px) from its centre."""
    return np.exp(-(np.asarray(xr, float) / HILL_W) ** 2)


def ball_path(x_start, travel_time):
    """Ball released from x_start (left of the hill) rolling right; returns a
    function t -> x (px offset from hill centre), out and back in travel_time."""
    x_turn = -HILL_W * math.sqrt(math.log(1 / BALL_E))
    xs = np.linspace(x_start, x_turn, 4000)
    v = np.sqrt(np.clip(2 * (BALL_E - hill(xs)), 1e-6, None))
    dt = np.diff(xs) / (0.5 * (v[1:] + v[:-1]))
    tt = np.concatenate([[0], np.cumsum(dt)])
    tt *= (travel_time / 2) / tt[-1]

    def x_of(t):
        t = min(max(t, 0.0), travel_time)
        if t <= travel_time / 2:
            return float(np.interp(t, tt, xs))
        return float(np.interp(travel_time - t, tt, xs))
    return x_of, x_turn


# ------------------------------------------- wave packet sim (P2, P3) ----

V0 = 0.66
A = 2.4
K0 = 1.0
SIGMA = 8.0
X0 = -42.0
S_END = 80.0
FRAMES = 461
XS = np.linspace(-60, 60, 1000)


def packet():
    def build():
        N, L = 8192, 640.0
        x = (np.arange(N) - N / 2) * (L / N)
        dx = x[1] - x[0]
        k = 2 * np.pi * np.fft.fftfreq(N, dx)
        V = V0 * 0.5 * (np.tanh((x + A / 2) / 0.1) - np.tanh((x - A / 2) / 0.1))
        absorb = np.exp(-np.clip((np.abs(x) - 280) / 20, 0, None) ** 2)
        psi = (2 * np.pi * SIGMA ** 2) ** -0.25 * np.exp(-(x - X0) ** 2 / (4 * SIGMA ** 2) + 1j * K0 * x)
        dt = 0.01
        hv = np.exp(-0.5j * V * dt)
        kin = np.exp(-0.5j * k ** 2 * dt)
        per = int(round(S_END / (FRAMES - 1) / dt))
        sel = np.searchsorted(x, XS)
        re, im = [], []
        for f in range(FRAMES):
            re.append(psi[sel].real)
            im.append(psi[sel].imag)
            if f == FRAMES - 1:
                break
            for _ in range(per):
                psi = hv * np.fft.ifft(kin * np.fft.fft(hv * psi)) * absorb
        p = np.abs(psi) ** 2
        return dict(re=np.array(re, np.float32), im=np.array(im, np.float32),
                    T=np.float32(p[x > A / 2].sum() * dx), R=np.float32(p[x < -A / 2].sum() * dx))
    return _cached("packet", build)


# ------------------------------------ stationary electron scattering (P4) -

HB2M = 0.0380998   # hbar^2 / 2 m_e  in eV nm^2
E_EV = 1.0
V_EV = 2.0


def stationary(a_nm, x_nm, E=E_EV, V=V_EV):
    """psi(x) for a unit incident wave on a barrier [0, a]; returns (psi, T)."""
    k = math.sqrt(E / HB2M)
    q = math.sqrt((V - E) / HB2M)
    ea, eb = math.exp(q * a_nm), math.exp(-q * a_nm)
    ph = np.exp(1j * k * a_nm)
    # unknowns r, C, D, t
    M = np.array([[1, -1, -1, 0],
                  [-1j * k, -q, q, 0],
                  [0, ea, eb, -ph],
                  [0, q * ea, -q * eb, -1j * k * ph]], complex)
    rhs = np.array([-1, -1j * k, 0, 0], complex)
    r, C, D, t = np.linalg.solve(M, rhs)
    x = np.asarray(x_nm, float)
    psi = np.where(x < 0, np.exp(1j * k * x) + r * np.exp(-1j * k * x),
                   np.where(x <= a_nm, C * np.exp(q * np.minimum(x, a_nm)) + D * np.exp(-q * np.minimum(x, a_nm)),
                            t * np.exp(1j * k * x)))
    return psi, abs(t) ** 2


def transmission(a_nm, E=E_EV, V=V_EV):
    q = math.sqrt((V - E) / HB2M)
    return 1.0 / (1.0 + V * V * math.sinh(q * a_nm) ** 2 / (4 * E * (V - E)))


if __name__ == "__main__":
    d = packet()
    print("packet T=%.3f R=%.3f" % (d["T"], d["R"]))
    for a in (0.3, 0.5, 1.0, 2.0):
        _, T = stationary(a, [0.0])
        print(f"a={a} nm  T={T:.3e}  closed form {transmission(a):.3e}")
    print("ratio 1nm/2nm = %.0f" % (transmission(1.0) / transmission(2.0)))
