"""Physics behind the pictures (deterministic, cached in .cache/).

* 2D time-dependent Schroedinger equation, split-step Fourier (hbar = m = 1):
  a Gaussian wave packet hits a full-height barrier slab.
* exact stationary scattering off a rectangular barrier for a real electron
  (1 eV electron, 2 eV barrier) for the thickness numbers.
"""

import math

import numpy as np

ROOT = __import__("pathlib").Path(__file__).resolve().parent
CACHE = ROOT / ".cache"
CACHE.mkdir(exist_ok=True)


def _cached(name, fn):
    p = CACHE / f"{name}.npz"
    if p.exists():
        return dict(np.load(p))
    d = fn()
    np.savez(p, **d)
    return d


# ------------------------------------------------------------ 2D packet ---

NX, NY = 512, 256          # grid cells (dx = 1)
K0 = 0.5                   # wave number -> E = K0^2 / 2 = 0.125
V0 = 0.16                  # barrier height (above E)
BAR_X0, BAR_W = 254, 5     # barrier slab: cells [254, 259)
PX0, PY0, SIG = 104.0, 128.0, 18.0
T_END = 640.0
FRAMES = 361


def packet2d():
    def build():
        x = np.arange(NX, dtype=np.float64)
        y = np.arange(NY, dtype=np.float64)
        X, Y = np.meshgrid(x, y)
        kx = 2 * np.pi * np.fft.fftfreq(NX)
        ky = 2 * np.pi * np.fft.fftfreq(NY)
        KX, KY = np.meshgrid(kx, ky)
        edge = 0.5 * (np.tanh((X - BAR_X0) / 0.35) - np.tanh((X - BAR_X0 - BAR_W) / 0.35))
        V = V0 * edge
        absorb = (np.exp(-np.clip((18 - X) / 6, 0, None) ** 2) * np.exp(-np.clip((X - (NX - 19)) / 6, 0, None) ** 2)
                  * np.exp(-np.clip((10 - Y) / 5, 0, None) ** 2) * np.exp(-np.clip((Y - (NY - 11)) / 5, 0, None) ** 2))
        psi = np.exp(-((X - PX0) ** 2 + (Y - PY0) ** 2) / (4 * SIG ** 2) + 1j * K0 * X)
        psi /= np.sqrt((np.abs(psi) ** 2).sum())
        dt = 0.25
        hv = np.exp(-0.5j * V * dt)
        kin = np.exp(-0.5j * (KX ** 2 + KY ** 2) * dt)
        per = int(round(T_END / (FRAMES - 1) / dt))
        prob, re = [], []
        for f in range(FRAMES):
            prob.append((np.abs(psi) ** 2).astype(np.float32))
            re.append(psi.real.astype(np.float32))
            if f == FRAMES - 1:
                break
            for _ in range(per):
                psi = hv * np.fft.ifft2(kin * np.fft.fft2(hv * psi)) * absorb
        p = np.abs(psi) ** 2
        T = p[:, BAR_X0 + BAR_W:].sum()
        R = p[:, :BAR_X0].sum()
        return dict(prob=np.array(prob, np.float16), re=np.array(re, np.float16), T=np.float32(T),
                    R=np.float32(R))
    return _cached("packet2d", build)


# ------------------------------------ stationary electron scattering ----

HB2M = 0.0380998   # hbar^2 / 2 m_e  in eV nm^2
E_EV, V_EV = 1.0, 2.0


def transmission(a_nm, E=E_EV, V=V_EV):
    q = math.sqrt((V - E) / HB2M)
    return 1.0 / (1.0 + V * V * math.sinh(q * a_nm) ** 2 / (4 * E * (V - E)))


if __name__ == "__main__":
    d = packet2d()
    print("2D packet  T=%.3f  R=%.3f  frames %s" % (d["T"], d["R"], d["prob"].shape))
    for a in (0.5, 1.0, 2.0):
        print(f"a={a} nm  T={transmission(a):.3e}  (1 in {1 / transmission(a):,.0f})")
    print("ratio 1nm/2nm = %.0f" % (transmission(1.0) / transmission(2.0)))
