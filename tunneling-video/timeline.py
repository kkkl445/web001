"""Shared timeline for picture and score (global seconds).

Spine: 'impossible' is only 'improbable' - and an improbable thing, tried
enough times, happens.  Uranium decays, the Sun shines, your phone stores
photos: all by knocking on a wall until it gives.

Every random event that is both seen and heard (Geiger clicks, wall hits,
detections) is generated here from fixed seeds so the score lands on the
same frames as the picture."""

import math

import numpy as np

FPS = 30

SCENES = [
    ("prologue", 26),   # 00 序   不撞南墙不__
    ("wall", 15),       # 01 墙   classical: impossible
    ("seep", 32),       # 02 渗   a cloud of possibilities seeps through (2D simulation)
    ("layers", 50),     # 03 层   exponential: impossible -> improbable; a person: zeros
    ("prison", 48),     # 04 狱   1928: alpha decay, 10^38 knocks, Geiger clicks
    ("light", 58),      # 05 光   1926/1929: the Sun; sparks become sunlight; your phone
    ("epilogue", 30),   # 06 终   no absolute wall; 不撞南墙不__
]

START = {}
_t = 0
for _n, _d in SCENES:
    START[_n] = _t
    _t += _d
DURATION = _t

# ------------------------------------------------------------ prologue ---
FILL = 3.6
BALL_HIT = 6.2
SHOTS = [11.2, 12.3, 13.4, 14.5, 15.6]       # launches; impact 1.0 s later
THROUGH_SHOT = 3
BURST = 20.0
TITLE = 20.4

# ---------------------------------------------------------------- wall ---
IMPOSSIBLE = START["wall"] + 10.2

# ---------------------------------------------------------------- seep ---
_s = START["seep"]
DETECT = (_s + 4.0, _s + 9.0)
SIM0 = _s + 10.4                     # simulation clock starts
COLLIDE = SIM0 + 5.4
EMERGE = SIM0 + 7.6
SPLIT = _s + 25.4                    # 80 % / 20 % readout


def detections():
    rng = np.random.default_rng(12)
    pts = rng.normal(0, 1, (40, 2))
    return pts, np.sort(rng.uniform(DETECT[0], DETECT[1], 40))


# -------------------------------------------------------------- layers ---
_l = START["layers"]
LAYERS = ([_l + 1.8 + 0.55 * k for k in range(10)]
          + [_l + 9.4 + 0.32 * k for k in range(10)])
NM1 = _l + 7.0
NM2 = _l + 12.6
FORMULA = _l + 17.8
MORPH0 = _l + 22.4                   # local morph clock
MORPH = (MORPH0 + 2.0, MORPH0 + 4.2)
ZEROS = (_l + 31.4, _l + 41.6)
PIVOT = _l + 45.2                    # 可如果，撞得足够多次呢？

# -------------------------------------------------------------- prison ---
_p = START["prison"]
YEAR28 = _p + 8.0
VOLCANO = _p + 12.0
GAMOW = _p + 24.6
COUNT = (_p + 27.4, _p + 40.2)       # hits counter runs 1 -> 1.4e38
BLUR = _p + 31.0                     # hits too fast to see
ESCAPE = _p + 41.0                   # tunnel lights; particle emerges ESCAPE + 0.45
FIELD = _p + 43.0


def prison_hits():
    """Visible wall hits inside the nucleus (global seconds)."""
    ts, t, dt = [], _p + 15.0, 0.85
    while t < COUNT[0]:
        ts.append(t)
        t += dt
    while t < BLUR + 0.4:
        ts.append(t)
        dt = max(0.035, dt * 0.86)
        t += dt
    return ts


def geiger():
    """Geiger counter clicks: (global time, nucleus index, direction)."""
    rng = np.random.default_rng(1928)
    ev = []
    for a, b, rate0, rate1 in ((_p + 0.6, _p + 9.6, 1.1, 2.6), (FIELD + 0.4, _p + 47.6, 2.4, 5.0)):
        t = a
        while True:
            u = (t - a) / (b - a)
            t += max(0.07, rng.exponential(1 / (rate0 + (rate1 - rate0) * u)))
            if t >= b:
                break
            ev.append((t, int(rng.integers(0, 240)), float(rng.uniform(0, 2 * math.pi))))
    return ev


# --------------------------------------------------------------- light ---
_g = START["light"]
QUOTE = _g + 4.8
YEAR29 = _g + 14.0
BOUNCES = [_g + 20.6, _g + 22.3, _g + 23.9, _g + 25.6]   # proton pairs repel
ZOOM = (_g + 27.6, _g + 36.2)                          # 10^2 -> 10^57 protons
FIRST_SPARK = _g + 29.2
SUNFORM = _g + 36.2                                    # field gathers into the disc
EARTH = _g + 40.6
PHONE = _g + 48.6
FLASH = _g + 52.4                                      # electrons tunnel into the cell

# ------------------------------------------------------------ epilogue ---
_e = START["epilogue"]
NOWALL = _e + 1.2
CALLBACK = _e + 9.4
FINAL = _e + 21.0
