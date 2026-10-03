"""Shared timeline for picture and score (global seconds)."""

FPS = 30

SCENES = [
    ("prologue", 26),   # 00 序
    ("wall", 20),       # 01 墙
    ("wave", 22),       # 02 波
    ("through", 30),    # 03 穿
    ("thin", 26),       # 04 薄
    ("light", 26),      # 05 光
    ("epilogue", 16),   # 终
]

START = {}
_t = 0
for _n, _d in SCENES:
    START[_n] = _t
    _t += _d
DURATION = _t

# sync points for the score
BALL_HIT = 2.6
REEL = (10.2, 18.6)
ELECTRON_HITS = (11.6, 15.6)
BURST = 20.0
TITLE = 20.6
CHAPTERS = [START[n] for n in ("wall", "wave", "through", "thin", "light", "epilogue")]
IMPOSSIBLE = START["wall"] + 16.8
DETECT = (START["wave"] + 10.0, START["wave"] + 16.0)
COLLIDE = START["through"] + 10.2
EMERGE = START["through"] + 13.8
REVEAL = START["through"] + 22.4
STEPS = [START["thin"] + s for s in (3.0, 7.5, 11.0, 14.5)]
FORMULA = START["thin"] + 18.0
VIGNETTES = [START["light"] + s for s in (0.0, 9.0, 17.0)]
SPARKS = (START["light"] + 23.5, START["epilogue"] + 5.0)
MORPH = (START["epilogue"] + 7.5, START["epilogue"] + 10.0)
FINAL = START["epilogue"] + 13.0
