"""Shared timeline for picture and score (global seconds).

Spine: 'impossible' is only 'improbable' - and we live inside that tiny
probability (the Sun burns because protons tunnel, slowly)."""

FPS = 30

SCENES = [
    ("prologue", 26),   # 00 序   不撞南墙不__
    ("wall", 16),       # 01 墙   classical: impossible
    ("wave", 18),       # 02 波   an electron is a cloud of possibilities
    ("seep", 44),       # 03 渗   seeping, slices, e^(-2κa), impossible -> improbable
    ("zeros", 16),      # 04 零   why we never walk through walls
    ("sun", 42),        # 05 光   if it were zero, the Sun would be dark
    ("epilogue", 16),   # 06 终   不撞南墙不__ (open)
]

START = {}
_t = 0
for _n, _d in SCENES:
    START[_n] = _t
    _t += _d
DURATION = _t

# prologue
FILL = 3.6
BALL_HIT = 6.2
SHOTS = [11.2, 12.3, 13.4, 14.5, 15.6]       # launches; impact 1.0 s later
THROUGH_SHOT = 3
BURST = 20.0
TITLE = 20.4
# wall / wave
IMPOSSIBLE = START["wall"] + 10.8
DETECT = (START["wave"] + 6.0, START["wave"] + 11.0)
# seep
COLLIDE = START["seep"] + 5.4
EMERGE = START["seep"] + 7.6
LAYERS = ([START["seep"] + 17.6 + 0.55 * k for k in range(10)]
          + [START["seep"] + 25.2 + 0.32 * k for k in range(10)])
NM1 = START["seep"] + 22.8
NM2 = START["seep"] + 28.4
FORMULA = START["seep"] + 31.4
MORPH = (START["seep"] + 38.0, START["seep"] + 40.2)
# zeros
ZEROS = (START["zeros"] + 0.8, START["zeros"] + 11.0)
# sun
DARK = START["sun"] + 0.6
BLACK_SUN = START["sun"] + 5.4
IGNITE = START["sun"] + 8.6
EARTH = START["sun"] + 30.0
PEAK = START["sun"] + 35.0
# epilogue
CALLBACK = START["epilogue"] + 1.0
FINAL = START["epilogue"] + 10.5
