"""Shared timeline for picture and score.  120 BPM: one bar = 2 s, and every
scene length is a multiple of 4 s, so cuts land on the downbeat."""

FPS = 30

SCENES = [
    ("opening", 12),
    ("p1", 16),   # 经典的墙
    ("p2", 16),   # 电子是一团波
    ("p3", 20),   # 渗进墙里
    ("p4", 20),   # 一纳米的差距
    ("p5", 28),   # 它一直都在
    ("outro", 12),
]

START = {}
_t = 0
for _n, _d in SCENES:
    START[_n] = _t
    _t += _d
DURATION = _t

WIPES = [START["p1"], START["p4"], START["p5"], START["outro"]]
TITLE_HIT = 2.0
RELEASE = START["p2"] + 8.0          # the wave packet is let go
TUNNEL_HIT = START["p3"] + 2.0       # transmitted packet emerges
TILE_HITS = [START["p5"] + d for d in (4.0, 8.0, 12.0, 16.0)]
STRIKE_HIT = START["outro"] + 4.0
