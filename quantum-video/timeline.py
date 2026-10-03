"""Shared timeline for the video renderer and the BGM synthesiser.

Every scene length is a multiple of 4 s (one bar at 60 BPM), so picture and
music cut on the same grid.
"""

FPS = 30

SCENES = [
    ("opening", 16),
    ("ch1", 20),   # 量子化
    ("ch2", 24),   # 波粒二象性
    ("ch3", 20),   # 测量
    ("ch4", 24),   # 叠加与坍缩
    ("ch5", 20),   # 不确定性原理
    ("ch6", 20),   # 量子隧穿
    ("ch7", 24),   # 量子纠缠
    ("ch8", 20),   # 无处不在
    ("outro", 20),
]

START = {}
_t = 0
for _name, _dur in SCENES:
    START[_name] = _t
    _t += _dur
DURATION = _t

# Moments the music hits on (seconds, global time).
TITLE_HIT = 8.0
COLLAPSE_HIT = START["ch4"] + 16.0
MEASURE_HIT = START["ch7"] + 12.0
LIFT = START["ch8"]
FINAL_HIT = START["outro"] + 12.0
