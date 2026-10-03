"""Shared timeline for picture and score (global seconds).

Spine: you already do attention - instantly, without thinking.  Before 2017
machines read one word at a time and forgot the beginning; the Transformer
lets every word look at every other word at once.  Q/K/V, eight heads,
positions, and what came after; and attention, for people, too."""

FPS = 30

SCENES = [
    ("prologue", 24),    # 它是谁 -> title
    ("sequential", 34),  # 一个字一个字地读: the telephone game
    ("atonce", 32),      # 一眼: every word looks at every word; the attention table
    ("qkv", 42),         # 问与答: query, key, value; the one line
    ("heads", 26),       # 八双眼睛: multi-head attention
    ("order", 26),       # 顺序: 猫追狗 / 狗追猫; positional waves
    ("after", 30),       # 之后: June 2017, the byline, GPT, every conversation
    ("ending", 24),      # 你的注意力
]

START = {}
_t = 0
for _n, _d in SCENES:
    START[_n] = _t
    _t += _d
DURATION = _t

# prologue (local seconds, same beats as the approved teaser)
P_T1, P_T2 = 0.25, 1.35
P_HOP, P_LIE = 1.9, 2.5
P_IT, P_FAN, P_SWAP, P_RE, P_NAME = 2.6, 3.2, 5.0, 5.6, 7.8
P_TITLE = 15.4

# sequential: the message passed person to person, the reader pointer word to word
SEQ_PASS0, SEQ_PASS_DT = 5.0, 2.4          # local: first hand-over, then every 2.4 s
SEQ_READ0, SEQ_READ_DT = 8.0, 1.6          # local: pointer steps through the tokens (reaches 它 at 16 s)

# atonce
ATO_WEB, ATO_TABLE = 5.2, 16.6             # local

# qkv
QKV_Q, QKV_K, QKV_V, QKV_MATCH, QKV_FORMULA = 5.0, 10.0, 15.0, 20.5, 27.5   # local

# heads
HEADS_T0, HEADS_DT, HEADS_ALL = 3.0, 1.35, 15.0   # local

# order
ORD_SWAP, ORD_WAVES = 6.0, 11.5            # local

# after
AFT_BYLINE, AFT_BEATLES, AFT_GPT, AFT_TOWER = 5.4, 11.2, 17.0, 22.6   # local

# ending
END_LINE, END_TITLE = 12.4, 19.2           # local
