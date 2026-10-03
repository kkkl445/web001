"""Shared timeline for picture and score (global seconds), third cut.

Spine: ChatGPT only ever guesses the next word; to guess well it must know
who 「它」 is.  Words are numbers - stars in a sky of meaning.  Attention is
how each star looks around and moves to where it belongs: question, label,
content; multiply and add; softmax; mix.  Layers and training.  Then the
bigger question - why guessing the next word looks like intelligence: to
predict you must understand, you cannot memorise so you must generalise, and
generalising (概括) is what minds have always done.  Emergence."""

FPS = 30

SCENES = [
    ("cold", 6),         # cold open: the formula lands; every AI chatbot runs on this line
    ("hook", 38),        # 猜下一个字; the chat; 它是谁; title
    ("sky", 46),         # words are numbers, numbers are stars
    ("sequential", 30),  # the old way: the telephone game
    ("core", 98),        # Q, K, V; the dot product; softmax; the mix; the formula; the grid
    ("heads", 22),       # eight pairs of eyes
    ("order", 20),       # 猫追狗 / 狗追猫; position waves
    ("learn", 46),       # layers; who fills in the numbers; the paper's result
    ("mind", 98),        # why guessing becomes intelligence; 概括; emergence
    ("after", 30),       # the page, the song, GPT, every conversation
    ("ending", 24),      # 你的注意力
]

START = {}
_t = 0
for _n, _d in SCENES:
    START[_n] = _t
    _t += _d
DURATION = _t

# cold open (local seconds)
CO_FORM, CO_L1, CO_L2, CO_L3, CO_GO = 0.15, 1.0, 2.6, 3.9, 5.0

# hook (local seconds)
H_TYPE, H_BARS, H_CHAT, H_PLATE = 0.3, 4.2, 8.4, 18.0
H_HOP, H_LIE, H_IT, H_LINK, H_SWAP, H_RE = 19.2, 19.8, 20.0, 20.4, 23.0, 23.6
H_TURN, H_TITLE = 27.6, 31.8

# sky
S_NUM, S_COUNT, S_STAR, S_GROUPS, S_KING, S_APPLE, S_IT, S_PULL = 0.4, 4.2, 9.8, 10.6, 19.4, 30.0, 35.4, 39.8

# sequential
SEQ_PASS0, SEQ_PASS_DT = 4.0, 2.2
SEQ_READ0, SEQ_READ_DT = 6.0, 1.2

# core
C_WEB, C_ASK, C_MAT, C_Q, C_K, C_V = 2.0, 5.4, 9.4, 15.4, 18.8, 22.2
C_QARROW, C_KARROWS, C_SUM, C_ALIGN, C_SCORES = 26.6, 31.6, 36.6, 41.4, 46.4
C_SOFT, C_MIX, C_MOVE, C_KNOW = 52.0, 58.4, 64.0, 68.0
C_FORMULA, C_QK, C_SM, C_TV, C_SQ = 72.4, 77.2, 79.2, 81.2, 84.0
C_GRID, C_GPU = 90.0, 94.4

# heads
HEADS_T0, HEADS_DT, HEADS_ALL, HEADS_MERGE = 2.0, 1.1, 11.6, 16.0

# order
ORD_SWAP, ORD_WAVES = 4.6, 10.0

# learn
L_LAYER, L_STACK, L_96, L_MOVE, L_WHO, L_MASK, L_NUDGE, L_COUNT, L_PAPER = 0.8, 5.0, 7.4, 9.8, 15.4, 19.8, 25.0, 30.8, 36.4

# mind
M_ASK, M_BOOK, M_CLUES, M_FORCED, M_CANT, M_GEN, M_HUMAN = 0.6, 5.4, 10.4, 16.4, 20.4, 27.0, 30.4
M_KEPLER, M_NEWTON, M_CAT, M_SKY, M_EMERGE, M_BIRDS, M_NEURON, M_SCALE = 34.0, 41.0, 46.0, 52.4, 59.0, 62.4, 69.4, 75.0
M_DEBATE, M_GUESS, M_QUESTION = 81.4, 85.4, 91.8

# after
A_PAGE, A_NOTE, A_SONG, A_GPT, A_CHAT = 0.3, 3.6, 7.0, 12.6, 17.4

# ending
END_LINE, END_TITLE = 12.4, 19.2
