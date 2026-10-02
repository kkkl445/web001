import json, math, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import sim

sim.DRAG = 2.0
W, H, FPS, SS = 1280, 720, 30, 2   # SS = 超采样倍数，抗锯齿

CJK = '/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc'
LAT = '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf'
LATB = '/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf'
MONO = '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf'
_fc = {}
def font(path, size):
    k = (path, size)
    if k not in _fc:
        _fc[k] = ImageFont.truetype(path, int(size * SS))
    return _fc[k]

BG = (13, 18, 36); PANEL = (24, 32, 56); GRID = (48, 60, 92)
TXT = (228, 233, 245); DIM = (140, 152, 180)
CP = (249, 132, 52); CI = (52, 211, 120); CD = (80, 150, 255)
TGT = (250, 204, 21); CURVE = (90, 225, 255); RED = (244, 80, 90); PURP = (190, 120, 255)

# ---------------- 仿真数据 ----------------
RUNS = {
    'bang': sim.simulate(kind='bang'),
    'p5': sim.simulate(kind='p', kp=5),
    'p12': sim.simulate(kind='p', kp=12),
    'pi': sim.simulate(kind='pi', kp=5, ki=2),
    'pid': sim.simulate(kind='pid', kp=6, ki=2.5, kd=2.5),
}
SIM_T = sim.T
PLAY = 1.4   # 播放倍速

# ---------------- 时间线 ----------------
raw = json.load(open('timeline_raw.json'))
NEED = {  # 场景所需的最短时长（相对场景起点，由动画决定）
    'bang': lambda s: s[0][0] + 0.5 + SIM_T / PLAY + 1.0,
    'p': lambda s: s[5][0] + SIM_T / PLAY + 1.0,
    'i': lambda s: s[1][0] + SIM_T / PLAY + 1.0,
    'd': lambda s: s[3][0] + SIM_T / PLAY + 0.8,
    'end': lambda s: s[-1][1] + 2.5,
}
SC = []; t = 0.0
for sc in raw['scenes']:
    dur = sc['audio_len']
    if sc['id'] in NEED:
        dur = max(dur, NEED[sc['id']](sc['segs']))
    SC.append(dict(sc, start=t, dur=dur)); t += dur
TOTAL = t

# ---------------- 绘图工具 ----------------
class Cv:
    def __init__(self):
        self.im = Image.new('RGB', (W * SS, H * SS), BG)
        self.d = ImageDraw.Draw(self.im, 'RGBA')
    def P(self, *xy): return [v * SS for v in xy]
    def rect(self, x0, y0, x1, y1, fill=None, outline=None, w=1, r=0):
        self.d.rounded_rectangle(self.P(x0, y0, x1, y1), radius=r * SS, fill=fill, outline=outline, width=int(w * SS))
    def line(self, pts, fill, w=2):
        self.d.line([(x * SS, y * SS) for x, y in pts], fill=fill, width=int(w * SS), joint='curve')
    def dashed(self, x0, y, x1, fill, w=2, dash=10):
        x = x0
        while x < x1:
            self.line([(x, y), (min(x + dash, x1), y)], fill, w); x += dash * 1.8
    def circle(self, x, y, r, fill=None, outline=None, w=1):
        self.d.ellipse(self.P(x - r, y - r, x + r, y + r), fill=fill, outline=outline, width=int(w * SS))
    def ellipse(self, x0, y0, x1, y1, fill=None, outline=None, w=1):
        self.d.ellipse(self.P(x0, y0, x1, y1), fill=fill, outline=outline, width=int(w * SS))
    def poly(self, pts, fill):
        self.d.polygon([(x * SS, y * SS) for x, y in pts], fill=fill)
    def text(self, x, y, s, size=24, fill=TXT, f=CJK, anchor='la'):
        self.d.text((x * SS, y * SS), s, font=font(f, size), fill=fill, anchor=anchor)
    def tw(self, s, size, f=CJK):
        return self.d.textlength(s, font=font(f, size)) / SS
    def arrow(self, x0, y0, x1, y1, fill, w=4, head=12):
        self.line([(x0, y0), (x1, y1)], fill, w)
        ang = math.atan2(y1 - y0, x1 - x0)
        p1 = (x1 - head * math.cos(ang - 0.45), y1 - head * math.sin(ang - 0.45))
        p2 = (x1 - head * math.cos(ang + 0.45), y1 - head * math.sin(ang + 0.45))
        self.poly([(x1, y1), p1, p2], fill)
    def rich(self, x, y, parts, size=30, anchor_center=False):
        """parts: [(text, color, fontpath)]；按片段拼接绘制"""
        total = sum(self.tw(s, size, f) for s, c, f in parts)
        if anchor_center: x -= total / 2
        for s, c, f in parts:
            self.text(x, y, s, size, c, f, anchor='ls'); x += self.tw(s, size, f)
        return total
    def out(self):
        return self.im.resize((W, H), Image.LANCZOS)

def ease(x): x = min(max(x, 0), 1); return x * x * (3 - 2 * x)
def fade(t, t0, d=0.5): return ease((t - t0) / d)
def mix(c, a, bg=BG): return tuple(int(bg[i] + (c[i] - bg[i]) * a) for i in range(3))
def rgba(c, a): return (c[0], c[1], c[2], int(255 * max(0, min(1, a))))

def sim_idx(run, t_sim):
    return int(min(max(t_sim, 0), SIM_T - 1e-6) / sim.DT)

# ---------------- 无人机世界面板 ----------------
WX0, WY0, WX1, WY1 = 40, 90, 400, 610
def wy(h): return WY1 - 30 - h / 10.0 * (WY1 - WY0 - 60)

def world(c, h=0.0, thrust=None, t=0.0, spring=False, show_err=False, label=True):
    c.rect(WX0, WY0, WX1, WY1, fill=PANEL, r=14)
    for m in range(0, 11, 2):
        y = wy(m)
        c.line([(WX0 + 38, y), (WX0 + 46, y)], DIM, 2)
        c.text(WX0 + 32, y, f'{m}m', 15, DIM, LAT, 'rm')
    c.line([(WX0 + 46, wy(0)), (WX0 + 46, wy(10))], GRID, 2)
    gy = wy(0) + 8
    c.line([(WX0 + 20, gy), (WX1 - 20, gy)], DIM, 3)
    for x in range(WX0 + 24, WX1 - 20, 16):
        c.line([(x, gy + 2), (x - 9, gy + 12)], GRID, 2)
    yt = wy(5)
    c.dashed(WX0 + 50, yt, WX1 - 16, TGT, 2)
    c.text(WX0 + 54, yt - 6, '目标 5m', 17, TGT, CJK, 'lb')
    cx = (WX0 + WX1) / 2 + 20
    yd = wy(h) - 8
    if spring:
        sx = cx - 70; n = 10; pts = []
        for k in range(n * 2 + 1):
            yy = yt + (yd - yt) * k / (n * 2)
            xx = sx + (0 if k in (0, n * 2) else (9 if k % 2 else -9))
            pts.append((xx, yy))
        c.line(pts, CP, 3)
        c.circle(sx, yt, 4, fill=CP); c.circle(sx, yd, 4, fill=CP)
        c.text(sx - 14, (yt + yd) / 2, '弹簧', 16, CP, CJK, 'rm')
    if show_err and abs(yt - yd) > 8:
        ex = cx + 80
        c.arrow(ex, yd, ex, yt + (3 if yt < yd else -3), RED, 3, 10)
        c.text(ex + 6, (yt + yd) / 2, f'e={5 - h:+.2f}', 15, RED, CJK, 'lm')
    # 推力箭头 & 重力
    if thrust is not None:
        L = thrust * 4.2
        if L > 4:
            c.arrow(cx, yd - 14, cx, yd - 14 - L, CURVE, 5, 13)
        c.arrow(cx + 50, yd + 8, cx + 50, yd + 8 + 9.8 * 3.2, RED, 3, 9)
        c.text(cx + 58, yd + 26, 'mg', 15, RED, LAT, 'lm')
    drone(c, cx, yd, t, thrust if thrust is not None else 10)
    if label:
        c.text(cx, yd + 32, f'h={h:.2f}m', 16, TXT, LAT, 'mm')

def drone(c, cx, y, t, thrust):
    c.rect(cx - 30, y - 7, cx + 30, y + 7, fill=(210, 220, 240), r=6)
    c.rect(cx - 12, y - 3, cx + 12, y + 3, fill=(40, 60, 100), r=2)
    for sgn in (-1, 1):
        ax = cx + sgn * 44
        c.line([(cx + sgn * 28, y), (ax, y - 6)], (180, 190, 210), 4)
        c.line([(ax, y - 6), (ax, y - 12)], (180, 190, 210), 3)
        ph = t * (8 + thrust * 1.5)
        wv = abs(math.cos(ph)) * 22 + 4
        c.ellipse(ax - 22, y - 15, ax + 22, y - 11, fill=(120, 200, 255, 50))
        c.line([(ax - wv, y - 13), (ax + wv, y - 13)], (230, 240, 255), 3)
    c.line([(cx - 16, y + 7), (cx - 22, y + 15)], (180, 190, 210), 3)
    c.line([(cx + 16, y + 7), (cx + 22, y + 15)], (180, 190, 210), 3)

# ---------------- 曲线面板 ----------------
PX0, PY0, PX1, PY1 = 500, 250, 1235, 600
def plot_frame(c, box=None, tmax=SIM_T, ymax=10):
    x0, y0, x1, y1 = box or (PX0, PY0, PX1, PY1)
    c.rect(x0 - 70, y0 - 20, x1 + 15, y1 + 52, fill=PANEL, r=14)
    fx = lambda tt: x0 + tt / tmax * (x1 - x0)
    fy = lambda h: y1 - h / ymax * (y1 - y0)
    for m in range(0, ymax + 1, 2):
        c.line([(x0, fy(m)), (x1, fy(m))], GRID if m else DIM, 1)
        c.text(x0 - 10, fy(m), f'{m}', 15, DIM, LAT, 'rm')
    for s in range(0, int(tmax) + 1, 2):
        c.text(fx(s), y1 + 8, f'{s}s', 15, DIM, LAT, 'ma')
    c.text(x0 - 55, (y0 + y1) / 2, '高度', 16, DIM, CJK, 'mm')
    c.text((x0 + x1) / 2, y1 + 32, '时间', 16, DIM, CJK, 'mm')
    c.dashed(x0, fy(5), x1, TGT, 2)
    return fx, fy

def curve(c, fx, fy, run, t_sim, color, w=3, ymax=10):
    n = sim_idx(run, t_sim) + 1
    hh = run['h'][:n:6]; tt = run['t'][:n:6]
    if len(hh) < 2: return
    pts = [(fx(a), fy(min(b, ymax))) for a, b in zip(tt, hh)]
    c.line(pts, color, w)
    c.circle(pts[-1][0], pts[-1][1], 5, fill=color)

def err_area(c, fx, fy, run, t_sim, color):
    n = sim_idx(run, t_sim) + 1
    hh = run['h'][:n:10]; tt = run['t'][:n:10]
    for k in range(len(hh) - 1):
        a, b = hh[k], hh[k + 1]
        col = rgba(color, 0.38) if (a + b) / 2 < 5 else rgba(RED, 0.30)
        c.poly([(fx(tt[k]), fy(5)), (fx(tt[k + 1]), fy(5)), (fx(tt[k + 1]), fy(min(b, 10))), (fx(tt[k]), fy(min(a, 10)))], col)

def run_state(run, ts):
    i = sim_idx(run, ts)
    return run['h'][i], run['u'][i]

# ---------------- 公共元素 ----------------
def header(c, sc, t):
    if not sc['name']: return
    a = fade(t, 0, 0.4)
    c.rect(40, 24, 40 + 8, 64, fill=mix(CURVE, a))
    c.text(60, 44, sc['name'], 30, mix(TXT, a), CJK, 'lm')
    c.text(W - 40, 44, 'PID 控制算法科普', 16, mix(DIM, a), CJK, 'rm')

def wrap(c, s, size, maxw):
    lines, cur = [], ''
    for ch in s:
        if c.tw(cur + ch, size) > maxw and cur:
            lines.append(cur); cur = ch
        else:
            cur += ch
    if cur: lines.append(cur)
    # 避免标点出现在行首
    for k in range(1, len(lines)):
        if lines[k][0] in '，。！？、；：':
            lines[k - 1] += lines[k][0]; lines[k] = lines[k][1:]
    return [l for l in lines if l]

def subtitle(c, sc, t):
    for s0, s1, line in sc['segs']:
        if s0 - 0.05 <= t <= s1 + 0.35:
            lines = wrap(c, line, 28, W - 160)
            hgt = 40 * len(lines) + 16
            y0 = H - 22 - hgt
            c.rect(60, y0, W - 60, H - 22, fill=(0, 0, 0, 165), r=10)
            for k, l in enumerate(lines):
                c.text(W / 2, y0 + 8 + 20 + 40 * k, l, 28, (255, 255, 255), CJK, 'mm')
            return

def seg(sc, k): return sc['segs'][k][0]

def formula_box(c, x, y, parts, size=30, a=1.0, center=False):
    parts = [(s, mix(col, a, PANEL if False else BG), f) for s, col, f in parts]
    c.rich(x, y, parts, size, center)

# ---------------- 各场景 ----------------
def sc_title(c, sc, t):
    run = RUNS['pid']
    ts = (t * 0.9) % SIM_T if t < 99 else 0
    h, u = run_state(run, min(t * 1.2, SIM_T - 0.1))
    bob = math.sin(t * 2) * 0.12
    # 背景飘动的曲线
    for k in range(3):
        pts = [(x, 470 + 40 * math.sin(x / 90 + t * 0.8 + k * 2) * math.exp(-x / 900) + k * 30) for x in range(0, W + 10, 10)]
        c.line(pts, mix([CP, CI, CD][k], 0.25), 3)
    drone(c, W / 2, 200 - 60 * (1 - ease(t / 2.5)) + bob * 30, t, 12)
    a = fade(t, 0.3, 0.8)
    c.text(W / 2, 320, 'PID 控制算法', 78, mix(TXT, a), CJK, 'mm')
    a2 = fade(t, 1.0, 0.8)
    x = W / 2; size = 34
    parts = [('P', CP, LATB), (' 比例   ', TXT, CJK), ('I', CI, LATB), (' 积分   ', TXT, CJK), ('D', CD, LATB), (' 微分', TXT, CJK)]
    c.rich(x, 410, [(s, mix(col, a2), f) for s, col, f in parts], size, True)
    a3 = fade(t, seg(sc, 3), 0.6)
    c.text(W / 2, 470, '—— 几分钟看懂自动控制的核心 ——', 22, mix(DIM, a3), CJK, 'mm')

def box(c, x, y, w, h, label, col, a=1.0, sub=None):
    c.rect(x, y, x + w, y + h, fill=mix(PANEL, a), outline=mix(col, a), w=3, r=10)
    c.text(x + w / 2, y + h / 2 - (9 if sub else 0), label, 22, mix(TXT, a), CJK, 'mm')
    if sub: c.text(x + w / 2, y + h / 2 + 16, sub, 15, mix(DIM, a), CJK, 'mm')

def sc_loop(c, sc, t):
    run = RUNS['pid']
    t0 = seg(sc, 1)
    ts = max(0, (t - t0) * 0.55)
    h, u = run_state(run, ts) if t > t0 else (0.0, 0.0)
    world(c, h, u if t > t0 else None, t, show_err=t > seg(sc, 2) - 0.2)
    # 方框图
    a1 = fade(t, seg(sc, 2) - 0.2); a2 = fade(t, seg(sc, 3) - 0.2)
    y = 200; bh = 70
    c.text(470, y + bh / 2, '目标高度', 20, mix(TGT, a1), CJK, 'lm')
    c.text(470, y + bh / 2 + 24, 'r = 5m', 16, mix(TGT, a1), LAT, 'lm')
    sx = 620; sy = y + bh / 2
    c.arrow(560, sy, sx - 22, sy, mix(DIM, a1), 3)
    c.circle(sx, sy, 20, outline=mix(TXT, a1), w=3)
    c.text(sx, sy, 'Σ', 22, mix(TXT, a1), LAT, 'mm')
    c.text(sx - 30, sy - 26, '+', 20, mix(TXT, a1), LAT, 'mm')
    c.text(sx + 12, sy + 36, '−', 24, mix(TXT, a1), LAT, 'mm')
    c.arrow(sx + 20, sy, 720, sy, mix(RED, a1), 3)
    c.text(670, sy - 18, 'e', 22, mix(RED, a1), LATB, 'mm')
    box(c, 720, y, 160, bh, '控制器', CURVE, a2, '计算推力 u')
    c.arrow(880, sy, 950, sy, mix(DIM, a2), 3)
    c.text(915, sy - 16, 'u', 20, mix(CURVE, a2), LATB, 'mm')
    box(c, 950, y, 150, bh, '无人机', TXT, a2, '推力→高度')
    c.arrow(1100, sy, 1210, sy, mix(DIM, a2), 3)
    c.text(1160, sy - 18, '高度 y', 18, mix(TXT, a2), CJK, 'mm')
    # 反馈线
    fb = y + bh + 70
    c.line([(1180, sy), (1180, fb), (sx, fb)], mix(CI, a2), 3)
    c.arrow(sx, fb, sx, sy + 22, mix(CI, a2), 3)
    c.text(900, fb + 22, '传感器反馈（闭环）', 18, mix(CI, a2), CJK, 'mm')
    # 误差公式
    c.rich(850, 470, [('误差  ', mix(TXT, a1), CJK), ('e', mix(RED, a1), LATB), (' = ', mix(TXT, a1), LAT),
                      ('目标', mix(TGT, a1), CJK), (' − ', mix(TXT, a1), LAT), ('实际', mix(CURVE, a1), CJK)], 34, True)
    if t > t0:
        c.text(850, 520, f'当前：e = 5.00 - {h:.2f} = {5 - h:+.2f} m', 20, mix(DIM, a1), CJK, 'mm')

def sim_scene(c, sc, t, main, t_start, faded=(), spring=False, area=False, show_err=False):
    run = RUNS[main]
    ts = max(0.0, (t - t_start) * PLAY)
    h, u = run_state(run, ts) if t >= t_start else (0.0, 0.0)
    world(c, h, u if t >= t_start else None, t, spring=spring and t >= t_start, show_err=show_err)
    fx, fy = plot_frame(c)
    for name, col in faded:
        curve(c, fx, fy, RUNS[name], SIM_T, mix(col, 0.35), 2)
    if area and t >= t_start:
        err_area(c, fx, fy, run, ts, CI)
    return fx, fy, run, ts, h, u

def legend(c, x, y, items, size=17):
    for k, (name, col) in enumerate(items):
        c.line([(x, y + k * 26), (x + 30, y + k * 26)], col, 4)
        c.text(x + 40, y + k * 26, name, size, col, CJK, 'lm')

def sc_bang(c, sc, t):
    t0 = seg(sc, 0) + 0.5
    fx, fy, run, ts, h, u = sim_scene(c, sc, t, 'bang', t0)
    if t >= t0: curve(c, fx, fy, run, ts, RED, 3)
    a = fade(t, 0.3)
    c.rich(870, 140, [('u = ', mix(TXT, a), LAT), ('最大推力', mix(CURVE, a), CJK), ('   (e > 0，低了)', mix(DIM, a), CJK)], 26, True)
    c.rich(870, 182, [('u = ', mix(TXT, a), LAT), ('0', mix(CURVE, a), LATB), ('                (e < 0，高了)', mix(DIM, a), CJK)], 26, True)
    if t >= t0:
        on = u > 9
        c.rect(1110, 270, 1225, 305, fill=(CI if on else RED) + (60,), outline=CI if on else RED, w=2, r=8)
        c.text(1167, 288, '油门 开' if on else '油门 关', 18, TXT, CJK, 'mm')
    a3 = fade(t, seg(sc, 2) - 0.2)
    if a3 > 0:
        c.text(870, 222, '× 只看方向，不看大小', 22, mix(RED, a3), CJK, 'mm')

def sc_p(c, sc, t):
    t1 = seg(sc, 1); t5 = seg(sc, 5)
    second = t >= t5
    main, start = ('p12', t5) if second else ('p5', t1)
    faded = [('p5', CP)] if second else []
    fx, fy, run, ts, h, u = sim_scene(c, sc, t, main, start, faded, spring=True)
    if t >= start: curve(c, fx, fy, run, ts, CP, 3)
    a = fade(t, 0.3)
    c.rich(870, 150, [('u', mix(TXT, a), LATB), (' = ', mix(TXT, a), LAT), ('Kp', mix(CP, a), LATB), (' · e', mix(TXT, a), LAT)], 44, True)
    kp = 12 if second else 5
    c.text(870, 200, f'当前 Kp = {kp}', 20, mix(CP, a), CJK, 'mm')
    legend_items = [('Kp = 5', mix(CP, 0.35 if second else 1))] + ([('Kp = 12', CP)] if second else [])
    legend(c, 1060, 270, legend_items)
    # 稳态误差标注
    a3 = fade(t, seg(sc, 3) - 0.2) * (1 - fade(t, t5 - 0.3, 0.3))
    if a3 > 0:
        hf = 5 - 9.8 / 5
        x = fx(10.5)
        c.arrow(x, fy(hf) - 2, x, fy(5) + 6, mix(RED, a3), 3, 10)
        c.arrow(x, fy(5) + 2, x, fy(hf) - 6, mix(RED, a3), 3, 10)
        c.text(x - 10, (fy(hf) + fy(5)) / 2, '稳态误差 ≈ 2m', 20, mix(RED, a3), CJK, 'rm')
    a4 = fade(t, seg(sc, 4) - 0.2) * (1 - fade(t, t5 - 0.3, 0.3))
    if a4 > 0:
        c.text(870, 224, '平衡时：Kp · e = mg  →  e = mg / Kp ≠ 0', 20, mix(TXT, a4), CJK, 'mm')
    if second:
        a5 = fade(t, t5 + 3)
        c.text(870, 224, 'Kp 越大：误差越小，但振荡越厉害', 20, mix(TGT, a5), CJK, 'mm')

def sc_i(c, sc, t):
    t1 = seg(sc, 1)
    fx, fy, run, ts, h, u = sim_scene(c, sc, t, 'pi', t1, [('p5', CP)], area=True)
    if t >= t1: curve(c, fx, fy, run, ts, CI, 3)
    a = fade(t, 0.3)
    c.rich(870, 150, [('u = ', mix(TXT, a), LAT), ('Kp', mix(CP, a), LATB), ('·e + ', mix(TXT, a), LAT), ('Ki', mix(CI, a), LATB), ('·∫e dt', mix(TXT, a), LAT)], 40, True)
    c.text(870, 198, '积分 = 误差曲线下的面积（误差的累积）', 19, mix(CI, fade(t, t1)), CJK, 'mm')
    legend(c, 1040, 270, [('纯 P (Kp=5)', mix(CP, 0.35)), ('PI', CI)])
    # 积分值条
    if t >= t1:
        iv = run['i'][sim_idx(run, ts)]
        bx, by = 870, 232
        c.text(bx - 200, by, '积分累积', 17, CI, CJK, 'lm')
        c.rect(bx - 120, by - 8, bx + 180, by + 8, fill=GRID, r=4)
        fr = max(0, min(1, iv / 8))
        c.rect(bx - 120, by - 8, bx - 120 + 300 * fr, by + 8, fill=CI, r=4)
        c.text(bx + 190, by, f'{iv:.2f}', 16, CI, LAT, 'lm')
    a3 = fade(t, seg(sc, 3) - 0.2)
    if a3 > 0:
        r = RUNS['pi']; k = int(np.argmax(r['h'])); tp, hp = r['t'][k], r['h'][k]
        if ts >= tp:
            c.circle(fx(tp), fy(hp), 9, outline=mix(RED, a3), w=3)
            c.text(fx(tp) + 16, fy(hp) - 6, f'超调！最高 {hp:.1f}m', 19, mix(RED, a3), CJK, 'lm')

def sc_d(c, sc, t):
    t3 = seg(sc, 3)
    showing_pid = t >= t3
    if showing_pid:
        fx, fy, run, ts, h, u = sim_scene(c, sc, t, 'pid', t3, [('pi', CI)])
        curve(c, fx, fy, run, ts, CURVE, 4)
    else:
        # 前半段：讲解微分 = 斜率，用 PI 曲线演示刹车概念
        run = RUNS['pi']
        ts = min(SIM_T, max(0.0, t - seg(sc, 1)) * 0.9 + 1.2)
        h, u = run_state(run, ts)
        world(c, h, u, t, show_err=True)
        fx, fy = plot_frame(c)
        curve(c, fx, fy, run, ts, mix(CI, 0.8), 3)
        i = sim_idx(run, ts)
        slope = (run['h'][min(i + 20, len(run['h']) - 1)] - run['h'][max(i - 20, 0)]) / (40 * sim.DT)
        x, y = fx(ts), fy(min(run['h'][i], 10))
        dx = 55; dy = -slope * (fy(1) - fy(0)) / (fx(1) - fx(0)) * dx * -1
        dy = slope / 10 * (PY1 - PY0) / (SIM_T) * (fx(1) - fx(0)) / (fx(1) - fx(0))
        # 切线：单位换算
        k_pix = slope * ((PY1 - PY0) / 10) / ((PX1 - PX0) / SIM_T)
        if abs(k_pix) > 1e-6:
            dx = min(dx, (y - PY0) / abs(k_pix), (PY1 - y) / abs(k_pix))
        c.line([(x - dx, y + k_pix * dx), (x + dx, y - k_pix * dx)], CD, 3)
        c.text(x + dx + 8, y - k_pix * dx, f'速度 {slope:+.1f} m/s', 17, CD, CJK, 'lm')
        a2 = fade(t, seg(sc, 1))
        brake = -2.5 * slope
        c.text(870, 224, f'D 项输出 = -Kd·速度 = {brake:+.1f}  ' + ('（刹车）' if slope > 0.2 else ''), 19, mix(CD, a2), CJK, 'mm')
    a = fade(t, 0.3)
    c.rich(870, 150, [('u = ', mix(TXT, a), LAT), ('Kp', mix(CP, a), LATB), ('·e + ', mix(TXT, a), LAT), ('Ki', mix(CI, a), LATB), ('·∫e dt + ', mix(TXT, a), LAT), ('Kd', mix(CD, a), LATB), ('·de/dt', mix(TXT, a), LAT)], 36, True)
    c.text(870, 198, '微分 = 误差变化的快慢（曲线的斜率）', 19, mix(CD, fade(t, seg(sc, 1) - 0.2)), CJK, 'mm')
    if showing_pid:
        legend(c, 1040, 270, [('PI', mix(CI, 0.35)), ('PID', CURVE)])
        a4 = fade(t, t3 + 2.5)
        c.text(870, 224, '√ 几乎无超调，约 3 秒稳定在 5m', 20, mix(CI, a4), CJK, 'mm')

def sc_sum(c, sc, t):
    a = fade(t, 0.2)
    c.rich(W / 2, 140, [('u(t) = ', mix(TXT, a), LAT), ('Kp', mix(CP, a), LATB), ('·e(t)', mix(TXT, a), LAT), ('  +  ', mix(DIM, a), LAT),
                        ('Ki', mix(CI, a), LATB), ('·∫e(t)dt', mix(TXT, a), LAT), ('  +  ', mix(DIM, a), LAT),
                        ('Kd', mix(CD, a), LATB), ('·de(t)/dt', mix(TXT, a), LAT)], 42, True)
    t2 = seg(sc, 2) - 0.3
    cards_a = fade(t, seg(sc, 1) - 0.3) * (1 - fade(t, t2, 0.4))
    if cards_a > 0:
        info = [('P', '比例', '看现在', '误差有多大？', '出力快，单独用有稳态误差', CP),
                ('I', '积分', '看过去', '误差累积了多少？', '消除稳态误差，但会超调', CI),
                ('D', '微分', '看未来', '误差变化有多快？', '提前刹车，抑制振荡', CD)]
        for k, (L, n, when, q, eff, col) in enumerate(info):
            ak = cards_a * fade(t, seg(sc, 1) - 0.3 + k * 0.5)
            x = 120 + k * 360; y = 200
            c.rect(x, y, x + 320, y + 330, fill=mix(PANEL, ak), outline=mix(col, ak), w=3, r=16)
            c.text(x + 160, y + 70, L, 80, mix(col, ak), LATB, 'mm')
            c.text(x + 160, y + 140, n, 26, mix(TXT, ak), CJK, 'mm')
            c.text(x + 160, y + 195, when, 38, mix(col, ak), CJK, 'mm')
            c.text(x + 160, y + 250, q, 20, mix(TXT, ak), CJK, 'mm')
            c.text(x + 160, y + 290, eff, 17, mix(DIM, ak), CJK, 'mm')
    a3 = fade(t, t2 + 0.2, 0.5)
    if a3 > 0:
        box = (260, 220, 1100, 540)
        fx, fy = plot_frame(c, box)
        items = [('bang', '开关控制', RED), ('p5', '纯 P', CP), ('pi', 'PI', CI), ('pid', 'PID', CURVE)]
        for k, (name, lab, col) in enumerate(items):
            tk = (t - t2 - 0.5 - k * 0.6) * 3
            if tk > 0:
                curve(c, fx, fy, RUNS[name], tk, mix(col, a3), 4 if name == 'pid' else 2)
        legend(c, 930, 245, [(lab, col) for _, lab, col in items])

CODE = [
    ('# 每个控制周期执行一次（dt 为周期）', DIM),
    ('error = target - measured', TXT),
    ('integral += error * dt', CI),
    ('derivative = (error - prev_error) / dt', CD),
    ('output = Kp*error + Ki*integral + Kd*derivative', CP),
    ('prev_error = error', TXT),
]
def sc_code(c, sc, t):
    c.rect(40, 100, 760, 400, fill=(10, 14, 28), outline=GRID, w=2, r=14)
    for k, col in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        c.circle(66 + k * 22, 124, 7, fill=col)
    c.text(400, 124, 'pid.py', 16, DIM, LAT, 'mm')
    for k, (line, col) in enumerate(CODE):
        a = fade(t, 0.4 + k * 0.9 * (seg(sc, 0) + sc['segs'][0][1]) / 10, 0.4)
        ff = CJK if k == 0 else MONO
        c.text(66, 170 + k * 38, line, 19 if k else 18, mix(col, a), ff, 'lm')
    # 调参步骤
    t1 = seg(sc, 1)
    steps = [('1', '先调 P', '增大 Kp 直到响应够快、开始轻微振荡', CP),
             ('2', '再加 D', '增大 Kd 压住振荡和超调', CD),
             ('3', '最后加 I', '小步增加 Ki，消除残余误差', CI)]
    c.text(800, 112, '调参经验', 26, mix(TXT, fade(t, t1 - 0.3)), CJK, 'lm')
    for k, (n, title, desc, col) in enumerate(steps):
        a = fade(t, t1 + k * 2.2, 0.5)
        y = 160 + k * 80
        c.rect(800, y, 1240, y + 66, fill=mix(PANEL, a), outline=mix(col, a), w=2, r=10)
        c.circle(830, y + 33, 17, fill=mix(col, a))
        c.text(830, y + 33, n, 20, mix(BG, 1) if a > 0.5 else mix(BG, 1), LATB, 'mm')
        c.text(860, y + 20, title, 21, mix(col, a), CJK, 'lm')
        c.text(860, y + 47, desc, 16, mix(TXT, a), CJK, 'lm')
    t2 = seg(sc, 2)
    tips = [('积分饱和', '误差长期存在时积分无限增大 → 给积分限幅'), ('微分噪声', '传感器噪声被微分放大 → 对 D 项低通滤波')]
    for k, (a_, b_) in enumerate(tips):
        a = fade(t, t2 + k * 1.5, 0.5)
        y = 430 + k * 64
        c.rect(40, y, 1240, y + 50, fill=mix(PANEL, a), r=10)
        c.text(64, y + 25, '⚠', 22, mix(TGT, a), LAT, 'lm')
        c.text(96, y + 25, a_, 20, mix(TGT, a), CJK, 'lm')
        c.text(210, y + 25, b_, 18, mix(TXT, a), CJK, 'lm')

def sc_end(c, sc, t):
    apps = [('恒温热水器', '温度控制', CP), ('定速巡航', '车速控制', CI), ('机械臂', '位置/角度控制', CD),
            ('3D 打印机', '喷头温度', PURP), ('平衡车', '姿态平衡', TGT), ('火箭 / 无人机', '姿态控制', CURVE)]
    t2 = seg(sc, 2)
    fo = 1 - fade(t, t2 - 0.3, 0.5)
    for k, (n, d, col) in enumerate(apps):
        a = fade(t, 0.3 + k * 0.6, 0.4) * fo
        x = 110 + (k % 3) * 360; y = 120 + (k // 3) * 170
        c.rect(x, y, x + 330, y + 140, fill=mix(PANEL, a), outline=mix(col, a), w=3, r=14)
        c.text(x + 165, y + 55, n, 30, mix(col, a), CJK, 'mm')
        c.text(x + 165, y + 100, d, 18, mix(TXT, a), CJK, 'mm')
    a1 = fade(t, seg(sc, 1) - 0.2) * fo
    if a1 > 0:
        c.text(W / 2, 500, '1922 年 Minorsky 为船舶自动操舵提出 PID 理论；如今绝大多数工业控制回路仍在使用 PID', 18, mix(DIM, a1), CJK, 'mm')
    a2 = fade(t, t2 - 0.1, 0.6)
    if a2 > 0:
        drone(c, W / 2, 220 + math.sin(t * 2) * 6, t, 12)
        c.text(W / 2, 330, '感谢观看', 66, mix(TXT, a2), CJK, 'mm')
        parts = [('P', CP, LATB), (' 看现在 · ', TXT, CJK), ('I', CI, LATB), (' 看过去 · ', TXT, CJK), ('D', CD, LATB), (' 看未来', TXT, CJK)]
        c.rich(W / 2, 425, [(s, mix(col, a2), f) for s, col, f in parts], 30, True)
    # 结尾淡出
    c.rect(0, 0, W, H, fill=(0, 0, 0, int(255 * fade(t, sc['dur'] - 0.8, 0.8))))

DRAW = dict(title=sc_title, loop=sc_loop, bang=sc_bang, p=sc_p, i=sc_i, d=sc_d, sum=sc_sum, code=sc_code, end=sc_end)

def frame(n):
    T = n / FPS
    for sc in SC:
        if T < sc['start'] + sc['dur'] or sc is SC[-1]:
            break
    t = T - sc['start']
    c = Cv()
    DRAW[sc['id']](c, sc, t)
    header(c, sc, t)
    subtitle(c, sc, t)
    # 场景切换淡入
    if t < 0.35 and sc is not SC[0]:
        c.rect(0, 0, W, H, fill=(0, 0, 0, int(255 * (1 - t / 0.35))))
    # 进度条
    c.rect(0, H - 4, W * T / TOTAL, H, fill=CURVE)
    return c.out()

if __name__ == '__main__':
    # 预览指定时间点：python render.py 12.5 30 ...
    for s in sys.argv[1:]:
        frame(int(float(s) * FPS)).save(f'prev_{s}.png')
    print('total', TOTAL)
