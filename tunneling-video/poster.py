"""Cover art for 量子隧穿 · 南墙之外.

The hero image is the film's own 2D simulation, caught just after the wave
has hit the wall: the blue part that turned back, the gold part that got
through.  Rendered with the same engine and palette as the film.

  python3 poster.py   -> poster-3x4.png (1080x1440), cover-4x3.png (1440x1080)   wave at the wall
                         poster-wall-3x4.png, poster-wall-4x3.png             the wall at sunrise
"""

import math

import cv2
import numpy as np

import physics as P
import style as S
from style import E

FRAME = 240                      # reflected and transmitted packets both clear of the wall


def canvas(w, h):
    """Point the engine at a w x h canvas and return a starfield background + vignette."""
    E.W, E.H, S.W, S.H = w, h, w, h
    rng = np.random.default_rng(4)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    F = np.empty((h, w, 3), np.float32)
    F[:] = S.rgb("030407")
    d = ((xx - w / 2) / (0.6 * w)) ** 2 + ((yy - 0.45 * h) / (0.6 * h)) ** 2
    F += (S.rgb("0b1322") - S.rgb("030407")) * np.exp(-d * 1.4)[..., None]
    stars = np.zeros((h, w), np.float32)
    n = int(720 * w * h / (1920 * 864))
    for x, y, b in zip(rng.uniform(0, w, n), rng.uniform(0, h, n), rng.power(4, n) * 0.22 + 0.02):
        cv2.circle(stars, (int(x * 16), int(y * 16)), int(rng.uniform(0.5, 1.2) * 16), float(b), -1, cv2.LINE_AA, 4)
    F += stars[..., None] * np.array([0.85, 0.9, 1.0], np.float32)
    r = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2)
    vig = (1 - 0.55 * np.clip((r - 0.5) / 0.9, 0, 1) ** 1.5).astype(np.float32)
    return F, vig


def finish(F, vig, path):
    h, w = F.shape[:2]
    small = cv2.resize(F, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    br = np.maximum(small - 0.42, 0)
    b = cv2.GaussianBlur(br, (0, 0), 2.5) * 0.55 + cv2.GaussianBlur(br, (0, 0), 11) * 0.75
    F += cv2.resize(b, (w, h), interpolation=cv2.INTER_LINEAR)
    F *= vig[..., None]
    F += np.random.default_rng(8).normal(0, 0.006, (h, w))[..., None].astype(np.float32)
    knee = 0.8
    hi = F > knee
    F[hi] = knee + (1 - knee) * (1 - np.exp(-(F[hi] - knee) / (1 - knee)))
    out = (np.clip(F, 0, 1) * 255 + 0.5).astype(np.uint8)
    cv2.imwrite(str(S.ROOT / path), out[..., ::-1])
    print("wrote", path, out.shape[1], "x", out.shape[0])


def palette(v, stops, cols):
    v = np.clip(v, 0, 1)
    return np.stack([np.interp(v, stops, [c[i] for c in cols]) for i in range(3)], -1).astype(np.float32)


COOL = ([0, 0.18, 0.45, 0.75, 1.0],
        [(0, 0, 0), (0.02, 0.05, 0.11), (0.1, 0.3, 0.52), (0.48, 0.76, 0.92), (0.96, 0.97, 0.95)])
WARM = ([0, 0.16, 0.42, 0.72, 1.0],
        [(0, 0, 0), (0.18, 0.07, 0.01), (0.7, 0.4, 0.1), (1.0, 0.8, 0.45), (1.0, 0.97, 0.9)])


def sim_image(x0, x1, y0, y1, boost=1.0):
    """Coloured simulation crop (cells x0..x1 along the motion, y0..y1 across), motion along +x."""
    D = P.packet2d()
    p = D["prob"][FRAME].astype(np.float32)
    r = D["re"][FRAME].astype(np.float32)
    p0 = float(D["prob"][0].astype(np.float32).max())
    v = np.clip((0.55 * p + 0.45 * r * r) / p0, 0, None)
    x = np.arange(P.NX)
    gain = np.where(x >= P.BAR_X0 + P.BAR_W, boost, 1.0)[None, :]          # lift the faint gold side
    v = (v * gain) ** 0.62
    side = np.clip((x - (P.BAR_X0 + P.BAR_W)) / 6.0, 0, 1)[None, :, None].astype(np.float32)
    col = palette(v, *COOL) * (1 - side) + palette(v, *WARM) * side
    return col[y0:y1, x0:x1]


def glass(F, a, b, across0, across1, vertical, scale):
    """The wall: a dark glass slab with glowing edges, fading out at its ends."""
    n = int(across1 - across0)
    prof = (np.clip(np.arange(n) / (0.22 * n), 0, 1) * np.clip((n - np.arange(n)) / (0.22 * n), 0, 1)) ** 1.5
    prof = prof.astype(np.float32)
    th = int(b - a)
    slab = np.repeat(prof[:, None], th, 1)
    edge = np.zeros((n, th + 40), np.float32)
    edge[:, 20] = prof
    edge[:, 20 + th] = prof
    glow = cv2.GaussianBlur(edge, (0, 0), 0.6 * scale)
    core = cv2.GaussianBlur(edge, (0, 0), 0.6)
    if not vertical:                       # wall runs horizontally across the frame
        slab, glow, core = slab.T, glow.T, core.T
        S.blend(F, slab, across0, a, np.array([0.03, 0.05, 0.09], np.float32), 0.9)
        S.add_light(F, glow, across0, a - 20, S.CYAN, 3.2)
        S.add_light(F, core, across0, a - 20, S.mix(S.CYAN, S.WHITE, 0.4), 1.3)
    else:
        S.blend(F, slab, a, across0, np.array([0.03, 0.05, 0.09], np.float32), 0.9)
        S.add_light(F, glow, a - 20, across0, S.CYAN, 3.2)
        S.add_light(F, core, a - 20, across0, S.mix(S.CYAN, S.WHITE, 0.4), 1.3)


def readout(F, x, y, value, zh, en, color, align="left"):
    S.T(value, "stix", 64, None, per_char=False).draw(F, x, y, color, 1.0, align=align)
    zl = S.serif(zh, 18, 500, 0.3)
    el = S.caps(en, 11, 0.4)
    w = zl.width + 12 + el.width
    x0 = x if align == "left" else x - w
    zl.draw(F, x0, y + 34, color, 0.9)
    el.draw(F, x0 + zl.width + 12, y + 33, S.DIM, 0.9)


def title_block(F, cx, y, size=150):
    t = S.serif("量子隧穿", size, 500, 0.32)
    t.draw(F, cx, y, S.IVORY, 1.0, align="center", glow=0.35, glow_color=S.GOLD, glow_sigma=18)
    S.serif("南墙之外", int(size * 0.27), 500, 0.6).draw(F, cx, y + size * 0.62, S.GOLD, 1.0, align="center")
    k = size / 150
    S.hairline(F, cx - 70 * k, int(y + size * 0.78), cx + 70 * k, S.GOLD, 0.6)
    S.caps("QUANTUM TUNNELING   ·   BEYOND THE WALL", int(13 * k + 1), 0.42).draw(
        F, cx, y + size * 0.98, S.DIM, 1.0, align="center")


def portrait():
    w, h = 1080, 1440
    F, vig = canvas(w, h)
    sc = 4.6                                       # px per simulation cell
    wall_y = 840
    x0, x1 = 140, 380                              # cells along the motion (wall at 254..259)
    y0, y1 = 11, 245
    img = sim_image(x0, x1, y0, y1, boost=1.35)
    img = np.rot90(img, 1)                         # motion now points up the page: beyond the wall is above
    ih, iw = int(round(img.shape[0] * sc)), int(round(img.shape[1] * sc))
    img = cv2.resize(np.ascontiguousarray(img), (iw, ih), interpolation=cv2.INTER_CUBIC)
    top = wall_y - int(round((x1 - P.BAR_X0) * sc))
    left = (w - iw) // 2
    S.E.add_rgb(F, img * 0.95, left, top, 1.0)
    glass(F, wall_y - P.BAR_W * sc, wall_y, 0, w, vertical=False, scale=sc)
    D = P.packet2d()
    readout(F, 96, 560, f"{round(100 * float(D['T']))}%", "穿过", "GOT THROUGH", S.GOLD)
    readout(F, w - 96, 1120, f"{round(100 * float(D['R']))}%", "回头", "TURNED BACK", S.CYAN, align="right")
    # fade the top of the frame into darkness so the title sits on black
    yy = np.arange(h, dtype=np.float32)
    shade = np.clip((yy - 250) / 260, 0, 1) ** 1.2 * 0.75 + 0.25
    shade = np.where(yy > 510, 1.0, shade)
    F *= shade[:, None, None]
    title_block(F, w / 2, 230, 150)
    S.serif("撞了南墙，它却不一定回头。", 34, 400, 0.18).draw(F, w / 2, 1318, S.IVORY, 0.95, align="center")
    S.italic("Hit the wall, and it may not turn back.", 22).draw(F, w / 2, 1360, S.DIM, 1.0, align="center")
    finish(F, vig, "poster-3x4.png")


def wide():
    """4:3 cover: title across the top, the wave meeting the wall below it."""
    w, h = 1440, 1080
    F, vig = canvas(w, h)
    sc = 6.2
    wall_x, mid_y = 712, 700
    x0, x1 = 140, 380
    y0, y1 = 11, 245
    img = sim_image(x0, x1, y0, y1, boost=1.35)
    ih, iw = int(round(img.shape[0] * sc)), int(round(img.shape[1] * sc))
    img = cv2.resize(img, (iw, ih), interpolation=cv2.INTER_CUBIC)
    left = wall_x - int(round((P.BAR_X0 - x0) * sc))
    top = mid_y - int(round((P.PY0 - y0) * sc))
    S.E.add_rgb(F, img * 0.95, left, top, 1.0)
    glass(F, wall_x, wall_x + P.BAR_W * sc, 330, 1000, vertical=True, scale=sc)
    yy = np.arange(h, dtype=np.float32)                            # the title sits on black
    shade = np.where(yy > 470, 1.0, 0.25 + 0.75 * np.clip((yy - 330) / 140, 0, 1) ** 1.2)
    F *= shade[:, None, None]
    D = P.packet2d()
    readout(F, 90, 470, f"{round(100 * float(D['R']))}%", "回头", "TURNED BACK", S.CYAN)
    readout(F, w - 90, 470, f"{round(100 * float(D['T']))}%", "穿过", "GOT THROUGH", S.GOLD, align="right")
    title_block(F, w / 2, 210, 136)
    S.serif("撞了南墙，它却不一定回头。", 30, 400, 0.18).draw(F, w / 2, 1032, S.IVORY, 0.95, align="center")
    finish(F, vig, "cover-4x3.png")


# ------------------------------------------------- the wall at sunrise ---

WARM_WHITE = np.array([1.0, 0.93, 0.8], np.float32)


def _noise1d(n, sigma, rng):
    v = cv2.GaussianBlur(rng.normal(0, 1, (1, n + int(8 * sigma))).astype(np.float32), (0, 0), sigma)[0]
    v = v[int(4 * sigma):int(4 * sigma) + n]
    return v / (np.abs(v).max() + 1e-9)


def ridge(w, base, amp, seed, rough=1.0):
    """Mountain crest: broad swells plus finer, sharper detail."""
    rng = np.random.default_rng(seed)
    n = w + 40
    x = np.arange(n, dtype=np.float32) - 20
    y = (0.62 * _noise1d(n, w * 0.09, rng) + 0.26 * _noise1d(n, w * 0.03, rng) * rough
         + 0.12 * np.abs(_noise1d(n, w * 0.008, rng)) * rough)
    return np.column_stack([x, base - amp * y])


def _weighted_line(F, pts, color, gain, weight_fn, th=1, sigma=4.0, glow=1.0):
    """Polyline whose brightness varies along x (rim light strongest towards the sun)."""
    h, w = F.shape[:2]
    y0 = max(0, int(pts[:, 1].min()) - 30)
    y1 = min(h, int(pts[:, 1].max()) + 30)
    st = S.Stroke(0, y0, w, y1)
    st.poly(pts, th=th)
    m = st.m.astype(np.float32) / 255 * weight_fn(np.arange(w, dtype=np.float32))[None, :]
    S.add_light(F, cv2.GaussianBlur(m, (0, 0), sigma), 0, y0, color, gain * glow)
    S.add_light(F, m, 0, y0, S.mix(color, S.WHITE, 0.3), gain)


def _fill_below(F, pts, color, a=1.0):
    h, w = F.shape[:2]
    y0 = max(0, int(pts[:, 1].min()) - 2)
    st = S.Stroke(0, y0, w, h)
    st.fill(np.vstack([pts, [[w + 20, h + 5], [-20, h + 5]]]))
    S.blend(F, st.m, 0, y0, color, a)


def _crenellate(xs, ys, merlon, depth):
    """Top edge of a wall: square teeth every `merlon` px."""
    out = []
    for x, y in zip(xs, ys):
        up = int(x // merlon) % 2 == 0
        out.append((x, y - (depth if up else 0)))
    return np.array(out, np.float32)


def great_wall(F, crest, x_from, x_to, height, merlon, rim_fn, gain, body, towers=(), tower_w=34, tower_h=44):
    """The wall riding a crest: dark body, glowing crenellated top, watchtowers at chosen x."""
    xs = np.arange(x_from, x_to, 2.0, dtype=np.float32)
    base = np.interp(xs, crest[:, 0], crest[:, 1]) + height * 0.25
    top = base - height
    poly = np.vstack([np.column_stack([xs, top - 0.6 * height * 0.4]), np.column_stack([xs[::-1], base[::-1] + 2])])
    st = S.Stroke(0, max(0, int(top.min()) - 20), F.shape[1], F.shape[0])
    st.fill(poly)
    S.blend(F, st.m, 0, st.y0, body, 1.0)
    cren = _crenellate(xs, top, merlon, height * 0.35)
    _weighted_line(F, cren, S.GOLD, gain, rim_fn, th=1, sigma=3.0, glow=1.4)
    _weighted_line(F, np.column_stack([xs, base]), S.GOLD, 0.25 * gain, rim_fn, th=1, sigma=2.0, glow=0.6)
    for tx in towers:
        span = (crest[:, 0] > tx - tower_w / 2) & (crest[:, 0] < tx + tower_w / 2)
        tb = float(crest[span, 1].max()) + height * 0.25
        tt = tb - tower_h
        q = tower_w / 6
        outline = [(tx - tower_w / 2, tb), (tx - tower_w / 2, tt)]
        for k in range(6):
            xa = tx - tower_w / 2 + k * q
            hgt = tt - (tower_h * 0.12 if k % 2 == 0 else 0)
            outline += [(xa, hgt), (xa + q, hgt)]
        outline += [(tx + tower_w / 2, tt), (tx + tower_w / 2, tb)]
        body_poly = np.array(outline, np.float32)
        stt = S.Stroke(tx - tower_w, tt - 20, tx + tower_w, tb + 4)
        stt.fill(body_poly)
        S.blend(F, stt.m, stt.x0, stt.y0, body, 1.0)
        _weighted_line(F, body_poly, S.GOLD, gain * 1.1, rim_fn, th=1, sigma=3.0, glow=1.4)
        dw, dh = tower_w * 0.22, tower_h * 0.34                    # an arched doorway
        door = [(tx - dw / 2, tb - 2), (tx - dw / 2, tb - dh)]
        for k in range(9):
            a = math.pi * k / 8
            door.append((tx - dw / 2 * math.cos(a), tb - dh - dw / 2 * math.sin(a)))
        door += [(tx + dw / 2, tb - dh), (tx + dw / 2, tb - 2)]
        _weighted_line(F, np.array(door, np.float32), S.GOLD, 0.6 * gain, rim_fn, th=1, sigma=2.0, glow=0.8)


def dawn(w, h, sun, ridges, walls, title_y, title_size, tag_y, path):
    """Mountains at sunrise and the old wall along the ridges; the sun is the thing beyond the wall."""
    F, vig = canvas(w, h)
    sx, sy, sr = sun
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # dawn sky: warm around the sun, violet along the horizon, stars only high up
    G = np.exp(-((xx - sx) / (0.42 * w)) ** 2 - ((yy - sy) / (0.30 * h)) ** 2)
    band = np.exp(-((yy - sy) / (0.16 * h)) ** 2)
    F *= (1 - 0.85 * np.clip(G * 1.4 + band * 0.6, 0, 1))[..., None]
    F += (G ** 1.3)[..., None] * np.array([0.62, 0.32, 0.12], np.float32) * 0.85
    F += (band * (1 - G))[..., None] * np.array([0.10, 0.07, 0.13], np.float32)
    F += np.exp(-((xx - sx) ** 2 + (yy - sy) ** 2) / (2 * (0.08 * w) ** 2))[..., None] * np.array(
        [1.0, 0.62, 0.28], np.float32) * 0.45
    # rays climbing out from behind the mountains
    ang = np.arctan2(yy - sy, xx - sx)
    rr = np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2)
    rng = np.random.default_rng(61)
    ray = np.zeros_like(ang)
    for k in range(7):
        f = rng.uniform(9, 40)
        ray += np.maximum(np.sin(ang * f + rng.uniform(0, 6.28)), 0) ** 6 * rng.uniform(0.4, 1.0)
    above = np.clip((sy + 30 - yy) / 90, 0, 1)                    # fade out below the horizon, no hard edge
    F += (ray / 7 * np.exp(-rr / (0.55 * w)) * above)[..., None] * np.array([1.0, 0.75, 0.45], np.float32) * 0.35
    # the sun itself, sitting just above the far crest
    d = np.sqrt((xx - sx) ** 2 + (yy - sy) ** 2)
    disc = np.clip((sr - d) / 2.0, 0, 1)
    F += disc[..., None] * WARM_WHITE * 1.4
    S.add_sprite(F, sx, sy, sr * 1.6, S.GOLD, 0.9)
    S.add_sprite(F, sx, sy, sr * 4.0, S.mix(S.GOLD, S.AMBER, 0.5), 0.35)

    def rim(scale):
        return lambda x: 0.12 + 0.88 * np.exp(-((x - sx) / (scale * w)) ** 2)

    # ridges from far to near: hazy and sunlit at the back, nearly black at the front
    for i, (base, amp, seed, rough, col, rim_gain, mist) in enumerate(ridges):
        crest = ridge(w, base, amp, seed, rough)
        _fill_below(F, crest, np.array(col, np.float32))
        if mist > 0:                                       # haze settling at the foot of the next range
            foot = crest[:, 1].mean() + amp * 0.9
            haze = np.exp(-((yy - foot) / (0.05 * h)) ** 2) * (0.25 + 0.75 * np.exp(-((xx - sx) / (0.5 * w)) ** 2))
            F += haze[..., None] * np.array([0.42, 0.30, 0.26], np.float32) * mist
        _weighted_line(F, crest, S.mix(S.GOLD, S.IVORY, 0.25), rim_gain, rim(0.22), th=1, sigma=4.0)
        for wl in walls:
            if wl["on"] == i:
                great_wall(F, crest, wl["x0"], wl["x1"], wl["height"], wl["merlon"], rim(wl.get("rim", 0.35)),
                           wl["gain"], np.array(col, np.float32) * 0.9, wl.get("towers", ()), wl.get("tw", 34),
                           wl.get("th", 44))
    # motes of light drifting through the wall towards us
    rng = np.random.default_rng(44)
    for _ in range(70):
        x = rng.normal(sx, 0.22 * w)
        y = rng.uniform(sy - 0.05 * h, sy + 0.28 * h)
        S.add_sprite(F, x, y, rng.uniform(1.0, 2.2), S.GOLD, rng.uniform(0.25, 0.9) * math.exp(-((x - sx) / (0.3 * w)) ** 2))
    title_block(F, w / 2, title_y, title_size)
    S.serif("宇宙里，没有绝对的南墙。", int(title_size * 0.22), 400, 0.2).draw(F, w / 2, tag_y, S.IVORY, 0.95,
                                                                      align="center")
    S.italic("In this universe, no wall is absolute.", int(title_size * 0.15)).draw(
        F, w / 2, tag_y + title_size * 0.29, S.DIM, 1.0, align="center")
    finish(F, vig, path)


def dawn_portrait():
    w, h = 1080, 1440
    dawn(w, h, sun=(650, 880, 34),
         ridges=[(890, 70, 3, 0.8, (0.20, 0.13, 0.15), 0.35, 0.5),
                 (960, 110, 7, 1.0, (0.085, 0.07, 0.095), 0.5, 0.45),
                 (1070, 150, 11, 1.0, (0.035, 0.035, 0.05), 0.85, 0.3),
                 (1290, 120, 19, 1.0, (0.012, 0.014, 0.022), 0.25, 0.0)],
         walls=[dict(on=1, x0=60, x1=1020, height=7, merlon=4, gain=0.55, rim=0.3, towers=(330, 860), tw=14, th=18),
                dict(on=2, x0=-20, x1=1100, height=16, merlon=7, gain=1.1, rim=0.4, towers=(190, 560, 930),
                     tw=34, th=46)],
         title_y=250, title_size=150, tag_y=1330, path="poster-wall-3x4.png")


def dawn_landscape():
    w, h = 1440, 1080
    dawn(w, h, sun=(860, 690, 32),
         ridges=[(700, 60, 23, 0.8, (0.20, 0.13, 0.15), 0.35, 0.5),
                 (760, 90, 29, 1.0, (0.085, 0.07, 0.095), 0.5, 0.45),
                 (850, 120, 31, 1.0, (0.035, 0.035, 0.05), 0.85, 0.3),
                 (1010, 90, 37, 1.0, (0.012, 0.014, 0.022), 0.25, 0.0)],
         walls=[dict(on=1, x0=80, x1=1380, height=6, merlon=4, gain=0.55, rim=0.3, towers=(420, 1130), tw=13, th=17),
                dict(on=2, x0=-20, x1=1460, height=15, merlon=7, gain=1.1, rim=0.4, towers=(230, 700, 1220),
                     tw=32, th=42)],
         title_y=215, title_size=132, tag_y=975, path="poster-wall-4x3.png")


if __name__ == "__main__":
    portrait()
    wide()
    dawn_portrait()
    dawn_landscape()
