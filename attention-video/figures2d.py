"""Constellation line-art: stars joined by fine glowing lines, used only where a
picture helps (the cat and the table, the telephone game, the library, the
chase, the stargazer).  Each figure returns screen points and edges; draw()
joins them one line at a time and turns them from cool steel to gold."""

import math

import numpy as np

from style import GOLD, IVORY, add_sprite, glow_poly, mix, orb, ramp

STEEL = mix(np.array([0.45, 0.84, 1.0], np.float32), IVORY, 0.45)


def draw(F, pts, edges, t=1e9, t0=0.0, dur=1e-3, glow=0.0, seed=0, a=1.0, extra=(), color=None, star=1.0,
         rings=()):
    """Stars joined one line at a time.  rings: (centre, radius) circles drawn as part of the figure."""
    if a <= 0.003:
        return
    col = color if color is not None else mix(STEEL, GOLD, glow)
    a_line = (0.35 + 0.65 * glow) * a
    n = len(edges) + len(rings)
    prog = ramp(t, t0, dur) * n
    lit = set()
    for k, (i, j) in enumerate(edges):
        f = min(1.0, max(0.0, prog - k))
        if f <= 0:
            continue
        p, q = pts[i], pts[j]
        glow_poly(F, [p, p + (q - p) * f], col, a_line, th=1, glow=0.5 + 0.8 * glow, sigma=3 + 2 * glow)
        lit.add(i)
        if f >= 1:
            lit.add(j)
    for k, (c, r) in enumerate(rings):
        f = min(1.0, max(0.0, prog - len(edges) - k))
        if f <= 0:
            continue
        th = np.linspace(-math.pi / 2, -math.pi / 2 + 2 * math.pi * f, max(3, int(40 * f)))
        glow_poly(F, np.column_stack([c[0] + r * np.cos(th), c[1] + r * np.sin(th)]), col, a_line, th=1,
                  glow=0.5 + 0.8 * glow, sigma=3 + 2 * glow)
    if prog >= n * 0.6:
        lit.update(extra)
    rng = np.random.default_rng(seed)
    ph = rng.uniform(0, 6.28, len(pts))
    for i in lit:
        tw = 0.8 + 0.2 * math.sin(t * 2.3 + ph[i])
        orb(F, pts[i][0], pts[i][1], (3.5 + 2.5 * glow) * star, col, (0.55 + 0.45 * glow) * tw * a)


def _place(units, x, ground, s, flip=False):
    u = np.asarray(units, float)
    if flip:
        u = u * np.array([-1.0, 1.0])
    return np.column_stack([x + u[:, 0] * s, ground + u[:, 1] * s])


# ------------------------------------------------------------------ cat ---

#          tail tip      tail base      hip           back          neck          head back     ear
CAT_SIT = [(-1.0, -0.15), (-0.55, -0.12), (-0.5, -0.45), (-0.25, -0.85), (0.12, -1.05), (0.12, -1.3), (0.15, -1.62),
           (0.28, -1.42), (0.42, -1.62), (0.5, -1.32), (0.58, -1.18), (0.38, -1.05), (0.32, -0.6), (0.32, 0.0),
           (-0.3, 0.0), (0.4, -1.24), (0.82, -1.24), (0.8, -1.1)]
#          ear dip       ear           forehead      nose          chin          chest         front paw
#          back paw      eye           whisker       whisker
CAT_LIE = [(-1.15, -0.05), (-0.7, -0.08), (-0.6, -0.32), (-0.2, -0.45), (0.2, -0.42), (0.25, -0.62), (0.3, -0.9),
           (0.42, -0.72), (0.55, -0.88), (0.62, -0.6), (0.72, -0.42), (0.5, -0.3), (0.35, -0.15), (0.85, 0.0),
           (-0.35, 0.0), (0.56, -0.56), (0.96, -0.5), (0.94, -0.36)]
CAT_RUN_A = [(-1.3, -0.8), (-0.85, -0.6), (-0.7, -0.52), (-0.25, -0.66), (0.22, -0.62), (0.32, -0.8), (0.35, -1.1),
             (0.47, -0.92), (0.6, -1.08), (0.68, -0.84), (0.8, -0.7), (0.58, -0.6), (0.33, -0.38), (0.8, -0.06),
             (-1.05, -0.06), (0.58, -0.78), (0.98, -0.76), (0.96, -0.62)]
CAT_RUN_B = [(-1.2, -0.95), (-0.75, -0.62), (-0.6, -0.55), (-0.2, -0.76), (0.2, -0.68), (0.3, -0.86), (0.33, -1.16),
             (0.45, -0.98), (0.58, -1.14), (0.66, -0.9), (0.78, -0.76), (0.56, -0.66), (0.25, -0.4), (0.05, 0.0),
             (-0.3, 0.0), (0.56, -0.84), (0.96, -0.82), (0.94, -0.68)]
CAT_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 4),
             (11, 12), (12, 13), (2, 14), (12, 2), (10, 16), (10, 17)]
CAT_EYE = 15


def cat(x, ground, s, down=0.0, hop=0.0, look_up=0.0, run=None, flip=False):
    """Sitting -> lying blend, a hop, a lifted head; or a running stride (run = phase in radians)."""
    if run is not None:
        u = 0.5 + 0.5 * math.sin(run)
        p = np.array(CAT_RUN_A) + (np.array(CAT_RUN_B) - np.array(CAT_RUN_A)) * u
        p = p + np.array([0.0, -0.12 * abs(math.cos(run))])
    else:
        p = np.array(CAT_SIT) + (np.array(CAT_LIE) - np.array(CAT_SIT)) * down
        p = p + np.array([0.25, -0.75]) * hop
        head = list(range(5, 12)) + [15, 16, 17]
        p[head] = p[head] + np.array([0.0, -0.08]) * look_up
    return _place(p, x, ground, s, flip), CAT_EDGES, [CAT_EYE]


# ------------------------------------------------------------------ dog ---

#          tail tip      tail base     hip           back          neck          head back     ear top
DOG_A = [(-1.3, -1.25), (-0.95, -0.85), (-0.85, -0.66), (-0.25, -0.76), (0.35, -0.82), (0.45, -1.08), (0.42, -1.16),
         (0.28, -0.86), (0.66, -1.12), (1.0, -1.0), (1.1, -0.92), (0.78, -0.84), (0.45, -0.48), (0.95, -0.06),
         (-1.2, -0.06), (-0.25, -0.42), (0.68, -1.04)]
#          ear tip       forehead      snout top     nose          jaw           chest         front paw
#          back paw      belly         eye
DOG_B = [(-1.2, -1.35), (-0.9, -0.9), (-0.8, -0.72), (-0.22, -0.86), (0.35, -0.9), (0.45, -1.16), (0.42, -1.24),
         (0.3, -0.94), (0.66, -1.2), (1.0, -1.08), (1.1, -1.0), (0.78, -0.92), (0.4, -0.52), (0.15, 0.0),
         (-0.4, 0.0), (-0.22, -0.48), (0.68, -1.12)]
DOG_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (5, 8), (8, 9), (9, 10), (10, 11), (11, 4),
             (4, 12), (12, 13), (2, 14), (12, 15), (15, 2)]
DOG_EYE = 16


def dog(x, ground, s, run=0.0, flip=False):
    u = 0.5 + 0.5 * math.sin(run)
    p = np.array(DOG_A) + (np.array(DOG_B) - np.array(DOG_A)) * u
    p = p + np.array([0.0, -0.1 * abs(math.cos(run))])
    return _place(p, x, ground, s, flip), DOG_EDGES, [DOG_EYE]


# ---------------------------------------------------------------- table ---

TAB_EDGES = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 5), (2, 6), (3, 7)]


def table(x, ground, s, h=1.5):
    top = np.array([(0, -h), (2.6, -h), (3.0, -h - 0.35), (0.4, -h - 0.35)])
    feet = np.array([(0, 0), (2.6, 0), (3.0, -0.35), (0.4, -0.35)])
    return _place(np.vstack([top, feet]), x, ground, s), TAB_EDGES


# --------------------------------------------------------------- person ---

PERSON_EDGES = [(0, 1), (0, 2), (1, 3), (3, 4), (2, 5), (5, 6), (0, 7), (7, 8), (8, 9), (8, 10), (9, 11), (11, 12),
                (10, 13), (13, 14)]


def person(x, ground, s, arm_l=(0.12, -0.08), arm_r=(0.12, -0.08), leg_l=(0.07, -0.03), leg_r=(0.07, -0.03),
           lean=0.0, head_up=0.0, flip=False):
    """A standing figure seen from the front.  Limb angles are (upper, lower) in radians from straight
    down, positive outwards (left limbs swing to -x, right limbs to +x).  Returns points, edges and the
    head ring (centre, radius)."""
    def limb(root, a1, a2, l1, l2, sgn=1.0):
        e = root + np.array([sgn * math.sin(a1), math.cos(a1)]) * l1
        h = e + np.array([sgn * math.sin(a1 + a2), math.cos(a1 + a2)]) * l2
        return e, h

    hip = np.array([0.0, -0.9])
    up = np.array([math.sin(lean), -math.cos(lean)])
    side = np.array([math.cos(lean), math.sin(lean)])
    neck = hip + up * 0.52
    chest = hip + up * 0.3
    sh_l, sh_r = neck - side * 0.17 + up * -0.05, neck + side * 0.17 + up * -0.05
    head = neck + up * 0.2 + np.array([0.05 * head_up, -0.02 * head_up])
    el_l, ha_l = limb(sh_l, arm_l[0] - lean, arm_l[1], 0.3, 0.28, -1.0)
    el_r, ha_r = limb(sh_r, arm_r[0] + lean, arm_r[1], 0.3, 0.28)
    hp_l, hp_r = hip - np.array([0.1, 0.0]), hip + np.array([0.1, 0.0])
    kn_l, ft_l = limb(hp_l, leg_l[0], leg_l[1], 0.45, 0.43, -1.0)
    kn_r, ft_r = limb(hp_r, leg_r[0], leg_r[1], 0.45, 0.43)
    p = np.array([neck, sh_l, sh_r, el_l, ha_l, el_r, ha_r, chest, hip, hp_l, hp_r, kn_l, ft_l, kn_r, ft_r])
    p[:, 1] -= max(ft_l[1], ft_r[1])                      # stand on the ground
    head = head - np.array([0.0, max(ft_l[1], ft_r[1])])
    pts = _place(p, x, ground, s, flip)
    hc = _place([head], x, ground, s, flip)[0]
    return pts, PERSON_EDGES, (hc, 0.12 * s)


HAND_L, HAND_R = 4, 6


# ---------------------------------------------------------- bookshelf ---

def shelf(x0, y0, w, h, books):
    """Shelf frame plus book spines.  books: list of (width, height); returns points, edges, spine boxes."""
    pts = [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    boxes = []
    gap = (w - sum(b[0] for b in books)) / (len(books) + 1)
    x = x0 + gap
    for bw, bh in books:
        k = len(pts)
        pts += [(x, y0 + h), (x + bw, y0 + h), (x + bw, y0 + h - bh), (x, y0 + h - bh)]
        edges += [(k, k + 1), (k + 1, k + 2), (k + 2, k + 3), (k + 3, k)]
        boxes.append((x, y0 + h - bh, bw, bh))
        x += bw + gap
    return np.array(pts, float), edges, boxes


def iso_box(cx, cy, w, h, d=0.5):
    """A block in isometric line-art: front face, top and right side.  (cx, cy) is the front-bottom centre."""
    dx, dy = w * d * 0.5, -w * d * 0.28
    x0, x1 = cx - w / 2, cx + w / 2
    p = [(x0, cy), (x1, cy), (x1, cy - h), (x0, cy - h), (x0 + dx, cy - h + dy), (x1 + dx, cy - h + dy),
         (x1 + dx, cy + dy)]
    e = [(0, 1), (1, 2), (2, 3), (3, 0), (3, 4), (4, 5), (5, 2), (5, 6), (6, 1)]
    return np.array(p, float), e


def phone(cx, cy, w, h):
    r = w * 0.12
    pts, edges = [], []
    corners = [(cx + w / 2 - r, cy - h / 2 + r, -90), (cx + w / 2 - r, cy + h / 2 - r, 0),
               (cx - w / 2 + r, cy + h / 2 - r, 90), (cx - w / 2 + r, cy - h / 2 + r, 180)]
    for ccx, ccy, a0 in corners:
        for k in range(4):
            a = math.radians(a0 + 90 * k / 3)
            pts.append((ccx + r * math.cos(a), ccy + r * math.sin(a)))
    n = len(pts)
    edges = [(i, (i + 1) % n) for i in range(n)]
    return np.array(pts, float), edges


def page(x0, y0, w, h, fold=40):
    p = [(x0, y0), (x0 + w - fold, y0), (x0 + w, y0 + fold), (x0 + w, y0 + h), (x0, y0 + h),
         (x0 + w - fold, y0 + fold)]
    e = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 5), (5, 2)]
    return np.array(p, float), e


def record(F, cx, cy, r, t, a, glow=0.0):
    """A spinning record: grooves as faint rings, a label, and one bright glint going round."""
    col = mix(STEEL, GOLD, glow)
    for k, rr in enumerate(np.linspace(r * 0.38, r, 9)):
        th = np.linspace(0, 2 * math.pi, 120)
        glow_poly(F, np.column_stack([cx + rr * np.cos(th), cy + rr * np.sin(th) * 0.32]), col,
                  a * (0.55 if k in (0, 8) else 0.22), th=1, glow=0.4, sigma=3)
    ang = t * 2.2
    for k in range(3):
        rr = r * (0.55 + 0.15 * k)
        add_sprite(F, cx + rr * math.cos(ang + k), cy + rr * math.sin(ang + k) * 0.32, 2.2, GOLD, 0.8 * a)
    orb(F, cx, cy, 5, GOLD, a)


# ---------------------------------------------------- figure silhouettes ---
# People are drawn as a single fine contour (like a pen drawing) traced around a
# body built from tapered limbs, with a few stars at the joints - the way old
# star atlases drew their figures - instead of stick bones.

import cv2  # noqa: E402

from style import E as _E, W as _W, H as _H  # noqa: E402

_SS = 3            # supersampling of the silhouette mask


def _pose_joints(arm_l, arm_r, leg_l, leg_r, lean, head_up, head_turn):
    """Joint positions (units, y up negative, standing height ~1.75)."""
    def seg(root, a, l, sgn):
        return root + np.array([sgn * math.sin(a), math.cos(a)]) * l

    hip = np.array([0.0, -0.9])
    up = np.array([math.sin(lean), -math.cos(lean)])
    side = np.array([math.cos(lean), math.sin(lean)])
    waist = hip + up * 0.18
    neck = hip + up * 0.5
    J = {"hip": hip, "waist": waist, "neck": neck}
    J["sh_l"], J["sh_r"] = neck - side * 0.155 - up * 0.04, neck + side * 0.155 - up * 0.04
    J["hp_l"], J["hp_r"] = hip - side * 0.085, hip + side * 0.085
    J["head"] = neck + up * 0.17 + np.array([0.035 * head_turn, -0.01]) + np.array([0.03, 0.0]) * head_up
    for nm, (a1, a2), sgn, l1, l2 in (("l", arm_l, -1, 0.29, 0.26), ("r", arm_r, 1, 0.29, 0.26)):
        J["el_" + nm] = seg(J["sh_" + nm], a1 + sgn * lean, l1, sgn)
        J["ha_" + nm] = seg(J["el_" + nm], a1 + a2 + sgn * lean, l2, sgn)
    for nm, (a1, a2), sgn in (("l", leg_l, -1), ("r", leg_r, 1)):
        J["kn_" + nm] = seg(J["hp_" + nm], a1, 0.43, sgn)
        J["ft_" + nm] = seg(J["kn_" + nm], a1 + a2, 0.42, sgn)
    lift = max(J["ft_l"][1], J["ft_r"][1])
    for k in J:
        J[k] = J[k] - np.array([0.0, lift])
    return J, up, side


def _contours(mask, x0, y0):
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = []
    for c in cs:
        if len(c) < 12:
            continue
        p = c[:, 0, :].astype(np.float64)
        k = 7                                          # smooth the traced outline into a pen line
        pad = np.vstack([p[-k:], p, p[:k]])
        ker = np.ones(2 * k + 1) / (2 * k + 1)
        sm = np.column_stack([np.convolve(pad[:, 0], ker, "same"), np.convolve(pad[:, 1], ker, "same")])[k:-k]
        out.append(sm[::2] / _SS + np.array([x0, y0]))
    return out


def silhouette(x, ground, s, arm_l=(0.14, -0.1), arm_r=(0.14, -0.1), leg_l=(0.06, -0.02), leg_r=(0.06, -0.02),
               lean=0.0, head_up=0.0, head_turn=0.0, flip=False):
    """A person as pen contours: body (head, torso, legs) and each arm traced separately so an arm
    in front of the body still reads.  Returns (contours, joints in screen coords, head centre)."""
    J, up, side = _pose_joints(arm_l, arm_r, leg_l, leg_r, lean, head_up, head_turn)
    sx = -1.0 if flip else 1.0
    to = lambda p: np.array([x + sx * p[0] * s, ground + p[1] * s])          # noqa: E731
    P = {k: to(v) for k, v in J.items()}
    allp = np.array(list(P.values()))
    x0, y0 = int(allp[:, 0].min() - 0.35 * s), int(allp[:, 1].min() - 0.3 * s)
    x1, y1 = int(allp[:, 0].max() + 0.35 * s), int(allp[:, 1].max() + 0.15 * s)
    shape = ((y1 - y0) * _SS, (x1 - x0) * _SS)

    def q(p):
        return tuple(np.round((p - [x0, y0]) * _SS * 16).astype(int))

    def limb(m, a, b, w0, w1):
        """Tapered limb as a filled quad plus round ends."""
        a, b = P[a], P[b]
        d = (b - a) / (np.linalg.norm(b - a) + 1e-9)
        n = np.array([-d[1], d[0]])
        r0, r1 = w0 * s / 2, w1 * s / 2
        poly = np.array([a + n * r0, b + n * r1, b - n * r1, a - n * r0])
        cv2.fillPoly(m, [np.round((poly - [x0, y0]) * _SS * 16).astype(np.int32)], 255, cv2.LINE_AA, 4)
        cv2.circle(m, q(a), int(r0 * _SS * 16), 255, -1, cv2.LINE_AA, 4)
        cv2.circle(m, q(b), int(r1 * _SS * 16), 255, -1, cv2.LINE_AA, 4)

    body = np.zeros(shape, np.uint8)
    sl, sr, hl, hr = P["sh_l"], P["sh_r"], P["hp_l"], P["hp_r"]
    across = (sr - sl) / (np.linalg.norm(sr - sl) + 1e-9)
    down = np.array([-across[1], across[0]]) * (1 if across[0] >= 0 else -1)
    if down[1] < 0:
        down = -down
    w_sh = np.linalg.norm(sr - sl) / 2
    mid = (sl + sr) / 2
    waist = P["waist"]
    hem = P["hip"] + down * 0.34 * s
    # a long coat: soft shoulders, a little waist, the hem flaring over the thighs
    coat = np.array([mid - across * w_sh * 0.55 - down * 0.02 * s, sl + down * 0.03 * s,
                     waist - across * w_sh * 0.74, hem - across * w_sh * 1.12, hem + across * w_sh * 1.12,
                     waist + across * w_sh * 0.74, sr + down * 0.03 * s, mid + across * w_sh * 0.55 - down * 0.02 * s])
    cv2.fillPoly(body, [np.round((coat - [x0, y0]) * _SS * 16).astype(np.int32)], 255, cv2.LINE_AA, 4)
    for p_ in (sl, sr):
        cv2.circle(body, q(p_ + down * 0.035 * s), int(0.05 * s * _SS * 16), 255, -1, cv2.LINE_AA, 4)
    limb(body, "neck", "head", 0.06, 0.06)
    hc = P["head"]
    cv2.ellipse(body, q(hc), (int(0.088 * s * _SS * 16), int(0.112 * s * _SS * 16)), 0, 0, 360, 255, -1,
                cv2.LINE_AA, 4)
    for nm in ("l", "r"):
        limb(body, "hp_" + nm, "kn_" + nm, 0.12, 0.085)
        limb(body, "kn_" + nm, "ft_" + nm, 0.085, 0.055)
        f = P["ft_" + nm]
        cv2.ellipse(body, q(f + np.array([sx * 0.035 * s, -0.012 * s])), (int(0.065 * s * _SS * 16),
                    int(0.024 * s * _SS * 16)), 0, 0, 360, 255, -1, cv2.LINE_AA, 4)
        limb(body, "sh_" + nm, "el_" + nm, 0.08, 0.065)
        limb(body, "el_" + nm, "ha_" + nm, 0.065, 0.045)
        cv2.circle(body, q(P["ha_" + nm]), int(0.03 * s * _SS * 16), 255, -1, cv2.LINE_AA, 4)
    arms = []
    cont = _contours(body, x0, y0)
    return cont, P, hc, (body, arms, x0, y0)


def draw_silhouette(F, sil, t=1e9, t0=0.0, dur=1e-3, glow=0.0, a=1.0, stars=("head", "ha_r"),
                    fill=0.05, seed=0):
    """Fine pen contour drawn on progressively, a breath of light inside, and a few stars at the joints."""
    if a <= 0.003:
        return
    cont, P, hc, (body, arms, x0, y0) = sil
    col = mix(STEEL, GOLD, glow)
    u = ramp(t, t0, dur)
    if fill > 0 and u > 0:
        m = body.astype(np.float32)
        for am in arms:
            m = np.maximum(m, am.astype(np.float32))
        m = cv2.resize(m, (m.shape[1] // _SS, m.shape[0] // _SS), interpolation=cv2.INTER_AREA) / 255
        _E.add_light(F, cv2.GaussianBlur(m, (0, 0), 6), x0, y0, col, fill * a * u * (1 + 1.5 * glow))
    for c in cont:
        k = max(2, int(len(c) * u))
        glow_poly(F, c[:k], col, (0.55 + 0.45 * glow) * a, th=1, glow=0.55 + 0.8 * glow, sigma=3 + 2 * glow)
    if u >= 0.7:
        rng = np.random.default_rng(seed)
        sa = a * min(1.0, (u - 0.7) / 0.3)
        for nm in stars:
            p = hc if nm == "head" else P[nm]
            orb(F, p[0], p[1], 2.6 + rng.uniform(0, 1.0) + 2 * glow, col, (0.55 + 0.45 * glow) * sa)


# ------------------------------------------------------------ profiles ---
# People appear only as profile busts: one smooth pen line from the back of the
# shoulder over the head, down the face and throat to the chest, an ear, and a
# single star for the eye.  Units: head height ~1, facing +x.

PROFILE = [(-0.62, 1.02), (-0.30, 0.66), (-0.27, 0.42), (-0.36, 0.16), (-0.33, -0.12), (-0.18, -0.36),
           (0.06, -0.46), (0.27, -0.33), (0.35, -0.13), (0.355, -0.04), (0.345, 0.03), (0.43, 0.165),
           (0.355, 0.205), (0.365, 0.265), (0.335, 0.30), (0.36, 0.345), (0.32, 0.425), (0.21, 0.49),
           (0.13, 0.55), (0.14, 0.76), (0.42, 0.98)]
EAR = [(-0.10, -0.01), (-0.045, -0.05), (-0.02, 0.03), (-0.045, 0.12), (-0.09, 0.13)]
EYE = (0.235, -0.05)
MOUTH = (0.38, 0.30)
EAR_C = (-0.06, 0.04)
HAIR = {
    "bun": [(-0.30, -0.30), (-0.52, -0.38), (-0.6, -0.2), (-0.5, -0.05), (-0.34, -0.06)],
    "short": [(-0.33, -0.12), (-0.24, -0.38), (0.0, -0.5), (0.22, -0.42), (0.3, -0.3)],
    "long": [(-0.18, -0.36), (-0.38, -0.08), (-0.42, 0.3), (-0.36, 0.62)],
    None: [],
}


def catmull(pts, n=10, closed=False):
    P = np.asarray(pts, float)
    if closed:
        P = np.vstack([P[-1], P, P[0], P[1]])
    else:
        P = np.vstack([P[0] * 2 - P[1], P, P[-1] * 2 - P[-2]])
    out = []
    for i in range(1, len(P) - 2):
        p0, p1, p2, p3 = P[i - 1], P[i], P[i + 1], P[i + 2]
        for u in np.linspace(0, 1, n, endpoint=False):
            u2, u3 = u * u, u * u * u
            out.append(0.5 * (2 * p1 + (-p0 + p2) * u + (2 * p0 - 5 * p1 + 4 * p2 - p3) * u2
                              + (-p0 + 3 * p1 - 3 * p2 + p3) * u3))
    out.append(P[-2])
    return np.array(out)


def profile(x, y, s, tilt=0.0, flip=False, hair="short"):
    """Screen-space curves of a profile bust anchored at the ear; tilt > 0 lifts the face upwards."""
    pivot = np.array([0.0, 0.52])                      # the head nods about the top of the neck

    def tf(pts):
        p = np.asarray(pts, float)
        if tilt:
            w = np.clip((0.62 - p[:, 1]) / 0.3, 0, 1)[:, None]
            a = -tilt * w
            d = p - pivot
            p = pivot + np.column_stack([d[:, 0] * np.cos(a[:, 0]) - d[:, 1] * np.sin(a[:, 0]),
                                         d[:, 0] * np.sin(a[:, 0]) + d[:, 1] * np.cos(a[:, 0])])
        p = p - np.array(EAR_C)
        if flip:
            p[:, 0] = -p[:, 0]
        return p * s + np.array([x, y])

    out = {"outline": tf(catmull(PROFILE, 12)), "ear": tf(catmull(EAR, 8)),
           "hair": tf(catmull(HAIR[hair], 10)) if HAIR[hair] else None,
           "eye": tf([EYE])[0], "mouth": tf([MOUTH])[0], "ear_pt": tf([EAR_C])[0]}
    return out


def draw_profile(F, pr, t=1e9, t0=0.0, dur=1e-3, glow=0.0, a=1.0, fill=0.06, eye=True):
    """Pen line drawn on from the shoulder over the head, then the ear and hair; a star for the eye."""
    if a <= 0.003:
        return
    col = mix(STEEL, GOLD, glow)
    u = ramp(t, t0, dur)
    if u <= 0:
        return
    line = pr["outline"]
    if fill > 0:
        poly = np.vstack([line, [line[-1] + [0, 40], line[0] + [0, 40]]])
        x0, y0 = int(poly[:, 0].min()) - 20, int(poly[:, 1].min()) - 20
        x1, y1 = int(poly[:, 0].max()) + 20, int(poly[:, 1].max()) + 20
        m = np.zeros((y1 - y0, x1 - x0), np.float32)
        cv2.fillPoly(m, [np.round((poly - [x0, y0]) * 16).astype(np.int32)], 1.0, cv2.LINE_AA, 4)
        fade = np.clip((np.arange(y1 - y0)[:, None] + y0 - poly[:, 1].min()) / max(1, (y1 - y0)), 0, 1)
        m *= (1 - 0.8 * fade).astype(np.float32)
        _E.add_light(F, cv2.GaussianBlur(m, (0, 0), 5), x0, y0, col, fill * a * u * (1 + 1.2 * glow))
    k = max(2, int(len(line) * min(1.0, u * 1.25)))
    glow_poly(F, line[:k], col, (0.6 + 0.4 * glow) * a, th=1, glow=0.6 + 0.8 * glow, sigma=3 + 2 * glow)
    v = ramp(u, 0.6, 0.4)
    if v > 0:
        e = pr["ear"]
        glow_poly(F, e[:max(2, int(len(e) * v))], col, 0.5 * a, th=1, glow=0.4, sigma=2.5)
        if pr["hair"] is not None:
            hcv = pr["hair"]
            glow_poly(F, hcv[:max(2, int(len(hcv) * v))], col, 0.45 * a, th=1, glow=0.4, sigma=2.5)
    if eye and u >= 0.8:
        orb(F, pr["eye"][0], pr["eye"][1], 2.6 + 1.5 * glow, col, (0.7 + 0.3 * glow) * a * ramp(u, 0.8, 0.2))
