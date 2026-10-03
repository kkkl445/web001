"""Constellation figures with depth: stars scattered over a 3D body and joined by
fine lines (a plexus), drawn in perspective.  Lines near the silhouette and on
the near side shine brightest, the far side fades, so the figure reads as a
solid made of starlight rather than a flat outline.

The cat's stars are sampled once on its sitting pose and carried by the body
part they sit on, so the net keeps its shape while the cat hops, lies down
and looks up."""

import math

import cv2
import numpy as np

import figures as FG
from render3d import Camera, sd_ellipsoid, sd_round_cone
from style import E, GOLD, H, W, add_sprite, blur, mix

STEEL = mix(np.array([0.45, 0.84, 1.0], np.float32), np.array([0.925, 0.902, 0.847], np.float32), 0.45)


def _fps(P, n, seed=0):
    """Farthest-point sampling: n well-spread points from P."""
    rng = np.random.default_rng(seed)
    idx = [int(rng.integers(len(P)))]
    d = np.linalg.norm(P - P[idx[0]], axis=1)
    for _ in range(n - 1):
        i = int(np.argmax(d))
        idx.append(i)
        d = np.minimum(d, np.linalg.norm(P - P[i], axis=1))
    return np.array(idx)


def _knn_edges(P, k, max_len):
    D = np.linalg.norm(P[:, None] - P[None], axis=2)
    np.fill_diagonal(D, np.inf)
    edges = set()
    for i in range(len(P)):
        for j in np.argsort(D[i])[:k]:
            if D[i, j] < max_len:
                edges.add((min(i, j), max(i, j)))
    return np.array(sorted(edges))


def _rot_between(a, b):
    a, b = a / np.linalg.norm(a), b / np.linalg.norm(b)
    v = np.cross(a, b)
    c = float(a @ b)
    if np.linalg.norm(v) < 1e-9:
        return np.eye(3)
    vx = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    return np.eye(3) + vx + vx @ vx * (1 / (1 + c))


class WireCat:
    def __init__(self, n=210, k=4, seed=3, rest_down=0.0):
        v, f, nrm, _ = FG.cat_mesh(down=rest_down, step=0.016)
        idx = _fps(v, n, seed)
        self.p0, self.n0 = v[idx], nrm[idx]
        self.prims0, _ = FG.cat_pose(down=rest_down)
        d = np.stack([self._sd(kind, args, self.p0) for kind, args in self.prims0], 1)
        self.owner = np.argmin(np.abs(d), axis=1)
        self.edges = _knn_edges(self.p0, k, 0.24)
        self.rest = np.linalg.norm(self.p0[self.edges[:, 0]] - self.p0[self.edges[:, 1]], axis=1)
        self.local = [self._to_local(i) for i in range(len(self.prims0))]

    @staticmethod
    def _sd(kind, args, P):
        return sd_ellipsoid(P, *args) if kind == "ell" else sd_round_cone(P, *args)

    def _to_local(self, i):
        kind, args = self.prims0[i]
        m = self.owner == i
        P, N = self.p0[m], self.n0[m]
        if kind == "ell":
            c, r = args
            return m, ("ell", (P - c) / r, N * r)
        a, b, r1, r2 = args
        L = np.linalg.norm(b - a)
        d = (b - a) / L
        u = np.clip((P - a) @ d / L, 0, 1)
        radial = P - (a + (u * L)[:, None] * d)
        rad = r1 + (r2 - r1) * u
        return m, ("cone", u, radial / rad[:, None], N)

    def pose(self, down=0.0, hop=0.0, look_up=0.0, origin=(0.0, 0.0, 0.0)):
        """World positions and normals of every star, plus the eye positions."""
        prims, eyes = FG.cat_pose(down, hop, look_up, origin)
        P = np.zeros_like(self.p0)
        N = np.zeros_like(self.n0)
        for i, (kind, args) in enumerate(prims):
            m, loc = self.local[i]
            if not m.any():
                continue
            if kind == "ell":
                c, r = args
                P[m] = c + loc[1] * r
                n = loc[2] / r
            else:
                a, b, r1, r2 = args
                a0, b0 = self.prims0[i][1][0], self.prims0[i][1][1]
                R = _rot_between(b0 - a0, b - a)
                L = np.linalg.norm(b - a)
                d = (b - a) / L
                u = loc[1]
                rad = r1 + (r2 - r1) * u
                P[m] = a + (u * L)[:, None] * d + (loc[2] @ R.T) * rad[:, None]
                n = loc[3] @ R.T
            N[m] = n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-9)
        return P, N, eyes


def draw_wire(F, cam, P, N, edges, glow, a, star=1.0, line=1.0, reveal=1.0, seed=0, rest=None):
    """Plexus render: brightness from silhouette (normal across the view) and near/far side."""
    if a <= 0.003 or len(P) == 0:
        return
    pr = cam.project(P)
    view = -cam.fwd
    if N is None:
        facing = np.ones(len(P))
        rim = np.full(len(P), 0.65)
    else:
        ndv = N @ view
        facing = np.clip(ndv * 2 + 0.6, 0.18, 1.0)              # far side dims
        rim = (1 - np.abs(ndv)) ** 1.5
    b = (0.35 + 0.65 * rim) * facing
    ew = np.ones(len(edges))
    if rest is not None:                                         # lines stretched by the pose fade away
        cur = np.linalg.norm(P[edges[:, 0]] - P[edges[:, 1]], axis=1)
        ew = np.clip((1.6 - cur / rest) / 0.35, 0, 1)
        keep = ew > 0.02
        edges, ew = edges[keep], ew[keep]
    if reveal < 1:                                               # stars light up one by one, then join
        rank = np.random.default_rng(seed).permutation(len(P)) / len(P)
        on = np.clip((reveal * 1.25 - rank) / 0.25, 0, 1)
        b = b * on
        vis = (on[edges[:, 0]] > 0.5) & (on[edges[:, 1]] > 0.5)
        edges, ew = edges[vis], ew[vis]
    col = mix(STEEL, np.asarray(GOLD, np.float32), glow)
    x0, y0 = int(pr[:, 0].min()) - 30, int(pr[:, 1].min()) - 30
    x1, y1 = int(pr[:, 0].max()) + 31, int(pr[:, 1].max()) + 31
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return
    m = np.zeros((y1 - y0, x1 - x0), np.uint8)
    q = np.round((pr[:, :2] - [x0, y0]) * 16).astype(np.int32)
    order = np.argsort(b[edges[:, 0]] + b[edges[:, 1]])           # brightest drawn last
    for e in order:
        i, j = edges[e]
        v = int(255 * min(1.0, 0.5 * (b[i] + b[j])) * ew[e])
        cv2.line(m, tuple(q[i]), tuple(q[j]), v, 1, cv2.LINE_AA, 4)
    mf = m.astype(np.float32) / 255
    gain = (0.55 + 0.45 * glow) * a * line
    E.add_light(F, blur(mf, 3.0 + 2 * glow), x0, y0, col, gain * (0.6 + 0.9 * glow))
    E.add_light(F, mf, x0, y0, mix(col, np.ones(3, np.float32), 0.3), gain)
    for k in np.argsort(-pr[:, 2]):                              # stars, far to near
        s = b[k]
        add_sprite(F, pr[k, 0], pr[k, 1], 1.0 + 1.4 * s, mix(col, np.ones(3, np.float32), 0.5), 0.9 * s * a * star)
        if s > 0.6:
            add_sprite(F, pr[k, 0], pr[k, 1], 4 + 3 * glow, col, 0.18 * s * a * star)


def table_wire(h, x0=0.0, width=2.0, depth=1.0, slab=0.08, leg=0.075):
    """Points and segments of a table drawn as a lattice of stars."""
    xs = [x0, x0 + width]
    zs = [-depth / 2, depth / 2]
    pts, edges = [], []

    def add(p):
        pts.append(p)
        return len(pts) - 1

    top = {(i, j, k): add((xs[i], h - slab * k, zs[j])) for i in (0, 1) for j in (0, 1) for k in (0, 1)}
    for k in (0, 1):
        edges += [(top[0, 0, k], top[1, 0, k]), (top[1, 0, k], top[1, 1, k]), (top[1, 1, k], top[0, 1, k]),
                  (top[0, 1, k], top[0, 0, k])]
    edges += [(top[i, j, 0], top[i, j, 1]) for i in (0, 1) for j in (0, 1)]
    for t in (0.25, 0.5, 0.75):                                   # a few lines across the top
        a = add((x0 + width * t, h, zs[0]))
        b = add((x0 + width * t, h, zs[1]))
        edges.append((a, b))
    a = add((x0, h, 0.0))
    b = add((x0 + width, h, 0.0))
    edges.append((a, b))
    for lx in (x0 + 0.1, x0 + width - 0.1):
        for lz in (zs[0] + 0.1, zs[1] - 0.1):
            for dx, dz in ((-leg / 2, -leg / 2), (leg / 2, leg / 2)):
                a = add((lx + dx, h - slab, lz + dz))
                b = add((lx + dx, 0.0, lz + dz))
                edges.append((a, b))
    return np.array(pts, float), np.array(edges)


def floor_stars(F, cam, a, x=(-2.6, 3.4), z=(-1.6, 1.6), step=0.3):
    """A perspective grid of faint stars on the floor: the ground the figures stand on."""
    if a <= 0.003:
        return
    gx, gz = np.meshgrid(np.arange(x[0], x[1] + 1e-6, step), np.arange(z[0], z[1] + 1e-6, step))
    P = np.stack([gx.ravel(), np.zeros(gx.size), gz.ravel()], 1)
    p = cam.project(P)
    fade = np.exp(-((P[:, 0] - 0.4) / 2.6) ** 2 - (P[:, 2] / 1.5) ** 2)
    for (sx, sy, _), f in zip(p, fade):
        add_sprite(F, sx, sy, 1.1, STEEL, 0.3 * f * a)
