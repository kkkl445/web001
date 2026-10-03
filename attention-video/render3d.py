"""A small software 3D renderer for the film's figures.

Models are written as signed distance functions (rounded primitives blended
with a smooth union), meshed with marching cubes, rasterised into a
z-buffered G-buffer (numba), and shaded in the series' palette: a cool
porcelain body, a warm key light, and a gold or cyan rim that glows.
Rendering is supersampled 2x inside the region the figures occupy."""

import math

import cv2
import numba
import numpy as np
from skimage.measure import marching_cubes

from style import E, GOLD, H, W, add_sprite, blur, glow_poly, mix, orb


# ------------------------------------------------------------- SDFs ---

def sd_ellipsoid(P, c, r):
    q = (P - c) / r
    k0 = np.linalg.norm(q, axis=1)
    k1 = np.linalg.norm(q / r, axis=1)
    return k0 * (k0 - 1.0) / np.maximum(k1, 1e-9)


def sd_round_cone(P, a, b, r1, r2):
    a, b = np.asarray(a, float), np.asarray(b, float)
    ba = b - a
    l2 = ba @ ba
    rr = r1 - r2
    a2 = l2 - rr * rr
    il2 = 1.0 / l2
    pa = P - a
    y = pa @ ba
    z = y - l2
    xv = pa * l2 - y[:, None] * ba
    x2 = np.einsum("ij,ij->i", xv, xv)
    y2 = y * y * l2
    z2 = z * z * l2
    k = np.sign(rr) * rr * rr * x2
    d_end = np.sqrt(x2 + z2) * il2 - r2
    d_start = np.sqrt(x2 + y2) * il2 - r1
    d_side = (np.sqrt(x2 * a2 * il2) + y * rr) * il2 - r1
    return np.where(np.sign(z) * a2 * z2 > k, d_end, np.where(np.sign(y) * a2 * y2 < k, d_start, d_side))


def sd_round_box(P, c, b, r):
    q = np.abs(P - c) - (np.asarray(b) - r)
    return np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0) - r


def smin(a, b, k):
    h = np.maximum(k - np.abs(a - b), 0.0) / k
    return np.minimum(a, b) - h * h * k * 0.25


def sdf_union(P, prims, k=0.05):
    d = None
    for kind, args in prims:
        di = {"ell": sd_ellipsoid, "cone": sd_round_cone, "box": sd_round_box}[kind](P, *args)
        d = di if d is None else smin(d, di, k)
    return d


def mesh(prims, lo, hi, step, k=0.05):
    """Marching-cubes mesh of a primitive blend; normals from the exact SDF gradient."""
    xs = [np.arange(lo[i], hi[i] + step, step) for i in range(3)]
    G = np.stack(np.meshgrid(*xs, indexing="ij"), -1).reshape(-1, 3)
    vol = sdf_union(G, prims, k).reshape(len(xs[0]), len(xs[1]), len(xs[2]))
    if vol.min() >= 0:
        return np.zeros((0, 3)), np.zeros((0, 3), np.int32), np.zeros((0, 3))
    v, f, _, _ = marching_cubes(vol, 0.0, spacing=(step, step, step))
    v = v + np.array(lo)
    e = step * 0.5
    n = np.stack([sdf_union(v + d, prims, k) - sdf_union(v - d, prims, k)
                  for d in (np.array([e, 0, 0]), np.array([0, e, 0]), np.array([0, 0, e]))], 1)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    return v, f.astype(np.int32), n


# ----------------------------------------------------------- camera ---

class Camera:
    def __init__(self, target, dist, yaw, pitch, focal, cx, cy):
        t = np.asarray(target, float)
        d = np.array([math.sin(yaw) * math.cos(pitch), math.sin(pitch), math.cos(yaw) * math.cos(pitch)])
        self.pos = t + d * dist
        f = t - self.pos
        f /= np.linalg.norm(f)
        r = np.cross(f, [0.0, 1.0, 0.0])
        r /= np.linalg.norm(r)
        u = np.cross(r, f)
        self.R = np.stack([r, u, f])
        self.focal, self.cx, self.cy = focal, cx, cy
        self.fwd = f

    def project(self, P):
        c = (np.asarray(P, float) - self.pos) @ self.R.T
        z = np.maximum(c[..., 2], 1e-6)
        return np.stack([self.cx + self.focal * c[..., 0] / z, self.cy - self.focal * c[..., 1] / z, z], -1)


# --------------------------------------------------------- raster ---

@numba.njit(cache=True, fastmath=True)
def _raster(px, py, pz, nrm, faces, oid, zbuf, nbuf, ibuf):
    Hh, Ww = zbuf.shape
    for f in range(faces.shape[0]):
        i0, i1, i2 = faces[f, 0], faces[f, 1], faces[f, 2]
        x0, y0, x1, y1, x2, y2 = px[i0], py[i0], px[i1], py[i1], px[i2], py[i2]
        area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
        if abs(area) < 1e-9:
            continue
        mnx = max(int(min(x0, min(x1, x2))), 0)
        mxx = min(int(max(x0, max(x1, x2))) + 1, Ww - 1)
        mny = max(int(min(y0, min(y1, y2))), 0)
        mxy = min(int(max(y0, max(y1, y2))) + 1, Hh - 1)
        for y in range(mny, mxy + 1):
            fy = y + 0.5
            for x in range(mnx, mxx + 1):
                fx = x + 0.5
                w0 = ((x1 - fx) * (y2 - fy) - (x2 - fx) * (y1 - fy)) / area
                w1 = ((x2 - fx) * (y0 - fy) - (x0 - fx) * (y2 - fy)) / area
                w2 = 1.0 - w0 - w1
                if w0 < 0 or w1 < 0 or w2 < 0:
                    continue
                z = w0 * pz[i0] + w1 * pz[i1] + w2 * pz[i2]
                if z < zbuf[y, x]:
                    zbuf[y, x] = z
                    for c in range(3):
                        nbuf[y, x, c] = w0 * nrm[i0, c] + w1 * nrm[i1, c] + w2 * nrm[i2, c]
                    ibuf[y, x] = oid


STEEL_ALB = np.array([0.20, 0.27, 0.38], np.float32)
GOLD_ALB = np.array([0.95, 0.72, 0.40], np.float32)
AMBIENT = np.array([0.035, 0.045, 0.08], np.float32)
KEY_DIR = np.array([-0.55, 0.75, 0.55]) / np.linalg.norm([-0.55, 0.75, 0.55])
KEY_COL = np.array([1.0, 0.92, 0.80], np.float32) * 1.05
FILL_DIR = np.array([0.8, 0.25, 0.35]) / np.linalg.norm([0.8, 0.25, 0.35])
FILL_COL = np.array([0.35, 0.45, 0.75], np.float32) * 0.3
BACK_DIR = np.array([0.25, 0.45, -1.0]) / np.linalg.norm([0.25, 0.45, -1.0])
CYAN = np.array([0.45, 0.84, 1.0], np.float32)


class Region:
    """Supersampled render target for the part of the frame the figures live in."""

    def __init__(self, x0, y0, x1, y1, ss=2):
        self.x0, self.y0, self.x1, self.y1, self.ss = x0, y0, x1, y1, ss
        self.w, self.h = (x1 - x0) * ss, (y1 - y0) * ss

    def render(self, F, cam, objects, glow_gain=0.45):
        """objects: list of (verts, faces, normals, glow 0..1, alpha).  Composites onto F; returns the
        z-buffer at frame resolution (for occlusion tests) in camera depth units."""
        ss = self.ss
        zbuf = np.full((self.h, self.w), np.inf, np.float32)
        nbuf = np.zeros((self.h, self.w, 3), np.float32)
        ibuf = np.full((self.h, self.w), -1, np.int16)
        for k, (v, f, n, g, a) in enumerate(objects):
            if len(f) == 0 or a <= 0.003:
                continue
            p = cam.project(v)
            px = ((p[:, 0] - self.x0) * ss).astype(np.float64)
            py = ((p[:, 1] - self.y0) * ss).astype(np.float64)
            _raster(px, py, p[:, 2].astype(np.float64), n.astype(np.float64), f, k, zbuf, nbuf, ibuf)
        cov = np.isfinite(zbuf)
        if not cov.any():
            return
        n = nbuf / (np.linalg.norm(nbuf, axis=2, keepdims=True) + 1e-9)
        view = -cam.fwd
        ndv = np.clip(n @ view, 0, 1)
        lam = np.clip(n @ KEY_DIR, 0, 1)
        fill = np.clip(n @ FILL_DIR, 0, 1)
        hv = (KEY_DIR + view) / np.linalg.norm(KEY_DIR + view)
        spec = np.clip(n @ hv, 0, 1) ** 60
        rim = (1 - ndv) ** 3.5 * (0.25 + 0.75 * np.clip(n @ BACK_DIR + 0.45, 0, 1))
        gl = np.zeros(ibuf.shape, np.float32)
        al = np.zeros(ibuf.shape, np.float32)
        for k, (_, _, _, g, a) in enumerate(objects):
            m = ibuf == k
            gl[m] = g
            al[m] = a
        alb = STEEL_ALB + (GOLD_ALB - STEEL_ALB) * gl[..., None]
        rim_col = CYAN + (np.asarray(GOLD, np.float32) - CYAN) * gl[..., None]
        col = (alb * (AMBIENT * 3 + lam[..., None] * KEY_COL + fill[..., None] * FILL_COL)
               + spec[..., None] * 0.5 + rim[..., None] * rim_col * (0.75 + 0.9 * gl[..., None]))
        col *= cov[..., None]
        alpha = cov.astype(np.float32) * al
        rim_l = rim[..., None] * rim_col * alpha[..., None]
        size = (self.x1 - self.x0, self.y1 - self.y0)
        col_s = cv2.resize(col * alpha[..., None], size, interpolation=cv2.INTER_AREA)
        a_s = cv2.resize(alpha, size, interpolation=cv2.INTER_AREA)
        rim_s = cv2.resize(rim_l, size, interpolation=cv2.INTER_AREA)
        reg = F[self.y0:self.y1, self.x0:self.x1]
        reg *= 1 - a_s[..., None]
        reg += col_s
        reg += blur(rim_s, 7) * glow_gain
        zs = cv2.resize(np.where(cov, zbuf, 1e9).astype(np.float32), size, interpolation=cv2.INTER_NEAREST)
        return zs


def contact_shadow(F, cam, cx, cz, rx, rz, strength=0.5, sigma=10):
    """Soft dark ellipse on the floor (y = 0) under an object."""
    th = np.linspace(0, 2 * math.pi, 48)
    P = np.stack([cx + rx * np.cos(th), np.zeros_like(th), cz + rz * np.sin(th)], 1)
    p = cam.project(P)[:, :2]
    x0, y0 = int(p[:, 0].min()) - 40, int(p[:, 1].min()) - 40
    x1, y1 = int(p[:, 0].max()) + 40, int(p[:, 1].max()) + 40
    m = np.zeros((y1 - y0, x1 - x0), np.float32)
    cv2.fillPoly(m, [np.round((p - [x0, y0]) * 16).astype(np.int32)], 1.0, cv2.LINE_AA, 4)
    m = cv2.GaussianBlur(m, (0, 0), sigma)
    xa, ya, xb, yb = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
    if xa < xb and ya < yb:
        F[ya:yb, xa:xb] *= (1 - strength * m[ya - y0:yb - y0, xa - x0:xb - x0])[..., None]


def floor_glow(F, cam, cx, cz, rx, rz, color, gain):
    """A faint pool of light on the floor so the figures stand somewhere."""
    th = np.linspace(0, 2 * math.pi, 64)
    P = np.stack([cx + rx * np.cos(th), np.zeros_like(th), cz + rz * np.sin(th)], 1)
    p = cam.project(P)[:, :2]
    x0, y0 = int(p[:, 0].min()) - 60, int(p[:, 1].min()) - 60
    x1, y1 = int(p[:, 0].max()) + 60, int(p[:, 1].max()) + 60
    m = np.zeros((y1 - y0, x1 - x0), np.float32)
    cv2.fillPoly(m, [np.round((p - [x0, y0]) * 16).astype(np.int32)], 1.0, cv2.LINE_AA, 4)
    m = cv2.GaussianBlur(m, (0, 0), 40)
    E.add_light(F, m, x0, y0, color, gain)
