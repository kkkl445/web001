"""3D figures for the film, modelled as blends of rounded primitives.

The cat faces +x.  Poses share one primitive list so any two can be blended:
SIT, LIE (tired, head on paws), plus a hop and a head that tilts up."""

import math

import numpy as np

from render3d import mesh

# name: (kind, params...) ; ellipsoid = centre, radii ; cone = a, b, r1, r2
CAT_SIT = {
    "body": ("ell", (-0.05, 0.42, 0.0), (0.26, 0.34, 0.21)),
    "haunch_l": ("ell", (-0.14, 0.20, 0.12), (0.20, 0.17, 0.10)),
    "haunch_r": ("ell", (-0.14, 0.20, -0.12), (0.20, 0.17, 0.10)),
    "chest": ("ell", (0.10, 0.48, 0.0), (0.17, 0.22, 0.16)),
    "head": ("ell", (0.14, 0.88, 0.0), (0.19, 0.165, 0.18)),
    "muzzle": ("ell", (0.30, 0.83, 0.0), (0.08, 0.065, 0.085)),
    "ear_l": ("cone", (0.10, 0.98, 0.09), (0.07, 1.15, 0.12), 0.06, 0.012),
    "ear_r": ("cone", (0.10, 0.98, -0.09), (0.07, 1.15, -0.12), 0.06, 0.012),
    "fleg_l": ("cone", (0.14, 0.42, 0.08), (0.17, 0.05, 0.08), 0.055, 0.045),
    "fleg_r": ("cone", (0.14, 0.42, -0.08), (0.17, 0.05, -0.08), 0.055, 0.045),
    "paw_l": ("ell", (0.20, 0.035, 0.08), (0.07, 0.035, 0.05)),
    "paw_r": ("ell", (0.20, 0.035, -0.08), (0.07, 0.035, 0.05)),
    "tail1": ("cone", (-0.28, 0.12, 0.0), (-0.38, 0.06, 0.18), 0.05, 0.04),
    "tail2": ("cone", (-0.38, 0.06, 0.18), (-0.10, 0.035, 0.30), 0.04, 0.032),
    "tail3": ("cone", (-0.10, 0.035, 0.30), (0.15, 0.03, 0.24), 0.032, 0.024),
}
CAT_LIE = {
    "body": ("ell", (-0.15, 0.20, 0.0), (0.42, 0.19, 0.22)),
    "haunch_l": ("ell", (-0.42, 0.17, 0.10), (0.20, 0.15, 0.12)),
    "haunch_r": ("ell", (-0.42, 0.17, -0.10), (0.20, 0.15, 0.12)),
    "chest": ("ell", (0.15, 0.20, 0.0), (0.20, 0.17, 0.18)),
    "head": ("ell", (0.42, 0.24, 0.0), (0.19, 0.155, 0.18)),
    "muzzle": ("ell", (0.58, 0.19, 0.0), (0.08, 0.065, 0.085)),
    "ear_l": ("cone", (0.38, 0.34, 0.09), (0.33, 0.50, 0.13), 0.06, 0.012),
    "ear_r": ("cone", (0.38, 0.34, -0.09), (0.33, 0.50, -0.13), 0.06, 0.012),
    "fleg_l": ("cone", (0.22, 0.12, 0.08), (0.62, 0.045, 0.10), 0.05, 0.042),
    "fleg_r": ("cone", (0.22, 0.12, -0.08), (0.62, 0.045, -0.10), 0.05, 0.042),
    "paw_l": ("ell", (0.66, 0.035, 0.10), (0.07, 0.035, 0.05)),
    "paw_r": ("ell", (0.66, 0.035, -0.10), (0.07, 0.035, 0.05)),
    "tail1": ("cone", (-0.60, 0.10, 0.0), (-0.85, 0.05, 0.12), 0.05, 0.04),
    "tail2": ("cone", (-0.85, 0.05, 0.12), (-1.03, 0.04, 0.30), 0.04, 0.032),
    "tail3": ("cone", (-1.03, 0.04, 0.30), (-0.98, 0.03, 0.46), 0.032, 0.024),
}
EYES_SIT = [(0.295, 0.905, 0.075), (0.295, 0.905, -0.075)]
EYES_LIE = [(0.575, 0.265, 0.075), (0.575, 0.265, -0.075)]
HEAD_PARTS = ("head", "muzzle", "ear_l", "ear_r")


def _lerp_prim(a, b, u):
    out = [a[0]]
    for pa, pb in zip(a[1:], b[1:]):
        out.append(tuple(np.asarray(pa, float) + (np.asarray(pb, float) - np.asarray(pa, float)) * u)
                   if isinstance(pa, tuple) else pa + (pb - pa) * u)
    return out


def _rot_about(p, c, ang):
    """Rotate point p about pivot c in the x-y plane (nodding the head up)."""
    p, c = np.asarray(p, float), np.asarray(c, float)
    d = p - c
    ca, sa = math.cos(ang), math.sin(ang)
    return tuple(c + np.array([d[0] * ca - d[1] * sa, d[0] * sa + d[1] * ca, d[2]]))


def cat_pose(down=0.0, hop=0.0, look_up=0.0, origin=(0.0, 0.0, 0.0)):
    """Primitive list for the cat: down blends SIT->LIE, hop lifts it, look_up tilts the head."""
    origin = np.asarray(origin, float)
    lift = np.array([0.28, 0.55, 0.0]) * hop
    prims, eyes = [], []
    head_c = np.asarray(CAT_SIT["head"][1]) * (1 - down) + np.asarray(CAT_LIE["head"][1]) * down
    pivot = head_c + np.array([-0.12, -0.12, 0.0])
    for name in CAT_SIT:
        kind, *params = _lerp_prim(CAT_SIT[name], CAT_LIE[name], down)
        npos = 1 if kind == "ell" else 2                  # leading params are positions, the rest sizes
        for k in range(npos):
            p = params[k]
            if name in HEAD_PARTS and look_up > 0:
                p = _rot_about(p, pivot, 0.45 * look_up)
            params[k] = tuple(np.asarray(p) + lift + origin)
        if kind == "ell":
            prims.append(("ell", (np.asarray(params[0]), np.asarray(params[1]))))
        else:
            prims.append(("cone", (np.asarray(params[0]), np.asarray(params[1]), params[2], params[3])))
    for es, el in zip(EYES_SIT, EYES_LIE):
        e = np.asarray(es) * (1 - down) + np.asarray(el) * down
        if look_up > 0:
            e = np.asarray(_rot_about(tuple(e), pivot, 0.45 * look_up))
        eyes.append(e + lift + origin)
    return prims, eyes


def cat_mesh(down=0.0, hop=0.0, look_up=0.0, origin=(0.0, 0.0, 0.0), step=0.018):
    prims, eyes = cat_pose(down, hop, look_up, origin)
    o = np.asarray(origin) + np.array([0.28, 0.55, 0.0]) * hop
    lo = o + np.array([-1.15, -0.02, -0.4])
    hi = o + np.array([0.82, 1.25, 0.58])
    v, f, n = mesh([(k, a) for k, a in prims], lo, hi, step, k=0.06)
    return v, f, n, eyes


def table_mesh(h, x0=0.0, width=2.0, depth=1.0, step=0.022):
    top_t, leg = 0.08, 0.075
    prims = [("box", (np.array([x0 + width / 2, h - top_t / 2, 0.0]), np.array([width / 2, top_t / 2, depth / 2]),
                      0.025))]
    for lx in (x0 + 0.1, x0 + width - 0.1):
        for lz in (-depth / 2 + 0.1, depth / 2 - 0.1):
            prims.append(("box", (np.array([lx, (h - top_t) / 2, lz]),
                                  np.array([leg / 2, (h - top_t) / 2, leg / 2]), 0.012)))
    lo = np.array([x0 - 0.05, -0.02, -depth / 2 - 0.05])
    hi = np.array([x0 + width + 0.05, h + 0.05, depth / 2 + 0.05])
    return mesh(prims, lo, hi, step, k=0.02)
