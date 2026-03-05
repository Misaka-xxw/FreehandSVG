"""
Advanced + Fast Sketch Vectorizer (Raster -> SVG path)

Goals:
- Stable: no missing strokes (graph-based tracing)
- Smooth: curvature segmentation + arc detection + Schneider-style cubic Bezier fitting
- Clean: straight parts output as SVG 'L', circular parts as 'A', others as 'C'
- Fast: fixed 8-neighborhood graph + directed-edge visited + optional Numba JIT

Dependencies:
    pip install numpy scipy scikit-image pillow
Optional (recommended for speed):
    pip install numba

Input:
    img_0_255: np.ndarray (H,W), binary-like or grayscale, white bg=255, black stroke=0
Output:
    SVG file with multiple <path>.

Notes:
- Coordinates: internally (y,x), SVG uses (x,y).
- Arc flags (large-arc/sweep) are heuristics; for most sketches they look correct.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional
import math
import numpy as np
from scipy.ndimage import convolve, label, binary_closing
from skimage.morphology import skeletonize
from PIL import Image

# ----------------------------
# Optional numba
# ----------------------------
try:
    from numba import njit
    NUMBA_OK = True
except Exception:
    NUMBA_OK = False

    def njit(*args, **kwargs):
        def deco(f):
            return f
        return deco


# =========================
# Constants
# =========================
# 8-neighborhood offsets in fixed order
N8 = np.array(
    [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),           (0, 1),
        (1, -1),  (1, 0),  (1, 1),
    ],
    dtype=np.int8,
)

# reverse direction mapping for this N8 ordering
REV8 = np.array([7, 6, 5, 4, 3, 2, 1, 0], dtype=np.int8)

IPt = Tuple[int, int]   # (y,x) int pixel
Pt = Tuple[float, float]  # (y,x) float


# =========================
# Params
# =========================
@dataclass
class Params:
    # preprocessing
    close_size: int = 3
    min_component_size: int = 10

    # spur pruning (light)
    prune_spurs: bool = True
    spur_rounds: int = 2
    spur_max_len: int = 8

    # speed knob: keep every k-th point in traced polylines
    sample_step: int = 1  # 1=all, 2=every other, ...

    # segmentation: split when turning angle > threshold
    # smaller -> more segments -> smoother circles but more paths/fit work
    angle_split_rad: float = 0.45  # ~25.8 degrees

    # line detection tolerance (px)
    line_tol: float = 0.9

    # arc fitting tolerance (px RMS radial error)
    arc_tol: float = 1.4

    # bezier fitting (Schneider) max error (px)
    bezier_max_error: float = 1.5
    bezier_max_depth: int = 18

    # svg style
    stroke: str = "black"
    stroke_width: float = 1.0


# ============================================================
# 1) Preprocess -> binary
# ============================================================
def _remove_small_components(binary: np.ndarray, min_size: int) -> np.ndarray:
    if min_size <= 0:
        return binary
    structure = np.ones((3, 3), dtype=int)  # 8-connect
    lab, n = label(binary, structure=structure)
    if n == 0:
        return binary
    counts = np.bincount(lab.ravel())
    keep = counts >= min_size
    keep[0] = False
    return keep[lab]


def preprocess(img_0_255: np.ndarray, p: Params) -> np.ndarray:
    """
    Assumes white background=255, black stroke=0.
    If your strokes are white, invert before calling: img = 255-img.
    """
    binary = (img_0_255 < 128) if img_0_255.dtype != np.bool_ else img_0_255.copy()
    binary = _remove_small_components(binary, p.min_component_size)
    if p.close_size and p.close_size > 1:
        k = np.ones((p.close_size, p.close_size), dtype=bool)
        binary = binary_closing(binary, structure=k)
    return binary


# ============================================================
# 2) Skeleton + degree
# ============================================================
def skeleton_and_degree(binary: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    skel = skeletonize(binary).astype(bool)
    kernel = np.ones((3, 3), dtype=int)
    neigh = convolve(skel.astype(int), kernel, mode="constant", cval=0)
    deg = neigh - skel.astype(int)
    return skel, deg


# ============================================================
# 3) Spur pruning (optional)
# ============================================================
def prune_spurs(skel: np.ndarray, p: Params) -> np.ndarray:
    if not p.prune_spurs:
        return skel
    h, w = skel.shape
    sk = skel.copy()

    def degree_map(s: np.ndarray) -> np.ndarray:
        kernel = np.ones((3, 3), dtype=int)
        neigh = convolve(s.astype(int), kernel, mode="constant", cval=0)
        return neigh - s.astype(int)

    for _ in range(max(1, p.spur_rounds)):
        deg = degree_map(sk)
        endpoints = list(zip(*np.where(sk & (deg == 1))))
        removed = False
        for ep in endpoints:
            path = [ep]
            prev = None
            cur = ep
            while len(path) <= p.spur_max_len:
                y, x = cur
                nbrs = []
                for dy, dx in N8:
                    ny, nx = y + int(dy), x + int(dx)
                    if 0 <= ny < h and 0 <= nx < w and sk[ny, nx]:
                        if prev is None or (ny, nx) != prev:
                            nbrs.append((ny, nx))
                if len(nbrs) != 1:
                    break
                nxt = nbrs[0]
                path.append(nxt)
                prev, cur = cur, nxt
                if deg[cur] != 2:
                    break
            # remove short spur that ends at a node (keep node pixel)
            if len(path) <= p.spur_max_len and deg[path[-1]] != 1:
                for yy, xx in path[:-1]:
                    sk[yy, xx] = False
                removed = True
        if not removed:
            break
    return sk


# ============================================================
# 4) Graph build (fixed 8-neighborhood adjacency arrays)
# ============================================================
@njit(cache=True)
def _build_graph_fast_numba(skel: np.ndarray):
    h, w = skel.shape

    # count
    n = 0
    for y in range(h):
        for x in range(w):
            if skel[y, x]:
                n += 1

    coords = np.zeros((n, 2), np.int32)
    id_map = -np.ones((h, w), np.int32)

    idx = 0
    for y in range(h):
        for x in range(w):
            if skel[y, x]:
                coords[idx, 0] = y
                coords[idx, 1] = x
                id_map[y, x] = idx
                idx += 1

    nbr = -np.ones((n, 8), np.int32)
    deg = np.zeros((n,), np.uint8)

    for i in range(n):
        y = coords[i, 0]
        x = coords[i, 1]
        d = 0
        for k in range(8):
            ny = y + int(N8[k, 0])
            nx = x + int(N8[k, 1])
            if 0 <= ny < h and 0 <= nx < w:
                j = id_map[ny, nx]
                if j >= 0:
                    nbr[i, k] = j
                    d += 1
        deg[i] = d

    return coords, id_map, nbr, deg


def build_graph_fixed(skel: np.ndarray):
    # if numba unavailable, fall back to python version
    if NUMBA_OK:
        return _build_graph_fast_numba(skel)

    ys, xs = np.where(skel)
    coords = np.stack([ys, xs], axis=1).astype(np.int32)
    n = coords.shape[0]
    id_map = -np.ones(skel.shape, dtype=np.int32)
    id_map[ys, xs] = np.arange(n, dtype=np.int32)

    nbr = -np.ones((n, 8), dtype=np.int32)
    deg = np.zeros((n,), dtype=np.uint8)

    h, w = skel.shape
    for i in range(n):
        y, x = int(coords[i, 0]), int(coords[i, 1])
        d = 0
        for k, (dy, dx) in enumerate(N8):
            ny, nx = y + int(dy), x + int(dx)
            if 0 <= ny < h and 0 <= nx < w and skel[ny, nx]:
                j = int(id_map[ny, nx])
                if j >= 0:
                    nbr[i, k] = j
                    d += 1
        deg[i] = d
    return coords, id_map, nbr, deg


# ============================================================
# 5) Graph compression tracing (fast + stable, with visited directed edges)
# ============================================================
@njit(cache=True)
def _trace_polylines_numba(coords: np.ndarray, nbr: np.ndarray, deg: np.ndarray, sample_step: int):
    n = coords.shape[0]
    vis = np.zeros((n, 8), dtype=np.uint8)  # directed edge visited
    polylines = []  # list of list[(y,x)] in python mode; numba stores as list of lists of tuples

    is_node = deg != 2

    def mark(u: int, k: int, v: int):
        vis[u, k] = 1
        vis[v, int(REV8[k])] = 1

    step = 1 if sample_step < 1 else sample_step

    # 1) node-to-node edges
    for u in range(n):
        if not is_node[u]:
            continue
        for k in range(8):
            v = nbr[u, k]
            if v < 0 or vis[u, k] == 1:
                continue

            idxs = [u, int(v)]
            mark(u, k, int(v))
            prev = u
            cur = int(v)

            guard = n + 10
            while guard > 0:
                guard -= 1
                if is_node[cur] and cur != u:
                    break

                # choose next neighbor != prev, prefer unvisited
                next_v = -1
                next_k = -1
                for kk in range(8):
                    vv = nbr[cur, kk]
                    if vv < 0 or int(vv) == prev:
                        continue
                    if vis[cur, kk] == 0:
                        next_v = int(vv)
                        next_k = kk
                        break
                if next_v < 0:
                    for kk in range(8):
                        vv = nbr[cur, kk]
                        if vv < 0 or int(vv) == prev:
                            continue
                        next_v = int(vv)
                        next_k = kk
                        break
                if next_v < 0:
                    break

                idxs.append(next_v)
                mark(cur, next_k, next_v)
                prev, cur = cur, next_v
                if cur == u:
                    break

            pts = []
            for t in range(0, len(idxs), step):
                ii = idxs[t]
                pts.append((int(coords[ii, 0]), int(coords[ii, 1])))
            if len(pts) >= 2:
                polylines.append(pts)

    # 2) loops (deg==2) with unvisited incident edge
    for start in range(n):
        if deg[start] != 2:
            continue
        k0 = -1
        for k in range(8):
            if nbr[start, k] >= 0 and vis[start, k] == 0:
                k0 = k
                break
        if k0 < 0:
            continue

        v = int(nbr[start, k0])
        idxs = [start, v]
        mark(start, k0, v)
        prev, cur = start, v

        guard = n + 10
        while guard > 0:
            guard -= 1
            if cur == start:
                break
            next_v = -1
            next_k = -1
            for kk in range(8):
                vv = nbr[cur, kk]
                if vv < 0 or int(vv) == prev:
                    continue
                if vis[cur, kk] == 0:
                    next_v = int(vv)
                    next_k = kk
                    break
            if next_v < 0:
                for kk in range(8):
                    vv = nbr[cur, kk]
                    if vv < 0 or int(vv) == prev:
                        continue
                    next_v = int(vv)
                    next_k = kk
                    break
            if next_v < 0:
                break

            idxs.append(next_v)
            mark(cur, next_k, next_v)
            prev, cur = cur, next_v

        pts = []
        for t in range(0, len(idxs), step):
            ii = idxs[t]
            pts.append((int(coords[ii, 0]), int(coords[ii, 1])))
        if len(pts) >= 4:
            polylines.append(pts)

    return polylines


def trace_polylines(coords: np.ndarray, nbr: np.ndarray, deg: np.ndarray, p: Params) -> List[List[IPt]]:
    if NUMBA_OK:
        return _trace_polylines_numba(coords, nbr, deg, p.sample_step)

    # python fallback
    n = coords.shape[0]
    vis = np.zeros((n, 8), dtype=bool)
    polylines: List[List[IPt]] = []
    is_node = deg != 2
    step = max(1, p.sample_step)

    def mark(u: int, k: int, v: int):
        vis[u, k] = True
        vis[v, int(REV8[k])] = True

    nodes = np.where(is_node)[0]
    for u in nodes:
        u = int(u)
        for k in range(8):
            v = int(nbr[u, k])
            if v < 0 or vis[u, k]:
                continue
            idxs = [u, v]
            mark(u, k, v)
            prev, cur = u, v
            guard = n + 10
            while guard > 0:
                guard -= 1
                if is_node[cur] and cur != u:
                    break
                next_v, next_k = -1, -1
                for kk in range(8):
                    vv = int(nbr[cur, kk])
                    if vv < 0 or vv == prev:
                        continue
                    if not vis[cur, kk]:
                        next_v, next_k = vv, kk
                        break
                if next_v < 0:
                    for kk in range(8):
                        vv = int(nbr[cur, kk])
                        if vv < 0 or vv == prev:
                            continue
                        next_v, next_k = vv, kk
                        break
                if next_v < 0:
                    break
                idxs.append(next_v)
                mark(cur, next_k, next_v)
                prev, cur = cur, next_v
                if cur == u:
                    break
            pts = [(int(coords[ii, 0]), int(coords[ii, 1])) for ii in idxs[::step]]
            if len(pts) >= 2:
                polylines.append(pts)

    # loops
    deg2 = np.where(deg == 2)[0]
    for start in deg2:
        start = int(start)
        k0 = -1
        for k in range(8):
            if nbr[start, k] >= 0 and not vis[start, k]:
                k0 = k
                break
        if k0 < 0:
            continue
        v = int(nbr[start, k0])
        idxs = [start, v]
        mark(start, k0, v)
        prev, cur = start, v
        guard = n + 10
        while guard > 0:
            guard -= 1
            if cur == start:
                break
            next_v, next_k = -1, -1
            for kk in range(8):
                vv = int(nbr[cur, kk])
                if vv < 0 or vv == prev:
                    continue
                if not vis[cur, kk]:
                    next_v, next_k = vv, kk
                    break
            if next_v < 0:
                for kk in range(8):
                    vv = int(nbr[cur, kk])
                    if vv < 0 or vv == prev:
                        continue
                    next_v, next_k = vv, kk
                    break
            if next_v < 0:
                break
            idxs.append(next_v)
            mark(cur, next_k, next_v)
            prev, cur = cur, next_v

        pts = [(int(coords[ii, 0]), int(coords[ii, 1])) for ii in idxs[::step]]
        if len(pts) >= 4:
            polylines.append(pts)

    return polylines


# ============================================================
# 6) Simple junction smoothing on each polyline (remove tiny “almost straight” kinks)
# ============================================================
def simplify_by_angle(points: np.ndarray, angle_thresh_rad: float) -> np.ndarray:
    """
    Remove middle points where turning angle is small (< angle_thresh_rad).
    """
    n = points.shape[0]
    if n < 3:
        return points
    keep = [0]
    for i in range(1, n - 1):
        v1 = points[i] - points[i - 1]
        v2 = points[i + 1] - points[i]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        c = float(np.dot(v1, v2) / (n1 * n2))
        c = max(-1.0, min(1.0, c))
        ang = math.acos(c)
        if ang > angle_thresh_rad:
            keep.append(i)
    keep.append(n - 1)
    keep = sorted(set(keep))
    return points[keep]


# ============================================================
# 7) Curvature segmentation (split when turning angle is large)
# ============================================================
def split_by_turning_angle(pts: np.ndarray, angle_thresh: float) -> List[np.ndarray]:
    n = pts.shape[0]
    if n < 4:
        return [pts]
    cut = [0]
    for i in range(1, n - 1):
        v1 = pts[i] - pts[i - 1]
        v2 = pts[i + 1] - pts[i]
        n1 = np.linalg.norm(v1)
        n2 = np.linalg.norm(v2)
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        c = float(np.dot(v1, v2) / (n1 * n2))
        c = max(-1.0, min(1.0, c))
        ang = math.acos(c)
        if ang > angle_thresh:
            cut.append(i)
    cut.append(n - 1)
    cut = sorted(set(cut))
    segs = []
    for a, b in zip(cut[:-1], cut[1:]):
        if b - a + 1 >= 2:
            segs.append(pts[a:b + 1])
    return segs if segs else [pts]


# ============================================================
# 8) Classification: line / arc / bezier
# ============================================================
def point_line_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    denom = np.linalg.norm(ab)
    if denom < 1e-12:
        return float(np.linalg.norm(p - a))
    return float(abs(np.cross(ab, p - a)) / denom)


def is_line(seg: np.ndarray, tol: float) -> bool:
    if seg.shape[0] <= 2:
        return True
    a = seg[0]
    b = seg[-1]
    for i in range(1, seg.shape[0] - 1):
        if point_line_dist(seg[i], a, b) > tol:
            return False
    return True


def fit_circle(seg: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Algebraic least squares circle fit.
    seg points are (y,x). We'll fit in (x,y) plane.
    Returns cx, cy, r, rms_radial_error
    """
    y = seg[:, 0]
    x = seg[:, 1]
    A = np.c_[2 * x, 2 * y, np.ones_like(x)]
    b = x * x + y * y
    c, *_ = np.linalg.lstsq(A, b, rcond=None)
    cx, cy, c0 = float(c[0]), float(c[1]), float(c[2])
    r = math.sqrt(max(1e-12, c0 + cx * cx + cy * cy))
    rr = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    rms = float(np.sqrt(np.mean((rr - r) ** 2)))
    return cx, cy, r, rms


def arc_svg_command(seg: np.ndarray, cx: float, cy: float, r: float) -> str:
    """
    Create SVG 'A' command ending at seg[-1].
    large-arc and sweep are heuristics based on endpoint angles and cross product.
    """
    y0, x0 = seg[0]
    y1, x1 = seg[-1]
    a0 = math.atan2(y0 - cy, x0 - cx)
    a1 = math.atan2(y1 - cy, x1 - cx)
    da = a1 - a0
    # normalize to (-pi,pi]
    while da <= -math.pi:
        da += 2 * math.pi
    while da > math.pi:
        da -= 2 * math.pi
    large_arc = 1 if abs(da) > math.pi / 2 else 0  # heuristic
    v0 = np.array([x0 - cx, y0 - cy])
    v1 = np.array([x1 - cx, y1 - cy])
    cross = float(v0[0] * v1[1] - v0[1] * v1[0])
    sweep = 1 if cross > 0 else 0
    return f"A {r:.2f} {r:.2f} 0 {large_arc} {sweep} {x1:.2f} {y1:.2f}"


def is_arc(seg: np.ndarray, arc_tol: float, min_points: int = 8) -> Optional[Tuple[float, float, float]]:
    if seg.shape[0] < min_points:
        return None
    cx, cy, r, rms = fit_circle(seg)
    if rms <= arc_tol:
        return cx, cy, r
    return None


# ============================================================
# 9) Schneider / Graphics-Gems cubic Bezier fitting
# ============================================================
def chord_length_parameterize(pts: np.ndarray) -> np.ndarray:
    d = np.sqrt(((pts[1:] - pts[:-1]) ** 2).sum(axis=1))
    u = np.zeros((pts.shape[0],), dtype=np.float64)
    u[1:] = np.cumsum(d)
    if u[-1] > 1e-9:
        u /= u[-1]
    return u


def bezier_eval(ctrl: np.ndarray, t: np.ndarray) -> np.ndarray:
    mt = 1.0 - t
    b0 = (mt ** 3)[:, None]
    b1 = (3 * mt * mt * t)[:, None]
    b2 = (3 * mt * t * t)[:, None]
    b3 = (t ** 3)[:, None]
    return b0 * ctrl[0] + b1 * ctrl[1] + b2 * ctrl[2] + b3 * ctrl[3]


def bezier_prime(ctrl: np.ndarray, t: np.ndarray) -> np.ndarray:
    mt = 1.0 - t
    return (
        (3 * mt * mt)[:, None] * (ctrl[1] - ctrl[0])
        + (6 * mt * t)[:, None] * (ctrl[2] - ctrl[1])
        + (3 * t * t)[:, None] * (ctrl[3] - ctrl[2])
    )


def bezier_prime2(ctrl: np.ndarray, t: np.ndarray) -> np.ndarray:
    mt = 1.0 - t
    return (
        (6 * mt)[:, None] * (ctrl[2] - 2 * ctrl[1] + ctrl[0])
        + (6 * t)[:, None] * (ctrl[3] - 2 * ctrl[2] + ctrl[1])
    )


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def left_tangent(pts: np.ndarray) -> np.ndarray:
    return normalize(pts[1] - pts[0])


def right_tangent(pts: np.ndarray) -> np.ndarray:
    return normalize(pts[-2] - pts[-1])


def center_tangent(pts: np.ndarray, i: int) -> np.ndarray:
    return normalize((pts[i - 1] - pts[i]) + (pts[i] - pts[i + 1]))


def generate_bezier(pts: np.ndarray, u: np.ndarray, t1: np.ndarray, t2: np.ndarray) -> np.ndarray:
    """
    Solve alpha1, alpha2 for ctrl points:
      P0, P0 + alpha1*t1, P3 + alpha2*t2, P3
    using least squares as in Graphics Gems.
    """
    P0 = pts[0]
    P3 = pts[-1]

    A1 = (3 * (1 - u) * (1 - u) * u)[:, None] * t1[None, :]
    A2 = (3 * (1 - u) * u * u)[:, None] * t2[None, :]

    mt = 1 - u
    B0 = (mt ** 3)[:, None]
    B3 = (u ** 3)[:, None]
    tmp = pts - (B0 * P0 + B3 * P3)

    C00 = float(np.sum(A1[:, 0] * A1[:, 0] + A1[:, 1] * A1[:, 1]))
    C01 = float(np.sum(A1[:, 0] * A2[:, 0] + A1[:, 1] * A2[:, 1]))
    C11 = float(np.sum(A2[:, 0] * A2[:, 0] + A2[:, 1] * A2[:, 1]))
    X0 = float(np.sum(A1[:, 0] * tmp[:, 0] + A1[:, 1] * tmp[:, 1]))
    X1 = float(np.sum(A2[:, 0] * tmp[:, 0] + A2[:, 1] * tmp[:, 1]))

    det = C00 * C11 - C01 * C01
    if abs(det) > 1e-12:
        alpha1 = (X0 * C11 - X1 * C01) / det
        alpha2 = (C00 * X1 - C01 * X0) / det
    else:
        chord = np.linalg.norm(P3 - P0)
        alpha1 = alpha2 = chord / 3.0

    seg_len = np.linalg.norm(P3 - P0)
    eps = 1e-6 * seg_len
    if alpha1 < eps or alpha2 < eps:
        alpha1 = alpha2 = seg_len / 3.0

    ctrl = np.zeros((4, 2), dtype=np.float64)
    ctrl[0] = P0
    ctrl[3] = P3
    ctrl[1] = P0 + alpha1 * t1
    ctrl[2] = P3 + alpha2 * t2
    return ctrl


def max_error(pts: np.ndarray, ctrl: np.ndarray, u: np.ndarray) -> Tuple[float, int]:
    curve = bezier_eval(ctrl, u)
    d = np.sqrt(((curve - pts) ** 2).sum(axis=1))
    idx = int(np.argmax(d))
    return float(d[idx]), idx


def reparameterize(pts: np.ndarray, ctrl: np.ndarray, u: np.ndarray) -> np.ndarray:
    Q = bezier_eval(ctrl, u)
    Q1 = bezier_prime(ctrl, u)
    Q2 = bezier_prime2(ctrl, u)
    diff = Q - pts
    num = (diff * Q1).sum(axis=1)
    den = (Q1 * Q1).sum(axis=1) + (diff * Q2).sum(axis=1)
    du = np.where(np.abs(den) > 1e-12, num / den, 0.0)
    u2 = u - du
    return np.clip(u2, 0.0, 1.0)


def fit_cubic_recursive(
    pts: np.ndarray,
    t1: np.ndarray,
    t2: np.ndarray,
    maxerr: float,
    depth: int,
    max_depth: int,
) -> List[np.ndarray]:
    n = pts.shape[0]
    if n == 2:
        P0, P3 = pts[0], pts[1]
        d = np.linalg.norm(P3 - P0) / 3.0
        ctrl = np.array([P0, P0 + d * t1, P3 + d * t2, P3], dtype=np.float64)
        return [ctrl]

    u = chord_length_parameterize(pts)
    ctrl = generate_bezier(pts, u, t1, t2)
    err, split = max_error(pts, ctrl, u)
    if err <= maxerr or depth >= max_depth:
        return [ctrl]

    # try improving parameterization a few times
    for _ in range(4):
        u = reparameterize(pts, ctrl, u)
        ctrl = generate_bezier(pts, u, t1, t2)
        err, split = max_error(pts, ctrl, u)
        if err <= maxerr:
            return [ctrl]

    split = max(1, min(n - 2, split))
    tc = center_tangent(pts, split)
    left = fit_cubic_recursive(pts[: split + 1], t1, tc, maxerr, depth + 1, max_depth)
    right = fit_cubic_recursive(pts[split:], -tc, t2, maxerr, depth + 1, max_depth)
    return left + right


def fit_beziers(pts: np.ndarray, maxerr: float, max_depth: int) -> List[np.ndarray]:
    if pts.shape[0] < 2:
        return []
    t1 = left_tangent(pts)
    t2 = right_tangent(pts)
    return fit_cubic_recursive(pts, t1, t2, maxerr, 0, max_depth)


# ============================================================
# 10) SVG assembly per polyline (mix L / A / C)
# ============================================================
def segment_to_svg_commands(seg: np.ndarray, p: Params) -> List[str]:
    if seg.shape[0] < 2:
        return []

    # line?
    if is_line(seg, p.line_tol):
        y1, x1 = seg[-1]
        return [f"L {x1:.2f} {y1:.2f}"]

    # arc?
    arc = is_arc(seg, p.arc_tol)
    if arc is not None:
        cx, cy, r = arc
        return [arc_svg_command(seg, cx, cy, r)]

    # bezier
    ctrls = fit_beziers(seg, p.bezier_max_error, p.bezier_max_depth)
    cmds: List[str] = []
    for c in ctrls:
        b1, b2, b3 = c[1], c[2], c[3]
        cmds.append(
            f"C {b1[1]:.2f} {b1[0]:.2f} {b2[1]:.2f} {b2[0]:.2f} {b3[1]:.2f} {b3[0]:.2f}"
        )
    return cmds


def polyline_to_path(poly: List[IPt], p: Params) -> str:
    pts = np.array(poly, dtype=np.float64)  # (y,x)

    # local kink cleanup (helps circles and long lines)
    pts = simplify_by_angle(pts, angle_thresh_rad=0.18)  # ~10.3 deg remove tiny wiggles

    # curvature segmentation for classification
    segs = split_by_turning_angle(pts, p.angle_split_rad)

    y0, x0 = segs[0][0]
    cmds = [f"M {x0:.2f} {y0:.2f}"]
    for seg in segs:
        cmds.extend(segment_to_svg_commands(seg, p))
    return " ".join(cmds)


def save_svg(paths: List[str], w: int, h: int, out_file: str, p: Params):
    with open(out_file, "w", encoding="utf-8") as f:
        f.write(
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}">\n'
        )
        for d in paths:
            f.write(
                f'  <path d="{d}" stroke="{p.stroke}" fill="none" stroke-width="{p.stroke_width}"/>\n'
            )
        f.write("</svg>\n")


# ============================================================
# 11) Main API
# ============================================================
def vectorize_final(img_0_255: np.ndarray, p: Params = Params()) -> List[str]:
    binary = preprocess(img_0_255, p)
    skel, _deg_img = skeleton_and_degree(binary)
    skel = prune_spurs(skel, p)

    coords, _id_map, nbr, deg = build_graph_fixed(skel)
    polylines = trace_polylines(coords, nbr, deg, p)

    paths: List[str] = []
    for poly in polylines:
        if len(poly) < 2:
            continue
        paths.append(polyline_to_path(poly, p))
    return paths


# ============================================================
# Example usage
# ============================================================
if __name__ == "__main__":
    # Your binary image: white background 255, black strokes 0
    img = np.array(Image.open(r"D:\GitHub\FreehandSVG\test\img\clean_edges3.png").convert("L"))

    p = Params(
        close_size=3,
        min_component_size=10,

        prune_spurs=True,
        spur_rounds=2,
        spur_max_len=8,

        # speed / quality tradeoffs:
        sample_step=1,          # set to 2 for more speed

        # make circles smoother:
        angle_split_rad=0.45,   # smaller -> smoother circles, more segments

        # line / arc / bezier:
        line_tol=0.9,
        arc_tol=1.4,
        bezier_max_error=1.5,
        bezier_max_depth=18,

        stroke="black",
        stroke_width=1.0,
    )

    paths = vectorize_final(img, p)
    save_svg(paths, img.shape[1], img.shape[0], "output3.svg", p)

    print("done",
          "paths=", len(paths),
          "numba=", NUMBA_OK)