"""
CPU perturbation kernels.

Three jobs:
  1. ground truth for verifying the GLSL shader (same maths, no GPU involved)
  2. the probe renderer that the auto-descent uses to find deep coordinates
  3. the offline high-quality render path, where wall-clock matters less than
     getting every pixel right

The Numba version runs `prange` over image rows: every physical core gets a
slab, and the inner loop is straight-line float64 arithmetic that the compiler
vectorises into AVX. It is the honest answer to "use the CPU" -- unlike the
reference orbit, which is strictly serial and cannot be parallelised by anyone.

Derivatives are carried *pre-scaled by the pixel size*. The raw dz/dc at a 1e-300
zoom is around 1e300 and overflows on the way to the interesting part; folding
the pixel scale into the recurrence keeps it near unity and makes the distance
estimate fall out in units of pixels, which is what the shader wants anyway.
"""
from __future__ import annotations

import numpy as np

BAILOUT2 = 1.0e12

try:
    from numba import njit, prange
    HAVE_NUMBA = True
except ImportError:  # pragma: no cover
    HAVE_NUMBA = False

    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not a or callable(a[0]) is False else a[0]

    prange = range


# --------------------------------------------------------------------------
# scalar core: one pixel, perturbation + Zhuoran rebasing
# --------------------------------------------------------------------------
def _pixel_core(refx, refy, reflen, dcx, dcy, maxiter, pixel_scale):
    dzx = 0.0
    dzy = 0.0
    dx = 0.0                # dz/dc, pre-scaled by pixel_scale
    dy = 0.0
    m = 0
    n = 0
    while n < maxiter - 1:
        Zx = refx[m]
        Zy = refy[m]
        # full z at this step, needed for the derivative recurrence
        zx_full = Zx + dzx
        zy_full = Zy + dzy
        ndx = 2.0 * (zx_full * dx - zy_full * dy) + pixel_scale
        ndy = 2.0 * (zx_full * dy + zy_full * dx)
        dx = ndx
        dy = ndy

        # dz <- 2*Z*dz + dz^2 + dc
        ax = 2.0 * (Zx * dzx - Zy * dzy) + (dzx * dzx - dzy * dzy) + dcx
        ay = 2.0 * (Zx * dzy + Zy * dzx) + 2.0 * dzx * dzy + dcy
        m += 1
        n += 1
        zx = refx[m] + ax
        zy = refy[m] + ay
        zmag = zx * zx + zy * zy
        dmag = ax * ax + ay * ay

        if zmag > BAILOUT2:
            return n, zmag, dx * dx + dy * dy
        if zmag < dmag or m >= reflen:
            # rebase: the true value is now smaller than the delta (or we ran
            # off the end of the reference), so restart the reference at zero
            dzx = zx
            dzy = zy
            m = 0
        else:
            dzx = ax
            dzy = ay
    return maxiter, 0.0, dx * dx + dy * dy


# --------------------------------------------------------------------------
def _render_impl(refx, refy, reflen, cx0, cy0, span_x, span_y,
                 w, h, maxiter, out_n, out_de):
    pixel_scale = span_x / w
    for j in prange(h):
        dcy = cy0 + span_y * ((j + 0.5) / h)
        for i in range(w):
            dcx = cx0 + span_x * ((i + 0.5) / w)
            n, zmag, dmag = _pixel_core(refx, refy, reflen, dcx, dcy,
                                        maxiter, pixel_scale)
            if n >= maxiter:
                out_n[j, i] = -1.0                 # interior
                out_de[j, i] = 0.0
            else:
                # smooth (continuous) escape count
                lz = 0.5 * np.log(zmag)
                out_n[j, i] = n + 1.0 - np.log(lz) / 0.6931471805599453
                # distance estimate, already in pixel units thanks to scaling
                if dmag > 0.0:
                    out_de[j, i] = np.sqrt(zmag) * lz / np.sqrt(dmag)
                else:
                    out_de[j, i] = 1e30


if HAVE_NUMBA:
    _pixel_core = njit(cache=True, fastmath=True, inline="always")(_pixel_core)
    _render_par = njit(cache=True, fastmath=True, parallel=True)(_render_impl)
    _render_ser = njit(cache=True, fastmath=True, parallel=False)(_render_impl)
else:  # pragma: no cover
    _render_par = _render_ser = _render_impl


# --------------------------------------------------------------------------
def render(ref, cx_off, cy_off, span_x, span_y, w, h, maxiter, parallel=True):
    """
    Render a tile in *offset* coordinates: (cx_off, cy_off) is the top-left
    corner expressed as a delta from the reference point, and span_* is the
    width/height in the same delta units.

    Returns (smooth_iter, distance_estimate) float64 arrays; smooth_iter is -1
    for interior pixels.
    """
    out_n = np.empty((h, w), dtype=np.float64)
    out_de = np.empty((h, w), dtype=np.float64)
    fn = _render_par if parallel else _render_ser
    fn(ref.zx, ref.zy, ref.length, float(cx_off), float(cy_off),
       float(span_x), float(span_y), int(w), int(h), int(maxiter), out_n, out_de)
    return out_n, out_de


# --------------------------------------------------------------------------
# vectorised NumPy variant -- no Numba needed, used by the test suite so that
# validation never depends on a JIT compiling correctly
# --------------------------------------------------------------------------
def escape_counts_numpy(refx, refy, reflen, dcx, dcy, maxiter, dtype=np.float64):
    refx = refx.astype(dtype)
    refy = refy.astype(dtype)
    dcx = np.asarray(dcx, dtype=dtype).ravel()
    dcy = np.asarray(dcy, dtype=dtype).ravel()

    npix = dcx.size
    dzx = np.zeros(npix, dtype=dtype)
    dzy = np.zeros(npix, dtype=dtype)
    m = np.zeros(npix, dtype=np.int64)
    out = np.full(npix, maxiter, dtype=np.int64)
    live = np.ones(npix, dtype=bool)
    bail = dtype(BAILOUT2)
    two = dtype(2.0)

    for n in range(maxiter - 1):
        idx = np.flatnonzero(live)
        if idx.size == 0:
            break
        mi = m[idx]
        Zx = refx[mi]
        Zy = refy[mi]
        dx = dzx[idx]
        dy = dzy[idx]
        ndx = two * (Zx * dx - Zy * dy) + (dx * dx - dy * dy) + dcx[idx]
        ndy = two * (Zx * dy + Zy * dx) + two * dx * dy + dcy[idx]
        mi = mi + 1
        zx = refx[mi] + ndx
        zy = refy[mi] + ndy
        zmag = zx * zx + zy * zy
        dmag = ndx * ndx + ndy * ndy
        escaped = zmag > bail
        rebase = (~escaped) & ((zmag < dmag) | (mi >= reflen))
        dzx[idx] = np.where(rebase, zx, ndx)
        dzy[idx] = np.where(rebase, zy, ndy)
        m[idx] = np.where(rebase, 0, mi)
        esc = idx[escaped]
        out[esc] = n + 1
        live[esc] = False
    return out
