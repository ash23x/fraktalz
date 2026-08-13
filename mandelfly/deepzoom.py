"""
Finding places worth falling into.

Deep zoom coordinates cannot be typed by hand or found by bisection -- bisecting
between an interior and exterior point converges onto a *level set* of the
escape-time function, which is an analytic curve. Zoom into one of those far
enough and you get a straight line. Very deep. Very boring.

Real deep coordinates come from the self-similar structure of the set itself:
every neighbourhood of the boundary contains a scaled copy of the whole thing (a
"minibrot"), and each minibrot has an exact centre -- a *nucleus* -- where the
critical orbit is periodic. Nuclei are roots of z_p(c) = 0 and Newton's method
finds them to arbitrary precision. Aim the camera at a high-period nucleus and
you fall through layer after layer of structure to get there.

  atom_domain_period()  -- which period's minibrot dominates this neighbourhood
  newton_nucleus()      -- solve for its exact centre
  descend()             -- repeat, shrinking the view, to arbitrary depth

This is both the test-fixture generator and the app's "find me somewhere
interesting" button.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import gmpy2
from gmpy2 import mpfr, get_context


# --------------------------------------------------------------------------
def prec_for_radius(radius, guard_bits: int = 80) -> int:
    """Working precision needed to resolve a view of the given half-width."""
    try:
        exp10 = -math.log10(float(radius))
    except (ValueError, OverflowError):
        exp10 = 320.0
    if not math.isfinite(exp10):
        exp10 = 320.0
    return int(max(64, exp10 * 3.3219280948873626 + guard_bits))


def set_prec(bits: int) -> None:
    get_context().precision = int(bits)


# --------------------------------------------------------------------------
@dataclass
class Nucleus:
    cx: mpfr
    cy: mpfr
    period: int
    size: float          # approximate radius of the minibrot around it
    converged: bool


# --------------------------------------------------------------------------
def atom_domain_period(cx, cy, maxiter: int, bailout: float = 4.0) -> int:
    """
    Period of the atom domain containing c.

    Iterate the critical orbit; every time |z_n| hits a new minimum, n is a
    candidate period. The last such n before escape is the period of the
    smallest atom whose domain contains c -- i.e. the minibrot we are closest
    to being inside of.
    """
    zx = mpfr(0)
    zy = mpfr(0)
    minmag = None
    period = 1
    for n in range(1, maxiter + 1):
        zx, zy = zx * zx - zy * zy + cx, 2 * zx * zy + cy
        mag = zx * zx + zy * zy
        if minmag is None or mag < minmag:
            minmag = mag
            period = n
        if mag > bailout:
            break
    return period


# --------------------------------------------------------------------------
def newton_nucleus(cx0, cy0, period: int, max_steps: int = 256,
                   trust_radius=None) -> Nucleus:
    """
    Newton's method on z_p(c) = 0.

    Also accumulates the second derivative so we can estimate the minibrot's
    physical size, which tells the caller how far to zoom before the structure
    stops being interesting.
    """
    cx, cy = mpfr(cx0), mpfr(cy0)
    eps = mpfr(2) ** (-(get_context().precision - 24))
    converged = False

    for _ in range(max_steps):
        zx = zy = mpfr(0)
        dx = dy = mpfr(0)          # dz/dc
        for _ in range(period):
            ndx = 2 * (zx * dx - zy * dy) + 1
            ndy = 2 * (zx * dy + zy * dx)
            zx, zy = zx * zx - zy * zy + cx, 2 * zx * zy + cy
            dx, dy = ndx, ndy

        den = dx * dx + dy * dy
        if den == 0:
            break
        sx = (zx * dx + zy * dy) / den        # z / dz  (complex division)
        sy = (zy * dx - zx * dy) / den
        cx -= sx
        cy -= sy

        step = gmpy2.sqrt(sx * sx + sy * sy)
        if trust_radius is not None and step > trust_radius * 4:
            break                              # ran away; not our nucleus
        if step < eps:
            converged = True
            break

    # minibrot size ~ 1 / (|dz/dc| * |d(z_p)/dz along the cycle|); the standard
    # cheap estimate is 1/|dc|^2 using the derivative at the converged nucleus.
    zx = zy = mpfr(0)
    dx = dy = mpfr(0)
    for _ in range(period):
        ndx = 2 * (zx * dx - zy * dy) + 1
        ndy = 2 * (zx * dy + zy * dx)
        zx, zy = zx * zx - zy * zy + cx, 2 * zx * zy + cy
        dx, dy = ndx, ndy
    dmag = float(gmpy2.sqrt(dx * dx + dy * dy)) if (dx or dy) else 1.0
    size = 1.0 / (dmag * dmag) if dmag > 0 else 1.0

    return Nucleus(cx, cy, period, size, converged)


# --------------------------------------------------------------------------
def descend(cx0="-0.7436438870371587", cy0="0.13182590420531197",
            radius0="1e-2", target_radius="1e-100",
            shrink: float = 4.0, grid: int = 96,
            iter_cap: int = 400_000, verbose: bool = False,
            progress=None):
    """
    Walk downward toward ever-deeper structure, image-driven.

    Two earlier attempts at this failed instructively:

      * bisecting between an interior and an exterior point converges onto a
        *level set* of the escape-time function -- an analytic curve that looks
        like a straight line once you zoom past its curvature. Deep, featureless.

      * hunting minibrot nuclei and centring on them puts the camera inside the
        minibrot, whose interior is a solid black disc. Every subsequent probe
        reports the same period and the descent stalls forever.

    What actually works is to look at the picture. Render a coarse grid, then aim
    at the pixel that *escaped* with the highest iteration count -- the point in
    view that came closest to the boundary without being swallowed by it. That
    can never wander into the interior, because interior points are excluded by
    construction, and it always lands where the structure is densest.

    Returns (cx_str, cy_str, radius, maxiter) -- the centre as a full-precision
    decimal string, ready to hand to the renderer.
    """
    from .reference import compute_reference, prec_for_radius as _prec
    from .kernels import render

    target = float(target_radius) if isinstance(target_radius, str) else float(target_radius)
    radius = float(radius0) if isinstance(radius0, str) else float(radius0)
    set_prec(_prec(target) + 96)
    cx, cy = mpfr(cx0), mpfr(cy0)
    maxiter = 3000
    step = 0

    while radius > target:
        digits = max(1, int(-math.log10(radius)) + 24)
        prec = max(64, _prec(radius))
        set_prec(prec + 64)
        cx = mpfr(cx)
        cy = mpfr(cy)

        ref = compute_reference(format_coord(cx, digits), format_coord(cy, digits),
                                maxiter, prec)
        n, de = render(ref, -radius, -radius, 2 * radius, 2 * radius,
                       grid, grid, maxiter)

        interior = n < 0
        frac_interior = float(interior.mean())
        escaped = ~interior

        peak = 0.0
        if escaped.any():
            vals = n[escaped]
            # Aim at the 92nd percentile of escape time, not the maximum.
            #
            # Targeting the single highest-escape pixel looks right and is a
            # trap: that pixel sits by definition just under maxiter, which
            # trips the "raise maxiter" rule, which raises the ceiling, which
            # moves the peak up again. The budget ran away to 400k iterations
            # and the descent chased one pathological filament tip. A high
            # percentile lands in the same rich territory without the feedback.
            thresh = float(np.percentile(vals, 92.0))
            cand = escaped & (n >= thresh)
            jj, ii = np.nonzero(cand)
            # among equally-interesting candidates prefer the one nearest the
            # current centre, so the descent path stays smooth
            d2 = (ii - grid / 2.0) ** 2 + (jj - grid / 2.0) ** 2
            k = int(np.argmin(d2))
            ix, jy = int(ii[k]), int(jj[k])
            peak = float(vals.max())
            dx = -radius + 2 * radius * (ix / grid)
            dy = -radius + 2 * radius * (jy / grid)
            cx = cx + mpfr(dx)
            cy = cy + mpfr(dy)

        # Iteration budget tracks depth directly. Deeper really does need more
        # iterations, but the growth is roughly linear in decades, not runaway.
        want = int(2500 + 700 * max(0.0, -math.log10(radius)))
        if frac_interior > 0.60:
            want = int(want * 1.5)
        maxiter = int(min(iter_cap, max(2000, want)))

        radius /= shrink
        step += 1
        if verbose:
            print(f"    r={radius:.3e}  maxiter={maxiter:<7} "
                  f"peak={peak:>9.1f}  interior={frac_interior:5.1%}")
        if progress is not None:
            progress(step, radius, maxiter)

    digits = max(1, int(-math.log10(radius)) + 24)
    return format_coord(cx, digits), format_coord(cy, digits), radius, maxiter


# --------------------------------------------------------------------------
def format_coord(v, digits: int) -> str:
    """Full-precision decimal string, for storing coordinates in a preset file."""
    return format(gmpy2.mpfr(v), f".{int(digits)}f")
