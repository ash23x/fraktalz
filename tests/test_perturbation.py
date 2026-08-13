"""
Ground-truth validation of the perturbation + rebasing kernel.

The whole deep-zoom architecture rests on one claim:

    iterating the *delta* in float64 (or even float32) against a high-precision
    reference orbit, with Zhuoran rebasing, reproduces the same escape iteration
    count as iterating the point itself at full arbitrary precision.

If that holds we can render 1e-300 zooms on a GPU that has never heard of a
300-digit number. If it doesn't, everything downstream is decoration.
So: check it against MPFR, pixel for pixel, at increasing depth.
"""
import math

import numpy as np
import gmpy2
from gmpy2 import mpfr, get_context

BAILOUT2 = 1e12  # |z|^2 escape threshold


def set_prec(bits):
    get_context().precision = bits


# --------------------------------------------------------------------------
# ground truth: full arbitrary precision, one point at a time. Slow, honest.
# --------------------------------------------------------------------------
def gt_escape(cx, cy, maxiter):
    zx = mpfr(0)
    zy = mpfr(0)
    for n in range(maxiter):
        zx2 = zx * zx
        zy2 = zy * zy
        if zx2 + zy2 > BAILOUT2:
            return n
        zy = 2 * zx * zy + cy
        zx = zx2 - zy2 + cx
    return maxiter


# --------------------------------------------------------------------------
# reference orbit: computed once at high precision, stored as float64
# --------------------------------------------------------------------------
def reference_orbit(cx, cy, maxiter):
    zx = mpfr(0)
    zy = mpfr(0)
    ox = np.zeros(maxiter + 1, dtype=np.float64)
    oy = np.zeros(maxiter + 1, dtype=np.float64)
    n = 0
    while n < maxiter:
        zx2 = zx * zx
        zy2 = zy * zy
        zy = 2 * zx * zy + cy
        zx = zx2 - zy2 + cx
        n += 1
        fx = float(zx)
        fy = float(zy)
        ox[n] = fx
        oy[n] = fy
        if fx * fx + fy * fy > BAILOUT2:      # reference escaped; orbit ends here
            return ox[: n + 1], oy[: n + 1]
    return ox, oy


# --------------------------------------------------------------------------
# perturbation + rebasing, vectorised over an array of deltas. This is exactly
# what the GLSL kernel does per fragment, just transposed into SIMD lanes.
# --------------------------------------------------------------------------
def perturb_escape(ox, oy, dcx, dcy, maxiter, dtype=np.float64):
    ox = ox.astype(dtype)
    oy = oy.astype(dtype)
    dcx = np.asarray(dcx, dtype=dtype)
    dcy = np.asarray(dcy, dtype=dtype)
    reflen = len(ox) - 1

    npix = dcx.size
    dzx = np.zeros(npix, dtype=dtype)
    dzy = np.zeros(npix, dtype=dtype)
    m = np.zeros(npix, dtype=np.int64)          # index into the reference orbit
    out = np.full(npix, maxiter, dtype=np.int64)
    live = np.ones(npix, dtype=bool)
    bail = dtype(BAILOUT2)
    two = dtype(2.0)

    # loop index n produces z_{n+1}; z_0 = 0 never escapes, and we only inspect
    # z_1 .. z_{maxiter-1} so the convention matches gt_escape exactly.
    for n in range(maxiter - 1):
        idx = np.flatnonzero(live)
        if idx.size == 0:
            break
        mi = m[idx]
        Zx = ox[mi]
        Zy = oy[mi]
        dx = dzx[idx]
        dy = dzy[idx]

        # dz <- 2*Z*dz + dz^2 + dc          (complex, written out real/imag)
        ndx = two * (Zx * dx - Zy * dy) + (dx * dx - dy * dy) + dcx[idx]
        ndy = two * (Zx * dy + Zy * dx) + two * dx * dy + dcy[idx]

        mi = mi + 1
        zx = ox[mi] + ndx
        zy = oy[mi] + ndy
        zmag = zx * zx + zy * zy
        dmag = ndx * ndx + ndy * ndy

        escaped = zmag > bail
        # Zhuoran rebasing: when the true point is smaller than the delta, or we
        # ran off the end of the stored reference, restart the reference at 0 and
        # promote the true value to the delta. One reference, zero glitches.
        rebase = (~escaped) & ((zmag < dmag) | (mi >= reflen))

        dzx[idx] = np.where(rebase, zx, ndx)
        dzy[idx] = np.where(rebase, zy, ndy)
        m[idx] = np.where(rebase, 0, mi)
        esc = idx[escaped]
        out[esc] = n + 1
        live[esc] = False

    return out


# --------------------------------------------------------------------------
# Deep boundary points by bisection.
#
# The first version of this test used Newton's method to hunt minibrot nuclei
# and it silently converged to c = 0 -- the dead centre of the main cardioid.
# Every sample point was interior, both methods dutifully returned maxiter, and
# the test "passed" while proving absolutely nothing. Hence the degeneracy guard
# in run_case() below: a comparison where every value is identical is not
# evidence of anything.
#
# Bisecting a segment between a known-interior and a known-exterior point
# converges onto the boundary to arbitrary precision, and any window around the
# result straddles wildly varying escape times -- which is exactly the regime
# that breaks a bad perturbation kernel.
# --------------------------------------------------------------------------
def bisect_boundary(ax, ay, bx, by, steps, maxiter):
    """a is inside (never escapes), b is outside. Returns a point on the edge."""
    ax, ay, bx, by = mpfr(ax), mpfr(ay), mpfr(bx), mpfr(by)
    half = mpfr(0.5)
    for _ in range(steps):
        mx = (ax + bx) * half
        my = (ay + by) * half
        if gt_escape(mx, my, maxiter) >= maxiter:
            ax, ay = mx, my
        else:
            bx, by = mx, my
    return ax, ay


# --------------------------------------------------------------------------
def run_case(name, cx, cy, width, maxiter, grid=9, dtype=np.float64):
    """Render a small grid two ways and compare escape counts exactly."""
    half = width / 2
    offs = [mpfr(-1) + mpfr(2) * i / (grid - 1) for i in range(grid)]
    dcx, dcy, gts = [], [], []
    for gy in offs:
        for gx in offs:
            ddx, ddy = gx * half, gy * half
            dcx.append(float(ddx))
            dcy.append(float(ddy))
            gts.append(gt_escape(cx + ddx, cy + ddy, maxiter))

    ox, oy = reference_orbit(cx, cy, maxiter)
    got = perturb_escape(ox, oy, np.array(dcx), np.array(dcy), maxiter, dtype=dtype)
    gts = np.array(gts)

    # --- degeneracy guard: a test whose ground truth is constant proves nothing.
    #
    # What makes a case informative is the number of DISTINCT ground-truth escape
    # counts, nothing else. An earlier version also demanded that between 10% and
    # 98% of samples be interior, which wrongly condemned the deep cases: a
    # filament view where every pixel escapes but does so at 43 different
    # iteration counts is an excellent test, and being told it was "vacuous"
    # was the guard malfunctioning, not the data.
    distinct = len(np.unique(gts))
    escaped_frac = float((gts < maxiter).mean())
    degenerate = distinct < 6

    diff = got != gts
    nbad = int(diff.sum())
    status = "VACUOUS" if degenerate else ("PASS" if nbad == 0 else "FAIL")
    print(f"  [{status:^7}] {name:<32} reflen={len(ox)-1:>6}  "
          f"distinct={distinct:>3}  esc={escaped_frac:5.0%}  "
          f"mismatch {nbad:>3}/{gts.size}")
    if 0 < nbad <= 8:
        for i in np.flatnonzero(diff):
            print(f"            pixel {i}: perturb={got[i]} truth={gts[i]}")
    return (nbad == 0) and not degenerate


def main():
    import os
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from mandelfly.locations import LOCATIONS

    passes, fails, expected = [], [], []
    print("\n=== perturbation + rebasing vs arbitrary-precision ground truth ===\n")

    # ---- shallow / mid, from bisected boundary points.
    # Bisection is fine at these depths: the level-set curve still has curvature
    # comparable to the window. It stops working deep -- see the note below.
    IN, OUT = ("-0.5", "0.0"), ("0.4", "0.4")
    for bits, steps, bmax, width, maxiter, dt, tag in (
        (128,  40, 3000, "1e-6",  3000, np.float32, "fp32"),
        (128,  40, 3000, "1e-6",  3000, np.float64, "fp64"),
        (160,  60, 4000, "1e-14", 4000, np.float32, "fp32"),
        (160,  60, 4000, "1e-14", 4000, np.float64, "fp64"),
    ):
        set_prec(bits)
        bx, by = bisect_boundary(IN[0], IN[1], OUT[0], OUT[1], steps, bmax)
        ok = run_case(f"boundary w={width:<7} {tag}", bx, by, mpfr(width),
                      maxiter, grid=7, dtype=dt)
        label = f"{tag} w={width}"
        if dt is np.float32:
            # EXPECTED FAILURE, and the reason the shader is fp64.
            # 24 mantissa bits cannot carry a delta through thousands of
            # compounding iterations. This is recorded rather than deleted
            # because it is the measurement that drove the design.
            expected.append((label, "PASSED?!" if ok else "failed as expected"))
        else:
            (passes if ok else fails).append(label)

    # ---- deep, at the curated destinations the app actually flies to.
    #
    # These replace an earlier set of bisected deep points that were reported
    # VACUOUS by the degeneracy guard: bisection converges onto a level set of
    # the escape-time function, which is analytic, so at 1e-55 the window
    # contained a straight edge and two flat regions. Ground truth with two
    # distinct values cannot validate anything. Real deep structure has to come
    # from the descent.
    for spot in LOCATIONS[:4]:
        r = mpfr(repr(spot["radius"])) * 3
        bits = int(max(128, -math.log10(spot["radius"]) * 3.33 + 96))
        set_prec(bits)
        ok = run_case(f"{spot['name']:<17} r={spot['radius']:.0e}",
                      mpfr(spot["cx"]), mpfr(spot["cy"]), r, 3000,
                      grid=7, dtype=np.float64)
        (passes if ok else fails).append(f"fp64 {spot['name']}")

    print()
    for label, note in expected:
        print(f"  expected-fail  {label:<16} {note}")
    print(f"\n  {len(passes)} passed, {len(fails)} failed, "
          f"{len(expected)} expected-fail (fp32 deltas -- see shaders.py)")
    if fails:
        print("  FAILURES:", ", ".join(fails))
        return 1
    if any(n == "PASSED?!" for _, n in expected):
        print("  fp32 unexpectedly passed -- re-examine the fp64 decision")
    print("\n  fp64 kernel is sound at every depth tested, to 1e-33.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
