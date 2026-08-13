"""
The reference orbit: the one thing that genuinely needs arbitrary precision.

At a zoom of 1e-300 the view is narrower than float64 can even represent as a
difference between two neighbouring pixels. Perturbation theory sidesteps this
by computing a *single* orbit at full precision (this file) and then iterating
every pixel as a small offset from it in ordinary float64 (kernels.py / the
GLSL shader).

The orbit is stored as plain float64 arrays. That is not a shortcut: the delta
iteration only ever needs Z to float64 accuracy, because the error it injects is
relative to the delta, not to the coordinate. Storing it as float32 does *not*
work -- each iteration would inject ~1e-7 of relative error into the delta and it
compounds over thousands of steps. Measured, not assumed; see tests/.

This computation is inherently serial -- Z_{n+1} depends on Z_n, so no amount of
AVX or CUDA parallelises it. It runs on a background thread so the render loop
never blocks on it.
"""
from __future__ import annotations

import math
import threading
from dataclasses import dataclass, field

import numpy as np

try:
    import gmpy2
    from gmpy2 import mpfr, get_context
    HAVE_GMPY2 = True
except ImportError:  # pragma: no cover - fallback path
    import mpmath
    HAVE_GMPY2 = False

BAILOUT2 = 1.0e12          # |z|^2 escape threshold; generous for smooth colour
ESCAPE_RADIUS = math.sqrt(BAILOUT2)


# --------------------------------------------------------------------------
@dataclass
class Reference:
    cx_str: str                    # full-precision centre, as decimal text
    cy_str: str
    zx: np.ndarray                 # float64 orbit, index 0..n  (z_0 = 0)
    zy: np.ndarray
    length: int                    # usable reference length
    escaped: bool                  # did the reference itself escape?
    maxiter: int
    prec_bits: int

    @property
    def nbytes(self) -> int:
        return self.zx.nbytes + self.zy.nbytes

    def interleaved_f64(self) -> np.ndarray:
        """(n+1, 2) float64, ready to upload as an SSBO of dvec2."""
        out = np.empty((self.length + 1, 2), dtype=np.float64)
        out[:, 0] = self.zx[: self.length + 1]
        out[:, 1] = self.zy[: self.length + 1]
        return out


# --------------------------------------------------------------------------
def prec_for_radius(radius: float, guard_bits: int = 64) -> int:
    """Mantissa bits needed so the view's half-width is still well resolved."""
    if radius <= 0 or not math.isfinite(radius):
        return 64
    digits = max(0.0, -math.log10(radius))
    return int(max(53, digits * 3.3219280948873626 + guard_bits))


# --------------------------------------------------------------------------
def compute_reference(cx_str: str, cy_str: str, maxiter: int,
                      prec_bits: int, cancel: threading.Event | None = None) -> Reference:
    """
    Iterate z -> z^2 + c at `prec_bits` of precision, recording the orbit in
    float64. Stops early if the reference escapes (its orbit is then complete --
    rebasing in the kernel copes with a short reference).
    """
    zx_out = np.zeros(maxiter + 1, dtype=np.float64)
    zy_out = np.zeros(maxiter + 1, dtype=np.float64)

    if prec_bits <= 53:
        # shallow: plain float64 is already exact enough, and ~100x faster
        cx = float(cx_str)
        cy = float(cy_str)
        zx = zy = 0.0
        n = 0
        while n < maxiter:
            zx, zy = zx * zx - zy * zy + cx, 2.0 * zx * zy + cy
            n += 1
            zx_out[n] = zx
            zy_out[n] = zy
            if zx * zx + zy * zy > BAILOUT2:
                return Reference(cx_str, cy_str, zx_out, zy_out, n, True, maxiter, 53)
            if cancel is not None and (n & 0x3FFF) == 0 and cancel.is_set():
                return Reference(cx_str, cy_str, zx_out, zy_out, n, False, maxiter, 53)
        return Reference(cx_str, cy_str, zx_out, zy_out, n, False, maxiter, 53)

    if HAVE_GMPY2:
        get_context().precision = int(prec_bits)
        cx = mpfr(cx_str)
        cy = mpfr(cy_str)
        zx = mpfr(0)
        zy = mpfr(0)
        n = 0
        while n < maxiter:
            zx, zy = zx * zx - zy * zy + cx, 2 * zx * zy + cy
            n += 1
            fx = float(zx)
            fy = float(zy)
            zx_out[n] = fx
            zy_out[n] = fy
            if fx * fx + fy * fy > BAILOUT2:
                return Reference(cx_str, cy_str, zx_out, zy_out, n, True, maxiter, prec_bits)
            if cancel is not None and (n & 0x1FFF) == 0 and cancel.is_set():
                return Reference(cx_str, cy_str, zx_out, zy_out, n, False, maxiter, prec_bits)
        return Reference(cx_str, cy_str, zx_out, zy_out, n, False, maxiter, prec_bits)

    # pragma: no cover -- mpmath fallback, ~20x slower than gmpy2
    mpmath.mp.prec = int(prec_bits)
    cx = mpmath.mpf(cx_str)
    cy = mpmath.mpf(cy_str)
    zx = mpmath.mpf(0)
    zy = mpmath.mpf(0)
    n = 0
    while n < maxiter:
        zx, zy = zx * zx - zy * zy + cx, 2 * zx * zy + cy
        n += 1
        fx = float(zx)
        fy = float(zy)
        zx_out[n] = fx
        zy_out[n] = fy
        if fx * fx + fy * fy > BAILOUT2:
            return Reference(cx_str, cy_str, zx_out, zy_out, n, True, maxiter, prec_bits)
        if cancel is not None and (n & 0x1FFF) == 0 and cancel.is_set():
            return Reference(cx_str, cy_str, zx_out, zy_out, n, False, maxiter, prec_bits)
    return Reference(cx_str, cy_str, zx_out, zy_out, n, False, maxiter, prec_bits)


# --------------------------------------------------------------------------
class ReferenceService:
    """
    Background reference-orbit computer.

    The render loop asks for a reference at some centre/depth and keeps drawing
    with whatever it already has. When a better one is ready it swaps in. The
    camera never stalls waiting for a 300-digit calculation.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._cancel = threading.Event()
        self.current: Reference | None = None
        self.pending_key = None
        self.generation = 0
        self.busy = False
        self.last_ms = 0.0

    def request(self, cx_str: str, cy_str: str, maxiter: int, prec_bits: int):
        key = (cx_str, cy_str, maxiter, prec_bits)
        with self._lock:
            if key == self.pending_key:
                return
            if self.current is not None and key == (self.current.cx_str, self.current.cy_str,
                                                    self.current.maxiter, self.current.prec_bits):
                return
            self.pending_key = key
            self._cancel.set()                      # tell any in-flight job to stop
            cancel = threading.Event()
            self._cancel = cancel
            self.busy = True

        def work():
            import time
            t0 = time.perf_counter()
            try:
                ref = compute_reference(cx_str, cy_str, maxiter, prec_bits, cancel)
            except Exception:
                with self._lock:
                    self.busy = False
                return
            with self._lock:
                if not cancel.is_set():
                    self.current = ref
                    self.generation += 1
                    self.last_ms = (time.perf_counter() - t0) * 1e3
                self.busy = False

        t = threading.Thread(target=work, daemon=True, name="reference-orbit")
        self._thread = t
        t.start()

    def get(self) -> Reference | None:
        with self._lock:
            return self.current

    def shutdown(self):
        self._cancel.set()
