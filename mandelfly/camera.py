"""
The camera, and the thing that flies it.

Two ideas do most of the work here.

*Zoom lives in log space.* Linearly interpolating a radius from 1.0 to 1e-30
spends 99.99% of the flight in the first decade and then falls off a cliff. What
reads as constant speed to a human eye is a constant number of *doublings* per
second, so the camera stores log(radius) and moves it at a constant rate. Every
zoom decade then takes exactly as long as every other one, which is what makes a
descent hypnotic rather than lurching.

*The centre is arbitrary precision, the motion is not.* Position needs hundreds
of digits at depth; velocity and smoothing only ever need a few. So the centre is
an mpfr and every increment applied to it is an ordinary float scaled by the
current radius. Precision grows automatically as we descend.
"""
from __future__ import annotations

import math
import threading

import gmpy2
from gmpy2 import mpfr, get_context

from .reference import prec_for_radius


def _set_prec(bits):
    get_context().precision = int(bits)


class Camera:
    MIN_LOG_RADIUS = math.log(1e-300)      # fp64 delta floor; see README
    MAX_LOG_RADIUS = math.log(4.0)

    def __init__(self, cx="-0.5", cy="0.0", radius=1.6):
        _set_prec(prec_for_radius(float(radius)) + 96)
        self.cx = mpfr(cx)
        self.cy = mpfr(cy)
        self.log_radius = math.log(float(radius))
        self.angle = 0.0

        # smoothed targets -- input sets the target, the camera eases toward it
        self.t_log_radius = self.log_radius
        self.t_angle = 0.0
        self.t_dx = 0.0                     # pending pan, in units of radius
        self.t_dy = 0.0

        self.zoom_rate = 0.0                # doublings per second, signed
        self.spin_rate = 0.0
        self.smoothing = 9.0                # higher = snappier
        self.moved = True
        self.lock = threading.Lock()

    # ------------------------------------------------------------------
    @property
    def radius(self) -> float:
        return math.exp(self.log_radius)

    @property
    def depth_decades(self) -> float:
        return max(0.0, -self.log_radius / 2.302585092994046)

    @property
    def zoom_power(self) -> float:
        """How many 2x magnifications from the full set."""
        return max(0.0, (math.log(1.6) - self.log_radius) / 0.6931471805599453)

    def precision_bits(self) -> int:
        return prec_for_radius(self.radius)

    def centre_strings(self):
        digits = int(self.depth_decades) + 24
        _set_prec(self.precision_bits() + 96)
        return (format(self.cx, f".{digits}f"), format(self.cy, f".{digits}f"))

    # ------------------------------------------------------------------
    def pan_pixels(self, dx_frac, dy_frac):
        """Queue a pan expressed as a fraction of the view radius."""
        self.t_dx += dx_frac
        self.t_dy += dy_frac

    def zoom_at(self, amount, fx=0.0, fy=0.0):
        """
        Zoom by `amount` doublings, keeping the point at view-relative offset
        (fx, fy) fixed on screen -- i.e. zoom toward the cursor, not the centre.
        """
        scale = 2.0 ** (-amount)
        self.t_log_radius = min(self.MAX_LOG_RADIUS,
                                max(self.MIN_LOG_RADIUS,
                                    self.t_log_radius - amount * 0.6931471805599453))
        self.t_dx += fx * (1.0 - scale)
        self.t_dy += fy * (1.0 - scale)

    def set_target(self, cx_str, cy_str, radius):
        _set_prec(prec_for_radius(float(radius)) + 96)
        self.cx = mpfr(cx_str)
        self.cy = mpfr(cy_str)
        self.log_radius = self.t_log_radius = math.log(float(radius))
        self.t_dx = self.t_dy = 0.0
        self.moved = True

    # ------------------------------------------------------------------
    def update(self, dt):
        """Ease everything toward its target. Returns True if anything moved."""
        dt = max(1e-4, min(0.1, dt))
        k = 1.0 - math.exp(-self.smoothing * dt)

        if self.zoom_rate != 0.0:
            self.t_log_radius = min(
                self.MAX_LOG_RADIUS,
                max(self.MIN_LOG_RADIUS,
                    self.t_log_radius - self.zoom_rate * dt * 0.6931471805599453))
        if self.spin_rate != 0.0:
            self.t_angle += self.spin_rate * dt

        moved = False

        # --- pan: apply a fraction of the queued offset, scaled by radius
        if abs(self.t_dx) > 1e-12 or abs(self.t_dy) > 1e-12:
            step_x = self.t_dx * k
            step_y = self.t_dy * k
            self.t_dx -= step_x
            self.t_dy -= step_y
            r = self.radius
            _set_prec(self.precision_bits() + 96)
            self.cx += mpfr(step_x * r)
            self.cy += mpfr(step_y * r)
            moved = True

        # --- zoom
        d = self.t_log_radius - self.log_radius
        if abs(d) > 1e-9:
            self.log_radius += d * k
            moved = True

        # --- roll
        da = self.t_angle - self.angle
        if abs(da) > 1e-6:
            self.angle += da * k
            moved = True

        self.moved = moved
        return moved

    # ------------------------------------------------------------------
    def basis(self, width, height):
        """Per-pixel step vectors (carrying rotation) and the top-left corner."""
        r = self.radius
        aspect = width / max(1, height)
        span_x = 2.0 * r * (aspect if aspect >= 1.0 else 1.0)
        span_y = 2.0 * r * (1.0 if aspect >= 1.0 else 1.0 / aspect)
        ca = math.cos(self.angle)
        sa = math.sin(self.angle)
        sx = span_x / width
        sy = span_y / height
        step_x = (sx * ca, sx * sa)
        step_y = (-sy * sa, sy * ca)
        corner = (-0.5 * span_x * ca + 0.5 * span_y * sa,
                  -0.5 * span_x * sa - 0.5 * span_y * ca)
        return corner, step_x, step_y, max(sx, sy)


# --------------------------------------------------------------------------
def auto_maxiter(radius: float, boost: float = 1.0) -> int:
    """
    Iteration budget as a function of depth.

    Calibrated empirically during the descent work: escape times grow roughly
    linearly in zoom decades, and an earlier version that adapted the budget from
    the observed peak escape count ran away to 400,000 iterations chasing a
    single pathological filament. A flat formula in depth is both cheaper and
    better behaved.
    """
    decades = max(0.0, -math.log10(max(radius, 1e-320)))
    return int(max(256, min(200_000, (700 + 620 * decades) * boost)))


# --------------------------------------------------------------------------
class Autopilot:
    """
    Cinematic flight.

    Three phases so that arriving somewhere far away does not look like a jump
    cut: pull back until the destination is in frame, pan across at constant
    altitude, then dive at a constant number of doublings per second.

    ENDLESS mode has no destination at all -- it just keeps falling, and a
    background probe (see engine.Steering) keeps nudging the centre toward
    whatever structure is richest in the current frame. That is the difference
    between a zoom and a flight.
    """
    IDLE, PULLBACK, PAN, DIVE, ENDLESS = "idle", "pullback", "pan", "dive", "endless"

    def __init__(self, camera: Camera):
        self.cam = camera
        self.state = self.IDLE
        self.target = None                  # (mpfr cx, mpfr cy, log_radius)
        self.dive_rate = 0.9                # doublings per second
        self.pan_rate = 1.6

    @property
    def active(self):
        return self.state != self.IDLE

    def stop(self):
        self.state = self.IDLE
        self.cam.zoom_rate = 0.0
        self.target = None

    def fly_to(self, cx_str, cy_str, radius, dive_rate=None):
        _set_prec(prec_for_radius(float(radius)) + 128)
        self.target = (mpfr(cx_str), mpfr(cy_str), math.log(float(radius)))
        if dive_rate:
            self.dive_rate = dive_rate
        self.state = self.PULLBACK

    def endless(self, dive_rate=None):
        if dive_rate:
            self.dive_rate = dive_rate
        self.state = self.ENDLESS
        self.target = None

    # ------------------------------------------------------------------
    def update(self, dt):
        cam = self.cam
        if self.state == self.IDLE:
            return
        if self.state == self.ENDLESS:
            cam.zoom_rate = self.dive_rate
            return

        tx, ty, tlog = self.target
        _set_prec(cam.precision_bits() + 128)
        offx = float(tx - cam.cx)
        offy = float(ty - cam.cy)
        dist = math.hypot(offx, offy)
        r = cam.radius

        if self.state == self.PULLBACK:
            if dist < r * 0.35 or cam.log_radius >= cam.MAX_LOG_RADIUS - 1e-6:
                self.state = self.PAN
            else:
                cam.zoom_rate = -self.pan_rate       # negative = zoom out
                return

        if self.state == self.PAN:
            cam.zoom_rate = 0.0
            if dist < r * 0.02:
                cam.cx, cam.cy = tx, ty
                self.state = self.DIVE
            else:
                cam.pan_pixels(offx / r * 0.6, offy / r * 0.6)
                return

        if self.state == self.DIVE:
            # keep the destination pinned to the centre as we fall
            cam.cx, cam.cy = tx, ty
            if cam.log_radius <= tlog + 1e-6:
                cam.zoom_rate = 0.0
                self.state = self.IDLE
            else:
                remaining = (cam.log_radius - tlog) / 0.6931471805599453
                # ease out over the last couple of doublings so the arrival
                # settles instead of slamming to a stop
                cam.zoom_rate = self.dive_rate * min(1.0, max(0.08, remaining / 2.0))
