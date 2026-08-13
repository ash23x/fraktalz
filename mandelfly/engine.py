"""
The render engine: everything that touches the GPU, deliberately kept free of
any window or input code so it can be driven headlessly by the test suite.

Smoothness here is not "make the shader fast enough" -- at a 1e-40 zoom with a
float64 delta path on a consumer card, it will not be. It is three mechanisms
working together:

  * adaptive resolution -- while the camera moves, render at whatever fraction of
    native resolution hits the frame-time budget, and upscale. Motion hides the
    softness; a dropped frame does not.

  * progressive accumulation -- the moment the camera stops, keep firing jittered
    samples into a float accumulation buffer. The image sharpens and the
    aliasing melts away over the next second while you sit there.

  * background reference orbits -- the arbitrary-precision work never happens on
    the render thread. The camera keeps flying on the previous orbit and swaps
    when the new one lands.
"""
from __future__ import annotations

import math
import threading
import time

import numpy as np
import moderngl

from . import shaders
from .camera import Camera, auto_maxiter
from .colour import PALETTES, PALETTE_NAMES
from .reference import ReferenceService, BAILOUT2, prec_for_radius

# Above this radius, plain float32 iteration of c resolves the view perfectly
# well and runs ~64x faster than the fp64 delta path on consumer NVIDIA silicon.
DIRECT_TIER_RADIUS = 3.0e-3


class Steering:
    """
    Background autopilot navigator.

    Renders a small probe grid on the CPU using the *existing* reference orbit
    and reports where the interesting structure is, as an offset in units of the
    view radius. The main loop feeds that in as a gentle pan, which is what turns
    a straight zoom into something that looks like flying.

    Aims at the 92nd percentile of escape time rather than the maximum: the
    maximum is a single filament tip that costs unbounded iterations and drags
    the whole budget with it.
    """

    def __init__(self, grid=64):
        self.grid = grid
        self.offset = (0.0, 0.0)
        self.confidence = 0.0
        self.busy = False
        self._lock = threading.Lock()
        self._last = 0.0

    def maybe_probe(self, ref, radius, maxiter, min_interval=0.25):
        now = time.perf_counter()
        with self._lock:
            if self.busy or ref is None or now - self._last < min_interval:
                return
            self.busy = True
            self._last = now

        def work():
            try:
                from .kernels import render
                g = self.grid
                n, _de = render(ref, -radius, -radius, 2 * radius, 2 * radius,
                                g, g, min(maxiter, 20000))
                esc = n >= 0
                if esc.sum() < g * g * 0.02:
                    with self._lock:
                        self.offset = (0.0, 0.0)
                        self.confidence = 0.0
                    return
                vals = n[esc]
                thresh = float(np.percentile(vals, 92.0))
                cand = esc & (n >= thresh)
                jj, ii = np.nonzero(cand)
                # weight toward candidates near the centre so steering is a
                # nudge rather than a swerve
                fx = (ii + 0.5) / g * 2.0 - 1.0
                fy = (jj + 0.5) / g * 2.0 - 1.0
                w = 1.0 / (1.0 + 3.0 * (fx * fx + fy * fy))
                ox = float(np.sum(fx * w) / np.sum(w))
                oy = float(np.sum(fy * w) / np.sum(w))
                with self._lock:
                    self.offset = (ox, oy)
                    self.confidence = float(cand.mean())
            except Exception:
                pass
            finally:
                with self._lock:
                    self.busy = False

        threading.Thread(target=work, daemon=True, name="steering").start()

    def take(self):
        with self._lock:
            return self.offset, self.confidence


class Engine:
    def __init__(self, ctx: moderngl.Context, width: int, height: int,
                 target_frame_ms: float = 12.0, max_samples: int = 256):
        self.ctx = ctx
        self.width = width
        self.height = height
        self.target_frame_ms = target_frame_ms
        self.max_samples = max_samples

        self.prog_perturb = ctx.program(vertex_shader=shaders.VERTEX,
                                        fragment_shader=shaders.FRAGMENT_PERTURB)
        self.prog_direct = ctx.program(vertex_shader=shaders.VERTEX,
                                       fragment_shader=shaders.FRAGMENT_DIRECT)
        self.prog_accum = ctx.program(vertex_shader=shaders.VERTEX,
                                      fragment_shader=shaders.FRAGMENT_ACCUM)
        self.prog_present = ctx.program(vertex_shader=shaders.VERTEX,
                                        fragment_shader=shaders.FRAGMENT_PRESENT)
        self.vao_perturb = ctx.vertex_array(self.prog_perturb, [])
        self.vao_direct = ctx.vertex_array(self.prog_direct, [])
        self.vao_accum = ctx.vertex_array(self.prog_accum, [])
        self.vao_present = ctx.vertex_array(self.prog_present, [])

        self.refs = ReferenceService()
        self.steering = Steering()
        self._ssbo = None
        self._ssbo_gen = -1
        self._ref_centre = None            # (cx_str, cy_str, radius) of current orbit
        self._ref_radius = None

        # look
        self.palette = "ember"
        self.cycle = 4.0
        self.colour_offset = 0.0
        self.de_strength = 0.78
        self.glow = 0.12
        self.gamma = 1.05
        self.exposure = 1.12
        self.vignette = 0.22
        self.interior = (0.02, 0.02, 0.035)
        self.iter_boost = 1.0

        # adaptive state
        self.scale = 1.0
        self.min_scale = 0.18
        self.samples = 0
        self.last_gpu_ms = 0.0
        self.maxiter = 512
        self.tier = "direct"
        self.quality_locked = False

        self._alloc(width, height)
        self._query = None
        try:
            self._query = ctx.query(time=True)
        except Exception:
            self._query = None

    # ------------------------------------------------------------------
    def _alloc(self, w, h):
        for name in ("tex_frac", "tex_accum", "tex_accum2"):
            old = getattr(self, name, None)
            if old is not None:
                old.release()
        for name in ("fbo_frac", "fbo_accum", "fbo_accum2"):
            old = getattr(self, name, None)
            if old is not None:
                old.release()
        self.tex_frac = self.ctx.texture((w, h), 4, dtype="f2")
        self.tex_accum = self.ctx.texture((w, h), 4, dtype="f4")
        self.tex_accum2 = self.ctx.texture((w, h), 4, dtype="f4")
        for t in (self.tex_frac, self.tex_accum, self.tex_accum2):
            t.filter = (moderngl.LINEAR, moderngl.LINEAR)
            t.repeat_x = t.repeat_y = False
        self.fbo_frac = self.ctx.framebuffer([self.tex_frac])
        self.fbo_accum = self.ctx.framebuffer([self.tex_accum])
        self.fbo_accum2 = self.ctx.framebuffer([self.tex_accum2])
        self.width, self.height = w, h

    def resize(self, w, h):
        if (w, h) != (self.width, self.height) and w > 0 and h > 0:
            self._alloc(w, h)
            self.samples = 0

    # ------------------------------------------------------------------
    def _set_colour_uniforms(self, prog):
        a, b, c, d = PALETTES[self.palette]
        for k, v in (("uPalA", a), ("uPalB", b), ("uPalC", c), ("uPalD", d),
                     ("uCycle", self.cycle), ("uColourOffset", self.colour_offset),
                     ("uDEStrength", self.de_strength),
                     ("uInteriorColour", self.interior), ("uGlow", self.glow)):
            if k in prog:
                prog[k].value = v

    # ------------------------------------------------------------------
    def ensure_reference(self, cam: Camera, force=False):
        """
        Decide whether the current reference orbit is still good enough, and if
        not kick off a new one in the background. Never blocks.
        """
        # Set the iteration budget HERE, not inside render_fractal.
        #
        # It used to be computed at draw time, which meant the very first
        # reference orbit was built with the constructor's placeholder budget of
        # 512 -- fine at the default view, catastrophic at 1e-25 where the orbit
        # then capped every pixel at 512 iterations and the whole frame came out
        # black. Interactively it self-corrected within a frame or two and was
        # nearly invisible; in the offline export path, which builds an Engine
        # and renders once, it was fatal.
        self.maxiter = auto_maxiter(cam.radius, self.iter_boost)
        radius = cam.radius
        if radius > DIRECT_TIER_RADIUS:
            return
        need_new = force or self._ref_centre is None
        if not need_new:
            try:
                import gmpy2
                from gmpy2 import mpfr, get_context
                get_context().precision = cam.precision_bits() + 96
                dx = float(cam.cx - mpfr(self._ref_centre[0]))
                dy = float(cam.cy - mpfr(self._ref_centre[1]))
                drift = math.hypot(dx, dy)
                # a reference stays usable while it is inside the frame; past
                # that the deltas grow and rebasing works harder for no reason
                if drift > radius * 0.6 or radius < self._ref_radius * 0.2:
                    need_new = True
            except Exception:
                need_new = True

        if need_new:
            cxs, cys = cam.centre_strings()
            prec = max(64, prec_for_radius(radius))
            self.refs.request(cxs, cys, self.maxiter, prec)
            self._ref_centre = (cxs, cys)
            self._ref_radius = radius

    def _upload_reference(self):
        ref = self.refs.get()
        if ref is None:
            return None
        if self.refs.generation != self._ssbo_gen:
            data = ref.interleaved_f64()
            if self._ssbo is not None:
                self._ssbo.release()
            self._ssbo = self.ctx.buffer(data.tobytes())
            self._ssbo_gen = self.refs.generation
        return ref

    # ------------------------------------------------------------------
    def render_fractal(self, cam: Camera, rw: int, rh: int, jitter=(0.0, 0.0)):
        """Draw one sample of the fractal into fbo_frac's top-left rw x rh."""
        self.maxiter = auto_maxiter(cam.radius, self.iter_boost)
        corner, step_x, step_y, pixel_scale = cam.basis(rw, rh)

        self.fbo_frac.use()
        self.ctx.viewport = (0, 0, rw, rh)

        if cam.radius > DIRECT_TIER_RADIUS:
            self.tier = "direct fp32"
            p = self.prog_direct
            self._set_colour_uniforms(p)
            cxf = float(cam.cx)
            cyf = float(cam.cy)
            vals = {
                "uResolution": (rw, rh),
                "uCentre": (cxf + corner[0], cyf + corner[1]),
                "uStepX": step_x, "uStepY": step_y,
                "uMaxIter": self.maxiter,
                "uPixelScale": pixel_scale,
                "uJitter": jitter,
                "uBailout2": float(BAILOUT2),
                "uRawOutput": 0,
            }
            for k, v in vals.items():
                if k in p:
                    p[k].value = v
            self.vao_direct.render(moderngl.TRIANGLES, vertices=3)
            return True

        ref = self._upload_reference()
        if ref is None:
            return False
        self.tier = "perturb fp64"

        # the reference may sit slightly off the view centre; express the
        # window relative to it
        try:
            import gmpy2
            from gmpy2 import mpfr, get_context
            get_context().precision = cam.precision_bits() + 96
            offx = float(cam.cx - mpfr(ref.cx_str))
            offy = float(cam.cy - mpfr(ref.cy_str))
        except Exception:
            offx = offy = 0.0

        p = self.prog_perturb
        self._set_colour_uniforms(p)
        vals = {
            "uResolution": (rw, rh),
            "uCorner": (corner[0] + offx, corner[1] + offy),
            "uStepX": step_x, "uStepY": step_y,
            "uRefLen": ref.length,
            "uMaxIter": min(self.maxiter, ref.maxiter),
            "uPixelScale": pixel_scale,
            "uJitter": jitter,
            "uBailout2": float(BAILOUT2),
            "uRawOutput": 0,
        }
        for k, v in vals.items():
            if k in p:
                p[k].value = v
        self._ssbo.bind_to_storage_buffer(0)
        self.vao_perturb.render(moderngl.TRIANGLES, vertices=3)
        return True

    # ------------------------------------------------------------------
    def frame(self, cam: Camera, moving: bool):
        """
        Render one frame's worth of work. Returns (image_texture, uv_scale).
        """
        self.ensure_reference(cam)
        t0 = time.perf_counter()
        if self._query is not None:
            self._query.__enter__()

        if moving:
            # Motion: one cheap sample at reduced resolution, upscaled on
            # present. Softness during movement is invisible; a stutter is not.
            self.samples = 0
            rw = max(64, int(self.width * self.scale))
            rh = max(64, int(self.height * self.scale))
            self.render_fractal(cam, rw, rh)
            img, uv = self.tex_frac, (rw / self.width, rh / self.height)
        elif self.samples >= self.max_samples:
            img, uv = self.tex_accum, (1.0, 1.0)          # fully converged
        else:
            # Still: keep firing sub-pixel-jittered samples into the float
            # accumulator. tex_accum always holds the running mean; we render
            # the blend into tex_accum2 and swap, so there is never a
            # read-and-write-the-same-texture hazard.
            rw, rh = self.width, self.height
            t_batch = time.perf_counter()
            while self.samples < self.max_samples:
                k = self.samples
                jx = _halton(k + 1, 2) - 0.5
                jy = _halton(k + 1, 3) - 0.5
                if not self.render_fractal(cam, rw, rh, jitter=(jx, jy)):
                    break
                self.fbo_accum2.use()
                self.ctx.viewport = (0, 0, rw, rh)
                self.tex_accum.use(0)
                self.tex_frac.use(1)
                for key, val in (("uPrev", 0), ("uNew", 1),
                                 ("uBlend", 1.0 if k == 0 else 1.0 / (k + 1.0))):
                    if key in self.prog_accum:
                        self.prog_accum[key].value = val
                self.vao_accum.render(moderngl.TRIANGLES, vertices=3)
                self.tex_accum, self.tex_accum2 = self.tex_accum2, self.tex_accum
                self.fbo_accum, self.fbo_accum2 = self.fbo_accum2, self.fbo_accum
                self.samples += 1
                self.ctx.finish()
                if (time.perf_counter() - t_batch) * 1e3 > self.target_frame_ms * 0.85:
                    break
            img, uv = self.tex_accum, (1.0, 1.0)

        if self._query is not None:
            self._query.__exit__(None, None, None)
            try:
                self.last_gpu_ms = self._query.elapsed / 1e6
            except Exception:
                self.last_gpu_ms = (time.perf_counter() - t0) * 1e3
        else:
            self.last_gpu_ms = (time.perf_counter() - t0) * 1e3

        if moving and not self.quality_locked:
            self._adapt()
        return img, uv

    def _adapt(self):
        """Nudge render scale toward the frame-time budget."""
        ms = max(0.05, self.last_gpu_ms)
        err = self.target_frame_ms / ms
        # scale area by the error, so linear scale moves as its square root
        target = self.scale * math.sqrt(max(0.25, min(4.0, err)))
        # rise slowly, fall fast: a stutter should be corrected immediately
        rate = 0.10 if target > self.scale else 0.5
        self.scale += (target - self.scale) * rate
        self.scale = max(self.min_scale, min(1.0, self.scale))

    # ------------------------------------------------------------------
    def present(self, target_fbo, image, uv_scale, overlay=None):
        target_fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)
        image.use(0)
        p = self.prog_present
        if "uImage" in p:
            p["uImage"].value = 0
        if overlay is not None:
            overlay.use(1)
            if "uOverlay" in p:
                p["uOverlay"].value = 1
            if "uShowOverlay" in p:
                p["uShowOverlay"].value = 1
        elif "uShowOverlay" in p:
            p["uShowOverlay"].value = 0
        for k, v in (("uGamma", self.gamma), ("uExposure", self.exposure),
                     ("uVignette", self.vignette), ("uUVScale", uv_scale)):
            if k in p:
                p[k].value = v
        self.vao_present.render(moderngl.TRIANGLES, vertices=3)

    def cycle_palette(self, delta=1):
        i = PALETTE_NAMES.index(self.palette)
        self.palette = PALETTE_NAMES[(i + delta) % len(PALETTE_NAMES)]
        self.samples = 0

    def release(self):
        self.refs.shutdown()


def _halton(index, base):
    f, r, i = 1.0, 0.0, index
    while i > 0:
        f /= base
        r += f * (i % base)
        i //= base
    return r
