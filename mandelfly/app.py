"""
The window: GLFW, input, HUD, main loop.

Deliberately thin. Everything that decides what a pixel looks like lives in
engine.py, which is driven headlessly by the test suite; this file only opens a
window, reads the keyboard, and calls it sixty (or a hundred and forty-four)
times a second.
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np

try:
    import glfw
except Exception:  # pragma: no cover
    glfw = None  # missing lib raises OSError, not ImportError
import moderngl
from PIL import Image, ImageDraw, ImageFont

from .camera import Camera, Autopilot, auto_maxiter
from .engine import Engine, DIRECT_TIER_RADIUS
from .colour import PALETTE_NAMES
from . import locations as loc

HELP = [
    ("wheel / W S",     "zoom toward cursor / in / out"),
    ("drag / arrows",   "pan"),
    ("Q E",             "roll"),
    ("SPACE",           "endless dive autopilot (steers itself)"),
    ("1 - 6",           "fly to a curated deep location"),
    ("0",               "home, the whole set"),
    ("N",               "dive to somewhere new (auto-descend)"),
    ("P / [ ]",         "palette / colour cycle"),
    (", .",             "shift colour phase"),
    ("- =",             "iteration budget down / up"),
    ("TAB",             "hide this panel"),
    ("F",               "fullscreen"),
    ("ENTER",           "save a PNG"),
    ("BACKSPACE",       "reset view"),
    ("ESC",             "quit"),
]


def _font(size):
    for path in (r"C:\Windows\Fonts\consola.ttf", r"C:\Windows\Fonts\segoeui.ttf",
                 "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
                 "/System/Library/Fonts/Menlo.ttc"):
        try:
            return ImageFont.truetype(path, size)
        except Exception:
            continue
    return ImageFont.load_default()


class App:
    def __init__(self, width=1600, height=900, fullscreen=False, vsync=True,
                 start=None, target_fps=60):
        if glfw is None:
            raise SystemExit("glfw is not installed.  pip install glfw")
        if not glfw.init():
            raise SystemExit("could not initialise GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        glfw.window_hint(glfw.SAMPLES, 0)

        monitor = glfw.get_primary_monitor() if fullscreen else None
        if fullscreen:
            mode = glfw.get_video_mode(monitor)
            width, height = mode.size.width, mode.size.height
        self.win = glfw.create_window(width, height, "mandelfly", monitor, None)
        if not self.win:
            glfw.terminate()
            raise SystemExit("could not create an OpenGL 4.3 window")
        glfw.make_context_current(self.win)
        glfw.swap_interval(1 if vsync else 0)

        self.ctx = moderngl.create_context()
        fbw, fbh = glfw.get_framebuffer_size(self.win)
        self.engine = Engine(self.ctx, fbw, fbh,
                             target_frame_ms=1000.0 / max(20, target_fps) * 0.72)
        self.cam = Camera()
        self.auto = Autopilot(self.cam)
        if start:
            self.cam.set_target(start["cx"], start["cy"], start["radius"])

        self.show_hud = True
        self.fullscreen = fullscreen
        self._win_geom = (width, height)
        self.dragging = False
        self.last_cursor = (0.0, 0.0)
        self.overlay_tex = None
        self._hud_t = 0.0
        self._fps = 0.0
        self._frames = 0
        self._fps_t = time.perf_counter()
        self.status = "ready"
        self.font = _font(15)
        self.font_small = _font(13)

        glfw.set_key_callback(self.win, self._on_key)
        glfw.set_scroll_callback(self.win, self._on_scroll)
        glfw.set_mouse_button_callback(self.win, self._on_mouse)
        glfw.set_cursor_pos_callback(self.win, self._on_cursor)
        glfw.set_framebuffer_size_callback(self.win, self._on_resize)

    # ------------------------------------------------------------------
    def _on_resize(self, win, w, h):
        if w > 0 and h > 0:
            self.engine.resize(w, h)

    def _cursor_frac(self):
        """Cursor position as an offset from centre, in units of the view radius."""
        cx, cy = glfw.get_cursor_pos(self.win)
        ww, wh = glfw.get_window_size(self.win)
        if ww == 0 or wh == 0:
            return 0.0, 0.0
        aspect = ww / wh
        fx = (cx / ww - 0.5) * 2.0 * (aspect if aspect >= 1 else 1.0)
        fy = (cy / wh - 0.5) * 2.0 * (1.0 if aspect >= 1 else 1.0 / aspect)
        ca, sa = math.cos(self.cam.angle), math.sin(self.cam.angle)
        return fx * ca - fy * sa, fx * sa + fy * ca

    def _on_scroll(self, win, dx, dy):
        self.auto.stop()
        fx, fy = self._cursor_frac()
        self.cam.zoom_at(dy * 0.55, fx, fy)

    def _on_mouse(self, win, button, action, mods):
        if button == glfw.MOUSE_BUTTON_LEFT:
            self.dragging = action == glfw.PRESS
            self.last_cursor = glfw.get_cursor_pos(self.win)
            if self.dragging:
                self.auto.stop()

    def _on_cursor(self, win, x, y):
        if not self.dragging:
            return
        ww, wh = glfw.get_window_size(self.win)
        dx = (x - self.last_cursor[0]) / max(1, wh) * 2.0
        dy = (y - self.last_cursor[1]) / max(1, wh) * 2.0
        self.last_cursor = (x, y)
        ca, sa = math.cos(self.cam.angle), math.sin(self.cam.angle)
        self.cam.pan_pixels(-(dx * ca - dy * sa), -(dx * sa + dy * ca))

    # ------------------------------------------------------------------
    def _on_key(self, win, key, scancode, action, mods):
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        eng, cam = self.engine, self.cam

        if key == glfw.KEY_ESCAPE:
            glfw.set_window_should_close(win, True)
        elif key == glfw.KEY_SPACE:
            if self.auto.state == self.auto.ENDLESS:
                self.auto.stop(); self.status = "autopilot off"
            else:
                self.auto.endless(); self.status = "endless dive"
        elif glfw.KEY_1 <= key <= glfw.KEY_6:
            d = loc.by_index(key - glfw.KEY_1)
            self.auto.fly_to(d["cx"], d["cy"], d["radius"])
            self.status = f"flying to {d['name']}"
        elif key == glfw.KEY_0:
            self.auto.stop()
            cam.set_target(loc.HOME["cx"], loc.HOME["cy"], loc.HOME["radius"])
            self.status = "home"
        elif key == glfw.KEY_N:
            self._auto_descend()
        elif key == glfw.KEY_P:
            eng.cycle_palette(-1 if mods & glfw.MOD_SHIFT else 1)
            self.status = f"palette: {eng.palette}"
        elif key == glfw.KEY_LEFT_BRACKET:
            eng.cycle = max(0.4, eng.cycle / 1.15); eng.samples = 0
        elif key == glfw.KEY_RIGHT_BRACKET:
            eng.cycle = min(120.0, eng.cycle * 1.15); eng.samples = 0
        elif key == glfw.KEY_COMMA:
            eng.colour_offset -= 0.04; eng.samples = 0
        elif key == glfw.KEY_PERIOD:
            eng.colour_offset += 0.04; eng.samples = 0
        elif key == glfw.KEY_MINUS:
            eng.iter_boost = max(0.15, eng.iter_boost / 1.3)
            eng.samples = 0; self.status = f"iteration budget x{eng.iter_boost:.2f}"
        elif key == glfw.KEY_EQUAL:
            eng.iter_boost = min(24.0, eng.iter_boost * 1.3)
            eng.samples = 0; self.status = f"iteration budget x{eng.iter_boost:.2f}"
        elif key == glfw.KEY_TAB:
            self.show_hud = not self.show_hud
        elif key == glfw.KEY_F:
            self._toggle_fullscreen()
        elif key == glfw.KEY_ENTER:
            self._screenshot()
        elif key == glfw.KEY_BACKSPACE:
            self.auto.stop()
            cam.set_target(loc.HOME["cx"], loc.HOME["cy"], loc.HOME["radius"])
            cam.angle = cam.t_angle = 0.0
        elif key == glfw.KEY_G:
            eng.de_strength = 0.0 if eng.de_strength > 0 else 0.78
            eng.samples = 0

    def _toggle_fullscreen(self):
        self.fullscreen = not self.fullscreen
        mon = glfw.get_primary_monitor()
        mode = glfw.get_video_mode(mon)
        if self.fullscreen:
            self._win_geom = glfw.get_window_size(self.win)
            glfw.set_window_monitor(self.win, mon, 0, 0,
                                    mode.size.width, mode.size.height,
                                    mode.refresh_rate)
        else:
            w, h = self._win_geom
            glfw.set_window_monitor(self.win, None, 80, 60, w, h, 0)
        glfw.swap_interval(1)

    def _auto_descend(self):
        """Hand the current view to the descent finder, then fly to what it finds."""
        import threading
        from .deepzoom import descend

        if getattr(self, "_descending", False):
            return
        self._descending = True
        self.status = "searching for somewhere deeper..."
        cxs, cys = self.cam.centre_strings()
        r = self.cam.radius
        target = max(1e-290, r * 1e-8)

        def work():
            try:
                fcx, fcy, fr, mi = descend(cxs, cys, r, target)
                self.auto.fly_to(fcx, fcy, fr)
                self.status = f"descending to 1e{math.log10(fr):.0f}"
            except Exception as exc:
                self.status = f"descent failed: {exc}"
            finally:
                self._descending = False

        threading.Thread(target=work, daemon=True, name="descend").start()

    def _screenshot(self):
        w, h = self.engine.width, self.engine.height
        data = self.ctx.screen.read(components=3, alignment=1)
        img = Image.frombytes("RGB", (w, h), data).transpose(Image.FLIP_TOP_BOTTOM)
        os.makedirs("captures", exist_ok=True)
        name = os.path.join("captures", time.strftime("mandelfly_%Y%m%d_%H%M%S.png"))
        img.save(name)
        self.status = f"saved {name}"

    # ------------------------------------------------------------------
    def _held_input(self, dt):
        cam = self.cam
        k = lambda code: glfw.get_key(self.win, code) == glfw.PRESS
        pan = 0.9 * dt
        moved = False
        if k(glfw.KEY_LEFT):
            cam.pan_pixels(-pan, 0); moved = True
        if k(glfw.KEY_RIGHT):
            cam.pan_pixels(pan, 0); moved = True
        if k(glfw.KEY_UP):
            cam.pan_pixels(0, -pan); moved = True
        if k(glfw.KEY_DOWN):
            cam.pan_pixels(0, pan); moved = True
        zr = 0.0
        if k(glfw.KEY_W):
            zr += 1.4
        if k(glfw.KEY_S):
            zr -= 1.4
        if zr != 0.0:
            self.auto.stop()
            cam.zoom_rate = zr
            moved = True
        elif not self.auto.active:
            cam.zoom_rate = 0.0
        sp = 0.0
        if k(glfw.KEY_Q):
            sp -= 0.45
        if k(glfw.KEY_E):
            sp += 0.45
        cam.spin_rate = sp
        if moved and self.auto.active:
            self.auto.stop()
            self.status = "manual control"

    # ------------------------------------------------------------------
    def _build_hud(self):
        eng, cam = self.engine, self.cam
        w = 430
        rows = 9 + (len(HELP) + 1 if self.show_hud else 0)
        h = 26 + rows * 19
        img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        d = ImageDraw.Draw(img)
        d.rounded_rectangle([0, 0, w - 1, h - 1], 8, fill=(8, 10, 16, 185),
                            outline=(90, 110, 150, 140))
        y = 10
        ref = eng.refs.get()
        reflen = ref.length if ref else 0
        prec = cam.precision_bits()
        lines = [
            (f"zoom      2^{cam.zoom_power:,.1f}   (radius {cam.radius:.3e})", (235, 235, 245)),
            (f"kernel    {eng.tier}", (150, 210, 255)),
            (f"iters     {eng.maxiter:,}  x{eng.iter_boost:.2f}", (200, 210, 230)),
            (f"precision {prec} bits  ({prec/3.32:.0f} digits)", (200, 210, 230)),
            (f"reference {reflen:,} steps"
             + ("  [computing]" if eng.refs.busy else f"  {eng.refs.last_ms:.0f} ms"),
             (255, 200, 140) if eng.refs.busy else (170, 200, 175)),
            (f"render    {eng.scale*100:.0f}% res   {eng.last_gpu_ms:.1f} ms gpu",
             (200, 210, 230)),
            (f"samples   {eng.samples}" + ("  converged" if eng.samples >= eng.max_samples else ""),
             (200, 210, 230)),
            (f"palette   {eng.palette}  cycle {eng.cycle:.2f}", (200, 210, 230)),
            (f"{self._fps:5.1f} fps    {self.status}", (255, 220, 150)),
        ]
        for text, col in lines:
            d.text((12, y), text, font=self.font, fill=col)
            y += 19
        if self.show_hud:
            y += 6
            d.line([12, y - 3, w - 12, y - 3], fill=(90, 110, 150, 120))
            for keys, what in HELP:
                d.text((12, y), f"{keys:<14}", font=self.font_small, fill=(160, 200, 255))
                d.text((132, y), what, font=self.font_small, fill=(190, 195, 205))
                y += 19
        return img

    def _upload_hud(self):
        img = self._build_hud()
        full = Image.new("RGBA", (self.engine.width, self.engine.height), (0, 0, 0, 0))
        full.paste(img, (18, 18))
        data = np.asarray(full, dtype=np.uint8)
        if (self.overlay_tex is None
                or self.overlay_tex.size != (self.engine.width, self.engine.height)):
            if self.overlay_tex is not None:
                self.overlay_tex.release()
            self.overlay_tex = self.ctx.texture(
                (self.engine.width, self.engine.height), 4, data.tobytes())
            self.overlay_tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        else:
            self.overlay_tex.write(data.tobytes())

    # ------------------------------------------------------------------
    def run(self):
        last = time.perf_counter()
        while not glfw.window_should_close(self.win):
            glfw.poll_events()
            now = time.perf_counter()
            dt = min(0.05, now - last)
            last = now

            self._held_input(dt)
            self.auto.update(dt)

            # endless mode: let the background probe steer us toward structure
            if self.auto.state == self.auto.ENDLESS:
                ref = self.engine.refs.get()
                self.engine.steering.maybe_probe(ref, self.cam.radius,
                                                 self.engine.maxiter)
                (ox, oy), conf = self.engine.steering.take()
                if conf > 0.0:
                    self.cam.pan_pixels(ox * dt * 0.55, oy * dt * 0.55)

            moving = self.cam.update(dt)
            img, uv = self.engine.frame(self.cam, moving)

            self._frames += 1
            if now - self._fps_t > 0.4:
                self._fps = self._frames / (now - self._fps_t)
                self._frames = 0
                self._fps_t = now
            if now - self._hud_t > 0.12:
                self._upload_hud()
                self._hud_t = now

            self.engine.present(self.ctx.screen, img, uv, self.overlay_tex)
            glfw.swap_buffers(self.win)

        self.engine.release()
        glfw.terminate()
