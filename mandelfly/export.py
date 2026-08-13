"""
Offline rendering: stills and zoom-video frame sequences.

Two backends.

  --gpu (default) renders through the same GLSL kernel the interactive app uses,
  offscreen, with heavy supersampling. Fast.

  --cpu runs the Numba kernel with prange across every physical core. This is
  the honest all-cores path: each thread owns a slab of rows and the inner loop
  is straight-line float64 that vectorises into AVX. It is slower per frame than
  the GPU but it is not competing with a display, it never drops a frame, and on
  a many-core box with a 4070 you can run both at once on alternate frames --
  see --hybrid.

Frames land as PNGs plus a ready-to-run ffmpeg command; muxing is left to ffmpeg
rather than reimplemented badly here.
"""
from __future__ import annotations

import math
import os
import time

import numpy as np
from PIL import Image

from .camera import Camera, auto_maxiter
from .colour import shade, to_png_array
from .reference import compute_reference, prec_for_radius
from .deepzoom import set_prec


# --------------------------------------------------------------------------
def render_still_cpu(cx, cy, radius, width, height, maxiter=None,
                     samples=1, palette="ember", cycle=4.0, de_strength=0.78,
                     progress=None):
    """Full-quality CPU render. Uses every core via the Numba prange kernel."""
    from .kernels import render

    maxiter = maxiter or auto_maxiter(radius)
    prec = max(64, prec_for_radius(radius))
    set_prec(prec + 64)
    digits = int(max(0.0, -math.log10(radius))) + 24
    ref = compute_reference(str(cx), str(cy), maxiter, prec)

    aspect = width / height
    span_x = 2.0 * radius * (aspect if aspect >= 1 else 1.0)
    span_y = 2.0 * radius * (1.0 if aspect >= 1 else 1.0 / aspect)

    acc = np.zeros((height, width, 3), dtype=np.float64)
    for s in range(samples):
        # stratified jitter, in units of one pixel
        jx = ((s % 4) + 0.5) / 4.0 - 0.5 if samples > 1 else 0.0
        jy = ((s // 4) + 0.5) / 4.0 - 0.5 if samples > 1 else 0.0
        ox = span_x / width * jx
        oy = span_y / height * jy
        n, de = render(ref, -span_x / 2 + ox, -span_y / 2 + oy,
                       span_x, span_y, width, height, maxiter)
        acc += shade(n, de, palette=palette, cycle=cycle, de_strength=de_strength)
        if progress:
            progress(s + 1, samples)
    return to_png_array(acc / samples)


# --------------------------------------------------------------------------
def render_still_gpu(ctx, cx, cy, radius, width, height, maxiter=None,
                     samples=8, **look):
    """Offscreen GPU render at arbitrary resolution, independent of any window."""
    import moderngl
    from .engine import Engine

    eng = Engine(ctx, width, height, max_samples=samples)
    for k, v in look.items():
        if hasattr(eng, k):
            setattr(eng, k, v)
    eng.quality_locked = True
    cam = Camera(str(cx), str(cy), radius)
    if maxiter:
        eng.iter_boost = maxiter / max(1, auto_maxiter(radius))

    eng.ensure_reference(cam, force=True)
    deadline = time.time() + 600
    while eng.refs.get() is None and time.time() < deadline:
        time.sleep(0.01)

    img = uv = None
    for _ in range(samples):
        img, uv = eng.frame(cam, moving=False)

    tex = ctx.texture((width, height), 4, dtype="f1")
    fbo = ctx.framebuffer([tex])
    eng.present(fbo, img, uv)
    raw = np.frombuffer(fbo.read(components=3, alignment=1), dtype=np.uint8)
    out = raw.reshape(height, width, 3)
    fbo.release(); tex.release(); eng.release()
    return out


# --------------------------------------------------------------------------
def render_sequence(outdir, cx, cy, start_radius, end_radius, seconds, fps=60,
                    width=1920, height=1080, samples=4, backend="gpu",
                    palette="ember", cycle=4.0, spin=0.0, progress=True):
    """
    A zoom video, one PNG per frame.

    Radius moves linearly in log space -- a constant number of doublings per
    second -- because that is what reads as constant speed. Linear in radius
    would spend the whole clip in the first decade and then fall off a cliff.
    """
    os.makedirs(outdir, exist_ok=True)
    nframes = max(1, int(round(seconds * fps)))
    lr0 = math.log(start_radius)
    lr1 = math.log(end_radius)
    doublings = (lr0 - lr1) / 0.6931471805599453

    ctx = None
    if backend in ("gpu", "hybrid"):
        import moderngl
        try:
            ctx = moderngl.create_context(standalone=True, require=430)
        except Exception:
            ctx = moderngl.create_context(standalone=True, backend="egl", require=430)

    t0 = time.time()
    for i in range(nframes):
        u = i / max(1, nframes - 1)
        # ease in and out so the clip starts and ends gently
        e = u * u * (3.0 - 2.0 * u)
        radius = math.exp(lr0 + (lr1 - lr0) * e)
        use_cpu = backend == "cpu" or (backend == "hybrid" and i % 2 == 1)
        if use_cpu:
            frame = render_still_cpu(cx, cy, radius, width, height,
                                     samples=samples, palette=palette, cycle=cycle)
        else:
            frame = render_still_gpu(ctx, cx, cy, radius, width, height,
                                     samples=samples, palette=palette, cycle=cycle)
        Image.fromarray(frame).save(os.path.join(outdir, f"frame_{i:06d}.png"))
        if progress:
            done = i + 1
            el = time.time() - t0
            eta = el / done * (nframes - done)
            print(f"\r  frame {done}/{nframes}  r={radius:.3e}  "
                  f"{el/done:5.2f}s/frame  eta {eta/60:5.1f} min   ",
                  end="", flush=True)
    if progress:
        print()
    print(f"\n{nframes} frames covering {doublings:.0f} doublings -> {outdir}")
    print("mux with:")
    print(f'  ffmpeg -framerate {fps} -i "{outdir}/frame_%06d.png" '
          f'-c:v libx264 -crf 16 -pix_fmt yuv420p "{outdir}/zoom.mp4"')
