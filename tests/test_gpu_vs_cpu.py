"""
Headless GPU verification.

The GLSL kernel and the CPU kernel implement the same recurrence, but "the same
maths" written twice in two languages is exactly where silent divergence lives:
a double literal that quietly became a float, an off-by-one in the orbit index,
a rebasing branch that fires one iteration late.

So we run the real shader against the already-MPFR-validated CPU kernel through
an offscreen EGL context and compare numbers, not screenshots. On this container
that context is llvmpipe -- software, slow, and completely irrelevant to the
answer, because we are testing arithmetic, not throughput. If it agrees here it
will agree on a 4070.
"""
import os, sys, math
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import moderngl

from mandelfly.shaders import VERTEX, FRAGMENT_PERTURB, FRAGMENT_DIRECT
from mandelfly.reference import compute_reference, prec_for_radius, BAILOUT2
from mandelfly.kernels import render
from mandelfly.deepzoom import set_prec


def gpu_render(ctx, prog, vao, w, h, uniforms, ssbo=None):
    tex = ctx.texture((w, h), 4, dtype="f4")
    fbo = ctx.framebuffer(color_attachments=[tex])
    fbo.use()
    ctx.viewport = (0, 0, w, h)
    for k, v in uniforms.items():
        if k in prog:
            prog[k].value = v
    if ssbo is not None:
        ssbo.bind_to_storage_buffer(0)
    vao.render(moderngl.TRIANGLES, vertices=3)
    data = np.frombuffer(fbo.read(components=4, dtype="f4"), dtype=np.float32)
    fbo.release(); tex.release()
    return data.reshape(h, w, 4)


def case(ctx, name, cx, cy, radius, maxiter, size=72):
    prec = max(64, prec_for_radius(radius))
    set_prec(prec + 64)
    ref = compute_reference(cx, cy, maxiter, prec)

    # ---- CPU (validated against MPFR in test_perturbation.py)
    n_cpu, de_cpu = render(ref, -radius, -radius, 2 * radius, 2 * radius,
                           size, size, maxiter)

    # ---- GPU, same points
    prog = ctx.program(vertex_shader=VERTEX, fragment_shader=FRAGMENT_PERTURB)
    vao = ctx.vertex_array(prog, [])
    orbit = ref.interleaved_f64()
    ssbo = ctx.buffer(orbit.tobytes())
    out = gpu_render(ctx, prog, vao, size, size, {
        "uResolution": (size, size),
        "uCorner": (-radius, -radius),
        "uStepX": (2 * radius / size, 0.0),
        "uStepY": (0.0, 2 * radius / size),
        "uRefLen": ref.length,
        "uMaxIter": maxiter,
        "uPixelScale": 2 * radius / size,
        "uJitter": (0.0, 0.0),
        "uBailout2": BAILOUT2,
        "uRawOutput": 1,
    }, ssbo)
    ssbo.release(); vao.release(); prog.release()

    n_gpu = out[:, :, 0].astype(np.float64)
    esc_gpu = out[:, :, 2] > 0.5
    esc_cpu = n_cpu >= 0

    n_distinct = len(np.unique(np.round(n_cpu[esc_cpu], 3))) if esc_cpu.any() else 0
    vacuous = n_distinct < 20 or esc_cpu.mean() < 0.02

    # Self-calibrating tolerance.
    #
    # A fixed "fewer than 0.1% of pixels may differ" threshold is arbitrary and
    # wrong: some pixels sit on a chaotic knife edge where a one-ULP change to
    # the window flips the escape count by hundreds of iterations. No two
    # implementations that round differently -- GPU FMA contraction vs Numba
    # fastmath reassociation -- will ever agree on those, and demanding they do
    # is demanding the Mandelbrot set stop being chaotic.
    #
    # So measure the instability directly: re-render on the CPU with the window
    # nudged by one ULP, count how many pixels move, and require the GPU to be
    # no worse than that intrinsic noise floor.
    eps = np.spacing(radius)
    n_ulp, _ = render(ref, -radius + eps, -radius + eps, 2 * radius, 2 * radius,
                      size, size, maxiter)
    b_ulp = (n_cpu >= 0) & (n_ulp >= 0)
    baseline = int((np.abs(n_cpu[b_ulp] - n_ulp[b_ulp]) > 0.5).sum())

    class_mismatch = int((esc_gpu != esc_cpu).sum())
    both = esc_gpu & esc_cpu
    if both.any():
        d = np.abs(n_gpu[both] - n_cpu[both])
        worst = float(d.max())
        med = float(np.median(d))
        big = int((d > 0.5).sum())
    else:
        worst = med = float("nan")
        big = 0

    total = size * size
    allowed = max(4, baseline * 3)
    ok = (not vacuous) and class_mismatch <= allowed and big <= allowed
    tag = "VACUOUS" if vacuous else ("PASS" if ok else "FAIL")
    print(f"  [{tag:^7}] {name:<26} distinct={n_distinct:>5}  "
          f"median|dn|={med:.1e}  knife-edge: gpu={big:>3} "
          f"cpu-1ULP={baseline:>3} (allow {allowed})  interior-mismatch={class_mismatch}")
    return ok


def main():
    ctx = moderngl.create_context(standalone=True, backend="egl", require=430)
    print("\n=== GLSL kernel vs MPFR-validated CPU kernel ===")
    print(f"    context: {ctx.info['GL_RENDERER']}\n")

    cases = [
        ("shallow  r=1e-3",  "-0.7436438870371587", "0.13182590420531197", 1e-3,  1500),
        ("mid      r=1e-9",  "-0.743643887037158704752191506114774",
                             "0.131825904205311970493132056385139",       1e-9,  3000),
        ("deep     r=1e-20", "-0.7434355537040193925502788418774160794647115491651981",
                             "0.1312009042057000242647592087978702654970027544786318", 1e-20, 6000),
        ("deeper   r=1e-30", "-0.743435553704019392550278841877416079464711549165198176",
                             "0.131200904205700024264759208797870265497002754478631877", 1e-30, 9000),
    ]
    ok = True
    for nm, cx, cy, r, mi in cases:
        ok &= case(ctx, nm, cx, cy, r, mi)

    print("\n" + ("GPU kernel agrees with ground truth." if ok
                  else "GPU kernel DIVERGES -- do not ship."))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
