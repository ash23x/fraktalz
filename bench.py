#!/usr/bin/env python3
"""
bench.py -- how fast does this machine iterate z^2 + c?

Identical workload on every backend: a 1024x1024 tile on the Seahorse
Valley boundary, maxiter 5000. Each backend reports its own measured
iteration count (read back from the render target / kernel output), so
Giter/s is self-consistent, not estimated. GPU fp64 is cross-checked
against the CPU fp64 result pixel-for-pixel.
"""
import time, platform
import numpy as np

def cpu_name():
    try:
        import winreg
        k = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                           r"HARDWARE\DESCRIPTION\System\CentralProcessor\0")
        return winreg.QueryValueEx(k, "ProcessorNameString")[0].strip()
    except Exception:
        return platform.processor()

W, H       = 1024, 1024
CX, CY     = -0.743643887037151, 0.131825904205330
STEP       = 2.0e-4 / W          # radius 1e-4
MAXITER    = 5000
GPU_FRAMES = 10

# ---------------------------------------------------------------- CPU
from numba import njit, prange, set_num_threads, get_num_threads

@njit(parallel=True, fastmath=True, cache=True)
def mandel_cpu(cx0, cy0, step, w, h, maxiter):
    iters = np.zeros((h, w), np.int64)
    for j in prange(h):
        cy = cy0 + (j - h * 0.5) * step
        for i in range(w):
            cx = cx0 + (i - w * 0.5) * step
            zx = 0.0; zy = 0.0; n = 0
            while n < maxiter:
                zx2 = zx * zx; zy2 = zy * zy
                if zx2 + zy2 > 4.0:
                    break
                zy = 2.0 * zx * zy + cy
                zx = zx2 - zy2 + cx
                n += 1
            iters[j, i] = n
    return iters

def bench_cpu(threads):
    set_num_threads(threads)
    t0 = time.perf_counter()
    iters = mandel_cpu(CX, CY, STEP, W, H, MAXITER)
    dt = time.perf_counter() - t0
    return iters, dt

# ---------------------------------------------------------------- GPU
VS = """#version 430
void main() {
    vec2 v = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(v * 2.0 - 1.0, 0.0, 1.0);
}"""

FS64 = """#version 430
uniform dvec2 uC; uniform double uStep; uniform ivec2 uSize; uniform int uMax;
out vec4 frag;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    dvec2 c = uC + dvec2((double(p.x) - double(uSize.x) * double(0.5)) * uStep,
                         (double(p.y) - double(uSize.y) * double(0.5)) * uStep);
    dvec2 z = dvec2(0.0);
    int n = 0;
    while (n < uMax) {
        double x2 = z.x * z.x, y2 = z.y * z.y;
        if (x2 + y2 > double(4.0)) break;
        z = dvec2(x2 - y2 + c.x, double(2.0) * z.x * z.y + c.y);
        n++;
    }
    frag = vec4(float(n), 0.0, 0.0, 1.0);
}"""

FS32 = """#version 430
uniform vec2 uC; uniform float uStep; uniform ivec2 uSize; uniform int uMax;
out vec4 frag;
void main() {
    ivec2 p = ivec2(gl_FragCoord.xy);
    vec2 c = uC + vec2((float(p.x) - float(uSize.x) * 0.5) * uStep,
                       (float(p.y) - float(uSize.y) * 0.5) * uStep);
    vec2 z = vec2(0.0);
    int n = 0;
    while (n < uMax) {
        float x2 = z.x * z.x, y2 = z.y * z.y;
        if (x2 + y2 > 4.0) break;
        z = vec2(x2 - y2 + c.x, 2.0 * z.x * z.y + c.y);
        n++;
    }
    frag = vec4(float(n), 0.0, 0.0, 1.0);
}"""

def bench_gpu(ctx, moderngl, fs, dbl):
    prog = ctx.program(vertex_shader=VS, fragment_shader=fs)
    prog["uC"].value = (CX, CY)
    prog["uStep"].value = STEP
    prog["uSize"].value = (W, H)
    prog["uMax"].value = MAXITER
    tex = ctx.texture((W, H), 1, dtype="f4")
    fbo = ctx.framebuffer(color_attachments=[tex])
    vao = ctx.vertex_array(prog, [])
    fbo.use(); ctx.viewport = (0, 0, W, H)
    vao.render(moderngl.TRIANGLES, vertices=3); ctx.finish()   # warm-up
    t0 = time.perf_counter()
    for _ in range(GPU_FRAMES):
        vao.render(moderngl.TRIANGLES, vertices=3)
    ctx.finish()
    dt = (time.perf_counter() - t0) / GPU_FRAMES
    iters = np.frombuffer(fbo.read(components=1, dtype="f4"),
                          dtype=np.float32).reshape(H, W)
    return iters.astype(np.int64), dt

# ---------------------------------------------------------------- run
def main():
    rows = []
    print("system:", cpu_name())

    print("JIT compiling CPU kernel...")
    set_num_threads(1); mandel_cpu(CX, CY, STEP, 32, 32, 64)   # compile pass

    it1, dt1 = bench_cpu(1)
    total = int(it1.sum())
    rows.append((cpu_name(), "fp64", "1", dt1, total / dt1))

    import numba
    nthreads = numba.config.NUMBA_DEFAULT_NUM_THREADS
    itN, dtN = bench_cpu(nthreads)
    rows.append(("CPU (AVX/numba)", "fp64", str(nthreads), dtN, int(itN.sum()) / dtN))

    import glfw, moderngl
    glfw.init()
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    win = glfw.create_window(64, 64, "bench", None, None)
    glfw.make_context_current(win)
    ctx = moderngl.create_context()
    gpu_name = ctx.info["GL_RENDERER"]
    print("GPU:", gpu_name)

    g32, dt32 = bench_gpu(ctx, moderngl, FS32, False)
    rows.append((gpu_name, "fp32", "-", dt32, int(g32.sum()) / dt32))

    g64, dt64 = bench_gpu(ctx, moderngl, FS64, True)
    rows.append((gpu_name, "fp64", "-", dt64, int(g64.sum()) / dt64))

    mismatch = int((g64 != it1).sum())
    print(f"\ncross-check: GPU fp64 vs CPU fp64 -> {mismatch}/{W*H} pixels differ")

    base = rows[0][4]
    print(f"\nworkload: {W}x{H} @ maxiter {MAXITER}, Seahorse Valley boundary, "
          f"{total/1e9:.2f} G iterations/frame\n")
    print("| backend | precision | threads | ms/frame | Giter/s | vs 1 core |")
    print("|---|---|---|---|---|---|")
    for name, prec, thr, dt, rate in rows:
        print(f"| {name} | {prec} | {thr} | {dt*1e3:,.1f} | "
              f"{rate/1e9:,.2f} | {rate/base:,.1f}x |")
    glfw.terminate()

if __name__ == "__main__":
    main()
