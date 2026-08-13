"""
Drive the whole render engine offscreen.

I cannot open a window in this container, so the plumbing that a windowed run
would exercise -- tier switching, reference-orbit handoff, ping-pong
accumulation, the present pass, adaptive scaling -- gets exercised here instead,
through a standalone EGL context. Everything except GLFW itself.

Writes a contact sheet so the output can be looked at, not just asserted.
"""
import os, sys, math, time
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import moderngl
from PIL import Image

from mandelfly.engine import Engine, DIRECT_TIER_RADIUS
from mandelfly.camera import Camera, Autopilot, auto_maxiter

W = H = 256
OUT = "/tmp/engine_sheet.png"


def grab(ctx, eng, cam, moving, wait_ref=True, samples=1):
    if wait_ref and cam.radius <= DIRECT_TIER_RADIUS:
        eng.ensure_reference(cam)
        for _ in range(400):
            if eng.refs.get() is not None and not eng.refs.busy:
                break
            time.sleep(0.02)
    img = uv = None
    for _ in range(samples):
        img, uv = eng.frame(cam, moving)
    tex = ctx.texture((W, H), 4, dtype="f1")
    fbo = ctx.framebuffer([tex])
    eng.present(fbo, img, uv)
    raw = np.frombuffer(fbo.read(components=3), dtype=np.uint8).reshape(H, W, 3)
    fbo.release(); tex.release()
    return raw


def main():
    ctx = moderngl.create_context(standalone=True, backend="egl", require=430)
    print(f"\n=== engine headless run on {ctx.info['GL_RENDERER']} ===\n")
    eng = Engine(ctx, W, H)
    eng.quality_locked = True          # deterministic output for the sheet
    ok = True
    tiles, labels = [], []

    checks = [
        ("full set",      "-0.5", "0.0", 1.6,   False, 1),
        ("direct tier",   "-0.743643887037158", "0.131825904205311", 5e-3, False, 1),
        ("perturb tier",  "-0.743643887037158704752191506114774",
                          "0.131825904205311970493132056385139", 1e-9, False, 1),
        ("deep 1e-20",    "-0.7434355537040193925502788418774160794647115491651981",
                          "0.1312009042057000242647592087978702654970027544786318", 1e-20, False, 1),
        ("deep 1e-30",    "-0.743435553704019392550278841877416079464711549165198176",
                          "0.131200904205700024264759208797870265497002754478631877", 1e-30, False, 1),
        ("accumulated",   "-0.743643887037158704752191506114774",
                          "0.131825904205311970493132056385139", 1e-9, False, 16),
    ]

    for name, cx, cy, r, moving, samples in checks:
        cam = Camera(cx, cy, r)
        # A FRESH engine per case, deliberately. Sharing one let state leak
        # between checks and masked a real bug: the first reference orbit was
        # built with the constructor's placeholder iteration budget, which only
        # looked fine because an earlier case had already raised it.
        eng.release()
        eng = Engine(ctx, W, H, max_samples=max(1, samples))
        eng.quality_locked = True
        img = grab(ctx, eng, cam, moving, samples=samples)
        var = float(img.std())
        black = float((img.max(axis=2) < 12).mean())
        good = var > 8.0 and black < 0.9
        ok &= good
        print(f"  [{'PASS' if good else 'FAIL'}] {name:<14} tier={eng.tier:<12} "
              f"maxiter={eng.maxiter:<6} stdev={var:6.2f}  black={black:5.1%}  "
              f"samples={eng.samples}")
        tiles.append(img)
        labels.append(name)

    # ---- moving vs still: adaptive scale must drop and accumulation must reset
    cam = Camera("-0.743643887037158704752191506114774",
                 "0.131825904205311970493132056385139", 1e-9)
    eng.quality_locked = False
    eng.scale = 1.0
    eng.ensure_reference(cam)
    for _ in range(400):
        if eng.refs.get() is not None and not eng.refs.busy:
            break
        time.sleep(0.02)
    for _ in range(12):
        eng.frame(cam, moving=True)
    moving_scale = eng.scale
    for _ in range(6):
        eng.frame(cam, moving=False)
    print(f"\n  adaptive scale after 12 moving frames: {moving_scale:.3f} "
          f"(gpu {eng.last_gpu_ms:.1f} ms/frame on llvmpipe)")
    print(f"  accumulated samples after 6 still frames: {eng.samples}")
    if eng.samples != 6:
        print("  [FAIL] accumulation did not advance one sample per still frame")
        ok = False

    # ---- autopilot state machine reaches its destination
    cam = Camera("-0.5", "0.0", 1.6)
    ap = Autopilot(cam)
    ap.fly_to("-0.743435553704019392550278841877416079464711549165198176",
              "0.131200904205700024264759208797870265497002754478631877", 1e-18,
              dive_rate=40.0)
    states = []
    for i in range(4000):
        ap.update(1 / 60)
        cam.update(1 / 60)
        if not states or states[-1] != ap.state:
            states.append(ap.state)
        if ap.state == ap.IDLE and i > 5:
            break
    reached = abs(math.log10(cam.radius) - (-18)) < 0.6
    print(f"  autopilot phases: {' -> '.join(states)}")
    print(f"  arrived at r={cam.radius:.3e} (wanted 1e-18): "
          f"{'PASS' if reached else 'FAIL'}")
    ok &= reached

    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (W * cols, H * rows), (8, 8, 12))
    for k, t in enumerate(tiles):
        sheet.paste(Image.fromarray(t), ((k % cols) * W, (k // cols) * H))
    sheet.save(OUT)
    print(f"\n  contact sheet -> {OUT}")
    print("\n" + ("ENGINE OK" if ok else "ENGINE HAS PROBLEMS"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
