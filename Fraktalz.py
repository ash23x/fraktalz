#!/usr/bin/env python3
"""
mandelfly -- a real-time Mandelbrot flythrough.

  python run.py                       open the window and fly
  python run.py --fullscreen --fps 144
  python run.py --location 3          start at a curated deep location
  python run.py --list                show the curated locations
  python run.py --still out.png --location 2 --width 3840 --height 2160
  python run.py --video out/ --location 1 --seconds 40 --backend hybrid
  python run.py --selftest            verify the maths and the shader, no window
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description="Mandelbrot flythrough",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__)
    ap.add_argument("--width", type=int, default=1600)
    ap.add_argument("--height", type=int, default=900)
    ap.add_argument("--fullscreen", action="store_true")
    ap.add_argument("--no-vsync", action="store_true",
                    help="uncap the framerate; adaptive resolution will chase --fps")
    ap.add_argument("--fps", type=int, default=60,
                    help="frame-time budget the adaptive resolution aims at")
    ap.add_argument("--location", type=int, default=None,
                    help="start at curated location N (1-based)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--still", metavar="PNG")
    ap.add_argument("--video", metavar="DIR")
    ap.add_argument("--seconds", type=float, default=30.0)
    ap.add_argument("--samples", type=int, default=8)
    ap.add_argument("--backend", choices=("gpu", "cpu", "hybrid"), default="gpu")
    ap.add_argument("--palette", default="ember")
    ap.add_argument("--cycle", type=float, default=4.0)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--bench", action="store_true",
                    help="benchmark every backend, print a markdown table")
    args = ap.parse_args()

    from mandelfly import locations as loc

    if args.list:
        print(f"\n  0  {loc.HOME['name']:<20} radius {loc.HOME['radius']:.1e}")
        for i, d in enumerate(loc.LOCATIONS, 1):
            print(f"  {i}  {d['name']:<20} radius {d['radius']:.1e}   "
                  f"{d['maxiter']:,} iters")
        print()
        return 0

    if args.bench:
        import bench
        bench.main()
        return 0

    if args.selftest:
        import subprocess
        here = os.path.dirname(os.path.abspath(__file__))
        rc = 0
        for t in ("test_perturbation.py", "test_gpu_vs_cpu.py", "test_engine_headless.py"):
            print(f"\n----- {t} " + "-" * (56 - len(t)))
            rc |= subprocess.call([sys.executable, os.path.join(here, "tests", t)])
        return rc

    start = loc.HOME
    if args.location:
        start = loc.by_index(args.location - 1)

    if args.still:
        from mandelfly.export import render_still_cpu, render_still_gpu
        from PIL import Image
        print(f"rendering {args.width}x{args.height} at {start['name']} "
              f"(radius {start['radius']:.2e}, {args.samples} samples, {args.backend})")
        if args.backend == "cpu":
            img = render_still_cpu(start["cx"], start["cy"], start["radius"],
                                   args.width, args.height, samples=args.samples,
                                   palette=args.palette, cycle=args.cycle,
                                   progress=lambda a, b: print(f"\r  sample {a}/{b}",
                                                               end="", flush=True))
            print()
        else:
            import moderngl
            try:
                ctx = moderngl.create_context(standalone=True, require=430)
            except Exception:
                ctx = moderngl.create_context(standalone=True, backend="egl", require=430)
            img = render_still_gpu(ctx, start["cx"], start["cy"], start["radius"],
                                   args.width, args.height, samples=args.samples,
                                   palette=args.palette, cycle=args.cycle)
        Image.fromarray(img).save(args.still)
        print(f"wrote {args.still}")
        return 0

    if args.video:
        from mandelfly.export import render_sequence
        render_sequence(args.video, start["cx"], start["cy"],
                        start_radius=1.6, end_radius=start["radius"],
                        seconds=args.seconds, fps=args.fps,
                        width=args.width, height=args.height,
                        samples=args.samples, backend=args.backend,
                        palette=args.palette, cycle=args.cycle)
        return 0

    from mandelfly.app import App
    app = App(width=args.width, height=args.height, fullscreen=args.fullscreen,
              vsync=not args.no_vsync, start=start, target_fps=args.fps)
    app.engine.palette = args.palette
    app.engine.cycle = args.cycle
    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
