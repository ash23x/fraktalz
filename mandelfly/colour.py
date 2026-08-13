"""
Turning iteration counts into something worth looking at.

Two ingredients:

  * continuous escape time -- the raw integer count bands horribly, so we use
    n + 1 - log2(log|z|), which is smooth across the escape boundary

  * the distance estimate -- |z|log|z|/|dz|, in pixel units. Near the boundary
    filaments are thinner than a pixel and plain escape-time colouring turns
    them into aliased confetti. Weighting by DE draws them as clean glowing
    threads instead, and doubles as free antialiasing on the set's edge.

Palettes are cosine-based (Inigo Quilez's trick): three sinusoids at different
phases give a continuous, cyclic, perceptually smooth ramp with four numbers per
channel and no lookup table to interpolate.
"""
from __future__ import annotations

import numpy as np

# (a, b, c, d) per channel -- colour = a + b*cos(2pi*(c*t + d))
PALETTES = {
    "ember":     ((0.55, 0.42, 0.32), (0.45, 0.38, 0.28), (1.0, 1.0, 1.0), (0.00, 0.12, 0.22)),
    "ice":       ((0.48, 0.53, 0.62), (0.42, 0.40, 0.38), (1.0, 1.0, 1.0), (0.62, 0.55, 0.42)),
    "aurora":    ((0.50, 0.50, 0.50), (0.45, 0.45, 0.45), (1.0, 1.0, 1.0), (0.30, 0.42, 0.58)),
    "gold":      ((0.62, 0.50, 0.28), (0.38, 0.36, 0.30), (1.0, 1.0, 1.0), (0.08, 0.15, 0.28)),
    "spectral":  ((0.50, 0.50, 0.50), (0.50, 0.50, 0.50), (1.0, 1.0, 1.0), (0.00, 0.33, 0.67)),
    "monochrome":((0.55, 0.55, 0.57), (0.42, 0.42, 0.43), (1.0, 1.0, 1.0), (0.10, 0.10, 0.10)),
    "copper":    ((0.50, 0.35, 0.25), (0.45, 0.32, 0.22), (1.0, 1.0, 1.0), (0.05, 0.10, 0.16)),
    "deepsea":   ((0.36, 0.46, 0.55), (0.34, 0.36, 0.40), (1.0, 1.0, 1.0), (0.55, 0.48, 0.30)),
}
PALETTE_NAMES = list(PALETTES)


def cosine_palette(t, name="ember"):
    a, b, c, d = PALETTES[name]
    t = np.asarray(t, dtype=np.float64)
    out = np.empty(t.shape + (3,), dtype=np.float64)
    for k in range(3):
        out[..., k] = a[k] + b[k] * np.cos(2.0 * np.pi * (c[k] * t + d[k]))
    return np.clip(out, 0.0, 1.0)


def shade(smooth_n, dist_est, palette="ember", cycle=4.0, offset=0.0,
          interior_colour=(0.02, 0.02, 0.035), de_strength=0.85, gamma=1.0):
    """
    smooth_n : continuous escape count, -1 for interior
    dist_est : distance estimate in pixel units (0 for interior)
    """
    interior = smooth_n < 0
    n = np.where(interior, 0.0, smooth_n)

    # sqrt keeps the number of visible colour cycles roughly constant with
    # depth; log does not (see shaders.py for the measurements)
    t = np.sqrt(np.maximum(n, 0.0)) / max(cycle, 0.05) + offset
    rgb = cosine_palette(t, palette)

    if de_strength > 0.0:
        # DE in pixels: <1 means the filament is sub-pixel. Map through a soft
        # curve so thin structure glows rather than aliasing into speckle.
        d = np.clip(dist_est, 0.0, 1e6)
        edge = 1.0 - np.exp(-2.2 * d)
        edge = edge ** 0.55
        rgb *= (1.0 - de_strength + de_strength * edge)[..., None]

    if gamma != 1.0:
        rgb = np.power(np.clip(rgb, 0, 1), 1.0 / gamma)

    rgb[interior] = interior_colour
    return np.clip(rgb, 0.0, 1.0)


def to_png_array(rgb) -> np.ndarray:
    return (np.clip(rgb, 0.0, 1.0) * 255.0 + 0.5).astype(np.uint8)
