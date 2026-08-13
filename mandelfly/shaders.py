"""
GLSL sources.

The fragment shader is the whole show: one invocation per pixel per sample,
iterating a float64 delta against the reference orbit sitting in an SSBO.

Why float64 and not float32, given that a 4070 runs fp64 at 1/64 rate: because
fp32 deltas are measurably wrong. Validated against MPFR ground truth, fp32
perturbation mismatches 5 pixels in 49 at a zoom of only 1e-6, and not by one
iteration -- one sample read 2800 against a true 3000, which on screen is a
glitch blob, not noise. 24 mantissa bits cannot survive thousands of compounding
delta iterations. So: fp64 for the delta, and we buy the smoothness back with
adaptive resolution and progressive accumulation instead of raw throughput.

The shallow tier (zoom > ~1e-5) skips perturbation entirely and iterates c
directly in fp32, where the GPU is 64x faster and the extra precision buys
nothing. Most of the opening minute of any flythrough lives there.
"""

VERTEX = """
#version 430
// fullscreen triangle -- no vertex buffer, gl_VertexID does the work
out vec2 vUV;
void main() {
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    vUV = p;
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""


COMMON_COLOUR = """
// ---- cosine palette: colour = a + b*cos(2pi*(c*t + d))
uniform vec3 uPalA, uPalB, uPalC, uPalD;
uniform float uCycle;
uniform float uColourOffset;
uniform float uDEStrength;
uniform vec3  uInteriorColour;
uniform float uGlow;

vec3 palette(float t) {
    return clamp(uPalA + uPalB * cos(6.28318530718 * (uPalC * t + uPalD)), 0.0, 1.0);
}

vec3 shadePixel(float smoothN, float dePixels, bool interior) {
    if (interior) return uInteriorColour;
    // sqrt, not log. Measured: the full set spans log 0.8..6.5 while a 1e-29
    // view spans only 7.2..8.0 -- 7x narrower, so any fixed log cycle either
    // over-bands the whole set or flattens deep views to one colour. sqrt
    // self-normalises: 24 units of range against 18, near enough the same.
    float t = sqrt(max(smoothN, 0.0)) / max(uCycle, 0.05) + uColourOffset;
    vec3 col = palette(t);
    // distance estimate in pixel units: sub-pixel filaments would otherwise
    // alias into speckle, so fade them by thickness instead
    float edge = 1.0 - exp(-2.2 * clamp(dePixels, 0.0, 1e6));
    edge = pow(edge, 0.55);
    col *= (1.0 - uDEStrength + uDEStrength * edge);
    // a touch of bloom on the thinnest structure keeps deep fields from
    // looking flat and muddy
    col += uGlow * pow(1.0 - edge, 3.0) * palette(t + 0.35);
    return col;
}
"""


# --------------------------------------------------------------------------
# deep tier: perturbation + Zhuoran rebasing, float64 deltas
# --------------------------------------------------------------------------
FRAGMENT_PERTURB = """
#version 430
#extension GL_ARB_gpu_shader_fp64 : enable

layout(location = 0) out vec4 fragColour;
in vec2 vUV;

layout(std430, binding = 0) readonly buffer RefOrbit { dvec2 orbit[]; };

uniform ivec2  uResolution;
uniform dvec2  uCorner;        // delta-c at the centre of pixel (0,0)
uniform dvec2  uStepX;         // delta-c per pixel along x (carries rotation)
uniform dvec2  uStepY;
uniform int    uRefLen;
uniform int    uMaxIter;
uniform double uPixelScale;    // |uStepX|, folded into the derivative
uniform int    uSampleCount;
uniform vec2   uJitter;
uniform double uBailout2;
uniform int    uRawOutput;

__COLOUR__

dvec2 cmul(dvec2 a, dvec2 b) {
    return dvec2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

void main() {
    vec2 px = vUV * vec2(uResolution);
    dvec2 base = uCorner
               + uStepX * double(px.x + uJitter.x)
               + uStepY * double(px.y + uJitter.y);

    dvec2 dz  = dvec2(0.0LF);
    dvec2 der = dvec2(0.0LF);      // dz/dc, pre-scaled by the pixel size so it
                                   // stays near unity instead of hitting 1e300
    int m = 0;
    int n = 0;
    bool escaped = false;
    double zmag = 0.0LF;

    while (n < uMaxIter) {
        dvec2 Z = orbit[m];
        dvec2 zf = Z + dz;

        der = 2.0LF * cmul(zf, der) + dvec2(uPixelScale, 0.0LF);
        dz  = 2.0LF * cmul(Z, dz) + cmul(dz, dz) + base;

        m++; n++;
        dvec2 zn = orbit[m] + dz;
        zmag = dot(zn, zn);
        double dmag = dot(dz, dz);

        if (zmag > uBailout2) { escaped = true; break; }
        // Zhuoran rebasing: when the true value drops below the delta (or we
        // reach the end of the stored reference) restart the reference at zero.
        // One reference orbit covers the whole frame; no glitch hunting.
        if (zmag < dmag || m >= uRefLen) { dz = zn; m = 0; }
    }

    float smoothN = 0.0;
    float dePixels = 0.0;
    if (escaped) {
        // GLSL has no transcendentals for double, and does not need them here:
        // both magnitudes are O(1e24) at worst and the derivative is pre-scaled
        // to near unity, so float has ample range and the log only feeds colour.
        float zmagf = float(zmag);
        float dmagf = float(dot(der, der));
        float lz = 0.5 * log(zmagf);
        smoothN = float(n) + 1.0 - log(lz) / 0.6931471805599453;
        dePixels = dmagf > 0.0 ? sqrt(zmagf) * lz / sqrt(dmagf) : 1e6;
    }
    if (uRawOutput == 1) {
        fragColour = vec4(smoothN, dePixels, escaped ? 1.0 : 0.0, float(n));
        return;
    }
    fragColour = vec4(shadePixel(smoothN, dePixels, !escaped), 1.0);
}
""".replace("__COLOUR__", COMMON_COLOUR)


# --------------------------------------------------------------------------
# shallow tier: direct iteration in float32, 64x the arithmetic throughput
# --------------------------------------------------------------------------
FRAGMENT_DIRECT = """
#version 430

layout(location = 0) out vec4 fragColour;
in vec2 vUV;

uniform ivec2 uResolution;
uniform vec2  uCentre;
uniform vec2  uStepX;
uniform vec2  uStepY;
uniform int   uMaxIter;
uniform float uPixelScale;
uniform vec2  uJitter;
uniform float uBailout2;
uniform int   uRawOutput;

__COLOUR__

void main() {
    vec2 px = vUV * vec2(uResolution);
    vec2 c = uCentre + uStepX * (px.x + uJitter.x) + uStepY * (px.y + uJitter.y);

    vec2 z = vec2(0.0);
    vec2 der = vec2(0.0);
    int n = 0;
    bool escaped = false;
    float zmag = 0.0;

    while (n < uMaxIter) {
        der = 2.0 * vec2(z.x * der.x - z.y * der.y, z.x * der.y + z.y * der.x)
            + vec2(uPixelScale, 0.0);
        z = vec2(z.x * z.x - z.y * z.y, 2.0 * z.x * z.y) + c;
        n++;
        zmag = dot(z, z);
        if (zmag > uBailout2) { escaped = true; break; }
    }

    float smoothN = 0.0;
    float dePixels = 0.0;
    if (escaped) {
        float lz = 0.5 * log(zmag);
        smoothN = float(n) + 1.0 - log(lz) / 0.6931471805599453;
        float dmag = dot(der, der);
        dePixels = dmag > 0.0 ? sqrt(zmag) * lz / sqrt(dmag) : 1e6;
    }
    if (uRawOutput == 1) {
        fragColour = vec4(smoothN, dePixels, escaped ? 1.0 : 0.0, float(n));
        return;
    }
    fragColour = vec4(shadePixel(smoothN, dePixels, !escaped), 1.0);
}
""".replace("__COLOUR__", COMMON_COLOUR)


# --------------------------------------------------------------------------
# accumulation + present
# --------------------------------------------------------------------------
FRAGMENT_ACCUM = """
#version 430
layout(location = 0) out vec4 fragColour;
in vec2 vUV;
uniform sampler2D uPrev;
uniform sampler2D uNew;
uniform float uBlend;          // 1/(sample index + 1)
void main() {
    vec3 prev = texture(uPrev, vUV).rgb;
    vec3 cur  = texture(uNew,  vUV).rgb;
    fragColour = vec4(mix(prev, cur, uBlend), 1.0);
}
"""

FRAGMENT_PRESENT = """
#version 430
layout(location = 0) out vec4 fragColour;
in vec2 vUV;
uniform sampler2D uImage;
uniform sampler2D uOverlay;
uniform float uGamma;
uniform float uExposure;
uniform float uVignette;
uniform int   uShowOverlay;
// while the camera moves we render into the corner of a full-size target at
// reduced resolution; this scales the fetch to just that region
uniform vec2  uUVScale;

vec3 tonemap(vec3 x) {
    // gentle filmic shoulder -- keeps the bright interiors of the palette from
    // clipping to flat white when exposure is pushed
    x *= uExposure;
    return (x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14);
}

void main() {
    vec3 col = texture(uImage, vUV * uUVScale).rgb;
    col = tonemap(col);
    col = pow(max(col, 0.0), vec3(1.0 / uGamma));
    vec2 q = vUV - 0.5;
    col *= 1.0 - uVignette * dot(q, q) * 1.6;
    if (uShowOverlay == 1) {
        vec4 ov = texture(uOverlay, vec2(vUV.x, 1.0 - vUV.y));
        col = mix(col, ov.rgb, ov.a);
    }
    fragColour = vec4(col, 1.0);
}
"""
