#!/usr/bin/env python3
"""Render the Min Dark wallpapers into backgrounds/.

  code        a few lines of TypeScript as VS Code shows them in Min Dark: line
              numbers, an indent guide, the current line outlined in Min's
              #303030 across the whole screen, the cursor at its end
  silk        twisting ribbons of silk in Min's colors, lit from above, the back
              ones out of focus
  dunes       sand dunes at dusk: faces turned to a low sun and faces in shade
  lake        a mountain lake at night: the moon's path on the water, mist on
              the far shore, pines in front
  planet      the lit limb of a banded planet rising into a starfield
  flow        thousands of hairlines drawn along a smooth flow field
  topography  contour lines of an imaginary terrain, one traced in lavender
  window      an editor and a terminal as macOS-style windows over the silk

Every color comes from colors.toml or Min Dark's token colors, and each
picture keeps its subject inside the middle 3456 px so a 16:10 crop keeps it.

Usage: tools/make-wallpaper.py [name ...|all] [--width 3840 --height 2160] [--seed N]
"""

import argparse
import math
import os
import re
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

THEME_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT = "/usr/share/fonts/TTF/JetBrainsMonoNerdFont-Regular.ttf"
UI_FONT = "/usr/share/fonts/noto/NotoSans-Medium.ttf"


def theme_colors():
    with open(os.path.join(THEME_DIR, "colors.toml")) as f:
        return {k: "#" + v for k, v in re.findall(r'^(\w+)\s*=\s*"#([0-9a-fA-F]{6})"', f.read(), re.M)}


C = theme_colors()

# Token colors, by min-dark.json scope. The ones colors.toml has no key for are
# spelled out.
DEFAULT = C["magenta"]        # the scope-less rule: plain identifiers are lavender
KEYWORD = C["red"]            # keyword, storage.type, keyword.operator
FUNCTION = C["magenta"]       # entity.name.function, meta.function-call
CONSTANT = C["blue"]          # support.type, variable.other.object
PARAMETER = C["orange"]       # variable.parameter.function
COMMENT = C["brown"]          # comment
PUNCTUATION = "#bbbbbb"       # punctuation.separator, punctuation.definition.arguments
NUMBER = "#f8f8f8"            # constant.numeric
LINE_NUMBER = C["muted"]      # editorLineNumber.foreground
LINE_NUMBER_ACTIVE = C["foreground"]
INDENT_GUIDE = C["lighter_background"]  # editorIndentGuide.background, #2a2a2a
LINE_BORDER = "#303030"       # editor.lineHighlightBorder
CURSOR = C["foreground"]

K, F, T, P, D, N, X, S = KEYWORD, FUNCTION, CONSTANT, PARAMETER, DEFAULT, NUMBER, COMMENT, PUNCTUATION

CODE = [
    [("// less, but better.", X)],
    [("export", K), (" ", D), ("function", K), (" ", D), ("min", F), ("(", S), ("...", K), ("values", P),
     (":", K), (" ", D), ("number", T), ("[]", S), (")", S), (":", K), (" ", D), ("number", T), (" ", D), ("{", S)],
    [("  ", D), ("return", K), (" ", D), ("values", T), (".", F), ("reduce", F), ("((", S), ("a", P), (", ", S),
     ("b", P), (")", S), (" ", D), ("=>", K), (" ", D), ("(", S), ("a", D), (" ", D), ("<", K), (" ", D), ("b", D),
     (" ", D), ("?", K), (" ", D), ("a", D), (" ", D), (":", K), (" ", D), ("b", D), ("));", S)],
    [("}", S)],
    [],
    [("min", F), ("(", S), ("3", N), (", ", S), ("1", N), (", ", S), ("2", N), (");", S), (" ", D), ("// 1", X)],
]
ACTIVE = 5            # the line holding the cursor, from 0
INDENTED = (2,)       # lines inside the function body get an indent guide


def rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


def col(color):
    return np.array(rgb(color), np.float32)


def mix(a, b, t):
    return a + (b - a) * t


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def coords(w, h):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return xx, yy


def gradient(w, h, stops):
    """A vertical gradient through (position 0..1, color) stops."""
    ys = np.linspace(0, 1, h, dtype=np.float32)
    chans = [np.interp(ys, [p for p, _ in stops], [rgb(c)[i] for _, c in stops]) for i in range(3)]
    return np.broadcast_to(np.stack(chans, -1)[:, None, :], (h, w, 3)).astype(np.float32)


def noise(w, h, rng, cell, octaves=5, persistence=0.5):
    """Smooth value noise in 0..1: random grids, bicubic-upscaled, summed."""
    out = np.zeros((h, w), np.float32)
    amp = total = 1.0
    total = 0.0
    for o in range(octaves):
        size = max(2.0, cell / 2 ** o)
        grid = rng.random((max(2, int(h / size) + 2), max(2, int(w / size) + 2))).astype(np.float32)
        out += np.asarray(Image.fromarray(grid, "F").resize((w, h), Image.BICUBIC), np.float32) * amp
        total += amp
        amp *= persistence
    out /= total
    return (out - out.min()) / (out.max() - out.min() + 1e-6)


def add_glow(arr, xx, yy, cx, cy, sigma, color, strength):
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma * sigma))
    arr += g[..., None] * (col(color) * strength)


def add_stars(arr, rng, count, top, bottom, tint="#e8e4f4", s=1.0):
    """Small soft points, most faint, a few bright with a halo."""
    h, w, _ = arr.shape
    c = col(tint)
    for _ in range(count):
        x, y = rng.uniform(0, w), top * h + (bottom - top) * h * rng.random() ** 1.6
        r = rng.choice([0.55, 0.75, 1.0, 1.4], p=[0.45, 0.3, 0.17, 0.08]) * s
        b = rng.uniform(0.15, 1.0) ** 2.2
        halo = r > 1.2 * s
        reach = int(math.ceil((9 if halo else 4) * r)) + 1
        x0, x1, y0, y1 = max(0, int(x) - reach), min(w, int(x) + reach + 1), max(0, int(y) - reach), min(h, int(y) + reach + 1)
        py, px = np.mgrid[y0:y1, x0:x1].astype(np.float32)
        d2 = (px - x) ** 2 + (py - y) ** 2
        g = np.exp(-d2 / (2 * r * r)) * 1.7
        if halo:
            g += np.exp(-d2 / (2 * (4 * r) ** 2)) * 0.12
        arr[y0:y1, x0:x1] += g[..., None] * (c * b)


def vignette(arr, amount=0.3):
    h, w, _ = arr.shape
    xx, yy = coords(w, h)
    r2 = ((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2) / 0.5
    arr *= (1 - amount * r2)[..., None]
    return arr


def finish(arr, seed, grain=1.6):
    """Float RGB array -> image, with a little grain so gradients don't band."""
    rng = np.random.default_rng(seed + 99)
    arr = arr + rng.normal(0, grain, arr.shape[:2]).astype(np.float32)[..., None]
    return Image.fromarray(np.clip(arr + 0.5, 0, 255).astype(np.uint8))


def paint(arr, cover, color):
    """Lay color over arr where cover (h, w, 0..1) says so; color is an RGB
    triple or an (h, w, 3) array."""
    a = cover[..., None]
    arr *= 1 - a
    arr += a * color
    return arr


# --- code --------------------------------------------------------------------

def code_metrics(scale):
    size = round(34 * scale)
    font = ImageFont.truetype(FONT, size)
    cell = font.getlength("M")
    line_h = round(size * 1.7)
    gutter, gap = 4 * cell, 3 * cell  # right-aligned line numbers, then a gap
    widest = max(sum(len(t) for t, _ in line) for line in CODE)
    return {"font": font, "cell": cell, "line_h": line_h, "gutter": gutter, "gap": gap,
            "w": gutter + gap + widest * cell, "h": len(CODE) * line_h}


def draw_code(img, x0, y0, scale, line_span=None):
    """Draw CODE with its gutter's left edge at (x0, y0). The current line's
    border runs across line_span (left, right), the whole image by default."""
    m = code_metrics(scale)
    font, cell, line_h = m["font"], m["cell"], m["line_h"]
    draw = ImageDraw.Draw(img)
    code_x = x0 + m["gutter"] + m["gap"]
    ascent, descent = font.getmetrics()

    border = max(1, round(2 * scale))
    top = y0 + ACTIVE * line_h
    left, right = line_span or (-border, img.width + border)
    draw.rectangle([left, top, right, top + line_h - 1], outline=LINE_BORDER, width=border)

    for i, line in enumerate(CODE):
        # Centre the glyphs in the line box.
        y = y0 + i * line_h + (line_h - ascent - descent) / 2
        number = str(i + 1)
        draw.text((x0 + m["gutter"] - len(number) * cell, y), number, font=font,
                  fill=LINE_NUMBER_ACTIVE if i == ACTIVE else LINE_NUMBER)
        if i in INDENTED:
            gx = round(code_x)
            draw.line([(gx, y0 + i * line_h), (gx, y0 + (i + 1) * line_h - 1)], fill=INDENT_GUIDE,
                      width=max(1, round(scale)))
        x = code_x
        for text, color in line:
            draw.text((x, y), text, font=font, fill=color)
            x += len(text) * cell
        if i == ACTIVE:
            cursor_w = max(2, round(3 * scale))
            draw.rectangle([round(x) + 1, top + round(line_h * 0.18), round(x) + cursor_w, top + round(line_h * 0.82)],
                           fill=CURSOR)


def render_code(w, h, seed):
    scale = h / 2160
    img = Image.new("RGB", (w, h), C["background"])
    m = code_metrics(scale)
    draw_code(img, round((w - m["w"]) / 2), round((h - m["h"]) / 2), scale)
    return img


# --- shapes ------------------------------------------------------------------

def squircle_mask(w, h, radius, power=4, supersample=4):
    """An antialiased rounded rectangle whose corners are superellipses, like
    Hyprland's rounding_power and Apple's continuous corners."""
    mask = Image.new("L", (w, h), 255)
    n = radius * supersample
    yy, xx = np.mgrid[0:n, 0:n].astype(np.float32) + 0.5
    inside = ((1 - xx / n) ** power + (1 - yy / n) ** power) <= 1
    corner = Image.fromarray(inside.astype(np.uint8) * 255).resize((radius, radius), Image.LANCZOS)
    for flip_x, flip_y, pos in ((False, False, (0, 0)), (True, False, (w - radius, 0)),
                                (False, True, (0, h - radius)), (True, True, (w - radius, h - radius))):
        c = corner
        if flip_x:
            c = c.transpose(Image.FLIP_LEFT_RIGHT)
        if flip_y:
            c = c.transpose(Image.FLIP_TOP_BOTTOM)
        mask.paste(c, pos)
    return mask


def ridge(w, rng, roughness, smooth=0):
    """A 1D fractal skyline in -1..1 by midpoint displacement; `smooth` drops
    the finest octaves for rounder hills."""
    n = 1 << math.ceil(math.log2(w))
    hgt = np.zeros(n + 1)
    hgt[[0, n]] = rng.uniform(-1, 1, 2)
    step, amp, level = n, 1.0, 0
    while step > 1:
        half = step // 2
        mids = np.arange(half, n, step)
        noise = rng.normal(0, amp, len(mids)) if level < 14 - smooth else 0
        hgt[mids] = (hgt[mids - half] + hgt[mids + half]) / 2 + noise
        amp *= roughness
        step = half
        level += 1
    hgt = np.interp(np.linspace(0, n, w), np.arange(n + 1), hgt)
    return hgt / (np.abs(hgt).max() or 1)


# --- silk --------------------------------------------------------------------

def silk(w, h, seed, ribbons=None):
    """Twisting ribbons, lit from above: a bright fold near the lit edge, the
    underside darker where a ribbon turns over, a soft shadow cast below each,
    and the back ones softer, as if out of focus."""
    xx, yy = coords(w, h)
    X = xx / w
    arr = gradient(w, h, [(0, "#16141b"), (0.55, "#121116"), (1, "#0e0e11")]).copy()
    add_glow(arr, xx, yy, 0.58 * w, 0.42 * h, 0.42 * h, C["magenta"], 0.07)
    add_glow(arr, xx, yy, 0.30 * w, 0.75 * h, 0.35 * h, C["blue"], 0.04)
    gray = col(C["dark_background"])
    white = col("#ffffff")
    for y0, tilt, amp, freq, phase, width, twist, tphase, color, depth in ribbons:
        c = h * (y0 + tilt * (X - 0.5) + amp * np.sin(2 * np.pi * (freq * X + phase))
                 + 0.35 * amp * np.sin(2 * np.pi * (2.3 * freq * X + 1.7 * phase)))
        tw = np.sin(2 * np.pi * (twist * X + tphase))
        half = h * width * (0.10 + 0.90 * np.abs(tw))
        # The shadow it casts on what's behind it.
        ts = (yy - c - 0.04 * h) / (half * 1.3)
        arr *= (1 - (1 - smoothstep(0.4, 1.6, np.abs(ts))) * (0.30 + 0.25 * depth))[..., None]
        t = (yy - c) / half
        soft = (1.2 + 9.0 * (1 - depth)) / half
        inside = smoothstep(-1 - soft, -1 + soft, t) * (1 - smoothstep(1 - soft, 1 + soft, t))
        tc = np.clip(t, -1, 1)

        def face(side, dim):
            # 1 at the lit edge: shading falls off across the width, with a
            # bright fold near the lit edge and a broad sheen.
            shade = (0.20 + 0.80 * ((side + 1) / 2) ** 1.7) * dim
            fold = np.exp(-((side - 0.74) / 0.10) ** 2) * (0.20 + 0.55 * np.abs(tw)) * dim
            sheen = np.exp(-((side + 0.2) / 0.45) ** 2) * 0.10
            return shade + sheen, fold

        # Where the ribbon turns over (tw crosses zero, at its narrowest) the
        # front blends into the darker back face instead of jumping.
        front = smoothstep(-0.35, 0.35, tw)
        shade_f, fold_f = face(-tc, 1.0)
        shade_b, fold_b = face(tc, 0.62)
        shade = mix(shade_b, shade_f, front)
        fold = mix(fold_b, fold_f, front)
        base = mix(gray, col(color), 0.66)
        rib = base * shade[..., None] + white * (fold * 0.42)[..., None]
        rib *= 0.50 + 0.50 * depth
        paint(arr, inside * 0.95, rib)
    return arr


SILK = [
    # y, tilt, amplitude, frequency, phase, half-width, twists, twist phase, color, depth
    (0.30, -0.22, 0.05, 0.60, 0.70, 0.085, 0.55, 0.15, "#6b5a99", 0.0),
    (0.72, -0.20, 0.05, 0.75, 0.10, 0.080, 0.80, 0.30, C["cyan"], 0.15),
    (0.58, -0.34, 0.07, 0.65, 0.55, 0.115, 0.65, 0.05, C["blue"], 0.40),
    (0.46, -0.40, 0.06, 0.80, 0.20, 0.140, 0.55, 0.62, C["magenta"], 0.75),
    (0.64, -0.26, 0.05, 1.05, 0.85, 0.060, 1.15, 0.40, C["red"], 1.0),
]


def render_silk(w, h, seed):
    return finish(vignette(silk(w, h, seed, SILK), 0.35), seed)


# --- dunes -------------------------------------------------------------------

def render_dunes(w, h, seed):
    rng = np.random.default_rng(seed)
    xx, yy = coords(w, h)
    x = np.arange(w, dtype=np.float32) / w
    hz = 0.56
    arr = gradient(w, h, [(0, "#120f16"), (0.28, "#1b151d"), (hz, "#4d2d36"), (1, "#24181f")]).copy()
    sx, sy = 0.63 * w, 0.535 * h
    add_glow(arr, xx, yy, sx, sy, 0.34 * h, C["red"], 0.16)
    add_glow(arr, xx, yy, sx, sy, 0.11 * h, C["yellow"], 0.30)
    disc = smoothstep(0.046 * h + 1, 0.046 * h - 1, np.hypot(xx - sx, yy - sy))
    paint(arr, disc, col("#ffd9bd"))
    add_stars(arr, rng, int(160 * w * h / 8.3e6), 0.0, 0.30, s=h / 2160)

    layers = [
        # base, amplitude, frequency, lit color, shadow color, haze
        (0.585, 0.022, 2.3, "#8c5a5c", "#47303a", 0.60),
        (0.625, 0.036, 1.7, "#9d5e55", "#3a2832", 0.42),
        (0.690, 0.052, 1.25, "#b06650", "#2c1f27", 0.24),
        (0.785, 0.066, 0.95, "#ba6a4e", "#21171d", 0.10),
        (0.905, 0.072, 0.70, "#a85c45", "#170f14", 0.0),
    ]
    haze_color = col("#5a3640")
    glow = col("#ffc9a8")
    for base, amp, freq, lit_c, dark_c, haze in layers:
        # Rolling crests, a little skewed, so the slope (and the light) changes smoothly.
        u = freq * x + rng.random() + 0.10 * np.sin(2 * np.pi * (0.7 * freq * x + rng.random()))
        crest = 0.5 + 0.5 * np.sin(2 * np.pi * u) + 0.22 * np.sin(2 * np.pi * (2.2 * freq * x + rng.random()))
        yc = (base - amp * crest) * h
        slope = np.gradient(yc) / (amp * h * freq / w * 2 * np.pi)
        # Faces that drop away to the right look at the sun; they catch it
        # along the crest and fall into shade below.
        lit = smoothstep(-0.2, 0.7, slope)[None, :]
        below = yy - yc[None, :]
        top = mix(col(dark_c), col(lit_c), (0.15 + 0.70 * lit)[..., None])
        color = mix(top, col(dark_c) * 0.85, smoothstep(0, amp * h * 0.9, below)[..., None])
        rim = np.exp(-np.clip(below, 0, None) / (3.0 * h / 2160)) * (0.15 + 0.6 * lit)
        color += (rim * (1 - haze))[..., None] * glow * 0.45
        color = mix(color, haze_color, haze)
        paint(arr, np.clip(below + 0.5, 0, 1), color)
    return finish(vignette(arr, 0.32), seed)


# --- lake --------------------------------------------------------------------

def pine(draw, x, base, height, ss):
    """A pine silhouette: a trunk and stacked, slightly ragged tiers."""
    tiers = 7
    width = height * 0.30
    draw.rectangle([x - height * 0.012, base - height * 0.12, x + height * 0.012, base], fill=255)
    for i in range(tiers):
        t = i / tiers
        top = base - height * (0.10 + 0.90 * (i + 1) / tiers) - height * 0.06
        bottom = base - height * (0.08 + 0.80 * t)
        half = width * (1 - t * 0.82) / 2
        draw.polygon([(x - half, bottom), (x, top), (x + half, bottom)], fill=255)


def render_lake(w, h, seed):
    rng = np.random.default_rng(seed)
    s = h / 2160
    xx, yy = coords(w, h)
    hz = int(0.58 * h)
    arr = gradient(w, h, [(0, "#0c0c10"), (0.40, "#16141c"), (0.58, "#2a2531"), (1, "#2a2531")]).copy()
    mx, my = 0.67 * w, 0.21 * h
    add_glow(arr, xx, yy, mx, my, 0.32 * h, C["magenta"], 0.11)
    add_glow(arr, xx, yy, mx, my, 0.07 * h, "#efe9fb", 0.16)
    add_stars(arr, rng, int(520 * w * h / 8.3e6), 0.0, 0.50, s=s)

    # Ranges behind the lake: sharp peaks, moonlight along the faces that
    # turn toward the moon, the far ones paler in the haze.
    layers = [(0.26, 0.58, 0, "#3a3545", 0.50, 1.7), (0.18, 0.55, 1, "#282530", 0.25, 1.4),
              (0.10, 0.50, 2, "#17151c", 0.0, 1.2)]
    mist = col("#3a3442")
    moonlit = col("#a59bbf")
    for height, rough, smooth, color, haze, sharp in layers:
        prof = ((ridge(w, rng, rough, smooth) + 1) / 2) ** sharp
        yc = hz - height * h * (0.12 + 0.88 * prof)
        # Light follows the broad shape of the range, not every jag in it.
        k = max(6, int(110 * s))
        kernel = np.exp(-np.linspace(-2.5, 2.5, 2 * k + 1) ** 2)
        kernel /= kernel.sum()
        smooth_yc = np.convolve(np.pad(yc, k, mode="edge"), kernel, mode="valid")
        slope = np.gradient(smooth_yc) / (height * h / w * 25)
        # A face dropping away to the right looks right, toward the moon.
        lit = (smoothstep(0.0, 0.8, slope) if mx > w / 2 else smoothstep(0.0, 0.8, -slope))[None, :]
        below = yy - yc[None, :]
        below_smooth = np.clip(yy - smooth_yc[None, :], 0, None)
        rim = np.exp(-np.clip(below, 0, None) / (10 * s)) * lit * 0.55
        face = col(color) + (lit * np.exp(-below_smooth / (0.08 * h)) * 0.10)[..., None] * moonlit
        layer = face + rim[..., None] * moonlit * 0.6
        layer = mix(layer, mist, haze)
        cover = np.clip(below + 0.5, 0, 1) * (yy < hz)
        paint(arr, cover, layer)
    # Mist lying on the far shore.
    fog = np.exp(-((yy - hz) / (0.018 * h)) ** 2) * 0.55
    paint(arr, fog, mist)

    # The water mirrors everything above (the moon's own disc only shows as
    # glints), darker, broken into ripples.
    above = arr[:hz].copy()
    r = 0.032 * h
    moon = smoothstep(r + 1, r - 1, np.hypot(xx - mx, yy - my))
    mottle = noise(w, h, rng, 0.012 * h, 3)
    paint(arr, moon, col("#ece6f6")[None, None, :] * (0.88 + 0.12 * mottle)[..., None])
    rows = np.arange(hz, h)
    depth = (rows - hz) / (h - hz)
    src = np.clip(2 * hz - rows - 1, 0, hz - 1)
    ripple = (np.sin(rows * 0.9 * (1 + depth * 0.4)) + 0.6 * np.sin(rows * 2.3 + 1.1)) * (2 + 26 * depth) * s
    cols_ = np.clip((np.arange(w)[None, :] + ripple[:, None]).astype(int), 0, w - 1)
    water = above[src[:, None], cols_] * (0.50 - 0.18 * depth)[:, None, None] + col("#101016") * 0.45
    streaks = noise(w, h - hz, rng, 0.004 * h, 2)
    water *= (0.92 + 0.16 * streaks)[..., None]
    # The moon's path: short horizontal glints in a column that widens toward you.
    width = (8 + 0.16 * (rows - hz)) * s
    glints = np.exp(-((np.arange(w)[None, :] - mx) / width[:, None]) ** 2)
    xs = np.arange(w, dtype=np.float32)[None, :]
    freq = rng.uniform(0.02, 0.06, len(rows))[:, None] / s
    dash = smoothstep(0.55, 0.95, np.sin(xs * freq + rng.uniform(0, 2 * np.pi, len(rows))[:, None]))
    row_bright = (rng.random(len(rows)) ** 2.5)[:, None] * (rows % max(1, round(3 * s)) == 0)[:, None]
    water += (glints * dash * row_bright * (1.0 - 0.5 * depth)[:, None])[..., None] * col("#ece6f6") * 1.1
    arr[hz:] = water
    arr[hz - 1:hz + 1] = mix(arr[hz - 1:hz + 1], mist, 0.5)

    # Shores and pines in front, drawn 2x for clean edges.
    ss = 2
    mask = Image.new("L", (w * ss, h * ss), 0)
    d = ImageDraw.Draw(mask)
    for x0, x1, top, side in ((0, 0.34, 0.86, -1), (0.74, 1.0, 0.89, 1)):
        pts = [(x0 * w * ss, h * ss), (x1 * w * ss, h * ss)]
        for t in np.linspace(0, 1, 40):
            xs = x0 + (x1 - x0) * (1 - t) if side < 0 else x0 + (x1 - x0) * t
            edge = abs(t - (0 if side < 0 else 1))
            pts.append((xs * w * ss, (top + 0.10 * (1 - t) ** 2 + 0.006 * math.sin(t * 17)) * h * ss))
        d.polygon(pts, fill=255)
    for x0, x1, base, tallest, n in ((0.03, 0.26, 0.88, 0.46, 9), (0.80, 0.97, 0.905, 0.34, 7)):
        for _ in range(n):
            px = rng.uniform(x0, x1)
            tall = tallest * rng.uniform(0.45, 1.0) * (1 - 0.6 * abs(px - (x0 + x1) / 2) / (x1 - x0))
            pine(d, px * w * ss, (base + rng.uniform(0, 0.02)) * h * ss, tall * h * ss, ss)
    mask = np.asarray(mask.resize((w, h), Image.LANCZOS), np.float32) / 255
    paint(arr, mask, col("#09090c"))
    return finish(vignette(arr, 0.30), seed)


# --- planet ------------------------------------------------------------------

def sphere(arr, xx, yy, cx, cy, radius, light, surface, atmosphere=None, glow_width=0.0):
    """A lit sphere: Lambert shading from `light`, a soft terminator, and an
    atmosphere glowing around the lit limb."""
    nx, ny = (xx - cx) / radius, (yy - cy) / radius
    r2 = nx * nx + ny * ny
    nz = np.sqrt(np.clip(1 - r2, 0, 1))
    lx, ly, lz = light
    lam = nx * lx + ny * ly + nz * lz
    lit = smoothstep(-0.08, 0.55, lam)
    r = np.sqrt(r2)
    if atmosphere is not None:
        outside = np.clip(r - 1, 0, None) * radius
        side = np.clip((nx * lx + ny * ly) / (r + 1e-6) * 0.7 + 0.45, 0, 1)
        halo = np.exp(-outside / glow_width) * side * (r >= 1)
        arr += halo[..., None] * col(atmosphere) * 0.55
    body = surface * (0.04 + 0.96 * lit)[..., None]
    if atmosphere is not None:
        rim = (1 - nz) ** 3 * smoothstep(-0.2, 0.6, lam)
        body = body + rim[..., None] * col(atmosphere) * 0.85
    edge = smoothstep(1 + 1.2 / radius, 1 - 1.2 / radius, r)
    paint(arr, edge, body)


def render_planet(w, h, seed):
    rng = np.random.default_rng(seed)
    s = h / 2160
    xx, yy = coords(w, h)
    arr = gradient(w, h, [(0, "#0b0b0f"), (1, "#111017")]).copy()
    neb = noise(w, h, rng, 0.45 * h, 5) * noise(w, h, rng, 0.25 * h, 4)
    arr += (smoothstep(0.25, 0.75, neb) * 0.10)[..., None] * mix(col(C["magenta"]), col(C["blue"]), 0.4)
    add_stars(arr, rng, int(700 * w * h / 8.3e6), 0.0, 1.0, s=s)
    light = np.array([-0.82, -0.38, 0.30], np.float32)
    light /= np.linalg.norm(light)
    cx, cy, radius = 0.86 * w, 1.06 * h, 0.60 * h
    # Gas-giant bands: latitude stripes, gently warped by low noise.
    turb = noise(w, h, rng, 0.18 * h, 3)
    fine = noise(w, h, rng, 0.03 * h, 2)
    lat = (yy - cy) / radius
    band = 0.5 + 0.5 * np.sin(lat * 24 + turb * 1.6 + fine * 0.35)
    band2 = 0.5 + 0.5 * np.sin(lat * 9 + turb * 0.8 + 1.3)
    surface = mix(col("#2b2736"), col("#5d5270"), (band * 0.7)[..., None])
    surface = mix(surface, col("#41506a"), (band2 * 0.45)[..., None])
    sphere(arr, xx, yy, cx, cy, radius, light, surface, atmosphere="#8fb4ff", glow_width=0.05 * h)
    moon_surface = mix(col("#4b4954"), col("#77737f"), noise(w, h, rng, 0.01 * h, 3)[..., None])
    sphere(arr, xx, yy, 0.24 * w, 0.30 * h, 0.042 * h, light, moon_surface)
    return finish(vignette(arr, 0.25), seed)


# --- flow --------------------------------------------------------------------

def render_flow(w, h, seed):
    """Hairlines traced through a smooth flow field, thousands of them, each
    tinted by where it started."""
    rng = np.random.default_rng(seed)
    s = h / 2160
    xx, yy = coords(w, h)
    arr = gradient(w, h, [(0, "#131217"), (1, "#0f0f12")]).copy()
    add_glow(arr, xx, yy, 0.5 * w, 0.5 * h, 0.5 * h, C["magenta"], 0.05)
    q = 4
    # Curl noise: the flow runs along the contours of a smooth potential, so
    # lines swirl side by side instead of piling into a few veins.
    # A steady drift to the right turns the eddies into long streaming threads.
    psi = noise(w // q, h // q, rng, 700 / q * s, 3, 0.45)
    gy, gx = np.gradient(psi)
    m = np.hypot(gx, gy).mean() + 1e-9
    field = np.arctan2(-gx / m, gy / m + 1.1)
    n, steps, step = int(6500 * w * h / 8.3e6), 700, 1.4 * s
    # Start some threads off the left edge so they drift in and fill it.
    px, py = rng.uniform(-0.35 * w, w, n), rng.uniform(0, h, n)
    palette = [col(C["cyan"]), col(C["blue"]), col(C["magenta"]), col(C["red"])]
    t = np.clip(px / w + rng.normal(0, 0.08, n), 0, 0.999) * (len(palette) - 1)
    i0 = t.astype(int)
    colors = np.stack([palette[i] for i in i0]) * (1 - (t - i0))[:, None] + np.stack([palette[i + 1] for i in i0]) * (t - i0)[:, None]
    dens = np.zeros(h * w, np.float32)
    acc = np.zeros((h * w, 3), np.float32)
    for k in range(steps):
        a = field[np.clip((py / q).astype(int), 0, h // q - 1), np.clip((px / q).astype(int), 0, w // q - 1)]
        px += np.cos(a) * step
        py += np.sin(a) * step
        on = (px >= 0) & (px < w) & (py >= 0) & (py < h)
        idx = (py[on].astype(int) * w + px[on].astype(int))
        weight = math.sin(math.pi * (k + 0.5) / steps) * 0.11
        np.add.at(dens, idx, weight)
        np.add.at(acc, idx, colors[on] * weight)
    dens = dens.reshape(h, w)
    tint = acc.reshape(h, w, 3) / np.maximum(dens, 1e-6)[..., None]
    alpha = 1 - np.exp(-dens * 1.6)
    lines = tint * (0.85 * alpha)[..., None]
    halo = np.asarray(Image.fromarray(np.clip(lines, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(6 * s)), np.float32)
    arr = arr * (1 - alpha[..., None] * 0.9) + lines + halo * 0.35
    return finish(vignette(arr, 0.35), seed)


# --- topography --------------------------------------------------------------

def render_topography(w, h, seed):
    """Contour lines of an imaginary terrain: thin gray lines, every fifth
    stronger, one traced in Min's lavender, a summit marked."""
    rng = np.random.default_rng(seed)
    s = h / 2160
    xx, yy = coords(w, h)
    height = noise(w, h, rng, 0.55 * h, 5, 0.48)
    for cx, cy, sg, amp in ((0.62, 0.40, 0.16, 0.45), (0.30, 0.66, 0.12, 0.25)):
        height += amp * np.exp(-(((xx / w - cx) * w / h) ** 2 + (yy / h - cy) ** 2) / (2 * sg * sg))
    height = (height - height.min()) / (height.max() - height.min())
    gy, gx = np.gradient(height)
    shade = np.clip((-gx * 0.7 - gy * 0.7) * 900, -1, 1)
    arr = np.zeros((h, w, 3), np.float32) + col("#1b1b1d")
    arr += (shade * 5)[..., None]
    levels = 34
    v = height * levels
    grad = np.hypot(gx, gy) * levels + 1e-6
    level = np.round(v)
    dist = np.abs(v - level) / grad                      # pixels to the nearest contour
    index = (level % 5 == 0)
    width = np.where(index, 1.5, 0.9) * s
    line = np.clip(width + 0.5 - dist, 0, 1)
    color = np.where(index[..., None], col("#4c4c50"), col("#343437"))
    special = level == int(levels * 0.62)
    color = np.where(special[..., None], col(C["magenta"]), color)
    paint(arr, line * np.where(special, 1.0, 0.95), color)
    # Mark the summit.
    sy, sx = np.unravel_index(np.argmax(height), height.shape)
    img = finish(vignette(arr, 0.25), seed, grain=1.0)
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, round(26 * s))
    r = 7 * s
    d.polygon([(sx, sy - r), (sx - r, sy + r * 0.8), (sx + r, sy + r * 0.8)], fill=C["foreground"])
    d.text((sx + 16 * s, sy - 13 * s), "2184", font=font, fill=C["light_foreground"])
    return img


# --- windows -----------------------------------------------------------------

TS_KEYWORDS = {"export", "function", "return", "const", "let", "type", "extends", "if", "throw", "new",
               "import", "from"}
TS_TYPES = {"number", "string", "boolean", "void"}
STRING = C["yellow"]  # string.quoted

HERO_CODE = """// min.ts: less, but better.
type Comparable = number | string;

export function min<T extends Comparable>(...values: T[]): T {
  if (values.length === 0) throw new RangeError("min() needs a value");
  return values.reduce((a, b) => (a < b ? a : b));
}

export const clamp = (v: number, lo: number, hi: number) =>
  Math.max(lo, Math.min(v, hi));

console.log(min(3, 1, 2)); // 1""".split("\n")
HERO_PARAMS = {"values", "a", "b", "v", "lo", "hi"}


def ts_tokens(line, params=HERO_PARAMS):
    """Split a line of TypeScript into (text, color) runs, Min Dark style."""
    out, i = [], 0
    while i < len(line):
        ch = line[i]
        if line.startswith("//", i):
            out.append((line[i:], COMMENT))
            break
        if ch in "\"'`":
            j = line.find(ch, i + 1)
            j = len(line) if j < 0 else j + 1
            out.append((line[i:j], STRING))
            i = j
            continue
        if ch.isdigit():
            j = i
            while j < len(line) and (line[j].isalnum() or line[j] == "."):
                j += 1
            out.append((line[i:j], NUMBER))
            i = j
            continue
        if ch.isalpha() or ch in "_$":
            j = i
            while j < len(line) and (line[j].isalnum() or line[j] in "_$"):
                j += 1
            word, rest = line[i:j], line[j:].lstrip()
            if word in TS_KEYWORDS:
                color = KEYWORD
            elif word in TS_TYPES or rest.startswith("."):
                color = CONSTANT
            elif rest.startswith("(") or rest.startswith("<"):
                color = FUNCTION
            elif word in params:
                color = PARAMETER
            else:
                color = FUNCTION if word[0].isupper() else DEFAULT
            out.append((word, color))
            i = j
            continue
        for op in ("===", "=>", "...", "<=", ">=", "&&", "||"):
            if line.startswith(op, i):
                out.append((op, KEYWORD))
                i += len(op)
                break
        else:
            out.append((ch, KEYWORD if ch in "=<>?:|+-*/!&" else (DEFAULT if ch == " " else PUNCTUATION)))
            i += 1
    return out


def window(img, x0, y0, body, radius, s, active):
    """Composite a window body onto img at (x0, y0): squircle corners, two
    shadows, a hairline, and traffic lights (gray when inactive)."""
    w, h = img.size
    ww, wh = body.size
    mask = squircle_mask(ww, wh, radius)
    for blur, dy, alpha in ((60 * s, 30 * s, 0.55), (10 * s, 5 * s, 0.35)):
        layer = Image.new("L", (w, h), 0)
        layer.paste(mask, (x0, round(y0 + dy)))
        layer = layer.filter(ImageFilter.GaussianBlur(blur)).point(lambda v: v * alpha)
        img.paste(Image.new("RGB", (w, h), (0, 0, 0)), (0, 0), layer)
    img.paste(body, (x0, y0), mask)
    from PIL import ImageChops
    edge = ImageChops.subtract(mask, mask.filter(ImageFilter.MinFilter(5)))
    img.paste(Image.new("RGB", (ww, wh), (255, 255, 255)), (x0, y0), edge.point(lambda v: v * 0.11))
    ss = 4
    bar_h = round(56 * s)
    dots = Image.new("RGBA", (round(130 * s) * ss, bar_h * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(dots)
    size, gap, cx = 22 * s * ss, 14 * s * ss, 24 * s * ss
    for color in (("#ff5f57", "#febc2e", "#28c840") if active else ("#4a4a4c",) * 3):
        d.ellipse([cx, bar_h * ss / 2 - size / 2, cx + size, bar_h * ss / 2 + size / 2], fill=color)
        cx += size + gap
    dots = dots.resize((dots.width // ss, bar_h), Image.LANCZOS)
    img.paste(dots, (x0, y0), dots)


def render_window(w, h, seed):
    s = h / 2160
    img = Image.fromarray(np.clip(silk(w, h, seed, SILK) * 0.72, 0, 255).astype(np.uint8))
    mono = ImageFont.truetype(FONT, round(28 * s))
    small = ImageFont.truetype(FONT, round(22 * s))
    ui = ImageFont.truetype(UI_FONT, round(23 * s))
    cell = mono.getlength("M")
    asc, desc = mono.getmetrics()

    # The editor: activity bar, explorer, a tab, the code, a status bar.
    ew, eh = round(0.58 * w), round(0.60 * h)
    body = Image.new("RGB", (ew, eh), C["background"])
    d = ImageDraw.Draw(body)
    bar, tabs, status, act, side = round(56 * s), round(50 * s), round(34 * s), round(58 * s), round(330 * s)
    d.rectangle([0, 0, ew, bar], fill=C["dark_background"])
    d.text((ew / 2, bar / 2), "min.ts — min-dark", font=ui, fill=C["muted"], anchor="mm")
    d.rectangle([0, bar, act + side, eh], fill=C["dark_background"])
    for i, glyph in enumerate(("", "", "", "")):
        d.text((act / 2, bar + 46 * s + i * 66 * s), glyph, font=ImageFont.truetype(FONT, round(30 * s)),
               fill=C["foreground"] if i == 0 else "#5c5c5c", anchor="mm")
    d.line([(act, bar), (act, eh)], fill="#232323", width=max(1, round(s)))
    d.text((act + 22 * s, bar + 24 * s), "EXPLORER", font=ImageFont.truetype(UI_FONT, round(19 * s)),
           fill=C["muted"])
    tree = [("v min-dark", 0, C["foreground"]), ("> backgrounds", 1, "#9e9e9e"), ("v tools", 1, "#9e9e9e"),
            ("make-wallpaper.py", 2, "#9e9e9e"), ("colors.toml", 1, "#9e9e9e"), ("min.ts", 1, "#f5f5f5"),
            ("README.md", 1, "#9e9e9e")]
    chevron = ImageFont.truetype(FONT, round(20 * s))
    for i, (name, depth, color) in enumerate(tree):
        y = bar + 70 * s + i * 40 * s
        if name == "min.ts":
            d.rectangle([act, y - 6 * s, act + side, y + 32 * s], fill="#262626")
        x = act + (24 + depth * 20) * s
        if name[:2] in ("v ", "> "):
            d.text((x, y + 15 * s), "▾" if name[0] == "v" else "▸", font=chevron, fill="#7d7d7d", anchor="lm")
            name = name[2:]
        d.text((x + 20 * s, y), name, font=ui, fill=color)
    ex = act + side
    d.rectangle([ex, bar, ew, bar + tabs], fill=C["dark_background"])
    d.rectangle([ex, bar, ex + 170 * s, bar + tabs], fill=C["background"])
    d.text((ex + 85 * s, bar + tabs / 2), "min.ts", font=ui, fill="#fafafa", anchor="mm")
    d.text((ex + 250 * s, bar + tabs / 2), "colors.toml", font=ui, fill=C["muted"], anchor="mm")
    line_h = round(28 * s * 1.75)
    top = bar + tabs + 26 * s
    active = 11
    gutter = ex + 36 * s
    code_x = gutter + 4 * cell + 2.5 * cell
    lt = top + active * line_h
    d.rectangle([ex, lt, ew, lt + line_h - 1], outline=LINE_BORDER, width=max(1, round(2 * s)))
    for i, src in enumerate(HERO_CODE):
        y = top + i * line_h + (line_h - asc - desc) / 2
        num = str(i + 1)
        d.text((gutter + (4 - len(num)) * cell, y), num, font=mono, fill=LINE_NUMBER_ACTIVE if i == active else LINE_NUMBER)
        if 4 <= i <= 5 or i == 9:
            gx = round(code_x)
            d.line([(gx, top + i * line_h), (gx, top + (i + 1) * line_h - 1)], fill=INDENT_GUIDE, width=max(1, round(s)))
        x = code_x
        for text, color in ts_tokens(src):
            d.text((x, y), text, font=mono, fill=color)
            x += len(text) * cell
        if i == active:
            d.rectangle([x + 2, lt + line_h * 0.2, x + 2 + 3 * s, lt + line_h * 0.8], fill=CURSOR)
    d.rectangle([0, eh - status, ew, eh], fill=C["dark_background"])
    d.text((act + 18 * s, eh - status / 2), " main", font=small, fill="#7e7e7e", anchor="lm")
    d.text((ew - 24 * s, eh - status / 2), "TypeScript   UTF-8   Ln 12, Col 30", font=small, fill="#7e7e7e", anchor="rm")
    window(img, round(0.115 * w), round(0.13 * h), body, round(24 * s), s, active=False)

    # The terminal, in front.
    tw, th = round(0.34 * w), round(0.33 * h)
    term = Image.new("RGB", (tw, th), C["background"])
    d = ImageDraw.Draw(term)
    d.rectangle([0, 0, tw, bar], fill=C["dark_background"])
    d.text((tw / 2, bar / 2), "~/min-dark", font=ui, fill=C["muted"], anchor="mm")
    lines = [
        [("❯ ", C["magenta"]), ("bun test", C["foreground"])],
        [("bun test v1.2", C["muted"])],
        [],
        [("min.test.ts:", C["foreground"])],
        [("✓ ", C["green"]), ("min() picks the smallest value ", C["foreground"]), ("[0.09ms]", C["muted"])],
        [("✓ ", C["green"]), ("min() throws on no values ", C["foreground"]), ("[0.02ms]", C["muted"])],
        [],
        [(" 2 pass", C["green"])],
        [(" 0 fail", C["muted"])],
        [("Ran 2 tests across 1 file. ", C["muted"]), ("[14.00ms]", C["muted"])],
        [("❯ ", C["magenta"])],
    ]
    tline = round(26 * s * 1.6)
    tfont = ImageFont.truetype(FONT, round(26 * s))
    tcell = tfont.getlength("M")
    for i, runs in enumerate(lines):
        x, y = 30 * s, bar + 26 * s + i * tline
        for text, color in runs:
            d.text((x, y), text, font=tfont, fill=color)
            x += len(text) * tcell
        if i == len(lines) - 1:
            d.rectangle([x, y + 2 * s, x + tcell, y + tline - 8 * s], fill=CURSOR)
    window(img, round(0.535 * w), round(0.535 * h), term, round(24 * s), s, active=True)
    return img


# --- main --------------------------------------------------------------------

RENDERERS = {
    "code": ("1-min-code.png", render_code),
    "silk": ("2-silk.jpg", render_silk),
    "dunes": ("3-dunes.jpg", render_dunes),
    "lake": ("4-lake.jpg", render_lake),
    "planet": ("5-planet.jpg", render_planet),
    "flow": ("6-flow.jpg", render_flow),
    "topography": ("7-topography.jpg", render_topography),
    "window": ("8-window.jpg", render_window),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("names", nargs="*", default=["all"], help="which to render: " + ", ".join(RENDERERS) + ", or all")
    ap.add_argument("--width", type=int, default=3840)
    ap.add_argument("--height", type=int, default=2160)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default=os.path.join(THEME_DIR, "backgrounds"))
    args = ap.parse_args()

    names = list(RENDERERS) if "all" in args.names else args.names
    os.makedirs(args.out, exist_ok=True)
    for name in names:
        filename, render = RENDERERS[name]
        img = render(args.width, args.height, args.seed)
        path = os.path.join(args.out, filename)
        if path.endswith(".jpg"):
            img.save(path, quality=92, subsampling=0, optimize=True)
        else:
            img.save(path, optimize=True)
        print(f"Wrote {path} ({args.width}x{args.height}, {os.path.getsize(path) // 1024} KB)")


if __name__ == "__main__":
    sys.exit(main())
