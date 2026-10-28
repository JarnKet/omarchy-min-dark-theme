#!/usr/bin/env python3
"""Render the Min Dark wallpapers into backgrounds/.

  code      a few lines of TypeScript as VS Code shows them in Min Dark: line
            numbers, an indent guide, the current line outlined in Min's
            #303030 across the whole screen, the cursor at its end
  aurora    soft lavender and blue light on Min's dark gray, macOS-style
  dusk      the same, pink and orange low on the canvas
  window    the code in a macOS-style window (squircle corners, shadow,
            traffic lights) over the aurora
  dots      a dot grid fading out toward the edges
  mountains a night range: layered ridges, mist, stars, a lavender moon
  hills     rounder hills at dawn with a faint pink horizon

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


def finish(arr, seed, grain=1.4):
    """Float RGB array -> image, with a little grain so gradients don't band."""
    rng = np.random.default_rng(seed + 99)
    arr = arr + rng.normal(0, grain, arr.shape[:2])[..., None]
    return Image.fromarray(np.clip(arr + 0.5, 0, 255).astype(np.uint8))


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


# --- soft light (aurora, dusk) -----------------------------------------------

def soft_light(w, h, blobs, seed, base=None):
    """Blobs of colored light added onto Min's chrome gray. Each blob is
    (cx, cy, rx, ry, degrees, color, strength), positions and radii as
    fractions of the width (x) and height (y)."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    arr = np.zeros((h, w, 3), np.float32) + np.array(rgb(base or C["dark_background"]), np.float32)
    for cx, cy, rx, ry, degrees, color, strength in blobs:
        a = math.radians(degrees)
        dx, dy = (xx - cx * w) / w, (yy - cy * h) / h * (h / w)
        u = dx * math.cos(a) + dy * math.sin(a)
        v = -dx * math.sin(a) + dy * math.cos(a)
        falloff = np.exp(-((u / rx) ** 2 + (v / (ry * h / w)) ** 2))
        arr += falloff[..., None] * (np.array(rgb(color), np.float32) * strength)
    # Vignette: the corners fall back toward black.
    r2 = ((xx / w - 0.5) ** 2 + (yy / h - 0.5) ** 2) / 0.5
    arr *= (1 - 0.32 * r2)[..., None]
    return arr


def render_aurora(w, h, seed):
    return finish(soft_light(w, h, [
        (0.34, 0.30, 0.42, 0.40, -18, C["magenta"], 0.42),
        (0.70, 0.56, 0.32, 0.34, 24, C["blue"], 0.30),
        (0.55, 0.88, 0.50, 0.16, 0, C["cyan"], 0.12),
    ], seed), seed)


def render_dusk(w, h, seed):
    return finish(soft_light(w, h, [
        (0.40, 0.98, 0.56, 0.34, 8, C["red"], 0.40),
        (0.74, 1.02, 0.30, 0.26, -10, C["yellow"], 0.30),
        (0.20, 0.14, 0.34, 0.26, 0, C["magenta"], 0.12),
    ], seed), seed)


# --- a macOS-style window ----------------------------------------------------

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


def render_window(w, h, seed):
    s = h / 2160
    img = render_aurora(w, h, seed)
    code_scale = 1.3 * s
    m = code_metrics(code_scale)
    pad_x, pad_y, title_h = round(84 * s), round(64 * s), round(64 * s)
    win_w, win_h = round(m["w"] + 2 * pad_x), round(title_h + m["h"] + 2 * pad_y)
    x0, y0 = (w - win_w) // 2, (h - win_h) // 2
    radius = round(24 * s)  # 12 px corners, drawn at 2x
    mask = squircle_mask(win_w, win_h, radius)

    # Shadow: a wide soft one below the window and a tight one around it.
    for blur, dy, alpha in ((70 * s, 36 * s, 0.62), (12 * s, 6 * s, 0.40)):
        layer = Image.new("L", (w, h), 0)
        layer.paste(mask, (x0, round(y0 + dy)))
        layer = layer.filter(ImageFilter.GaussianBlur(blur)).point(lambda v: v * alpha)
        img.paste(Image.new("RGB", (w, h), (0, 0, 0)), (0, 0), layer)

    body = Image.new("RGB", (win_w, win_h), C["background"])
    ImageDraw.Draw(body).rectangle([0, 0, win_w, title_h - 1], fill=C["dark_background"])
    ImageDraw.Draw(body).line([(0, title_h - 1), (win_w, title_h - 1)], fill=C["lighter_background"],
                              width=max(1, round(s)))
    img.paste(body, (x0, y0), mask)

    # Hairline: a 2 px band just inside the edge, white at low alpha.
    edge = ImageChops_subtract(mask, mask.filter(ImageFilter.MinFilter(5)))
    img.paste(Image.new("RGB", (win_w, win_h), (255, 255, 255)), (x0, y0), edge.point(lambda v: v * 0.12))

    # Traffic lights and the title, drawn 4x and scaled down for smooth edges.
    ss = 4
    bar = Image.new("RGBA", (win_w * ss, title_h * ss), (0, 0, 0, 0))
    d = ImageDraw.Draw(bar)
    dot, gap, cx = 24 * s * ss, 16 * s * ss, 28 * s * ss
    cy = title_h * ss / 2
    for color in ("#ff5f57", "#febc2e", "#28c840"):
        d.ellipse([cx, cy - dot / 2, cx + dot, cy + dot / 2], fill=color)
        cx += dot + gap
    font = ImageFont.truetype(UI_FONT, round(25 * s * ss))
    d.text((win_w * ss / 2, cy), "min.ts", font=font, fill=C["light_foreground"], anchor="mm")
    bar = bar.resize((win_w, title_h), Image.LANCZOS)
    img.paste(bar, (x0, y0), bar)

    draw_code(img, x0 + pad_x - round(m["cell"]), y0 + title_h + pad_y, code_scale,
              line_span=(x0 + 1, x0 + win_w - 2))
    return img


def ImageChops_subtract(a, b):
    from PIL import ImageChops
    return ImageChops.subtract(a, b)


# --- dots --------------------------------------------------------------------

def render_dots(w, h, seed):
    s = h / 2160
    pitch, radius = 48 * s, 3.0 * s
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    # Distance to the nearest grid point, the grid centred on the screen.
    gx = (xx - w / 2) / pitch
    gy = (yy - h / 2) / pitch
    dist = np.hypot((gx - np.round(gx)) * pitch, (gy - np.round(gy)) * pitch)
    cover = np.clip(radius + 0.5 - dist, 0, 1)
    # Dots fade out toward the edges.
    fade = np.clip(1.15 - 1.3 * (((xx / w - 0.5) * 1.1) ** 2 + (yy / h - 0.5) ** 2) ** 0.5 * 1.6, 0, 1)
    bg, dot = np.array(rgb(C["background"]), np.float32), np.array(rgb("#424242"), np.float32)
    arr = bg + (dot - bg) * (cover * fade)[..., None]
    # One dot, a little off centre, in Min's function lavender.
    px, py = np.round((0.618 * w - w / 2) / pitch) * pitch + w / 2, np.round((0.382 * h - h / 2) / pitch) * pitch + h / 2
    lav = np.clip(radius * 1.25 + 0.5 - np.hypot(xx - px, yy - py), 0, 1)
    arr += (np.array(rgb(C["magenta"]), np.float32) - arr) * lav[..., None]
    return Image.fromarray(np.clip(arr + 0.5, 0, 255).astype(np.uint8))


# --- landscapes --------------------------------------------------------------

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


def sky(w, h, top, horizon, horizon_y):
    t = np.clip(np.arange(h, dtype=np.float32) / (horizon_y * h), 0, 1)[:, None, None]
    return np.broadcast_to(np.array(rgb(top), np.float32) * (1 - t) + np.array(rgb(horizon), np.float32) * t,
                           (h, w, 3)).copy()


def paint_layers(arr, layers, mist_color, rng):
    """Paint ridges far to near; mist settles at the foot of each farther one."""
    h, w, _ = arr.shape
    yy = np.arange(h, dtype=np.float32)[:, None]
    mist = np.array(rgb(mist_color), np.float32)
    for i, (base, amp, rough, smooth, color) in enumerate(layers):
        line = base * h - amp * h * ridge(w, rng, rough, smooth)
        if i:
            # Mist: thickest just above this layer's lowest crest.
            top = line.min()
            t = np.clip((yy - (top - 0.10 * h)) / (0.12 * h), 0, 1) * 0.45
            arr += (mist - arr) * t[..., None]
        cover = np.clip(yy - line[None, :] + 0.5, 0, 1)
        arr += (np.array(rgb(color), np.float32) - arr) * cover[..., None]
    return arr


def glow(arr, cx, cy, sigma, color, strength):
    h, w, _ = arr.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    g = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * sigma ** 2))
    arr += g[..., None] * np.array(rgb(color), np.float32) * strength


def render_mountains(w, h, seed):
    rng = np.random.default_rng(seed)
    arr = sky(w, h, "#111114", "#2a2733", 0.62)
    # Stars, mostly faint, thinning toward the horizon.
    n = int(420 * w * h / (3840 * 2160))
    xs, ys = rng.uniform(0, w, n), h * 0.58 * rng.power(0.6, n)
    for x, y, b in zip(xs, ys, rng.uniform(0.15, 0.7, n) ** 2):
        xi, yi = int(x), int(y)
        arr[yi:yi + 2, xi:xi + 2] += np.array(rgb("#d9d4ea"), np.float32) * b
    # A lavender moon with a wide glow.
    mx, my = 0.72 * w, 0.21 * h
    glow(arr, mx, my, 0.10 * h, C["magenta"], 0.22)
    glow(arr, mx, my, 0.025 * h, "#efe9fb", 0.35)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    disc = np.clip(0.021 * h + 0.5 - np.hypot(xx - mx, yy - my), 0, 1)
    arr += (np.array(rgb("#ece6f7"), np.float32) - arr) * disc[..., None]
    paint_layers(arr, [
        (0.60, 0.13, 0.56, 0, "#3a3843"),
        (0.66, 0.11, 0.54, 0, "#2e2c35"),
        (0.72, 0.09, 0.52, 1, "#232229"),
        (0.79, 0.08, 0.50, 1, "#1a191e"),
        (0.86, 0.07, 0.48, 2, "#131216"),
        (0.94, 0.05, 0.45, 3, "#0d0c0f"),
    ], "#2b2932", rng)
    return finish(arr, seed)


def render_hills(w, h, seed):
    rng = np.random.default_rng(seed)
    arr = sky(w, h, "#16151a", "#3b2c33", 0.64)
    glow(arr, 0.5 * w, 0.66 * h, 0.16 * h, C["red"], 0.20)
    glow(arr, 0.58 * w, 0.66 * h, 0.06 * h, C["yellow"], 0.12)
    paint_layers(arr, [
        (0.66, 0.06, 0.42, 3, "#4a3d46"),
        (0.71, 0.06, 0.40, 4, "#382f37"),
        (0.77, 0.06, 0.40, 4, "#29242a"),
        (0.84, 0.06, 0.38, 5, "#1d1a1e"),
        (0.92, 0.05, 0.36, 5, "#141215"),
    ], "#3a3036", rng)
    return finish(arr, seed)


# --- main --------------------------------------------------------------------

RENDERERS = {
    "code": ("1-min-code.png", render_code),
    "aurora": ("2-abstract-aurora.jpg", render_aurora),
    "dusk": ("3-abstract-dusk.jpg", render_dusk),
    "window": ("4-window.jpg", render_window),
    "dots": ("5-dots.png", render_dots),
    "mountains": ("6-mountains.jpg", render_mountains),
    "hills": ("7-hills.jpg", render_hills),
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
