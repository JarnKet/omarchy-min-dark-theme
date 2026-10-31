#!/usr/bin/env python3
"""Render the Min Dark wallpapers into backgrounds/.

One series of quiet charts, all drawn the same way: hairlines in two grays on
near-black, a few small mono labels, and one feature picked out in one of
Min's syntax colors.

  topography  contour lines of an imaginary terrain, one in lavender
  ridgelines  stacked elevation profiles across a range, one in pink
  rivers      the river network draining a range, the main river in blue
  globe       a wireframe globe with the land in dots, the flight from
              Singapore to San Francisco in orange
  circuit     a circuit board's chips and traces, one trace in green
  lunar       a crater field as contours, one crater's wall in orange
  nautical    a coast with depth contours and soundings, the 20 m line in blue
  stars       a star chart of invented constellations, one in lavender
  rings       the growth rings of a sawn log, one ring in green

Every color comes from colors.toml, and each picture keeps its subject inside
the middle 3456 px so a 16:10 crop keeps it.

Usage: tools/make-wallpaper.py [name ...|all] [--width 3840 --height 2160] [--seed N]
"""

import argparse
import heapq
import math
import os
import re
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

THEME_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT = "/usr/share/fonts/TTF/JetBrainsMonoNerdFont-Regular.ttf"
LAND = os.path.join(THEME_DIR, "tools", "world-land.png")  # Natural Earth 1:50m, 0.1° a pixel


def theme_colors():
    with open(os.path.join(THEME_DIR, "colors.toml")) as f:
        return {k: "#" + v for k, v in re.findall(r'^(\w+)\s*=\s*"#([0-9a-fA-F]{6})"', f.read(), re.M)}


C = theme_colors()

GROUND = "#1b1b1d"
MINOR = "#343437"             # a line among many
MAJOR = "#4c4c50"             # every fifth line, outlines
LABEL = C["light_foreground"]
MARK = C["foreground"]
# The one accent each picture gets: Min's syntax colors.
LAVENDER = C["magenta"]       # functions
PINK = C["red"]               # keywords
BLUE = C["blue"]              # constants
ORANGE = C["yellow"]          # strings
GREEN = C["green"]


def rgb(color):
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) for i in (0, 2, 4))


def col(color):
    return np.array(rgb(color), np.float32)


def mix(a, b, t):
    return a + (b - a) * t


def hexmix(a, b, t):
    return "#%02x%02x%02x" % tuple(round(float(v)) for v in mix(col(a), col(b), t))


def smoothstep(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3 - 2 * t)


def coords(w, h):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    return xx, yy


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


def normalize(a):
    return (a - a.min()) / (a.max() - a.min())


def ground(w, h):
    return np.zeros((h, w, 3), np.float32) + col(GROUND)


def relief(arr, field, strength=5):
    """Shade arr very slightly as if field were lit from the top left."""
    gy, gx = np.gradient(field)
    shade = np.clip((-gx * 0.7 - gy * 0.7) * 900, -1, 1)
    arr += (shade * strength)[..., None]


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


def contour_lines(arr, field, levels, s, every=5, widths=(0.9, 1.5), minor=MINOR, major=MAJOR,
                  accent=None, accent_level=None, where=None, mask=None):
    """Draw the contours of field (where field * levels crosses a whole
    number) onto arr as smooth hairlines: every `every`th one wider and in
    `major`, the one at `accent_level` (only `where`, if given) in `accent`.
    `mask` (0..1) hides lines."""
    gy, gx = np.gradient(field)
    v = field * levels
    grad = np.hypot(gx, gy) * levels + 1e-6
    level = np.round(v)
    dist = np.abs(v - level) / grad                      # pixels to the nearest contour
    index = (level % every == 0)
    width = np.where(index, widths[1], widths[0]) * s
    line = np.clip(width + 0.5 - dist, 0, 1)
    color = np.where(index[..., None], col(major), col(minor))
    opacity = 0.95
    if accent is not None:
        special = level == accent_level
        if where is not None:
            special &= where
        color = np.where(special[..., None], col(accent), color)
        opacity = np.where(special, 1.0, 0.95)
    if mask is not None:
        line = line * mask
    paint(arr, line * opacity, color)


class Plate:
    """Line work for a picture, drawn three times the size and scaled down so
    hairlines come out smooth, over a float RGB array (the ground, and any
    contours already on it). Labels go on last, at full size, crisp."""

    SS = 3

    def __init__(self, arr):
        self.h, self.w = arr.shape[:2]
        base = Image.fromarray(np.clip(arr + 0.5, 0, 255).astype(np.uint8))
        self.img = base.resize((self.w * self.SS, self.h * self.SS), Image.NEAREST)
        self.draw = ImageDraw.Draw(self.img)
        self.labels = []

    def _xy(self, points):
        k = self.SS
        return [(float(x) * k, float(y) * k) for x, y in points]

    def _px(self, width):
        return max(1, round(width * self.SS))

    def line(self, points, color, width):
        if len(points) > 1:
            self.draw.line(self._xy(points), fill=color, width=self._px(width), joint="curve")

    def stroke(self, points, widths, colors):
        """A polyline whose width and color change along it, one of each per
        point; drawn in runs where they hold steady."""
        run, key = [points[0]], (self._px(widths[0]), colors[0])
        for p, wd, c in zip(points[1:], widths[1:], colors[1:]):
            run.append(p)
            k = (self._px(wd), c)
            if k != key:
                self.draw.line(self._xy(run), fill=key[1], width=key[0], joint="curve")
                run, key = [p], k
        if len(run) > 1:
            self.draw.line(self._xy(run), fill=key[1], width=key[0], joint="curve")

    def dashed(self, points, color, width, dash, gap):
        pts = np.asarray(points, np.float64)
        seg = np.hypot(*np.diff(pts, axis=0).T)
        at = np.concatenate([[0], np.cumsum(seg)])
        start = 0.0
        while start < at[-1]:
            end = min(start + dash, at[-1])
            t = np.linspace(start, end, max(2, int((end - start) / 4) + 2))
            self.line(list(zip(np.interp(t, at, pts[:, 0]), np.interp(t, at, pts[:, 1]))), color, width)
            start = end + gap

    def polygon(self, points, fill):
        self.draw.polygon(self._xy(points), fill=fill)

    def circle(self, x, y, r, fill=None, outline=None, width=1.0):
        k = self.SS
        box = [(x - r) * k, (y - r) * k, (x + r) * k, (y + r) * k]
        if outline:
            self.draw.ellipse(box, fill=fill, outline=outline, width=self._px(width))
        else:
            self.draw.ellipse(box, fill=fill)

    def rect(self, x0, y0, x1, y1, fill=None, outline=None, width=1.0):
        k = self.SS
        box = [x0 * k, y0 * k, x1 * k, y1 * k]
        if outline:
            self.draw.rectangle(box, fill=fill, outline=outline, width=self._px(width))
        else:
            self.draw.rectangle(box, fill=fill)

    def summit(self, x, y, s):
        """The topography wallpaper's summit mark: a small solid triangle."""
        r = 7 * s
        self.polygon([(x, y - r), (x - r, y + r * 0.8), (x + r, y + r * 0.8)], MARK)

    def label(self, x, y, text, s, color=LABEL, size=26, anchor="ls"):
        self.labels.append((x, y, text, color, round(size * s), anchor))

    def finish(self, seed, vignette_amount=0.25, grain=1.0):
        arr = np.asarray(self.img.resize((self.w, self.h), Image.BOX), np.float32)
        img = finish(vignette(arr, vignette_amount), seed, grain)
        d = ImageDraw.Draw(img)
        for x, y, text, color, size, anchor in self.labels:
            d.text((x, y), text, font=ImageFont.truetype(FONT, size), fill=color, anchor=anchor)
        return img


def chaikin(points, n=2):
    """Round a polyline's corners, keeping its two ends."""
    pts = np.asarray(points, np.float64)
    for _ in range(n):
        if len(pts) < 3:
            break
        q = 0.75 * pts[:-1] + 0.25 * pts[1:]
        r = 0.25 * pts[:-1] + 0.75 * pts[1:]
        mid = np.empty((2 * len(q), 2))
        mid[0::2], mid[1::2] = q, r
        pts = np.vstack([pts[:1], mid[1:-1], pts[-1:]])
    return pts


def scatter(w, h, r, rng, accept=None, attempts=None):
    """Random points at least r apart (dart throwing on a grid), only where
    accept(x, y) agrees."""
    cell = r / math.sqrt(2)
    gw, gh = int(w / cell) + 1, int(h / cell) + 1
    grid = -np.ones((gh, gw), np.int64)
    pts = []
    attempts = attempts or int(3 * w * h / (r * r))
    for x, y in zip(rng.uniform(0, w, attempts), rng.uniform(0, h, attempts)):
        if accept is not None and not accept(x, y):
            continue
        gx, gy = int(x / cell), int(y / cell)
        near = grid[max(0, gy - 2):gy + 3, max(0, gx - 2):gx + 3]
        if any((pts[k][0] - x) ** 2 + (pts[k][1] - y) ** 2 < r * r for k in near[near >= 0]):
            continue
        grid[gy, gx] = len(pts)
        pts.append((x, y))
    return pts


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
    height = normalize(height)
    arr = ground(w, h)
    relief(arr, height)
    levels = 34
    contour_lines(arr, height, levels, s, accent=LAVENDER, accent_level=int(levels * 0.62))
    # Mark the summit.
    sy, sx = np.unravel_index(np.argmax(height), height.shape)
    img = finish(vignette(arr, 0.25), seed, grain=1.0)
    d = ImageDraw.Draw(img)
    font = ImageFont.truetype(FONT, round(26 * s))
    r = 7 * s
    d.polygon([(sx, sy - r), (sx - r, sy + r * 0.8), (sx + r, sy + r * 0.8)], fill=MARK)
    d.text((sx + 16 * s, sy - 13 * s), "2184", font=font, fill=LABEL)
    return img


# --- ridgelines --------------------------------------------------------------

def render_ridgelines(w, h, seed):
    """Elevation profiles of a range, one behind the other, each hiding what's
    behind it; the far ones dimmer. One profile in pink, its peak marked."""
    rng = np.random.default_rng(seed + 1)
    s = h / 2160
    rows, top, bottom = 64, 0.27 * h, 0.86 * h
    gap = (bottom - top) / (rows - 1)
    step = 3 * s
    xs = np.arange(-4 * step, w + 4 * step, step)
    # One terrain, sliced: neighbouring profiles share their mountains.
    terrain = noise(len(xs), rows * 4, rng, 0.11 * len(xs), 6, 0.5)
    envelope = 0.18 + 0.82 * np.exp(-((xs / w - 0.5) / 0.26) ** 2)
    pink = int(rows * 0.58)
    plate = Plate(ground(w, h))
    peak = None
    for i in range(rows):
        n = terrain[i * 4]
        ridged = 1 - np.abs(2 * n - 1)
        lift = (0.65 * ridged ** 2.2 + 0.35 * n) * envelope * gap * 13
        ys = top + i * gap - lift
        pts = list(zip(xs, ys))
        plate.polygon(pts + [(xs[-1], h + 10), (xs[0], h + 10)], GROUND)
        if i == pink:
            plate.line(pts, PINK, 1.4 * s)
            k = int(np.argmin(np.where(np.abs(xs - w / 2) < 0.4 * w, ys, np.inf)))
            peak = (xs[k], ys[k])
        else:
            plate.line(pts, hexmix("#2d2d30", "#56565a", (i / (rows - 1)) ** 0.8), 1.15 * s)
    px, py = peak
    plate.summit(px, py - 20 * s, s)
    plate.label(px + 16 * s, py - 13 * s, "1450", s)
    return plate.finish(seed)


# --- rivers ------------------------------------------------------------------

def drainage(cost, mouth):
    """Grow a river network back from its mouth: every cell drains along its
    cheapest route there (Dijkstra over the cells, with knight's moves too so
    streams aren't held to eight directions). Returns each cell's route cost,
    the next cell downstream (-1 at the mouth), and the cells in the order
    they were reached."""
    h, w = cost.shape
    c = cost.ravel().tolist()
    dist, down, done, order = [math.inf] * (h * w), [-1] * (h * w), bytearray(h * w), []
    moves = [(dy, dx, math.hypot(dy, dx)) for dy in range(-2, 3) for dx in range(-2, 3)
             if (dy or dx) and math.gcd(dy, dx) == 1]
    dist[mouth] = 0.0
    heap = [(0.0, mouth)]
    while heap:
        di, i = heapq.heappop(heap)
        if done[i]:
            continue
        done[i] = 1
        order.append(i)
        y, x = divmod(i, w)
        ci = c[i]
        for dy, dx, step in moves:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w:
                j = ny * w + nx
                if not done[j]:
                    dj = di + (ci + c[j]) * 0.5 * step
                    if dj < dist[j]:
                        dist[j], down[j] = dj, i
                        heapq.heappush(heap, (dj, j))
    return np.array(dist).reshape(h, w), down, order


def render_rivers(w, h, seed):
    """One river basin: every stream runs down the cheapest way to a single
    mouth at the bottom edge, through the low ground of a noise field, and a
    stream shows once enough cells drain into it. Width grows with the area
    drained; the main river is blue."""
    rng = np.random.default_rng(seed + 2)
    s = h / 2160
    q = 5                                                # terrain cell, px
    gw, gh = w // q, h // q
    cost = (0.05 + noise(gw, gh, rng, 0.3 * gh, 6, 0.55) ** 3 + 0.3 * noise(gw, gh, rng, 0.05 * gh, 3, 0.5) ** 2
            + 0.08 * rng.random((gh, gw)))
    mouth = (gh - 1) * gw + gw // 2
    dist, down, order = drainage(cost, mouth)
    acc = [1.0] * len(down)
    for i in reversed(order):
        if down[i] >= 0:
            acc[down[i]] += acc[i]
    acc = np.array(acc)

    # The land rises with the cost of the way down; it only places the summit.
    small = Image.fromarray(dist.astype(np.float32), "F").resize((gw // 8, gh // 8), Image.BOX)
    height = np.asarray(small.resize((w, h), Image.BICUBIC), np.float32)
    plate = Plate(ground(w, h))

    threshold = 140
    river = acc >= threshold
    top = math.log(acc.max() / threshold)

    def look(a):
        t = np.log(np.maximum(a, threshold) / threshold) / top
        return s * (0.7 + 3.6 * t ** 1.3), [hexmix("#3a3a3e", "#66666b", round(v * 6) / 6) for v in t]

    def point(i):
        y, x = divmod(i, gw)
        return ((x + 0.5) * q, (y + 0.5) * q)

    def draw(path, color=None):
        pts = chaikin([point(i) for i in path], 3)
        a = np.interp(np.linspace(0, len(path) - 1, len(pts)), np.arange(len(path)), acc[path])
        widths, colors = look(a)
        plate.stroke(pts, widths, [color] * len(pts) if color else colors)

    # Trace every stream from its source down to where it joins another.
    upstream = np.zeros(len(down), np.int64)
    for i in np.flatnonzero(river):
        if down[i] >= 0:
            upstream[down[i]] += 1
    seen = bytearray(len(down))
    for src in np.flatnonzero(river & (upstream == 0)):
        path, i = [src], src
        seen[i] = 1
        while down[i] >= 0:
            i = down[i]
            path.append(i)
            if seen[i]:
                break
            seen[i] = 1
        draw(path)

    # The main river: from the mouth, always up the bigger branch.
    ups = {}
    for i in np.flatnonzero(river):
        if down[i] >= 0:
            ups.setdefault(down[i], []).append(i)
    i, trunk = mouth, [mouth]
    while i in ups:
        i = max(ups[i], key=lambda k: acc[k])
        trunk.append(i)
    draw(trunk[::-1], BLUE)

    xx, yy = coords(w, h)
    inside = (np.abs(xx - w / 2) < 0.42 * w) & (yy > 0.1 * h) & (yy < 0.9 * h)
    sy, sx = np.unravel_index(np.argmax(np.where(inside, height, 0)), height.shape)
    plate.summit(sx, sy, s)
    plate.label(sx + 16 * s, sy + 7 * s, "1630", s)
    return plate.finish(seed)


# --- globe -------------------------------------------------------------------

def unit(lat, lon):
    la, lo = np.radians(lat), np.radians(lon)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], -1)


def render_globe(w, h, seed):
    """A wireframe globe: a 15° graticule, the land as a field of dots, and a
    flight from Singapore to San Francisco flown along its great circle, a
    little above the surface."""
    s = h / 2160
    cx, cy, radius = w / 2, h / 2, 0.40 * h
    sin_city, sfo = unit(1.36, 103.99), unit(37.62, -122.38)
    # Face the middle of the route.
    # Look at the route from 22° south of its middle, so the great circle bends
    # (seen face on through its middle it would be a straight line).
    mid = (sin_city + sfo) / np.linalg.norm(sin_city + sfo)
    lat0, lon0 = math.degrees(math.asin(mid[2])) - 22, math.degrees(math.atan2(mid[1], mid[0]))
    view = unit(lat0, lon0)
    east = np.array([-math.sin(math.radians(lon0)), math.cos(math.radians(lon0)), 0])
    north = np.cross(view, east)

    def project(p):
        """3-D points -> screen x, y and depth (> 0 faces us)."""
        return cx + radius * (p @ east), cy - radius * (p @ north), p @ view

    plate = Plate(ground(w, h))

    def curve(lat, lon, color, width):
        x, y, z = project(unit(lat, lon))
        run = []
        for px, py, pz in zip(x, y, z):
            if pz > 0:
                run.append((px, py))
            elif run:
                plate.line(run, color, width)
                run = []
        plate.line(run, color, width)

    fine = np.arange(-180, 180.5, 0.5)
    for lat in range(-75, 90, 15):
        curve(np.full_like(fine, lat), fine, MAJOR if lat == 0 else MINOR, (1.3 if lat == 0 else 0.9) * s)
    meridian = np.arange(-90, 90.5, 0.5)
    for lon in range(-180, 180, 15):
        curve(meridian, np.full_like(meridian, lon), MINOR, 0.9 * s)

    # Land: dots evenly spread over the sphere, kept where the map says land.
    land = np.asarray(Image.open(LAND))
    step = 0.85
    for lat in np.arange(-85, 85, step):
        lon = np.arange(-180, 180, step / max(0.05, math.cos(math.radians(lat))))
        row = int((90 - lat) / 180 * land.shape[0])
        on = land[row, ((lon + 180) / 360 * land.shape[1]).astype(int) % land.shape[1]]
        x, y, z = project(unit(np.full(on.sum(), lat), lon[on]))
        for px, py, pz in zip(x, y, z):
            if pz > 0.02:
                plate.circle(px, py, (0.9 + 1.4 * pz ** 0.5) * s, hexmix("#38383b", "#5c5c60", pz ** 0.6))

    plate.circle(cx, cy, radius, outline=MAJOR, width=1.5 * s)

    # The flight, on its great circle: its ground track dashed, the flight
    # itself raised off the surface at mid-route.
    angle = math.acos(float(sin_city @ sfo))
    t = np.linspace(0, 1, 400)
    path = (np.sin((1 - t) * angle)[:, None] * sin_city + np.sin(t * angle)[:, None] * sfo) / math.sin(angle)
    gx, gy, _ = project(path)
    plate.dashed(list(zip(gx, gy)), MAJOR, 1.1 * s, 10 * s, 9 * s)
    ax, ay, _ = project(path * (1 + 0.11 * np.sin(np.pi * t))[:, None])
    plate.line(list(zip(ax, ay)), ORANGE, 1.8 * s)
    for (px, py), name, side in (((gx[0], gy[0]), "SIN", -1), ((gx[-1], gy[-1]), "SFO", 1)):
        plate.circle(px, py, 7 * s, outline=ORANGE, width=1.6 * s)
        plate.circle(px, py, 2.4 * s, MARK)
        plate.label(px + side * 16 * s, py + 30 * s, name, s, anchor="ls" if side > 0 else "rs")
    k = len(t) // 2
    nx, ny = ay[k + 1] - ay[k - 1], ax[k - 1] - ax[k + 1]
    if ny > 0:
        nx, ny = -nx, -ny
    norm = math.hypot(nx, ny)
    plate.label(ax[k] + nx / norm * 44 * s, ay[k] + ny / norm * 44 * s, "13 590 km", s, anchor="mm")
    return plate.finish(seed)


# --- circuit -----------------------------------------------------------------

def render_circuit(w, h, seed):
    """A circuit board: three chips, small two-pad parts, and traces fanning
    out of the chips' pins at 0, 45 and 90 degrees to end in vias. The
    longest trace in green."""
    rng = np.random.default_rng(seed + 4)
    s = h / 2160
    g = 26 * s                                           # routing grid, px
    cols, rows = int(w / g), int(h / g)
    free = np.ones((rows, cols), bool)
    margin = 3
    free[:margin], free[-margin:], free[:, :margin], free[:, -margin:] = False, False, False, False

    def at(c, r):
        return ((c + 0.5) * g, (r + 0.5) * g)

    plate = Plate(ground(w, h))
    chips = [("U1", 0.30, 0.38, 20, 20), ("U2", 0.64, 0.30, 14, 8), ("U3", 0.60, 0.66, 9, 9)]
    pins = []
    for name, fx, fy, cw, ch in chips:
        c0, r0 = int(cols * fx - cw / 2), int(rows * fy - ch / 2)
        free[r0 - 1:r0 + ch + 1, c0 - 1:c0 + cw + 1] = False
        x0, y0 = c0 * g, r0 * g
        x1, y1 = (c0 + cw) * g, (r0 + ch) * g
        plate.rect(x0, y0, x1, y1, outline=MAJOR, width=1.6 * s)
        plate.circle(x0 + 1.2 * g, y0 + 1.2 * g, 0.3 * g, outline=MAJOR, width=1.2 * s)
        plate.label((x0 + x1) / 2, (y0 + y1) / 2 + 9 * s, name, s, anchor="ms")
        sides = [(range(c0 + 1, c0 + cw - 1), lambda c: (c, r0 - 1), (0, -1)),
                 (range(c0 + 1, c0 + cw - 1), lambda c: (c, r0 + ch), (0, 1)),
                 (range(r0 + 1, r0 + ch - 1), lambda r: (c0 - 1, r), (-1, 0)),
                 (range(r0 + 1, r0 + ch - 1), lambda r: (c0 + cw, r), (1, 0))]
        for span, cell, out in sides:
            for k in span:
                c, r = cell(k)
                px, py = at(c, r)
                plate.rect(px - 0.22 * g, py - 0.22 * g, px + 0.22 * g, py + 0.22 * g, fill=MAJOR)
                if rng.random() < 0.8:
                    pins.append((c, r, out))

    # Small two-pad parts dotted over the board, placed before the traces so
    # they route around them.
    for _ in range(40):
        c, r = int(rng.integers(margin + 2, cols - margin - 4)), int(rng.integers(margin + 2, rows - margin - 4))
        across = rng.random() < 0.5
        cw, ch = (3, 1) if across else (1, 3)
        if not free[r - 1:r + ch + 1, c - 1:c + cw + 1].all():
            continue
        free[r - 1:r + ch + 1, c - 1:c + cw + 1] = False
        (ax, ay), (bx, by) = at(c, r), at(c + cw - 1, r + ch - 1)
        plate.rect(ax - 0.4 * g, ay - 0.4 * g, bx + 0.4 * g, by + 0.4 * g, outline=MINOR, width=1.0 * s)
        for px, py in ((ax, ay), (bx, by)):
            plate.rect(px - 0.28 * g, py - 0.28 * g, px + 0.28 * g, py + 0.28 * g, fill=MAJOR)

    directions = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    taken = {}

    def can(c, r, dc, dr):
        nc, nr = c + dc, r + dr
        if not (0 <= nr < rows and 0 <= nc < cols and free[nr, nc]):
            return False
        # Don't cross another trace's diagonal.
        return not (dc and dr and taken.get((c + dc, r)) is not None and taken.get((c + dc, r)) == taken.get((c, r + dr)))

    traces = []
    rng.shuffle(pins)
    for n, (c, r, out) in enumerate(pins):          # pin cells are kept from other traces
        taken[(c, r)] = n
        heading = directions.index(out)
        path = [(c, r)]
        straight = int(rng.integers(1, 4))
        for step in range(int(rng.integers(10, 80))):
            options = [heading] if step < straight else [heading, (heading + 1) % 8, (heading - 1) % 8]
            if step >= straight and rng.random() < 0.1:
                options = options[1:] + options[:1]
            for hd in options:
                dc, dr = directions[hd]
                if can(c, r, dc, dr) and abs(((hd - directions.index(out)) + 4) % 8 - 4) <= 2:
                    heading = hd
                    c, r = c + dc, r + dr
                    free[r, c] = False
                    taken[(c, r)] = n
                    path.append((c, r))
                    break
            else:
                break
        if len(path) > 3:
            traces.append(path)

    longest = max(traces, key=len)
    for path in traces:
        accent = path is longest
        color = GREEN if accent else "#3e3e42"
        pts = [at(c, r) for c, r in path]
        plate.line(pts, color, 2.2 * s)
        ex, ey = pts[-1]
        plate.circle(ex, ey, 0.34 * g, GROUND, outline=color, width=1.6 * s)
    for fx, fy in ((0.12, 0.14), (0.88, 0.86)):
        plate.circle(w * fx, h * fy, 1.3 * g, outline=MAJOR, width=1.4 * s)
        plate.circle(w * fx, h * fy, 0.8 * g, outline=MINOR, width=1.0 * s)
    return plate.finish(seed)


# --- lunar -------------------------------------------------------------------

def render_lunar(w, h, seed):
    """A crater field as a contour map: each crater a bowl inside a raised
    rim, larger ones deeper, over gently rolling ground. One contour round a
    big crater's inner wall in orange, its name and size beside it."""
    rng = np.random.default_rng(seed + 5)
    s = h / 2160
    xx, yy = coords(w, h)
    height = 0.05 * noise(w, h, rng, 0.35 * h, 5, 0.5)
    marked = (0.61 * w, 0.43 * h, 0.16 * h)
    craters = [(0.30 * w, 0.64 * h, 0.11 * h), (0.83 * w, 0.76 * h, 0.075 * h), (0.18 * w, 0.24 * h, 0.06 * h)]
    for _ in range(260):
        r = min(0.05 * h, 9 * s * (1 - rng.random()) ** (-1 / 1.5))
        craters.append((rng.uniform(-0.05, 1.05) * w, rng.uniform(-0.05, 1.05) * h, r))
    craters.sort(key=lambda c: -c[2])                    # big and old first, small ones on top
    craters.insert(0, marked)
    for x, y, r in craters:
        reach = 2.8 * r
        x0, x1 = int(max(0, x - reach)), int(min(w, x + reach + 1))
        y0, y1 = int(max(0, y - reach)), int(min(h, y + reach + 1))
        if x0 >= x1 or y0 >= y1:
            continue
        d = np.hypot(xx[y0:y1, x0:x1] - x, yy[y0:y1, x0:x1] - y) / r
        rim = 0.24 * np.exp(-((d - 1) / 0.14) ** 2)
        profile = np.where(d < 1, 0.62 * (d * d - 1), 0.07 * np.exp(-(d - 1) * 2.4)) + rim
        height[y0:y1, x0:x1] += (r / h) ** 0.8 * profile
    height = normalize(height)
    arr = ground(w, h)
    relief(arr, height)
    levels = 46
    mx, my, mr = marked
    theta = np.linspace(0, 2 * np.pi, 360, endpoint=False)
    # A level the ground around the crater stays above, so it closes once,
    # on the inner wall.
    around = height[(my + 1.35 * mr * np.sin(theta)).astype(int), (mx + 1.35 * mr * np.cos(theta)).astype(int)]
    near = np.hypot(xx - mx, yy - my) < 1.12 * mr
    contour_lines(arr, height, levels, s, accent=ORANGE, accent_level=math.floor(around.min() * levels), where=near)
    plate = Plate(arr)
    plate.label(mx + 0.86 * mr, my - 0.9 * mr, "C-17  Ø 42 km", s, anchor="ls")
    return plate.finish(seed)


# --- nautical ----------------------------------------------------------------

DEPTHS = [0, 2, 5, 10, 20, 30, 50, 100, 200]             # contours drawn, m


def render_nautical(w, h, seed):
    """A stretch of coast as a sea chart: land stippled, depth contours, the
    soundings in metres scattered over the water, a compass rose. The 20 m
    line in blue."""
    rng = np.random.default_rng(seed + 6)
    s = h / 2160
    xx, yy = coords(w, h)
    e = noise(w, h, rng, 0.5 * h, 6, 0.52) + 0.75 * (1 - (0.6 * xx / w + 0.4 * yy / h))  # land to the north-west
    coast = np.quantile(e, 0.68)
    sea = e < coast
    depth = 200 * ((coast - e) / (coast - e.min())).clip(0) ** 1.7
    # Contour index: whole numbers fall on the depths above; land below zero.
    g = np.where(sea, np.interp(depth, DEPTHS, np.arange(len(DEPTHS))), (coast - e) / (e.max() - coast) * 3)
    arr = ground(w, h)
    contour_lines(arr, g, 1, s, every=len(DEPTHS) + 1, accent=BLUE, accent_level=DEPTHS.index(20),
                  mask=(g > -0.5).astype(np.float32))
    contour_lines(arr, np.where(sea, -1.0, 1.0) * np.minimum(np.abs(e - coast), 0.004) / 0.004 * 0.5, 1, s,
                  every=1, mask=np.abs(e - coast) < 0.01)  # the coastline, stronger
    plate = Plate(arr)

    def wet(x, y):
        return sea[int(y), int(x)]

    for x, y in scatter(w, h, 13 * s, rng, lambda x, y: not wet(x, y)):
        plate.circle(x, y, 1.6 * s, "#434346")

    # The compass rose sits in deep water to the south-east.
    window = (xx > 0.62 * w) & (xx < 0.82 * w) & (yy > 0.58 * h) & (yy < 0.78 * h)
    ry, rx = np.unravel_index(np.argmax(np.where(window, depth, -1)), depth.shape)
    rr = 170 * s
    plate.circle(rx, ry, rr, outline=MAJOR, width=1.4 * s)
    plate.circle(rx, ry, 0.86 * rr, outline=MINOR, width=1.0 * s)
    for deg in range(0, 360, 2):
        a = math.radians(deg)
        inner = 0.86 if deg % 10 else 0.80
        ux, uy = math.sin(a), -math.cos(a)
        plate.line([(rx + ux * rr * inner, ry + uy * rr * inner), (rx + ux * rr, ry + uy * rr)],
                   MAJOR if deg % 10 == 0 else MINOR, (1.2 if deg % 10 == 0 else 0.9) * s)
        if deg % 30 == 0:
            plate.label(rx + ux * rr * 0.68, ry + uy * rr * 0.68 + 7 * s, str(deg), s, size=18, anchor="ms")
    plate.line([(rx, ry - 0.5 * rr), (rx, ry + 0.5 * rr)], MAJOR, 1.0 * s)
    plate.line([(rx - 0.5 * rr, ry), (rx + 0.5 * rr, ry)], MAJOR, 1.0 * s)
    plate.polygon([(rx, ry - 0.5 * rr), (rx - 9 * s, ry - 0.5 * rr + 26 * s), (rx + 9 * s, ry - 0.5 * rr + 26 * s)], MARK)

    def open_water(x, y):
        return wet(x, y) and depth[int(y), int(x)] > 1.5 and math.hypot(x - rx, y - ry) > 1.25 * rr

    for x, y in scatter(w, h, 150 * s, rng, open_water):
        plate.label(x, y, str(int(depth[int(y), int(x)])), s, color="#5e5e62", size=19, anchor="mm")
    plate.label(0.06 * w, 0.93 * h, "DEPTHS IN METRES", s, size=20)
    return plate.finish(seed)


# --- stars -------------------------------------------------------------------

def render_stars(w, h, seed):
    """A star chart: right ascension and declination as curved lines, the
    ecliptic dashed, stars sized by brightness, and six made-up
    constellations joined the way chart makers do, the lines stopping short
    of the stars. One constellation in lavender."""
    rng = np.random.default_rng(seed + 7)
    s = h / 2160
    ra0, dec0 = math.radians(90), math.radians(20)
    scale = (w / 2) / (2 * math.tan(math.radians(56) / 2))

    def project(ra, dec):
        """Stereographic, east to the left like the sky seen from below."""
        ra, dec = np.radians(ra), np.radians(dec)
        cos_c = math.sin(dec0) * np.sin(dec) + math.cos(dec0) * np.cos(dec) * np.cos(ra - ra0)
        k = 2 / np.maximum(1 + cos_c, 1e-6)
        x = k * np.cos(dec) * np.sin(ra - ra0)
        y = k * (math.cos(dec0) * np.sin(dec) - math.sin(dec0) * np.cos(dec) * np.cos(ra - ra0))
        return w / 2 - scale * x, h / 2 - scale * y, cos_c

    plate = Plate(ground(w, h))
    pad = 80 * s

    def curve(ra, dec, color, width, dashed=False):
        x, y, c = project(ra, dec)
        keep = (c > 0) & (x > -pad) & (x < w + pad) & (y > -pad) & (y < h + pad)
        runs, run = [], []
        for px, py, k in zip(x, y, keep):
            if k:
                run.append((px, py))
            elif run:
                runs.append(run)
                run = []
        runs.append(run)
        for run in runs:
            if len(run) > 1:
                if dashed:
                    plate.dashed(run, color, width, 16 * s, 12 * s)
                else:
                    plate.line(run, color, width)
        return x, y

    fine = np.arange(0, 360.5, 0.5)
    for dec in range(-30, 90, 10):
        x, y = curve(fine, np.full_like(fine, dec), MAJOR if dec == 0 else "#2e2e31", (1.2 if dec == 0 else 0.9) * s)
        near = np.flatnonzero((np.abs(x - 70 * s) < 6 * s) & (y > 0.08 * h) & (y < 0.92 * h))
        if len(near):
            plate.label(70 * s, y[near[0]] - 10 * s, f"{dec:+d}°" if dec else "0°", s, size=20)
    pole = np.arange(-80, 80.5, 0.5)
    for hour in range(24):
        x, y = curve(np.full_like(pole, hour * 15), pole, "#2e2e31", 0.9 * s)
        near = np.flatnonzero((np.abs(y - (h - 64 * s)) < 6 * s) & (x > 0.06 * w) & (x < 0.94 * w))
        if len(near):
            plate.label(x[near[0]] + 10 * s, h - 64 * s, f"{hour:02d}h", s, size=20)
    # The ecliptic, 23.44° to the equator.
    lam, eps = np.radians(np.arange(0, 360.5, 0.5)), math.radians(23.44)
    curve(np.degrees(np.arctan2(math.cos(eps) * np.sin(lam), np.cos(lam))) % 360,
          np.degrees(np.arcsin(math.sin(eps) * np.sin(lam))), MAJOR, 1.0 * s, dashed=True)

    def star(x, y, b):
        r = (0.8 + 4.2 * b ** 1.4) * s
        plate.circle(x, y, r, hexmix("#3e3e42", "#dcdcdc", b ** 0.6))
        return r

    for _ in range(1700):
        star(rng.uniform(0, w), rng.uniform(0, h), rng.random() ** 5)

    centres = scatter(w, h, 620 * s, rng, lambda x, y: 0.12 * w < x < 0.88 * w and 0.16 * h < y < 0.84 * h)[:6]
    accent = min(range(len(centres)), key=lambda i: math.hypot(centres[i][0] - 0.62 * w, centres[i][1] - 0.42 * h))
    for n, (cx, cy) in enumerate(centres):
        k = int(rng.integers(5, 9))
        pts = [(cx + rng.normal(0, 160 * s), cy + rng.normal(0, 130 * s)) for _ in range(k)]
        bright = rng.uniform(0.35, 1.0, k)
        radii = [star(x, y, b) for (x, y), b in zip(pts, bright)]
        # Join them by their minimum spanning tree.
        inside, edges = {0}, []
        while len(inside) < k:
            i, j = min(((i, j) for i in inside for j in range(k) if j not in inside),
                       key=lambda e: math.dist(pts[e[0]], pts[e[1]]))
            inside.add(j)
            edges.append((i, j))
        for i, j in edges:
            (ax, ay), (bx, by) = pts[i], pts[j]
            length = math.dist(pts[i], pts[j])
            gap_a, gap_b = radii[i] + 12 * s, radii[j] + 12 * s
            if length > gap_a + gap_b + 6 * s:
                ux, uy = (bx - ax) / length, (by - ay) / length
                plate.line([(ax + ux * gap_a, ay + uy * gap_a), (bx - ux * gap_b, by - uy * gap_b)],
                           LAVENDER if n == accent else MAJOR, 1.3 * s)
        if n == accent:
            b = int(np.argmax(bright))
            plate.label(pts[b][0] + radii[b] + 10 * s, pts[b][1] - radii[b] - 4 * s, "α", s, size=28)
    return plate.finish(seed)


# --- rings -------------------------------------------------------------------

def render_rings(w, h, seed):
    """The end of a sawn log: growth rings around an off-centre pith, wide
    when the tree was young, the bark around them, and a few cracks where the
    wood dried. One ring in green."""
    rng = np.random.default_rng(seed + 8)
    s = h / 2160
    xx, yy = coords(w, h)
    px, py = 0.52 * w, 0.52 * h
    theta = np.arctan2(yy - py, xx - px)
    phases = rng.uniform(0, 2 * np.pi, 4)

    def outline(t):
        """The log's radius from the pith at angle t."""
        return 0.40 * h * (1 + 0.09 * np.cos(t - 2.4)) * (1 + 0.03 * np.sin(3 * t + phases[0])
                                                          + 0.018 * np.sin(5 * t + phases[1]) + 0.01 * np.sin(9 * t + phases[2]))

    rho = np.hypot(xx - px, yy - py) / outline(theta)
    rho = rho + 0.006 * (noise(w, h, rng, 0.05 * h, 3, 0.5) - 0.5)
    years = 74
    widths = rng.gamma(5, 1, years) * np.exp(-np.arange(years) / years * 1.1)
    bounds = np.concatenate([[0], np.cumsum(widths)])
    bounds /= bounds[-1]
    ring = np.interp(rho, bounds, np.arange(years + 1))
    arr = ground(w, h)
    contour_lines(arr, ring, 1, s, every=10, accent=GREEN, accent_level=42,
                  mask=(1 - smoothstep(0.985, 1.0, rho)).astype(np.float32))
    plate = Plate(arr)

    t = np.linspace(0, 2 * np.pi, 1440)
    r_in, r_out = outline(t), outline(t) * (1.05 + 0.012 * np.sin(17 * t + phases[3]))
    plate.line(list(zip(px + r_in * np.cos(t), py + r_in * np.sin(t))), MAJOR, 1.6 * s)
    plate.line(list(zip(px + r_out * np.cos(t), py + r_out * np.sin(t))), MAJOR, 1.4 * s)
    for a in rng.uniform(0, 2 * np.pi, 260):              # the bark's furrows
        r0 = float(outline(a)) * 1.008
        r1 = r0 * rng.uniform(1.012, 1.04)
        plate.line([(px + r0 * math.cos(a), py + r0 * math.sin(a)), (px + r1 * math.cos(a), py + r1 * math.sin(a))],
                   MINOR, 1.0 * s)
    for a in rng.uniform(0, 2 * np.pi) + np.arange(3) * 2.1 + rng.uniform(-0.4, 0.4, 3):  # checks, opening toward the bark
        r_edge = float(outline(a)) * 1.002
        depth = rng.uniform(0.35, 0.65)
        steps = np.linspace(1, depth, 24)
        wobble = np.cumsum(rng.normal(0, 0.012, 24))
        half = 9 * s * (steps - depth) / (1 - depth) + 0.4 * s
        left, right = [], []
        for f, dw, hw in zip(steps, wobble, half):
            ang = a + dw
            x, y = px + r_edge * f * math.cos(ang), py + r_edge * f * math.sin(ang)
            nx, ny = -math.sin(ang), math.cos(ang)
            left.append((x + nx * hw, y + ny * hw))
            right.append((x - nx * hw, y - ny * hw))
        plate.polygon(left + right[::-1], "#131315")
    plate.circle(px, py, 2.5 * s, MARK)
    # Name the green ring where it crosses the horizontal to the right.
    xs = np.arange(int(px), w)
    k = int(np.argmin(np.abs(ring[int(py), xs] - 42)))
    plate.label(xs[k] + 12 * s, py - 12 * s, "42", s)
    return plate.finish(seed)


# --- main --------------------------------------------------------------------

RENDERERS = {
    "topography": ("1-topography.jpg", render_topography),
    "ridgelines": ("2-ridgelines.jpg", render_ridgelines),
    "rivers": ("3-rivers.jpg", render_rivers),
    "globe": ("4-globe.jpg", render_globe),
    "circuit": ("5-circuit.jpg", render_circuit),
    "lunar": ("6-lunar.jpg", render_lunar),
    "nautical": ("7-nautical.jpg", render_nautical),
    "stars": ("8-stars.jpg", render_stars),
    "rings": ("9-rings.jpg", render_rings),
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
        img.save(path, quality=92, subsampling=0, optimize=True)
        print(f"Wrote {path} ({args.width}x{args.height}, {os.path.getsize(path) // 1024} KB)")


if __name__ == "__main__":
    sys.exit(main())
