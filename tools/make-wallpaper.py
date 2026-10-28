#!/usr/bin/env python3
"""Render the Min Dark code wallpaper.

A few lines of TypeScript as VS Code shows them in Min Dark: line numbers, an
indent guide, the current line outlined in Min's #303030 (here across the whole
screen) and the cursor at its end. Token colors follow min-dark.json.

Usage: tools/make-wallpaper.py [output.png] [--width 3840 --height 2160]
"""

import argparse
import os
import re
import sys

from PIL import Image, ImageDraw, ImageFont

THEME_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FONT = "/usr/share/fonts/TTF/JetBrainsMonoNerdFont-Regular.ttf"


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output", nargs="?", default=os.path.join(THEME_DIR, "backgrounds", "1-min-code.png"))
    ap.add_argument("--width", type=int, default=3840)
    ap.add_argument("--height", type=int, default=2160)
    args = ap.parse_args()

    w, h = args.width, args.height
    scale = h / 2160
    size = round(34 * scale)
    font = ImageFont.truetype(FONT, size)
    cell = font.getlength("M")
    line_h = round(size * 1.7)

    img = Image.new("RGB", (w, h), C["background"])
    draw = ImageDraw.Draw(img)

    gutter = 4 * cell          # right-aligned line numbers
    gap = 3 * cell
    widest = max(sum(len(t) for t, _ in line) for line in CODE)
    block_w = gutter + gap + widest * cell
    x0 = round((w - block_w) / 2)
    y0 = round((h - len(CODE) * line_h) / 2)
    code_x = x0 + gutter + gap

    def text_y(i):
        # Baseline-ish placement: centre the glyphs in the line box.
        ascent, descent = font.getmetrics()
        return y0 + i * line_h + (line_h - ascent - descent) / 2

    # The current line's border runs the full width, as in an editor that
    # fills the screen.
    border = max(1, round(2 * scale))
    top = y0 + ACTIVE * line_h
    draw.rectangle([-border, top, w + border, top + line_h - 1], outline=LINE_BORDER, width=border)

    for i, line in enumerate(CODE):
        y = text_y(i)
        number = str(i + 1)
        draw.text((x0 + gutter - len(number) * cell, y), number, font=font,
                  fill=LINE_NUMBER_ACTIVE if i == ACTIVE else LINE_NUMBER)
        if i in INDENTED:
            gx = round(code_x)
            draw.line([(gx, y0 + i * line_h), (gx, y0 + (i + 1) * line_h - 1)], fill=INDENT_GUIDE, width=max(1, round(scale)))
        x = code_x
        for text, color in line:
            draw.text((x, y), text, font=font, fill=color)
            x += len(text) * cell
        if i == ACTIVE:
            cursor_w = max(2, round(3 * scale))
            draw.rectangle([round(x) + 1, top + round(line_h * 0.18), round(x) + cursor_w, top + round(line_h * 0.82)],
                           fill=CURSOR)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    img.save(args.output, optimize=True)
    print(f"Wrote {args.output} ({w}x{h})")


if __name__ == "__main__":
    sys.exit(main())
