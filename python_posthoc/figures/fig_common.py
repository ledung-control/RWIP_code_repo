# -*- coding: utf-8 -*-
"""Shared drawing helpers for the IJCAS single-column block diagrams.

Geometry contract
-----------------
The figure is authored at its FINAL published size: 8.0 cm wide.  The axes fill
the whole canvas and use equal aspect with x in [0, 100], so

        1 drawing unit = 8.0 cm / 100 = 0.8 mm = 2.2677 pt

Every text object is set in points, so a label declared at 10 pt is exactly
10 pt once the figure is placed at 8.0 cm in the manuscript - no Word scaling.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
from matplotlib.patches import FancyArrowPatch

CM = 8.0                      # final published width, centimetres
W_IN = CM / 2.54              # 3.1496 in
UNIT_PT = 72.0 * W_IN / 100.0 # points per drawing unit  (2.2677)
FS = 10.0                     # every label: exactly 10 pt at final size
LINE_UNITS = 12.2 / UNIT_PT   # 10 pt text on ~12.2 pt leading, in units

# palette taken from the existing figures
C_BLUE_E, C_BLUE_F = "#1F3864", "#DCE3F0"
C_ORNG_E, C_ORNG_F = "#C05A16", "#FDF0E2"
C_RED_E,  C_RED_F  = "#B41F2E", "#FBE0E4"
C_GOLD_E, C_GOLD_F = "#B8860B", "#FBEFD0"
C_GREY_E, C_GREY_F = "#595959", "#F0F0F0"

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": FS,
    "text.antialiased": True,
})


def new_canvas(height_units):
    """axes with 1 unit = 0.8 mm, canvas exactly 8.0 cm wide"""
    h_in = W_IN * height_units / 100.0
    fig = plt.figure(figsize=(W_IN, h_in))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 100)
    ax.set_ylim(0, height_units)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def box(ax, x, y, w, lines, ec, fc, pad=2.6, lead=None, weights=None,
        fs=FS, radius=1.6, heights=None):
    """centred multi-line box whose top edge is at y; returns its bottom y.

    `lines` is a list of strings (mathtext allowed); `weights` an optional
    list of 'normal'/'bold' of the same length."""
    lead = lead or LINE_UNITS
    hs = [(heights[i] if heights else 1.0) * lead for i in range(len(lines))]
    h = sum(hs) + 2 * pad
    ax.add_patch(FancyBboxPatch(
        (x, y - h), w, h,
        boxstyle="round,pad=0,rounding_size=%.3f" % radius,
        linewidth=1.1, edgecolor=ec, facecolor=fc, zorder=2))
    cur = y - pad
    for i, s in enumerate(lines):
        wt = (weights[i] if weights else "normal")
        ax.text(x + w / 2.0, cur - hs[i] / 2.0, s, ha="center", va="center",
                fontsize=fs, color="black", zorder=3, fontweight=wt)
        cur -= hs[i]
    return y - h


def arrow(ax, p0, p1, color, lw=1.1, ms=9):
    ax.add_patch(FancyArrowPatch(
        p0, p1, arrowstyle="-|>", mutation_scale=ms, linewidth=lw,
        color=color, shrinkA=0, shrinkB=0, zorder=4,
        joinstyle="miter", capstyle="butt"))


def poly_arrow(ax, pts, color, lw=1.1, ms=9):
    """orthogonal poly-line with an arrow head on the final segment"""
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    ax.plot(xs[:-1], ys[:-1], color=color, lw=lw, solid_joinstyle="miter",
            solid_capstyle="butt", zorder=4)
    arrow(ax, pts[-2], pts[-1], color, lw=lw, ms=ms)


def label(ax, x, y, s, color="black", ha="left", va="center", fs=FS):
    ax.text(x, y, s, color=color, ha=ha, va=va, fontsize=fs, zorder=5)


def summing_junction(ax, x, y, r=2.4):
    ax.add_patch(Circle((x, y), r, facecolor="white", edgecolor="black",
                        linewidth=1.1, zorder=5))
    ax.text(x, y, "+", ha="center", va="center", fontsize=FS, zorder=6)


def save(fig, stem):
    """vector master + 1200 dpi raster fallback, both at exactly 8.0 cm wide"""
    fig.savefig(stem + ".pdf", facecolor="white")          # vector master
    fig.savefig(stem + ".svg", facecolor="white")
    fig.savefig(stem + "_1250dpi.png", dpi=1250, facecolor="white")
    plt.close(fig)


def audit_fit(fig, ax, boxes):
    """report any text that overflows the box it belongs to"""
    r = fig.canvas.get_renderer()
    bad = []
    for t in ax.texts:
        bb = t.get_window_extent(renderer=r)
        bb = bb.transformed(ax.transData.inverted())
        for (bx, by0, bw, bh) in boxes:
            cx = (bb.x0 + bb.x1) / 2
            if by0 <= (bb.y0 + bb.y1) / 2 <= by0 + bh and bx <= cx <= bx + bw:
                if bb.x0 < bx + 0.6 or bb.x1 > bx + bw - 0.6:
                    bad.append((t.get_text()[:52], round(bb.x0, 1),
                                round(bb.x1, 1), bx, bx + bw))
                break
    return bad
