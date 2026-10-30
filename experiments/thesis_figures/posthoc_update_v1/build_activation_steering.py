#!/usr/bin/env python3
"""Reproducible vector schematic of activation steering (not measured geometry).

Run: python build_activation_steering.py --output-dir Figures
Matplotlib and NumPy are the only dependencies. No model or data is loaded.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle


BLUE = "#2F6595"
COPPER = "#B47B48"
EDIT = "#A2547A"
INK = "#283039"
GREY = "#7C858C"


def arrow(ax, start, end, colour=GREY, width=1.05, head=10, zorder=4):
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=head,
        linewidth=width,
        color=colour,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def block(ax, x, y, w, h, label, fill="#F7F8F9", edge="#AAB0B5", size=11.5):
    ax.add_patch(
        Rectangle((x, y), w, h, facecolor=fill, edgecolor=edge, linewidth=0.85)
    )
    ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=size)


def build(out: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 11,
            "mathtext.fontset": "stix",
            "text.color": INK,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    fig = plt.figure(figsize=(11.65, 6.25), facecolor="white")
    ax = fig.add_axes([0.035, 0.08, 0.43, 0.83])
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.set_axis_off()

    fig.text(0.043, 0.952, "(a)  Edit during the forward pass", fontsize=13.3,
             weight="bold", family="DejaVu Sans", va="top")
    fig.text(0.528, 0.952, "(b)  Vector addition", fontsize=13.3,
             weight="bold", family="DejaVu Sans", va="top")
    fig.add_artist(plt.Line2D([0.494, 0.494], [0.13, 0.91],
                            transform=fig.transFigure, color="#E1E5E8", lw=0.8))

    # Panel (a): the added vector changes an activation, not the learned weights.
    x, w, centre = 0.055, 0.455, 0.2825
    block(ax, x, 0.832, w, 0.106, "Earlier model layers")
    block(ax, x, 0.637, w, 0.126, "Hidden state  $h$", fill="#E2EDF6", edge="#9DAFBD")
    block(ax, x, 0.265, w, 0.126, "Edited state  $h'$", fill="#F7E7D7", edge="#BCAC9C")
    block(ax, x, 0.035, w, 0.142, "Later layers\nand output head")
    arrow(ax, (centre, 0.832), (centre, 0.771))
    arrow(ax, (centre, 0.637), (centre, 0.548))
    arrow(ax, (centre, 0.456), (centre, 0.399))
    arrow(ax, (centre, 0.265), (centre, 0.185))

    ax.add_patch(Circle((centre, 0.501), 0.044, fc="white", ec=GREY, lw=1.0))
    ax.text(centre, 0.503, "+", fontsize=25, va="center", ha="center")
    block(ax, 0.667, 0.454, 0.278, 0.096, r"$\alpha d$", fill="#F1E3EA",
          edge="#BAA2AF", size=19)
    ax.text(0.806, 0.584, "Steering edit", ha="center", va="center", fontsize=11)
    arrow(ax, (0.659, 0.501), (centre + 0.045, 0.501), colour=EDIT, width=1.5)
    ax.text(0.74, 0.323, r"$h' = h + \alpha d$", fontsize=19, ha="center", va="center")
    fig.text(0.043, 0.051, "Base model weights remain fixed.", fontsize=10.3, color="#5F666C")

    # Panel (b): an orthographic projection of explicit 3D vectors. Projection is
    # linear, so the plotted triangle preserves h' = h + edit exactly. Geometry
    # is intentionally illustrative and does not encode experimental activations.
    geo = fig.add_axes([0.53, 0.299, 0.435, 0.578])
    geo.set_axis_off()
    azimuth, elevation = np.deg2rad([25.0, 20.0])
    projection = np.array(
        [[np.cos(azimuth), -np.sin(azimuth), 0.0],
         [np.sin(elevation) * np.sin(azimuth),
          np.sin(elevation) * np.cos(azimuth), np.cos(elevation)]]
    )
    assert np.allclose(projection @ projection.T, np.eye(2))

    def p(v):
        return projection @ np.asarray(v, dtype=float)

    origin = np.zeros(3)
    h = np.array([2.2, 0.4, 0.4])
    edit = np.array([0.9, 0.25, 1.5])
    hp = h + edit
    assert np.allclose(p(hp) - p(h), p(edit))

    # A restrained floor grid and three coordinate axes provide depth cues.
    floor = np.array([p([0, 0, 0]), p([3.8, 0, 0]),
                      p([3.8, 1.75, 0]), p([0, 1.75, 0])])
    geo.add_patch(Polygon(floor, facecolor="#F7F9FA", edgecolor="none", zorder=0))
    for value in [0.95, 1.9, 2.85, 3.8]:
        endpoints = np.array([p([value, 0, 0]), p([value, 1.75, 0])])
        geo.plot(*endpoints.T, color="#E3E8EC", lw=0.65, zorder=1)
    for value in [0.4375, 0.875, 1.3125, 1.75]:
        endpoints = np.array([p([0, value, 0]), p([3.8, value, 0])])
        geo.plot(*endpoints.T, color="#E3E8EC", lw=0.65, zorder=1)
    for endpoint in ([4.08, 0, 0], [0, 2.10, 0], [0, 0, 3.35]):
        arrow(geo, p(origin), p(endpoint), colour="#C9D1D7", width=0.9, head=7, zorder=2)

    # The dashed edges are translated copies of the two addition vectors.
    polygon = np.array([p(origin), p(h), p(hp), p(edit)])
    geo.add_patch(Polygon(polygon, facecolor=EDIT, alpha=0.027, edgecolor="none", zorder=2))
    for a, b, colour in [(origin, edit, EDIT), (edit, hp, BLUE)]:
        pts = np.array([p(a), p(b)])
        geo.plot(*pts.T, ls=(0, (3.5, 3.5)), color=colour, alpha=0.38, lw=1.0, zorder=3)
    # Dashed vertical projection is a spatial cue, not an extra steering action.
    drop = np.array([p(hp), p([hp[0], hp[1], 0])])
    geo.plot(*drop.T, ls=(0, (2, 4)), color="#CCD3D8", lw=0.7, zorder=1)

    arrow(geo, p(origin), p(hp), colour=COPPER, width=2.1, head=13, zorder=5)
    arrow(geo, p(origin), p(h), colour=BLUE, width=2.3, head=13, zorder=6)
    arrow(geo, p(h), p(hp), colour=EDIT, width=2.3, head=13, zorder=7)
    geo.scatter(*p(origin), s=12, c=GREY, zorder=8)
    geo.scatter(*p(h), s=14, c=BLUE, zorder=8)
    geo.scatter(*p(hp), s=14, c=COPPER, zorder=8)
    geo.text(-0.10, -0.11, "0", fontsize=10, color="#727B83", ha="right", va="top")
    geo.text(*(p(h) + [0.065, -0.09]), r"$h$", color=BLUE, fontsize=20, ha="left", va="top")
    geo.text(*(p(hp) + [0.03, 0.105]), r"$h'$", color=COPPER, fontsize=21, ha="center", va="bottom")
    geo.text(*((p(h) + p(hp)) / 2 + [0.22, 0.03]), r"$\alpha d$",
             color=EDIT, fontsize=20, va="center")
    geo.set(xlim=(-1.06, 3.83), ylim=(-0.17, 3.26), aspect="equal")

    fig.text(0.745, 0.281, "Illustrative hidden-state geometry", fontsize=10,
             ha="center", color="#657079")
    # The three definitions align directly below the diagram, with ample spacing.
    for y, symbol, description, colour in [
        (0.217, r"$h$", "original hidden state", BLUE),
        (0.172, r"$d$", "steering direction", EDIT),
        (0.127, r"$\alpha$", "steering strength", EDIT),
    ]:
        fig.text(0.579, y, symbol, fontsize=16, ha="center", va="center", color=colour)
        fig.text(0.603, y, description, fontsize=11.2, va="center")
    fig.text(0.945, 0.051, r"$\alpha=0$: no edit", fontsize=11.3, ha="right", color="#5F666C")

    out.mkdir(parents=True, exist_ok=True)
    name = "how_activation_steering_works"
    metadata = {"Title": "How activation steering works", "Author": "RouteGate thesis"}
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight", pad_inches=0.11,
                metadata=metadata)
    fig.savefig(out / f"{name}.svg", bbox_inches="tight", pad_inches=0.11,
                metadata={"Title": metadata["Title"],
                          "Description": "Schematic additive edit with orthographic 3D vector geometry; not measured data."})
    fig.savefig(out / f"{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.11)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "Figures")
    args = parser.parse_args()
    build(args.output_dir)
