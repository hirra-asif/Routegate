"""Shared figure style.

Style only. No figure is produced at the reproduction stage -- publication plots
belong to the corrected analysis, once the numbers they would display have been
established.

The palette is a validated two-series categorical pair plus a diverging pair for
signed quantities. Series colours are assigned to entities, never to rank, so a
chart that drops a series does not repaint the survivors:

    B, benign compliance   slot 1, blue
    E, over-refusal        slot 2, orange
    H, harmful request     slot 3, teal

Slot 3 was added when H stopped being a class that is carried and not analysed:
the causal results report an H outcome beside the E and B ones, so H needs a
colour of its own. It is a third categorical hue, not a status colour -- the
panel that plots it says in words which direction is better. The three hues were
checked, not chosen by eye: every pair clears the CVD and normal-vision
separation floors against :data:`SURFACE`, and each clears the lightness band,
the chroma floor and 3:1 contrast.

:data:`STATUS_CRITICAL` is separate and is reserved. It marks a harmful outcome
*as* an outcome -- a failed cell, a harm count -- and is never handed out as
"series 4".
"""
from __future__ import annotations

from typing import Any, Dict, Sequence

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

SERIES_B = "#2a78d6"          # categorical slot 1
SERIES_E = "#eb6834"          # categorical slot 2
SERIES_H = "#0b8f7b"          # categorical slot 3

DIVERGING_LOW = "#2a78d6"
DIVERGING_MID = "#f0efec"
DIVERGING_HIGH = "#e34948"

#: Reserved. A harmful outcome, never a categorical series.
STATUS_CRITICAL = "#e34948"
STATUS_CRITICAL_FILL = "#f7dbdb"

CLASS_COLOURS: Dict[str, str] = {"B": SERIES_B, "E": SERIES_E, "H": SERIES_H}
CLASS_LABELS: Dict[str, str] = {"B": "B  benign compliance",
                                "E": "E  over-refusal",
                                "H": "H  harmful request"}

RC_PARAMS: Dict[str, Any] = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.edgecolor": AXIS,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK_SECONDARY,
    "axes.titlecolor": INK,
    "axes.titlesize": 11.5,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "xtick.labelcolor": INK_SECONDARY,
    "ytick.labelcolor": INK_SECONDARY,
    "legend.frameon": False,
    "figure.dpi": 160,
    # Deterministic SVG element ids. Without a salt matplotlib derives clip-path
    # ids from object identity, so two runs over identical inputs produce SVGs
    # that differ in every id -- which would make a figure built from frozen
    # inputs unable to demonstrate that it is reproducible.
    "svg.hashsalt": "routegate",
}

STATE_AXIS_LABEL = "state  (0 = embedding output, 1-48 = transformer-block outputs)"


def configure(backend: str = "Agg"):
    """Apply the shared style and return ``pyplot``."""
    import matplotlib
    matplotlib.use(backend)
    import matplotlib.pyplot as plt
    plt.rcParams.update(RC_PARAMS)
    return plt


def tidy(ax) -> None:
    """Recede the chrome: no top or right spine, grid behind the marks."""
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def diverging_colormap():
    """Two poles with a neutral midpoint, for signed quantities."""
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list(
        "routegate_diverging",
        [DIVERGING_LOW, DIVERGING_MID, DIVERGING_HIGH], N=256)


def title_block(ax, title: str, subtitle: str, wrap: int = 92) -> None:
    """Title above a wrapped subtitle above the axes, never overlapping."""
    import textwrap
    lines = [line for chunk in subtitle.split("\n")
             for line in (textwrap.wrap(chunk, width=wrap) or [""])]
    ax.text(0.0, 1.0, "\n".join(lines), transform=ax.transAxes,
            va="bottom", ha="left", color=INK_SECONDARY, fontsize=9.0,
            linespacing=1.35)
    ax.set_title(title, loc="left", pad=6.0 + 12.5 * len(lines))


def save_figure(fig, directory, stem: str, dpi: int = 400) -> Dict[str, str]:
    """Write one figure twice: a high-resolution raster and an editable vector.

    Both are needed and neither substitutes for the other. The PNG is what a
    document embeds; the SVG is what a reader can open and correct a label in
    without rerunning anything. Returns the two paths, keyed by extension.

    ``Date: None`` suppresses the ``dc:date`` matplotlib otherwise stamps into
    every SVG. A wall clock inside the file would make two runs over the same
    frozen inputs differ byte-for-byte, which is exactly the thing a figure
    built from frozen inputs should be able to prove it does not do. When a run
    happened belongs in its run record, not in the artifact.
    """
    from pathlib import Path
    directory = Path(directory)
    written = {}
    for extension, options in (("png", {"dpi": dpi}),
                               ("svg", {"metadata": {"Date": None}})):
        path = directory / f"{stem}.{extension}"
        fig.savefig(path, bbox_inches="tight", facecolor=SURFACE, **options)
        written[extension] = str(path)
    return written
