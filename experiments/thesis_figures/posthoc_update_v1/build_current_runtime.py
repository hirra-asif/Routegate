"""Draw the two frozen final-evaluation runtimes without executing a model.

Usage:
    python build_current_runtime.py --output-dir Figures

Grounding: src/routegate/finalrunner.py::execute_item and
src/routegate/gatesensors.py::GateThresholds.fires in the supplied RouteGate repo.
The strict H threshold, the two-pass execution and zero-based layer conventions
are intentional. SVG labels remain text; PDF output is vector with embedded fonts.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


INK = "#292C32"
MUTED = "#58626C"
EDGE = "#B2BAC0"
BLUE = "#2F6595"
COPPER = "#B47B48"
RUST = "#9E594C"
PALE_BLUE = "#EDF4F8"
PALE_COPPER = "#FCF4EB"
PALE_NEUTRAL = "#F7F8F8"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parent / "Figures")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.family": "DejaVu Serif", "font.size": 9.6,
        "mathtext.fontset": "dejavuserif", "svg.fonttype": "none",
        "pdf.fonttype": 42, "savefig.facecolor": "white",
    })
    fig = plt.figure(figsize=(7.25, 5.35), facecolor="white")
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set(xlim=(0, 7.25), ylim=(0, 5.35))
    ax.axis("off")

    def text(x, y, s, size=9.6, ha="center", va="center", colour=INK,
             weight="normal", family="DejaVu Serif", **kwargs):
        return ax.text(x, y, s, fontsize=size, ha=ha, va=va, color=colour,
                       fontweight=weight, fontfamily=family, **kwargs)

    def card(x, y, w, h, fill=PALE_NEUTRAL, edge=EDGE):
        ax.add_patch(FancyBboxPatch((x, y - h / 2), w, h,
                                   boxstyle="round,pad=0,rounding_size=0.025",
                                   facecolor=fill, edgecolor=edge, linewidth=.8))

    def arrow(x0, x1, y):
        ax.add_patch(FancyArrowPatch((x0, y), (x1, y), arrowstyle="-|>",
                                    mutation_scale=9, linewidth=.95, color=MUTED,
                                    shrinkA=3, shrinkB=3))

    def step_label(x, y, s, colour=MUTED):
        text(x, y, s, size=7.8, colour=colour, weight="bold",
             family="DejaVu Sans")

    def title(y, prefix, heading, subtitle):
        text(.08, y, prefix + "  " + heading, size=12, ha="left",
             weight="bold", family="DejaVu Sans")
        text(.08, y - .26, subtitle, size=9.1, ha="left", colour=MUTED)

    # Four aligned columns separate the capture, decision, prefill edit and decode.
    cap_x, cap_w = .08, 1.19
    ctl_x, ctl_w = 1.60, 1.65
    edit_x, edit_w = 3.59, 2.11
    gen_x, gen_w = 6.04, 1.13
    cap_c, ctl_c = cap_x + cap_w / 2, ctl_x + ctl_w / 2
    edit_c, gen_c = edit_x + edit_w / 2, gen_x + gen_w / 2

    def shared_steps(y):
        card(cap_x, y, cap_w, 1.02, PALE_BLUE, "#97B0C3")
        step_label(cap_c, y + .30, "UNSTEERED PASS", BLUE)
        text(cap_c, y - .02, "Process\nprompt", size=10, linespacing=1.35)
        card(gen_x, y, gen_w, 1.02)
        step_label(gen_c, y + .30, "DECODE")
        text(gen_c, y - .02, "Generate\nresponse", size=9.8, linespacing=1.35)
        arrow(cap_x + cap_w, ctl_x, y)
        arrow(ctl_x + ctl_w, edit_x, y)
        arrow(edit_x + edit_w, gen_x, y)

    title(4.97, "(a)", "Dynamic strength", "The controller chooses the strength of the edit.")
    y = 3.88
    shared_steps(y)
    card(ctl_x, y, ctl_w, 1.02)
    step_label(ctl_c, y + .30, "READ + DECIDE")
    text(ctl_c, y - .02, "Read state 32", size=10)
    text(ctl_c, y - .27, r"Compute $g_R(x)$", size=10)

    card(edit_x, y, edit_w, 1.28, PALE_COPPER, "#C5A586")
    step_label(edit_c, y + .40, "NEW PROMPT PASS", COPPER)
    text(edit_c, y + .07, r"$\alpha_R(x)=0.15\,g_R(x)$", size=11.4)
    text(edit_c, y - .24, r"Direction $d^{(1)}$", size=10)
    text(edit_c, y - .48, "Layers 31–32", size=10)

    ax.plot([.08, 7.17], [2.96, 2.96], color="#DCE1E4", linewidth=.8)
    title(2.67, "(b)", "Safety-first RouteGate", "The gate chooses whether to apply the fixed edit.")
    y = 1.53
    shared_steps(y)
    card(ctl_x, y, ctl_w, 1.15)
    step_label(ctl_c, y + .38, "READ + DECIDE")
    text(ctl_c, y + .07, "Read state 45", size=10)
    text(ctl_c, y - .24, r"$p_E\geq\tau_E$ and $p_H<\tau_H$", size=9.5)

    card(edit_x, y, edit_w, 1.54, PALE_COPPER, "#C5A586")
    step_label(edit_c, y + .55, "NEW PROMPT PASS", COPPER)
    text(edit_c, y + .24, "Open: strength 0.15", size=10.6)
    text(edit_c, y - .02, r"$d^{(23)}$, layers 31–33", size=10)
    ax.plot([edit_x + .12, edit_x + edit_w - .12], [y - .23, y - .23],
            color="#D7C4B0", linewidth=.75)
    text(edit_c, y - .47, "Closed: no edit", size=10, colour=MUTED)

    # Keep the scientific scope clear without implying generation-time editing.
    text(.08, .45, "The first pass records activations and unsteered reference norms.",
         size=8.5, ha="left", colour=MUTED)
    text(.08, .24, "Fixed model weights · final prompt-token edits only · no decode-token edits",
         size=8.5, ha="left", colour=MUTED)

    for extension in ("pdf", "svg", "png"):
        destination = args.output_dir / f"routegate_current_runtime.{extension}"
        fig.savefig(destination, dpi=300, metadata={"Creator": "RouteGate scientific figure source"})
        print(destination)
    plt.close(fig)


if __name__ == "__main__":
    main()
