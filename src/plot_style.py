"""Shared plotting style for the paper figures.

Import `use_paper_style()` once, then use METHOD_COLORS / METHOD_MARKERS for
consistency across every figure in the paper. Designed for a 2-column
NeurIPS/ICML/ACL-style template: figures are sized to fit one column
(~3.3in) or a full text width (~6.9in) at 300dpi, with fonts that stay
legible after LaTeX shrinks them.
"""
from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt

METHODS = ("SFT", "DPO", "ORPO", "KTO")
PERSONAS = ("level-0", "level-1", "level-2", "level-3")
PERSONA_LABELS = {"level-0": "NT", "level-1": "L1", "level-2": "L2", "level-3": "L3"}

# Okabe-Ito colorblind-safe palette, reused from the original code.
METHOD_COLORS = {"SFT": "#4D4D4D", "DPO": "#0072B2", "ORPO": "#E69F00", "KTO": "#CC79A7"}
METHOD_MARKERS = {"SFT": "o", "DPO": "s", "ORPO": "^", "KTO": "D"}

# One column of a NeurIPS/ICML/ACL-style template is ~3.3in; full width ~6.9in.
COLUMN_WIDTH_IN = 3.3
TEXT_WIDTH_IN = 6.9


def use_paper_style() -> None:
    """Call once at import time / start of the analysis script."""
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.5,
        "axes.titlesize": 9,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.7,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#111111",
        "text.color": "#111111",
        "xtick.color": "#333333",
        "ytick.color": "#333333",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linewidth": 0.5,
        "grid.color": "#888888",
        "axes.axisbelow": True,
        "savefig.dpi": 300,
        "figure.dpi": 150,
        "pdf.fonttype": 42,  # embed as real text, not paths, for camera-ready PDFs
        "ps.fonttype": 42,
    })


def style_axis(ax) -> None:
    """Strip chartjunk: top/right spines gone, ticks outward, thin remaining spines."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)
    ax.tick_params(direction="out")
    ax.grid(axis="y", zorder=0)
    ax.set_axisbelow(True)
