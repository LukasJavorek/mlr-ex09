"""The one plotting style for this folder's figures.

Same reference palette as the audit figures in `../task3_audit_figs/` (slots 1-3 of the
categorical set plus the neutral grays), so a figure here and a figure there can sit
side by side in RESULTS.md and read as one set. Every figure is saved as a .png at
200 dpi into `figures/`; nothing calls `plt.show()`, because the whole point is that a
run is reproducible headless and every number in a figure is also in a .json.

Colour is used for identity only - which method, which arm - never for rank, and every
panel with two or more series has a legend, so nothing depends on telling hues apart.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

FIGURES = Path(__file__).resolve().parent / "figures"

INK, MUTED, GREY = "#0b0b0b", "#52514e", "#9b9a94"
BLUE, ORANGE, GREEN, RED = "#2a78d6", "#eb6834", "#1baf7a", "#c2453d"
GRID, SURFACE = "#e8e6e0", "#fcfcfb"
SERIES = (BLUE, ORANGE, GREEN, RED)


def style(size: int = 8) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": size,
        "axes.edgecolor": GREY, "axes.linewidth": 0.8,
        "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "figure.autolayout": False})


def save(fig, name: str) -> Path:
    FIGURES.mkdir(exist_ok=True)
    path = FIGURES / name
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path.relative_to(FIGURES.parent)}")
    return path


def annotate_peak(ax, x: float, y: float, label: str = "best epoch", color: str = ORANGE) -> None:
    """Mark the epoch a run was selected at, so the eye lands on it first. `y` is where
    the label goes, in data coordinates."""
    ax.axvline(x, color=color, lw=1.0, ls=(0, (3, 3)), zorder=1)
    ax.annotate(label, (x, y), textcoords="offset points", xytext=(4, -2),
                fontsize=7, color=color, va="top")
